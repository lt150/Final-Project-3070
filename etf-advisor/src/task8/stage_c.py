"""Task 8 Stage C: one-shot test evaluation of the frozen T1xA1 model.

test is touched exactly once, after gates C1-C3 pass, 
and whatever comes out is final. test.npz is not loaded
until every gate has passed and the val context table is built. No re-runs
after test numbers exist; anomalies are reported with a diagnosis, not
fixed. The frozen configuration is the seed-42 T1xA1 checkpoint from
Stage B (best epoch 1) — seed 44 scored better in the stability check but
the pre-registered checkpoint is seed 42 and is not swapped.

Context baselines: OLS-pooled (y_vol = a + b*vol_trail_20, train-only, 2 params)
and OLS-per-ticker (same per ticker, 16 params).
OLS-per-ticker receives ticker identity,
which the LSTM never sees — deliberate: its question is "what does a
trivially simple practical method achieve," not "what is a fair fight."
They calibrate how much of the LSTM's shrinkage-driven skill is generic;
they play no selection role.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import torch

from ..data.config import PROCESSED_DIR, SEED
from .metrics import basic_metrics, rmse_skill
from .modeling import TargetSpec, build_model
from .stage_b import OUT_DIR, MODEL_DIR, predict_raw

CKPT_PATH = MODEL_DIR / "T1_A1_seed42.pt"
STAGE_B_CSV = OUT_DIR / "stage_b_results.csv"
GATE_C1_TOL = 1e-9
N_TEST_ROWS = 3856

FORECASTERS = ("persist", "lstm", "ols_pooled", "ols_ticker")


# --------------------------------------------------------------------------- #
# Gates (no test.npz contact anywhere in this section)
# --------------------------------------------------------------------------- #
def gate_c1_checkpoint_roundtrip(d: dict, lines: list[str]) -> tuple:
    ckpt = torch.load(CKPT_PATH, weights_only=True)
    assert (
        ckpt["target"] == "T1" and ckpt["arch_id"] == "A1"
        and ckpt["seed"] == SEED and ckpt["best_epoch"] == 1
    ), f"checkpoint metadata unexpected: {({k: ckpt[k] for k in ('target','arch_id','seed','best_epoch')})}"

    model = build_model(ckpt["arch_id"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    spec = TargetSpec(name="T1", mu=ckpt["mu"], sd=ckpt["sd"])

    X_val = torch.from_numpy(d["X_val"].astype(np.float32))
    pred_val = predict_raw(model, X_val, spec, d["trail_val"])
    rmse = float(np.sqrt(np.mean((pred_val - d["y_val"]) ** 2)))

    results = pd.read_csv(STAGE_B_CSV)
    ref = float(results.loc[results["run"] == "T1xA1", "val_rmse"].iloc[0])
    diff = abs(rmse - ref)
    assert diff < GATE_C1_TOL, (
        f"reloaded checkpoint val RMSE {rmse!r} differs from Stage B's "
        f"{ref!r} by {diff:.2g} >= {GATE_C1_TOL} — loaded model is not the "
        "validated model, STOP"
    )
    lines.append(
        f"GATE C1 PASS — checkpoint round-trip: reloaded T1xA1 val RMSE "
        f"{rmse:.10f} matches stage_b_results.csv {ref:.10f} (|diff| "
        f"{diff:.2g} < {GATE_C1_TOL:g})"
    )
    return model, spec, pred_val


def gate_c2_test_alignment(d: dict, lines: list[str]) -> None:
    b = d["base_test"].reset_index(drop=True)
    m = d["meta_test"].reset_index(drop=True)
    assert len(b) == N_TEST_ROWS, (
        f"baseline.parquet test rows {len(b)} != {N_TEST_ROWS}"
    )
    assert len(m) == N_TEST_ROWS
    rng = np.random.default_rng(SEED)
    idx = rng.choice(N_TEST_ROWS, size=50, replace=False)
    for i in idx:
        assert (
            b.at[i, "ticker"] == m.at[i, "ticker"]
            and b.at[i, "window_end"] == m.at[i, "window_end"]
        ), f"baseline/meta test row mismatch at [{i}]"
    lines.append(
        "GATE C2 PASS — baseline.parquet holds all 3856 test rows in npz "
        "row order (Stage A wrote them; no vol_trail_20 extension needed); "
        "50-row (window_end, ticker) spot check vs meta (seed 42) matches"
    )


def gate_c3_ols_fit(d: dict, lines: list[str]) -> dict:
    """Fit both OLS baselines on TRAIN rows only. Never reads val/test targets."""
    x, y = d["trail_train"], d["y_train"]
    A = np.column_stack([np.ones_like(x), x])
    coef_pooled, *_ = np.linalg.lstsq(A, y, rcond=None)  # [a, b]

    coef_ticker = {}
    tickers = d["tickers_train"]
    for t in np.unique(tickers):
        mask = tickers == t
        At = np.column_stack([np.ones(mask.sum()), x[mask]])
        coef_ticker[t], *_ = np.linalg.lstsq(At, y[mask], rcond=None)

    all_coefs = np.concatenate([coef_pooled] + list(coef_ticker.values()))
    assert np.all(np.isfinite(all_coefs)), "non-finite OLS coefficient — STOP"
    lines.append(
        f"GATE C3 PASS — OLS fit on train rows only, all coefficients "
        f"finite; pooled: a={coef_pooled[0]:.6g}, b={coef_pooled[1]:.6g} "
        f"(2 params); per-ticker: 8 x (a, b) = 16 params"
    )
    return {"pooled": coef_pooled, "ticker": coef_ticker}


def ols_predict(ols: dict, which: str, trail: np.ndarray, tickers: np.ndarray) -> np.ndarray:
    if which == "pooled":
        a, b = ols["pooled"]
        return a + b * trail
    pred = np.empty_like(trail)
    for t, (a, b) in ols["ticker"].items():
        mask = tickers == t
        pred[mask] = a + b * trail[mask]
    return pred


# --------------------------------------------------------------------------- #
# Tables (shared by the val context table and the single test pass)
# --------------------------------------------------------------------------- #
def headline_table(preds: dict[str, np.ndarray], actual: np.ndarray) -> pd.DataFrame:
    rmse_p = float(np.sqrt(np.mean((preds["persist"] - actual) ** 2)))
    rows = []
    for name in FORECASTERS:
        m = basic_metrics(preds[name], actual)
        rows.append({"forecaster": name, **m, "skill": rmse_skill(m["rmse"], rmse_p)})
    return pd.DataFrame(rows)


def per_ticker_wide(
    preds: dict[str, np.ndarray], actual: np.ndarray, tickers: np.ndarray
) -> pd.DataFrame:
    rows = []
    for t in np.unique(tickers):
        mask = tickers == t
        row = {"ticker": t, "n": int(mask.sum())}
        rmse_p = float(np.sqrt(np.mean((preds["persist"][mask] - actual[mask]) ** 2)))
        row["rmse_persist"] = rmse_p
        for name in FORECASTERS[1:]:
            r = float(np.sqrt(np.mean((preds[name][mask] - actual[mask]) ** 2)))
            row[f"rmse_{name}"] = r
            row[f"skill_{name}"] = rmse_skill(r, rmse_p)
        rows.append(row)
    df = pd.DataFrame(rows)
    mean_row = {"ticker": "MEAN(eq-wt)", "n": int(df["n"].sum())}
    for c in df.columns[2:]:
        mean_row[c] = df[c].mean()
    return pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)


def decile_calibration_wide(
    preds: dict[str, np.ndarray], actual: np.ndarray
) -> pd.DataFrame:
    """Each forecaster bucketed by its OWN predicted decile."""
    out = []
    for name in FORECASTERS:
        df = pd.DataFrame({"pred": preds[name], "actual": actual})
        df["decile"] = pd.qcut(df["pred"].rank(method="first"), 10, labels=False) + 1
        g = df.groupby("decile").agg(
            **{f"{name}_pred": ("pred", "mean"), f"{name}_actual": ("actual", "mean")}
        )
        out.append(g)
    return pd.concat(out, axis=1).reset_index()


def conditional_skill_wide(
    preds: dict[str, np.ndarray],
    actual: np.ndarray,
    trail: np.ndarray,
    tickers: np.ndarray,
) -> pd.DataFrame:
    df = pd.DataFrame({"ticker": tickers, "trail": trail})
    df["decile"] = (
        df.groupby("ticker")["trail"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
        + 1
    )
    rows = []
    for dec in range(1, 11):
        mask = (df["decile"] == dec).to_numpy()
        row = {"trail_decile": dec, "n": int(mask.sum())}
        rmse_p = float(np.sqrt(np.mean((preds["persist"][mask] - actual[mask]) ** 2)))
        row["rmse_persist"] = rmse_p
        for name in FORECASTERS[1:]:
            r = float(np.sqrt(np.mean((preds[name][mask] - actual[mask]) ** 2)))
            row[f"rmse_{name}"] = r
            row[f"skill_{name}"] = rmse_skill(r, rmse_p)
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_df(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:.6f}")


# --------------------------------------------------------------------------- #
# Finding statement (generated from the numbers)
# --------------------------------------------------------------------------- #
def finding_statement(
    head: pd.DataFrame, per_ticker: pd.DataFrame, cond: pd.DataFrame,
    pred_lstm: np.ndarray, val_skill_lstm: float,
) -> list[str]:
    h = head.set_index("forecaster")
    s_lstm = h.at["lstm", "skill"]
    lines = ["", "— finding statement —"]

    lines.append(
        f"(a) LSTM vs persistence on test: skill {s_lstm:+.4f} "
        f"(RMSE {h.at['lstm', 'rmse']:.6f} vs {h.at['persist', 'rmse']:.6f}) — "
        + ("the LSTM BEATS persistence on test."
           if s_lstm > 0 else "the LSTM DOES NOT beat persistence on test.")
        + f" Val skill was {val_skill_lstm:+.4f}."
    )

    for name, label in (("ols_pooled", "OLS-pooled"), ("ols_ticker", "OLS-per-ticker")):
        rel = 1.0 - h.at["lstm", "rmse"] / h.at[name, "rmse"]
        lines.append(
            f"(b) LSTM vs {label}: {label} skill {h.at[name, 'skill']:+.4f} "
            f"(RMSE {h.at[name, 'rmse']:.6f}); LSTM RMSE is {abs(rel):.2%} "
            + ("LOWER than" if rel > 0 else "HIGHER than") + f" {label}."
        )

    pt = per_ticker[per_ticker["ticker"] != "MEAN(eq-wt)"]
    pos = pt.loc[pt["skill_lstm"] > 0, "ticker"].tolist()
    eq = per_ticker.loc[per_ticker["ticker"] == "MEAN(eq-wt)", "skill_lstm"].iloc[0]
    top = cond.loc[cond["trail_decile"] == 10, "skill_lstm"].iloc[0]
    bot = cond.loc[cond["trail_decile"].isin([1, 2]), "skill_lstm"].mean()
    mid = cond.loc[cond["trail_decile"].isin(range(4, 10)), "skill_lstm"].mean()
    lines.append(
        f"(c) Where the skill sits: positive LSTM skill on {len(pos)}/8 "
        f"tickers ({', '.join(pos) if pos else 'none'}); equal-weighted mean "
        f"skill {eq:+.4f} vs pooled {s_lstm:+.4f}. Conditional on trailing "
        f"vol: top within-ticker decile skill {top:+.4f}, deciles 1-2 mean "
        f"{bot:+.4f}, deciles 4-9 mean {mid:+.4f}."
    )

    floor_hits = int((pred_lstm <= 1e-6).sum())
    anomalies = []
    if not np.all(np.isfinite(pred_lstm)):
        anomalies.append("non-finite LSTM predictions")
    if floor_hits:
        anomalies.append(f"{floor_hits} prediction(s) at the 1e-6 floor")
    lines.append(
        "(d) Anomalies: " + ("; ".join(anomalies) if anomalies else "none")
        + f". LSTM prediction range [{pred_lstm.min():.6f}, "
        f"{pred_lstm.max():.6f}]; floor hits {floor_hits}; std ratio "
        f"{h.at['lstm', 'std_ratio']:.4f} (under-dispersion is the known, "
        "pre-registered shrinkage behaviour, not an anomaly). Reported as-is; "
        "test not touched again."
    )
    return lines


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    lines = ["Task 8 Stage C — one-shot test evaluation (T1xA1, frozen)", ""]

    # ---- load everything EXCEPT test.npz --------------------------------- #
    npz = {s: np.load(PROCESSED_DIR / f"{s}.npz") for s in ("train", "val")}
    base = pd.read_parquet(OUT_DIR / "baseline.parquet")
    meta = pd.read_parquet(PROCESSED_DIR / "meta.parquet")
    d = {
        "X_val": npz["val"]["X"],
        "y_val": npz["val"]["y_vol"].astype(np.float64),
        "y_train": npz["train"]["y_vol"].astype(np.float64),
        "base_test": base[base["split"] == "test"],
        "meta_test": meta[meta["split"] == "test"],
    }
    for split in ("train", "val", "test"):
        b = base[base["split"] == split]
        d[f"trail_{split}"] = b["vol_trail_20"].to_numpy(dtype=np.float64)
        d[f"tickers_{split}"] = b["ticker"].to_numpy()

    # ---- gates ------------------------------------------------------------ #
    model, spec, pred_val_lstm = gate_c1_checkpoint_roundtrip(d, lines)
    gate_c2_test_alignment(d, lines)
    ols = gate_c3_ols_fit(d, lines)

    # ---- val context table (reported before test) ------------------------- #
    preds_val = {
        "persist": d["trail_val"],
        "lstm": pred_val_lstm,
        "ols_pooled": ols_predict(ols, "pooled", d["trail_val"], d["tickers_val"]),
        "ols_ticker": ols_predict(ols, "ticker", d["trail_val"], d["tickers_val"]),
    }
    head_val = headline_table(preds_val, d["y_val"])
    val_skill_lstm = float(
        head_val.set_index("forecaster").at["lstm", "skill"]
    )
    lines += ["", "— val context table (pre-test; OLS baselines are "
              "interpretive only, no selection role) —", fmt_df(head_val)]

    # ---- THE single test pass --------------------------------------------- #
    lines += ["", "=" * 68,
              "TEST — single pass, first and only contact with test.npz",
              "=" * 68]
    test = np.load(PROCESSED_DIR / "test.npz")
    X_test = torch.from_numpy(test["X"].astype(np.float32))
    y_test = test["y_vol"].astype(np.float64)

    preds = {
        "persist": d["trail_test"],
        "lstm": predict_raw(model, X_test, spec, d["trail_test"]),
        "ols_pooled": ols_predict(ols, "pooled", d["trail_test"], d["tickers_test"]),
        "ols_ticker": ols_predict(ols, "ticker", d["trail_test"], d["tickers_test"]),
    }

    head = headline_table(preds, y_test)
    pt = per_ticker_wide(preds, y_test, d["tickers_test"])
    cal = decile_calibration_wide(preds, y_test)
    cond = conditional_skill_wide(preds, y_test, d["trail_test"], d["tickers_test"])

    lines += ["", "1. headline (raw daily vol units; skill vs persistence):",
              fmt_df(head),
              "", "2. per-ticker RMSE and skill:", fmt_df(pt),
              "", "3. decile calibration (each forecaster bucketed by its own "
              "predicted decile):", fmt_df(cal),
              "", "4. conditional skill by within-ticker decile of vol_trail_20 "
              "(test is 2023-24, calmer than val's 2021-22):", fmt_df(cond)]

    lines += finding_statement(head, pt, cond, preds["lstm"], val_skill_lstm)

    # ---- deliverables ------------------------------------------------------ #
    head.to_csv(OUT_DIR / "stage_c_results.csv", index=False)
    hand_off = d["base_test"][["ticker", "window_end"]].reset_index(drop=True).assign(
        y_vol=y_test,
        vol_trail_20=d["trail_test"],
        pred_lstm=preds["lstm"],
        pred_ols_pooled=preds["ols_pooled"],
        pred_ols_ticker=preds["ols_ticker"],
    )
    hand_off.to_parquet(OUT_DIR / "predictions_test.parquet", index=False)
    lines += ["", f"deliverables: stage_c_results.csv, predictions_test.parquet "
              f"({len(hand_off)} rows), stage_c_report.txt — all in outputs/task8/",
              "", "STOP — Task 8 evaluation closed. Test was touched exactly "
              "once; these numbers are final regardless of outcome. Any further "
              "modeling is a new, separately registered task."]

    report = "\n".join(lines)
    (OUT_DIR / "stage_c_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, RuntimeError) as e:
        print(f"STAGE C FAILURE — STOP: {e}", file=sys.stderr)
        sys.exit(1)
