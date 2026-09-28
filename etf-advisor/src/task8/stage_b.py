"""Task 8 Stage B: three-target LSTM sweep vs the persistence baseline.

Pre-registered protocol:
  * Gates B1 (alignment), B2 (persistence checksum vs Stage A, |diff| <
    1e-9), B3 (log-safety: min targets > 0) run before any training.
  * Fixed grid: 4 architectures x 3 targets = 12 runs, identical recipe
    (Adam lr 1e-3, batch 256, MSE in standardised target space, max 100
    epochs, early stopping on VAL RMSE IN RAW VOL SPACE with patience 10,
    restore best weights, grad clip 1.0, seed 42, deterministic algorithms).
  * Selection: lowest val RMSE in raw space; near-ties (< 1% relative)
    resolve to the simpler target (T1 > T2 > T3), then fewer parameters.
  * Stability: winner re-run at seeds 43/44; mean/min/max reported.

vol_trail_20 is sourced ONLY from Stage A's frozen baseline.parquet
(single source, single failure point).
"""

from __future__ import annotations

import copy
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..data.config import PROCESSED_DIR, ROOT, SEED
from .metrics import basic_metrics, per_ticker_table, rmse_skill
from .modeling import ARCHS, TARGETS, TargetSpec, build_model, fit_target_spec, set_determinism

OUT_DIR = ROOT / "outputs" / "task8"
MODEL_DIR = OUT_DIR / "models"

# Gate B2 reference: Stage A's val persistence metrics at full precision
# (recomputed with src.task8.metrics.basic_metrics on the frozen baseline.parquet.
STAGE_A_VAL_PERSIST = {
    "rmse": 0.003543504837961903,
    "mae": 0.0025830856197616436,
    "pearson_r": 0.7811475481024399,
    "std_ratio": 0.9983551837636984,
}
GATE_B2_TOL = 1e-9

BATCH_SIZE = 256
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
GRAD_CLIP = 1.0
RUNTIME_GUARD_S = 900  # ~15 min: STOP rather than shrink the recipe
STABILITY_SEEDS = (43, 44)
TARGET_SIMPLICITY = {"T1": 0, "T2": 1, "T3": 2}  # tie-break: lower = simpler
NEAR_TIE_REL = 0.01


# --------------------------------------------------------------------------- #
# Data + gates
# --------------------------------------------------------------------------- #
def load_stage_b_data() -> dict:
    npz = {s: np.load(PROCESSED_DIR / f"{s}.npz") for s in ("train", "val")}
    base = pd.read_parquet(OUT_DIR / "baseline.parquet")
    meta = pd.read_parquet(PROCESSED_DIR / "meta.parquet")
    d = {
        "meta": meta,
        "base": base,
        "X_train": npz["train"]["X"],
        "y_train": npz["train"]["y_vol"].astype(np.float64),
        "X_val": npz["val"]["X"],
        "y_val": npz["val"]["y_vol"].astype(np.float64),
    }
    for split in ("train", "val"):
        b = base[base["split"] == split]
        d[f"trail_{split}"] = b["vol_trail_20"].to_numpy(dtype=np.float64)
        d[f"tickers_{split}"] = b["ticker"].to_numpy()
    return d


def gate_b1_alignment(d: dict, lines: list[str]) -> None:
    base, meta = d["base"], d["meta"]
    rng = np.random.default_rng(SEED)
    for split, n in (("train", len(d["y_train"])), ("val", len(d["y_val"]))):
        b = base[base["split"] == split].reset_index(drop=True)
        m = meta[meta["split"] == split].reset_index(drop=True)
        assert len(b) == n, f"baseline {split} rows {len(b)} != npz {n}"
        idx = rng.choice(n, size=50, replace=False)
        for i in idx:
            assert (
                b.at[i, "ticker"] == m.at[i, "ticker"]
                and b.at[i, "window_end"] == m.at[i, "window_end"]
            ), f"baseline/meta row mismatch at {split}[{i}]"
    # test rows exist in baseline.parquet but are untouched in Stage B
    lines.append(
        "GATE B1 PASS — baseline.parquet row counts match npz (train/val); "
        "(window_end, ticker) spot check 50 rows/split (seed 42) matches "
        "meta's npz row order"
    )


def gate_b2_persistence_checksum(d: dict, lines: list[str]) -> None:
    m = basic_metrics(d["trail_val"], d["y_val"])
    for k, ref in STAGE_A_VAL_PERSIST.items():
        diff = abs(m[k] - ref)
        assert diff < GATE_B2_TOL, (
            f"val persistence {k} = {m[k]!r} drifted {diff:.2g} >= "
            f"{GATE_B2_TOL} from Stage A's {ref!r} — alignment or metric "
            "changed"
        )
    lines.append(
        "GATE B2 PASS — val persistence metrics recomputed from "
        "baseline.parquet match Stage A's frozen values to < 1e-9 "
        f"(RMSE {m['rmse']:.6f}, MAE {m['mae']:.6f}, r {m['pearson_r']:.4f}, "
        f"std ratio {m['std_ratio']:.4f})"
    )


def gate_b3_log_safety(d: dict, lines: list[str]) -> None:
    mins = {
        "min y_vol train": d["y_train"].min(),
        "min y_vol val": d["y_val"].min(),
        "min vol_trail_20 train": d["trail_train"].min(),
        "min vol_trail_20 val": d["trail_val"].min(),
    }
    for k, v in mins.items():
        assert v > 0, f"GATE B3: {k} = {v!r} <= 0 — log targets unsafe, STOP"
    lines.append(
        "GATE B3 PASS — log-safety: "
        + " | ".join(f"{k} {v:.6g}" for k, v in mins.items())
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def predict_raw(
    model: nn.Module, X: torch.Tensor, spec: TargetSpec, trail: np.ndarray
) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            preds.append(model(X[i : i + 2048]).numpy())
    z = np.concatenate(preds).astype(np.float64)
    return spec.invert(z, trail)


def train_one_run(target: str, arch_id: str, seed: int, d: dict) -> dict:
    t0 = time.perf_counter()
    set_determinism(seed)

    spec = fit_target_spec(target, d["y_train"], d["trail_train"])
    y_tr_z = spec.transform(d["y_train"], d["trail_train"]).astype(np.float32)

    X_tr = torch.from_numpy(d["X_train"].astype(np.float32))
    X_va = torch.from_numpy(d["X_val"].astype(np.float32))
    y_tr = torch.from_numpy(y_tr_z)

    model = build_model(arch_id)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    shuffle_gen = torch.Generator().manual_seed(seed)

    best_rmse, best_epoch, best_state, bad = np.inf, 0, None, 0
    n = len(X_tr)
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n, generator=shuffle_gen)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        # early stopping monitors val RMSE in RAW vol space (primary metric)
        pred_val = predict_raw(model, X_va, spec, d["trail_val"])
        val_rmse = float(np.sqrt(np.mean((pred_val - d["y_val"]) ** 2)))
        if val_rmse < best_rmse:
            best_rmse, best_epoch, bad = val_rmse, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE:
                break

        if time.perf_counter() - t0 > RUNTIME_GUARD_S:
            raise RuntimeError(
                f"runtime guard: {target}x{arch_id} seed {seed} exceeded "
                f"{RUNTIME_GUARD_S}s at epoch {epoch} — STOP per spec §3"
            )

    model.load_state_dict(best_state)
    pred_val = predict_raw(model, X_va, spec, d["trail_val"])
    m = basic_metrics(pred_val, d["y_val"])
    wall = time.perf_counter() - t0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODEL_DIR / f"{target}_{arch_id}_seed{seed}.pt"
    torch.save(
        {
            "state_dict": best_state,
            "target": target,
            "arch_id": arch_id,
            "mu": spec.mu,
            "sd": spec.sd,
            "seed": seed,
            "best_epoch": best_epoch,
        },
        ckpt_path,
    )

    layers, hidden, dropout = ARCHS[arch_id]
    return {
        "run": f"{target}x{arch_id}",
        "target": target,
        "arch": arch_id,
        "layers": layers,
        "hidden": hidden,
        "dropout": dropout,
        "params": n_params,
        "seed": seed,
        "val_rmse": m["rmse"],
        "val_mae": m["mae"],
        "val_r": m["pearson_r"],
        "val_std_ratio": m["std_ratio"],
        "skill": rmse_skill(m["rmse"], STAGE_A_VAL_PERSIST["rmse"]),
        "epochs": epoch,
        "best_epoch": best_epoch,
        "wall_s": round(wall, 1),
        "pred_val": pred_val,  # dropped before CSV
    }


# --------------------------------------------------------------------------- #
# Exhibits (best config per target, on val)
# --------------------------------------------------------------------------- #
def decile_side_by_side(
    pred_model: np.ndarray, pred_persist: np.ndarray, actual: np.ndarray
) -> pd.DataFrame:
    """Calibration by predicted decile, each forecaster bucketed by its own
    predictions (a calibration table is about a forecaster's own buckets)."""
    out = []
    for name, pred in (("model", pred_model), ("persist", pred_persist)):
        df = pd.DataFrame({"pred": pred, "actual": actual})
        df["decile"] = pd.qcut(df["pred"].rank(method="first"), 10, labels=False) + 1
        g = df.groupby("decile").agg(
            **{f"{name}_mean_pred": ("pred", "mean"),
               f"{name}_mean_actual": ("actual", "mean")}
        )
        out.append(g)
    return pd.concat(out, axis=1).reset_index()


def conditional_skill_table(
    pred_model: np.ndarray,
    pred_persist: np.ndarray,
    actual: np.ndarray,
    trail: np.ndarray,
    tickers: np.ndarray,
) -> pd.DataFrame:
    """Pre-registered exhibit: RMSE and skill by decile of vol_trail_20,
    deciles ranked WITHIN ticker on val, pooled by decile bucket."""
    df = pd.DataFrame(
        {"ticker": tickers, "trail": trail, "model": pred_model,
         "persist": pred_persist, "actual": actual}
    )
    df["decile"] = (
        df.groupby("ticker")["trail"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
        + 1
    )
    rows = []
    for dec, g in df.groupby("decile"):
        rmse_m = float(np.sqrt(np.mean((g["model"] - g["actual"]) ** 2)))
        rmse_p = float(np.sqrt(np.mean((g["persist"] - g["actual"]) ** 2)))
        rows.append(
            {"trail_decile": dec, "n": len(g), "rmse_model": rmse_m,
             "rmse_persist": rmse_p, "skill": rmse_skill(rmse_m, rmse_p)}
        )
    return pd.DataFrame(rows)


def fmt_df(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:.6f}")


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def select_winner(rows: list[dict]) -> tuple[dict, list[str]]:
    best_rmse = min(r["val_rmse"] for r in rows)
    ties = [
        r for r in rows
        if (r["val_rmse"] - best_rmse) / best_rmse < NEAR_TIE_REL
    ]
    ties.sort(
        key=lambda r: (TARGET_SIMPLICITY[r["target"]], r["params"], r["val_rmse"])
    )
    winner = ties[0]
    note = [
        f"near-tie set (< {NEAR_TIE_REL:.0%} relative of best "
        f"{best_rmse:.6f}): " + ", ".join(
            f"{r['run']} ({r['val_rmse']:.6f})" for r in ties
        ),
        "tie-break order: simpler target (T1 > T2 > T3), then fewer params",
    ]
    return winner, note


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    lines = ["Task 8 Stage B — three-target LSTM sweep vs persistence", ""]
    d = load_stage_b_data()

    gate_b1_alignment(d, lines)
    gate_b2_persistence_checksum(d, lines)
    gate_b3_log_safety(d, lines)

    # ---- 12-run sweep, seed 42 ------------------------------------------- #
    rows = []
    for target in TARGETS:
        for arch_id in ARCHS:
            r = train_one_run(target, arch_id, SEED, d)
            rows.append(r)
            print(
                f"done {r['run']}: val RMSE {r['val_rmse']:.6f} "
                f"(skill {r['skill']:+.4f}), best epoch {r['best_epoch']}/"
                f"{r['epochs']}, {r['wall_s']}s",
                flush=True,
            )

    results = pd.DataFrame([{k: v for k, v in r.items() if k != "pred_val"}
                            for r in rows])
    persist_row = {
        "run": "persistence", "target": "-", "arch": "-", "layers": None,
        "hidden": None, "dropout": None, "params": None, "seed": None,
        "val_rmse": STAGE_A_VAL_PERSIST["rmse"],
        "val_mae": STAGE_A_VAL_PERSIST["mae"],
        "val_r": STAGE_A_VAL_PERSIST["pearson_r"],
        "val_std_ratio": STAGE_A_VAL_PERSIST["std_ratio"],
        "skill": 0.0, "epochs": None, "best_epoch": None, "wall_s": None,
    }
    results = pd.concat(
        [results, pd.DataFrame([persist_row])], ignore_index=True
    ).sort_values("val_rmse").reset_index(drop=True)
    results.to_csv(OUT_DIR / "stage_b_results.csv", index=False)

    lines += ["", "— val results, all 12 runs + persistence (sorted by RMSE) —",
              fmt_df(results.drop(columns=["seed"]))]

    # ---- exhibits: best config per target -------------------------------- #
    for target in TARGETS:
        best = min((r for r in rows if r["target"] == target),
                   key=lambda r: r["val_rmse"])
        pred = best["pred_val"]
        lines += ["", f"=== best {target}: {best['run']} "
                  f"(val RMSE {best['val_rmse']:.6f}, skill {best['skill']:+.4f}) ==="]

        pt = per_ticker_table(pred, d["y_val"], d["tickers_val"],
                              pred_persistence=d["trail_val"])
        mean_row = pd.DataFrame([{
            "ticker": "MEAN(eq-wt)", "n": pt["n"].sum(),
            "rmse": pt["rmse"].mean(), "rmse_persist": pt["rmse_persist"].mean(),
            "skill": pt["skill"].mean(),
        }])
        lines += ["", "per-ticker RMSE and skill (val):",
                  fmt_df(pd.concat([pt, mean_row], ignore_index=True))]

        lines += ["", "decile calibration (each forecaster bucketed by its own "
                  "predicted decile):",
                  fmt_df(decile_side_by_side(pred, d["trail_val"], d["y_val"]))]

        lines += ["", "conditional skill by within-ticker decile of vol_trail_20 "
                  "(pre-registered exhibit; diagnostic only):",
                  fmt_df(conditional_skill_table(
                      pred, d["trail_val"], d["y_val"],
                      d["trail_val"], d["tickers_val"]))]

    # ---- selection + stability ------------------------------------------- #
    winner, tie_note = select_winner(rows)
    lines += ["", "— selection (pre-registered rule) —", *tie_note,
              f"SELECTED: {winner['run']} — target {winner['target']}, arch "
              f"{winner['arch']} ({winner['layers']}x{winner['hidden']} "
              f"dropout {winner['dropout']}, {winner['params']} params), "
              f"best epoch {winner['best_epoch']}, val RMSE "
              f"{winner['val_rmse']:.6f}, skill {winner['skill']:+.4f}",
              f"frozen checkpoint: outputs/task8/models/"
              f"{winner['target']}_{winner['arch']}_seed42.pt"]

    seed_rmses = {SEED: winner["val_rmse"]}
    for s in STABILITY_SEEDS:
        r = train_one_run(winner["target"], winner["arch"], s, d)
        seed_rmses[s] = r["val_rmse"]
        print(f"stability seed {s}: val RMSE {r['val_rmse']:.6f}", flush=True)

    vals = np.array(list(seed_rmses.values()))
    runner_up = min(
        (r for r in rows if r["run"] != winner["run"]),
        key=lambda r: r["val_rmse"],
    )
    persist_rmse = STAGE_A_VAL_PERSIST["rmse"]
    lines += ["", "— stability check, winner only (seeds 42/43/44) —",
              "  " + " | ".join(f"seed {s}: {v:.6f}" for s, v in seed_rmses.items()),
              f"  mean {vals.mean():.6f} | min {vals.min():.6f} | max {vals.max():.6f}",
              f"  persistence bar {persist_rmse:.6f}: "
              + ("seed spread STRADDLES the bar"
                 if vals.min() < persist_rmse < vals.max()
                 else ("all seeds beat the bar" if vals.max() < persist_rmse
                       else "no seed beats the bar")),
              f"  runner-up {runner_up['run']} at {runner_up['val_rmse']:.6f}: "
              + ("worst seed REORDERS the top of the table"
                 if vals.max() > runner_up["val_rmse"]
                 else "no reordering at any seed")]

    lines += ["", "STOP — Stage B complete. No test-set contact. Stage C "
              f"evaluates ONLY {winner['run']} (seed-42 weights, best epoch "
              f"{winner['best_epoch']}) against persistence, once."]

    report = "\n".join(lines)
    (OUT_DIR / "stage_b_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, RuntimeError) as e:
        print(f"STAGE B FAILURE — STOP: {e}", file=sys.stderr)
        sys.exit(1)
