"""Task 12 Stage D2: LSTM post-mortem by Integrated Gradients.

DeepSHAP under the pinned shap 0.52.0
has no op handler for the fused ``nn.LSTM`` and fails local accuracy
structurally; that measurement is the Gate D3 failure record in
``stage_d_report.md``, which this stage does NOT modify. The post-mortem method
is substituted, not the post-mortem.

Method: Integrated Gradients in pure PyTorch autograd. **The shap library
appears nowhere in this module** -- not imported, not referenced. Straight-line
path from baseline to explicand, midpoint Riemann rule, 128 steps. A
completeness failure at 128 steps is a HALT-and-report, never a step-count
negotiation.

Quarantine unchanged: this stage is a diagnostic exhibit only and feeds nothing
back into the Task 12 verdict, into any Task 8-11 result, or into any chapter
claim beyond the single permitted sentence.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone

import numpy as np
import torch

from ..data.config import FEATURE_ORDER, WINDOW
from ..task8.modeling import build_model
from .fixtures import OUT_DIR
from .reporting import env_header, md_table, write_report
from .stage_d import (
    APPENDIX_DIR,
    BACKGROUND_SEED,
    CKPT_PATH,
    N_BACKGROUND,
    OPENED,
    STAMP,
    TORCH_SEED,
    _load_split,
    sha256,
)


IG_STEPS = 128                 # midpoint Riemann rule; no unilateral escalation
D2_3_ABS_TOL = 1e-3            # ABSOLUTE, on the standardized output scale
EXPLICAND_BATCH = 128          # execution detail only; changes no value
METHOD_LABEL = (
    "Integrated Gradients (amendment T12-A3; DeepSHAP inapplicable -- see the "
    "Stage D failure record)"
)
STAGE_D_REPORT = OUT_DIR / "stage_d_report.md"


def stage_d_anchor() -> str:
    """The checkpoint SHA-256 recorded by Stage D, read from its report."""
    text = STAGE_D_REPORT.read_text(encoding="utf-8")
    hits = re.findall(r"\|\s*checkpoint SHA-256\s*\|\s*`([0-9a-f]{64})`\s*\|", text)
    assert len(hits) == 1, (
        f"GATE D2-1: {len(hits)} checkpoint SHA-256 rows in "
        f"{STAGE_D_REPORT.name}, expected exactly 1 -- cannot identify the "
        "anchor, STOP"
    )
    return hits[0]


# --------------------------------------------------------------------------- #
# Integrated Gradients (pure autograd; no shap)
# --------------------------------------------------------------------------- #
def integrated_gradients(
    model: torch.nn.Module,
    explicands: torch.Tensor,
    baseline: torch.Tensor,
    steps: int = IG_STEPS,
) -> np.ndarray:
    """IG with the midpoint Riemann rule on the straight-line path.

    IG_i(x) = (x_i - b_i) * (1/m) * sum_k d f(b + a_k (x - b)) / d x_i
    with a_k = (k + 0.5) / m -- the midpoint rule, so no endpoint is evaluated.
    """
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    out = np.empty(tuple(explicands.shape), dtype=np.float64)

    for start in range(0, len(explicands), EXPLICAND_BATCH):
        x = explicands[start:start + EXPLICAND_BATCH]
        delta = x - baseline                      # (B, 30, 6)
        grad_sum = torch.zeros_like(x, dtype=torch.float64)
        for alpha in alphas:
            point = (baseline + alpha * delta).clone().requires_grad_(True)
            y = model(point)                      # (B,)
            grad = torch.autograd.grad(y.sum(), point)[0]
            grad_sum += grad.to(torch.float64)
        out[start:start + EXPLICAND_BATCH] = (
            (delta.to(torch.float64) * grad_sum / steps).numpy()
        )
    return out


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #
def gate_d2_1(lines: list[str], ckpt_meta: dict, opened: list[str],
              bg_idx: np.ndarray, model: torch.nn.Module) -> None:
    digest = sha256(CKPT_PATH)
    anchor = stage_d_anchor()
    matches = digest == anchor
    forbidden = [p for p in opened if "test" in p.lower()]

    # The background draw is reproduced through Stage D's own constants; a second
    # independent draw with a fresh generator must give the identical index set.
    check = np.random.default_rng(BACKGROUND_SEED).choice(
        int(_POPULATION[0]), size=N_BACKGROUND, replace=False
    )
    identical = bool(np.array_equal(np.sort(bg_idx), np.sort(check)))

    lines += [
        ("**GATE D2-1 PASS** -- " if (matches and not forbidden and identical)
         else "**GATE D2-1 FAIL** -- ") + "posture affirmations.",
        "",
        md_table(
            ["affirmation", "evidence"],
            [
                ["`predictions_test.parquet` not opened",
                 "never referenced in `src/task12`; this stage IMPORTS Stage D's "
                 "`_load_split`, the structural data door that raises on any "
                 "split but train/val -- the firewall is retained, not restated"],
                ["no test-period windows used anywhere",
                 "<br>".join(f"`{p}`" for p in sorted(set(opened)))],
                ["checkpoint path", f"`{CKPT_PATH}`"],
                ["checkpoint SHA-256 (recomputed)", f"`{digest}`"],
                ["Stage D anchor (read from `stage_d_report.md`)", f"`{anchor}`"],
                ["anchor comparison",
                 "**MATCH**" if matches else "**MISMATCH -- STOP**"],
                ["checkpoint metadata",
                 ", ".join(f"{k}={ckpt_meta[k]!r}" for k in
                           ("target", "arch_id", "seed", "best_epoch"))],
                ["model eval mode asserted",
                 f"`model.training` = {model.training} (False required)"],
                ["parameter grads disabled",
                 f"{sum(p.requires_grad for p in model.parameters())} of "
                 f"{len(list(model.parameters()))} parameters require grad "
                 "(0 required: gradients are taken w.r.t. the INPUT only)"],
                ["background index set identical to Stage D's",
                 f"{identical} -- same `numpy.random.default_rng("
                 f"{BACKGROUND_SEED})`, same size {N_BACKGROUND}, same "
                 "population; reproduced with a fresh generator and compared"],
                ["output stamp", f"\"{STAMP}\""],
            ],
        ),
        "",
        "This is the checkpoint hash's FIRST USE AS A COMPARISON. Stage D "
        "created the anchor (no prior hash existed anywhere in the repository); "
        "Stage D2 reads it back from the file of record at execution time and "
        "compares, so the two stages are certified to have explained the same "
        "frozen weights.",
    ]
    assert not forbidden, f"GATE D2-1: test artifact opened: {forbidden} -- STOP"
    assert matches, (
        f"GATE D2-1: checkpoint SHA-256 {digest} != Stage D anchor {anchor} -- "
        "the frozen checkpoint changed between stages, STOP"
    )
    assert identical, "GATE D2-1: background index set differs from Stage D -- STOP"
    assert not model.training, "GATE D2-1: model is not in eval mode -- STOP"


def gate_d2_2(lines: list[str]) -> None:
    import pandas as pd
    import pyarrow
    import scipy

    pins = [
        ["numpy", "2.4.6", np.__version__],
        ["pandas", "3.0.3", pd.__version__],
        ["scipy", "1.17.1", scipy.__version__],
        ["pyarrow", "25.0.0", pyarrow.__version__],
    ]
    bad = [(n, want, got) for n, want, got in pins if want != got]
    lines += [
        ("**GATE D2-2 PASS** -- " if not bad else "**GATE D2-2 FAIL** -- ")
        + "environment: NO new installs were made for this stage. The four "
        "registered pins verified by import:",
        "",
        md_table(
            ["pinned package", "registered version", "imported version", "verdict"],
            [[n, want, got, "UNDISTURBED" if want == got else "**DISTURBED**"]
             for n, want, got in pins],
        ),
        "",
        f"- torch {torch.__version__}, CPU execution, `torch.manual_seed("
        f"{TORCH_SEED})`.",
        "- **The shap library is not imported, referenced or used in this "
        "module.** Integrated Gradients is computed with `torch.autograd.grad` "
        "alone. shap remains installed from Stage D's ratified attempt but "
        "plays no part here.",
    ]
    assert not bad, f"GATE D2-2: pinned versions disturbed: {bad} -- STOP"


def gate_d2_3(lines: list[str], ig_sum: np.ndarray, fx: np.ndarray,
              f_base: float) -> bool:
    gap = fx - f_base
    abs_dev = np.abs(ig_sum - gap)
    worst_abs = float(abs_dev.max())
    worst_i = int(np.argmax(abs_dev))
    rel = abs_dev / np.abs(gap)
    passed = worst_abs < D2_3_ABS_TOL

    lines += [
        ("**GATE D2-3 PASS** -- " if passed else "**GATE D2-3 FAIL** -- ")
        + "completeness: per explicand, "
        "`|sum(IG) - (f(x) - f(x_baseline))|`, gated **ABSOLUTE** at "
        f"{D2_3_ABS_TOL:g} on the standardized output scale (amended criterion, "
        "registered by T12-A3 BEFORE any IG value existed):",
        "",
        md_table(
            ["quantity", "value"],
            [
                ["explicands checked", f"{len(fx)}"],
                ["worst ABSOLUTE deviation (gated)",
                 f"`{worst_abs!r}` (= {worst_abs:.4g})"],
                ["registered threshold", f"{D2_3_ABS_TOL:g} absolute"],
                ["at explicand index", f"{worst_i}"],
                ["  its sum(IG)", f"`{float(ig_sum[worst_i])!r}`"],
                ["  its f(x) - f(x_baseline)", f"`{float(gap[worst_i])!r}`"],
                ["median ABSOLUTE deviation", f"{float(np.median(abs_dev)):.4g}"],
                ["fraction over the absolute threshold",
                 f"{float((abs_dev >= D2_3_ABS_TOL).mean()):.4f}"],
                ["worst RELATIVE deviation (descriptive, NOT gated)",
                 f"{float(np.max(rel)):.4g}"],
                ["median RELATIVE deviation (descriptive, NOT gated)",
                 f"{float(np.median(rel)):.4g}"],
                ["smallest |f(x) - f(x_baseline)| over explicands",
                 f"{float(np.abs(gap).min()):.4g}"],
                ["f(x_baseline)", f"`{f_base!r}`"],
            ],
        ),
        "",
        "**Why absolute, recorded.** Stage D's pure-relative criterion was "
        "pathological under near-zero denominators -- its worst case had "
        "`|f(x) - mean f(bg)| = 8.937932550907163e-05`, so the relative form "
        "partly measured denominator size rather than attribution error. The "
        "explained output is the standardized z, which is O(1), so an absolute "
        f"{D2_3_ABS_TOL:g} is the meaningful float32-class criterion here. The "
        "relative figures are reported above for continuity with the Stage D "
        "record; they are descriptive and carry no gate.",
    ]
    return passed


# --------------------------------------------------------------------------- #
# Registered summary and deliverables
# --------------------------------------------------------------------------- #
def write_summary(ig: np.ndarray) -> tuple[str, str, np.ndarray, np.ndarray]:
    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
    by_lag = np.abs(ig).mean(axis=(0, 2))       # (30,)
    by_channel = np.abs(ig).mean(axis=(0, 1))   # (6,)
    by_lag_channel = np.abs(ig).mean(axis=0)    # (30, 6)

    rows = ["method,scope,index,name,mean_abs_attribution"]
    tag = "integrated_gradients_T12A3"
    for i, v in enumerate(by_lag):
        rows.append(f"{tag},lag,{i},lag_{i},{float(v)!r}")
    for j, v in enumerate(by_channel):
        rows.append(f"{tag},channel,{j},{FEATURE_ORDER[j]},{float(v)!r}")
    for i in range(by_lag_channel.shape[0]):
        for j in range(by_lag_channel.shape[1]):
            rows.append(
                f"{tag},lag_channel,{i},lag_{i}|{FEATURE_ORDER[j]},"
                f"{float(by_lag_channel[i, j])!r}"
            )
    csv_path = APPENDIX_DIR / "attribution_by_lag.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(csv_path), _figure(by_lag, by_channel), by_lag, by_channel


def _figure(by_lag: np.ndarray, by_channel: np.ndarray) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .exhibits import (
        BASELINE,
        GRID,
        INK_MUTED,
        INK_PRIMARY,
        INK_SECONDARY,
        SLOT_1_BLUE,
        SLOT_2_ORANGE,
        SURFACE,
    )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.4, 5.4), width_ratios=[2.3, 1.0], constrained_layout=True
    )
    fig.patch.set_facecolor(SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(BASELINE)
        ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, length=3)
        ax.grid(axis="y", color=GRID, linewidth=1.0)
        ax.set_axisbelow(True)

    ax1.bar(range(len(by_lag)), by_lag, color=SLOT_1_BLUE, width=0.72,
            edgecolor=SURFACE, linewidth=1.2)
    ax1.set_xlabel(
        f"lag position   (0 = oldest step in the window, {len(by_lag) - 1} = "
        "window_end, the step the head reads)",
        color=INK_SECONDARY, fontsize=10,
    )
    ax1.set_ylabel("mean |IG attribution|", color=INK_SECONDARY, fontsize=10)
    ax1.set_title("by lag position", color=INK_PRIMARY, fontsize=11.5, loc="left")

    ax2.bar(range(len(by_channel)), by_channel, color=SLOT_2_ORANGE, width=0.66,
            edgecolor=SURFACE, linewidth=1.2)
    ax2.set_xticks(range(len(by_channel)))
    ax2.set_xticklabels(FEATURE_ORDER, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel("mean |IG attribution|", color=INK_SECONDARY, fontsize=10)
    ax2.set_title("by input channel", color=INK_PRIMARY, fontsize=11.5, loc="left")

    fig.suptitle(
        "Appendix: attribution profile of the frozen Task 8 LSTM (T1xA1, seed 42)\n"
        f"{METHOD_LABEL}   |   {STAMP}",
        color=INK_PRIMARY, fontsize=12, ha="left", x=0.006,
    )
    path = APPENDIX_DIR / "attribution_by_lag.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
_POPULATION: list[int] = [0]   # train-window count, filled at run time


def main() -> None:
    torch.manual_seed(TORCH_SEED)

    lines = env_header(
        "Task 12 Stage D2 report -- LSTM post-mortem by Integrated Gradients",
    )
    lines += [
        f"**{STAMP.upper()}**",
        "",
        f"Method: {METHOD_LABEL}. Registered after the Stage D "
        "measurement established that shap 0.52.0's PyTorch "
        "`DeepExplainer` has no op handler for the fused `nn.LSTM` and fails "
        "local accuracy structurally. `stage_d_report.md` is that failure "
        "record and is NOT modified by this stage; `stage_d.py` carries an "
        "overwrite guard refusing to regenerate it.",
        "",
        "Quarantine unchanged: nothing computed here feeds back into the Task 12 "
        "verdict, into any Task 8-11 result, or into any chapter claim beyond "
        "the single permitted sentence.",
        "",
    ]
    try:
        X_train = _load_split("train")
        X_val = _load_split("val")
        _POPULATION[0] = len(X_train)
        # _load_split is Stage D's; it records every file it opens in OPENED.
        opened = list(OPENED)
        assert X_train.shape[1] == WINDOW and X_train.shape[2] == len(FEATURE_ORDER)

        ckpt = torch.load(CKPT_PATH, weights_only=True)
        model = build_model(ckpt["arch_id"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        rng = np.random.default_rng(BACKGROUND_SEED)
        bg_idx = rng.choice(len(X_train), size=N_BACKGROUND, replace=False)
        background = torch.from_numpy(X_train[bg_idx].astype(np.float32))
        baseline = background.mean(dim=0, keepdim=True)     # (1, 30, 6)
        explicands = torch.from_numpy(X_val.astype(np.float32))

        lines += ["## Gates", ""]
        gate_d2_1(lines, ckpt, opened, bg_idx, model)
        lines.append("")
        gate_d2_2(lines)
        lines.append("")

        ig = integrated_gradients(model, explicands, baseline, IG_STEPS)
        assert ig.shape == X_val.shape, (
            f"IG shape {ig.shape} != explicand shape {X_val.shape} -- STOP"
        )

        with torch.no_grad():
            fx = model(explicands).numpy().ravel().astype(np.float64)
            f_base = float(model(baseline).numpy().ravel()[0])
        ig_sum = ig.reshape(ig.shape[0], -1).sum(axis=1)

        passed = gate_d2_3(lines, ig_sum, fx, f_base)

        # Frozen weights untouched: re-read the checkpoint and compare tensors.
        fresh = torch.load(CKPT_PATH, weights_only=True)["state_dict"]
        drift = [
            k for k, v in model.state_dict().items()
            if not torch.equal(v, fresh[k])
        ]
        assert not drift, f"GATE D2-1: frozen weights changed during the run: {drift}"


        if not passed:
            lines += [
                "",
                "## Diagnosis (Gate D2-3 FAIL -- halt, do not patch forward)",
                "",
                "The registered step count is 128 with the midpoint rule. A "
                "completeness failure at 128 steps is a HALT-and-report, NOT a "
                "step-count negotiation: raising the step count until the check "
                "passes would tune a registered constant against its own gate. "
                "No appendix deliverable is written on a failed stage.",
            ]
            raise AssertionError(
                f"GATE D2-3: worst absolute completeness deviation exceeds "
                f"{D2_3_ABS_TOL:g} at {IG_STEPS} midpoint steps -- STOP"
            )

        csv_path, fig_path, by_lag, by_channel = write_summary(ig)
        dom_lag = int(np.argmax(by_lag))
        dom_ch = int(np.argmax(by_channel))
        lag_share = by_lag / by_lag.sum()
        ch_share = by_channel / by_channel.sum()
        last8 = float(lag_share[-8:].sum())

        lines += [
            "",
            "## Registered summary -- computed, not pre-committed",
            "",
            "Mean absolute IG attribution aggregated by lag position and by "
            "input channel, over all "
            f"{len(X_val)} validation explicands. The frozen schema is "
            f"multichannel ({WINDOW} lags x {len(FEATURE_ORDER)} channels), so "
            "both aggregations are registered and both are reported.",
            "",
            md_table(
                ["lag", *[f"{i}" for i in range(WINDOW)]],
                [["mean abs IG", *[f"{v:.3e}" for v in by_lag]],
                 ["share", *[f"{v:.3f}" for v in lag_share]]],
            ),
            "",
            md_table(
                ["channel", *FEATURE_ORDER],
                [["mean abs IG", *[f"{v:.3e}" for v in by_channel]],
                 ["share", *[f"{v:.3f}" for v in ch_share]]],
            ),
            "",
            md_table(
                ["scope", "dominant", "mean abs IG", "share of total"],
                [
                    ["lag position", f"lag {dom_lag}"
                     + (" (window_end)" if dom_lag == WINDOW - 1 else ""),
                     f"{by_lag[dom_lag]:.6g}", f"{lag_share[dom_lag]:.4f}"],
                    ["input channel", f"`{FEATURE_ORDER[dom_ch]}`",
                     f"{by_channel[dom_ch]:.6g}", f"{ch_share[dom_ch]:.4f}"],
                ],
            ),
            "",
            f"- Concentration, recorded: the most recent "
            f"8 of {WINDOW} lag positions carry {last8:.4f} of the total mean "
            f"absolute attribution; a flat profile would put {8 / WINDOW:.4f} "
            "there.",
            f"- Dominant lag is index {dom_lag} of {WINDOW - 1}; dominant "
            f"channel is `{FEATURE_ORDER[dom_ch]}`.",
        ]
    except AssertionError as exc:
        lines += [
            "",
            f"**GATE FAILURE -- STOP**: {exc}",
            "",
            "Diagnosis recorded above; nothing patched. "
            "No appendix deliverable is written on a failed stage.",
            "",
            "**STOP -- Stage D2 boundary.**",
        ]
        print(write_report("stage_d2_report.md", lines))
        sys.exit(1)

    lines += [
        "",
        "## Deliverables (Stage D2, appendix placement)",
        "",
        f"- `{csv_path}`",
        f"- `{fig_path}` (two panels: by lag position, by input channel)",
        "- `outputs/task12/stage_d2_report.md`",
        "",
        f"Method label carried on every deliverable: \"{METHOD_LABEL}\".",
        "",
        "**STOP -- Stage D2 boundary. Task 12 execution complete.**",
    ]
    print(write_report("stage_d2_report.md", lines))


if __name__ == "__main__":
    main()
