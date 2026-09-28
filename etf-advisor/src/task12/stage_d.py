"""Task 12 Stage D driver: LSTM post-mortem (quarantined diagnostic).

DeepSHAP over the frozen Task 8 LSTM checkpoint.
This stage is quarantined: its output is a diagnostic exhibit only and feeds
nothing back into any Task 12 verdict, any Task 8-11 result, or any chapter
claim other than the single permitted sentence.
"""

from __future__ import annotations

import hashlib
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn

from ..data.config import FEATURE_ORDER, PROCESSED_DIR, ROOT, WINDOW
from ..task8.modeling import build_model
from .fixtures import IMPORT_MAP_STAGE_D, OUT_DIR
from .reporting import env_header, md_table, write_report


APPENDIX_DIR = OUT_DIR / "appendix_lstm"
CKPT_PATH = ROOT / "outputs" / "task8" / "models" / "T1_A1_seed42.pt"

# constants.
N_BACKGROUND = 100
BACKGROUND_SEED = 42
TORCH_SEED = 42
D3_REL_TOL = 1e-3           # float32-class check, NOT an engine-precision gate

STAMP = "diagnostic exhibit only; feeds nothing back."

# Every frozen data file this driver opens, recorded for the Gate D1 affirmation.
OPENED: list[str] = []


def _load_split(name: str) -> np.ndarray:
    """The only data door in this module. Admits train and val; never test."""
    if name not in ("train", "val"):
        raise ValueError(
            f"Stage D may only read the train and val splits, got {name!r} -- "
            "the test-period firewall is structural, not advisory"
        )
    path = PROCESSED_DIR / f"{name}.npz"
    OPENED.append(str(path))
    with np.load(path) as z:
        return z["X"]


class Rank2(nn.Module):
    """The frozen model with its output kept at [B, 1].

    ``LSTMVol.forward`` ends with ``.squeeze(-1)`` and returns shape [B]; shap's
    ``PyTorchDeep`` indexes ``outputs.shape[1]`` and therefore requires
    [B, n_outputs]. This adapter re-adds exactly the dimension the frozen
    forward squeezes off. No weight, no activation and no output VALUE is
    altered -- ``Rank2(m)(x)`` is ``m(x)`` with a trailing axis of length 1.
    ``src/task8`` is untouched.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x).unsqueeze(-1)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Frozen input schema (reported before anything is computed)
# --------------------------------------------------------------------------- #
def schema_section(lines: list[str], X_train: np.ndarray, X_val: np.ndarray) -> None:
    n_lags, n_channels = X_train.shape[1], X_train.shape[2]
    multichannel = n_channels > 1
    assert n_lags == WINDOW, (
        f"frozen arrays carry {n_lags} lags, config says WINDOW={WINDOW} -- STOP"
    )
    assert n_channels == len(FEATURE_ORDER), (
        f"frozen arrays carry {n_channels} channels, config names "
        f"{len(FEATURE_ORDER)} -- STOP"
    )
    lines += [
        "## Frozen input schema (read from the arrays, reported before any "
        "attribution is computed)",
        "",
        md_table(
            ["array", "shape", "dtype", "meaning"],
            [
                [f"`train.npz['X']`", f"{X_train.shape}", f"{X_train.dtype}",
                 "background universe"],
                [f"`val.npz['X']`", f"{X_val.shape}", f"{X_val.dtype}",
                 "explicand universe (ALL validation windows)"],
            ],
        ),
        "",
        f"The frozen input is **multichannel**: each sample is a {n_lags} x "
        f"{n_channels} window -- {n_lags} lag positions by {n_channels} input "
        "channels. The registered summary is to be aggregated "
        "BOTH by lag position AND by input channel. Channel order "
        "(the stored column order of every window):",
        "",
        md_table(
            ["channel index", "name"],
            [[i, f"`{name}`"] for i, name in enumerate(FEATURE_ORDER)],
        ),
        "",
        f"Lag convention: index 0 is the OLDEST step in the window and index "
        f"{n_lags - 1} is the step at `window_end`, i.e. the most recent "
        "observation. The frozen architecture reads the last timestep's hidden "
        "state (`out[:, -1, :]`), so lag index "
        f"{n_lags - 1} is the one the head sees directly.",
        "",
        f"Multichannel: {multichannel}. Model output explained: the network's "
        "own scalar output, i.e. the standardised T1 prediction z before the "
        "checkpoint's raw-vol inversion. The inversion is a post-hoc affine map "
        "with a floor and is not part of the network.",
    ]


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #
def gate_d1(lines: list[str], ckpt_meta: dict) -> str:
    digest = sha256(CKPT_PATH)
    opened = sorted(set(OPENED))
    forbidden = [p for p in opened if "test" in p.lower()]
    assert not forbidden, (
        f"GATE D1: a test-period artifact was opened: {forbidden} -- STOP"
    )
    lines += [
        "**GATE D1 PASS** -- posture affirmations.",
        "",
        md_table(
            ["affirmation", "evidence"],
            [
                ["`predictions_test.parquet` not opened",
                 "never referenced in `src/task12`; the only data door in this "
                 "module is `_load_split`, which reads `data/processed/"
                 "{train,val}.npz` and raises on any other split"],
                ["no test-period windows used anywhere",
                 "background drawn from `train.npz`, explicands from `val.npz`; "
                 "the files actually opened are recorded below"],
                ["files opened by this stage", "<br>".join(f"`{p}`" for p in opened)],
                ["checkpoint path", f"`{CKPT_PATH}`"],
                ["checkpoint SHA-256", f"`{digest}`"],
                ["checkpoint metadata",
                 ", ".join(f"{k}={ckpt_meta[k]!r}" for k in
                           ("target", "arch_id", "seed", "best_epoch"))],
                ["output stamp", f"\"{STAMP}\""],
            ],
        ),
        "",
        "No prior SHA-256 of this checkpoint exists anywhere in the repository, so this "
        "recording CREATES the anchor, per the Task 9 Gate B2 precedent. It is a "
        "record, not a comparison.",
        "",
        "Rationale for explicands = validation, not test: "
        "the attribution profile is a property of the frozen weights and is "
        "visible on any samples; val is already spent for selection, so no "
        "further contact with test occurs.",
    ]
    return digest


def gate_d2(lines: list[str]) -> None:
    import pandas as pd
    import pyarrow
    import scipy
    import shap

    pins = [
        ["numpy", "2.4.6", np.__version__],
        ["pandas", "3.0.3", pd.__version__],
        ["scipy", "1.17.1", scipy.__version__],
        ["pyarrow", "25.0.0", pyarrow.__version__],
    ]
    bad = [(n, want, got) for n, want, got in pins if want != got]

    try:
        import llvmlite
        import numba
        extra = f"numba {numba.__version__}, llvmlite {llvmlite.__version__}"
    except ImportError:  # pragma: no cover - recorded either way
        extra = "numba / llvmlite not importable"

    lines += [
        ("**GATE D2 PASS** -- " if not bad else "**GATE D2 FAIL** -- ")
        + "single dependency install attempt into the venv, followed by pin "
        "verification BY IMPORT:",
        "",
        md_table(
            ["pinned package", "registered version", "imported version", "verdict"],
            [[n, want, got, "UNDISTURBED" if want == got else "**DISTURBED**"]
             for n, want, got in pins],
        ),
        "",
        f"- installed this stage: shap {shap.__version__}, {extra}, plus "
        "`slicer`, `cloudpickle`, `tqdm` (pure additions).",
        f"- torch {torch.__version__}, CPU execution, `torch.manual_seed("
        f"{TORCH_SEED})`.",
        "- The install added packages only: no pinned package was upgraded, "
        "downgraded or removed. `shap` declares `numpy>=2`, which the "
        "registered numpy 2.4.6 already satisfies.",
    ]
    assert not bad, f"GATE D2: pinned versions disturbed: {bad} -- STOP"


def gate_d3(
    lines: list[str],
    phi_sum: np.ndarray,
    fx: np.ndarray,
    f_bg_mean: float,
) -> bool:
    gap = fx - f_bg_mean
    abs_dev = np.abs(phi_sum - gap)
    rel = abs_dev / np.abs(gap)
    worst_rel = float(np.nanmax(rel))
    worst_i = int(np.nanargmax(rel))
    passed = worst_rel < D3_REL_TOL

    lines += [
        ("**GATE D3 PASS** -- " if passed else "**GATE D3 FAIL** -- ")
        + "local accuracy: DeepSHAP attribution sums against "
        "`f(x) - mean over background of f`, per explicand. **This is a "
        "float32-class check, not an engine-precision gate** -- the network is "
        f"float32, so the registered threshold is {D3_REL_TOL:g} RELATIVE.",
        "",
        md_table(
            ["quantity", "value"],
            [
                ["explicands checked", f"{len(fx)}"],
                ["worst relative deviation", f"`{worst_rel!r}` (= {worst_rel:.4g})"],
                ["registered threshold", f"{D3_REL_TOL:g} relative"],
                ["at explicand index", f"{worst_i}"],
                ["  its sum(phi)", f"`{float(phi_sum[worst_i])!r}`"],
                ["  its f(x) - mean f(bg)", f"`{float(gap[worst_i])!r}`"],
                ["median relative deviation", f"{float(np.median(rel)):.4g}"],
                ["fraction of explicands over threshold",
                 f"{float((rel >= D3_REL_TOL).mean()):.4f}"],
                ["worst absolute deviation", f"{float(abs_dev.max()):.6g}"],
                ["mean f over background", f"`{f_bg_mean!r}`"],
            ],
        ),
    ]
    return passed


# --------------------------------------------------------------------------- #
# Registered summary and deliverables (written only if Gate D3 passes)
# --------------------------------------------------------------------------- #
def write_summary(phi: np.ndarray) -> tuple[str, str]:
    """Mean absolute attribution by lag position and by input channel."""
    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
    by_lag = np.abs(phi).mean(axis=(0, 2))        # (30,)
    by_channel = np.abs(phi).mean(axis=(0, 1))    # (6,)
    by_lag_channel = np.abs(phi).mean(axis=0)     # (30, 6)

    rows = ["scope,index,name,mean_abs_attribution"]
    for i, v in enumerate(by_lag):
        rows.append(f"lag,{i},lag_{i},{float(v)!r}")
    for j, v in enumerate(by_channel):
        rows.append(f"channel,{j},{FEATURE_ORDER[j]},{float(v)!r}")
    for i in range(by_lag_channel.shape[0]):
        for j in range(by_lag_channel.shape[1]):
            rows.append(
                f"lag_channel,{i},lag_{i}|{FEATURE_ORDER[j]},"
                f"{float(by_lag_channel[i, j])!r}"
            )
    csv_path = APPENDIX_DIR / "attribution_by_lag.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    fig_path = _figure(by_lag, by_channel)
    return str(csv_path), fig_path


def _figure(by_lag: np.ndarray, by_channel: np.ndarray) -> str:
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

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.0, 5.2), width_ratios=[2.2, 1.0], constrained_layout=True
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
    ax1.set_xlabel("lag position  (0 = oldest step, "
                   f"{len(by_lag) - 1} = window_end)",
                   color=INK_SECONDARY, fontsize=10)
    ax1.set_ylabel("mean |attribution|", color=INK_SECONDARY, fontsize=10)
    ax1.set_title("by lag position", color=INK_PRIMARY, fontsize=11, loc="left")

    ax2.bar(range(len(by_channel)), by_channel, color=SLOT_2_ORANGE, width=0.66,
            edgecolor=SURFACE, linewidth=1.2)
    ax2.set_xticks(range(len(by_channel)))
    ax2.set_xticklabels(FEATURE_ORDER, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel("mean |attribution|", color=INK_SECONDARY, fontsize=10)
    ax2.set_title("by input channel", color=INK_PRIMARY, fontsize=11, loc="left")

    fig.suptitle(
        "Appendix: DeepSHAP attribution profile of the frozen Task 8 LSTM "
        f"(T1xA1, seed 42)\n{STAMP}",
        color=INK_PRIMARY, fontsize=12.5, ha="left", x=0.008,
    )
    path = APPENDIX_DIR / "attribution_by_lag.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    import shap

    # Ratified: stage_d_report.md is the Gate D3 FAILURE RECORD and is
    # preserved verbatim and untouched hereafter.
    existing = OUT_DIR / "stage_d_report.md"
    if existing.exists() and "GATE D3 FAIL" in existing.read_text(encoding="utf-8"):
        raise SystemExit(
            f"{existing} is the ratified Gate D3 failure record; regenerating it "
            "would rewrite chronology. Refusing to overwrite. The substituted "
            "Integrated-Gradients post-mortem is `src.task12.stage_d2`."
        )

    torch.manual_seed(TORCH_SEED)

    lines = env_header(
        "Task 12 Stage D report -- LSTM post-mortem (quarantined diagnostic)",
    )
    lines += [
        f"**{STAMP.upper()}**",
        "",
        "DeepSHAP over the frozen Task 8 LSTM checkpoint. This stage "
        "is quarantined: nothing computed here feeds back into the Task 12 "
        "verdict, into any Task 8-11 result, or into any chapter claim beyond "
        "the single permitted sentence.",
        "",
    ]
    try:
        X_train = _load_split("train")
        X_val = _load_split("val")
        schema_section(lines, X_train, X_val)

        ckpt = torch.load(CKPT_PATH, weights_only=True)
        inner = build_model(ckpt["arch_id"])
        inner.load_state_dict(ckpt["state_dict"])
        inner.eval()
        model = Rank2(inner)
        model.eval()

        lines += ["", "## Gates", ""]
        gate_d1(lines, ckpt)
        lines.append("")
        gate_d2(lines)
        lines.append("")

        rng = np.random.default_rng(BACKGROUND_SEED)
        bg_idx = rng.choice(len(X_train), size=N_BACKGROUND, replace=False)
        background = torch.from_numpy(X_train[bg_idx].astype(np.float32))
        explicands = torch.from_numpy(X_val.astype(np.float32))

        caught: list[str] = []
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            explainer = shap.DeepExplainer(model, background)
            raw = explainer.shap_values(explicands, check_additivity=False)
            caught = sorted({f"{x.category.__name__}: {x.message}" for x in w})

        phi = np.asarray(raw)
        if phi.ndim == 4 and phi.shape[-1] == 1:
            phi = phi[..., 0]
        assert phi.shape == X_val.shape, (
            f"attribution shape {phi.shape} != explicand shape {X_val.shape} -- STOP"
        )

        with torch.no_grad():
            fx = model(explicands).numpy().ravel().astype(np.float64)
            f_bg_mean = float(model(background).numpy().ravel().astype(np.float64).mean())
        phi_sum = phi.reshape(phi.shape[0], -1).sum(axis=1)

        lines += [
            "**Registered computation.** Background: "
            f"{N_BACKGROUND} training windows drawn WITHOUT replacement via "
            f"`numpy.random.default_rng({BACKGROUND_SEED})` (the Task 9 Gate A2 "
            f"sampling convention). Explicands: ALL {len(X_val)} validation "
            "windows -- deterministic, no sampling choice. CPU execution, "
            f"`torch.manual_seed({TORCH_SEED})`.",
            "",
        ]
        if caught:
            lines += [
                "**Warnings raised by the explainer, recorded verbatim:**",
                "",
                *[f"- `{c}`" for c in caught],
                "",
            ]

        d3_ok = gate_d3(lines, phi_sum, fx, f_bg_mean)

        if not d3_ok:
            lines += [
                "",
                "## Diagnosis (Gate D3 FAIL -- halt)",
                "",
                "The failure is structural, not marginal. `shap` 0.52.0's "
                "PyTorch `DeepExplainer` walks the module graph and dispatches "
                "each module to an op handler; the frozen architecture's "
                "`nn.LSTM` is a fused recurrent module for which that handler "
                "table has **no entry**, which is exactly what the recorded "
                "`unrecognized nn.Module: LSTM` warning reports. The explainer "
                "does not fail loudly on the unknown module -- it falls through "
                "to a default treatment that does not implement the "
                "rescale/reveal-cancel rule for a recurrent cell, so the "
                "returned values are not DeepSHAP attributions for this network "
                "and do not decompose its output. The measured worst relative "
                "deviation above is orders of magnitude beyond the registered "
                f"{D3_REL_TOL:g}; a float32 rounding story cannot account for it.",
                "",
                "**Nothing is patched forward. ** "
                "No appendix deliverable is written on a "
                "failed stage -- publishing a by-lag profile computed from "
                "values that fail local accuracy would present an artifact of "
                "the explainer as a property of the model.",
                "",
                "**Scope of the failure, stated precisely.** This is a finding "
                "about the DeepSHAP method's applicability to a fused LSTM under "
                "the pinned library, not about the frozen model and not about "
                "Stages A-C. The Stage C verdict is untouched: Stages A-C use "
                "the own-implementation exact enumeration, share no code with "
                "this stage, and never invoke the shap library.",
            ]
            raise AssertionError(
                f"GATE D3: worst relative deviation exceeds {D3_REL_TOL:g} -- "
                "DeepSHAP does not satisfy local accuracy on the frozen LSTM "
                "(unrecognized nn.Module: LSTM) -- STOP"
            )

        csv_path, fig_path = write_summary(phi)
        by_lag = np.abs(phi).mean(axis=(0, 2))
        by_channel = np.abs(phi).mean(axis=(0, 1))
        dom_lag = int(np.argmax(by_lag))
        dom_ch = int(np.argmax(by_channel))
        lines += [
            "",
            "## Registered summary",
            "",
            md_table(
                ["scope", "dominant", "mean |attribution|", "share of total"],
                [
                    ["lag position", f"lag {dom_lag}", f"{by_lag[dom_lag]:.6g}",
                     f"{by_lag[dom_lag] / by_lag.sum():.4f}"],
                    ["input channel", f"`{FEATURE_ORDER[dom_ch]}`",
                     f"{by_channel[dom_ch]:.6g}",
                     f"{by_channel[dom_ch] / by_channel.sum():.4f}"],
                ],
            ),
            "",
            f"- `{csv_path}`",
            f"- `{fig_path}`",
        ]
    except AssertionError as exc:
        lines += [
            "",
            f"**GATE FAILURE -- STOP**: {exc}",
            "",
            "Diagnosis recorded above; nothing patched. "
            "No appendix deliverable is written on a failed stage.",
            "",
            "**STOP -- Stage D boundary.**",
        ]
        print(write_report("stage_d_report.md", lines))
        sys.exit(1)

    lines += [
        "",
        "## Notes",
        "",
        f"- Output stamp: \"{STAMP}\"",
        "- `predictions_test.parquet` was not opened. No test-period window was "
        "used as background or as an explicand.",
        "",
        "**STOP -- Stage D boundary. Task 12 execution complete.**",
    ]
    print(write_report("stage_d_report.md", lines))


if __name__ == "__main__":
    main()
