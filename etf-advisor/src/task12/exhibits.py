"""Task 12 exhibits X1 and X2.

Both exhibits were fixed before any attribution existed and are shown for ALL
cells regardless of outcome: a measured failure is carried by these same
figures. This module only renders them.

X1 is a diverging encoding -- the blue/red poles with a
neutral gray midpoint, equal step count per arm,
symmetric about zero -- and every cell also carries its value as text, so
magnitude is never colour-alone. X2 is a two-series categorical scatter on
slots 1 and 2 (blue, orange), the first two slots, which validate on the
all-pairs pairlist that scatter forms require; both series are direct-labelled
in addition to the legend, and the full 96-row table lives in
``stability_probes.csv``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import NullFormatter

from .fixtures import OUT_DIR, TICKERS

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SLOT_1_BLUE = "#2a78d6"    # monitored pair
SLOT_2_ORANGE = "#eb6834"  # control pair
DIVERGING_NEG = "#2a78d6"  # blue pole
DIVERGING_MID = "#f0efec"  # neutral gray midpoint
DIVERGING_POS = "#e34948"  # red pole

DIVERGING = LinearSegmentedColormap.from_list(
    "t12_diverging", [DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS], N=256
)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, length=3, width=1.0)


# --------------------------------------------------------------------------- #
# X1 -- attribution heatmap at the anchor cell
# --------------------------------------------------------------------------- #
def x1_attribution_heatmap(
    phi: np.ndarray,
    date: str,
    profile: str,
    filename: str = "X1_attribution_heatmap_anchor.png",
) -> str:
    """8x8 phi at the anchor cell, with column sums annotated."""
    n = len(TICKERS)
    col_sums = phi.sum(axis=0)
    vmax = float(np.abs(phi).max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12.6, 10.4), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)

    im = ax.imshow(phi, cmap=DIVERGING, norm=norm, aspect="equal")
    # The column-sum strip lives on THIS axes, in the same data coordinates as
    # the cells. A separate axes cannot be used: aspect="equal" shrinks the
    # heatmap's box inside its allotted space, so a full-width strip below it
    # would not line up with the columns it is annotating.
    ax.set_ylim(n - 0.5 + 1.75, -0.5)
    ax.set_xticks(range(n), TICKERS)
    ax.set_yticks(range(n), TICKERS)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xlabel(
        "output weight component  w_j", color=INK_SECONDARY, fontsize=11, labelpad=10
    )
    ax.set_ylabel(
        "input feature  d_i  (forecast vol)", color=INK_SECONDARY, fontsize=11
    )
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 2px surface gap between cells, drawn as a grid on the
    # minor ticks so no fill touches its neighbour.
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.0)
    ax.tick_params(which="minor", length=0)

    for i in range(n):
        for j in range(n):
            v = float(phi[i, j])
            shade = abs(v) / vmax if vmax > 0 else 0.0
            ax.text(
                j,
                i,
                f"{v:+.4f}",
                ha="center",
                va="center",
                fontsize=8.0,
                color="#ffffff" if shade > 0.62 else INK_PRIMARY,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(
        "phi[i, j]  (attribution of d_i to w_j)", color=INK_SECONDARY, fontsize=10
    )
    cbar.ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, length=3)
    cbar.outline.set_visible(False)

    for j in range(n):
        ax.text(
            j,
            n - 0.5 + 0.62,
            f"{col_sums[j]:+.4f}",
            ha="center",
            va="center",
            fontsize=8.5,
            color=INK_PRIMARY,
        )
    ax.text(
        -0.5,
        n - 0.5 + 1.28,
        "column sums  =  w_advisor - w_classical   (efficiency identity, "
        "Sec. 1.1)",
        ha="left",
        va="center",
        fontsize=10,
        color=INK_SECONDARY,
    )

    ax.set_title(
        f"X1  Exact Shapley attributions, anchor cell {date} / {profile}\n"
        "explicand d_forecast(t), background d_classical(t), full 2^8 enumeration",
        color=INK_PRIMARY,
        fontsize=13,
        pad=34,
        loc="left",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# X2 -- stability exhibit
# --------------------------------------------------------------------------- #
def x2_stability(
    cells: list[tuple[str, str]],
    monitored: dict[tuple[str, str], list[tuple[str, float]]],
    control: dict[tuple[str, str], list[tuple[str, float]]],
    threshold: float,
    monitored_label: str,
    control_label: str,
    filename: str = "X2_stability.png",
) -> str:
    """rho at every cell x probe for both pairs, with the threshold line.

    ``monitored`` / ``control`` map a cell to its [(probe name, rho), ...].
    All cells are shown regardless of outcome.
    """
    # X2 is a two-panel exhibit.
    # TOP is the registered full-range view, unchanged: log axis, threshold line
    # at 10, all 48 monitored points plus the control distribution. BOTTOM is a
    # linear-scale companion over the observed band showing the SAME complete 96
    # points, with the threshold annotated as off-scale above.
    fig, (ax, ax_zoom) = plt.subplots(
        2, 1, figsize=(13.0, 11.6), height_ratios=[1.0, 1.0],
        sharex=True, constrained_layout=True,
    )
    fig.patch.set_facecolor(SURFACE)
    for target in (ax, ax_zoom):
        _style(target)
        target.grid(axis="y", color=GRID, linewidth=1.0)
        target.set_axisbelow(True)

    def spread(k: int, count: int, half: float) -> np.ndarray:
        if count == 1:
            return np.array([0.0])
        return np.linspace(-half, half, count)

    all_rho = [r for cell in cells for _, r in monitored[cell] + control[cell]]
    positive = [r for r in all_rho if r > 0.0]
    log_ok = len(positive) == len(all_rho)

    # The same 96 points are drawn on both panels.
    for target in (ax, ax_zoom):
        for k, cell in enumerate(cells):
            m = [r for _, r in monitored[cell]]
            c = [r for _, r in control[cell]]
            target.scatter(
                k - 0.17 + spread(k, len(m), 0.075),
                m,
                s=52,
                marker="o",
                color=SLOT_1_BLUE,
                edgecolors=SURFACE,
                linewidths=1.6,
                zorder=3,
                label=monitored_label if (k == 0 and target is ax) else None,
            )
            target.scatter(
                k + 0.17 + spread(k, len(c), 0.075),
                c,
                s=52,
                marker="D",
                color=SLOT_2_ORANGE,
                edgecolors=SURFACE,
                linewidths=1.6,
                zorder=3,
                label=control_label if (k == 0 and target is ax) else None,
            )

    # Reserved right margin: end-labels and the threshold caption live here, so
    # neither can sit on top of a data point.
    right = len(cells) - 0.4
    margin = 1.35

    ax.axhline(
        threshold, color=INK_SECONDARY, linewidth=2.0, linestyle=(0, (6, 3)), zorder=2
    )
    ax.text(
        right + 0.06,
        threshold,
        f"registered threshold = {threshold:g}",
        color=INK_SECONDARY,
        fontsize=10,
        va="bottom",
        ha="left",
    )

    if log_ok:
        ax.set_yscale("log")
        # DC-1 lesson: the log locator also emits minor ticks, which matplotlib
        # labels in scientific notation. Suppress the minor LABELS only.
        ax.yaxis.set_minor_formatter(NullFormatter())

    # Bottom panel: linear scale over the observed band only.
    lo_obs, hi_obs = min(all_rho), max(all_rho)
    pad = (hi_obs - lo_obs) * 0.12 or hi_obs * 0.05
    ax_zoom.set_ylim(lo_obs - pad, hi_obs + pad)
    ax_zoom.set_ylabel("rho  (linear, observed band)",
                       color=INK_SECONDARY, fontsize=11)
    ax_zoom.text(
        right + 0.06,
        hi_obs + pad,
        f"registered threshold = {threshold:g}\noff scale above",
        color=INK_SECONDARY,
        fontsize=10,
        va="top",
        ha="left",
    )
    ax_zoom.set_title(
        f"Companion panel (amendment T12-A2): same {len(all_rho)} points, "
        f"linear scale over the observed band {lo_obs:.3g} .. {hi_obs:.3g}",
        color=INK_SECONDARY, fontsize=11, pad=8, loc="left",
    )

    ax_zoom.set_xticks(range(len(cells)))
    ax_zoom.set_xticklabels(
        [f"{d}\n{p}" for d, p in cells], fontsize=9, color=INK_SECONDARY
    )
    ax.set_xlim(-0.6, right + margin)
    ax.set_ylabel("rho  (log, full registered range)",
                  color=INK_SECONDARY, fontsize=11)
    ax.set_title(
        "X2  Attribution stability under the registered perturbation probes\n"
        f"{len(cells)} cells x 4 probes per pair, eps = 0.01 on the explicand; "
        "all cells shown regardless of outcome",
        color=INK_PRIMARY,
        fontsize=13,
        pad=32,   # clears the legend row, which sits just above the axes
        loc="left",
    )

    # Direct end-labels in addition to the legend. Text wears ink tokens, never
    # the series colour: a colour+shape marker glyph beside each label carries
    # identity. Anchored to the last cell's group medians, pushed apart when
    # those medians are too close to render as two lines.
    last = cells[-1]
    y_mon = float(np.median([r for _, r in monitored[last]]))
    y_ctl = float(np.median([r for _, r in control[last]]))
    lo, hi = ax.get_ylim()
    if log_ok:
        span = np.log10(hi) - np.log10(lo)
        gap = abs(np.log10(y_mon) - np.log10(y_ctl)) / span
    else:
        span = hi - lo
        gap = abs(y_mon - y_ctl) / span
    if gap < 0.09:
        push = (0.09 - gap) / 2.0
        if log_ok:
            mid = (np.log10(y_mon) + np.log10(y_ctl)) / 2.0
            y_mon = 10 ** (mid + (push + gap / 2) * span * (1 if y_mon >= y_ctl else -1))
            y_ctl = 10 ** (mid - (push + gap / 2) * span * (1 if y_mon >= y_ctl else -1))
        else:
            mid = (y_mon + y_ctl) / 2.0
            y_mon = mid + (push + gap / 2) * span * (1 if y_mon >= y_ctl else -1)
            y_ctl = mid - (push + gap / 2) * span * (1 if y_mon >= y_ctl else -1)

    for y, label, colour, marker in (
        (y_mon, monitored_label, SLOT_1_BLUE, "o"),
        (y_ctl, control_label, SLOT_2_ORANGE, "D"),
    ):
        ax.scatter(
            [right + 0.14], [y], s=52, marker=marker, color=colour,
            edgecolors=SURFACE, linewidths=1.6, zorder=4, clip_on=False,
        )
        ax.text(
            right + 0.28, y, label, fontsize=9.5, color=INK_SECONDARY,
            va="center", ha="left",
        )

    # Legend above the plot area, so it can never occlude a data point.
    legend = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=2,
        frameon=False,
        fontsize=10,
        labelcolor=INK_SECONDARY,
    )
    legend.set_zorder(5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return str(path)
