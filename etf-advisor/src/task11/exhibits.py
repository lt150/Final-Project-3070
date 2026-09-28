"""Registered Task 11 exhibits.

Four body figures (F1–F4) and two appendix equity curves, all PNG, all
to ``outputs/task11/``. The exhibit list, panel content and body/appendix split
are pre-registered and are not open to post-run discretion; this module only
renders them.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

from ..data.config import UNIVERSE
from .h1 import EQUITY_SLEEVE, equity_sleeve
from .metrics import drawdown_curve

# --------------------------------------------------------------------------- #
# Palette (light mode, validated reference instance)
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

SERIES = {
    "rb_advisor": "#2a78d6",
    "rb_classical": "#eb6834",
    "static_bucket": "#1baf7a",
    "one_over_n": "#eda100",
}
STYLE = {
    "rb_advisor": "-",
    "rb_classical": "-",
    "static_bucket": (0, (6, 2)),
    "one_over_n": (0, (1.6, 2)),
}
LABEL = {
    "rb_advisor": "rb_advisor",
    "rb_classical": "rb_classical",
    "static_bucket": "static_bucket",
    "one_over_n": "1/N",
}
# Curve order = slot order; benchmarks last so the two allocators under test
# take the leading (highest-contrast) slots.
CURVE_ORDER = ("rb_advisor", "rb_classical", "static_bucket", "one_over_n")

STACK_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]

LINE_W = 2.0
MARKER_S = 8.0

HINDSIGHT = (
    "the forecaster family was selected on 2021–24 evidence, so pre-2021 "
    "exhibits apply a method chosen with hindsight — universal to retrospective "
    "backtests, low severity for a 16-parameter regression, dissolved at "
    "coefficient level by walk-forward refit."
)


def _style_axes(ax, *, ylabel: str = "", xlabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=0.8)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)


def _title(fig, title: str, subtitle: str = "") -> None:
    fig.text(0.012, 0.975, title, ha="left", va="top",
             fontsize=13, color=INK_PRIMARY, fontweight="bold")
    if subtitle:
        fig.text(0.012, 0.925, subtitle, ha="left", va="top",
                 fontsize=9.5, color=INK_SECONDARY)


def _footnote(fig, text: str, y: float = 0.012) -> None:
    fig.text(0.012, y, text, ha="left", va="bottom", fontsize=7.6,
             color=INK_MUTED, wrap=True)


def _end_labels(ax, entries: list[tuple[float, str]], x_pos, min_gap_pt: float = 11.0):
    """Direct labels at the right edge, de-collided in display space.

    Works on log and linear axes alike because the de-collision happens after
    the scale transform. Call only once the final layout is fixed — the
    transforms it reads are invalidated by any later ``subplots_adjust``.

    Label text carries a text token, never the series colour; the arriving
    curve is the colour mark beside it. Where de-collision displaces a label
    off its curve, a hairline leader in muted ink restores the association.
    """
    gap = min_gap_pt * ax.figure.dpi / 72.0
    items = sorted(entries, key=lambda e: e[0])
    disp = [ax.transData.transform((0.0, y))[1] for y, _ in items]
    for i in range(1, len(disp)):
        if disp[i] - disp[i - 1] < gap:
            disp[i] = disp[i - 1] + gap

    inv = ax.transData.inverted()
    for (y, text), d in zip(items, disp):
        y_lab = float(inv.transform((0.0, d))[1])
        moved = abs(d - ax.transData.transform((0.0, y))[1]) > 1.0
        ax.annotate(
            text,
            xy=(x_pos, y_lab), xytext=(7, 0), textcoords="offset points",
            va="center", ha="left", fontsize=9, color=INK_SECONDARY,
            annotation_clip=False, zorder=5,
        )
        if moved:
            ax.plot([x_pos, x_pos], [y, y_lab], color=INK_MUTED, linewidth=0.6,
                    zorder=4, clip_on=False)


def _pct(y, _pos):
    return f"{y * 100:.0f}%"


# --------------------------------------------------------------------------- #
# F1 / appendix — equity curves
# --------------------------------------------------------------------------- #
def equity_curve_figure(
    values: dict[str, pd.Series],
    profile: str,
    path,
    *,
    with_hindsight: bool,
    figure_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.9), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax, ylabel="portfolio value (log scale, funded at 1.000)")

    ends = []
    for key in CURVE_ORDER:
        s = values[key]
        ax.plot(s.index, s.to_numpy(), color=SERIES[key], linewidth=LINE_W,
                linestyle=STYLE[key], label=LABEL[key], zorder=3,
                solid_capstyle="round")
        ends.append((float(s.iloc[-1]), f"{LABEL[key]}  {float(s.iloc[-1]):.2f}"))

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _p: f"{y:.2f}"))
    lo = min(float(s.min()) for s in values.values())
    hi = max(float(s.max()) for s in values.values())
    ticks = [t for t in (0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0)
             if lo * 0.98 <= t <= hi * 1.02]
    if len(ticks) >= 3:
        ax.set_yticks(ticks)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    idx = next(iter(values.values())).index
    ax.set_xlim(idx[0], idx[-1])

    leg = ax.legend(loc="upper left", frameon=False, fontsize=9, handlelength=2.6)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    _title(fig, f"{figure_label}  Net-of-cost equity curves — {profile} profile",
           "monthly rebalancing, 10 bps one-way transaction costs, "
           "2012-01-03 → 2024-12-31; log vertical scale")
    if with_hindsight:
        _footnote(fig, "Hindsight disclosure: " + HINDSIGHT)
    fig.subplots_adjust(left=0.075, right=0.845, top=0.855,
                        bottom=0.145 if with_hindsight else 0.09)
    fig.canvas.draw()
    _end_labels(ax, ends, idx[-1])
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# F2 — drawdown curves
# --------------------------------------------------------------------------- #
def drawdown_figure(values: dict[str, pd.Series], profile: str, path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax, ylabel="drawdown from running peak")

    ends = []
    troughs = {}
    for key in CURVE_ORDER:
        dd = drawdown_curve(values[key])
        ax.plot(dd.index, dd.to_numpy(), color=SERIES[key], linewidth=LINE_W,
                linestyle=STYLE[key], label=LABEL[key], zorder=3,
                solid_capstyle="round")
        ends.append((float(dd.iloc[-1]), LABEL[key]))
        troughs[key] = (dd.idxmin(), float(dd.min()))

    ax.axhline(0.0, color=BASELINE, linewidth=1.0, zorder=2)
    ax.yaxis.set_major_formatter(FuncFormatter(_pct))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    idx = next(iter(values.values())).index
    ax.set_xlim(idx[0], idx[-1])

    leg = ax.legend(loc="lower left", frameon=False, fontsize=9, handlelength=2.6)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    worst = min(troughs.items(), key=lambda kv: kv[1][1])
    ax.annotate(
        f"deepest: {LABEL[worst[0]]} {worst[1][1] * 100:.1f}% "
        f"({worst[1][0].date()})",
        xy=(worst[1][0], worst[1][1]), xytext=(10, 14),
        textcoords="offset points", fontsize=8.5, color=INK_SECONDARY,
        arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=0.8),
    )

    _title(fig, f"F2  Drawdown curves — {profile} profile",
           "net of costs, on the daily value path, 2012-01-03 → 2024-12-31")
    fig.subplots_adjust(left=0.075, right=0.845, top=0.845, bottom=0.09)
    fig.canvas.draw()
    _end_labels(ax, ends, idx[-1])
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# F3 — stacked-area target weights, two panels
# --------------------------------------------------------------------------- #
def weights_stack_figure(
    target_weights: dict[str, pd.DataFrame], profile: str, path
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.6), dpi=200, sharex=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, key in zip(axes, ("rb_advisor", "rb_classical")):
        w = target_weights[key][list(UNIVERSE)]
        _style_axes(ax, ylabel="target weight")
        ax.stackplot(
            w.index, *[w[tk].to_numpy() for tk in UNIVERSE],
            labels=list(UNIVERSE), colors=STACK_COLORS,
            edgecolor=SURFACE, linewidth=1.0, zorder=3,
        )
        ax.set_ylim(0.0, 1.0)
        ax.yaxis.set_major_formatter(FuncFormatter(_pct))
        ax.set_xlim(w.index[0], w.index[-1])
        ax.grid(False)
        ax.text(0.008, 0.945, LABEL[key], transform=ax.transAxes,
                fontsize=10, color=INK_PRIMARY, fontweight="bold",
                va="top", ha="left",
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2.5))

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=8, frameon=False,
                     fontsize=9, bbox_to_anchor=(0.5, 0.005), handlelength=1.6)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    _title(fig, f"F3  Target-weight trajectories at grid dates — {profile} profile",
           "stacked to 1.000; stack order equity (SPY QQQ EFA EEM) · bonds "
           "(AGG IEF) · alternatives (GLD VNQ), so the equity sleeve is the "
           "lower band")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.885, bottom=0.105, hspace=0.13)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# F4 — equity sleeve through the crash
# --------------------------------------------------------------------------- #
def crash_sleeve_figure(
    target_weights: dict[str, pd.DataFrame],
    window: pd.DatetimeIndex,
    q3_dates: dict[str, pd.Timestamp],
    profile: str,
    path,
) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax, ylabel="equity-sleeve target weight E")

    ax.axvspan(q3_dates["Feb"], q3_dates["Mar"], color=GRID, alpha=0.75, zorder=1)
    ax.text(q3_dates["Feb"], 0.995, " registered H1-Q3 leg", transform=ax.get_xaxis_transform(),
            fontsize=8.5, color=INK_MUTED, va="top", ha="left")

    ends = []
    for key in ("rb_advisor", "rb_classical"):
        e = equity_sleeve(target_weights[key]).loc[window]
        ax.plot(e.index, e.to_numpy(), color=SERIES[key], linewidth=LINE_W,
                marker="o", markersize=MARKER_S, markeredgecolor=SURFACE,
                markeredgewidth=1.6, label=LABEL[key], zorder=3)
        d = abs(float(e.loc[q3_dates["Mar"]]) - float(e.loc[q3_dates["Feb"]]))
        ends.append((float(e.iloc[-1]), f"{LABEL[key]}   |ΔE| Feb→Mar = {d:.3f}"))

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_xlim(window[0] - pd.Timedelta(days=8), window[-1] + pd.Timedelta(days=8))

    # Legend below the axes: the right margin is reserved for the direct
    # end-labels, and an in-axes legend collides with them whenever a curve
    # finishes high.
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower left", ncol=2, frameon=False,
                     fontsize=9, handlelength=2.6, bbox_to_anchor=(0.085, 0.005))
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    _title(fig, f"F4  Equity-sleeve target weight through the 2020 crash — {profile} profile",
           f"E = w_SPY + w_QQQ + w_EFA + w_EEM at grid dates "
           f"{window[0].date()} → {window[-1].date()}; shaded leg is the "
           "registered H1-Q3 comparison")
    fig.subplots_adjust(left=0.085, right=0.665, top=0.845, bottom=0.185)
    fig.canvas.draw()
    _end_labels(ax, ends, window[-1])
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


__all__ = [
    "EQUITY_SLEEVE", "HINDSIGHT", "SERIES", "crash_sleeve_figure",
    "drawdown_figure", "equity_curve_figure", "weights_stack_figure",
]
