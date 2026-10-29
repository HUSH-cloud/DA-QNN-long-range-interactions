"""Shared figure style: canvas size, type sizes, colors, axes treatment, confidence intervals.

Every figure is included at \\linewidth, so a fixed canvas width (COL_W for one column, FULL_W for two)
gives the same printed type size in every figure. Figures are saved without re-cropping for that reason.
"""
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator
from scipy import stats

COL_W = 5.00
FULL_W = 10.35
FIG_H = 4.25

FS_TICK = 13.0
FS_LABEL = 15.0
FS_PANEL = 16.0
FS_LEGEND = 13.0
FS_ANNOT = 11.0
FS_CBAR = 13.0
FS_CELL = 8.4

LW_AXES = 1.1
LW_DATA = 1.8
MS_DATA = 5.5
CAPSIZE = 2.6

ACC_VMIN, ACC_VMAX = 0.5, 1.0

# Long range (small alpha) is red, nearest neighbor is blue.
LR_NN_CMAP = LinearSegmentedColormap.from_list(
    "lr_to_nn",
    [(0.00, "#4a7fb8"), (0.25, "#7fb5d8"), (0.50, "#ffc94e"), (0.75, "#f2724d"), (1.00, "#d7312d")],
    N=256,
)

CMAP_ACC = LinearSegmentedColormap.from_list(
    "nature_div",
    [(0.00, "#05306b"), (0.15, "#134b87"), (0.30, "#327db7"), (0.45, "#6fafd2"), (0.50, "#c7e0ed"),
     (0.55, "#fbd2bc"), (0.65, "#feab88"), (0.80, "#b71c2c"), (0.90, "#8b0824"), (1.00, "#6a0624")],
    N=256,
)

ALPHAS_5 = ["0.2", "1.0", "3.0", "6.0", "Inf"]
ALPHAS_10 = ["0.2", "0.4", "0.6", "0.8", "1.0", "1.5", "2.0", "3.0", "6.0", "Inf"]


def ordered_colors(n):
    """n colors for an ordered sweep, first = red end, last = blue end."""
    if n == 1:
        return [LR_NN_CMAP(0.5)]
    return [LR_NN_CMAP(1.0 - i / (n - 1)) for i in range(n)]


def alpha_label(alpha):
    """'0.2' -> '$\\alpha=0.2$', 'Inf' -> '$\\alpha\\to\\infty$'."""
    return r"$\alpha\to\infty$" if alpha == "Inf" else rf"$\alpha={alpha}$"


def use():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": FS_TICK,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "axes.linewidth": LW_AXES,
        "lines.linewidth": LW_DATA,
        "xtick.direction": "in", "ytick.direction": "in",
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    })


def journal_axes(ax, minor_x=True, minor_y=True, direction="in", sides=("top", "right", "bottom", "left")):
    """Closed frame, ticks on the given sides, minor ticks halfway between majors."""
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(LW_AXES)
    ax.tick_params(which="both", direction=direction, width=LW_AXES,
                   top="top" in sides, right="right" in sides, bottom="bottom" in sides, left="left" in sides)
    ax.tick_params(which="major", length=3.8)
    ax.tick_params(which="minor", length=2.1)
    if minor_x:
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    else:
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    if minor_y:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    else:
        ax.tick_params(axis="y", which="minor", left=False, right=False)
    ax.grid(False)


def panel_label(ax, letter, dx=-0.02, dy=1.015):
    """Bold (a), (b) above the frame."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=FS_PANEL, fontweight="bold", va="bottom", ha="left")


def cell_value(ax, i, j, value, ncols):
    """Write a heat-map value centered in cell (i, j), shifted inward if it would cross the frame."""
    t = ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=FS_CELL, color="white", zorder=5)
    fig = ax.figure
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    x0, x1 = ax.transData.inverted().transform([(bb.x0, 0), (bb.x1, 0)])[:, 0]
    if j == 0 and x0 < -0.5:
        t.set_ha("left"); t.set_x(-0.46)
    elif j == ncols - 1 and x1 > ncols - 0.5:
        t.set_ha("right"); t.set_x(ncols - 1 + 0.46)
    return t


def errorbar_endpoints(ax, x, y, yerr, *, curve_idx, n_curves, color, marker="o", label=None,
                       markersize=6, linewidth=1.4, capsize=3, capthick=1.0, elinewidth=1.2):
    """Draw every curve, but error bars only on the first and last curve (all curves when there are three or fewer)."""
    show = set(range(n_curves)) if n_curves <= 3 else {0, n_curves - 1}
    line, = ax.plot(x, y, "-" + marker, color=color, linewidth=linewidth, markersize=markersize, label=label, zorder=2)
    bars = None
    if curve_idx in show:
        bars = ax.errorbar(x, y, yerr=yerr, fmt="none", color=color, capsize=capsize, capthick=capthick,
                           elinewidth=elinewidth, alpha=0.9, zorder=3)
    return line, bars


def t_factor(n):
    """Two-sided 95% Student-t multiplier for sample size n (array-aware)."""
    n = np.asarray(n, dtype=float)
    out = np.full(n.shape, np.nan)
    ok = n >= 2
    out[ok] = stats.t.ppf(0.975, n[ok] - 1)
    return out


def ci_from_samples(samples, axis=0):
    """Half-width of the 95% confidence interval of the mean along axis, ignoring NaNs."""
    samples = np.asarray(samples, dtype=float)
    n = np.sum(np.isfinite(samples), axis=axis)
    s = np.nanstd(samples, axis=axis, ddof=1)
    return t_factor(n) * s / np.sqrt(np.where(n >= 2, n, np.nan))


def save(fig, out_dir, stem):
    """Write out_dir/stem.pdf at the authored canvas size."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches=None)
    return out_dir / f"{stem}.pdf"
