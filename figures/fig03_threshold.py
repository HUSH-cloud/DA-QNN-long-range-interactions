"""Fig. 3: minimum resources needed to reach the test accuracy acc* = 0.70, versus alpha.

Left axis (blue): tau_min/2pi, from the tau scan at ell = 2 on the grid tau = 0.1 pi ... 1.0 pi.
Right axis (orange): ell_min, from the depth scan ell = 1 ... 8 at tau = 0.4 pi.
h = 1.0 is drawn with solid lines and circles, h = 3.0 with dashed lines and squares, shifted
left and right of each alpha so the two fields do not overlap.

Point: the resource at which the seed-averaged accuracy curve first reaches acc*, found by linear
interpolation between the two grid points around the crossing.
Error bar: 95% bootstrap confidence interval of that same point. The 20 seeds are resampled with
replacement N_BOOT times, the crossing of each resampled seed-average is recomputed, and the bar spans
the 2.5% to 97.5% quantiles. The bar is therefore asymmetric and never leaves the scanned grid.

Inputs (20 seeds, relative to paths.RESULTS):
    fig3_tau/h{1,3}/acc_a{alpha}_t{01..10}_s{seed}.npy        tau scan
    fig3_layers/h{1,3}/acc_a{alpha}_tl{ell}_s{seed}.npy       depth scan
Output: paths.OUT/fig_threshold_line.pdf
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

import style as S
from paths import OUT, RESULTS, SEEDS_20

ACC_STAR = 0.70
ALPHAS = S.ALPHAS_5
XTICK = [r"$0.2$", r"$1.0$", r"$3.0$", r"$6.0$", r"$\infty$"]
TAU_PI = np.array([0.1 * i for i in range(1, 11)])    # tau / pi, file index t01 ... t10
ELLS = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
X_OFFSET = 0.08                                        # h = 1 left of each alpha, h = 3 right
N_BOOT = 10000                                         # bootstrap resamples per point
BOOT_SEED = 0                                          # fixed so that the figure is reproducible

C_TAU, C_ELL = "#2E7EBC", "#E25508"
DASH = (0, (3.2, 1.6))
FIELDS = {
    "1": dict(ls="-", marker="o"),
    "3": dict(ls=DASH, marker="s"),
}
TAU_DIR = {h: RESULTS / "fig3_tau" / f"h{h}" for h in FIELDS}
ELL_DIR = {h: RESULTS / "fig3_layers" / f"h{h}" for h in FIELDS}


def load_acc(folder, grid_tags):
    """acc[alpha, grid point, seed] from acc_a{alpha}_{tag}_s{seed}.npy, shape (5, len(grid_tags), 20)."""
    acc = np.zeros((len(ALPHAS), len(grid_tags), len(SEEDS_20)))
    for ai, alpha in enumerate(ALPHAS):
        for gi, tag in enumerate(grid_tags):
            for si, seed in enumerate(SEEDS_20):
                acc[ai, gi, si] = float(np.load(folder / f"acc_a{alpha}_{tag}_s{seed}.npy").flat[0])
    return acc


def first_crossing(x, acc):
    """Smallest x at which acc reaches ACC_STAR, linearly interpolated; NaN if it never does."""
    for i in range(len(acc)):
        if acc[i] >= ACC_STAR:
            if i == 0:
                return x[0]
            y0, y1 = acc[i - 1], acc[i]
            x0, x1 = x[i - 1], x[i]
            return x1 if y1 == y0 else x0 + (ACC_STAR - y0) * (x1 - x0) / (y1 - y0)
    return np.nan


def resource_min(x, acc):
    """Crossing of the seed-averaged curve and its 95% bootstrap interval.

    x   : grid of the scanned resource, shape (n_grid,)
    acc : acc[alpha, grid point, seed], shape (5, n_grid, 20)
    Returns point, shape (5,), and err, shape (2, 5): err[0] = point - lower end, err[1] = upper end - point,
    the format matplotlib's errorbar expects for asymmetric bars.
    """
    rng = np.random.default_rng(BOOT_SEED)
    n_seed = acc.shape[2]
    point = np.array([first_crossing(x, acc[ai].mean(axis=1)) for ai in range(len(ALPHAS))])
    err = np.zeros((2, len(ALPHAS)))
    for ai in range(len(ALPHAS)):
        boot = np.empty(N_BOOT)
        for b in range(N_BOOT):
            pick = rng.integers(0, n_seed, n_seed)              # n_seed seed indices drawn with replacement
            boot[b] = first_crossing(x, acc[ai][:, pick].mean(axis=1))
        low, high = np.nanpercentile(boot, [2.5, 97.5])
        err[:, ai] = [point[ai] - low, high - point[ai]]
    return point, err


def round_up(v, step):
    return step * np.ceil(v / step + 1e-9)


def main():
    S.use()

    stats = {}
    for h in FIELDS:
        acc_tau = load_acc(TAU_DIR[h], [f"t{i:02d}" for i in range(1, 11)])
        acc_ell = load_acc(ELL_DIR[h], [f"tl{ell}" for ell in range(1, 9)])
        tau_min, tau_err = resource_min(TAU_PI, acc_tau)
        ell_min, ell_err = resource_min(ELLS, acc_ell)
        stats[h] = (tau_min / 2.0, ell_min, tau_err / 2.0, ell_err)     # tau in units of 2 pi

    x = np.arange(len(ALPHAS))
    # Canvas lower than S.FIG_H: the two y axes take 1.6 in of width, so a full-height canvas would leave a
    # nearly square plot box. 3.55 in gives the plot box the same aspect ratio (h/w ~ 0.73) as Fig. 5.
    fig_w, fig_h = S.COL_W, 3.55
    ml, mr, mb, mt = 0.80, 0.80, 0.62, 0.40
    fig = plt.figure(figsize=(fig_w, fig_h))
    axL = fig.add_axes((ml / fig_w, mb / fig_h, 1 - (ml + mr) / fig_w, 1 - (mb + mt) / fig_h))
    axR = axL.twinx()

    for h, line_style in FIELDS.items():
        tau_min, ell_min, tau_err, ell_err = stats[h]
        xs = x + (X_OFFSET if h == "3" else -X_OFFSET)
        for ax, y, yerr, color in ((axL, tau_min, tau_err, C_TAU), (axR, ell_min, ell_err, C_ELL)):
            ax.errorbar(xs, y, yerr=yerr, color=color, capsize=S.CAPSIZE, markersize=S.MS_DATA,
                        linewidth=S.LW_DATA, markeredgecolor="white", markeredgewidth=0.6, elinewidth=1.0,
                        **line_style)

    # Both axes start at 0. The tau axis is opened in steps of 0.05 until every tau_min point sits
    # clearly below every ell_min point at the same alpha, so the two families never cross.
    tau_top = round_up(max(np.nanmax(stats[h][0] + stats[h][2][1]) for h in FIELDS), 0.05)
    ell_top = round_up(max(np.nanmax(stats[h][1] + stats[h][3][1]) for h in FIELDS), 0.5)
    tau_highest = np.nanmax([stats[h][0] for h in FIELDS], axis=0)
    ell_lowest = np.nanmin([stats[h][1] for h in FIELDS], axis=0)
    while np.any(tau_highest / tau_top > ell_lowest / ell_top - 0.06) and tau_top < 1.0:
        tau_top = round(tau_top + 0.05, 2)
    axL.set_ylim(0, tau_top)
    axR.set_ylim(0, ell_top)

    axL.yaxis.set_major_locator(MultipleLocator(0.05 if tau_top <= 0.36 else 0.1))
    axL.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axR.yaxis.set_major_locator(MultipleLocator(0.5))
    axR.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axL.set_xlim(-0.32, len(x) - 0.68)
    axL.set_xticks(x)
    axL.set_xticklabels(XTICK)
    axL.set_xlabel(r"Interaction exponent  $\alpha$", fontsize=S.FS_LABEL)
    S.journal_axes(axL, minor_x=False)
    S.journal_axes(axR, minor_x=False)
    axL.tick_params(axis="y", which="both", colors=C_TAU, right=False)
    axR.tick_params(axis="y", which="both", colors=C_ELL, left=False)
    axR.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    axL.tick_params(axis="x", top=True)
    axL.set_ylabel(r"$\tau_{\min}/2\pi$", fontsize=S.FS_LABEL, color=C_TAU)
    axR.set_ylabel(r"$\ell_{\min}$", fontsize=S.FS_LABEL, color=C_ELL)
    for ax, color in ((axL, C_TAU), (axR, C_ELL)):
        for label in ax.get_yticklabels():
            label.set_color(color)

    # One legend entry per field, each showing the blue and the orange marker side by side.
    marker_kw = dict(markeredgecolor="white", markeredgewidth=0.6, markersize=S.MS_DATA, linewidth=S.LW_DATA)
    handles = [(Line2D([], [], color=C_TAU, markerfacecolor=C_TAU, **s, **marker_kw),
                Line2D([], [], color=C_ELL, markerfacecolor=C_ELL, **s, **marker_kw)) for s in FIELDS.values()]
    axL.legend(handles, [r"$h=1.0$", r"$h=3.0$"], handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)},
               loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=S.FS_LEGEND,
               handlelength=3.6, columnspacing=2.0, handletextpad=0.6, borderaxespad=0.0)

    S.save(fig, OUT, "fig_threshold_line")
    plt.close(fig)


if __name__ == "__main__":
    main()
