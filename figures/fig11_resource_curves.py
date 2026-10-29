"""Fig. 11: test accuracy versus the analog-block resources at h = 1.0.

(a) accuracy versus tau/2pi at ell = 2, tau = 0.1 pi ... 1.0 pi.
(b) accuracy versus the circuit depth ell = 1 ... 8 at tau = 0.4 pi.
One curve per alpha in {0.2, 1.0, 3.0, 6.0, Inf}, both panels on the same accuracy scale.

Point: mean test accuracy over 20 seeds. Error bar: 95% Student-t confidence interval of that mean,
drawn only on the extremal curves alpha = 0.2 and alpha -> Inf.

Inputs (relative to paths.RESULTS):
    fig3_tau/h1/acc_a{alpha}_t{01..10}_s{seed}.npy
    fig3_layers/h1/acc_a{alpha}_tl{ell}_s{seed}.npy
Output: paths.OUT/fig_resource_combined.pdf
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

import style as S
from paths import OUT, RESULTS, SEEDS_20

ALPHAS = S.ALPHAS_5
TAU_PI = np.array([0.1 * i for i in range(1, 11)])    # tau / pi, file index t01 ... t10
ELLS = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
ELL_FIXED = 2
TAU_FIXED = 0.4
TAU_DIR = RESULTS / "fig3_tau" / "h1"
ELL_DIR = RESULTS / "fig3_layers" / "h1"

# Each alpha keeps the color it has in the ten-alpha palette of the other line figures.
COLORS = [S.ordered_colors(10)[S.ALPHAS_10.index(a)] for a in ALPHAS]


def load_acc(folder, grid_tags):
    """acc[alpha, grid point, seed] from acc_a{alpha}_{tag}_s{seed}.npy, shape (5, len(grid_tags), 20)."""
    acc = np.zeros((len(ALPHAS), len(grid_tags), len(SEEDS_20)))
    for ai, alpha in enumerate(ALPHAS):
        for gi, tag in enumerate(grid_tags):
            for si, seed in enumerate(SEEDS_20):
                acc[ai, gi, si] = float(np.load(folder / f"acc_a{alpha}_{tag}_s{seed}.npy").flat[0])
    return acc


def draw(ax, x, acc, label):
    """One curve per alpha; mean over seeds, CI bars on the first and last curve only."""
    mean = acc.mean(axis=2)
    err = S.ci_from_samples(acc, axis=2)
    for ai, alpha in enumerate(ALPHAS):
        S.errorbar_endpoints(ax, list(x), mean[ai], err[ai], curve_idx=ai, n_curves=len(ALPHAS),
                             color=COLORS[ai], marker="o", markersize=S.MS_DATA, capsize=S.CAPSIZE,
                             linewidth=S.LW_DATA, elinewidth=1.0,
                             label=S.alpha_label(alpha) if label else None)


def main():
    S.use()
    acc_tau = load_acc(TAU_DIR, [f"t{i:02d}" for i in range(1, 11)])
    acc_ell = load_acc(ELL_DIR, [f"tl{ell}" for ell in range(1, 9)])

    # Two stacked panels of height 1.75 in; the gap holds panel (a)'s x axis and panel (b)'s letter.
    fig_w = S.COL_W
    leg_h, xlab_h = 0.34, 0.62
    ml, mr, mt, gap = 0.68, 0.12, 0.26, 0.88
    mb = leg_h + xlab_h
    ph = 1.75
    fig_h = mb + 2 * ph + gap + mt
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = [fig.add_axes((ml / fig_w, (mb + (1 - k) * (ph + gap)) / fig_h, 1 - (ml + mr) / fig_w, ph / fig_h))
            for k in range(2)]

    tau_2pi = TAU_PI / 2
    draw(axes[0], tau_2pi, acc_tau, label=True)
    axes[0].set_xlim(tau_2pi[0] - 0.0125, tau_2pi[-1] + 0.0125)
    axes[0].xaxis.set_major_locator(MultipleLocator(0.1))
    axes[0].set_xlabel(r"$\tau/2\pi$", fontsize=S.FS_LABEL, labelpad=2)

    draw(axes[1], ELLS, acc_ell, label=False)
    axes[1].set_xlim(0.55, 8.45)
    axes[1].xaxis.set_major_locator(MultipleLocator(1))
    axes[1].set_xlabel(r"Circuit depth  $\ell$", fontsize=S.FS_LABEL, labelpad=2)

    # 0.5 is the two-class random baseline; the frame opens slightly below it.
    for k, ax in enumerate(axes):
        S.journal_axes(ax, minor_x=(k == 0))
        ax.set_ylim(0.47, 1.00)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        S.panel_label(ax, "ab"[k], dx=-0.005, dy=1.02)
    for ax, fixed in ((axes[0], rf"$\ell={ELL_FIXED}$"), (axes[1], rf"$\tau={TAU_FIXED}\pi$")):
        ax.text(0.975, 0.955, fixed, transform=ax.transAxes, ha="right", va="top", fontsize=S.FS_LABEL)
    fig.text(0.115 / fig_w, (mb + ph + gap / 2) / fig_h, "Test accuracy",
             rotation=90, va="center", ha="left", fontsize=S.FS_LABEL)

    handles = axes[0].get_legend_handles_labels()[0]
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center", ncol=5, fontsize=S.FS_LEGEND,
               frameon=False, columnspacing=0.35, handlelength=1.1, handletextpad=0.25, labelspacing=0.3,
               bbox_to_anchor=(0.5 + (ml - mr) / (2 * fig_w), 0.015 / fig_h))

    S.save(fig, OUT, "fig_resource_combined")
    plt.close(fig)


if __name__ == "__main__":
    main()
