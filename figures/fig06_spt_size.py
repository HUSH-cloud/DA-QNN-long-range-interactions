"""Fig. 6: cluster-SPT test accuracy versus alpha for N = 6, 8, 10 at (a) ell = 1 and (b) ell = 2, side by side.

Balanced training set (25 SPT, 25 non-SPT), h = 1.0, tau = 0.4 pi, 500 iterations. Each point is the mean
over 10 seeds (SEEDS_10) of the test accuracy on the 441 test points, SPT probability 1 - p0 (test_prob_spt)
thresholded at 0.5.
Error bars: 95% confidence interval of the mean over seeds (Student t, ddof = 1), drawn on every curve.
The canvas is the two-panel row of Fig. 4 (fig04_size_scan.row_axes).

Input:  RESULTS/spt/N{N}_l{ell}_a{alpha}_s{seed}.npz   (key acc_test), ell = 1, 2
Output: OUT/fig_spt_size.pdf
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

import style as S
from fig04_size_scan import ROW_PANEL_H, ROW_W, row_axes
from paths import OUT, RESULTS, SEEDS_10

NS = [6, 8, 10]
YLIM = (0.70, 0.95)


def load(ell):
    """acc[N, alpha, seed] of the runs at depth ell."""
    acc = np.zeros((len(NS), len(S.ALPHAS_5), len(SEEDS_10)))
    for ni, n in enumerate(NS):
        for ai, a in enumerate(S.ALPHAS_5):
            for si, s in enumerate(SEEDS_10):
                f = RESULTS / "spt" / f"N{n}_l{ell}_a{a}_s{s}.npz"
                acc[ni, ai, si] = float(np.load(f)["acc_test"][0])
    return acc


def draw(ells, out_name):
    """One panel per ell side by side, one curve per N; the right panel shares the accuracy ticks, legend below."""
    S.use()
    fig, axes, (ml, mr, mb, fig_h) = row_axes()
    x = np.arange(len(S.ALPHAS_5))
    colors = S.ordered_colors(len(NS))

    for k, (ax, ell) in enumerate(zip(axes, ells)):
        acc = load(ell)
        mean = acc.mean(axis=2)
        err = S.ci_from_samples(acc, axis=2)
        lo, hi = (mean - err).min(), (mean + err).max()
        assert YLIM[0] < lo and hi < YLIM[1], f"ell={ell}: data {lo:.3f}..{hi:.3f} clipped by {YLIM}"
        for ni, n in enumerate(NS):
            S.errorbar_endpoints(ax, list(x), mean[ni], err[ni], curve_idx=ni, n_curves=len(NS),
                                 color=colors[ni], marker="o", markersize=S.MS_DATA, capsize=S.CAPSIZE,
                                 linewidth=S.LW_DATA, elinewidth=1.0,
                                 label=rf"$N={n}$" if k == 0 else None)
        S.journal_axes(ax, minor_x=False)
        S.panel_label(ax, "ab"[k], dx=-0.005, dy=1.02)
        ax.set_xlim(-0.32, len(x) - 0.68)
        ax.set_xticks(x)
        ax.set_xticklabels([r"$\infty$" if a == "Inf" else a for a in S.ALPHAS_5])
        ax.set_xlabel(r"Interaction exponent  $\alpha$", fontsize=S.FS_LABEL, labelpad=2)
        ax.set_ylim(*YLIM)
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_minor_locator(MultipleLocator(0.025))
    axes[1].set_yticklabels([])
    fig.text(0.10 / ROW_W, (mb + ROW_PANEL_H / 2) / fig_h, "Test accuracy", rotation=90, va="center", ha="left",
             fontsize=S.FS_LABEL)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(NS), fontsize=S.FS_LEGEND, frameon=False,
               columnspacing=1.2, handlelength=1.4, handletextpad=0.35,
               bbox_to_anchor=(0.5 + (ml - mr) / (2 * ROW_W), 0.015 / fig_h))
    S.save(fig, OUT, out_name)
    plt.close(fig)


if __name__ == "__main__":
    draw([1, 2], "fig_spt_size")
