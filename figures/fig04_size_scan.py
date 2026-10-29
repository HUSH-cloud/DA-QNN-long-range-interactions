"""Fig. 4: test accuracy versus system size N at h = 1.0 for (a) ell = 1 and (b) ell = 2, side by side.

Each curve is one alpha (0.2, 1.0, 3.0, 6.0, inf), N = 3..10, averaged over the six digit pairs
(3,8), (2,7), (4,9), (0,1), (5,8), (0,6) and 20 seeds, i.e. 120 runs per point.
Error bars: +-1 standard deviation (ddof = 0) of those 120 accuracies, pooled over pairs and seeds.
They are drawn on the alpha = 0.2 and alpha -> inf curves only.
draw_row() is the two-panel row of Fig. 4; draw() is the single-panel layout of Fig. 12 (fig12_size_scan_ell4.py).
row_axes() is shared with Fig. 6 (fig06_spt_size.py).

Input:  RESULTS/fig4_sizes/acc_sitA_l{ell}_N{N}_a{alpha}_p{pair}_s{seed}.npy
Output: OUT/fig_alphaN_acc.pdf
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

import style as S
from paths import OUT, RESULTS, SEEDS_20

SIZES = RESULTS / "fig4_sizes"
NS = list(range(3, 11))
PAIRS = ["38", "27", "49", "01", "58", "06"]
# Each alpha keeps the color it has in the ten-alpha palette of the other line figures.
COLORS = [S.ordered_colors(10)[S.ALPHAS_10.index(a)] for a in S.ALPHAS_5]
ROW_W = 7.6          # canvas width of the two-panel row (inserted at \textwidth in the paper)
ROW_PANEL_H = 2.75   # plot-box height of each panel of the row


def load(ell):
    """acc[N, alpha, pair, seed] at depth ell."""
    acc = np.zeros((len(NS), len(S.ALPHAS_5), len(PAIRS), len(SEEDS_20)))
    for ni, n in enumerate(NS):
        for ai, a in enumerate(S.ALPHAS_5):
            for pi, p in enumerate(PAIRS):
                for si, s in enumerate(SEEDS_20):
                    acc[ni, ai, pi, si] = np.load(SIZES / f"acc_sitA_l{ell}_N{n}_a{a}_p{p}_s{s}.npy").flat[0]
    return acc


def row_axes():
    """Two side-by-side plot boxes sharing the accuracy axis; returns fig, [ax_a, ax_b], (ml, mr, mb, fig_h) in inches."""
    ml, mr, mt, gap = 0.70, 0.12, 0.34, 0.22
    mb = 0.42 + 0.62                      # legend strip + x-label strip
    pw = (ROW_W - ml - mr - gap) / 2
    fig_h = mb + ROW_PANEL_H + mt
    fig = plt.figure(figsize=(ROW_W, fig_h))
    axes = [fig.add_axes(((ml + k * (pw + gap)) / ROW_W, mb / fig_h, pw / ROW_W, ROW_PANEL_H / fig_h))
            for k in range(2)]
    return fig, axes, (ml, mr, mb, fig_h)


def draw_row(ells, out_name):
    """One panel per ell side by side; the right panel shares the accuracy ticks of the left one, legend below."""
    S.use()
    fig, axes, (ml, mr, mb, fig_h) = row_axes()
    for k, (ax, ell) in enumerate(zip(axes, ells)):
        acc = load(ell)
        mean, std = acc.mean(axis=(2, 3)), acc.std(axis=(2, 3))
        for ai, a in enumerate(S.ALPHAS_5):
            S.errorbar_endpoints(ax, NS, mean[:, ai], std[:, ai], curve_idx=ai, n_curves=len(S.ALPHAS_5),
                                 color=COLORS[ai], marker="o", markersize=S.MS_DATA, capsize=S.CAPSIZE,
                                 linewidth=S.LW_DATA, elinewidth=1.0, label=S.alpha_label(a) if k == 0 else None)
        ax.set_xlim(min(NS) - 0.35, max(NS) + 0.35)
        ax.set_xticks(NS)
        ax.set_ylim(0.47, 1.00)
    for k, ax in enumerate(axes):
        S.journal_axes(ax, minor_x=False)
        S.panel_label(ax, "ab"[k], dx=-0.005, dy=1.02)
        ax.set_xlabel(r"Number of qubits  $N$", fontsize=S.FS_LABEL, labelpad=2)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axes[1].set_yticklabels([])
    fig.text(0.10 / ROW_W, (mb + ROW_PANEL_H / 2) / fig_h, "Test accuracy", rotation=90, va="center", ha="left",
             fontsize=S.FS_LABEL)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(S.ALPHAS_5), fontsize=S.FS_LEGEND, frameon=False,
               columnspacing=0.9, handlelength=1.4, handletextpad=0.35,
               bbox_to_anchor=(0.5 + (ml - mr) / (2 * ROW_W), 0.015 / fig_h))
    S.save(fig, OUT, out_name)
    plt.close(fig)


def draw(ells, out_name):
    """One stacked panel per ell; panel letters only when there is more than one panel."""
    S.use()
    n_panels = len(ells)
    fig_w, fig_h = S.COL_W, 2.4 + 1.75 * n_panels
    ml, mr, mt, gap = 0.68, 0.10, 0.26, 0.48
    mb = 0.42 + 0.62                      # legend strip + x-label strip
    ph = (fig_h - mb - mt - gap * (n_panels - 1)) / n_panels
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = [fig.add_axes((ml / fig_w, (mb + (n_panels - 1 - k) * (ph + gap)) / fig_h,
                          1 - (ml + mr) / fig_w, ph / fig_h)) for k in range(n_panels)]

    for k, (ax, ell) in enumerate(zip(axes, ells)):
        acc = load(ell)
        mean = acc.mean(axis=(2, 3))
        std = acc.std(axis=(2, 3))
        for ai, a in enumerate(S.ALPHAS_5):
            S.errorbar_endpoints(ax, NS, mean[:, ai], std[:, ai], curve_idx=ai, n_curves=len(S.ALPHAS_5),
                                 color=COLORS[ai], marker="o", markersize=S.MS_DATA, capsize=S.CAPSIZE,
                                 linewidth=S.LW_DATA, elinewidth=1.0,
                                 label=S.alpha_label(a) if k == 0 else None)
        S.journal_axes(ax, minor_x=False)
        if n_panels > 1:
            S.panel_label(ax, "abc"[k], dx=-0.005, dy=1.02)
        ax.set_xlim(min(NS) - 0.35, max(NS) + 0.35)
        ax.set_xticks(NS)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_ylim(0.47, 1.00)
        if k < n_panels - 1:
            ax.set_xticklabels([])

    axes[-1].set_xlabel(r"Number of qubits  $N$", fontsize=S.FS_LABEL, labelpad=2)
    fig.text(0.115 / fig_w, (mb + (n_panels * ph + (n_panels - 1) * gap) / 2) / fig_h, "Test accuracy",
             rotation=90, va="center", ha="left", fontsize=S.FS_LABEL)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=S.FS_LEGEND, frameon=False,
               columnspacing=0.5, handlelength=1.2, handletextpad=0.3,
               bbox_to_anchor=(0.5 + (ml - mr) / (2 * fig_w), 0.015 / fig_h))
    S.save(fig, OUT, out_name)
    plt.close(fig)


if __name__ == "__main__":
    draw_row([1, 2], "fig_alphaN_acc")
