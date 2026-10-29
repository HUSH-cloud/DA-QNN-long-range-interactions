"""Fig. 16: probe-pair contrasts of the XXZ loss maps for the four training points.

N = 8, ell = 2, h = 1.0, tau = 0.4 pi; panels (a)-(d) = training points zAFM, qzAFM, XY-QLRO, VBS;
bars = alpha in {0.2, 1.0, 3.0, 6.0}. Bar height: seed mean of |L~(xi_i) - L~(xi_j)|, L~ = min-max normalized loss map;
error bars: 95% t interval over the ten seeds. Probes xi_k = grid points nearest to the training points.

Input:  RESULTS/xxz and DATA/xxz_data/xxz_N8_grid50.npz (see xxz_common.py)
Output: OUT/fig_xxz_pairdiff.pdf
"""
import matplotlib.pyplot as plt
import numpy as np

import style as S
import xxz_common as X
from paths import OUT


def draw(means, ci):
    """means, ci: shape (4 groups, 4 alphas, 6 pairs)."""
    S.use()
    pair_labels = [rf"$\xi_{a[1]}${X.DASH}$\xi_{b[1]}$" for a, b in X.PAIR_ORDER]
    x = np.arange(len(pair_labels), dtype=float)
    width = 0.18
    n_alpha = len(X.ALPHAS_XXZ)
    offsets = (np.arange(n_alpha) - (n_alpha - 1) / 2) * width

    fig_w, fig_h = S.COL_W, 5.30
    leg_h, xlab_h = 0.34, 0.86
    ml, mr, mt = 0.80, 0.12, 0.26
    gap_x, gap_y = 0.26, 0.74
    mb = leg_h + xlab_h
    pw = (fig_w - ml - mr - gap_x) / 2
    ph = (fig_h - mb - mt - gap_y) / 2
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = []
    for gi in range(len(X.GROUPS)):
        row, col = divmod(gi, 2)
        axes.append(fig.add_axes(((ml + col * (pw + gap_x)) / fig_w, (mb + (1 - row) * (ph + gap_y)) / fig_h,
                                  pw / fig_w, ph / fig_h)))

    top = 0.0
    for gi, ax in enumerate(axes):
        for ai, (_, aval) in enumerate(X.ALPHAS_XXZ):
            ax.bar(x + offsets[ai], means[gi, ai], width=width, yerr=ci[gi, ai], capsize=S.CAPSIZE,
                   color=X.muted(X.COLORS_ALPHA_BARS[ai]), edgecolor="white", linewidth=0.5,
                   error_kw=dict(elinewidth=0.9, capthick=0.9, ecolor=X.darker(X.COLORS_ALPHA_BARS[ai], 0.62)),
                   label=rf"$\alpha={aval:.1f}$" if gi == 0 else None)
        top = max(top, float(np.nanmax(means[gi] + ci[gi])))
        ax.set_xticks(x)
        if gi >= 2:
            ax.set_xticklabels(pair_labels, rotation=40, ha="right", rotation_mode="anchor")
            ax.set_xlabel("Probe pair", fontsize=S.FS_LABEL, labelpad=1)
        else:
            ax.set_xticklabels([])
        S.journal_axes(ax, minor_x=False)
        S.panel_label(ax, chr(ord("a") + gi), dx=-0.02, dy=1.02)

    for gi, ax in enumerate(axes):                 # one y range: the four groups are compared with each other
        ax.set_ylim(0, top * 1.10)
        if gi % 2:
            ax.set_yticklabels([])
    fig.text(0.13 / fig_w, (mb + ph + gap_y / 2) / fig_h, r"Mean $|\Delta\tilde L_s|$", rotation=90,
             va="center", ha="left", fontsize=S.FS_LABEL)
    h_, lab_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, lab_, loc="lower center", ncol=len(X.ALPHAS_XXZ), fontsize=S.FS_LEGEND, frameon=False,
               columnspacing=1.0, handlelength=1.4, handletextpad=0.35,
               bbox_to_anchor=(0.5 + (ml - mr) / (2 * fig_w), 0.015 / fig_h))
    S.save(fig, OUT, "fig_xxz_pairdiff")
    plt.close(fig)


def main():
    xs, lams, anchors = X.grid()
    probe_idx, _ = X.probes(xs, lams, anchors)
    means, ci = X.pairdiff_stats(X.normalized_maps, probe_idx)
    draw(means, ci)


if __name__ == "__main__":
    main()
