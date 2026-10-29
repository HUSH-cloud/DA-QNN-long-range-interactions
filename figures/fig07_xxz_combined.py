"""Fig. 7: XXZ loss maps at alpha = 1.0 and 6.0, and the probe-pair contrasts of the circuit trained on the VBS point.

(a1)-(a4) seed-averaged min-max normalized loss maps, alpha = 1.0, one training point per column
          (zAFM, qzAFM, XY-QLRO, VBS); (b1)-(b4) the same at alpha = 6.0.
          Dashed curves: order-parameter iso-contours of Fig. 15; triangles: training points; stars: probes xi_0-xi_3.
(c)       |Delta L~| for the VBS training point at alpha in {0.2, 1.0, 6.0}, seed mean with 95% t interval.

N = 8, ell = 2, h = 1.0, tau = 0.4 pi, ten seeds (SEEDS_10).

Input:  RESULTS/xxz and DATA/xxz_data/xxz_N8_grid50.npz (see xxz_common.py)
Output: OUT/fig_xxz_combined.pdf
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import style as S
import xxz_common as X
from paths import OUT

ALPHA_ROWS = ["1.0", "6.0"]
BAR_ALPHAS = ["0.2", "1.0", "6.0"]
BAR_GROUP = 3                                            # the VBS training point


def draw(maps, anchors, probe_xy, boundaries, means, ci, extent, xlim, ylim):
    """maps[tag][k] (rows = lambda, cols = J3/J6); anchors[tag][k] = (J3/J6, lambda); probe_xy {name: (x, y)};
    boundaries: list of (color, [segments]) in drawing order; means, ci: (4 groups, 4 alphas, 6 pairs)."""
    S.use()
    keep = [i for i, (t, _) in enumerate(X.ALPHAS_XXZ) if t in BAR_ALPHAS]
    gi = X.GROUPS.index(BAR_GROUP)
    pair_labels = [rf"$\xi_{a[1]}${X.DASH}$\xi_{b[1]}$" for a, b in X.PAIR_ORDER]

    def off(px, py):                                     # label offset of a probe star, in points
        dx, dy = 7, 7
        if px < xlim[0] + .15 * (xlim[1] - xlim[0]): dx = 9
        if px > xlim[1] - .15 * (xlim[1] - xlim[0]): dx = -26
        if py > ylim[1] - .12 * (ylim[1] - ylim[0]): dy = -13
        if py < ylim[0] + .12 * (ylim[1] - ylim[0]): dy = 9
        return dx, dy

    FW, FH = S.FULL_W, 4.72
    fig = plt.figure(figsize=(FW, FH))
    gs = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 1, 0.085, 0.42, 1.70], left=0.088, right=0.943,
                          bottom=0.205, top=0.938, wspace=0.095, hspace=0.235)
    map_axes = [[None] * 4 for _ in range(2)]
    last = None
    for r, tag in enumerate(ALPHA_ROWS):
        for c in range(len(X.GROUPS)):
            ax = fig.add_subplot(gs[r, c])
            last = ax.imshow(maps[tag][c], origin="lower", extent=extent, aspect="auto", cmap="RdYlBu",
                             interpolation="bicubic", vmin=0, vmax=1, rasterized=True, zorder=1)
            ax.grid(False)
            for color, segs in boundaries:
                for seg in segs:
                    ax.plot(seg[:, 0], seg[:, 1], ls="--", lw=1.15, color=color, alpha=.95, zorder=4, clip_on=False)
            for name, (px, py) in probe_xy.items():
                X.scatter_probe_star(ax, px, py)
                odx, ody = off(px, py)
                ax.annotate(name, xy=(px, py), xytext=(odx, ody), textcoords="offset points",
                            fontsize=S.FS_ANNOT - 1.5, color="#6a6cff", clip_on=False)
            X.scatter_train_anchor(ax, *anchors[tag][c])
            ax.text(-0.02, 1.025, f"({'ab'[r]}{c + 1})", transform=ax.transAxes, fontsize=S.FS_PANEL,
                    fontweight="bold", va="bottom", ha="left")
            X.style_heatmap_axes(ax, xlim, ylim)
            if c == 0:
                ax.set_ylabel(r"$\lambda$", fontsize=S.FS_LABEL)
            else:
                ax.tick_params(labelleft=False)
            if r == 0:
                ax.tick_params(labelbottom=False)
            ax.set_xticks([0.0, 0.8, 1.6, 2.4])
            if r == 1:
                ax.set_xticklabels(["0.0", "0.8", "1.6", "" if c < 3 else "2.4"])
            map_axes[r][c] = ax
            if c == 0:                                   # row label: alpha
                av = dict(X.ALPHAS_XXZ)[tag]
                ax.text(-0.52, 0.5, rf"$\alpha={av:.1f}$", transform=ax.transAxes, fontsize=S.FS_LABEL,
                        rotation=90, va="center", ha="center")

    cax = fig.add_subplot(gs[:, 4])
    cb = fig.colorbar(last, cax=cax)
    cb.set_ticks([0, .2, .4, .6, .8, 1.])
    cb.set_label("normalized loss", fontsize=S.FS_CBAR, labelpad=6)
    cb.ax.tick_params(labelsize=S.FS_TICK)

    axb = fig.add_subplot(gs[:, 6])                      # panel (c)
    x = np.arange(len(pair_labels), dtype=float)
    w = .27
    offs = (np.arange(len(keep)) - (len(keep) - 1) / 2) * w
    for j, ai in enumerate(keep):
        axb.bar(x + offs[j], means[gi, ai], width=w, yerr=ci[gi, ai], capsize=S.CAPSIZE,
                color=X.muted(X.COLORS_ALPHA_BARS[ai]), edgecolor="white", linewidth=.6,
                error_kw=dict(elinewidth=1.0, capthick=1.0, ecolor=X.darker(X.COLORS_ALPHA_BARS[ai], .62)),
                label=rf"$\alpha={X.ALPHAS_XXZ[ai][1]:.1f}$")
    axb.set_xticks(x)
    axb.set_xticklabels(pair_labels, rotation=28, ha="right", rotation_mode="anchor", fontsize=S.FS_TICK - 2)
    axb.set_ylim(0, float(np.nanmax(means[gi, keep] + ci[gi, keep])) * 1.10)
    axb.set_ylabel(r"Mean $|\Delta\tilde L_s|$", fontsize=S.FS_LABEL)
    axb.set_xlabel("Probe pair", fontsize=S.FS_LABEL)
    S.journal_axes(axb, minor_x=False)
    X.minor_ticks_within_majors(axb.yaxis)
    axb.yaxis.set_label_position("right")               # keep clear of the color-bar label
    axb.tick_params(axis="y", labelleft=False, labelright=True)
    pa, pj = map_axes[0][0].get_position(), axb.get_position()
    fig.text(pj.x0 - 0.02 * pj.width, pa.y1 + 0.025 * pa.height, "(c)", fontsize=S.FS_PANEL, fontweight="bold",
             va="bottom", ha="left")
    hb, lb = axb.get_legend_handles_labels()

    x0, x1 = map_axes[1][0].get_position().x0, map_axes[1][3].get_position().x1
    xc = (x0 + x1) / 2
    fig.text(xc, 0.098, r"$J_3/J_6$", ha="center", va="bottom", fontsize=S.FS_LABEL)
    fig.legend(handles=[
        Line2D([0], [0], marker="^", color="none", markerfacecolor="lime", markeredgecolor="deepskyblue",
               markeredgewidth=1.1, markersize=10, label="training point"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#6a6cff", markeredgecolor="white",
               markeredgewidth=.7, markersize=12, label=rf"test-grid probe $\xi_0${X.DASH}$\xi_3$")],
        loc="lower center", ncol=2, fontsize=S.FS_LEGEND, frameon=False, handletextpad=.4, columnspacing=2.2,
        bbox_to_anchor=(xc, 0.008))
    pb = axb.get_position()
    fig.legend(hb, lb, loc="lower center", ncol=3, fontsize=S.FS_LEGEND - 1, frameon=False, handlelength=1.1,
               handletextpad=.32, columnspacing=1.0, bbox_to_anchor=((pb.x0 + pb.x1) / 2, 0.008))
    S.save(fig, OUT, "fig_xxz_combined")
    plt.close(fig)


def main():
    xs, lams, anchors = X.grid()
    probe_idx, probe_coords = X.probes(xs, lams, anchors)
    maps = {tag: [X.normalized_maps(tag, k).mean(axis=0) for k in X.GROUPS] for tag in ALPHA_ROWS}
    anchor_xy = {tag: [tuple(anchors[k]) for k in X.GROUPS] for tag in ALPHA_ROWS}
    fields = X.order_parameter_fields()
    seg = X.boundary_segments(xs, lams, fields)
    boundaries = [seg["XY-QLRO"], seg["qzAFM"], seg["zAFM"], seg["VBS"]]      # drawing order, last on top
    means, ci = X.pairdiff_stats(X.normalized_maps, probe_idx)
    probe_xy = {rf"$\xi_{k}$": tuple(probe_coords[k]) for k in range(4)}
    draw(maps, anchor_xy, probe_xy, boundaries, means, ci, X.cell_extent(xs, lams), (0.0, 2.5), (0.0, 1.0))


if __name__ == "__main__":
    main()
