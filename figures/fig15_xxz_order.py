"""Fig. 15: the four XXZ order-parameter fields with their quantile iso-contours.

N = 8 ground states on the 50 x 50 (J3/J6, lambda) grid. Panels: (a) zAFM, (b) qzAFM, (c) XY-QLRO, (d) VBS,
each with its own color scale; dashed curve = iso-contour at the f-quantile of the field
(f = 0.91, 0.97, 0.50, 0.82), the finite-size boundary drawn in Fig. 7; triangle = the training point of that panel's phase.
Order parameters as defined in App. E.2 (xxz_common.order_parameter_fields).

Input:  DATA/xxz_data/xxz_N8_grid50.npz (see xxz_common.py)
Output: OUT/fig_xxz_order_boundaries.pdf
"""
import matplotlib.pyplot as plt
import numpy as np

import style as S
import xxz_common as X
from paths import OUT

PANELS = [("(a) zAFM", "zAFM"), ("(b) qzAFM", "qzAFM"), ("(c) XY-QLRO", "XY-QLRO"), ("(d) VBS", "VBS")]


def draw(fields, extent, xlim, ylim, boundaries, anchors):
    """fields: name -> (rows = lambda, cols = J3/J6); boundaries: name -> (color, [segments]); anchors (4, 2)."""
    S.use()
    cmap = plt.get_cmap("RdYlBu").reversed()
    H, LEG = 4.72, 0.30                                   # bottom strip reserved for the legend
    fig, axes = plt.subplots(2, 2, figsize=(S.COL_W, H), sharex=True, sharey=True, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.004, h_pad=0.004, wspace=0.01, hspace=0.02)
    for k, (ax, (title, key)) in enumerate(zip(axes.ravel(), PANELS)):
        field = np.asarray(fields[key], float)
        im = ax.imshow(field, origin="lower", extent=extent, aspect="auto", cmap=cmap, interpolation="bicubic",
                       vmin=float(np.nanmin(field)), vmax=float(np.nanmax(field)), rasterized=True, zorder=1)
        color, segs = boundaries[key]
        for seg in segs:
            ax.plot(seg[:, 0], seg[:, 1], ls="--", lw=1.5, color=color, alpha=0.95, zorder=4, clip_on=False)
        X.scatter_train_anchor(ax, anchors[k:k+1, 0], anchors[k:k+1, 1], label="training point")  # only this phase's point
        ax.set_title(title, fontsize=S.FS_LABEL, pad=4)
        X.style_heatmap_axes(ax, xlim, ylim)
        cbar = fig.colorbar(im, ax=ax, fraction=0.050, pad=0.018)
        cbar.ax.tick_params(labelsize=S.FS_TICK)
        cbar.outline.set_linewidth(S.LW_AXES)
    for c in range(2):
        axes[1, c].set_xlabel(r"$J_3/J_6$", fontsize=S.FS_LABEL, labelpad=1)
    for r in range(2):
        axes[r, 0].set_ylabel(r"$\lambda$", fontsize=S.FS_LABEL, labelpad=1)
    h_, lab_ = axes[0, 0].get_legend_handles_labels()
    fig.get_layout_engine().set(rect=(0, LEG / H, 1, 1 - LEG / H))
    fig.legend(h_, lab_, loc="lower center", ncol=1, fontsize=S.FS_LEGEND, frameon=False, handletextpad=0.35,
               bbox_to_anchor=(0.5, 0.015 / H))
    fig.canvas.draw()                                     # one layout pass before saving fixes the panel margins
    S.save(fig, OUT, "fig_xxz_order_boundaries")
    plt.close(fig)


def main():
    xs, lams, anchors = X.grid()
    fields = X.order_parameter_fields()
    draw(fields, X.cell_extent(xs, lams), (0.0, 2.5), (0.0, 1.0), X.boundary_segments(xs, lams, fields), anchors)


if __name__ == "__main__":
    main()
