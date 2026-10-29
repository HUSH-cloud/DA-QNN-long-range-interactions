"""Fig. 14: seed-averaged cluster-SPT output probability on the (h1/J, h2/J) test grid.

N = 8, alpha = 0.2, h = 1.0, tau = 0.4 pi, 500 iterations, panels (a)-(d) ell = 1, 2, 4, 8.
Color: mean over the 10 seeds (SEEDS_10) of the SPT probability 1 - p0 (test_prob_spt) at each of the 441 test points.
Circles: the training states. Dashed curves: the two label boundaries of the dataset (cubic fits in h1/J),
lower (anti) in blue and upper (para) in red.

Input:  RESULTS/spt/N8_l{ell}_a0.2_s{seed}.npz            (key test_prob_spt)
        DATA/spt_data/cluster_spt_N8.npz                    (keys test_grid_params, test_params, train_params,
                                                             anti_boundary, para_boundary)
Output: OUT/fig_haldane_prob_a0p2.pdf
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

import style as S
from paths import DATA, OUT, RESULTS, SEEDS_10

ELLS = [1, 2, 4, 8]
GRID = 21
PROB_CMAP = LinearSegmentedColormap.from_list("green_white_yellow", [(0.0, "#00B094"), (0.5, "white"), (1.0, "#FBFFA0")])


def load():
    """Display grid (441, 2), seed-mean SPT probability (4 ells, 441), training points (50, 2) and the anti and para
    boundary polynomials."""
    data = np.load(DATA / "spt_data" / "cluster_spt_N8.npz")
    grid = np.asarray(data["test_grid_params"], float)              # (441, 2), h1 outer, h2 inner
    params = np.asarray(data["test_params"], float)                 # (441, 2), the points the runs were tested on
    train = np.asarray(data["train_params"], float)                 # (50, 2) training states
    # The five boundary-safe test points are displaced slightly in h2; draw them on their grid cell.
    assert np.allclose(params[:, 0], grid[:, 0]) and np.abs(params[:, 1] - grid[:, 1]).max() < 0.05, \
        "test points do not match the 21 x 21 grid"
    mean = np.zeros((len(ELLS), GRID * GRID))
    for li, ell in enumerate(ELLS):
        stack = []
        for s in SEEDS_10:
            z = np.load(RESULTS / "spt" / f"N8_l{ell}_a0.2_s{s}.npz")
            stack.append(np.asarray(z["test_prob_spt"], float))
        mean[li] = np.mean(stack, axis=0)
    anti = np.poly1d(data["anti_boundary"])
    para = np.poly1d(data["para_boundary"])
    return grid, mean, train, anti, para


def main():
    S.use()
    grid, mean, train, anti, para = load()
    h1 = np.unique(np.round(grid[:, 0], 10))
    h2 = np.unique(np.round(grid[:, 1], 10))
    dx, dy = np.mean(np.diff(h1)), np.mean(np.diff(h2))
    extent = [h1.min() - dx / 2, h1.max() + dx / 2, h2.min() - dy / 2, h2.max() + dy / 2]

    fig, axes = plt.subplots(2, 2, figsize=(S.COL_W, 4.87), sharex=True, sharey=True, constrained_layout=True)
    for i, (ax, ell) in enumerate(zip(axes.ravel(), ELLS)):
        # reshape(21, 21): rows = h1, columns = h2; transpose so that h2 runs vertically.
        im = ax.imshow(mean[i].reshape(GRID, GRID).T, origin="lower", extent=extent, cmap=PROB_CMAP,
                       vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest", rasterized=True)
        ax.grid(False)
        ax.scatter(train[:, 0], train[:, 1], s=22, facecolors="none", edgecolors="black", linewidths=1.0, zorder=5)
        xs = np.linspace(h1.min(), h1.max(), 400)
        ax.plot(xs, anti(xs), color="#1F77B4", ls="--", lw=1.6, zorder=6)
        ax.plot(xs, para(xs), color="#C44E52", ls="--", lw=1.6, zorder=6)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        S.panel_label(ax, "abcd"[i], dx=-0.02, dy=1.02)
        if i >= 2:
            ax.set_xlabel(r"$h_1/J$", fontsize=S.FS_LABEL)
        if i % 2 == 0:
            ax.set_ylabel(r"$h_2/J$", fontsize=S.FS_LABEL)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(direction="out", length=3.8, width=S.LW_AXES, labelsize=S.FS_TICK)
        for spine in ax.spines.values():
            spine.set_linewidth(S.LW_AXES)

    cbar = fig.colorbar(im, ax=axes, fraction=0.042, pad=0.015, aspect=34)
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_label("Output probability", fontsize=S.FS_CBAR)
    cbar.ax.tick_params(labelsize=S.FS_TICK)
    cbar.outline.set_linewidth(S.LW_AXES)
    S.save(fig, OUT, "fig_haldane_prob_a0p2")
    plt.close(fig)


if __name__ == "__main__":
    main()
