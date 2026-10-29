"""Fig. 2: test accuracy over the (tau, ell) plane at h = 1.0 (a) and h = 3.0 (b).

N = 8, alpha = 1, mean over 10 seeds. Rows are tau = k pi / 5 for k = 1..8 (tau <= 1.6 pi),
labeled as tau / 2 pi; columns are ell = 1, 2, 3, 4, 6, 8, 10. Cells are square, one color bar
serves both panels, and the best and worst cell of each panel carry their value.

Inputs (under RESULTS):
    fig2_grid/h{1,3}/seed_{SEED}/acc_t{k}_l{j}.npy               ell = 2, 4, 6, 8, 10 (j = 1..5)
    fig2_columns/h{1,3}/acc_a1.0_tt{k}_l{ell}_s{SEED}.npy        ell = 1, 3
Output: OUT/fig_tl_heatmap.pdf
"""
import matplotlib.pyplot as plt
import numpy as np

import style as S
from paths import OUT, RESULTS, SEEDS_10

PANELS = [("h1", r"$h=1.0$"), ("h3", r"$h=3.0$")]
TAU_IDX = range(1, 9)                      # tau = k pi / 5, k = 1..8
ELL_GRID = [2, 4, 6, 8, 10]                # fig2_grid runs, file index j = 1..5
ELL_COLUMNS = [1, 3]                       # fig2_columns runs
ELLS = sorted(ELL_GRID + ELL_COLUMNS)


def read(path):
    return float(np.load(path).flat[0])


def mean_accuracy(h):
    """Seed-mean accuracy, shape (n_tau, n_ell) with ell in ELLS order."""
    acc = np.zeros((len(SEEDS_10), len(TAU_IDX), len(ELLS)))
    for si, seed in enumerate(SEEDS_10):
        for ti, k in enumerate(TAU_IDX):
            for ell in ELLS:
                if ell in ELL_GRID:
                    f = RESULTS / "fig2_grid" / h / f"seed_{seed}" / f"acc_t{k}_l{ELL_GRID.index(ell) + 1}.npy"
                else:
                    f = RESULTS / "fig2_columns" / h / f"acc_a1.0_tt{k}_l{ell}_s{seed}.npy"
                acc[si, ti, ELLS.index(ell)] = read(f)
    return acc.mean(axis=0)


def main():
    S.use()
    tau_over_pi = [k / 5 for k in TAU_IDX]
    n_tau, n_ell = len(TAU_IDX), len(ELLS)

    # Panel width from the canvas width, panel height from square cells.
    gap, cgap, cbw = 0.13, 0.10, 0.17
    left, right = 0.62, 0.55
    bottom, top = 0.62, 0.32
    pw = (S.COL_W - left - right - gap - cgap - cbw) / 2
    ph = pw / n_ell * n_tau
    fig_h = bottom + ph + top

    fig = plt.figure(figsize=(S.COL_W, fig_h))
    for k, (h, title) in enumerate(PANELS):
        m = mean_accuracy(h)
        ax = fig.add_axes(((left + k * (pw + gap)) / S.COL_W, bottom / fig_h, pw / S.COL_W, ph / fig_h))
        im = ax.imshow(m, cmap=S.CMAP_ACC, vmin=S.ACC_VMIN, vmax=S.ACC_VMAX,
                       origin="lower", aspect="auto", interpolation="nearest")
        ax.set_xticks(range(n_ell))
        ax.set_xticklabels([str(e) for e in ELLS])
        ax.set_yticks(range(n_tau))
        if k == 0:
            ax.set_yticklabels([f"{t / 2:.1f}" for t in tau_over_pi])
            ax.set_ylabel(r"$\tau/2\pi$", fontsize=S.FS_LABEL)
        else:
            ax.set_yticklabels([])
        S.journal_axes(ax, minor_x=False, minor_y=False, direction="out", sides=("bottom", "left"))
        ax.tick_params(colors="0.15")
        ax.set_title(title, fontsize=S.FS_LABEL, pad=5)
        S.panel_label(ax, "ab"[k], dx=-0.05, dy=1.02)
        for i, j in (np.unravel_index(np.argmax(m), m.shape), np.unravel_index(np.argmin(m), m.shape)):
            S.cell_value(ax, i, j, m[i, j], n_ell)

    fig.text((left + pw + gap / 2) / S.COL_W, 0.155 / fig_h, r"$\ell$",
             ha="center", va="center", fontsize=S.FS_LABEL)

    cax = fig.add_axes(((left + 2 * pw + gap + cgap) / S.COL_W, bottom / fig_h, cbw / S.COL_W, ph / fig_h))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Test accuracy", fontsize=S.FS_CBAR, labelpad=7)
    S.journal_axes(cb.ax, minor_x=False, minor_y=True, direction="in", sides=("left", "right"))
    cb.ax.tick_params(labelsize=S.FS_TICK)
    cb.outline.set_linewidth(S.LW_AXES)

    S.save(fig, OUT, "fig_tl_heatmap")
    plt.close(fig)


if __name__ == "__main__":
    main()
