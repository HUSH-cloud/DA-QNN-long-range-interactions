"""Fig. 10: test accuracy over the (h, alpha) plane at ell = 1 (a) and ell = 4 (b).

N = 8, tau = 0.4 pi per layer, mean over 10 seeds. h runs along x, alpha up y; cells are square,
one color bar serves both panels, and the best and worst cell of each panel carry their value.

Inputs (under RESULTS), files acc_h{H}_a{ALPHA}_s{SEED}.npy in
    (a) fig9_h_alpha/ell1
    (b) fig9_h_alpha/ell4
Output: OUT/fig_h_alpha.pdf
"""
import matplotlib.pyplot as plt
import numpy as np

import style as S
from paths import OUT, RESULTS, SEEDS_10

PANELS = [(r"$\ell=1$", "fig9_h_alpha/ell1"),
          (r"$\ell=4$", "fig9_h_alpha/ell4")]
H_VALS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
ALPHAS = S.ALPHAS_10


def mean_accuracy(folder):
    """Seed-mean accuracy, shape (n_alpha, n_h)."""
    acc = np.zeros((len(SEEDS_10), len(H_VALS), len(ALPHAS)))
    for si, seed in enumerate(SEEDS_10):
        for i, h in enumerate(H_VALS):
            for j, alpha in enumerate(ALPHAS):
                acc[si, i, j] = float(np.load(RESULTS / folder / f"acc_h{h}_a{alpha}_s{seed}.npy").flat[0])
    return acc.mean(axis=0).T


def main():
    S.use()
    n_alpha, n_h = len(ALPHAS), len(H_VALS)

    # Panel width from the canvas width, panel height from square cells.
    gap, cgap, cbw = 0.13, 0.10, 0.17
    left, right = 0.62, 0.55
    bottom, top = 0.80, 0.32
    pw = (S.COL_W - left - right - gap - cgap - cbw) / 2
    ph = pw / n_h * n_alpha
    fig_h = bottom + ph + top

    fig = plt.figure(figsize=(S.COL_W, fig_h))
    for k, (title, folder) in enumerate(PANELS):
        m = mean_accuracy(folder)
        ax = fig.add_axes(((left + k * (pw + gap)) / S.COL_W, bottom / fig_h, pw / S.COL_W, ph / fig_h))
        im = ax.imshow(m, cmap=S.CMAP_ACC, vmin=S.ACC_VMIN, vmax=S.ACC_VMAX,
                       origin="lower", aspect="auto", interpolation="nearest")
        ax.set_xticks(range(n_h))
        ax.set_xticklabels([f"{v:.1f}" for v in H_VALS], rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(n_alpha))
        if k == 0:
            ax.set_yticklabels([r"$\infty$" if a == "Inf" else a for a in ALPHAS])
            ax.set_ylabel(r"Interaction exponent  $\alpha$", fontsize=S.FS_LABEL)
        else:
            ax.set_yticklabels([])
        S.journal_axes(ax, minor_x=False, minor_y=False, direction="out", sides=("bottom", "left"))
        ax.tick_params(colors="0.15")
        ax.set_title(title, fontsize=S.FS_LABEL, pad=5)
        S.panel_label(ax, "ab"[k], dx=-0.05, dy=1.02)
        for i, j in (np.unravel_index(np.argmax(m), m.shape), np.unravel_index(np.argmin(m), m.shape)):
            S.cell_value(ax, i, j, m[i, j], n_h)

    fig.text((left + pw + gap / 2) / S.COL_W, 0.155 / fig_h, r"Transverse field  $h$",
             ha="center", va="center", fontsize=S.FS_LABEL)

    cax = fig.add_axes(((left + 2 * pw + gap + cgap) / S.COL_W, bottom / fig_h, cbw / S.COL_W, ph / fig_h))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Test accuracy", fontsize=S.FS_CBAR, labelpad=7)
    S.journal_axes(cb.ax, minor_x=False, minor_y=True, direction="in", sides=("left", "right"))
    cb.ax.tick_params(labelsize=S.FS_TICK)
    cb.outline.set_linewidth(S.LW_AXES)

    S.save(fig, OUT, "fig_h_alpha")
    plt.close(fig)


if __name__ == "__main__":
    main()
