"""Fig. 8: test-accuracy distribution per MNIST digit pair at h = 1.0.

Each pair pools ell in {1, 2, 4}, N = 3..10, five alpha and 20 seeds (2400 runs). Pairs are sorted
by median accuracy, easiest (red) on the right. Each pair shows a Gaussian-KDE density over the
observed range, the interquartile box, Tukey whiskers (1.5 IQR) and the median as a white dot.

Inputs (under RESULTS):
    fig4_sizes/acc_sitA_l{ell}_N{N}_a{alpha}_p{ab}_s{SEED}.npy
Output: OUT/fig_task_difficulty.pdf
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import style as S
from paths import OUT, RESULTS, SEEDS_20

ELLS = [1, 2, 4]
NS = range(3, 11)
PAIRS = [(3, 8), (2, 7), (4, 9), (0, 1), (5, 8), (0, 6)]

BOX_W = 0.16      # box width in data units
VIOLIN_W = 0.40   # half-width of the density at its widest
PAD = 0.03        # y headroom as a fraction of the data span


def pair_runs(pair):
    """All accuracies of one digit pair, shape (len(ELLS) * len(NS) * 5 * 20,)."""
    a, b = pair
    runs = []
    for ell in ELLS:
        for N in NS:
            for alpha in S.ALPHAS_5:
                for seed in SEEDS_20:
                    f = RESULTS / "fig4_sizes" / f"acc_sitA_l{ell}_N{N}_a{alpha}_p{a}{b}_s{seed}.npy"
                    runs.append(float(np.load(f).flat[0]))
    return np.array(runs)


def whiskers(v):
    """Tukey whiskers: the extreme datum within 1.5 IQR of the quartiles."""
    q1, q3 = np.percentile(v, [25, 75])
    r = 1.5 * (q3 - q1)
    return v[v >= q1 - r].min(), v[v <= q3 + r].max()


def main():
    S.use()
    data = [pair_runs(p) for p in PAIRS]
    order = np.argsort([np.median(d) for d in data])
    labels = [f"{PAIRS[i][0]}v{PAIRS[i][1]}" for i in order]
    data = [data[i] for i in order]
    n = len(labels)
    colors = list(reversed(S.ordered_colors(n)))

    fig_w, fig_h = S.COL_W, S.FIG_H
    ml, mr, mb, mt = 0.62, 0.14, 0.50, 0.14
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes((ml / fig_w, mb / fig_h, 1 - (ml + mr) / fig_w, 1 - (mb + mt) / fig_h))

    lo, hi = min(d.min() for d in data), max(d.max() for d in data)
    span = hi - lo

    for i, (v, c) in enumerate(zip(data, colors)):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        wlo, whi = whiskers(v)
        # Density over the observed support only, so it stops at the extreme run.
        grid = np.linspace(v.min(), v.max(), 256)
        dens = gaussian_kde(v)(grid)
        dens = dens / dens.max() * VIOLIN_W
        ax.fill_betweenx(grid, i - dens, i + dens, facecolor=c, alpha=0.55, edgecolor=c, linewidth=0.9, zorder=2)
        ax.add_patch(plt.Rectangle((i - BOX_W / 2, q1), BOX_W, q3 - q1,
                                   facecolor="0.22", edgecolor="0.15", linewidth=0.9, zorder=4))
        ax.plot([i, i], [wlo, q1], color="0.15", linewidth=0.9, zorder=3)
        ax.plot([i, i], [q3, whi], color="0.15", linewidth=0.9, zorder=3)
        for y in (wlo, whi):
            ax.plot([i - BOX_W / 2, i + BOX_W / 2], [y, y], color="0.15", linewidth=0.9, zorder=3)
        ax.plot([i], [med], marker="o", markersize=3.4, color="white",
                markeredgecolor="0.15", markeredgewidth=0.6, zorder=5)

    ax.set_ylim(lo - PAD * span, min(1.0, hi + PAD * span))
    ax.set_xlim(-0.62, n - 0.38)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Digit pair", fontsize=S.FS_LABEL)
    ax.set_ylabel("Test accuracy", fontsize=S.FS_LABEL)
    S.journal_axes(ax, minor_x=False)

    S.save(fig, OUT, "fig_task_difficulty")


if __name__ == "__main__":
    main()
