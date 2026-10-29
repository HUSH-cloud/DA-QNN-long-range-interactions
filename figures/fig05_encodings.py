"""Fig. 5: test accuracy versus alpha for the six encoding strategies E1-E6.

N = 8, h = 1.0, ell = 1, digit pair 3 vs 8; mean over 10 seeds (SEEDS_10).
Sites are counted from the readout site of the main scheme, so site 1 is the end of the chain and
site N/2 + 1 the central site carrying PC1 of the center-out placements (see data/prepare_data.py).
    E1  centerout,          read out on site 1        (the scheme used throughout the paper)
    E2  centerout_mirror,   read out on site 1
    E3  sequential_readout, read out on site 1        (PC1 on the readout site)
    E4  centerout,          read out on site N/2 + 1  (PC1 on the readout site)
    E5  centerout_mirror,   read out on site N/2 + 1  (PC1 on the readout site)
    E6  sequential_readout, read out on site N/2 + 1
Error bars: 95% confidence interval of the mean over seeds (Student t, ddof=1), drawn on E1 and E6 only.
Input:  RESULTS/fig5_encodings/acc_{placement}_q{site}_N{N}_l{ell}_a{alpha}_h{h}_s{seed}.npy
Output: OUT/fig_encodings.pdf
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

import style as S
from paths import OUT, RESULTS, SEEDS_10

RES = RESULTS / "fig5_encodings"
N_QUBITS = 8
H_FIELD = "1.0"
CENTER = N_QUBITS // 2 + 1
ENCODINGS = [("E1", "centerout", 1), ("E2", "centerout_mirror", 1), ("E3", "sequential_readout", 1),
             ("E4", "centerout", CENTER), ("E5", "centerout_mirror", CENTER), ("E6", "sequential_readout", CENTER)]


def load(ell):
    """acc[encoding, alpha, seed]."""
    acc = np.zeros((len(ENCODINGS), len(S.ALPHAS_5), len(SEEDS_10)))
    for ei, (_, placement, site) in enumerate(ENCODINGS):
        for ai, a in enumerate(S.ALPHAS_5):
            for si, s in enumerate(SEEDS_10):
                acc[ei, ai, si] = np.load(RES / f"acc_{placement}_q{site}_N{N_QUBITS}_l{ell}_a{a}_h{H_FIELD}_s{s}.npy").flat[0]
    return acc


def plot_encodings(ell, stem):
    """Draw the six encoding curves at the given ell and save OUT/stem.pdf."""
    S.use()
    acc = load(ell)
    x = np.arange(len(S.ALPHAS_5))
    colors = S.ordered_colors(len(ENCODINGS))

    fig_w, fig_h = S.COL_W, S.FIG_H
    leg_h, xlab_h = 0.34, 0.66
    ml, mr, mb, mt = 0.62, 0.14, leg_h + xlab_h, 0.14
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes((ml / fig_w, mb / fig_h, 1 - (ml + mr) / fig_w, 1 - (mb + mt) / fig_h))

    means, errs = [], []
    for ei, (label, _, _) in enumerate(ENCODINGS):
        m = acc[ei].mean(axis=1)
        e = S.ci_from_samples(acc[ei], axis=1)
        means.append(m)
        errs.append(e)
        S.errorbar_endpoints(ax, list(x), m, e, curve_idx=ei, n_curves=len(ENCODINGS), color=colors[ei],
                             marker="o", markersize=S.MS_DATA, capsize=S.CAPSIZE, linewidth=S.LW_DATA,
                             elinewidth=1.0, label=label)

    # Same accuracy ruler as the other accuracy figures; the frame opens slightly below 0.5.
    ylim = (0.47, 1.00)
    means, errs = np.array(means), np.array(errs)
    ends = [0, len(ENCODINGS) - 1]
    lo = min(means.min(), (means[ends] - errs[ends]).min())
    hi = max(means.max(), (means[ends] + errs[ends]).max())
    assert ylim[0] < lo and hi < ylim[1], f"data {lo:.3f}..{hi:.3f} is clipped by the frame {ylim}"
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.set_xlim(-0.32, len(x) - 0.68)
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\infty$" if a == "Inf" else a for a in S.ALPHAS_5])
    ax.set_xlabel(r"Interaction exponent  $\alpha$", fontsize=S.FS_LABEL)
    ax.set_ylabel("Test accuracy", fontsize=S.FS_LABEL)
    S.journal_axes(ax, minor_x=False)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(ENCODINGS), fontsize=S.FS_LEGEND, frameon=False,
               columnspacing=1.0, handlelength=1.5, handletextpad=0.35,
               bbox_to_anchor=(0.5 + (ml - mr) / (2 * fig_w), 0.015 / fig_h))

    S.save(fig, OUT, stem)
    plt.close(fig)


if __name__ == "__main__":
    plot_encodings(ell=1, stem="fig_encodings")
