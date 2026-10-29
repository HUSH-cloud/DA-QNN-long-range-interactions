"""Fig. 9: time-averaged magnetization M_z_bar(tau) at N = 8 for six alpha.

(a) h = 1.0 over tau in [0, 6 pi], (b) h = 3.0 over tau in [0, pi]. The dotted line marks 0.1,
the dashed line zero.

Inputs (under THERMAL, written by experiments/thermalization.py):
    tau_arr.npy                     shape (N_T,)
    Mz_avg_a{alpha}_h{h}.npy        shape (N_T,)
Output: OUT/fig_thermalization.pdf
"""
from fractions import Fraction

import matplotlib.pyplot as plt
import numpy as np

import style as S
from paths import OUT, THERMAL

ALPHAS = ["0.2", "0.4", "1.0", "3.0", "6.0", "Inf"]
# (h, x-axis upper limit in units of pi, tick step in units of pi)
PANELS = [(1.0, 6, Fraction(1)), (3.0, 1, Fraction(1, 4))]


def pi_tick(k):
    """Label for k * pi with k a Fraction: 0, pi/4, pi/2, 3pi/4, pi, 2pi, ..."""
    if k == 0:
        return r"$0$"
    head = "" if k.numerator == 1 else str(k.numerator)
    return rf"${head}\pi$" if k.denominator == 1 else rf"${head}\pi/{k.denominator}$"


def load(h):
    tau = np.load(THERMAL / "tau_arr.npy")
    return tau, [np.load(THERMAL / f"Mz_avg_a{a}_h{h}.npy") for a in ALPHAS]


def main():
    S.use()
    colors = S.ordered_colors(len(ALPHAS))

    fig_w, fig_h = S.COL_W, 5.45
    leg_h, xlab_h = 0.62, 0.56
    ml, mr, mt, gap = 0.72, 0.14, 0.26, 0.46
    mb = leg_h + xlab_h
    ph = (fig_h - mb - mt - gap) / 2
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = [fig.add_axes((ml / fig_w, (mb + (1 - k) * (ph + gap)) / fig_h, 1 - (ml + mr) / fig_w, ph / fig_h))
            for k in range(2)]

    shown = []
    for i, (ax, (h, xmax_pi, step_pi)) in enumerate(zip(axes, PANELS)):
        tau, curves = load(h)
        for y, alpha, color in zip(curves, ALPHAS, colors):
            ax.plot(tau, y, color=color, linewidth=S.LW_DATA, label=S.alpha_label(alpha) if i == 0 else None)
            shown.append(y[tau <= xmax_pi * np.pi])
        ax.axhline(0, color="0.45", linewidth=0.7, linestyle="--")
        ax.axhline(0.1, color="0.45", linewidth=0.7, linestyle=":")
        ax.set_xlim(0, xmax_pi * np.pi)
        ks = [step_pi * j for j in range(int(Fraction(xmax_pi) / step_pi) + 1)]
        ax.set_xticks([float(k) * np.pi for k in ks])
        ax.set_xticklabels([pi_tick(k) for k in ks])
        S.journal_axes(ax, minor_x=True)
        S.panel_label(ax, "ab"[i], dx=-0.005, dy=1.02)
    axes[1].set_xlabel(r"$\tau$", fontsize=S.FS_LABEL)
    fig.text(0.115 / fig_w, (mb + ph + gap / 2) / fig_h, r"$\overline{M}_z(\tau)$",
             rotation=90, va="center", ha="left", fontsize=S.FS_LABEL)

    # Both panels share one y range covering every visible curve.
    v = np.concatenate(shown)
    lo, hi = min(v.min() - 0.06, -0.08), max(v.max() + 0.06, 1.05)
    for ax in axes:
        ax.set_ylim(lo, hi)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=S.FS_LEGEND, frameon=False,
               columnspacing=0.7, handlelength=1.3, handletextpad=0.3,
               bbox_to_anchor=(0.5 + (ml - mr) / (2 * fig_w), 0.015 / fig_h))

    S.save(fig, OUT, "fig_thermalization")


if __name__ == "__main__":
    main()
