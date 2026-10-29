"""Shared data and drawing helpers for the XXZ figures (paper Figs. 7, 15, 16).

Inputs:
    DATA/xxz_data/xxz_N8_grid50.npz      keys j3_ratios (50,), lambdas (50,), anchor_params (4, 2) and
                                         test_states (2500, 256), the ground states on the 50 x 50 (J3/J6, lambda)
                                         grid with row = i_lambda * 50 + i_ratio
    RESULTS/xxz/a{alpha}_anchor{k}_s{seed}.npz, alpha in 0.2, 1.0, 3.0, 6.0
        one circuit trained on training point k (0 zAFM, 1 qzAFM, 2 XY-QLRO, 3 VBS; 'anchor k' in the file names) at ansatz exponent alpha;
        key test_loss (2500,) = loss on the grid, entry i_lambda * 50 + i_ratio (reshaped to (50, 50) with
        rows = lambda), loss = mean over the 8 qubits of the probability of reading 0.

Functions
    grid()                    J3/J6 axis (50,), lambda axis (50,), training points (4, 2)
    normalized_maps(tag, k)   min-max normalized loss maps of the ten seeds, shape (10, 50, 50)
    minmax_normalize(loss)    (L - min L) / (max L - min L) on the grid
    probes(xs, lams, anchors) grid index (row, col) of the test point nearest to each training point
    pairdiff_stats(...)       mean and 95% CI of |Delta L~| for every training point, alpha and probe pair
    order_parameter_fields()  the four order parameters of App. E.2 on the grid, each (50, 50)
    contour_threshold(...)    value of a field at a given quantile
    boundary_segments(...)    dashed phase boundaries: quantile iso-contours of the order parameters
    cell_extent(xs, lams)     imshow extent with pixel centers on the grid points
    scatter_train_anchor, scatter_probe_star, style_heatmap_axes, muted, darker, minor_ticks_within_majors
"""
import colorsys
from functools import reduce

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, MaxNLocator
from scipy import stats

import style as S
from paths import DATA, RESULTS, SEEDS_10

XXZ_DATA = DATA / "xxz_data" / "xxz_N8_grid50.npz"
XXZ_RESULTS = RESULTS / "xxz"
ALPHAS_XXZ = [("0.2", 0.2), ("1.0", 1.0), ("3.0", 3.0), ("6.0", 6.0)]     # (file-name spelling, value) of alpha
GROUPS = [0, 1, 2, 3]
PAIR_ORDER = [("p0", "p1"), ("p0", "p2"), ("p0", "p3"), ("p1", "p2"), ("p1", "p3"), ("p2", "p3")]
COLORS_ALPHA_BARS = list(S.ordered_colors(5)[:4])          # first four of five steps of the red-to-blue palette
# phase (= order-parameter key), boundary color, quantile of the iso-contour (App. E.2)
PHASES = [("zAFM", "red", 0.91), ("qzAFM", "green", 0.97), ("XY-QLRO", "blue", 0.50), ("VBS", "black", 0.82)]
DASH = "\u2013"                                             # en dash (mathtext prints "--" as two hyphens)


# ---------------------------------------------------------------- data
def grid():
    """J3/J6 axis (50,), lambda axis (50,) and the four training points (4, 2) as (J3/J6, lambda)."""
    z = np.load(XXZ_DATA)
    return np.asarray(z["j3_ratios"], float), np.asarray(z["lambdas"], float), np.asarray(z["anchor_params"], float)


def minmax_normalize(loss):
    """(loss - min) / (max - min) over the grid; 1e-12 guards a constant map."""
    lo, hi = float(np.nanmin(loss)), float(np.nanmax(loss))
    return (loss - lo) / (hi - lo + 1e-12)


def normalized_maps(tag, k):
    """Loss map of each seed normalized to [0, 1] on the grid, shape (10, 50, 50), rows = lambda."""
    maps = []
    for s in SEEDS_10:
        z = np.load(XXZ_RESULTS / f"a{tag}_anchor{k}_s{s}.npz")
        maps.append(minmax_normalize(np.asarray(z["test_loss"], float).reshape(50, 50)))
    return np.stack(maps)


def probes(xs, lams, anchors):
    """Probe xi_k = the grid point nearest to training point k; returns [(row, col)] and their coordinates (4, 2)."""
    idx = [(int(np.argmin(np.abs(lams - b))), int(np.argmin(np.abs(xs - a)))) for a, b in anchors]
    return idx, np.array([(xs[c], lams[r]) for r, c in idx])


def pairdiff_stats(maps_of, probe_idx):
    """means, ci [group, alpha, pair]: seed mean of |L~(xi_a) - L~(xi_b)| and its two-sided 95% t interval.

    maps_of(tag, k) returns the normalized maps of the ten seeds, shape (seeds, rows, cols).
    """
    means = np.zeros((len(GROUPS), len(ALPHAS_XXZ), len(PAIR_ORDER)))
    ci = np.zeros_like(means)
    for gi, k in enumerate(GROUPS):
        for ai, (tag, _) in enumerate(ALPHAS_XXZ):
            maps = maps_of(tag, k)
            t = stats.t.ppf(0.975, len(maps) - 1)
            for pi, (a, b) in enumerate(PAIR_ORDER):
                (ya, xa), (yb, xb) = probe_idx[int(a[1])], probe_idx[int(b[1])]
                d = np.abs(maps[:, ya, xa] - maps[:, yb, xb])
                means[gi, ai, pi] = d.mean()
                ci[gi, ai, pi] = t * d.std(ddof=1) / np.sqrt(len(d))
    return means, ci


def order_parameter_fields():
    """Order parameters of App. E.2 on the grid, dict name -> (50, 50), rows = lambda.

    O_mu(k) = Re sum_{r=1}^{N-1} e^{ikr} <sigma^mu_1 sigma^mu_{1+r}>, (mu, k) = (z, pi) zAFM, (z, pi/2) qzAFM, (x, pi) XY-QLRO;
    O_VBS = Re sum_{r=1}^{N} (-1)^r <D_1 D_{1+r}>, D_k = sigma_k . sigma_{k+1} - sigma_{k-1} . sigma_k, sites modulo N.
    """
    z = np.load(XXZ_DATA)
    states = np.asarray(z["test_states"], complex)            # (2500, 256), row = i_lambda * 50 + i_ratio
    n = 8
    I2 = np.eye(2)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.diag([1.0, -1.0]).astype(complex)

    def full(ops, sites):
        factors = [I2] * n
        for op, site in zip(ops, sites):
            factors[site % n] = op
        return reduce(np.kron, factors)

    def corr(op, k):
        return sum(np.exp(1j * k * r) * full([op, op], [0, r]) for r in range(1, n))

    def bond(i, j):
        return full([X, X], [i, j]) + full([Y, Y], [i, j]) + full([Z, Z], [i, j])

    def diff(k):
        return bond(k, k + 1) - bond(k - 1, k)

    ops = {"zAFM": corr(Z, np.pi), "qzAFM": corr(Z, np.pi / 2), "XY-QLRO": corr(X, np.pi),
           "VBS": sum((-1) ** r * (diff(0) @ diff(r)) for r in range(1, n + 1))}
    return {name: np.einsum("si,ij,sj->s", states.conj(), op, states).real.reshape(50, 50) for name, op in ops.items()}


def contour_threshold(field, fraction):
    """Value at the given fraction of the sorted field (the rule of App. E.2)."""
    values = np.sort(np.asarray(field, float).ravel())
    return float(values[round(fraction * values.size)])


def boundary_segments(xs, lams, fields):
    """Dict phase -> (color, [segments (m, 2) of (J3/J6, lambda)]): each field's quantile iso-contour."""
    out = {}
    for name, color, fraction in PHASES:
        fig, ax = plt.subplots()
        cs = ax.contour(xs, lams, fields[name], [contour_threshold(fields[name], fraction)])
        segs = [np.asarray(s, float) for s in cs.allsegs[0] if len(s)]
        plt.close(fig)
        out[name] = (color, segs)
    return out


def cell_extent(xs, lams):
    """imshow extent that puts every pixel center on its grid point."""
    dx, dy = np.mean(np.diff(xs)), np.mean(np.diff(lams))
    return [xs[0] - dx / 2, xs[-1] + dx / 2, lams[0] - dy / 2, lams[-1] + dy / 2]


# ---------------------------------------------------------------- drawing helpers
def scatter_train_anchor(ax, x, y, *, label=None):
    """Green triangle(s) at the training point(s), drawn over the frame."""
    kw = dict(marker="^", s=140, facecolor="lime", edgecolor="deepskyblue", linewidth=1.1, zorder=6, clip_on=False)
    if label is not None:
        kw["label"] = label
    return ax.scatter(np.atleast_1d(x), np.atleast_1d(y), **kw)


def scatter_probe_star(ax, x, y):
    """Purple star at one probe point, drawn over the frame."""
    return ax.scatter([x], [y], s=220, marker="*", facecolors="#6a6cff", edgecolors="white", linewidths=0.7,
                      zorder=5, clip_on=False)


def style_heatmap_axes(ax, xlim, ylim):
    """Limits, outward ticks and frame width of a heat-map panel."""
    ax.grid(False)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.tick_params(direction="out", length=3.8, width=S.LW_AXES, labelsize=S.FS_TICK)
    for spine in ax.spines.values():
        spine.set_linewidth(S.LW_AXES)


def muted(c, sat=0.68, val=1.06):
    """Calmer version of a palette color for filled bars: same hue, less saturation."""
    h, s_, v = colorsys.rgb_to_hsv(*mcolors.to_rgb(c))
    return colorsys.hsv_to_rgb(h, s_ * sat, min(1.0, v * val))


def darker(c, f=0.62):
    """Darker shade of a bar color, for its error bar."""
    r, g, b = mcolors.to_rgb(c)
    return (r * f, g * f, b * f)


def minor_ticks_within_majors(axis):
    """Drop minor ticks beyond the outermost major tick (call after S.journal_axes)."""
    v0, v1 = sorted(axis.get_view_interval())
    majors = [t for t in axis.get_majorticklocs() if v0 <= t <= v1]
    if majors:
        lo, hi = min(majors), max(majors)
        axis.set_minor_locator(FixedLocator([t for t in axis.get_minorticklocs() if lo <= t <= hi]))
