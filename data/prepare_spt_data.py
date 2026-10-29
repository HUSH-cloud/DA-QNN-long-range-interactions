"""Cluster-SPT ground states for the phase classification of Figs. 6 and 14 (Sec. 4.1 and App. E.1 of the paper).

    python3 prepare_spt_data.py [out dir]      (default: data/spt_data)

<out dir>/cluster_spt_N{N}.npz, N = 6, 8, 10, with the same points and labels for every N:
    train_states (50, 2^N) complex128, train_params (50, 2) as [h1/J, h2/J], train_labels (50,) with 1 = SPT, 0 = non-SPT
        rows 0-39 are the stratified points (15 SPT, 25 non-SPT), rows 40-49 the added SPT points (25 SPT, 25 non-SPT in all)
    test_states (441, 2^N), test_params (441, 2), test_labels (441,)
    test_grid_params (441, 2)     the 21 x 21 grid before the boundary shift, h1 outer, h2 inner
    anti_boundary, para_boundary  (4,) cubic coefficients in h1, in numpy.polyval order

Eq. (3), open chain, J = 1:  H_SPT = -J sum_{i=1}^{N-2} Z_i X_{i+1} Z_{i+2} - h1 sum_i X_i - h2 sum_{i=1}^{N-1} X_i X_{i+1}
Chain site i is bit i - 1 of the basis index (Yao qubit i), so the readout qubit 1 is an end of the chain.
The two boundaries are cubic fits to ten tabulated points of the phase diagram of the reference cited with Eq. (3).
A point is SPT if anti(h1) + 0.01 < h2 < para(h1) - 0.01 and non-SPT if h2 lies more than 0.01 outside that band.
data/spt_data holds the arrays used by the runs, with the same states up to a global phase.
"""
import sys
from pathlib import Path

import numpy as np

SIZES = (6, 8, 10)
SEED = 42
H1_MIN, H1_MAX = 0.0, 1.6
H2_MIN, H2_MAX = -1.6, 1.6
GRID_POINTS = 21
MARGIN = 0.01                  # minimum h2 distance of every labeled point from both boundaries
SHIFT_EPS = 1e-6               # extra distance of the shifted test points, which sit at MARGIN + SHIFT_EPS
N_BINS = 5
POINTS_PER_BIN = 8             # stratified points
ADDED_PER_BIN = 2              # SPT points added to balance the two classes
DEGENERACY_TOL = 1e-10

# Ten tabulated points of the two phase boundaries, h2 as a function of h1.
BOUNDARY_H1 = np.array([0.1000, 0.2556, 0.4111, 0.5667, 0.7222, 0.8778, 1.0333, 1.1889, 1.3444, 1.5000])
BOUNDARY_ANTI = np.array([-1.004, -1.0009, -1.024, -1.049, -1.079, -1.109, -1.154, -1.225, -1.285, -1.35])
BOUNDARY_PARA = np.array([0.8439, 0.6636, 0.5033, 0.3631, 0.2229, 0.09766, -0.02755, -0.1377, -0.2479, -0.3531])


def boundary_fits():
    """Cubic least-squares fits (anti, para) to the tabulated boundary points, each of shape (4,)."""
    return np.polyfit(BOUNDARY_H1, BOUNDARY_ANTI, deg=3), np.polyfit(BOUNDARY_H1, BOUNDARY_PARA, deg=3)


def phase_label(h1, h2, anti, para):
    """1 inside the SPT band, 0 outside it, -1 within MARGIN of a boundary, e.g. (0.8, 0.0) -> 1, (0.8, 1.0) -> 0."""
    lower = np.polyval(anti, h1)
    upper = np.polyval(para, h1)
    if lower + MARGIN < h2 < upper - MARGIN:
        return 1
    if h2 < lower - MARGIN or h2 > upper + MARGIN:
        return 0
    return -1


def bin_edges():
    """Edges of the N_BINS equal h1 bins on [H1_MIN, H1_MAX], shape (N_BINS + 1,)."""
    return np.linspace(H1_MIN, H1_MAX, N_BINS + 1)


def stratified_points(anti, para):
    """Stratified points, shape (40, 2): per h1 bin 8 points with h1 uniform in the bin and h2 uniform on
    [H2_MIN, H2_MAX], h2 redrawn while the point lies within MARGIN of a boundary."""
    rng = np.random.default_rng(SEED)
    edges = bin_edges()
    points = []
    for b in range(N_BINS):
        for _ in range(POINTS_PER_BIN):
            h1 = rng.uniform(edges[b], edges[b + 1])
            h2 = rng.uniform(H2_MIN, H2_MAX)
            while phase_label(h1, h2, anti, para) == -1:
                h2 = rng.uniform(H2_MIN, H2_MAX)
            points.append([h1, h2])
    return np.array(points)


def added_spt_points(anti, para):
    """The 10 added SPT points, shape (10, 2): per h1 bin 2 points with h1 uniform in the bin and
    h2 uniform between the two boundaries, MARGIN inside each."""
    rng = np.random.default_rng(SEED)
    edges = bin_edges()
    points = []
    for b in range(N_BINS):
        for _ in range(ADDED_PER_BIN):
            h1 = rng.uniform(edges[b], edges[b + 1])
            h2_low = max(np.polyval(anti, h1) + MARGIN, H2_MIN)
            h2_high = min(np.polyval(para, h1) - MARGIN, H2_MAX)
            points.append([h1, rng.uniform(h2_low, h2_high)])
    return np.array(points)


def grid_points():
    """The 21 x 21 test grid, shape (441, 2); row k is (h1[k // 21], h2[k % 21])."""
    points = []
    for h1 in np.linspace(H1_MIN, H1_MAX, GRID_POINTS):
        for h2 in np.linspace(H2_MIN, H2_MAX, GRID_POINTS):
            points.append([h1, h2])
    return np.array(points)


def shift_off_boundaries(grid, anti, para):
    """Test points, shape (441, 2). A grid point within MARGIN of a boundary moves in h2 to the nearest of the four
    positions MARGIN + SHIFT_EPS above or below a boundary at which it receives a label (0 or 1); every other point
    is unchanged."""
    step = MARGIN + SHIFT_EPS
    points = grid.copy()
    for k in range(len(grid)):
        h1, h2 = grid[k]
        if phase_label(h1, h2, anti, para) != -1:
            continue
        lower = np.polyval(anti, h1)
        upper = np.polyval(para, h1)
        best = None
        for candidate in (lower - step, lower + step, upper - step, upper + step):
            candidate = min(max(candidate, H2_MIN), H2_MAX)
            if phase_label(h1, candidate, anti, para) == -1:
                continue
            if best is None or abs(candidate - h2) < abs(best - h2):
                best = candidate
        points[k, 1] = best
    return points


def labels_of(points, anti, para):
    """Labels of all points, shape (len(points),) int64."""
    return np.array([phase_label(h1, h2, anti, para) for h1, h2 in points], dtype=np.int64)


def hamiltonian(n, h1, h2):
    """Dense real H_SPT of Eq. (3) with J = 1, shape (2^n, 2^n); chain site i (0-based here) is bit i."""
    dim = 2 ** n
    H = np.zeros((dim, dim))
    for s in range(dim):
        for i in range(n - 2):
            z_left = 1.0 - 2.0 * ((s >> i) & 1)
            z_right = 1.0 - 2.0 * ((s >> (i + 2)) & 1)
            H[s ^ (1 << (i + 1)), s] -= z_left * z_right
        for i in range(n):
            H[s ^ (1 << i), s] -= h1
        for i in range(n - 1):
            H[s ^ (1 << i) ^ (1 << (i + 1)), s] -= h2
    return H


def ground_state(n, h1, h2):
    """Ground state of shape (2^n,) complex128: |0...0> projected onto the ground space (all eigenvalues within
    DEGENERACY_TOL of the lowest), normalized, then multiplied by a phase that makes its first component above
    1e-12 real and positive."""
    H = hamiltonian(n, h1, h2)
    energies, vectors = np.linalg.eigh(H)
    n_ground = int(np.count_nonzero(np.abs(energies - energies[0]) <= DEGENERACY_TOL))
    assert n_ground < len(energies), f"no gap above the ground space at N = {n}, h1 = {h1}, h2 = {h2}"
    ground_space = vectors[:, :n_ground]
    reference = np.zeros(2 ** n, dtype=complex)
    reference[0] = 1.0
    state = ground_space @ (ground_space.conj().T @ reference)
    norm = np.linalg.norm(state)
    assert norm > 1e-12, f"|0...0> has no overlap with the ground space at N = {n}, h1 = {h1}, h2 = {h2}"
    state = np.asarray(state / norm, dtype=np.complex128)
    first = np.flatnonzero(np.abs(state) > 1e-12)[0]
    state = state * np.exp(-1j * np.angle(state[first]))
    residual = np.linalg.norm(H @ state - energies[0] * state)
    assert residual < 1e-10, f"state is not an eigenstate (residual {residual:.1e}) at N = {n}, h1 = {h1}, h2 = {h2}"
    return state


def main(out_dir):
    anti, para = boundary_fits()
    h1_fine = np.linspace(H1_MIN, H1_MAX, 161)
    assert np.all(np.polyval(anti, h1_fine) < np.polyval(para, h1_fine) - 2 * MARGIN), \
        "anti must lie below para by more than 2 * MARGIN"

    train_params = np.concatenate([stratified_points(anti, para), added_spt_points(anti, para)])
    train_labels = labels_of(train_params, anti, para)
    assert np.count_nonzero(train_labels[:40] == 1) == 15 and np.count_nonzero(train_labels[:40] == 0) == 25, \
        "the stratified points are 15 SPT and 25 non-SPT"
    assert np.count_nonzero(train_labels == 1) == 25 and np.count_nonzero(train_labels == 0) == 25, \
        "the training set has 25 SPT and 25 non-SPT points"

    test_grid_params = grid_points()
    test_params = shift_off_boundaries(test_grid_params, anti, para)
    test_labels = labels_of(test_params, anti, para)
    assert np.count_nonzero(test_params[:, 1] != test_grid_params[:, 1]) == 5, "expected five shifted test points"
    assert np.count_nonzero(test_labels == 1) == 176 and np.count_nonzero(test_labels == 0) == 265, \
        "the test set has 176 SPT and 265 non-SPT points"

    out_dir.mkdir(parents=True, exist_ok=True)
    for n in SIZES:
        train_states = np.array([ground_state(n, h1, h2) for h1, h2 in train_params])
        test_states = np.array([ground_state(n, h1, h2) for h1, h2 in test_params])
        np.savez_compressed(out_dir / f"cluster_spt_N{n}.npz",
                            train_states=train_states, train_params=train_params, train_labels=train_labels,
                            test_states=test_states, test_params=test_params, test_labels=test_labels,
                            test_grid_params=test_grid_params, anti_boundary=anti, para_boundary=para)
        print(f"N = {n} done")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else here / "spt_data")
