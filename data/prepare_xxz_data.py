"""Frustrated XXZ ground states, Eq. (4) of the paper, for Figs. 7, 15 and 16, by dense exact diagonalization.

    python3 prepare_xxz_data.py [out dir]      (default: data/xxz_data)

H = sum_i [ J3 (X_i X_{i+1} + Y_i Y_{i+1} + lambda (X_i X_{i+2} + Y_i Y_{i+2})) + J6 (Z_i Z_{i+1} + lambda^2 Z_i Z_{i+2}) ]
on N = 8 sites with periodic boundaries and J6 = 1, so J3 equals the ratio J3/J6. Site i is bit i - 1 of the state
index, as in src/daqnn.jl.

Writes <out dir>/xxz_N8_grid50.npz with
    j3_ratios (50,)            J3/J6 = linspace(0.01, 2.5, 50)
    lambdas (50,)              lambda = linspace(0.01, 1.0, 50)
    test_params (2500, 2)      (J3/J6, lambda) of row i_lambda * 50 + i_ratio
    test_states (2500, 256)    complex128 ground states on the grid
    anchor_params (4, 2)       (J3/J6, lambda) of the training anchors zAFM, qzAFM, XY-QLRO, VBS
    anchor_states (4, 256)     complex128 training states
Each state is a real eigenvector of H (numpy.linalg.eigh) with its first component above 1e-12 in magnitude made
positive: the eigenvector of the lowest eigenvalue on the grid and at the qzAFM, XY-QLRO and VBS anchors. At the zAFM
anchor the two lowest levels form a quasi-degenerate Neel doublet (splitting 2.7e-7, asserted below 1e-6), and the
training state is its odd member under spin flip and translation, eigenvector index 1.
data/xxz_data holds the arrays used by the runs, with the same states up to a global phase.
"""
import sys
from pathlib import Path

import numpy as np

N = 8
J6 = 1.0
J3_RATIOS = np.linspace(0.01, 2.5, 50)
LAMBDAS = np.linspace(0.01, 1.0, 50)
ANCHOR_PARAMS = np.array([
    (0.01, 0.2184),       # zAFM
    (0.01, 0.9479),       # qzAFM
    (1.8447, 0.1663),     # XY-QLRO
    (1.5826, 0.6353),     # VBS
])
ANCHOR_LEVELS = [1, 0, 0, 0]          # eigenvector index taken at each anchor (0 = lowest)
DOUBLET_SPLITTING_MAX = 1e-6


def bond_sums(n, distance):
    """Sums over i of X_i X_{i+d} + Y_i Y_{i+d} and of Z_i Z_{i+d} with d = distance, periodic, each (2^n, 2^n)."""
    dim = 2 ** n
    xy = np.zeros((dim, dim))
    zz = np.zeros((dim, dim))
    for s in range(dim):
        for i in range(n):
            j = (i + distance) % n
            z_i = 1 - 2 * ((s >> i) & 1)
            z_j = 1 - 2 * ((s >> j) & 1)
            zz[s, s] += z_i * z_j
            if z_i != z_j:
                # X X + Y Y maps |01> to 2 |10> and |10> to 2 |01>, and annihilates |00> and |11>
                xy[s ^ (1 << i) ^ (1 << j), s] += 2.0
    return xy, zz


XY_NN, ZZ_NN = bond_sums(N, 1)
XY_NNN, ZZ_NNN = bond_sums(N, 2)


def hamiltonian(j3_ratio, lam):
    """Eq. (4) with J6 = 1 and J3 = j3_ratio * J6, real symmetric of shape (2^N, 2^N)."""
    j3 = j3_ratio * J6
    return j3 * (XY_NN + lam * XY_NNN) + J6 * (ZZ_NN + lam ** 2 * ZZ_NNN)


def fix_sign(state):
    """The real state, shape (2^N,), multiplied by -1 if its first component above 1e-12 in magnitude is negative."""
    for amplitude in state:
        if abs(amplitude) > 1e-12:
            return state if amplitude > 0 else -state


def eigenstate(j3_ratio, lam, level):
    """All eigenvalues in ascending order, shape (2^N,), and eigenvector `level`, shape (2^N,)."""
    energies, vectors = np.linalg.eigh(hamiltonian(j3_ratio, lam))
    return energies, fix_sign(vectors[:, level])


def main(out_dir):
    test_params = np.array([(j3_ratio, lam) for lam in LAMBDAS for j3_ratio in J3_RATIOS])
    test_states = np.array([eigenstate(j3_ratio, lam, 0)[1] for j3_ratio, lam in test_params])

    anchor_states = []
    for (j3_ratio, lam), level in zip(ANCHOR_PARAMS, ANCHOR_LEVELS):
        energies, state = eigenstate(j3_ratio, lam, level)
        if level > 0:
            assert energies[level] - energies[0] < DOUBLET_SPLITTING_MAX, \
                f"no quasi-degenerate doublet at J3/J6 = {j3_ratio}, lambda = {lam}"
        anchor_states.append(state)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "xxz_N8_grid50.npz",
        j3_ratios=J3_RATIOS, lambdas=LAMBDAS,
        test_params=test_params, test_states=test_states.astype(np.complex128),
        anchor_params=ANCHOR_PARAMS, anchor_states=np.array(anchor_states, dtype=np.complex128),
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "xxz_data")
