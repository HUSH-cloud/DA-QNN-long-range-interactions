"""Fig. 9 data: time-averaged magnetization after a quench from |00...0> in the N = 8 chain, by exact diagonalization.

    python3 thermalization.py <output dir>

Writes tau_arr.npy, shape (N_T,), the Kac time grid tau = kac_norm * t on [0, 6 pi], and
Mz_avg_a{alpha}_h{h}.npy, shape (N_T,), for the six alpha at h = 1.0 and 3.0 drawn in Fig. 9.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import eigh

N = 8
TAU_MAX = 6 * np.pi
N_T = 4000
ALPHAS = [0.2, 0.4, 1.0, 3.0, 6.0, np.inf]
ALPHA_TAGS = ["0.2", "0.4", "1.0", "3.0", "6.0", "Inf"]
FIELDS = [1.0, 3.0]


def kac_norm(n, alpha):
    if np.isinf(alpha):
        return 1.0
    return sum(1.0 / (j - i) ** alpha for i in range(n) for j in range(i + 1, n)) / (n - 1)


def hamiltonian(n, alpha, h):
    """Dense H of shape (2^n, 2^n), same convention as src/daqnn.jl."""
    dim = 2 ** n
    H = np.zeros((dim, dim))
    bonds = [(i, i + 1) for i in range(n - 1)] if np.isinf(alpha) else [(i, j) for i in range(n) for j in range(i + 1, n)]
    for s in range(dim):
        diagonal = 0.0
        for i, j in bonds:
            z_i = 1 - 2 * ((s >> i) & 1)
            z_j = 1 - 2 * ((s >> j) & 1)
            J_ij = 1.0 if np.isinf(alpha) else 1.0 / (j - i) ** alpha
            diagonal -= J_ij * z_i * z_j
        H[s, s] = diagonal
    field = -kac_norm(n, alpha) * h
    for s in range(dim):
        for i in range(n):
            H[s, s ^ (1 << i)] += field
    return H


def averaged_magnetization(n, alpha, h, t):
    """Running time average (1/t) int_0^t M_z dt' of M_z = (1/N) sum_i <Z_i>, shape (len(t),)."""
    E, V = eigh(hamiltonian(n, alpha, h))
    amplitudes = V * V[0, :][None, :]
    psi_t = amplitudes @ np.exp(-1j * np.outer(E, t))
    mz_diagonal = np.array([(n - 2 * bin(s).count("1")) / n for s in range(2 ** n)])
    mz_inst = np.real(mz_diagonal @ np.abs(psi_t) ** 2)
    integral = np.zeros(len(t))
    integral[1:] = cumulative_trapezoid(mz_inst, t)
    mz_avg = np.where(t > 0, integral / np.maximum(t, 1e-15), mz_inst[0])
    return mz_avg


def main(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    tau = np.linspace(0, TAU_MAX, N_T)
    np.save(out_dir / "tau_arr.npy", tau)
    for h in FIELDS:
        for alpha, tag in zip(ALPHAS, ALPHA_TAGS):
            mz_avg = averaged_magnetization(N, alpha, h, tau / kac_norm(N, alpha))
            np.save(out_dir / f"Mz_avg_a{tag}_h{h}.npy", mz_avg)
            print(f"alpha={tag:>4}  h={h}  Mz_avg(6 pi)={mz_avg[-1]:+.4f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
