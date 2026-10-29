"""Angle-encoded MNIST digit pairs: standardize, PCA to 2N components, map each angle block linearly so the
training set spans [0, pi]; the test set uses the same map.

    python3 prepare_data.py [digit data dir] [encoding data dir]      (defaults: data/digit_data, data/encoding_data)

<digit data dir>/{a}_{b}/x_{train,test}_n{N}.npy    shape (2N, samples), the center-out placement, N = 3..10
<digit data dir>/{a}_{b}/y_{train,test}.npy         shape (samples, 2), one-hot, column 1 is digit a
<encoding data dir>/{placement}/x_{train,test}_n{N}.npy   pair 3 vs 8, N = 8, one folder per placement

Row j of x (1-based, j <= N) holds theta of encoded qubit j and row N + j its phi. The simulation
(src/daqnn.jl) puts encoded qubit j on chain site N + 1 - j, and the readout is site 1 (POS_MEASURE = 1),
so a placement is specified below by chain sites counted from the readout site and converted here.
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

SEED = 42
PAIRS = [(3, 8), (2, 7), (4, 9), (0, 1), (5, 8), (0, 6)]
QUBIT_COUNTS = range(3, 11)
ENCODING_COUNTS = (8,)
TRAIN_PER_DIGIT = 500
TEST_PER_DIGIT = 100


def outward(n, start, first_step):
    """Sites visited from `start` alternating to both sides, e.g. n = 8, start 5, first step -1: 5 4 6 3 7 2 8 1."""
    sites = [start]
    for d in range(1, n):
        for s in (start + first_step * d, start - first_step * d):
            if 1 <= s <= n and s not in sites:
                sites.append(s)
    return sites


def placements(n):
    """Chain sites (readout = site 1) receiving PC1, PC2, ... for every placement of the encoding runs."""
    far = n // 2 + 1                       # the central site farther from the readout
    return {
        "sequential_readout":    list(range(1, n + 1)),          # PC1 on the readout site
        "centerout":             outward(n, far, +1),            # PC1 on the far central site, then away from the readout
        "centerout_mirror":      outward(n, far, -1),            # same, but toward the readout first
    }


def encoded_rows(n, sites):
    """Row of the encoded array that carries the r-th component: encoded qubit N + 1 - site, 0-based."""
    return [n - s for s in sites]


def split(images, digits, a, b, rng):
    """Balanced train/test split of one pair. rng is consumed for every pair so splits do not depend on PAIRS."""
    mask = (digits == a) | (digits == b)
    X = images[mask]
    is_b = (digits[mask] == b).astype(np.float64)
    labels = np.stack([1 - is_b, is_b], axis=1)
    idx_a = np.where(is_b == 0)[0]
    idx_b = np.where(is_b == 1)[0]
    rng.shuffle(idx_a)
    rng.shuffle(idx_b)
    train = np.concatenate([idx_a[:TRAIN_PER_DIGIT], idx_b[:TRAIN_PER_DIGIT]])
    test = np.concatenate([idx_a[TRAIN_PER_DIGIT:TRAIN_PER_DIGIT + TEST_PER_DIGIT],
                           idx_b[TRAIN_PER_DIGIT:TRAIN_PER_DIGIT + TEST_PER_DIGIT]])
    rng.shuffle(train)
    rng.shuffle(test)
    return X[train], X[test], labels[train], labels[test]


def angles(components, rows, limits=None):
    """Component r goes to theta row rows[r] and component n + r to the matching phi row. Returns shape (2n, samples)."""
    n = len(rows)
    theta_column = [0] * n
    phi_column = [0] * n
    for rank, row in enumerate(rows):
        theta_column[row] = rank
        phi_column[row] = rank + n
    theta = components[:, theta_column]
    phi = components[:, phi_column]
    if limits is None:
        limits = (theta.min(), theta.max(), phi.min(), phi.max())
    t_min, t_max, p_min, p_max = limits
    theta = (theta - t_min) / (t_max - t_min + 1e-8) * np.pi
    phi = (phi - p_min) / (p_max - p_min + 1e-8) * np.pi
    return np.concatenate([theta, phi], axis=1).T, limits


def save_encoded(folder, n, pca_train, pca_test, sites):
    rows = encoded_rows(n, sites)
    x_train, limits = angles(pca_train, rows)
    x_test, _ = angles(pca_test, rows, limits)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f"x_train_n{n}.npy", x_train)
    np.save(folder / f"x_test_n{n}.npy", x_test)


def main(digit_dir, encoding_dir):
    mnist = fetch_openml("mnist_784", version=1, parser="auto")
    images = np.asarray(mnist.data, dtype=np.float64)
    digits = np.asarray(mnist.target, dtype=np.int32).ravel()
    rng = np.random.default_rng(SEED)

    for a, b in combinations(range(10), 2):
        X_train, X_test, y_train, y_test = split(images, digits, a, b, rng)
        if (a, b) not in PAIRS:
            continue
        scaler = StandardScaler().fit(X_train)
        Z_train, Z_test = scaler.transform(X_train), scaler.transform(X_test)
        folders = [digit_dir / f"{a}_{b}"]
        if (a, b) == (3, 8):
            folders += [encoding_dir / name for name in placements(8)]
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / "y_train.npy", y_train)
            np.save(folder / "y_test.npy", y_test)

        for n in QUBIT_COUNTS:
            pca = PCA(n_components=2 * n, random_state=SEED).fit(Z_train)
            pca_train, pca_test = pca.transform(Z_train), pca.transform(Z_test)
            save_encoded(digit_dir / f"{a}_{b}", n, pca_train, pca_test, placements(n)["centerout"])
            if (a, b) == (3, 8) and n in ENCODING_COUNTS:
                for name, sites in placements(n).items():
                    save_encoded(encoding_dir / name, n, pca_train, pca_test, sites)
        print(f"{a} vs {b} done")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else here / "digit_data",
         Path(sys.argv[2]) if len(sys.argv) > 2 else here / "encoding_data")
