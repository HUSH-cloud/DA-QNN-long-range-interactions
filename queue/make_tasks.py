"""Write queue shards for the computed data behind the paper figures.

    python3 make_tasks.py <results root> [n_shards]

Each phase becomes <root>/_queue/<phase>.shard{k}.txt and one line of <root>/_queue/manifest.tsv.
Result subfolders are the folders the figure scripts read.
Data locations default to the files in data/ and can be overridden with DIGIT_DATA, ENCODING_DATA (data/prepare_data.py),
SPT_DATA (data/prepare_spt_data.py), XXZ_DATA (data/prepare_xxz_data.py) and INIT_FILE (data/initial_params_julia110.jl).
"""
import math
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
EXPERIMENTS = REPO / "experiments"
DIGIT_DATA = os.environ.get("DIGIT_DATA", str(REPO / "data" / "digit_data"))
ENCODING_DATA = os.environ.get("ENCODING_DATA", str(REPO / "data" / "encoding_data"))
SPT_DATA = os.environ.get("SPT_DATA", str(REPO / "data" / "spt_data"))
XXZ_DATA = os.environ.get("XXZ_DATA", str(REPO / "data" / "xxz_data" / "xxz_N8_grid50.npz"))
INIT_FILE = os.environ.get("INIT_FILE", str(REPO / "data" / "initial_params_julia110.txt"))

SEEDS_10 = [42, 123, 456, 789, 1024, 2024, 2025, 2718, 3141, 5678]
SEEDS_20 = SEEDS_10 + [1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008, 9009, 10010]
ALPHAS_5 = ["0.2", "1.0", "3.0", "6.0", "Inf"]
ALPHAS_10 = ["0.2", "0.4", "0.6", "0.8", "1.0", "1.5", "2.0", "3.0", "6.0", "Inf"]
ALPHAS_XXZ = ["0.2", "1.0", "3.0", "6.0"]
FIELDS = {"1.0": "h1", "3.0": "h3"}
FIELDS_8 = ["0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "4.0", "5.0"]
PAIRS = [(3, 8), (2, 7), (4, 9), (0, 1), (5, 8), (0, 6)]
PLACEMENTS = ["sequential_readout", "centerout", "centerout_mirror"]   # folders of data/prepare_data.py
TAU_04 = repr(0.4 * math.pi)


def task(**settings):
    """One task line, e.g. task(SEED=42, ELL=2) -> 'SEED=42;ELL=2'."""
    return ";".join(f"{key}={value}" for key, value in settings.items())


def spt_task(n, ell, alpha, seed, root):
    """One cluster-SPT run at h = 1.0, tau = 0.4 pi, readout qubit 1."""
    return task(N_QUBITS=n, ELL=ell, H_FIELD="1.0", ALPHA=alpha, TAU_PER_LAYER=TAU_04, SEED=seed,
                POS_MEASURE=1, DATA_DIR=SPT_DATA, INIT_FILE=INIT_FILE, OUT_DIR=root / "spt")


def phases(root):
    # Fig. 2: tau = k pi / 5 for k = 1..8 (0.2 pi to 1.6 pi); ell = 2, 4, 6, 8, 10 in fig2_grid, ell = 1, 3 in fig2_columns.
    yield "fig2_grid", "tau_ell_grid.jl", [
        task(N_QUBITS=8, H_FIELD=h, TAU_IDX=t, ELL_IDX=e, SEED=s, POS_MEASURE=1, DATA_DIR=DIGIT_DATA,
             OUT_DIR=root / "fig2_grid" / tag / f"seed_{s}")
        for h, tag in FIELDS.items() for e in range(1, 6) for t in range(1, 9) for s in SEEDS_10
    ]
    yield "fig2_columns", "resource_scan.jl", [
        task(N_QUBITS=8, ELL=ell, H_FIELD=h, ALPHA="1.0", TAU_PER_LAYER=repr(k / 5 * math.pi), TAU_IDX=f"t{k}_l{ell}",
             SEED=s, POS_MEASURE=1, DATA_DIR=DIGIT_DATA, OUT_DIR=root / "fig2_columns" / tag)
        for h, tag in FIELDS.items() for ell in (1, 3) for k in range(1, 9) for s in SEEDS_10
    ]
    yield "fig3_tau", "resource_scan.jl", [
        task(N_QUBITS=8, ELL=2, H_FIELD=h, ALPHA=a, TAU_PER_LAYER=repr(k / 10 * math.pi), TAU_IDX=f"{k:02d}",
             SEED=s, POS_MEASURE=1, DATA_DIR=DIGIT_DATA, OUT_DIR=root / "fig3_tau" / tag)
        for h, tag in FIELDS.items() for a in ALPHAS_5 for k in range(1, 11) for s in SEEDS_20
    ]
    yield "fig3_layers", "resource_scan.jl", [
        task(N_QUBITS=8, ELL=ell, H_FIELD=h, ALPHA=a, TAU_PER_LAYER=TAU_04, TAU_IDX=f"l{ell}",
             SEED=s, POS_MEASURE=1, DATA_DIR=DIGIT_DATA, OUT_DIR=root / "fig3_layers" / tag)
        for h, tag in FIELDS.items() for a in ALPHAS_5 for ell in range(1, 9) for s in SEEDS_20
    ]
    yield "fig4_sizes", "size_scan.jl", [
        task(SIT="sitA", N_QUBITS=n, ELL=ell, ALPHA=a, PAIR_A=pa, PAIR_B=pb, TAU_PER_LAYER=TAU_04,
             SEED=s, POS_MEASURE=1, DATA_DIR=DIGIT_DATA, OUT_DIR=root / "fig4_sizes")
        for ell in (1, 2, 4) for n in range(3, 11) for a in ALPHAS_5 for pa, pb in PAIRS for s in SEEDS_20
    ]
    # Fig. 5 / Fig. 13: readout on the end site 1 or on the central site N/2 + 1; N = 8 at ell = 1, 4 and h = 1.0, 3.0
    # (the figures draw h = 1.0; the h = 3.0 runs support the remark on h = 3.0 in Sec. 3.4).
    encoding_runs = [(8, ell, h) for ell in (1, 4) for h in FIELDS]
    yield "fig5_encodings", "encodings.jl", [
        task(ENC_NAME=enc, Q_MEAS=q, N_QUBITS=n, ELL=ell, H_FIELD=h, ALPHA=a, TAU_PER_LAYER=TAU_04,
             SEED=s, ENCODING_ROOT=ENCODING_DATA, OUT_DIR=root / "fig5_encodings")
        for n, ell, h in encoding_runs for enc in PLACEMENTS for q in (1, n // 2 + 1) for a in ALPHAS_5 for s in SEEDS_10
    ]
    # Fig. 6: cluster-SPT classification, N = 6, 8, 10 at ell = 1, 2.
    yield "fig6_spt", "spt_classification.jl", [
        spt_task(n, ell, a, s, root) for n in (6, 8, 10) for ell in (1, 2) for a in ALPHAS_5 for s in SEEDS_10
    ]
    # Fig. 7 and Fig. 16: XXZ circuits at N = 8, ell = 2 trained on each of the four anchors.
    yield "fig7_xxz", "xxz_anchors.jl", [
        task(ANCHOR=k, ELL=2, H_FIELD="1.0", ALPHA=a, TAU_PER_LAYER=TAU_04, SEED=s, DATA_FILE=XXZ_DATA,
             INIT_FILE=INIT_FILE, OUT_DIR=root / "xxz")
        for a in ALPHAS_XXZ for k in range(4) for s in SEEDS_10
    ]
    yield "fig9_h_alpha", "h_alpha_plane.jl", [
        task(N_QUBITS=8, ELL=ell, H_FIELD=h, ALPHA=a, TAU_PER_LAYER=TAU_04,
             SEED=s, POS_MEASURE=1, DATA_DIR=DIGIT_DATA, OUT_DIR=root / "fig9_h_alpha" / f"ell{ell}")
        for ell in (1, 4) for h in FIELDS_8 for a in ALPHAS_10 for s in SEEDS_10
    ]
    # Fig. 14: N = 8, alpha = 0.2 at ell = 4, 8 (ell = 1, 2 come from fig6_spt).
    yield "fig14_spt", "spt_classification.jl", [
        spt_task(8, ell, "0.2", s, root) for ell in (4, 8) for s in SEEDS_10
    ]


def main(root, n_shards):
    queue = root / "_queue"
    queue.mkdir(parents=True, exist_ok=True)
    manifest = []
    for phase, script, tasks in phases(root):
        for k in range(n_shards):
            (queue / f"{phase}.shard{k}.txt").write_text("\n".join(tasks[k::n_shards]) + "\n")
        manifest.append(f"{phase}\t{EXPERIMENTS / script}")
        print(f"{phase:16s} {len(tasks):6d} tasks")
    (queue / "manifest.tsv").write_text("\n".join(manifest) + "\n")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]).resolve(), int(sys.argv[2]) if len(sys.argv) > 2 else 5)
