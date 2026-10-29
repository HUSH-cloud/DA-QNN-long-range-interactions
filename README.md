# DA-QNN code

Simulation code, raw results and plotting scripts for every figure of "Digital-Analog Quantum Neural Network with
Long-Range Interactions": the architecture schematic (Fig. 1), MNIST classification and thermalization (Figs. 2-5,
8-13), cluster-SPT phase classification (Figs. 6, 14) and unsupervised XXZ phase detection (Figs. 7, 15, 16).

```
src/daqnn.jl                  Hamiltonian, exact analog block, circuit, data and initial-parameter loading, training, test accuracy
experiments/
  tau_ell_grid.jl             Fig. 2 main grid
  resource_scan.jl            Fig. 2 columns ell = 1, 3, Fig. 3, Fig. 11
  size_scan.jl                Fig. 4, Fig. 8, Fig. 12
  encodings.jl                Fig. 5, Fig. 13
  spt_classification.jl       Fig. 6, Fig. 14
  xxz_anchors.jl              Fig. 7, Fig. 16
  h_alpha_plane.jl            Fig. 10
  thermalization.py           Fig. 9 data (exact diagonalization)
data/                         inputs of every run (see "Data") and the scripts that write them:
  prepare_data.py             digit_data/, encoding_data/
  prepare_spt_data.py         spt_data/
  prepare_xxz_data.py         xxz_data/
  initial_params_julia110.jl  initial_params_julia110.txt
queue/
  make_tasks.py               task list of every Julia run in results.tar.gz
  run_queue.sh, worker.jl     local parallel runner
tests/run_checks.jl           checks of the conventions and of the quantum-state data
figures/
  style.py, paths.py          shared figure style and input/output locations
  xxz_common.py               XXZ loading, order parameters and drawing helpers of Figs. 7, 15, 16
  figNN_*.py                  one script per paper figure
  assets/                     the MNIST digit drawn in Fig. 1 (image 7 of mnist_784, counted from 0)
results.tar.gz                every result behind the figures (see "Data")
results.sha256, data.sha256   SHA-256 of every file in results.tar.gz and of every data file in data/
Project.toml, Manifest.toml   pinned Julia environment
requirements.txt              pinned Python packages
LICENSE                       MIT for the code, CC BY 4.0 for the data and results
```

## Setup

- Julia 1.12: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
- Python 3.13: `pip install -r requirements.txt`
- The figures embed Times New Roman; without it matplotlib uses Times or STIX and the glyphs differ from the paper.
- Julia 1.10 (juliaup channel `+1.10`) only for `data/initial_params_julia110.jl` (see "Conventions"), which needs only
  the Random standard library and runs outside the project environment.

## Data

- `data/digit_data/{a}_{b}/`: angle-encoded MNIST pairs, `x_{train,test}_n{N}.npy` (2N, samples) for N = 3..10 and
  one-hot `y_{train,test}.npy` (samples, 2) for six digit pairs, 1000 training and 200 test images per pair, from
  OpenML `mnist_784`.
- `data/encoding_data/{placement}/`: pair 3 vs 8 at N = 8 for the PCA-to-site placements `sequential_readout`,
  `centerout`, `centerout_mirror`.
- `data/spt_data/cluster_spt_N{6,8,10}.npz`: `train_states` (50, 2^N), `train_params` (50, 2) as (h1/J, h2/J),
  `train_labels` (50,) with 1 = SPT and 0 = non-SPT, `test_states` (441, 2^N), `test_params` (441, 2), `test_labels`
  (441,), `test_grid_params` (441, 2) (the 21 x 21 grid before the boundary shift, h1 outer), `anti_boundary` and
  `para_boundary` (4,) (cubic coefficients in `numpy.polyval` order). The training set has 25 SPT and 25 non-SPT
  states; points, labels and boundaries are the same for every N. Each state is the normalized projection of |0...0> onto the ground space of Eq. (3) (levels within 1e-10 of the
  lowest), up to a global phase.
- `data/xxz_data/xxz_N8_grid50.npz`: `j3_ratios` (50,), `lambdas` (50,), `test_params` (2500, 2) as (J3/J6, lambda)
  with row i_lambda * 50 + i_ratio, `test_states` (2500, 256), `anchor_params` (4, 2) as (J3/J6, lambda) and
  `anchor_states` (4, 256) of the four training points zAFM, qzAFM, XY-QLRO, VBS (`anchor` in keys, file names and
  the `ANCHOR` variable denotes the training point of a phase). Each state is an eigenvector of Eq. (4), the lowest
  except at the zAFM point (see "Conventions"), up to a global phase.
- `data/initial_params_julia110.txt`: initial parameters of the SPT and XXZ runs, one header line, then lines
  `seed n_params v1 ... vn` for the ten seeds and n_params in {36, 48, 54, 60, 72, 90, 120, 216}.

`results.tar.gz` unpacks into `results/`, one file per (configuration, seed) of every run, plus the thermalization
curves:

| Folder in `results/` | Files | Content |
|---|---|---|
| fig2_grid/{h1,h3}/seed_* | 800 | MNIST test accuracy, shape (1,) |
| fig2_columns/{h1,h3} | 320 | MNIST test accuracy |
| fig3_tau/{h1,h3} | 2,000 | MNIST test accuracy |
| fig3_layers/{h1,h3} | 1,600 | MNIST test accuracy |
| fig4_sizes | 14,400 | MNIST test accuracy |
| fig5_encodings | 1,200 | MNIST test accuracy at h = 1.0 (Figs. 5, 13) and h = 3.0 (the remark in Sec. 3.4) |
| fig9_h_alpha/{ell1,ell4} | 1,600 | MNIST test accuracy |
| spt | 320 | `test_prob_spt` (441,), `acc_test` (1,), `acc_train` (1,), `loss_history` (500,), `final_params` |
| xxz | 160 | `test_loss` (2500,), `loss_history` (51,), `final_params` (72,) |
| thermalization | 13 | `Mz_avg_a{alpha}_h{h}.npy` (time-averaged magnetization) for 6 alpha at h = 1.0 and 3.0, shape (4000,), and the time grid `tau_arr.npy` |

That is 22,413 files: 21,920 MNIST, 320 SPT and 160 XXZ runs, each the output of one task of `queue/make_tasks.py`,
and the 13 thermalization files of `experiments/thermalization.py`. Checksums:

```bash
tar xzf results.tar.gz
shasum -a 256 -c results.sha256
shasum -a 256 -c data.sha256
```

## Reproducing

All commands run from the repository root. Without arguments the prepare scripts overwrite the shipped inputs in
`data/`; the commands below write to `rerun/`. `prepare_data.py` downloads `mnist_784` from OpenML through scikit-learn
on first use.

```bash
OPENBLAS_NUM_THREADS=1 python3 data/prepare_data.py rerun/digit_data rerun/encoding_data
OPENBLAS_NUM_THREADS=1 python3 data/prepare_spt_data.py rerun/spt_data
OPENBLAS_NUM_THREADS=1 python3 data/prepare_xxz_data.py rerun/xxz_data
julia +1.10 --startup-file=no data/initial_params_julia110.jl rerun/initial_params_julia110.txt
```

`make_tasks.py` writes the tasks for a results root (optional second argument: parallel workers per phase, default 5).
They read the shipped `data/` unless `DIGIT_DATA`, `ENCODING_DATA`, `SPT_DATA` (folders), `XXZ_DATA` or `INIT_FILE`
(files) is set when `make_tasks.py` runs, with relative paths read from the folder where `run_queue.sh` starts.
`run_queue.sh` starts each worker with `julia`, or with `JULIA_BIN`, on one BLAS thread and skips finished tasks, so a
stopped queue resumes where it left off; with `results` as the root, the unpacked archive counts as finished.

```bash
python3 queue/make_tasks.py rerun/results     # task shards under rerun/results/_queue
bash queue/run_queue.sh rerun/results         # touch rerun/results/_queue/STOP to stop before the next phase
python3 experiments/thermalization.py rerun/results/thermalization
```

To use the regenerated data instead, in a separate results root:

```bash
DIGIT_DATA=rerun/digit_data ENCODING_DATA=rerun/encoding_data SPT_DATA=rerun/spt_data \
XXZ_DATA=rerun/xxz_data/xxz_N8_grid50.npz INIT_FILE=rerun/initial_params_julia110.txt \
python3 queue/make_tasks.py rerun/results_regen
bash queue/run_queue.sh rerun/results_regen
```

A single configuration runs without the queue; the examples below reproduce `results/fig3_tau/h1/acc_a1.0_t04_s42.npy`,
`results/spt/N6_l1_a0.2_s42.npz` and `results/xxz/a1.0_anchor3_s42.npz`. Each example runs in under a minute,
including the Julia start-up.

```bash
N_QUBITS=8 ELL=2 H_FIELD=1.0 ALPHA=1.0 TAU_PER_LAYER=1.2566370614359172 TAU_IDX=04 SEED=42 POS_MEASURE=1 \
DATA_DIR=data/digit_data OUT_DIR=rerun/fig3_tau julia --project=. experiments/resource_scan.jl

N_QUBITS=6 ELL=1 H_FIELD=1.0 ALPHA=0.2 TAU_PER_LAYER=1.2566370614359172 SEED=42 POS_MEASURE=1 \
DATA_DIR=data/spt_data INIT_FILE=data/initial_params_julia110.txt OUT_DIR=rerun/spt \
julia --project=. experiments/spt_classification.jl

ANCHOR=3 ELL=2 H_FIELD=1.0 ALPHA=1.0 TAU_PER_LAYER=1.2566370614359172 SEED=42 DATA_FILE=data/xxz_data/xxz_N8_grid50.npz \
INIT_FILE=data/initial_params_julia110.txt OUT_DIR=rerun/xxz julia --project=. experiments/xxz_anchors.jl
```

`cmp` prints nothing for a byte-identical MNIST output; the Python lines print the largest difference per key, which
in the SPT example exceeds rounding only for the last-layer entries of `final_params` (see "Conventions"):

```bash
cmp rerun/fig3_tau/acc_a1.0_t04_s42.npy results/fig3_tau/h1/acc_a1.0_t04_s42.npy
compare='import numpy as np, sys; a, b = np.load(sys.argv[1]), np.load(sys.argv[2]); print({k: float(abs(a[k] - b[k]).max()) for k in b.files})'
python3 -c "$compare" rerun/spt/N6_l1_a0.2_s42.npz results/spt/N6_l1_a0.2_s42.npz
python3 -c "$compare" rerun/xxz/a1.0_anchor3_s42.npz results/xxz/a1.0_anchor3_s42.npz
```

## Drawing the figures

```bash
cd figures
python3 fig03_threshold.py        # and likewise for the other figNN_*.py
```

The scripts read the unpacked `results/` and `data/` and write PDFs to `figures/out/`. `DAQNN_RESULTS`, `DAQNN_DATA`,
`DAQNN_THERMAL` and `DAQNN_FIGS` (absolute, or relative to the current folder) set other locations, e.g.
`DAQNN_RESULTS=../rerun/results python3 fig06_spt_size.py`.

| Figure | Script | Output | Result folders under `results/` | Data files |
|---|---|---|---|---|
| 1 | fig01_architecture.py | fig1_architecture.pdf | none | figures/assets/mnist_3_01.png |
| 2 | fig02_tl_heatmap.py | fig_tl_heatmap.pdf | fig2_grid/{h1,h3}/seed_*, fig2_columns/{h1,h3} | |
| 3 | fig03_threshold.py | fig_threshold_line.pdf | fig3_tau/{h1,h3}, fig3_layers/{h1,h3} | |
| 4 | fig04_size_scan.py | fig_alphaN_acc.pdf | fig4_sizes | |
| 5 | fig05_encodings.py | fig_encodings.pdf | fig5_encodings | |
| 6 | fig06_spt_size.py | fig_spt_size.pdf | spt | |
| 7 | fig07_xxz_combined.py | fig_xxz_combined.pdf | xxz | data/xxz_data/xxz_N8_grid50.npz |
| 8 | fig08_task_difficulty.py | fig_task_difficulty.pdf | fig4_sizes | |
| 9 | fig09_thermalization.py | fig_thermalization.pdf | thermalization | |
| 10 | fig10_h_alpha.py | fig_h_alpha.pdf | fig9_h_alpha/{ell1,ell4} | |
| 11 | fig11_resource_curves.py | fig_resource_combined.pdf | fig3_tau/h1, fig3_layers/h1 | |
| 12 | fig12_size_scan_ell4.py | fig_alphaN_acc_ell4.pdf | fig4_sizes | |
| 13 | fig13_encodings_ell4.py | fig_encodings_ell4.pdf | fig5_encodings | |
| 14 | fig14_spt_probability.py | fig_haldane_prob_a0p2.pdf | spt (N = 8, alpha = 0.2) | data/spt_data/cluster_spt_N8.npz |
| 15 | fig15_xxz_order.py | fig_xxz_order_boundaries.pdf | none | data/xxz_data/xxz_N8_grid50.npz |
| 16 | fig16_xxz_pairdiff.py | fig_xxz_pairdiff.pdf | xxz | data/xxz_data/xxz_N8_grid50.npz |

## Conventions

- `TAU_PER_LAYER` is the dimensionless time tau of one analog block; the code evolves the Hamiltonian with its field
  term scaled by the Kac factor for t = tau / kac_norm, which equals evolving Eq. (1) for tau.
- Sites are counted from the readout site (`POS_MEASURE=1` reads out site 1). Rows j (theta) and N + j (phi) of a data
  file are loaded onto site N + 1 - j (`product_state` builds the register with `kron`, Yao numbers qubits from the
  least significant bit); `data/prepare_data.py` specifies placements in sites and converts them in `encoded_rows`.
  `centerout` (main scheme) places PC1 on site floor(N/2) + 1 (for even N the central site farther from the readout),
  `sequential_readout` on the readout site.
- Column 1 of `y_*.npy` is the first digit of the pair, assigned to outcome |0> on the readout site.
- Cluster SPT: Eq. (3), open chain, J = 1, chain site i on Yao qubit i (bit i - 1 of the state index), readout qubit 1.
  Label 0 (non-SPT) is outcome |0> and label 1 (SPT) outcome |1>, so the SPT probability is 1 - p0 (SPT if p0 < 0.5).
  Training: binary cross-entropy, AdaGrad (learning rate 0.1), 500 iterations, h = 1.0, tau = 0.4 pi, the whole
  training set (50 states) in every iteration. A point is SPT (non-SPT) when h2/J lies between the two boundary
  fits (beyond either fit) by more than 0.01; the five grid points within that margin are shifted in h2 to 0.010001
  from the boundary, and Fig. 14 draws them on their grid cells (`test_grid_params`).
- XXZ: Eq. (4), N = 8, periodic boundaries, J6 = 1. Each circuit (ell = 2, h = 1.0, tau = 0.4 pi) is trained on the
  state of one training point with AdaGrad (learning rate 0.1) for 50 steps on L = (1/N) sum_i P_i(0), the probability of reading 0
  averaged over the qubits; `test_loss` is L for every grid state.
- zAFM training point: at (J3/J6, lambda) = (0.01, 0.2184) the two lowest levels form a quasi-degenerate Neel doublet split by
  2.7e-7; the stored state is its odd member under spin flip and translation, eigenvector index 1 counted from 0.
- Initial parameters: `randn(MersenneTwister(seed), n_params)`. The MNIST runs draw it in `train_circuit!` under Julia
  1.12; the SPT and XXZ runs read it as drawn by Julia 1.10 (a different stream) from `data/initial_params_julia110.txt`
  through `INIT_FILE`. `spt_classification.jl` draws the per-iteration order of the training states with
  `MersenneTwister(seed)` under Julia 1.12, which affects only rounding.
- SPT runs: the last-layer rotations on qubits other than the readout commute with the readout projector, so they
  affect no output and have zero exact gradient. AdaGrad rescales their rounding-level gradient and moves them from
  their initial values by less than 5e-7, so reruns are compared by probabilities and losses rather than by these
  entries of `final_params`.

## Checks

`julia --project=. tests/run_checks.jl [pair folder, default data/digit_data/3_8]` asserts the conventions above on the
shipped data (sites, one-hot balanced labels, the SPT label margin, H / kac_norm against Eq. (1) and the Kac-time
equivalence), unitarity of the analog block inside a chain, reflection symmetry, the nearest-neighbor limit, one SPT and
one XXZ state as ground states of Eqs. (3) and (4), and the zAFM doublet (splitting below 1e-6, stored state equal to
eigenvector index 1 up to a phase). Run it after any change to `src/`, the prepare scripts or the data files.

## Verification

Measured with Julia 1.12.5 (Julia 1.10.12 for the parameter table) and the Python packages of `requirements.txt`:

- `shasum -a 256 -c` passes for all 22,413 lines of `results.sha256` and all 125 lines of `data.sha256`.
- `make_tasks.py` writes 22,400 tasks whose outputs are exactly the files of `results/` outside
  `results/thermalization/`.
- MNIST: the MNIST example and eleven `queue/worker.jl` tasks covering all five MNIST scripts reproduce the stored
  accuracies bit for bit.
- SPT: all 320 tasks rerun give identical predictions and accuracies, and `test_prob_spt` within 2.5e-5.
- SPT example: identical predictions and accuracies, `test_prob_spt` to 1.2e-14, `loss_history` to 2.3e-14,
  `final_params` to 7.2e-13 except the last-layer entries on qubits other than the readout (to 2.2e-7).
- XXZ: all 160 tasks rerun give `test_loss` to 6.7e-14, `loss_history` to 8.3e-14, `final_params` to 8.2e-12; the
  example agrees to 1.2e-15, 7.8e-16 and 1.4e-14.
- Prepare scripts, run into a separate folder: `prepare_data.py` labels exact, angles to 4.3e-13; `prepare_spt_data.py`
  labels exact, points and boundaries to 2e-15, states to 6e-11 up to a global phase; `prepare_xxz_data.py` axes and
  points exact, states to 1 - |overlap| below 1e-13 up to a global phase; `initial_params_julia110.jl` byte for byte;
  `thermalization.py` curves to 3.1e-14.
- Against Eqs. (3) and (4) built independently from Pauli matrices, every stored state has |H psi - <H> psi| at most
  1.3e-10 and, except the zAFM anchor state, <H> - E0 below 2e-13; the zAFM anchor state has |overlap| with eigenvector
  index 1 equal to 1 within 2e-14.
- Every figure script redraws its figure of the paper from `results/` and `data/`; `tests/run_checks.jl` passes.

## License

Code: MIT. Data and results: CC BY 4.0. The MNIST-derived files keep the terms of the MNIST database. See `LICENSE`.

## Citation

Zongyuan Ge, Daili Li, Zewen Zhang, "Digital-Analog Quantum Neural Network with Long-Range Interactions".
Code and data: https://github.com/HUSH-cloud/DA-QNN-long-range-interactions
