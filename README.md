# FedGen with AM-DAE Imputation

A federated learning research codebase that combines **FedGen-style data-free
knowledge distillation** with the **Adaptive-Learned Median-Filled Deep
Autoencoder (AM-DAE)** for missing-data imputation, then evaluates it
side-by-side with FedAvg, FedProx, FedDistill / FedDistill-FL, and
FedEnsemble on Mnist, EMnist, UCI HAR, WISDM, and (optionally) PAMAP2.

> **Method note for citation.**
> Every server class in this repository (`FedAvg`, `FedProx`, `FedDistill`,
> `FedEnsemble`, `FedGen`) wraps the *same* AM-DAE imputation front-end
> ([Cui et al., IEEE T-Cyber, 2023]) before training begins. When
> `--missing_rate = 0.0`, the imputer is a verified no-op and the runs
> reproduce the original-paper baselines. See
> [`results/zero_missing_baseline/amdae_declaration.txt`](#goal-1-amdae-declaration--zero-missing-baseline)
> for the paste-into-paper paragraph.

---

## Table of contents

1. [Project at a glance](#1-project-at-a-glance)
2. [Repository layout](#2-repository-layout)
3. [Installation](#3-installation)
4. [Datasets and how to prepare them](#4-datasets-and-how-to-prepare-them)
5. [Models, algorithms, and configs](#5-models-algorithms-and-configs)
6. [Running experiments](#6-running-experiments)
7. [The AM-DAE imputation pipeline](#7-the-am-dae-imputation-pipeline)
8. [Paper-pipeline scripts (Goal 1 / 2 / 3)](#8-paper-pipeline-scripts-goal-1--2--3)
9. [One-shot orchestrator (`run_paper_pipeline.py`)](#9-one-shot-orchestrator-run_paper_pipelinepy)
10. [Tables, plots, and metrics tooling](#10-tables-plots-and-metrics-tooling)
11. [Option A paper sweep (run this for the paper)](#11-option-a-paper-sweep-run-this-for-the-paper)
12. [Outputs you will get](#12-outputs-you-will-get)
13. [Troubleshooting](#13-troubleshooting)
14. [Citing this work](#14-citing-this-work)

---

## 1. Project at a glance

**Goal.** Quantify how robust five mainstream federated-learning algorithms
are when client features have missing values, and demonstrate that a shared
AM-DAE imputation front-end closes the gap on heterogeneous, non-IID data.

**Pipeline.** For each `(dataset, alpha, missing_rate, algorithm)` cell:

```
read_data(dataset)                       # JSON/.pt per-user dictionaries
  -> apply_amdae_imputation(missing_rate) # AM-DAE / Mean / Median / Zero
                                           #   auto-evaluated; best chosen
  -> build per-user DataLoaders
  -> server.train()                       # FedAvg / FedProx / FedDistill /
                                           #   FedEnsemble / FedGen
  -> per-round HDF5 logging               # acc, loss, y_true, y_pred, y_prob
```

**Datasets supported (out of the box).** Mnist, EMnist (letters), UCI HAR,
CelebA. PAMAP2 is supported via a data-prep helper plus a one-time config
patch hint (see [Goal 2](#goal-2-real-world-dataset-uci-har--pamap2)).

**Algorithms.** FedAvg, FedProx (`λ‖w − w*‖²` proximal term), FedDistill /
FedDistill-FL (label-wise logit averaging), FedEnsemble (test-time
averaging), and FedGen (server trains a conditional generator that
synthesises latent representations to regularise client training).

---

## 2. Repository layout

```
.
├─ main.py                          # argparse entry point (one run, one algorithm)
├─ run_paper_pipeline.py            # one-shot end-to-end paper-results driver
├─ produce_paper_results.py         # thin "rebuild every paper artifact" wrapper
├─ goal1_zero_missing_baseline.py   # 0% missing baseline + AMDAE declaration
├─ goal2_real_dataset_experiment.py # UCI HAR + PAMAP2 wrapper
├─ goal3_metrics_table.py           # F1 / Precision / Recall paper tables
├─ plot_per_class_f1_heatmap.py     # paper Figs 5-8 (per-class F1 heatmap)
├─ plot_experiment_results.py       # paper Fig 13 (acc + loss side-by-side)
├─ run_all.sh, run_experiments.sh,  # legacy shell drivers
│  batch_run_jobs.sh
│
├─ FLAlgorithms/
│  ├─ servers/  serverbase, serveravg, serverFedProx, serverFedDistill,
│  │            serverpFedEnsemble, serverpFedGen
│  ├─ users/    userbase, useravg, userFedProx, userFedDistill,
│  │            userpFedEnsemble, userpFedGen, userGen
│  ├─ optimizers/  fedoptimizer.py    (pFedIBOptimizer, FedProxOptimizer)
│  └─ trainmodel/  models.py (CNN), generator.py (cond. generator + diversity loss)
│
├─ utils/
│  ├─ data_imputation.py     # AM-DAE + Mean/Median/Zero imputers, metrics
│  ├─ model_utils.py         # data IO, DataLoader builders, model factories
│  ├─ model_config.py        # CONFIGS_, GENERATORCONFIGS, RUNCONFIGS
│  ├─ metrics_utils.py       # per-round y_true/y_pred dump
│  └─ plot_utils.py          # accuracy curves
│
├─ data/
│  ├─ Mnist/    generate_niid_dirichlet.py  (auto-downloads via torchvision)
│  ├─ EMnist/   generate_niid_dirichlet.py  (auto-downloads via torchvision)
│  ├─ UCI HAR/  generate_niid_dirichlet.py  (needs manual raw download)
│  └─ CelebA/   generate_niid_agg.py        (LEAF benchmark)
│
├─ evaluate_metrics.py            # per-dataset F1/ROC/CM analysis (legacy)
├─ f1score_all.py                 # F1-vs-round across all algorithms
├─ confusion_matrix_all.py        # last-round confusion matrices per algorithm
├─ plot_imputation_comparisons.py # accuracy across missing-rate sweeps
├─ plot_experiment_results.py     # paper-style accuracy curves + summary
├─ main_plot.py + utils/plot_utils.py  # original FedGen plotting
│
├─ requirements.txt
├─ FenGen-overview.png
├─ figs/         legacy figure store
├─ readme_figs/  paper visualisations
└─ results/      created at runtime (see Section 11)
```

---

## 3. Installation

### 3.1 Requirements

`requirements.txt` declares the minimum dependency set:

```
numpy
scipy
Pillow
torch
torchvision
matplotlib
tqdm
h5py
scikit-learn
seaborn
```

For Goal 3's table generator, also install `pandas` and `tabulate`:

```bash
pip install pandas tabulate
```

### 3.2 Recommended virtual env

```bash
# Linux / macOS
python -m venv fedgen_env
source fedgen_env/bin/activate
pip install -r requirements.txt pandas tabulate
```

```powershell
# Windows
py -3 -m venv fedgen_env
.\fedgen_env\Scripts\Activate.ps1
py -m pip install -r requirements.txt pandas tabulate
```

> A pre-existing `fedamd/` Linux venv ships with the repo for reference; on
> Windows you should build a new venv as above.

---

## 4. Datasets and how to prepare them

Every dataset uses a Dirichlet `α`-controlled non-IID split: smaller `α`
means more skewed per-client label distribution.

### 4.1 Mnist (auto-download via torchvision)

```bash
cd data/Mnist
python generate_niid_dirichlet.py --n_class 10 --sampling_ratio 0.5 \
       --alpha 0.1 --n_user 20
# writes data/Mnist/u20c10-alpha0.1-ratio0.5/{train,test}/*.pt
```

| Flag | Default | Meaning |
|---|---|---|
| `--alpha` | 0.5 | Dirichlet concentration. `0.1` is highly non-IID. |
| `--sampling_ratio` | 0.05 | Fraction of MNIST training samples to use overall. |
| `--n_user` | 20 | Number of clients (must stay 20; the FL pipeline hardcodes that prefix). |
| `--n_class` | 10 | Classes to retain (10 = full MNIST). |
| `--unknown_test` | 0 | If 1, allow unseen labels in each user's test set. |

### 4.2 EMnist (auto-download via torchvision)

```bash
cd data/EMnist
python generate_niid_dirichlet.py --sampling_ratio 0.1 --alpha 0.1 --n_user 20
# writes data/EMnist/u20-letters-alpha0.1-ratio0.1/{train,test}/*.pt
```

EMnist defaults to the `letters` split (26 classes); the `balanced` split
(47 classes) is also supported via `--split balanced`.

### 4.3 UCI HAR (manual raw download)

1. Download the **Human Activity Recognition Using Smartphones** dataset
   (UCI ML repository or Kaggle), unzip so the structure is:
   ```
   data/UCI HAR/data/UCI HAR Dataset/{train,test}/X_*.txt, y_*.txt
   ```
2. Generate the per-user split:
   ```bash
   cd "data/UCI HAR"
   python generate_niid_dirichlet.py --alpha 0.1 --sampling_ratio 0.5 --n_user 20
   # writes data/UCI HAR/u20-alpha0.1-ratio0.5/{train,test}/*.pt
   ```

The generator pads the 561 raw HAR features to 576 and reshapes them into
a `(N, 1, 24, 24)` "image" so the CNN in `FLAlgorithms/trainmodel/models.py`
can be reused unchanged.

### 4.4 CelebA (LEAF benchmark)

LEAF-style federated CelebA: see `data/CelebA/README.md` and
`generate_niid_agg.py`. Uses a hardcoded `user{N}-agg{M}` directory layout.

### 4.5 PAMAP2 (research extension)

Use `goal2_real_dataset_experiment.py --dataset_kind pamap2 --prepare_only`
to download and convert the raw archive into the same `{users, user_data}`
schema used elsewhere in the codebase. Pads the 51 IMU channels to 64 and
reshapes them to `(N, 1, 8, 8)`. Wiring PAMAP2 into the model layer is a
one-time edit to `utils/model_utils.py` + `utils/model_config.py` — the
script prints the exact patch as it finishes.

### 4.6 Dataset string format used by `main.py`

| Dataset | Token format | Example |
|---|---|---|
| Mnist | `Mnist-alpha<a>-ratio<r>` | `Mnist-alpha0.1-ratio0.5` |
| EMnist | `EMnist-alpha<a>-ratio<r>` | `EMnist-alpha0.1-ratio0.1` |
| UCI HAR | `'UCI HAR-alpha<a>-ratio<r>'` | `'UCI HAR-alpha0.1-ratio0.5'` |
| CelebA | `celeb-user<u>-agg<a>` | `celeb-user25-agg10` |
| PAMAP2 | `PAMAP2-alpha<a>-ratio<r>` | `PAMAP2-alpha0.1-ratio0.5` (after wiring) |

---

## 5. Models, algorithms, and configs

### 5.1 Core CNN (`FLAlgorithms/trainmodel/models.py :: Net`)

A configurable Conv → BatchNorm → ReLU → MaxPool → Flatten → FC → FC
classifier. Architecture is selected by the dataset key:

```python
CONFIGS_ = {
    'mnist'  : ([6, 16, 'F'], 1, 10, 784, 32),
    'emnist' : ([6, 16, 'F'], 1, 26, 784, 32),
    'ucihar' : ([6, 16, 'F'], 1,  6, 576, 32),
    'celeb'  : ([16,'M',32,'M',64,'M','F'], 3, 2, 64, 32),
    # cifar / cifar100-c25 / mnist_cnn{1,2}: extra entries ...
}
```

Format: `(layer_spec, in_channels, n_classes, conv_flat_dim, latent_dim)`.
`'M'` is `MaxPool2d(2,2)`, `'F'` is `Flatten`. Forward pass exposes
`start_layer_idx` so the FedGen generator can feed in latent activations
mid-network.

### 5.2 Conditional generator (`FLAlgorithms/trainmodel/generator.py`)

Used by FedGen only. Takes a label, samples Gaussian noise, concatenates
either a one-hot or a learned embedding of the label, and produces a
synthetic latent representation matching the chosen `latent_layer_idx`.
A `DiversityLoss` term encourages distinct outputs for the same label.
Sizes per dataset live in `GENERATORCONFIGS`.

### 5.3 Per-dataset run knobs

`RUNCONFIGS` (in `utils/model_config.py`) holds the ensemble-distillation
hyperparameters per dataset:

```python
RUNCONFIGS['mnist'] = {
    'ensemble_lr': 3e-4, 'ensemble_batch_size': 128, 'ensemble_epochs': 50,
    'num_pretrain_iters': 20,
    'ensemble_alpha': 1, 'ensemble_beta': 0, 'ensemble_eta': 1,
    'unique_labels': 10,
    'generative_alpha': 10, 'generative_beta': 10,
    'weight_decay': 1e-2,
}
```

These supersede the matching CLI flags for FedGen (`--ensemble_lr` is
read but `RUNCONFIGS` wins for the distillation phase).

### 5.4 Algorithms

| Algorithm | Key behaviour |
|---|---|
| `FedAvg` | Sample-weighted parameter averaging. Baseline. |
| `FedProx` | Local objective adds `λ/2 · ‖w − w*‖²`; controlled by `--lamda`. |
| `FedDistill` | Per-label averaged logits broadcast to clients; can pretrain (`--algorithm FedDistill-pretrain`) and/or share weights (`-FL` suffix). |
| `FedEnsemble` | Same training as FedAvg, but inference is the average of all client logits. |
| `FedGen` | Server trains a conditional generator from clients' label-aware logits; clients then use generated latents as auxiliary supervision (teacher loss + KL student loss + diversity loss). |

---

## 6. Running experiments

### 6.1 Single run via `main.py`

```bash
python main.py \
    --dataset Mnist-alpha0.1-ratio0.5 \
    --algorithm FedGen \
    --num_glob_iters 100 --local_epochs 20 \
    --num_users 10 --batch_size 64 --gen_batch_size 64 \
    --learning_rate 0.01 --personal_learning_rate 0.01 \
    --lamda 1 --beta 1.0 --K 1 \
    --embedding 0 --missing_rate 0.1 \
    --times 3 --device cuda \
    --result_path results/models
```

### 6.2 All `main.py` CLI flags

| Flag | Default | Purpose |
|---|---|---|
| `--dataset` | `Mnist-alpha0.1-ratio0.5` | Dataset token (see [4.6](#46-dataset-string-format-used-by-mainpy)). |
| `--model` | `cnn` | Backbone (informational; CNN spec is selected by dataset). |
| `--algorithm` | `pFedMe` | One of `FedAvg`, `FedProx`, `FedDistill`, `FedDistill-FL`, `FedDistill-pretrain`, `FedEnsemble`, `FedGen`, `FedGen-cnn{0,1,2,3}`. |
| `--train` | `1` | Set to 0 to only print parameter counts. |
| `--num_glob_iters` | `200` | Federated rounds. |
| `--local_epochs` | `20` | Local SGD epochs per round. |
| `--num_users` | `20` | **Active** clients per round (must be ≤ users in the data split). |
| `--batch_size` | `32` | Local mini-batch. |
| `--gen_batch_size` | `32` | Generated-sample batch (FedGen only). |
| `--learning_rate` | `0.01` | Local SGD/Adam lr. |
| `--personal_learning_rate` | `0.01` | Per-user fine-tune lr (pFedMe-style algorithms). |
| `--ensemble_lr` | `1e-4` | Server distillation lr. Overridden by `RUNCONFIGS` for FedGen. |
| `--beta` | `1.0` | Polyak/moving-average for personalised algorithms. |
| `--lamda` | `1` | FedProx proximal coefficient. |
| `--mix_lambda` | `0.1` | Mix coefficient for the FedMXI variant. |
| `--embedding` | `0` | If 1, the FedGen generator uses a learned label embedding instead of one-hot. |
| `--K` | `1` | Inner SGD steps per local epoch. |
| `--times` | `3` | Number of random seeds (separate `.h5` per seed). |
| `--device` | `cuda` | `cpu` or `cuda`. |
| `--result_path` | `results/models` | Where the per-run history HDF5 lands. |
| `--missing_rate` | `0.1` | Fraction of feature values to mask before AM-DAE imputation. `0.0` = no-op. |

### 6.3 Quick-start examples

```bash
# Single FedGen run on EMnist, 60 rounds, 1 seed
python main.py --dataset EMnist-alpha0.1-ratio0.1 --algorithm FedGen \
    --num_glob_iters 60 --times 1

# All five algorithms on Mnist, 200 rounds, 3 seeds
for a in FedAvg FedGen FedProx FedDistill-FL FedEnsemble; do
  python main.py --dataset Mnist-alpha0.1-ratio0.5 --algorithm "$a" \
      --num_glob_iters 200 --times 3 --batch_size 32 --learning_rate 0.01
done

# Use the legacy shell driver for one (alpha, dataset, missing) combo
./run_experiments.sh 0.1 EMnist 0.1
```

---

## 7. The AM-DAE imputation pipeline

`utils/data_imputation.py` implements:

- **`AMDAE`** — encoder/decoder MLP with configurable hidden-dim list,
  ReLU/Tanh/Sigmoid activations, and dropout; trained per call to
  `fit_impute()` for `max_epochs` epochs.
- **`MissingDataSimulator`** — three patterns (`random`, `fixed_intervals`,
  `continuous_periods`) that produce a corrupted matrix + binary mask.
- **`AMDAEImputer`** — uses the **adaptive loss**

  ```
  α_k(1 − M)·(x − x̂)² + β_k·M·(x − x̂)²
  α_k = 2(1 − 0.5·k/T),   β_k = k/T
  ```

  paired with the **median-update rule** `x_new = (x + x̂) / 2` applied
  only at masked positions.
- **`MeanImputer` / `MedianImputer` / `ZeroImputer`** — baselines.
- **`apply_amdae_imputation()`** — orchestrates the four imputers, computes
  RMSE, MAPE, KL-divergence, mean-difference, and an adaptive-loss score
  per imputer, normalises across methods, picks the lowest-score winner,
  and reconstructs the federated `(clients, groups, train, test, proxy)`
  tuple. Saves a comparison bar chart to
  `results/comprehensive_imputation_comparison.png`.

When `--missing_rate <= 0`, the function returns the original data
unchanged — verified short-circuit. This is what makes the 0%-missing
baseline reproducible.

---

## 8. Paper-pipeline scripts (Goal 1 / 2 / 3)

### Goal 1 — AMDAE declaration + zero-missing baseline

`goal1_zero_missing_baseline.py` does two things:

1. Emits a paste-into-paper paragraph documenting that all five FL
   algorithms in this repo share the AM-DAE imputer
   (`results/zero_missing_baseline/amdae_declaration.txt`).
2. Drives `main.py` for every algorithm with `--missing_rate 0.0` and
   produces a final summary
   (`results/zero_missing_baseline/<dataset>_zero_missing.{txt,csv}`).

```bash
# Print the methodology paragraph only (no training)
python goal1_zero_missing_baseline.py \
    --dataset EMnist-alpha0.1-ratio0.1 --declaration_only

# Reproduce all 5 baselines on EMnist alpha=0.1 (50 rounds, 1 seed)
python goal1_zero_missing_baseline.py \
    --dataset EMnist-alpha0.1-ratio0.1 \
    --num_glob_iters 50 --times 1
```

### Goal 2 — real-world dataset (UCI HAR / PAMAP2)

`goal2_real_dataset_experiment.py` takes care of (a) data preparation and
(b) running every selected FL algorithm on the chosen real-sensor dataset.

UCI HAR is fully wired end-to-end. PAMAP2 has a complete data-prep step
plus a printed wiring hint for `utils/model_utils.py` and
`utils/model_config.py`.

```bash
# UCI HAR, alpha=0.1, 10% missing, all 5 algorithms, 50 rounds
python goal2_real_dataset_experiment.py \
    --dataset_kind ucihar --alpha 0.1 --sampling_ratio 0.5 \
    --missing_rate 0.1 --num_glob_iters 50

# PAMAP2: only download + format (do not train)
python goal2_real_dataset_experiment.py --dataset_kind pamap2 --prepare_only
```

| Flag | Default | Notes |
|---|---|---|
| `--dataset_kind` | `ucihar` | `ucihar` or `pamap2`. |
| `--alpha` | `0.1` | Dirichlet concentration. |
| `--sampling_ratio` | `0.5` | Fraction of source training data to use. |
| `--n_user` | `20` | Total users in the per-dataset split (must remain 20). |
| `--missing_rate` | `0.1` | AM-DAE imputation knob. |
| `--algorithms` | five algos | Subset to run. |
| `--prepare_only` | off | Skip training; only download/format/split. |
| `--skip_generate` | off | Skip the per-user split if you already produced it. |

### Goal 3 — F1 / Precision / Recall paper tables

`goal3_metrics_table.py` walks one or more `results/metrics_mr<rate>/`
folders, extracts `(y_true, y_pred)` from the highest-numbered round per
`(algorithm, dataset, alpha, ratio)` cell, and produces:

- `results/tables/metrics_table_long.csv` — one row per cell, every
  metric (precision/recall/F1, both `macro` and `weighted`, `n_samples`).
- `results/tables/<dataset>_metrics_wide.csv` / `.md` / `.tex` — the
  paper-ready wide layout (rows = algorithm, columns = `Missing% × α ×
  Metric`).

```bash
python goal3_metrics_table.py \
    --input 0:results/metrics_mr0 \
    --input 10:results/metrics_mr10 \
    --input 20:results/metrics_mr20 \
    --avg both
```

| Flag | Default | Notes |
|---|---|---|
| `--input MR:DIR` | required | One per missing-rate; repeatable. |
| `--out_dir` | `results/tables` | Where to write the long + wide tables. |
| `--avg` | `both` | `macro`, `weighted`, or `both`. |
| `--zero_division` | `0` | sklearn `zero_division` kwarg. |

---

## 9. One-shot orchestrator (`run_paper_pipeline.py`)

This is the single command that produces every paper artifact end-to-end.

### 9.1 What it does (5 phases)

| Phase | What runs |
|---|---|
| **0** | `goal1` declaration → `results/zero_missing_baseline/amdae_declaration.txt` |
| **1** | Auto-runs the per-dataset Dirichlet generators if the per-user split is missing |
| **2** | Trains every `(dataset, alpha, missing_rate, algorithm, seed)` cell, writing `results/models_mr<R>/` and renaming `results/metrics/` → `results/metrics_mr<R>/` after each missing-rate batch (avoids the metrics-overwrite trap) |
| **3** | `goal3` over all `results/metrics_mr<R>/` dirs |
| **4** | Plots: `f1score_all.py` (F1-by-round per dataset), `confusion_matrix_all.py` (last-round CM per algorithm), `plot_experiment_results.py` (paper-style accuracy curves with mean ± std) |
| **5 (opt.)** | UCI HAR sweep via `goal2` when `--include_ucihar` is given |

### 9.2 Recommended workflows

**Quick sanity check (~30 min on CPU)** — one alpha, two missing rates,
5 rounds × 2 local-epochs, 5 algorithms × 2 datasets = 20 short trainings:

```bash
python run_paper_pipeline.py --quick -y
```

**Full paper run** — interactive confirmation prompt before kicking off:

```bash
python run_paper_pipeline.py --datasets Mnist EMnist \
    --alphas 0.1 1.0 10.0 \
    --missing_rates 0.0 0.1 0.2 \
    --num_glob_iters 100 --times 3 --device cuda
```

**Add UCI HAR** to the same sweep (raw archive must already be on disk):

```bash
python run_paper_pipeline.py --include_ucihar
```

**Re-run only the table + plot phases** on already-trained results:

```bash
python run_paper_pipeline.py --skip_train -y
```

**Plan only — print every command but execute none**:

```bash
python run_paper_pipeline.py --dry_run -y
```

### 9.2.1 The "one-button paper rebuild" wrapper (`produce_paper_results.py`)

If you just cloned a fresh copy of the repo on a new machine and want to
reproduce **everything the paper still needs in one go** — the 0% baseline
column, the alpha × missing-rate sweep, UCI HAR (MCAR + MAR + MNAR),
PAMAP2, the F1/Precision/Recall table, the per-class F1 heatmaps, and the
acc-vs-loss panels — run the thin wrapper:

```bash
python produce_paper_results.py
```

It calls `run_paper_pipeline.py` four times with the right flags
(MNIST/EMNIST sweep, then UCI HAR per-mechanism, then PAMAP2, then a final
goal3 sweep over every `results/metrics_mr*` directory it finds), and ends
with a checklist of where each artifact landed and which sentence in the
paper to update from it.

Useful flags:

```bash
python produce_paper_results.py --quick           # ~30-min sanity smoke
python produce_paper_results.py --dry_run -y      # print plan, run nothing
python produce_paper_results.py --skip_real       # MNIST/EMNIST only
python produce_paper_results.py --device cuda --times 3 --num_glob_iters 100
```

### 9.3 Resumability

- Every cell is checkpointed: the orchestrator computes the expected `.h5`
  path via `get_log_path` and skips trainings whose file already exists in
  the matching `results/models_mr<R>/`.
- Phase 3 and 4 only fire on missing-rate dirs that actually contain
  metrics files, so partial sweeps degrade gracefully.

### 9.4 Important flags

| Flag | Default | Purpose |
|---|---|---|
| `--datasets` | `Mnist EMnist` | Sweep dimension 1. |
| `--alphas` | `0.1 1.0 10.0` | Sweep dimension 2. |
| `--missing_rates` | `0.0 0.1 0.2` | Sweep dimension 3. |
| `--algorithms` | five algos | Sweep dimension 4. |
| `--num_glob_iters` | 100 | Federated rounds per cell. |
| `--local_epochs` | 20 | Local SGD epochs per round. |
| `--num_users` | 10 | Active users per round (must be ≤ 20). |
| `--batch_size` | 64 | Local + ensemble mini-batch. |
| `--gen_batch_size` | 64 | Synthetic sample batch (FedGen). |
| `--times` | 1 | Random seeds. Use ≥ 3 for paper. |
| `--device` | `cpu` | `cpu` or `cuda`. |
| `--include_ucihar` | off | Append UCI HAR to the sweep. |
| `--quick` | off | Tiny smoke run (5 rounds, 1 alpha, 2 missing). |
| `--skip_data_prep`, `--skip_train`, `--skip_table`, `--skip_plot` | off | Selective re-runs. |
| `--dry_run` | off | Print, don't execute. |
| `--yes / -y` | off | Skip confirmation prompt. |

---

## 10. Tables, plots, and metrics tooling

You can call any of the analysis scripts directly even without the
orchestrator.

### 10.1 F1 across all algorithms vs round

```bash
python f1score_all.py --dataset Mnist-alpha0.1-ratio0.5 \
       --rounds 100 --avg macro \
       --input-root results/metrics_mr10 \
       --output-root results/figures/mr10
```

### 10.2 Last-round confusion matrices per algorithm

```bash
python confusion_matrix_all.py --dataset Mnist-alpha0.1-ratio0.5 \
       --rounds 100 --normalize true \
       --input-root results/metrics_mr10 \
       --output-root results/figures/mr10
```

### 10.3 Per-dataset deep dive (F1-by-round + last-round ROC + last-round CM)

```bash
python evaluate_metrics.py --results_dir results/metrics_mr10 \
       --output_dir eval --normalize true
```

### 10.4 Paper-style accuracy curves with mean ± std + summary table

```bash
python plot_experiment_results.py \
    --dataset Mnist-alpha0.1-ratio0.5 \
    --algorithms FedAvg,FedGen,FedProx,FedDistill,FedEnsemble \
    --missing_rate 0.10 \
    --result_path results/models_mr10 \
    --num_glob_iters 100 --num_users 10 --batch_size 64 \
    --gen_batch_size 64 --local_epochs 20 --learning_rate 0.01 --times 3
```

### 10.5 Imputation-rate comparisons (across multiple sweeps)

`plot_imputation_comparisons.py` reads `imputation_comparisons/*.h5` (use
`embed00`, `embed01`, … as the missing-rate suffix in filenames) and
produces accuracy / loss / time line charts plus a summary CSV.

### 10.6 Goal 3 paper tables

See [Section 8](#goal-3--f1--precision--recall-paper-tables).

---

## 11. Option A paper sweep (run this for the paper)

Section 11 is the **single source of truth** for reproducing the paper's
headline numbers. It runs a clean, mechanism-free (MCAR-only) sweep across
EMNIST-letters, UCI HAR, and WISDM with the alpha and missing-rate grid
used in our previous paper, then emits the 7 paper tables and 4 dashboard
figures. If you only want to run one thing, run the commands below.

> **Jul 2026 revision:** the third dataset slot was **PAMAP2** in the
> original plan and was replaced with **WISDM v1.1** (much smaller: ~11 MB
> tarball vs ~700 MB, same 24x24 input shape, no model changes). PAMAP2's
> code paths are still wired up for opt-in reproduction (`--datasets PAMAP2`)
> but are not part of the default Option A grid.

### 11.1 Locked grid

| Axis | Values |
|------|--------|
| Datasets (run order) | `UCI HAR` -> `EMnist-letters` -> `WISDM` |
| Heterogeneity (alpha) | `0.1`, `1`, `10` |
| Missing rate | `0.0`, `0.10`, `0.20` (0% = no-missingness anchor) |
| Mechanism | MCAR (random) only |
| Algorithms | FedAvg, FedProx, FedDistill, FedEnsemble, FedGen |
| Imputer (main sweep) | AM-DAE, **forced** via `--force_imputer amdae` |
| Imputer (ablation) | FedGen x {AM-DAE, Mean, Median, Zero, no-imputation} |
| Headline cell (figures + ablation) | alpha=1, missing=10% |
| Communication rounds | EMNIST 200, UCI HAR 100, WISDM 100 |
| Seeds | Stage 1 = 1; Stage 2 = `--times 3` |

Total trainings (across both stages and ablation): ~135 (Stage 1) +
incremental (Stage 2 only adds the missing seeds) + 45 (ablation).
Wall-clock on a single GPU: ~35-50 GPU-h (WISDM is roughly 5-10x lighter
per cell than PAMAP2 was, so the revised sweep is meaningfully cheaper).

### 11.2 Auto-download

All three datasets self-bootstrap on first call -- you do not need to
download anything manually:

* **UCI HAR** (~60 MB) -- `run_optionA_sweep.py` itself fetches and
  unzips `https://archive.ics.uci.edu/ml/.../UCI HAR Dataset.zip` via
  Python stdlib (`urllib` + `zipfile`); no `wget` / `unzip` required, so
  it works on Linux, macOS, and Windows. Pass `--no_auto_download` to
  disable on no-internet boxes.
* **EMNIST-letters** (~562 MB) -- `data/EMnist/generate_niid_dirichlet.py`
  uses `torchvision.datasets.EMNIST(download=True)`.
* **WISDM v1.1** (~11 MB tarball, ~68 MB raw txt) --
  `data/WISDM/download_wisdm.py` streams the tarball from
  `http://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz`
  and extracts `WISDM_ar_v1.1_raw.txt` via Python stdlib. Also invoked
  automatically by the WISDM Dirichlet generator when the raw file is
  absent.
* *(Optional, opt-in)* **PAMAP2** (~700 MB zipped / ~3 GB unzipped) --
  `goal2_real_dataset_experiment.py --prepare_only` downloads + unzips
  via `urllib`. Only fetched when you explicitly pass `--datasets PAMAP2`.

All are idempotent (skip download if already present) and the splits are
reused across alphas where possible.

### 11.3 New scripts

* `run_optionA_sweep.py` -- master driver. Drives Dirichlet split
  generation, per-cell training (with per-seed resume-skip), and the
  imputer ablation phase.
* `paper_table_optionA.py` -- builds the 6 main paper tables (Accuracy
  and Macro-F1 on EMNIST / UCI HAR / WISDM) and the 1 imputer ablation
  table, in CSV + Markdown + LaTeX, with the row winner bolded.
* `paper_dashboard.py` -- composes the 3 per-dataset 2x3 dashboards
  (`<dataset>_dashboard.png`) and the 4x3 hero figure
  (`hero_figure.png`).

These three scripts read every cell from
`results/optionA/<dataset>/alpha<a>_miss<m>/<algo>/{models,metrics}/`
which is the namespacing the driver uses.

### 11.4 Recommended runbook

Designed to be safe to interrupt and resume at any time -- everything is
per-cell, per-seed skip-resumable.

```bash
# 0. (One-off, no GPU) Pre-flight: verify the driver builds the right
#    commands for the full grid before committing GPU time.
python run_optionA_sweep.py --dry_run --stage 1

# 1. Stage 1 -- single seed across the full 3 x 3 x 3 x 5 = 135-cell grid.
#    ~22-30 GPU-h on a single GPU. Run inside tmux / screen with logging.
tmux new -s optionA_stage1
python run_optionA_sweep.py --stage 1 --device cuda \
       2>&1 | tee logs/optionA_stage1.log

# 2. Triage Stage 1 results. Build the seed-1 tables and smell-test:
#    does FedGen win most rows? Did anything explode (NaN, < 50% accuracy
#    where it shouldn't be, training-loss divergence)?
python paper_table_optionA.py --metric all
grep -E "BEST PERFORMING METHOD|Selected method" logs/optionA_stage1.log \
     | sort -u

# 3. Stage 2 -- multi-seed (`--times 3` total). The driver re-runs only
#    the seeds that are missing for each cell, so seed 0 is NOT
#    retrained. ~40-60 GPU-h.
python run_optionA_sweep.py --stage 2 --device cuda --times 3 \
       2>&1 | tee logs/optionA_stage2.log

# 4. Imputer ablation -- FedGen x {amdae,mean,median,zero,none} at the
#    headline cell (alpha=1, miss=10%) for each dataset, --times 3.
#    ~6-8 GPU-h.
python run_optionA_sweep.py --ablation --device cuda --times 3 \
       2>&1 | tee logs/optionA_ablation.log

# 5. Final paper outputs (no GPU; ~1-2 minutes).
python paper_table_optionA.py --metric all --imputer_ablation
python paper_dashboard.py --headline-alpha 1 --headline-missing 0.10
```

**Subset / smoke-test variants**, useful while iterating or to do a
sanity run on a CPU box first:

```bash
# Just one dataset, one alpha, fewer rounds
python run_optionA_sweep.py --stage 1 --device cuda \
       --datasets EMnist-letters --alphas 0.1 \
       --num_glob_iters_emnist 5 --algorithms FedGen FedAvg

# Re-run only one algorithm across the full grid
python run_optionA_sweep.py --stage 1 --device cuda --algorithms FedGen

# Ablation only on UCI HAR (e.g. for a fast sanity pass)
python run_optionA_sweep.py --ablation --device cuda \
       --datasets "UCI HAR" --times 3
```

### 11.5 Output layout

```
results/optionA/
├─ <dataset>/                          # emnist | ucihar | wisdm
│  ├─ alpha0.1_miss0.0/<algo>/
│  │  ├─ models/<TOKEN>_<algo>_..._<seed>.h5
│  │  └─ metrics/seed_<s>/<TOKEN>/<algo>_<TOKEN>_round_<R>.h5
│  ├─ alpha0.1_miss0.1/<algo>/...
│  ├─ alpha0.1_miss0.2/<algo>/...
│  ├─ alpha1.0_miss0.0/<algo>/...
│  ├─ ...                              # (3 x 3 alpha x miss combinations)
│  └─ alpha1.0_miss0.1/FedGen/imputer_ablation/<imputer>/{models,metrics}/
│                                      # ablation cells live as a sub-bucket
├─ tables/
│  ├─ accuracy_emnist.{csv,md,tex}     # 3 datasets x 2 metrics = 6 tables
│  ├─ accuracy_ucihar.{csv,md,tex}
│  ├─ accuracy_wisdm.{csv,md,tex}
│  ├─ macro_f1_emnist.{csv,md,tex}
│  ├─ macro_f1_ucihar.{csv,md,tex}
│  ├─ macro_f1_wisdm.{csv,md,tex}
│  └─ imputer_ablation.{csv,md,tex}    # FedGen x 5 imputers x 3 datasets
└─ dashboards/
   ├─ emnist_dashboard.png             # 2x3 panels per dataset
   ├─ ucihar_dashboard.png
   ├─ wisdm_dashboard.png
   └─ hero_figure.png                  # 4 rows (panel types) x 3 cols (datasets)
```

### 11.6 Resume / skip behaviour

* Per-cell, per-seed: the driver inspects the expected
  `<TOKEN>_<algo>_..._<seed>.h5` filename for each seed and skips any
  seed whose summary file already exists.
* Per-round metrics dumps are namespaced under
  `metrics/seed_<s>/<TOKEN>/` so multi-seed runs do **not** overwrite
  each other (the per-round HDF5 filename does not include the seed,
  hence the per-seed sub-folder).
* If `results/metrics/` is left over from a previous crash, the driver
  now **auto-archives** it to `results/_stale_metrics_<UTC-timestamp>/`
  on startup (and again per-cell as a defensive net). No data is lost;
  the new sweep starts clean. See the "startup" line in
  `results/optionA/_status.md`.
* Per-round HDF5 dumps are now compact by default: `y_true`, `y_pred`,
  and `y_prob` are each stored **once** (schema_version=2), gzip-compressed,
  and cast to int32 / float32. Older sweeps that wrote 5-way aliased dumps
  are still readable by every downstream tool -- readers probe multiple
  candidate keys and fall back to the canonical ones.

### 11.7 What was deliberately removed

* MAR / MNAR mechanism sweeps. The code path stays in place
  (`utils/data_imputation.py::_mar_missing` / `_mnar_missing`) for future
  work, but it is **not part of this sweep**. The previous mechanism work
  on UCI HAR is parked in `results/_archive_mechanisms/`.
* MNIST. EMNIST-letters is the synthetic anchor.

### 11.8 force_imputer guarantee

Every "FedGen-AMDAE" row in the main tables is reviewer-bulletproof: the
driver passes `--force_imputer amdae` to `main.py`, which forces
`apply_amdae_imputation` to use AM-DAE unconditionally (bypassing the
composite-score auto-selection). The composite-score selection is still
correct and would still pick AM-DAE under `RELIABLE_METRICS`, but the
explicit force makes it impossible for a reviewer to argue the headline
numbers were silently produced by Mean / Median / Zero imputation.

The imputer-ablation table runs FedGen at the headline cell with each
of `{amdae, mean, median, zero, none}` so the paper can quantify how
much AM-DAE actually buys vs. simpler imputers and the no-imputation
baseline.

---

## 12. Outputs you will get

```
results/
├─ zero_missing_baseline/
│  ├─ amdae_declaration.txt              <- paste this paragraph in the paper
│  └─ <dataset>_zero_missing.{txt,csv}   <- 0% missing summary table
│
├─ models_mr0/, models_mr10/, models_mr20/
│  └─ <dataset>_<algo>_<lr>_<U>u_<B>b_<E>_<seed>[_embed0].h5
│        keys: glob_acc, glob_loss, per_acc, per_loss,
│              user_train_time, server_agg_time
│
├─ metrics_mr0/, metrics_mr10/, metrics_mr20/
│  └─ <dataset>/<algo>_<dataset>_round_<i>.h5
│        keys: y_true (alias: test_y, labels, targets, test_targets)
│              y_pred (alias: preds, test_pred, test_predictions, predictions)
│              y_prob (alias: probs, probabilities, logits, outputs)
│
├─ tables/
│  ├─ metrics_table_long.csv             <- all (algo,ds,alpha,miss) cells
│  ├─ <dataset>_metrics_wide.csv         <- paper-ready wide form
│  ├─ <dataset>_metrics_wide.md          <- markdown twin
│  └─ <dataset>_metrics_wide.tex         <- LaTeX twin
│
├─ figures/mr<R>/<dataset>/
│  ├─ f1_by_round.png + f1_by_round.csv
│  └─ confusion_matrix_round_<R>_<algo>.{png,csv}
│
├─ experiment_summary/
│  ├─ plot_<dataset>_alpha<a>_miss<m>.png
│  └─ table_<dataset>_alpha<a>_miss<m>.txt
│
├─ comprehensive_imputation_comparison.png   <- AM-DAE vs Mean/Median/Zero bar chart
└─ real_dataset_experiments/                  <- only when --include_ucihar
   └─ UCI_HAR-alpha<a>-ratio<r>_mr<m>.{txt,csv}
```

---

## 13. Troubleshooting

**"Dataset not recognized"** — the dataset token must match the
`get_data_dir()` patterns in `utils/model_utils.py`. Make sure the
hyphenation is exact (`Mnist-alpha0.1-ratio0.5`, not
`mnist-alpha-0.1-ratio-0.5`).

**`results/metrics/` keeps getting overwritten** — happens because
`utils/metrics_utils.py` writes there unconditionally. The orchestrator
fixes this by renaming the folder after each missing-rate batch. If you
run `main.py` directly across multiple missing rates, rename
`results/metrics` to `results/metrics_mr<rate>` between runs yourself
before invoking `goal3_metrics_table.py`.

**FedDistill vs FedDistill-FL vs FedDistill-pretrain** — the algorithm
string is parsed by substring: `FL` activates `share_model=True`,
`pretrain` activates a 20-iter pretrain phase. The default
`FedDistill` (no suffix) does logit-only distillation.

**Windows console garbles characters** — fixed in the goal-N scripts and
in the orchestrator. If you still see `?` or encoding errors, set
`PYTHONIOENCODING=utf-8` or run the scripts inside a UTF-8-aware
terminal.

**Out of memory on CUDA** — drop `--batch_size`, `--gen_batch_size`, or
`--num_users`. FedGen is the most memory-hungry because it carries the
generator and a frozen student copy.

**Why does `--ensemble_lr` seem ignored?** — for FedGen, `RUNCONFIGS`
in `utils/model_config.py` overrides it. Edit that table to change the
distillation lr.

**UCI HAR generator says "Could not find UCI HAR Dataset"** — you need
the raw archive at `data/UCI HAR/data/UCI HAR Dataset/` before running
the generator. There is no automatic download for HAR.

---

## 14. Citing this work

If you use this codebase, please cite:

- **FedGen (the underlying federated learning method).** Zhu, Hong, and
  Zhou, *"Data-Free Knowledge Distillation for Heterogeneous Federated
  Learning"*, ICML 2021.
- **AM-DAE (the imputation method).** Y. Cui et al., *"Imputation of
  Missing Values in Time Series Using an Adaptive-Learned Median-Filled
  Deep Autoencoder"*, IEEE Transactions on Cybernetics, 2023.

The exact paragraph documenting the AM-DAE integration (verbatim) lives
in `results/zero_missing_baseline/amdae_declaration.txt`. Generate it on
demand:

```bash
python goal1_zero_missing_baseline.py \
    --dataset Mnist-alpha0.1-ratio0.5 --declaration_only
```
