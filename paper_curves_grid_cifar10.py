#!/usr/bin/env python
"""


Workstation runbook
Pull the file, then from the repo root:

# Option A: EMNIST-letters, UCI HAR, WISDM
python3 paper_curves_individual.py --sweep optionA
# Run 3: MNIST, FedISIC, HAM10000
python3 paper_curves_individual.py --sweep run3
# CIFAR-10
python3 paper_curves_individual.py --sweep cifar10
# Or all three in one go
python3 paper_curves_individual.py --sweep all
Output layout
Everything goes to <input-root>/individual_curves/ by default. So after running all three sweeps you'll have:

results/optionA/individual_curves/
    EMNIST_acc_a0p1_m0.png    EMNIST_f1_a0p1_m0.png
    EMNIST_acc_a0p1_m10.png   EMNIST_f1_a0p1_m10.png
    ... (9 acc + 9 F1 per dataset)
    EMNIST_test_loss.png
    EMNIST_mean_acc.png
    UCIHAR_*.png
    WISDM_*.png
results/run3/individual_curves/
    MNIST_*.png
    FEDISIC_*.png
    HAM10000_*.png
results/cifar10/individual_curves/
    CIFAR10_*.png
Per-dataset file counts
Prefix	acc panels	F1 panels	test_loss	mean_acc	Total
Each dataset
9
9
1
1
20
So --sweep optionA gives 60 files, --sweep run3 gives 60 files, --sweep cifar10 gives 20 files. Total for --sweep all: 140 files.

Drop them straight into LaTeX
Point your LaTeX figure directory at results/<sweep>/individual_curves/ (or copy them across):

# Example: copy Option A + CIFAR-10 + Run 3 individual curves into your LaTeX figures folder
cp results/optionA/individual_curves/*.png    /path/to/latex/figures/
cp results/run3/individual_curves/*.png       /path/to/latex/figures/
cp results/cifar10/individual_curves/*.png    /path/to/latex/figures/
Useful flags
--no-f1 — skip the F1 panels (10× faster; F1 recompute from per-round dumps is the slow step)
--no-test-loss / --no-mean-acc — skip those two supplementary charts
--datasets WISDM CIFAR10 — only render specific dataset prefixes
--output-dir /path/to/latex/figures/ — write directly to your LaTeX folder, no copy step
--headline-alpha 1.0 --headline-missing 0.10 — change which cell the _test_loss.png uses

paper_curves_grid_cifar10.py
============================

Standalone renderer for the 3x3 (alpha x missing rate) accuracy- and
macro-F1-over-rounds grids for the CIFAR-10 sweep. Backport-style
counterpart to `paper_curves_grid_optionA.py`: same figure layout,
zero dependency on `paper_dashboard_cifar10.py`, safe to re-run any
number of times from the existing HDF5 checkpoints.

Reads directly from:

    results/cifar10/<dataset_short>/alpha<a>_miss<m>/<algo>/
        models/<TOKEN>_<algo>_<lr>_<num_users>u_<bs>b_<le>_<seed>.h5
        metrics/seed_<s>/<TOKEN>/
            <algo>_<TOKEN>_round_<R>.h5

Emits into `--output-dir` (default `results/cifar10/dashboards/`):

    cifar10_accuracy_curves.png    # 3 (alpha) x 3 (miss rate) panels
    cifar10_f1_curves.png          # same layout, Macro F1 over rounds

Cells with no completed seeds render as blank panels with a "-- no data --"
placeholder so the figure layout stays stable on a partially-complete sweep.

Usage::

    # Default: both accuracy and F1 grids
    python paper_curves_grid_cifar10.py

    # Only accuracy grid (fast; skips per-round F1 recomputation)
    python paper_curves_grid_cifar10.py --no-f1

    # Different result tree
    python paper_curves_grid_cifar10.py --input-root results/cifar10 \\
        --output-dir results/cifar10/dashboards
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np  # noqa: F401  -- type hints only

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "results" / "cifar10"
DEFAULT_OUTPUT = ROOT / "results" / "cifar10" / "dashboards"

ALGOS_ORDER = ["FedGen", "FedAvg", "FedProx", "FedEnsemble", "FedDistill"]
ALGO_DISPLAY = {
    "FedGen": "FedGen-AMDAE",
    "FedAvg": "FedAvg",
    "FedProx": "FedProx",
    "FedEnsemble": "FedEnsemble",
    "FedDistill": "FedDistill",
}

# (dataset_token_prefix, human_display_label, on-disk dataset_short)
DATASETS_ORDER = [
    ("CIFAR10", "CIFAR-10", "cifar10"),
]
ALPHAS = [0.1, 1.0, 10.0]
MISSING_RATES = [0.0, 0.10, 0.20]


# ---------------------------------------------------------------- adapters
def dataset_token(dataset: str, alpha: float,
                  sampling_ratio: float = 0.5) -> str:
    return f"{dataset}-alpha{alpha}-ratio{sampling_ratio}"


def cell_dir(input_root: Path, dataset_short: str, alpha: float,
             miss: float, algo: str) -> Path:
    return input_root / dataset_short / f"alpha{alpha}_miss{miss}" / algo


def _round_idx(p: Path) -> int:
    try:
        return int(p.stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


# ---------------------------------------------------------------- HDF5 I/O
def read_curves(models_dir: Path,
                key: str = "glob_acc") -> Optional["np.ndarray"]:
    """Stack `key` arrays from every per-seed summary HDF5 in `models_dir`,
    truncating to the shortest length. Returns None when no seed exists."""
    import h5py
    import numpy as np
    if not models_dir.is_dir():
        return None
    arrs = []
    for h5_path in sorted(models_dir.glob("*.h5")):
        try:
            with h5py.File(h5_path, "r") as hf:
                if key not in hf:
                    continue
                arr = np.asarray(hf[key][:], dtype=float)
                if arr.size:
                    arrs.append(arr)
        except OSError:
            continue
    if not arrs:
        return None
    minlen = min(len(a) for a in arrs)
    return np.stack([a[:minlen] for a in arrs], axis=0)


def read_f1_curves(metrics_dir: Path, dataset: str, alpha: float,
                   algo: str) -> Optional["np.ndarray"]:
    """For each seed under metrics_dir/seed_<s>/<TOKEN>/, iterate through
    every per-round HDF5 dump in ascending round order, compute Macro F1
    from (y_true, y_pred), and stack the resulting per-seed curves into
    a (n_seeds, n_rounds) ndarray (truncated to the shortest curve)."""
    import h5py
    import numpy as np
    from sklearn.metrics import f1_score
    if not metrics_dir.is_dir():
        return None

    token = dataset_token(dataset, alpha)
    seed_dirs = sorted(p for p in metrics_dir.iterdir()
                       if p.is_dir() and p.name.startswith("seed_"))
    if not seed_dirs:
        flat = metrics_dir / token
        seed_dirs = [flat] if flat.is_dir() else []

    per_seed_curves: List[List[float]] = []
    for sd in seed_dirs:
        token_dir = sd / token if sd.name.startswith("seed_") else sd
        if not token_dir.is_dir():
            continue
        rounds = sorted(token_dir.glob(f"{algo}_*round_*.h5"),
                        key=lambda p: _round_idx(p))
        curve: List[float] = []
        for h5_path in rounds:
            try:
                with h5py.File(h5_path, "r") as hf:
                    yt = np.asarray(hf["y_true"][:]).reshape(-1)
                    yp = np.asarray(hf["y_pred"][:]).reshape(-1)
                curve.append(float(
                    f1_score(yt, yp, average="macro", zero_division=0)))
            except (OSError, KeyError, ValueError):
                continue
        if curve:
            per_seed_curves.append(curve)

    if not per_seed_curves:
        return None
    minlen = min(len(c) for c in per_seed_curves)
    return np.stack([np.asarray(c[:minlen], dtype=float)
                     for c in per_seed_curves], axis=0)


# ---------------------------------------------------------------- panels
def _no_data_panel(ax, msg: str = "-- no data --") -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, fontsize=10, color="#888")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_curve_panel(ax, curves_per_algo: Dict[str, "np.ndarray"],
                      title: str, ylabel: str,
                      ylim: Optional[Tuple[float, float]] = None,
                      legend: bool = True) -> bool:
    """Given a mapping algo -> (n_seeds, n_rounds) ndarray already scaled
    to display units, draw a mean +/- std curve panel."""
    import numpy as np
    drew_any = False
    for algo in ALGOS_ORDER:
        arr = curves_per_algo.get(algo)
        if arr is None or arr.size == 0:
            continue
        mu = arr.mean(axis=0)
        sd = (arr.std(axis=0, ddof=1) if arr.shape[0] > 1
              else np.zeros_like(mu))
        x = np.arange(1, len(mu) + 1)
        ax.plot(x, mu, label=ALGO_DISPLAY[algo], linewidth=1.6)
        if arr.shape[0] > 1:
            ax.fill_between(x, mu - sd, mu + sd, alpha=0.15)
        drew_any = True

    ax.set_title(title, fontsize=10)
    if not drew_any:
        _no_data_panel(ax)
        return False

    ax.set_xlabel("Round", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(fontsize=7, loc="best", framealpha=0.85)
    return True


# ---------------------------------------------------------------- composer
def compose_curves_grid(input_root: Path, output_dir: Path,
                        dataset: str, dataset_disp: str,
                        dataset_short: str,
                        which: str = "accuracy") -> Path:
    """3 (alpha) x 3 (missing rate) grid of per-algo curve panels.

    which in {'accuracy', 'f1'}:
      - accuracy: reads glob_acc from summary HDF5s (fast)
      - f1:       computes Macro F1 from per-round y_true/y_pred (slow)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex="col")

    for i, alpha in enumerate(ALPHAS):
        for j, miss in enumerate(MISSING_RATES):
            ax = axes[i, j]
            title = f"alpha = {alpha}, missing = {int(miss * 100)}%"

            if which == "accuracy":
                curves: Dict[str, "np.ndarray"] = {}
                for algo in ALGOS_ORDER:
                    md = cell_dir(input_root, dataset_short,
                                  alpha, miss, algo) / "models"
                    arr = read_curves(md, "glob_acc")
                    if arr is None:
                        continue
                    curves[algo] = arr * 100.0
                _draw_curve_panel(ax, curves, title, "Accuracy (%)",
                                  legend=(i == 0 and j == 0))
            else:
                curves = {}
                for algo in ALGOS_ORDER:
                    md = cell_dir(input_root, dataset_short,
                                  alpha, miss, algo) / "metrics"
                    arr = read_f1_curves(md, dataset, alpha, algo)
                    if arr is None:
                        continue
                    curves[algo] = arr * 100.0
                _draw_curve_panel(ax, curves, title, "Macro F1 (%)",
                                  legend=(i == 0 and j == 0))

    kind_label = "Accuracy" if which == "accuracy" else "Macro F1"
    fig.suptitle(f"{dataset_disp}  --  {kind_label} over rounds "
                 f"(alpha x missing rate)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    suffix = "accuracy_curves" if which == "accuracy" else "f1_curves"
    out = output_dir / f"{dataset_short}_{suffix}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


# ---------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-root", default=str(DEFAULT_INPUT),
                   help=f"Root of the CIFAR-10 sweep results "
                        f"(default {DEFAULT_INPUT}).")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT),
                   help=f"Where to write PNG grids "
                        f"(default {DEFAULT_OUTPUT}).")
    p.add_argument("--no-accuracy", action="store_true",
                   help="Skip the accuracy-over-rounds grid.")
    p.add_argument("--no-f1", action="store_true",
                   help="Skip the F1-over-rounds grid "
                        "(this is the slower one: it recomputes F1 "
                        "from per-round y_true / y_pred dumps).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nReading from  : {input_root}")
    print(f"Writing figs  : {output_dir}")
    print(f"Grids         : "
          f"{'accuracy ' if not args.no_accuracy else ''}"
          f"{'f1' if not args.no_f1 else ''}".strip())
    print()

    import matplotlib
    matplotlib.use("Agg")

    if not input_root.is_dir():
        print(f"[ERROR] input root not found: {input_root}", file=sys.stderr)
        sys.exit(2)

    n_written = 0
    for dataset, dataset_disp, dataset_short in DATASETS_ORDER:
        if not args.no_accuracy:
            try:
                compose_curves_grid(input_root, output_dir,
                                    dataset, dataset_disp, dataset_short,
                                    which="accuracy")
                n_written += 1
            except Exception as e:
                print(f"[WARN] accuracy grid for {dataset} failed: {e}",
                      file=sys.stderr)

        if not args.no_f1:
            try:
                compose_curves_grid(input_root, output_dir,
                                    dataset, dataset_disp, dataset_short,
                                    which="f1")
                n_written += 1
            except Exception as e:
                print(f"[WARN] F1 grid for {dataset} failed: {e}",
                      file=sys.stderr)

    print(f"\nDone. {n_written} figure(s) written.")


if __name__ == "__main__":
    main()
