#!/usr/bin/env python
r"""
paper_curves_individual.py
==========================

Emit LaTeX-ready INDIVIDUAL PNG panels from any completed sweep
(Option A, Run 3, or CIFAR-10), following the naming convention:

    <PREFIX>_acc_a<ALPHA>_m<MISS>.png     # one per (alpha, missing) cell
    <PREFIX>_f1_a<ALPHA>_m<MISS>.png      # optional (see --no-f1)
    <PREFIX>_test_loss.png                # test loss curve at headline cell
    <PREFIX>_mean_acc.png                 # bar chart, mean acc across 9 cells

Alpha tag: `0.1 -> 0p1`, `1.0 -> 1p0`, `10.0 -> 10p0`
Miss tag : `0.0 -> 0`,   `0.10 -> 10`, `0.20 -> 20`
Prefix is per-dataset (WISDM, UCIHAR, EMNIST, MNIST, CIFAR10,
FEDISIC, HAM10000) matching how the paper's LaTeX subfigures name
their `\includegraphics` targets.

Usage::

    # Option A (EMNIST-letters, UCI HAR, WISDM)
    python paper_curves_individual.py --sweep optionA

    # Run 3 (MNIST, FedISIC, HAM10000)
    python paper_curves_individual.py --sweep run3

    # CIFAR-10
    python paper_curves_individual.py --sweep cifar10

    # All three at once
    python paper_curves_individual.py --sweep all

    # Skip F1 recompute (much faster)
    python paper_curves_individual.py --sweep optionA --no-f1

    # Change destination
    python paper_curves_individual.py --sweep run3 \\
        --output-dir /path/to/latex/figures/

Cells with no completed seeds render as blank panels with a "-- no data --"
placeholder, so partial sweeps still produce a full grid without crashing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np  # noqa: F401  -- type hints only

ROOT = Path(__file__).resolve().parent

ALGOS_ORDER = ["FedGen", "FedAvg", "FedProx", "FedEnsemble", "FedDistill"]
ALGO_DISPLAY = {
    "FedGen": "FedGen-AMDAE",
    "FedAvg": "FedAvg",
    "FedProx": "FedProx",
    "FedEnsemble": "FedEnsemble",
    "FedDistill": "FedDistill",
}
ALPHAS = [0.1, 1.0, 10.0]
MISSING_RATES = [0.0, 0.10, 0.20]
HEADLINE_ALPHA = 1.0
HEADLINE_MISS = 0.10

# ---------------------------------------------------------------- presets
# (dataset_token_prefix, on-disk dataset_short, LaTeX filename prefix)
SWEEP_PRESETS: Dict[str, Dict] = {
    "optionA": {
        "default_input": ROOT / "results" / "optionA",
        "datasets": [
            ("EMnist-letters", "emnist", "EMNIST"),
            ("UCI HAR",        "ucihar", "UCIHAR"),
            ("WISDM",          "wisdm",  "WISDM"),
        ],
    },
    "run3": {
        "default_input": ROOT / "results" / "run3",
        "datasets": [
            ("Mnist",    "mnist",    "MNIST"),
            ("FedISIC",  "fedisic",  "FEDISIC"),
            ("HAM10000", "ham10000", "HAM10000"),
        ],
    },
    "cifar10": {
        "default_input": ROOT / "results" / "cifar10",
        "datasets": [
            ("CIFAR10", "cifar10", "CIFAR10"),
        ],
    },
}


# ---------------------------------------------------------------- filename tags
def alpha_tag(a: float) -> str:
    """0.1 -> '0p1', 1.0 -> '1p0', 10.0 -> '10p0'."""
    return str(a).replace(".", "p")


def miss_tag(m: float) -> str:
    """0.0 -> '0', 0.10 -> '10', 0.20 -> '20'."""
    return str(int(round(m * 100)))


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
    truncating to the shortest length."""
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
    """For each seed, iterate every per-round HDF5 in ascending round
    order and compute Macro F1 from (y_true, y_pred). Returns
    (n_seeds, n_rounds), or None if no seed / no round produced valid
    predictions."""
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


# ---------------------------------------------------------------- panel primitive
def _no_data_panel(ax, msg: str = "-- no data --") -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, fontsize=11, color="#888")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_curve_panel(ax, curves_per_algo: Dict[str, "np.ndarray"],
                      ylabel: str, legend: bool = True) -> bool:
    """Given a mapping algo -> (n_seeds, n_rounds) ndarray already scaled
    to display units, draw a mean +/- std curve panel. Returns True iff
    at least one curve was drawn."""
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

    if not drew_any:
        _no_data_panel(ax)
        return False

    ax.set_xlabel("Round", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(fontsize=8, loc="best", framealpha=0.85)
    return True


# ---------------------------------------------------------------- renderers
def render_single_accuracy_panel(input_root: Path, dataset_short: str,
                                 alpha: float, miss: float,
                                 prefix: str, output_dir: Path) -> Path:
    """One PNG containing the 5-algo accuracy curve for a single
    (alpha, missing) cell."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    curves: Dict[str, "np.ndarray"] = {}
    for algo in ALGOS_ORDER:
        md = cell_dir(input_root, dataset_short, alpha, miss, algo) / "models"
        arr = read_curves(md, "glob_acc")
        if arr is None:
            continue
        curves[algo] = arr * 100.0
    _draw_curve_panel(ax, curves, "Accuracy (%)", legend=True)

    fig.tight_layout()
    out = output_dir / f"{prefix}_acc_a{alpha_tag(alpha)}_m{miss_tag(miss)}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def render_single_f1_panel(input_root: Path, dataset: str,
                           dataset_short: str, alpha: float, miss: float,
                           prefix: str, output_dir: Path) -> Path:
    """One PNG containing the 5-algo Macro F1 curve for a single
    (alpha, missing) cell."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    curves: Dict[str, "np.ndarray"] = {}
    for algo in ALGOS_ORDER:
        md = cell_dir(input_root, dataset_short, alpha, miss, algo) / "metrics"
        arr = read_f1_curves(md, dataset, alpha, algo)
        if arr is None:
            continue
        curves[algo] = arr * 100.0
    _draw_curve_panel(ax, curves, "Macro F1 (%)", legend=True)

    fig.tight_layout()
    out = output_dir / f"{prefix}_f1_a{alpha_tag(alpha)}_m{miss_tag(miss)}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def render_test_loss(input_root: Path, dataset_short: str,
                     prefix: str, output_dir: Path,
                     alpha: float = HEADLINE_ALPHA,
                     miss: float = HEADLINE_MISS) -> Path:
    """Standalone test-loss PNG at the headline cell (default
    alpha=1, miss=10%)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    curves: Dict[str, "np.ndarray"] = {}
    for algo in ALGOS_ORDER:
        md = cell_dir(input_root, dataset_short, alpha, miss, algo) / "models"
        arr = read_curves(md, "glob_loss")
        if arr is None:
            continue
        # Loss is stored raw, not as a percentage.
        curves[algo] = arr

    _draw_curve_panel(ax, curves, "Test loss", legend=True)

    fig.tight_layout()
    out = output_dir / f"{prefix}_test_loss.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def render_mean_acc_bar(input_root: Path, dataset_short: str,
                        prefix: str, output_dir: Path) -> Path:
    """Bar chart of mean +/- std final accuracy per algorithm across
    all 9 (alpha, missing) cells."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    means: List[float] = []
    stds:  List[float] = []
    labels: List[str] = []
    for algo in ALGOS_ORDER:
        per_cell_means: List[float] = []
        for alpha in ALPHAS:
            for miss in MISSING_RATES:
                md = cell_dir(input_root, dataset_short,
                              alpha, miss, algo) / "models"
                arr = read_curves(md, "glob_acc")
                if arr is None:
                    continue
                per_cell_means.append(float(arr[:, -1].mean()) * 100.0)
        if not per_cell_means:
            continue
        labels.append(ALGO_DISPLAY[algo])
        means.append(float(np.mean(per_cell_means)))
        stds.append(float(np.std(per_cell_means, ddof=1))
                    if len(per_cell_means) > 1 else 0.0)

    if not means:
        _no_data_panel(ax)
    else:
        x = np.arange(len(means))
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color="#4c8bf5", edgecolor="black", linewidth=0.6)
        win = int(np.argmax(means))
        bars[win].set_color("#f0a500")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("Mean acc (%) across 9 cells", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = output_dir / f"{prefix}_mean_acc.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- driver
def render_sweep(sweep_name: str, input_root: Path, output_dir: Path,
                 include_f1: bool = True,
                 include_accuracy: bool = True,
                 include_test_loss: bool = True,
                 include_mean_acc: bool = True,
                 headline_alpha: float = HEADLINE_ALPHA,
                 headline_miss: float = HEADLINE_MISS,
                 datasets_filter: Optional[List[str]] = None) -> int:
    """Render every LaTeX-named PNG for one sweep. Returns file count."""
    preset = SWEEP_PRESETS[sweep_name]
    n_written = 0

    print(f"\n[{sweep_name}]  input={input_root}")
    print(f"[{sweep_name}]  output={output_dir}")

    for dataset, dataset_short, prefix in preset["datasets"]:
        if datasets_filter is not None and prefix not in datasets_filter:
            continue

        print(f"\n  {prefix}:")
        for alpha in ALPHAS:
            for miss in MISSING_RATES:
                if include_accuracy:
                    try:
                        out = render_single_accuracy_panel(
                            input_root, dataset_short, alpha, miss,
                            prefix, output_dir)
                        print(f"    wrote {out.name}")
                        n_written += 1
                    except Exception as e:
                        print(f"    [WARN] acc a={alpha} m={miss} failed: {e}",
                              file=sys.stderr)
                if include_f1:
                    try:
                        out = render_single_f1_panel(
                            input_root, dataset, dataset_short, alpha, miss,
                            prefix, output_dir)
                        print(f"    wrote {out.name}")
                        n_written += 1
                    except Exception as e:
                        print(f"    [WARN] f1 a={alpha} m={miss} failed: {e}",
                              file=sys.stderr)

        if include_test_loss:
            try:
                out = render_test_loss(input_root, dataset_short, prefix,
                                       output_dir,
                                       alpha=headline_alpha,
                                       miss=headline_miss)
                print(f"    wrote {out.name}")
                n_written += 1
            except Exception as e:
                print(f"    [WARN] test_loss failed: {e}", file=sys.stderr)

        if include_mean_acc:
            try:
                out = render_mean_acc_bar(input_root, dataset_short, prefix,
                                          output_dir)
                print(f"    wrote {out.name}")
                n_written += 1
            except Exception as e:
                print(f"    [WARN] mean_acc failed: {e}", file=sys.stderr)

    return n_written


# ---------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep",
                   choices=list(SWEEP_PRESETS.keys()) + ["all"],
                   required=True,
                   help="Which sweep to render. 'all' renders Option A + "
                        "Run 3 + CIFAR-10 in one go.")
    p.add_argument("--input-root", default=None,
                   help="Override sweep's default results root "
                        "(e.g. results/optionA). Only makes sense with a "
                        "single --sweep value.")
    p.add_argument("--output-dir", default=None,
                   help="Where to write PNGs (default: "
                        "<input-root>/individual_curves).")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Filter by LaTeX prefix (WISDM UCIHAR EMNIST MNIST "
                        "CIFAR10 FEDISIC HAM10000). Default: all datasets "
                        "for the chosen sweep.")
    p.add_argument("--headline-alpha", type=float, default=HEADLINE_ALPHA,
                   help=f"Alpha for the standalone test-loss chart "
                        f"(default {HEADLINE_ALPHA}).")
    p.add_argument("--headline-missing", type=float, default=HEADLINE_MISS,
                   help=f"Missing rate for the standalone test-loss chart "
                        f"(default {HEADLINE_MISS}).")
    p.add_argument("--no-accuracy", action="store_true",
                   help="Skip the per-cell accuracy PNGs.")
    p.add_argument("--no-f1", action="store_true",
                   help="Skip the per-cell F1 PNGs (fast path: F1 recompute "
                        "from per-round dumps is the slow step).")
    p.add_argument("--no-test-loss", action="store_true",
                   help="Skip the <PREFIX>_test_loss.png chart.")
    p.add_argument("--no-mean-acc", action="store_true",
                   help="Skip the <PREFIX>_mean_acc.png bar chart.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    sweeps = (list(SWEEP_PRESETS.keys())
              if args.sweep == "all" else [args.sweep])

    if args.input_root is not None and len(sweeps) > 1:
        print("[ERROR] --input-root can only be used with a single --sweep, "
              "not --sweep all.", file=sys.stderr)
        sys.exit(2)

    import matplotlib
    matplotlib.use("Agg")

    total = 0
    for sweep_name in sweeps:
        preset = SWEEP_PRESETS[sweep_name]
        input_root = (Path(args.input_root).resolve()
                      if args.input_root else Path(preset["default_input"]))
        if not input_root.is_dir():
            print(f"[WARN] {sweep_name}: input root not found ({input_root}), "
                  f"skipping.", file=sys.stderr)
            continue

        output_dir = (Path(args.output_dir).resolve()
                      if args.output_dir else
                      input_root / "individual_curves")

        n = render_sweep(
            sweep_name=sweep_name,
            input_root=input_root,
            output_dir=output_dir,
            include_f1=not args.no_f1,
            include_accuracy=not args.no_accuracy,
            include_test_loss=not args.no_test_loss,
            include_mean_acc=not args.no_mean_acc,
            headline_alpha=args.headline_alpha,
            headline_miss=args.headline_missing,
            datasets_filter=args.datasets,
        )
        total += n
        print(f"\n[{sweep_name}] {n} PNG(s) written to {output_dir}")

    print(f"\nDone. {total} figure(s) written in total.")


if __name__ == "__main__":
    main()
