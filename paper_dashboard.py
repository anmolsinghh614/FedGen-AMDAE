#!/usr/bin/env python
"""
paper_dashboard.py
==================

Compose paper-ready dashboard figures from a completed Option-A sweep.

Two products:

1. Per-dataset dashboard (one PNG per dataset, 2x3 grid). For the headline
   cell (alpha=1, miss=10%) of `dataset`:

       (0,0) Test accuracy curve, all 5 algorithms (mean +/- std band)
       (0,1) Train loss curve,    all 5 algorithms (mean +/- std band)
       (0,2) Per-class F1 heatmap (rows = algorithms, cols = classes)
       (1,0) FedGen   confusion matrix
       (1,1) FedDistill confusion matrix
       (1,2) Mean +/- std final accuracy per algorithm across all 9
             (alpha, missing) cells (bar chart)

   Output: results/optionA/dashboards/<dataset>_dashboard.png

2. Global hero figure (1 PNG, 4 rows x 3 cols):

       row 0: accuracy-curve panel for each dataset
       row 1: per-class-F1-heatmap panel for each dataset
       row 2: FedGen confusion matrix per dataset
       row 3: per-algorithm bar-chart per dataset

   Output: results/optionA/dashboards/hero_figure.png

Cells that have no completed seeds are rendered as a blank panel with a
"-- no data --" placeholder so the figure layout stays stable even on a
partially-complete sweep.

Usage::

    python paper_dashboard.py --input-root results/optionA \
                              --output-dir results/optionA/dashboards \
                              --headline-alpha 1 --headline-missing 0.10
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np  # noqa: F401  -- only for type hints

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "results" / "optionA"
DEFAULT_OUTPUT = ROOT / "results" / "optionA" / "dashboards"

ALGOS_ORDER = ["FedGen", "FedAvg", "FedProx", "FedEnsemble", "FedDistill"]
ALGO_DISPLAY = {
    "FedGen": "FedGen-AMDAE",
    "FedAvg": "FedAvg",
    "FedProx": "FedProx",
    "FedEnsemble": "FedEnsemble",
    "FedDistill": "FedDistill",
}
DATASETS_ORDER = [
    ("EMnist-letters", "EMNIST",  "emnist"),
    ("UCI HAR",        "UCI HAR", "ucihar"),
    ("PAMAP2",         "PAMAP2",  "pamap2"),
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


# ---------------------------------------------------------------- HDF5 I/O
def read_curves(models_dir: Path,
                key: str = "glob_acc") -> Optional["np.ndarray"]:
    """Stack `key` arrays from every per-seed summary HDF5 in `models_dir`,
    truncating to the shortest length so a partial Stage-2 cell still
    produces a usable mean curve. Returns None when no seed exists."""
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


def read_yp_yt(metrics_dir: Path, dataset: str, alpha: float, algo: str
               ) -> Optional[Tuple["np.ndarray", "np.ndarray"]]:
    """Return (y_true, y_pred) for the highest-round HDF5 of this cell.
    Tries the seed_<s>/<TOKEN>/ layout first, then falls back to flat
    <metrics>/<TOKEN>/ for backwards compatibility."""
    import h5py
    import numpy as np
    if not metrics_dir.is_dir():
        return None

    token = dataset_token(dataset, alpha)

    seed_dirs = sorted(p for p in metrics_dir.iterdir()
                       if p.is_dir() and p.name.startswith("seed_"))
    if not seed_dirs:
        flat = metrics_dir / token
        if flat.is_dir():
            seed_dirs = [flat]

    cands: List[Path] = []
    for sd in seed_dirs:
        token_dir = sd / token if sd.name.startswith("seed_") else sd
        if not token_dir.is_dir():
            continue
        for f in token_dir.glob(f"{algo}_*round_*.h5"):
            cands.append(f)
    if not cands:
        return None

    cands.sort(key=lambda p: _round_idx(p))
    last = cands[-1]
    try:
        with h5py.File(last, "r") as hf:
            yt = np.asarray(hf["y_true"][:]).reshape(-1)
            yp = np.asarray(hf["y_pred"][:]).reshape(-1)
        return yt, yp
    except (OSError, KeyError, ValueError):
        return None


def _round_idx(p: Path) -> int:
    try:
        return int(p.stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


# ---------------------------------------------------------------- panel I/O
def _no_data_panel(ax, msg: str = "-- no data --") -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, fontsize=10, color="#888")
    ax.set_xticks([])
    ax.set_yticks([])


def panel_curve(ax, models_dir_per_algo: Dict[str, Path], key: str,
                title: str, ylabel: str, percent: bool = True) -> None:
    """Draw a per-algorithm mean +/- std curve panel. `models_dir_per_algo`
    maps algorithm -> models dir for one cell."""
    import numpy as np
    import matplotlib.pyplot as plt
    drew_any = False
    for algo in ALGOS_ORDER:
        md = models_dir_per_algo.get(algo)
        if md is None:
            continue
        arr = read_curves(md, key)
        if arr is None:
            continue
        scale = 100.0 if percent else 1.0
        mu = arr.mean(axis=0) * scale
        sd = (arr.std(axis=0, ddof=1) if arr.shape[0] > 1
              else np.zeros_like(mu)) * scale
        x = np.arange(1, len(mu) + 1)
        ax.plot(x, mu, label=ALGO_DISPLAY[algo], linewidth=1.6)
        if arr.shape[0] > 1:
            ax.fill_between(x, mu - sd, mu + sd, alpha=0.15)
        drew_any = True

    ax.set_title(title, fontsize=10)
    if not drew_any:
        _no_data_panel(ax)
        return

    ax.set_xlabel("Round", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best", framealpha=0.85)


def panel_per_class_f1_heatmap(ax, dataset: str, alpha: float,
                               metrics_dirs_per_algo: Dict[str, Path],
                               title: str) -> None:
    """Heatmap: rows = algorithms, cols = classes."""
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score
    rows = []
    row_labels = []
    n_classes = 0
    for algo in ALGOS_ORDER:
        md = metrics_dirs_per_algo.get(algo)
        if md is None:
            continue
        result = read_yp_yt(md, dataset, alpha, algo)
        if result is None:
            continue
        yt, yp = result
        labels = sorted(set(np.concatenate([yt, yp]).tolist()))
        f1 = f1_score(yt, yp, average=None, zero_division=0,
                      labels=labels)
        rows.append(np.asarray(f1, dtype=float))
        row_labels.append(ALGO_DISPLAY[algo])
        n_classes = max(n_classes, len(labels))
    if not rows:
        ax.set_title(title, fontsize=10)
        _no_data_panel(ax)
        return

    # Pad rows to common length
    n_classes = max(len(r) for r in rows)
    M = np.full((len(rows), n_classes), np.nan)
    for i, r in enumerate(rows):
        M[i, :len(r)] = r
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks(range(n_classes))
    if n_classes <= 12:
        ax.set_xticklabels(range(n_classes), fontsize=8)
    else:
        # subsample x-tick labels for high-class datasets like EMNIST-letters
        step = max(1, n_classes // 13)
        keep = list(range(0, n_classes, step))
        ax.set_xticks(keep)
        ax.set_xticklabels([str(i) for i in keep], fontsize=7)
    ax.set_xlabel("Class", fontsize=9)
    ax.set_title(title, fontsize=10)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("F1", fontsize=8)


def panel_confusion_matrix(ax, dataset: str, alpha: float, algo: str,
                           metrics_dir: Path, title: str) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    result = read_yp_yt(metrics_dir, dataset, alpha, algo) \
        if metrics_dir is not None else None
    if result is None:
        ax.set_title(title, fontsize=10)
        _no_data_panel(ax)
        return
    yt, yp = result
    labels = sorted(set(np.concatenate([yt, yp]).tolist()))
    cm = confusion_matrix(yt, yp, labels=labels)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    n = len(labels)
    if n <= 12:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
    else:
        step = max(1, n // 13)
        keep = list(range(0, n, step))
        ax.set_xticks(keep)
        ax.set_yticks(keep)
        ax.set_xticklabels([str(labels[i]) for i in keep], fontsize=6)
        ax.set_yticklabels([str(labels[i]) for i in keep], fontsize=6)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)


def panel_bar_per_algo(ax, input_root: Path, dataset: str,
                       dataset_short: str, title: str) -> None:
    """For each algorithm, show mean +/- std final accuracy across all
    9 (alpha, missing) cells of this dataset. One bar per algorithm."""
    import numpy as np
    import matplotlib.pyplot as plt
    means = []
    stds = []
    labels = []
    for algo in ALGOS_ORDER:
        per_cell_means: List[float] = []
        for alpha in ALPHAS:
            for miss in MISSING_RATES:
                models_dir = cell_dir(input_root, dataset_short,
                                      alpha, miss, algo) / "models"
                arr = read_curves(models_dir, "glob_acc")
                if arr is None:
                    continue
                # Final-round per-seed accuracies; take mean across seeds
                per_cell_means.append(float(arr[:, -1].mean()) * 100.0)
        if not per_cell_means:
            continue
        labels.append(ALGO_DISPLAY[algo])
        means.append(float(np.mean(per_cell_means)))
        stds.append(float(np.std(per_cell_means, ddof=1))
                    if len(per_cell_means) > 1 else 0.0)

    if not means:
        ax.set_title(title, fontsize=10)
        _no_data_panel(ax)
        return

    x = np.arange(len(means))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="#4c8bf5",
                  edgecolor="black", linewidth=0.6)
    # Bold the winner
    if means:
        win = int(np.argmax(means))
        bars[win].set_color("#f0a500")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Mean acc (%) across 9 cells", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", alpha=0.3)


# ---------------------------------------------------------------- composers
def compose_per_dataset_dashboard(input_root: Path, output_dir: Path,
                                  dataset: str, dataset_disp: str,
                                  dataset_short: str,
                                  headline_alpha: float,
                                  headline_miss: float) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Per-cell dirs for the headline cell, keyed by algorithm
    models_per_algo: Dict[str, Path] = {}
    metrics_per_algo: Dict[str, Path] = {}
    for algo in ALGOS_ORDER:
        cd = cell_dir(input_root, dataset_short,
                      headline_alpha, headline_miss, algo)
        models_per_algo[algo] = cd / "models"
        metrics_per_algo[algo] = cd / "metrics"

    headline_str = (f"{dataset_disp}, alpha={headline_alpha}, "
                    f"miss={int(headline_miss * 100)}%")

    panel_curve(axes[0, 0], models_per_algo, "glob_acc",
                title=f"Test accuracy ({headline_str})",
                ylabel="Test acc (%)", percent=True)
    panel_curve(axes[0, 1], models_per_algo, "glob_loss",
                title=f"Test loss ({headline_str})",
                ylabel="Test loss", percent=False)
    panel_per_class_f1_heatmap(axes[0, 2], dataset, headline_alpha,
                               metrics_per_algo,
                               title=f"Per-class F1 ({headline_str})")
    panel_confusion_matrix(axes[1, 0], dataset, headline_alpha, "FedGen",
                           metrics_per_algo.get("FedGen"),
                           title=f"FedGen-AMDAE confusion ({headline_str})")
    panel_confusion_matrix(axes[1, 1], dataset, headline_alpha, "FedDistill",
                           metrics_per_algo.get("FedDistill"),
                           title=f"FedDistill confusion ({headline_str})")
    panel_bar_per_algo(axes[1, 2], input_root, dataset, dataset_short,
                       title=f"{dataset_disp}: mean acc per algorithm")

    fig.suptitle(f"{dataset_disp}  --  Option A dashboard", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = output_dir / f"{dataset_short}_dashboard.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def compose_hero_figure(input_root: Path, output_dir: Path,
                        headline_alpha: float, headline_miss: float) -> Path:
    """4 rows (panel types) x 3 cols (datasets) hero figure."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(18, 22))

    for j, (dataset, dataset_disp, dataset_short) in enumerate(DATASETS_ORDER):
        models_per_algo: Dict[str, Path] = {}
        metrics_per_algo: Dict[str, Path] = {}
        for algo in ALGOS_ORDER:
            cd = cell_dir(input_root, dataset_short,
                          headline_alpha, headline_miss, algo)
            models_per_algo[algo] = cd / "models"
            metrics_per_algo[algo] = cd / "metrics"

        head = (f"{dataset_disp}\n(alpha={headline_alpha}, "
                f"miss={int(headline_miss * 100)}%)")

        # Row 0: accuracy curves
        panel_curve(axes[0, j], models_per_algo, "glob_acc",
                    title=f"Test accuracy -- {head}",
                    ylabel="Test acc (%)", percent=True)
        # Row 1: per-class F1 heatmap
        panel_per_class_f1_heatmap(axes[1, j], dataset, headline_alpha,
                                   metrics_per_algo,
                                   title=f"Per-class F1 -- {head}")
        # Row 2: FedGen-AMDAE confusion matrix
        panel_confusion_matrix(axes[2, j], dataset, headline_alpha, "FedGen",
                               metrics_per_algo.get("FedGen"),
                               title=f"FedGen-AMDAE confusion -- {head}")
        # Row 3: per-algorithm bar across all 9 cells
        panel_bar_per_algo(axes[3, j], input_root, dataset, dataset_short,
                           title=f"{dataset_disp}: mean acc across cells")

    fig.suptitle(
        "FedGen-AMDAE  --  hero figure across EMNIST / UCI HAR / PAMAP2",
        fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = output_dir / "hero_figure.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


# ---------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-root", default=str(DEFAULT_INPUT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--headline-alpha", type=float, default=1.0)
    p.add_argument("--headline-missing", type=float, default=0.10)
    p.add_argument("--datasets", nargs="*",
                   choices=[d for d, _, _ in DATASETS_ORDER],
                   default=[d for d, _, _ in DATASETS_ORDER],
                   help="Subset of datasets to dashboard.")
    p.add_argument("--no-hero", action="store_true",
                   help="Skip the 4x3 hero figure (only per-dataset PNGs).")
    p.add_argument("--no-per-dataset", action="store_true",
                   help="Skip per-dataset dashboards (only hero figure).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nReading from   : {input_root}")
    print(f"Writing figs   : {output_dir}")
    print(f"Headline cell  : alpha={args.headline_alpha} "
          f"miss={args.headline_missing}")

    # Use a non-interactive backend so the script works on headless boxes.
    import matplotlib
    matplotlib.use("Agg")

    if not args.no_per_dataset:
        for dataset, dataset_disp, dataset_short in DATASETS_ORDER:
            if dataset not in args.datasets:
                continue
            try:
                compose_per_dataset_dashboard(
                    input_root, output_dir,
                    dataset, dataset_disp, dataset_short,
                    args.headline_alpha, args.headline_missing)
            except Exception as e:
                print(f"[WARN] per-dataset dashboard for {dataset} failed: {e}",
                      file=sys.stderr)

    if not args.no_hero:
        try:
            compose_hero_figure(input_root, output_dir,
                                args.headline_alpha, args.headline_missing)
        except Exception as e:
            print(f"[WARN] hero figure failed: {e}", file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
