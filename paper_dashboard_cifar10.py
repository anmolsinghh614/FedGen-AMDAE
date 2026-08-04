#!/usr/bin/env python
"""
paper_dashboard_cifar10.py
==========================

Compose paper-ready dashboard figures from a completed CIFAR-10 sweep.
Single-dataset counterpart to `paper_dashboard_run3.py` (there is no
cross-dataset hero figure here; that layout is meaningless for a single
dataset).

Three products per invocation (written under `<output_dir>`, default
`results/cifar10/dashboards/`):

1. Headline dashboard (2x3 grid) for the (alpha=1, miss=10%) cell:

       (0,0) Test accuracy curve, all 5 algorithms (mean +/- std band)
       (0,1) Test loss curve,    all 5 algorithms (mean +/- std band)
       (0,2) Per-class F1 heatmap (rows = algorithms, cols = classes)
       (1,0) FedGen-AMDAE confusion matrix
       (1,1) FedDistill confusion matrix
       (1,2) Mean +/- std final accuracy per algorithm across all 9
             (alpha, missing) cells (bar chart)

   Output: `cifar10_dashboard.png`

2. Accuracy-over-rounds 3x3 grid:
       Rows: alpha in {0.1, 1.0, 10.0}
       Cols: missing rate in {0%, 10%, 20%}
       Each panel = mean +/- std accuracy over rounds for all 5 algorithms.
   Output: `cifar10_accuracy_curves.png`

3. Macro-F1-over-rounds 3x3 grid (same layout, F1 computed per round from
   stored y_true / y_pred dumps).
   Output: `cifar10_f1_curves.png`

Cells that have no completed seeds render as blank panels with
"-- no data --" placeholders so the figure layout stays stable on a
partially-complete sweep.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np  # noqa: F401  -- only for type hints

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


def read_yp_yt(metrics_dir: Path, dataset: str, alpha: float, algo: str
               ) -> Optional[Tuple["np.ndarray", "np.ndarray"]]:
    """Return (y_true, y_pred) for the highest-round HDF5 of this cell,
    aggregated from every seed. Used by confusion / heatmap panels."""
    import h5py
    import numpy as np
    if not metrics_dir.is_dir():
        return None

    token = dataset_token(dataset, alpha)
    seed_dirs = sorted(p for p in metrics_dir.iterdir()
                       if p.is_dir() and p.name.startswith("seed_"))
    if not seed_dirs:
        flat = metrics_dir / token
        seed_dirs = [flat] if flat.is_dir() else []

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


def panel_curve_summary(ax, models_dir_per_algo: Dict[str, Path], key: str,
                        title: str, ylabel: str,
                        percent: bool = True) -> None:
    curves: Dict[str, "np.ndarray"] = {}
    for algo, md in models_dir_per_algo.items():
        arr = read_curves(md, key)
        if arr is None:
            continue
        curves[algo] = arr * (100.0 if percent else 1.0)
    _draw_curve_panel(ax, curves, title, ylabel)


def panel_f1_curve(ax, metrics_dir_per_algo: Dict[str, Path],
                   dataset: str, alpha: float, title: str,
                   legend: bool = True) -> None:
    curves: Dict[str, "np.ndarray"] = {}
    for algo, md in metrics_dir_per_algo.items():
        arr = read_f1_curves(md, dataset, alpha, algo)
        if arr is None:
            continue
        curves[algo] = arr * 100.0
    _draw_curve_panel(ax, curves, title, "Macro F1 (%)", legend=legend)


def panel_per_class_f1_heatmap(ax, dataset: str, alpha: float,
                               metrics_dirs_per_algo: Dict[str, Path],
                               title: str) -> None:
    import numpy as np
    from sklearn.metrics import f1_score
    rows = []
    row_labels = []
    for algo in ALGOS_ORDER:
        md = metrics_dirs_per_algo.get(algo)
        if md is None:
            continue
        result = read_yp_yt(md, dataset, alpha, algo)
        if result is None:
            continue
        yt, yp = result
        labels = sorted(set(np.concatenate([yt, yp]).tolist()))
        f1 = f1_score(yt, yp, average=None, zero_division=0, labels=labels)
        rows.append(np.asarray(f1, dtype=float))
        row_labels.append(ALGO_DISPLAY[algo])
    if not rows:
        ax.set_title(title, fontsize=10)
        _no_data_panel(ax)
        return

    n_classes = max(len(r) for r in rows)
    M = np.full((len(rows), n_classes), np.nan)
    for i, r in enumerate(rows):
        M[i, :len(r)] = r
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(range(n_classes), fontsize=8)
    ax.set_xlabel("Class", fontsize=9)
    ax.set_title(title, fontsize=10)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("F1", fontsize=8)


def panel_confusion_matrix(ax, dataset: str, alpha: float, algo: str,
                           metrics_dir: Optional[Path], title: str) -> None:
    import numpy as np
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
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)


def panel_bar_per_algo(ax, input_root: Path, dataset: str,
                       dataset_short: str, title: str) -> None:
    """Mean +/- std final accuracy per algorithm across all 9 cells."""
    import numpy as np
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
    """2x3 panel PNG at the headline cell."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    models_per_algo: Dict[str, Path] = {}
    metrics_per_algo: Dict[str, Path] = {}
    for algo in ALGOS_ORDER:
        cd = cell_dir(input_root, dataset_short,
                      headline_alpha, headline_miss, algo)
        models_per_algo[algo] = cd / "models"
        metrics_per_algo[algo] = cd / "metrics"

    headline_str = (f"{dataset_disp}, alpha={headline_alpha}, "
                    f"miss={int(headline_miss * 100)}%")

    panel_curve_summary(axes[0, 0], models_per_algo, "glob_acc",
                        title=f"Test accuracy ({headline_str})",
                        ylabel="Test acc (%)", percent=True)
    panel_curve_summary(axes[0, 1], models_per_algo, "glob_loss",
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

    fig.suptitle(f"{dataset_disp}  --  CIFAR-10 dashboard", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = output_dir / f"{dataset_short}_dashboard.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


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
            title = (f"alpha = {alpha}, missing = {int(miss * 100)}%")

            if which == "accuracy":
                models_per_algo: Dict[str, Path] = {}
                for algo in ALGOS_ORDER:
                    cd = cell_dir(input_root, dataset_short, alpha, miss, algo)
                    models_per_algo[algo] = cd / "models"
                curves: Dict[str, "np.ndarray"] = {}
                for algo, md in models_per_algo.items():
                    arr = read_curves(md, "glob_acc")
                    if arr is None:
                        continue
                    curves[algo] = arr * 100.0
                _draw_curve_panel(ax, curves, title, "Accuracy (%)",
                                  legend=(i == 0 and j == 0))
            else:
                metrics_per_algo: Dict[str, Path] = {}
                for algo in ALGOS_ORDER:
                    cd = cell_dir(input_root, dataset_short, alpha, miss, algo)
                    metrics_per_algo[algo] = cd / "metrics"
                panel_f1_curve(ax, metrics_per_algo, dataset, alpha,
                               title=title, legend=(i == 0 and j == 0))

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
    p.add_argument("--input-root", default=str(DEFAULT_INPUT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--headline-alpha", type=float, default=1.0)
    p.add_argument("--headline-missing", type=float, default=0.10)
    p.add_argument("--no-per-dataset", action="store_true",
                   help="Skip per-dataset headline dashboards.")
    p.add_argument("--no-curves", action="store_true",
                   help="Skip the 3x3 accuracy/F1 curve grids.")
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

    import matplotlib
    matplotlib.use("Agg")

    for dataset, dataset_disp, dataset_short in DATASETS_ORDER:
        if not args.no_per_dataset:
            try:
                compose_per_dataset_dashboard(
                    input_root, output_dir,
                    dataset, dataset_disp, dataset_short,
                    args.headline_alpha, args.headline_missing)
            except Exception as e:
                print(f"[WARN] headline dashboard for {dataset} failed: {e}",
                      file=sys.stderr)

        if not args.no_curves:
            try:
                compose_curves_grid(input_root, output_dir,
                                    dataset, dataset_disp, dataset_short,
                                    which="accuracy")
            except Exception as e:
                print(f"[WARN] accuracy-curves grid for {dataset} failed: {e}",
                      file=sys.stderr)
            try:
                compose_curves_grid(input_root, output_dir,
                                    dataset, dataset_disp, dataset_short,
                                    which="f1")
            except Exception as e:
                print(f"[WARN] F1-curves grid for {dataset} failed: {e}",
                      file=sys.stderr)

    print("\nDone.")


if __name__ == "__main__":
    main()
