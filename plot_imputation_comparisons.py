import argparse
import logging
from pathlib import Path
import sys
import numpy as np
import h5py
import re
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import seaborn as sns

plt.rcParams.update({"font.size": 14})

COLORS = list(mcolors.TABLEAU_COLORS.values())

WANTED_KEYS = [
    "glob_acc",          # accuracy per round (0..1)
    "glob_loss",         # loss per round
    "per_acc",           # may be empty
    "per_loss",          # may be empty
    "server_agg_time",   # per-round server aggregation time
    "user_train_time",   # per-round user training time
]

EMBED_RE = re.compile(r'embed(\d+)', re.IGNORECASE)

def _embed_percent_from_name(name: str):
    m = EMBED_RE.search(name)
    if not m:
        return None
    return int(m.group(1)) * 10  # embed00 -> 0, embed01 -> 10, ...

def pretty_missing_label(name: str):
    pct = _embed_percent_from_name(name)
    return f"{pct}% missing" if pct is not None else re.sub(r"\.h5$", "", name)

def _sort_key(p: Path):
    pct = _embed_percent_from_name(p.name)
    return (pct if pct is not None else 999, p.name)

def make_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("impute_plot")
    if not logger.handlers:
        h = logging.StreamHandler(stream=sys.stdout)
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger

def list_datasets(hf: h5py.File):
    items = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            items.append((name, obj.shape, str(obj.dtype)))
    hf.visititems(lambda n, o: visitor(n, o))  # walk tree and collect datasets
    return items

def read_array_if_exists(hf: h5py.File, key: str):
    if key in hf:
        try:
            return np.array(hf[key][...])
        except Exception:
            return None
    return None

def load_file(h5_path: Path, logger: logging.Logger):
    with h5py.File(h5_path, "r") as hf:
        # Print all datasets for debugging
        logger.info(f"[{h5_path.name}] DATASETS:")
        for name, shape, dtype in list_datasets(hf):
            logger.info(f"  - {name}  shape={shape}  dtype={dtype}")

        data = {}
        for k in WANTED_KEYS:
            arr = read_array_if_exists(hf, k)
            if arr is not None:
                data[k] = np.asarray(arr)
        return data

def summarize_curves(curves: dict):
    # curves: key -> np.ndarray
    out = {}
    # Accuracy summary
    if "glob_acc" in curves and curves["glob_acc"].size > 0:
        a = np.asarray(curves["glob_acc"], dtype=float)
        out["acc_last"] = float(a[-1])
        out["acc_best"] = float(np.max(a))
        out["acc_best_round"] = int(np.argmax(a))
        last_k = min(10, len(a))
        out["acc_mean_lastk"] = float(np.mean(a[-last_k:]))
        # normalized AUC (mean over rounds equals trapezoid area / length for unit spacing)
        out["acc_auc_norm"] = float(np.trapz(a, dx=1.0) / max(1, len(a)))
        out["acc_std"] = float(np.std(a))
    # Loss summary
    if "glob_loss" in curves and curves["glob_loss"].size > 0:
        l = np.asarray(curves["glob_loss"], dtype=float)
        out["loss_last"] = float(l[-1])
        out["loss_min"] = float(np.min(l))
        out["loss_min_round"] = int(np.argmin(l))
        last_k = min(10, len(l))
        out["loss_mean_lastk"] = float(np.mean(l[-last_k:]))
        out["loss_std"] = float(np.std(l))
    # Time summary
    ut = curves.get("user_train_time", None)
    st = curves.get("server_agg_time", None)
    if ut is not None and ut.size > 0:
        out["user_time_total"] = float(np.sum(ut))
        out["user_time_mean"] = float(np.mean(ut))
    if st is not None and st.size > 0:
        out["server_time_total"] = float(np.sum(st))
        out["server_time_mean"] = float(np.mean(st))
    if ut is not None and ut.size > 0 and st is not None and st.size > 0:
        tot = ut + st
        out["round_time_mean"] = float(np.mean(tot))
        out["round_time_total"] = float(np.sum(tot))
    return out

def save_csv(header, rows, out_csv: Path, logger: logging.Logger):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    logger.info(f"Saved CSV: {out_csv}")

def plot_lines(curves_by_file, key, ylabel, out_png: Path, percent=False, logger=None):
    if not curves_by_file:
        return
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, arr) in enumerate(curves_by_file.items()):
        y = np.asarray(arr).astype(float).ravel()
        x = np.arange(len(y))
        color = COLORS[i % len(COLORS)]
        # Plot each file separately (one line per file)
        ax.plot(x, y, color=color, label=label, linewidth=1.8)
    ax.set_xlabel("Rounds")
    ax.set_ylabel(ylabel)
    if percent:
        ax.set_ylim(0.0, min(1.0, max([float(np.max(v)) for v in curves_by_file.values()] + [1.0]) + 0.02))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_yticks(np.arange(0, 1.01, 0.05))
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=350, bbox_inches="tight")
    plt.close(fig)
    if logger:
        logger.info(f"Saved plot: {out_png}")

def plot_bars(values_by_file, title, ylabel, out_png: Path, logger=None):
    labels = list(values_by_file.keys())
    vals = [float(values_by_file[k]) for k in labels]
    fig_w = min(1.0 * max(8, len(labels)), 28)
    fig, ax = plt.subplots(figsize=(fig_w, 5.2))
    ax.bar(np.arange(len(labels)), vals, color="#4C78A8")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=10)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=350, bbox_inches="tight")
    plt.close(fig)
    if logger:
        logger.info(f"Saved plot: {out_png}")

def main():
    parser = argparse.ArgumentParser(description="Plot and compare metrics stored in imputation .h5 files.")
    parser.add_argument("--input-dir", type=str, default="imputation_comparisons", help="Directory containing .h5 files.")
    parser.add_argument("--pattern", type=str, default="*.h5", help="Glob pattern for files (default: *.h5).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logs.")
    parser.add_argument("--percent-acc", action="store_true", help="Format accuracy axis as percent (0..1 to 0..100%).")
    args = parser.parse_args()

    logger = make_logger(args.verbose)

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        logger.error(f"Missing input directory: {input_dir}")
        sys.exit(2)

    files = sorted(input_dir.glob(args.pattern), key=_sort_key)
    logger.info(f"Found {len(files)} files in {input_dir}")
    if not files:
        logger.error("No files to process.")
        sys.exit(3)

    # Collect curves and summaries
    acc_curves = {}
    loss_curves = {}
    user_time = {}
    server_time = {}
    summary_rows = []
    header = [
        "file",
        "rounds",
        "acc_last",
        "acc_best",
        "acc_best_round",
        "acc_mean_lastk",
        "acc_auc_norm",
        "acc_std",
        "loss_last",
        "loss_min",
        "loss_min_round",
        "loss_mean_lastk",
        "loss_std",
        "user_time_total",
        "user_time_mean",
        "server_time_total",
        "server_time_mean",
        "round_time_mean",
        "round_time_total",
    ]

    for f in files:
        logger.info(f"Processing: {f.name}")
        curves = load_file(f, logger)

        label = pretty_missing_label(f.name)  # <<< use percentage label

        if "glob_acc" in curves and curves["glob_acc"].size > 0:
            acc_curves[label] = curves["glob_acc"]

        if "glob_loss" in curves and curves["glob_loss"].size > 0:
            loss_curves[label] = curves["glob_loss"]

        if "user_train_time" in curves and curves["user_train_time"].size > 0:
            user_time[label] = curves["user_train_time"]

        if "server_agg_time" in curves and curves["server_agg_time"].size > 0:
            server_time[label] = curves["server_agg_time"]


        # Summaries
        s = summarize_curves(curves)
        rounds = 0
        if "glob_acc" in curves and curves["glob_acc"].size > 0:
            rounds = int(len(curves["glob_acc"]))
        elif "glob_loss" in curves and curves["glob_loss"].size > 0:
            rounds = int(len(curves["glob_loss"]))
        row = [
            f.name,
            rounds,
            s.get("acc_last", ""),
            s.get("acc_best", ""),
            s.get("acc_best_round", ""),
            s.get("acc_mean_lastk", ""),
            s.get("acc_auc_norm", ""),
            s.get("acc_std", ""),
            s.get("loss_last", ""),
            s.get("loss_min", ""),
            s.get("loss_min_round", ""),
            s.get("loss_mean_lastk", ""),
            s.get("loss_std", ""),
            s.get("user_time_total", ""),
            s.get("user_time_mean", ""),
            s.get("server_time_total", ""),
            s.get("server_time_mean", ""),
            s.get("round_time_mean", ""),
            s.get("round_time_total", ""),
        ]
        summary_rows.append(row)

    # Save summary CSV in the same directory
    out_csv = input_dir / "imputation_metrics_summary.csv"
    save_csv(header, summary_rows, out_csv, logger)

    # Plots in the same directory
    if acc_curves:
        plot_lines(
            acc_curves,
            key="glob_acc",
            ylabel="Accuracy",
            out_png=input_dir / "accuracy_lines.png",
            percent=args.percent_acc or True,
            logger=logger,
        )
        # Compare last and best accuracy as bars
        last_vals = {k: float(v[-1]) for k, v in acc_curves.items() if len(v) > 0}
        best_vals = {k: float(np.max(v)) for k, v in acc_curves.items() if len(v) > 0}
        plot_bars(last_vals, "Last-round accuracy", "Accuracy", input_dir / "accuracy_last_bar.png", logger)
        plot_bars(best_vals, "Best accuracy", "Accuracy", input_dir / "accuracy_best_bar.png", logger)

    if loss_curves:
        plot_lines(
            loss_curves,
            key="glob_loss",
            ylabel="Loss",
            out_png=input_dir / "loss_lines.png",
            percent=False,
            logger=logger,
        )

    if user_time:
        plot_lines(
            user_time,
            key="user_train_time",
            ylabel="User train time",
            out_png=input_dir / "user_train_time_lines.png",
            percent=False,
            logger=logger,
        )
    if server_time:
        plot_lines(
            server_time,
            key="server_agg_time",
            ylabel="Server agg time",
            out_png=input_dir / "server_agg_time_lines.png",
            percent=False,
            logger=logger,
        )

if __name__ == "__main__":
    main()
