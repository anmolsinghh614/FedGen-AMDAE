#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
import re
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler

try:
    from sklearn.metrics import f1_score
except ImportError as e:
    raise SystemExit(
        "Missing dependency: scikit-learn is required for F1 computation. "
        "Install with: pip install scikit-learn"
    )

ALGOS_DEFAULT = ["FedAvg", "FedGen", "FedProx", "FedDistill", "FedEnsemble"]

def make_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("f1_debug")
    if not logger.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger

def walk_datasets(h: h5py.File):
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append((name, obj))
    datasets = []
    h.visititems(lambda n, o: visitor(n, o))
    return datasets

def read_array(hf: h5py.File, key: str):
    try:
        return np.array(hf[key][...])
    except Exception:
        return None

def first_available(hf: h5py.File, keys):
    for k in keys:
        if k in hf:
            arr = read_array(hf, k)
            if arr is not None:
                return k, arr
    return None, None

def as_label_vector(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 0:
        return a.reshape(1)
    if a.ndim == 1:
        return a
    if a.ndim == 2 and a.shape[1] > 1:
        row_sums = a.sum(axis=1)
        if np.all((row_sums == 1) | (row_sums == 0)):
            return np.argmax(a, axis=1)
        return np.argmax(a, axis=1)
    return a.reshape(-1)

def preds_from_prob(prob: np.ndarray) -> np.ndarray:
    p = np.asarray(prob)
    if p.ndim == 1:
        return (p >= 0.5).astype(int)
    if p.ndim >= 2:
        return np.argmax(p, axis=-1)
    return p.reshape(-1)

def compute_f1(y_true: np.ndarray, y_pred: np.ndarray, average: str) -> float:
    try:
        return float(f1_score(y_true, y_pred, average=average))
    except Exception:
        return float("nan")

def compute_f1_from_metrics_aliases(hf: h5py.File, average: str, logger: logging.Logger):
    true_keys = ["y_true", "test_y", "labels", "targets", "test_targets"]
    pred_keys = ["y_pred", "preds", "test_pred", "test_predictions", "predictions"]
    prob_keys = ["y_prob", "probs", "probabilities", "logits", "outputs"]
    k_ytrue, y_true = first_available(hf, true_keys)
    if y_true is None:
        logger.debug("    missing y_true in any of: " + ", ".join(true_keys))
        return float("nan"), "missing", None
    y_true = as_label_vector(y_true)
    k_ypred, y_pred = first_available(hf, pred_keys)
    if y_pred is not None:
        y_pred = as_label_vector(y_pred)
        logger.debug(f"    using predictions from '{k_ypred}' with shape={y_pred.shape}")
    else:
        k_yprob, y_prob = first_available(hf, prob_keys)
        if y_prob is None:
            logger.debug("    missing y_pred and y_prob in any of: " + ", ".join(pred_keys + prob_keys))
            return float("nan"), "missing", None
        y_pred = preds_from_prob(y_prob)
        logger.debug(f"    derived predictions from '{k_yprob}' via argmax/threshold with shape={y_pred.shape}")
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        logger.debug(f"    length mismatch y_true={y_true.shape[0]} vs y_pred={y_pred.shape[0]}")
        return float("nan"), "missing", None
    val = compute_f1(y_true, y_pred, average=average)
    return val, "derived-ytrue-ypred", (k_ypred if k_ypred else k_ytrue)

def extract_scalar(arr) -> float:
    a = np.array(arr)
    if a.ndim == 0:
        return float(a)
    if a.size == 0:
        return float("nan")
    return float(a.reshape(-1)[-1])

def read_f1_from_h5(path: Path, logger: logging.Logger, average: str):
    try:
        with h5py.File(path, "r") as hf:
            dsets = walk_datasets(hf)
            keys = [k for k, _ in dsets]
            logger.debug(f"  -> keys={keys[:10]}{' ...' if len(keys) > 10 else ''}")
            f1_candidates = [k for k in keys if "f1" in k.lower()]
            if f1_candidates:
                def score(k):
                    kl = k.lower()
                    s = 0
                    if "glob" in kl or "global" in kl:
                        s += 3
                    if "test" in kl or "val" in kl or "eval" in kl:
                        s += 2
                    if kl == "f1" or kl.endswith("/f1"):
                        s += 2
                    if "per" in kl or "personal" in kl:
                        s += 1
                    return s
                f1_candidates.sort(key=score, reverse=True)
                chosen = f1_candidates[0]
                try:
                    val = extract_scalar(hf[chosen][...])
                    logger.debug(f"  -> f1 source: dataset '{chosen}' = {val}")
                    return val, "f1-dataset", chosen
                except Exception as e:
                    logger.debug(f"  -> failed reading f1 '{chosen}': {e}")
            val, method, key = compute_f1_from_metrics_aliases(hf, average, logger)
            if not np.isnan(val):
                logger.debug(f"  -> f1 source: derived from saved metrics => {val}")
                return val, method, key
    except Exception as e:
        logger.debug(f"  -> failed opening/reading: {e}")
    logger.debug("  -> f1 source: missing")
    return float("nan"), "missing", None

def find_algorithms(dataset_dir: Path, dataset_token: str, logger: logging.Logger):
    algos = set()
    for p in list(dataset_dir.glob(f"*_{dataset_token}*_round_*.[hH][5dD][5]")) + \
             list(dataset_dir.glob(f"*_{dataset_token}*_round_*.[hH]5")) + \
             list(dataset_dir.glob(f"*_{dataset_token}*_round_*.[hH][dD]5")):
        base = p.name
        m = re.match(rf"^(?P<algo>[^_]+)_{re.escape(dataset_token)}.*_round_(\d+)\.(h5|hd5)$", base, flags=re.IGNORECASE)
        if m:
            algos.add(m.group("algo"))
    algos = sorted(algos)
    logger.info(f"Detected algorithms: {algos}")
    return algos

def find_round_files(dataset_dir: Path, algo: str, dataset_token: str, round_idx: int):
    patterns = [
        f"{algo}_{dataset_token}*_round_{round_idx}.h5",
        f"{algo}_{dataset_token}*_round_{round_idx}.hd5",
        f"{algo}_{dataset_token}*round_{round_idx}.h5",
        f"{algo}_{dataset_token}*round_{round_idx}.hd5",
    ]
    files = []
    for pat in patterns:
        files.extend(dataset_dir.glob(pat))
    files = sorted(set(files), key=lambda p: p.name)
    return files

def build_series_for_algo(dataset_dir: Path, dataset_token: str, algo: str, n_rounds: int, average: str, logger: logging.Logger):
    y = []
    counts = {"rounds_with_files": 0, "non_nan_points": 0, "total_files": 0}
    for r in range(1, n_rounds + 1):
        files = find_round_files(dataset_dir, algo, dataset_token, r)
        if not files:
            logger.debug(f"[{algo}] round={r}: 0 files matched")
            y.append(float("nan"))
            continue
        counts["rounds_with_files"] += 1
        counts["total_files"] += len(files)
        logger.debug(f"[{algo}] round={r}: {len(files)} file(s): {[f.name for f in files][:5]}{' ...' if len(files) > 5 else ''}")
        vals = []
        for f in files:
            v, method, key = read_f1_from_h5(f, logger, average)
            if not np.isnan(v):
                vals.append(v)
            logger.debug(f"    file={f.name} -> f1={v} via {method}{' [' + str(key) + ']' if key else ''}")
        if vals:
            m = float(np.mean(vals))
            y.append(m)
            counts["non_nan_points"] += 1
            logger.debug(f"    round={r}: mean_f1={m} from {len(vals)}/{len(files)} valid files")
        else:
            y.append(float("nan"))
            logger.debug(f"    round={r}: all NaN across {len(files)} files")
    return np.array(y, dtype=float), counts

def _repeat_to_length(seq, L):
    if len(seq) == 0:
        return seq
    reps = int(np.ceil(L / len(seq)))
    return (seq * reps)[:L]

def plot_series(series_map: dict, n_rounds: int, out_png: Path, dataset_token: str, average: str, logger: logging.Logger):
    x = np.arange(1, n_rounds + 1)
    plt.style.use("seaborn-v0_8-darkgrid")

    colors = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("Dark2").colors)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_prop_cycle(cycler(color=colors))

    lines_plotted = 0
    for algo, y in series_map.items():
        if np.all(np.isnan(y)):
            logger.warning(f"[plot] Skipping '{algo}' — no valid F1 points")
            continue

        mask = np.isfinite(y)
        x_plot = x[mask]
        y_plot = (y[mask] * 100.0)

        ax.plot(
            x_plot,
            y_plot,
            linestyle='-',       # force solid line
            marker=None,         # no dot markers
            linewidth=3.0,
            alpha=1.0,
            solid_capstyle='round',
            zorder=2,
            label=algo,
        )
        lines_plotted += 1

    ax.set_xlabel("Rounds")
    ax.set_ylabel("F1 Score(%)")
    ax.set_xlim(1, n_rounds)
    ax.set_ylim(5, 100)
    ax.set_yticks(np.arange(5, 101, 5))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6, zorder=0)

    if lines_plotted > 0:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, framealpha=0.98,
                  facecolor="white", edgecolor="#444444", title="Algorithm")
    else:
        logger.error("[plot] No lines plotted — verify file matching and F1 metrics")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=220, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_png}")

def save_csv(series_map: dict, n_rounds: int, out_csv: Path, logger: logging.Logger):
    algos = list(series_map.keys())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("round," + ",".join(algos) + "\n")
        for r in range(1, n_rounds + 1):
            row = [str(r)] + [str(series_map[a][r - 1]) for a in algos]
            f.write(",".join(row) + "\n")
    logger.info(f"Saved: {out_csv}")

def main():
    parser = argparse.ArgumentParser(description="Plot F1 across rounds for each algorithm from saved y_true/y_pred/y_prob.")
    parser.add_argument("--dataset", "-d", required=True, help="Dataset token as used in filenames, e.g., EMnist or EMnist-alpha0.1-ratio0.1")
    parser.add_argument("--rounds", "-r", type=int, default=100, help="Number of rounds to scan (default: 100)")
    parser.add_argument("--algos", "-a", nargs="*", default=None, help="Algorithms to include (default: auto-detect from files)")
    parser.add_argument("--avg", choices=["macro", "micro", "weighted"], default="macro", help="F1 averaging: macro|micro|weighted (default: macro)")
    parser.add_argument("--input-root", default=None, help="Root directory containing results/metrics; default: inferred from script directory")
    parser.add_argument("--output-root", default=None, help="Root directory for results/metrics/eval; default: inferred from script directory")
    parser.add_argument("--no-csv", action="store_true", help="Do not save CSV next to the PNG")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug logs")
    args = parser.parse_args()
    logger = make_logger(args.verbose)
    script_dir = Path(__file__).resolve().parent
    default_input_root = script_dir / "results" / "metrics"
    default_output_root = script_dir / "results" / "metrics" / "eval"
    input_root = Path(args.input_root) if args.input_root else default_input_root
    output_root = Path(args.output_root) if args.output_root else default_output_root
    dataset_dir = input_root / args.dataset
    out_dir = output_root / args.dataset
    out_png = out_dir / "f1_by_round.png"
    out_csv = out_dir / "f1_by_round.csv"
    logger.info(f"script_dir={script_dir}")
    logger.info(f"cwd={Path.cwd()}")
    logger.info(f"input_root={input_root}")
    logger.info(f"output_root={output_root}")
    logger.info(f"dataset_dir={dataset_dir}")
    if not dataset_dir.is_dir():
        logger.error(f"Missing dataset directory: {dataset_dir}")
        raise SystemExit(2)
    algos = args.algos if args.algos else find_algorithms(dataset_dir, args.dataset, logger)
    if not algos:
        logger.error("No algorithms detected — check dataset token and file naming pattern")
        raise SystemExit(3)
    series_map = {}
    for algo in algos:
        logger.info(f"Collecting series for algo='{algo}' across rounds=1..{args.rounds} (avg={args.avg})")
        y, counts = build_series_for_algo(dataset_dir, args.dataset, algo, args.rounds, args.avg, logger)
        logger.info(
            f"Summary '{algo}': rounds_with_files={counts['rounds_with_files']}, "
            f"non_nan_points={counts['non_nan_points']}, total_files={counts['total_files']}"
        )
        series_map[algo] = y
    plot_series(series_map, args.rounds, out_png, args.dataset, args.avg, logger)
    if not args.no_csv:
        save_csv(series_map, args.rounds, out_csv, logger)

if __name__ == "__main__":
    main()
