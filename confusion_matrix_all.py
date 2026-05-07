import argparse
import logging
from pathlib import Path
import re
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
except ImportError:
    raise SystemExit(
        "Missing dependency: scikit-learn is required for confusion matrix. "
        "Install with: pip install scikit-learn"
    )

ALGOS_DEFAULT = ["FedAvg", "FedGen", "FedProx", "FedDistill", "FedEnsemble"]

def make_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("cm_debug")
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

def read_ytrue_ypred_from_h5(path: Path, logger: logging.Logger):
    true_keys = ["y_true", "test_y", "labels", "targets", "test_targets"]
    pred_keys = ["y_pred", "preds", "test_pred", "test_predictions", "predictions"]
    prob_keys = ["y_prob", "probs", "probabilities", "logits", "outputs"]
    try:
        with h5py.File(path, "r") as hf:
            k_ytrue, y_true = first_available(hf, true_keys)
            if y_true is None:
                logger.debug(f"  -> {path.name}: missing y_true in any of: {', '.join(true_keys)}")
                return None, None, "missing"

            y_true = as_label_vector(y_true)

            k_ypred, y_pred = first_available(hf, pred_keys)
            if y_pred is not None:
                y_pred = as_label_vector(y_pred)
                method = f"preds:{k_ypred}"
                logger.debug(f"  -> {path.name}: using y_pred '{k_ypred}' shape={y_pred.shape}")
            else:
                k_yprob, y_prob = first_available(hf, prob_keys)
                if y_prob is None:
                    logger.debug(f"  -> {path.name}: missing y_pred/y_prob in any of: {', '.join(pred_keys + prob_keys)}")
                    return None, None, "missing"
                y_pred = preds_from_prob(y_prob)
                method = f"probs:{k_yprob}"
                logger.debug(f"  -> {path.name}: derived y_pred from '{k_yprob}' shape={y_pred.shape}")

            y_true = y_true.reshape(-1)
            y_pred = y_pred.reshape(-1)

            if y_true.shape[0] != y_pred.shape[0]:
                logger.debug(f"  -> {path.name}: length mismatch y_true={y_true.shape[0]} vs y_pred={y_pred.shape[0]}")
                return None, None, "mismatch"

            return y_true, y_pred, method
    except Exception as e:
        logger.debug(f"  -> {path.name}: failed opening/reading: {e}")
        return None, None, "error"

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

def find_last_round_with_any_files(dataset_dir: Path, algos: list, dataset_token: str, max_rounds: int, logger: logging.Logger) -> int:
    last = 0
    for r in range(1, max_rounds + 1):
        any_files = False
        for algo in algos:
            if find_round_files(dataset_dir, algo, dataset_token, r):
                any_files = True
                break
        if any_files:
            last = r
    return last

def aggregate_confusion_for_round(files, normalize: str, logger: logging.Logger):
    y_true_all = []
    y_pred_all = []
    for f in files:
        yt, yp, how = read_ytrue_ypred_from_h5(f, logger)
        if yt is None or yp is None:
            logger.debug(f"    skip {f.name}: {how}")
            continue
        y_true_all.append(yt)
        y_pred_all.append(yp)
        logger.debug(f"    use {f.name}: n={len(yt)} via {how}")
    if not y_true_all:
        return None, None, None

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)

    labels = np.unique(np.concatenate([y_true_all, y_pred_all], axis=0))
    norm = None if normalize == "none" else normalize
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels, normalize=norm)
    return cm, labels, (len(y_true_all))

def save_cm_csv(cm: np.ndarray, labels: np.ndarray, out_csv: Path, logger: logging.Logger):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("label," + ",".join(map(str, labels)) + "\n")
        for i, row in enumerate(cm):
            f.write(str(labels[i]) + "," + ",".join(map(str, row)) + "\n")
    logger.info(f"Saved: {out_csv}")

def plot_and_save_cm(cm: np.ndarray, labels: np.ndarray, algo: str, round_idx: int,
                     out_png: Path, title_suffix: str, logger: logging.Logger,
                     annotate_thresh: float = 0.01):
    n = len(labels)
    fig_w = min(0.55 * n + 4.8, 24.0)
    fig_h = min(0.55 * n + 4.3, 24.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ann_fs  = 11.0
    tick_fs = max(ann_fs + 2, 11)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(
        ax=ax,
        include_values=True,
        cmap="Blues",
        colorbar=False,
        values_format=".2f",
        xticks_rotation=90,
        im_kw=dict(interpolation="nearest"),
        text_kw=dict(fontsize=ann_fs, style="normal", weight="normal"),
    )

    if disp.text_ is not None:
        cm_float = cm.astype(float)
        for (i, j), t in np.ndenumerate(disp.text_):
            if i != j and cm_float[i, j] < annotate_thresh:
                t.set_visible(False)
            elif i == j:
                t.set_fontweight("bold")
            t.set_style("normal")
            t.set_fontsize(ann_fs)

    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='x', which='both',
                   bottom=False, top=True,
                   labelbottom=False, labeltop=True,
                   labelsize=tick_fs, labelrotation=90, pad=10)
    ax.tick_params(axis="y", labelsize=tick_fs)

    for lbl in ax.get_xticklabels():
        lbl.set_style("normal")
        lbl.set_weight("normal")
        lbl.set_ha("center")
        lbl.set_va("bottom")

    for lbl in ax.get_yticklabels():
        lbl.set_style("normal")
        lbl.set_weight("normal")

    ax.set_xlabel(ax.get_xlabel(), fontsize=tick_fs + 2, style='normal')
    ax.xaxis.labelpad = 12
    ax.set_ylabel(ax.get_ylabel(), fontsize=tick_fs + 2, style='normal')

    fig.tight_layout(pad=2.0)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=250, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_png}")

def main():
    parser = argparse.ArgumentParser(description="Create confusion matrices for the last available round per algorithm.")
    parser.add_argument("--dataset", "-d", required=True, help="Dataset token as used in filenames, e.g., EMnist or EMnist-alpha0.1-ratio0.1")
    parser.add_argument("--rounds", "-r", type=int, default=100, help="Upper bound on rounds to scan (default: 100)")
    parser.add_argument("--algos", "-a", nargs="*", default=None, help="Algorithms to include (default: auto-detect from files)")
    parser.add_argument("--normalize", choices=["none", "true", "pred", "all"], default="true", help="Normalization mode for confusion_matrix (default: true)")
    parser.add_argument("--input-root", default=None, help="Root directory containing results/metrics; default: inferred from script directory")
    parser.add_argument("--output-root", default=None, help="Root directory for results/metrics/eval; default: inferred from script directory")
    parser.add_argument("--no-csv", action="store_true", help="Do not save CSV for the confusion matrix")
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

    last_round = find_last_round_with_any_files(dataset_dir, algos, args.dataset, args.rounds, logger)
    if last_round <= 0:
        logger.error("No round files found up to the specified --rounds")
        raise SystemExit(4)

    logger.info(f"Using last available round: {last_round}")

    for algo in algos:
        files = find_round_files(dataset_dir, algo, args.dataset, last_round)
        if not files:
            logger.warning(f"[{algo}] No files for round={last_round}, skipping")
            continue

        logger.info(f"[{algo}] Aggregating confusion matrix from {len(files)} file(s) at round {last_round}")
        cm, labels, n = aggregate_confusion_for_round(files, args.normalize, logger)
        if cm is None:
            logger.warning(f"[{algo}] No valid y_true/y_pred for round={last_round}, skipping")
            continue

        suffix = f" — n={n}, norm={args.normalize}"
        out_png = out_dir / f"confusion_matrix_round_{last_round}_{algo}.png"
        plot_and_save_cm(cm, labels, algo, last_round, out_png, suffix, logger)

        if not args.no_csv:
            out_csv = out_dir / f"confusion_matrix_round_{last_round}_{algo}.csv"
            save_cm_csv(cm, labels, out_csv, logger)

if __name__ == "__main__":
    main()
