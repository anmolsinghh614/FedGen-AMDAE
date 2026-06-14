#!/usr/bin/env python
"""
Per-class F1 heatmap plotter (reproduces paper Figs 5-8).

Given a `results/metrics_mr<rate>/<dataset_token>/` directory, this script:
  1. picks the highest available round per algorithm,
  2. computes per-class precision / recall / F1 with sklearn,
  3. renders a heatmap whose rows are algorithms and columns are classes,
     with the cell value = F1 (and an annotation = "F1 (P/R)").

Outputs
-------
results/figures/heatmap_f1/<dataset_token>/heatmap_f1.png
results/figures/heatmap_f1/<dataset_token>/heatmap_f1.csv

Examples
--------
  # Default: scans results/metrics_mr10/<token>/ and writes results/figures/...
  py -3 plot_per_class_f1_heatmap.py --dataset Mnist-alpha0.1-ratio0.5 \
        --input-root results/metrics_mr10 \
        --output-root results/figures/heatmap_f1

  # All in one call (sweep every missing rate / alpha):
  for mr in 10 20; do
    for a in 0.1 1.0 10.0; do
      py -3 plot_per_class_f1_heatmap.py \
          --dataset "Mnist-alpha${a}-ratio0.5" \
          --input-root "results/metrics_mr${mr}" \
          --output-root "results/figures/heatmap_f1/mr${mr}";
    done;
  done
"""
import argparse
import re
from pathlib import Path
import sys

ALGOS_DEFAULT = ["FedAvg", "FedGen", "FedProx", "FedDistill", "FedEnsemble"]

FILE_RE = re.compile(r"^(?P<algo>[^_]+)_(?P<token>.+)_round_(?P<round>\d+)\.h5$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", "-d", required=True,
                   help="Dataset token, e.g. 'Mnist-alpha0.1-ratio0.5'.")
    p.add_argument("--input-root", required=True,
                   help="Where the metrics_mr<rate>/<dataset_token>/ files live.")
    p.add_argument("--output-root", default="results/figures/heatmap_f1",
                   help="Where to write the heatmap PNG + CSV.")
    p.add_argument("--algorithms", "-a", nargs="*", default=None,
                   help="Subset to plot (default: auto-detect, ordered).")
    p.add_argument("--metric", choices=["f1", "precision", "recall"], default="f1",
                   help="Cell value (annotation always shows F1 P/R).")
    p.add_argument("--zero_division", type=int, default=0)
    return p.parse_args()


def latest_round_per_algo(metrics_root: Path, dataset_token: str):
    best = {}
    if not metrics_root.is_dir():
        raise SystemExit(f"Not a directory: {metrics_root}")
    for h5 in metrics_root.glob("*.h5"):
        m = FILE_RE.match(h5.name)
        if not m:
            continue
        if m.group("token") != dataset_token:
            continue
        algo = m.group("algo")
        r = int(m.group("round"))
        prev = best.get(algo)
        if prev is None or prev[0] < r:
            best[algo] = (r, h5)
    return {a: v[1] for a, v in best.items()}


def load_y(path: Path):
    import h5py
    import numpy as np
    true_keys = ["y_true", "test_y", "labels", "targets", "test_targets"]
    pred_keys = ["y_pred", "preds", "test_pred", "test_predictions", "predictions"]
    prob_keys = ["y_prob", "probs", "probabilities", "logits", "outputs"]
    with h5py.File(path, "r") as hf:
        yt_key = next((k for k in true_keys if k in hf), None)
        if yt_key is None:
            return None
        y_true = np.asarray(hf[yt_key][()]).reshape(-1)
        yp_key = next((k for k in pred_keys if k in hf), None)
        if yp_key is not None:
            y_pred = np.asarray(hf[yp_key][()]).reshape(-1)
        else:
            yq_key = next((k for k in prob_keys if k in hf), None)
            if yq_key is None:
                return None
            q = np.asarray(hf[yq_key][()])
            y_pred = (q.reshape(-1) >= 0.5).astype(int) if q.ndim == 1 else q.argmax(axis=-1)
        if y_true.shape[0] != y_pred.shape[0]:
            return None
        return y_true, y_pred


def main() -> None:
    args = parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_fscore_support

    in_dir = Path(args.input_root) / args.dataset
    files = latest_round_per_algo(in_dir, args.dataset)
    if not files:
        raise SystemExit(f"No matching .h5 files under {in_dir}")

    algos = args.algorithms or [a for a in ALGOS_DEFAULT if a in files]
    if not algos:
        algos = sorted(files.keys())

    rows_p, rows_r, rows_f = [], [], []
    common_labels = None
    for algo in algos:
        path = files.get(algo)
        if path is None:
            print(f"[SKIP] {algo}: no h5 file present", file=sys.stderr)
            continue
        yp = load_y(path)
        if yp is None:
            print(f"[SKIP] {algo}: could not extract y_true/y_pred from {path.name}",
                  file=sys.stderr)
            continue
        y_true, y_pred = yp
        labels = np.unique(np.concatenate([y_true, y_pred]))
        if common_labels is None:
            common_labels = labels
        else:
            common_labels = np.unique(np.concatenate([common_labels, labels]))
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels,
            average=None, zero_division=args.zero_division)
        # remap onto the (eventual) common label set later
        rows_p.append((algo, dict(zip(labels.tolist(), p.tolist()))))
        rows_r.append((algo, dict(zip(labels.tolist(), r.tolist()))))
        rows_f.append((algo, dict(zip(labels.tolist(), f.tolist()))))

    if common_labels is None:
        raise SystemExit("No usable algorithm data found.")

    n_algos = len(rows_f)
    n_classes = len(common_labels)
    F = np.zeros((n_algos, n_classes), dtype=float)
    P = np.zeros_like(F)
    R = np.zeros_like(F)
    for i, ((_, fmap), (_, pmap), (_, rmap)) in enumerate(zip(rows_f, rows_p, rows_r)):
        for j, c in enumerate(common_labels):
            F[i, j] = fmap.get(int(c), float("nan"))
            P[i, j] = pmap.get(int(c), float("nan"))
            R[i, j] = rmap.get(int(c), float("nan"))

    metric_grid = {"f1": F, "precision": P, "recall": R}[args.metric]

    # ---- render
    fig_w = max(8, 0.55 * n_classes + 4)
    fig_h = max(3, 0.7 * n_algos + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(metric_grid, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels([str(int(c)) for c in common_labels], rotation=0)
    ax.set_yticks(range(n_algos))
    ax.set_yticklabels([algo for algo, _ in rows_f])
    ax.set_xlabel("Class")
    ax.set_ylabel("Algorithm")
    ax.set_title(f"Per-class {args.metric.upper()} on {args.dataset} "
                 f"(annotation: F1 (P/R))")

    for i in range(n_algos):
        for j in range(n_classes):
            v = metric_grid[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i,
                    f"{F[i, j]:.2f}\n({P[i, j]:.2f}/{R[i, j]:.2f})",
                    ha="center", va="center",
                    fontsize=8,
                    color="white" if v > 0.55 else "black")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(args.metric.upper())
    fig.tight_layout()

    out_dir = Path(args.output_root) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "heatmap_f1.png"
    csv = out_dir / "heatmap_f1.csv"

    fig.savefig(str(png), dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] heatmap -> {png}")

    with open(csv, "w", encoding="utf-8") as f:
        f.write("algorithm,class," "precision,recall,f1\n")
        for i, (algo, _) in enumerate(rows_f):
            for j, c in enumerate(common_labels):
                f.write(f"{algo},{int(c)},{P[i, j]:.6f},"
                        f"{R[i, j]:.6f},{F[i, j]:.6f}\n")
    print(f"[OK] csv     -> {csv}")


if __name__ == "__main__":
    main()
