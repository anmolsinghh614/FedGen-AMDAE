#!/usr/bin/env python
"""
paper_table_run3.py
====================

Generate paper-style stacked tables from a completed Run 3 sweep.
Extends the Option A table builder with two extra headline metrics
requested for the medical-imaging sweep: Macro Precision and Macro
Recall (both computable from stored y_true / y_pred without re-running
any experiment).

Produces, in `<output_dir>` (default `results/run3/tables/`):

    accuracy_mnist.{csv,md,tex}          precision_mnist.{csv,md,tex}
    accuracy_fedisic.{csv,md,tex}        precision_fedisic.{csv,md,tex}
    accuracy_ham10000.{csv,md,tex}       precision_ham10000.{csv,md,tex}
    macro_f1_mnist.{csv,md,tex}          recall_mnist.{csv,md,tex}
    macro_f1_fedisic.{csv,md,tex}        recall_fedisic.{csv,md,tex}
    macro_f1_ham10000.{csv,md,tex}       recall_ham10000.{csv,md,tex}
    imputer_ablation.{csv,md,tex}        (when --imputer_ablation is given)

The 12 main tables use the same layout as the Option A tables:
5 algorithm columns, rows grouped by alpha, missing-rate sub-rows.
Cells are mean +/- std across seeds; the winning algorithm per row is
bolded in Markdown and LaTeX.

The imputer-ablation table now reports 4 metric columns (Accuracy,
Macro F1, Precision, Recall) per imputer per dataset.

Usage::

    python paper_table_run3.py                       # all 12 + ablation
    python paper_table_run3.py --metric Accuracy     # subset
    python paper_table_run3.py --metric all --imputer_ablation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "results" / "run3"
DEFAULT_OUTPUT = ROOT / "results" / "run3" / "tables"

ALGOS_ORDER = ["FedGen", "FedAvg", "FedProx", "FedEnsemble", "FedDistill"]
ALGO_DISPLAY = {
    "FedGen": "FedGen",
    "FedAvg": "FedAvg",
    "FedProx": "FedProx",
    "FedEnsemble": "FedEns.",
    "FedDistill": "FedDist.",
}
DATASETS_ORDER = [
    ("Mnist",    "MNIST",    "mnist"),
    ("FedISIC",  "FedISIC",  "fedisic"),
    ("HAM10000", "HAM10000", "ham10000"),
]
ALPHAS = [0.1, 1.0, 10.0]
MISSING_RATES = [0.0, 0.10, 0.20]
HEADLINE_ALPHA = 1.0
HEADLINE_MISS = 0.10
ABLATION_IMPUTERS = [
    ("amdae",  "AM-DAE"),
    ("mean",   "Mean"),
    ("median", "Median"),
    ("zero",   "Zero"),
    ("none",   "No imputation"),
]

# Metric registry: (short-name in main-table filenames, display column
# header for the imputer ablation, sklearn-style aggregator).
METRIC_ORDER = ["Accuracy", "MacroF1", "Precision", "Recall"]
METRIC_LABEL = {
    "Accuracy":   "accuracy",
    "MacroF1":    "macro_f1",
    "Precision":  "precision",
    "Recall":     "recall",
}
METRIC_TITLE = {
    "Accuracy":   "Accuracy (%)",
    "MacroF1":    "Macro F1 (%)",
    "Precision":  "Macro Precision (%)",
    "Recall":     "Macro Recall (%)",
}


# ---------------------------------------------------------------- adapters
def dataset_token(dataset: str, alpha: float, sampling_ratio: float = 0.5) -> str:
    return f"{dataset}-alpha{alpha}-ratio{sampling_ratio}"


def cell_dir(input_root: Path, dataset_short: str, alpha: float,
             miss: float, algo: str, suffix: str = "") -> Path:
    base = input_root / dataset_short / f"alpha{alpha}_miss{miss}" / algo
    if suffix:
        base = base / suffix
    return base


def _round_idx(p: Path) -> int:
    try:
        return int(p.stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


# ---------------------------------------------------------------- HDF5 reads
def read_per_seed_accuracy(models_dir: Path) -> List[float]:
    """One value per per-seed summary HDF5: the FINAL glob_acc."""
    import h5py
    if not models_dir.is_dir():
        return []
    out: List[float] = []
    for h5_path in sorted(models_dir.glob("*.h5")):
        try:
            with h5py.File(h5_path, "r") as hf:
                if "glob_acc" in hf:
                    arr = hf["glob_acc"][:]
                    if len(arr):
                        out.append(float(arr[-1]))
        except (OSError, KeyError) as e:
            print(f"[WARN] could not read accuracy from {h5_path}: {e}",
                  file=sys.stderr)
    return out


def read_per_seed_metric_from_preds(metrics_dir: Path, dataset: str,
                                    alpha: float, algo: str,
                                    metric: str) -> List[float]:
    """One `metric` per `seed_<s>/<TOKEN>/` sub-folder, computed from the
    highest-round per-round HDF5.

    `metric` is one of 'MacroF1', 'Precision', 'Recall'."""
    import h5py
    import numpy as np
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
    )
    if not metrics_dir.is_dir():
        return []

    def _score(yt, yp) -> float:
        if metric == "MacroF1":
            return float(f1_score(yt, yp, average="macro", zero_division=0))
        if metric == "Precision":
            return float(precision_score(yt, yp, average="macro",
                                         zero_division=0))
        if metric == "Recall":
            return float(recall_score(yt, yp, average="macro",
                                      zero_division=0))
        raise ValueError(f"Unknown metric {metric!r}")

    token = dataset_token(dataset, alpha)
    out: List[float] = []
    seed_dirs = sorted(p for p in metrics_dir.iterdir()
                       if p.is_dir() and p.name.startswith("seed_"))
    if not seed_dirs:
        flat = metrics_dir / token
        seed_dirs = [flat] if flat.is_dir() else []

    for sd in seed_dirs:
        token_dir = sd / token if sd.name.startswith("seed_") else sd
        if not token_dir.is_dir():
            continue
        cands = sorted(token_dir.glob(f"{algo}_*round_*.h5"),
                       key=lambda p: _round_idx(p))
        if not cands:
            continue
        last = cands[-1]
        try:
            with h5py.File(last, "r") as hf:
                yt = np.asarray(hf["y_true"][:]).reshape(-1)
                yp = np.asarray(hf["y_pred"][:]).reshape(-1)
            out.append(_score(yt, yp))
        except (OSError, KeyError, ValueError) as e:
            print(f"[WARN] could not compute {metric} from {last}: {e}",
                  file=sys.stderr)
    return out


# ---------------------------------------------------------------- aggregation
def collect_cell_values(input_root: Path, dataset: str, dataset_short: str,
                        alpha: float, miss: float, algo: str, metric: str,
                        suffix: str = "") -> List[float]:
    base = cell_dir(input_root, dataset_short, alpha, miss, algo, suffix=suffix)
    if metric == "Accuracy":
        return read_per_seed_accuracy(base / "models")
    if metric in ("MacroF1", "Precision", "Recall"):
        return read_per_seed_metric_from_preds(base / "metrics", dataset,
                                               alpha, algo, metric)
    raise ValueError(f"Unknown metric: {metric}")


def mean_std_pct(values: List[float], pct: bool = True) -> Tuple[Optional[float], Optional[float], int]:
    import numpy as np
    if not values:
        return None, None, 0
    arr = np.asarray(values, dtype=float)
    if pct:
        arr = arr * 100.0
    mu = float(arr.mean())
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mu, sd, len(arr)


def fmt_cell(mu: Optional[float], sd: Optional[float], n: int) -> str:
    if mu is None:
        return "--"
    if n <= 1:
        return f"{mu:.2f}"
    return f"{mu:.2f} +/- {sd:.2f}"


# ---------------------------------------------------------------- table build
def build_main_table(input_root: Path, dataset: str, dataset_short: str,
                     metric: str) -> "list[dict]":
    rows: list[dict] = []

    rows.append({
        "kind": "header",
        "cells": ["Setting"] + [ALGO_DISPLAY[a] for a in ALGOS_ORDER],
    })

    for alpha in ALPHAS:
        rows.append({
            "kind": "block",
            "cells": [f"Heterogeneity: alpha = {alpha}"]
                     + [""] * len(ALGOS_ORDER),
        })
        for miss in MISSING_RATES:
            stats = []
            for algo in ALGOS_ORDER:
                vals = collect_cell_values(input_root, dataset, dataset_short,
                                           alpha, miss, algo, metric)
                stats.append(mean_std_pct(vals, pct=True))

            mus = [s[0] if s[0] is not None else float("-inf") for s in stats]
            best_idx = max(range(len(mus)), key=lambda i: mus[i]) \
                if any(s[0] is not None for s in stats) else None

            cells = []
            for idx, (mu, sd, n) in enumerate(stats):
                cells.append({
                    "text": fmt_cell(mu, sd, n),
                    "is_winner": (idx == best_idx and mu is not None),
                })

            rows.append({
                "kind": "data",
                "label": f"Miss. {int(miss * 100)}%",
                "cells": cells,
            })
    return rows


def build_imputer_ablation(input_root: Path) -> "list[dict]":
    """3 dataset blocks, each showing FedGen at the headline cell with
    each of the 5 imputers. 4 metric columns:
    Accuracy, Macro F1, Precision, Recall."""
    rows: list[dict] = []

    rows.append({
        "kind": "header",
        "cells": ["Imputer", "Accuracy (%)", "Macro F1 (%)",
                  "Precision (%)", "Recall (%)"],
    })

    for dataset, dataset_disp, dataset_short in DATASETS_ORDER:
        rows.append({
            "kind": "block",
            "cells": [f"Dataset: {dataset_disp}  "
                      f"(FedGen, alpha={HEADLINE_ALPHA}, miss="
                      f"{int(HEADLINE_MISS * 100)}%)",
                      "", "", "", ""],
        })

        per_imputer: List[Tuple[Tuple, Tuple, Tuple, Tuple]] = []
        for impid, _disp in ABLATION_IMPUTERS:
            suffix = f"imputer_ablation/{impid}"
            stats = tuple(
                mean_std_pct(
                    collect_cell_values(input_root, dataset, dataset_short,
                                        HEADLINE_ALPHA, HEADLINE_MISS,
                                        "FedGen", metric, suffix=suffix))
                for metric in METRIC_ORDER
            )
            per_imputer.append(stats)  # type: ignore[arg-type]

        # Per-metric winner index (None if no seeds anywhere).
        best_idx: List[Optional[int]] = []
        for col in range(len(METRIC_ORDER)):
            mus = [p[col][0] if p[col][0] is not None else float("-inf")
                   for p in per_imputer]
            if any(p[col][0] is not None for p in per_imputer):
                best_idx.append(
                    max(range(len(per_imputer)), key=lambda i: mus[i]))
            else:
                best_idx.append(None)

        for idx, (impid, disp) in enumerate(ABLATION_IMPUTERS):
            cells = []
            for col in range(len(METRIC_ORDER)):
                mu, sd, n = per_imputer[idx][col]
                cells.append({
                    "text": fmt_cell(mu, sd, n),
                    "is_winner": (best_idx[col] == idx and mu is not None),
                })
            rows.append({
                "kind": "data",
                "label": disp,
                "cells": cells,
            })
    return rows


# ---------------------------------------------------------------- writers
def _bold_md(s: str) -> str:
    return f"**{s}**"


def _bold_tex(s: str) -> str:
    return r"\textbf{" + s + "}"


def write_csv(rows: list, path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            if r["kind"] in ("header", "block"):
                w.writerow(r["cells"])
            else:
                line = [r["label"]] + [c["text"] for c in r["cells"]]
                w.writerow(line)


def write_markdown(rows: list, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"## {title}", ""]
    header_row = next((r for r in rows if r["kind"] == "header"), None)
    if header_row is None:
        return
    n_cols = len(header_row["cells"])
    lines.append("| " + " | ".join(header_row["cells"]) + " |")
    lines.append("|" + "|".join(["---"] * n_cols) + "|")

    for r in rows:
        if r["kind"] == "header":
            continue
        if r["kind"] == "block":
            lines.append("| " + " | ".join(
                [f"**{r['cells'][0]}**"] + [""] * (n_cols - 1)) + " |")
        else:
            cells = []
            for c in r["cells"]:
                t = c["text"]
                if c.get("is_winner"):
                    t = _bold_md(t)
                cells.append(t)
            lines.append("| " + " | ".join([r["label"]] + cells) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(rows: list, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header_row = next((r for r in rows if r["kind"] == "header"), None)
    if header_row is None:
        return
    n_cols = len(header_row["cells"])
    col_spec = "l" + "c" * (n_cols - 1)
    out = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + title.replace("&", r"\&") + ".}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        " & ".join(_tex_escape(c) for c in header_row["cells"]) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        if r["kind"] == "header":
            continue
        if r["kind"] == "block":
            out.append(r"\multicolumn{" + str(n_cols) + r"}{l}{\textit{"
                       + _tex_escape(r["cells"][0]) + r"}} \\")
        else:
            cells = []
            for c in r["cells"]:
                t = _tex_escape(c["text"])
                if c.get("is_winner"):
                    t = _bold_tex(t)
                cells.append(t)
            out.append(" & ".join([_tex_escape(r["label"])] + cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _tex_escape(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("+/-", r"$\pm$")
    s = s.replace("%", r"\%")
    s = s.replace("&", r"\&")
    return s


# ---------------------------------------------------------------- driver
def emit_main_tables(input_root: Path, output_dir: Path,
                     metrics: List[str]) -> None:
    for metric in metrics:
        for dataset, dataset_disp, dataset_short in DATASETS_ORDER:
            rows = build_main_table(input_root, dataset, dataset_short, metric)
            stem = f"{METRIC_LABEL[metric]}_{dataset_short}"
            title = (f"Performance comparison on {dataset_disp}: "
                     f"{METRIC_TITLE[metric]} (mean +/- std).")
            write_csv(rows, output_dir / f"{stem}.csv")
            write_markdown(rows, output_dir / f"{stem}.md", title)
            write_latex(rows, output_dir / f"{stem}.tex", title)
            print(f"  wrote {stem}.csv / .md / .tex  -> {output_dir}")


def emit_imputer_ablation_table(input_root: Path, output_dir: Path) -> None:
    rows = build_imputer_ablation(input_root)
    stem = "imputer_ablation"
    title = (f"Imputer ablation -- FedGen at headline cell "
             f"(alpha={HEADLINE_ALPHA}, miss={int(HEADLINE_MISS * 100)}%) "
             f"per dataset.")
    write_csv(rows, output_dir / f"{stem}.csv")
    write_markdown(rows, output_dir / f"{stem}.md", title)
    write_latex(rows, output_dir / f"{stem}.tex", title)
    print(f"  wrote {stem}.csv / .md / .tex  -> {output_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-root", default=str(DEFAULT_INPUT),
                   help=f"Root of the Run 3 sweep results (default {DEFAULT_INPUT}).")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT),
                   help=f"Where to write tables (default {DEFAULT_OUTPUT}).")
    p.add_argument("--metric",
                   choices=METRIC_ORDER + ["all"], default="all",
                   help="Which main table(s) to produce. Default 'all' "
                        "produces 12 main tables (3 datasets x 4 metrics).")
    p.add_argument("--imputer_ablation", action="store_true",
                   help="Additionally produce results/run3/tables/"
                        "imputer_ablation.{csv,md,tex}.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = METRIC_ORDER if args.metric == "all" else [args.metric]

    print(f"\nReading from   : {input_root}")
    print(f"Writing tables : {output_dir}")
    print(f"Main metrics   : {metrics}")
    print(f"Imputer ablation: {args.imputer_ablation}\n")

    emit_main_tables(input_root, output_dir, metrics)
    if args.imputer_ablation:
        emit_imputer_ablation_table(input_root, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
