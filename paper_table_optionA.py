#!/usr/bin/env python
"""
paper_table_optionA.py
======================

Generate paper-style stacked tables from a completed Option-A sweep.

Produces, in `<output_dir>` (default `results/optionA/tables/`):

    accuracy_emnist.{csv,md,tex}
    accuracy_ucihar.{csv,md,tex}
    accuracy_pamap2.{csv,md,tex}
    macro_f1_emnist.{csv,md,tex}
    macro_f1_ucihar.{csv,md,tex}
    macro_f1_pamap2.{csv,md,tex}
    imputer_ablation.{csv,md,tex}     (when --imputer_ablation is given)

The 6 main tables follow the layout of the paper's MNIST/EMNIST table:
  * 5 algorithm columns (FedGen, FedAvg, FedProx, FedEnsemble, FedDistill).
  * Rows grouped by alpha; each alpha block has a 0%/10%/20% missing
    sub-row.
  * Cells contain mean +/- std (computed across the seeds present in the
    sweep). Stage-1 cells show only mean (one seed = std undefined).
  * The winning algorithm per row is **bolded** in Markdown and LaTeX.

The imputer-ablation table is a single document with 3 dataset blocks,
each showing FedGen at the headline cell (alpha=1, missing=10%) trained
with each of the 5 imputer choices (AM-DAE, Mean, Median, Zero, no
imputation), reporting Accuracy and Macro-F1.

Usage::

    python paper_table_optionA.py --metric Accuracy
    python paper_table_optionA.py --metric MacroF1
    python paper_table_optionA.py --imputer_ablation
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "results" / "optionA"
DEFAULT_OUTPUT = ROOT / "results" / "optionA" / "tables"

ALGOS_ORDER = ["FedGen", "FedAvg", "FedProx", "FedEnsemble", "FedDistill"]
ALGO_DISPLAY = {
    "FedGen": "FedGen",
    "FedAvg": "FedAvg",
    "FedProx": "FedProx",
    "FedEnsemble": "FedEns.",
    "FedDistill": "FedDist.",
}
DATASETS_ORDER = [
    ("EMnist-letters", "EMNIST", "emnist"),
    ("UCI HAR",        "UCI HAR", "ucihar"),
    ("PAMAP2",         "PAMAP2",  "pamap2"),
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


# ---------------------------------------------------------------- adapters
def dataset_token(dataset: str, alpha: float, sampling_ratio: float = 0.5) -> str:
    return f"{dataset}-alpha{alpha}-ratio{sampling_ratio}"


def cell_dir(input_root: Path, dataset_short: str, alpha: float,
             miss: float, algo: str, suffix: str = "") -> Path:
    base = input_root / dataset_short / f"alpha{alpha}_miss{miss}" / algo
    if suffix:
        base = base / suffix
    return base


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


def read_per_seed_macro_f1(metrics_dir: Path, dataset: str, alpha: float,
                           algo: str) -> List[float]:
    """One Macro-F1 per `seed_<s>/<TOKEN>/` sub-folder, computed from the
    highest-round per-round HDF5 in that folder."""
    import h5py
    import numpy as np
    from sklearn.metrics import f1_score
    if not metrics_dir.is_dir():
        return []

    token = dataset_token(dataset, alpha)
    out: List[float] = []
    seed_dirs = sorted(p for p in metrics_dir.iterdir()
                       if p.is_dir() and p.name.startswith("seed_"))
    if not seed_dirs:
        # Backwards-compat: an older run may have written a flat
        # `<metrics>/<TOKEN>/...` (no per-seed namespacing). Treat it as
        # if it were a single-seed sub-folder.
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
            f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
            out.append(f1)
        except (OSError, KeyError, ValueError) as e:
            print(f"[WARN] could not compute F1 from {last}: {e}",
                  file=sys.stderr)
    return out


def _round_idx(p: Path) -> int:
    try:
        return int(p.stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


# ---------------------------------------------------------------- aggregation
def collect_cell_values(input_root: Path, dataset: str, dataset_short: str,
                        alpha: float, miss: float, algo: str, metric: str,
                        suffix: str = "") -> List[float]:
    base = cell_dir(input_root, dataset_short, alpha, miss, algo, suffix=suffix)
    if metric == "Accuracy":
        return read_per_seed_accuracy(base / "models")
    if metric == "MacroF1":
        return read_per_seed_macro_f1(base / "metrics", dataset, alpha, algo)
    raise ValueError(f"Unknown metric: {metric}")


def mean_std_pct(values: List[float], pct: bool = True) -> Tuple[Optional[float], Optional[float], int]:
    """Return (mean_in_percent, std_in_percent, n). If `pct=False` returns the
    raw values. Returns (None, None, 0) when no seeds are present."""
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
    """Cell display string. Uses the paper's '88.7 +/- 0.3' style with one
    decimal."""
    if mu is None:
        return "--"
    if n <= 1:
        return f"{mu:.2f}"
    return f"{mu:.2f} +/- {sd:.2f}"


# ---------------------------------------------------------------- table build
def build_main_table(input_root: Path, dataset: str, dataset_short: str,
                     metric: str) -> "list[list[str]]":
    """Build a 2-D table for one (dataset, metric) pair.

    Layout (rows):
        header row
        block-header for alpha=0.1
            row miss=0.0
            row miss=0.10
            row miss=0.20
        block-header for alpha=1
            ... (same)
        block-header for alpha=10
            ... (same)
    Cells store dictionaries with {'text', 'is_winner', 'block_header'}.
    """
    rows: list[dict] = []

    # Header
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


def build_imputer_ablation(input_root: Path) -> "list[list[str]]":
    """One table with 3 dataset blocks, each showing FedGen at the headline
    cell with each of the 5 imputers in `ABLATION_IMPUTERS`. Two metric
    columns (Accuracy, Macro F1)."""
    rows: list[dict] = []

    rows.append({
        "kind": "header",
        "cells": ["Imputer", "Accuracy (%)", "Macro F1 (%)"],
    })

    for dataset, dataset_disp, dataset_short in DATASETS_ORDER:
        rows.append({
            "kind": "block",
            "cells": [f"Dataset: {dataset_disp}  "
                      f"(FedGen, alpha={HEADLINE_ALPHA}, miss="
                      f"{int(HEADLINE_MISS * 100)}%)", "", ""],
        })

        # Collect (mu_acc, mu_f1) per imputer to find row-winner per metric
        per_imputer = []
        for impid, _disp in ABLATION_IMPUTERS:
            suffix = f"imputer_ablation/{impid}"
            acc_vals = collect_cell_values(input_root, dataset, dataset_short,
                                           HEADLINE_ALPHA, HEADLINE_MISS,
                                           "FedGen", "Accuracy", suffix=suffix)
            f1_vals = collect_cell_values(input_root, dataset, dataset_short,
                                          HEADLINE_ALPHA, HEADLINE_MISS,
                                          "FedGen", "MacroF1", suffix=suffix)
            per_imputer.append((mean_std_pct(acc_vals), mean_std_pct(f1_vals)))

        if any(p[0][0] is not None for p in per_imputer):
            acc_best = max(range(len(per_imputer)),
                           key=lambda i: (per_imputer[i][0][0]
                                          if per_imputer[i][0][0] is not None
                                          else float("-inf")))
        else:
            acc_best = None
        if any(p[1][0] is not None for p in per_imputer):
            f1_best = max(range(len(per_imputer)),
                          key=lambda i: (per_imputer[i][1][0]
                                         if per_imputer[i][1][0] is not None
                                         else float("-inf")))
        else:
            f1_best = None

        for idx, (impid, disp) in enumerate(ABLATION_IMPUTERS):
            (acc, f1) = per_imputer[idx]
            rows.append({
                "kind": "data",
                "label": disp,
                "cells": [
                    {"text": fmt_cell(*acc),
                     "is_winner": (idx == acc_best and acc[0] is not None)},
                    {"text": fmt_cell(*f1),
                     "is_winner": (idx == f1_best and f1[0] is not None)},
                ],
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
            if r["kind"] == "header":
                w.writerow(r["cells"])
            elif r["kind"] == "block":
                w.writerow(r["cells"])
            else:
                line = [r["label"]] + [c["text"] for c in r["cells"]]
                w.writerow(line)


def write_markdown(rows: list, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"## {title}", ""]
    # Table header (first 'header' row)
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
    metric_label = {"Accuracy": "accuracy", "MacroF1": "macro_f1"}
    metric_title = {"Accuracy": "Accuracy (%)", "MacroF1": "Macro F1 (%)"}
    for metric in metrics:
        for dataset, dataset_disp, dataset_short in DATASETS_ORDER:
            rows = build_main_table(input_root, dataset, dataset_short, metric)
            stem = f"{metric_label[metric]}_{dataset_short}"
            title = (f"Performance comparison on {dataset_disp}: "
                     f"{metric_title[metric]} (mean +/- std).")
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
                   help=f"Root of the Option A sweep results "
                        f"(default {DEFAULT_INPUT}).")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT),
                   help=f"Where to write tables (default {DEFAULT_OUTPUT}).")
    p.add_argument("--metric", choices=["Accuracy", "MacroF1", "all"],
                   default="all",
                   help="Which main table(s) to produce. Default 'all' "
                        "produces 6 main tables (3 datasets x 2 metrics).")
    p.add_argument("--imputer_ablation", action="store_true",
                   help="Additionally produce results/optionA/tables/"
                        "imputer_ablation.{csv,md,tex}.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.metric == "all":
        metrics = ["Accuracy", "MacroF1"]
    else:
        metrics = [args.metric]

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
