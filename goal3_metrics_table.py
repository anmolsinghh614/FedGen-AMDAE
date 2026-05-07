#!/usr/bin/env python
"""
Goal 3 -- F1 / Precision / Recall table per method, missing rate, and alpha.

What this script does
---------------------
Given one or more `results/metrics/` roots -- each tagged with the
missing-rate it was produced under -- this script:

  1. Walks every `<dataset>-alpha<a>-ratio<r>` subfolder
  2. Picks the highest available round per
     (algorithm, dataset, alpha, ratio, missing_rate)
  3. Reads y_true / y_pred (or y_prob) from the .h5 file using the
     same alias rules as evaluate_metrics.py
  4. Computes Precision, Recall, and F1 (macro AND weighted) per row
  5. Emits a long-form CSV plus paper-ready WIDE Markdown / LaTeX
     tables, one per dataset, indexed by Algorithm x MissingRate x
     Alpha (the same axes used by your existing accuracy table).

Inputs
------
Use --input multiple times, one per missing-rate root:

  --input 0:results/metrics_mr0
  --input 10:results/metrics_mr10
  --input 20:results/metrics_mr20

If you only have a single `results/metrics/` directory, you can
still call:

  --input 10:results/metrics

...and produce a one-column table.

Outputs
-------
results/tables/metrics_table_long.csv     # long-form (one row per cell)
results/tables/<dataset>_metrics_wide.csv # wide form, one per dataset
results/tables/<dataset>_metrics_wide.md  # markdown twin of the CSV
results/tables/<dataset>_metrics_wide.tex # LaTeX twin of the CSV
"""
import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CANDIDATE_TRUE_KEYS = ["y_true", "test_y", "labels", "targets", "test_targets"]
CANDIDATE_PRED_KEYS = ["y_pred", "preds", "test_pred", "test_predictions",
                       "predictions"]
CANDIDATE_PROBA_KEYS = ["y_prob", "probs", "probabilities", "logits", "outputs"]

# <algo>_<dataset_token>_round_<n>.h5  where dataset_token may contain spaces
FILE_RE = re.compile(r"^(?P<algo>[^_]+)_(?P<token>.+)_round_(?P<round>\d+)\.h5$")
TOKEN_RE = re.compile(
    r"^(?P<dataset>.+?)-alpha(?P<alpha>[0-9.]+)-ratio(?P<ratio>[0-9.]+)$"
)


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", "-i", action="append", required=True,
        metavar="MISSING_PCT:DIR",
        help="One per missing-rate root; example: '10:results/metrics_mr10'. "
             "Repeatable.")
    p.add_argument("--out_dir", default="results/tables")
    p.add_argument("--avg", choices=["macro", "weighted", "both"], default="both")
    p.add_argument("--zero_division", type=int, default=0,
                   help="Sklearn 'zero_division' kwarg (0 / 1 / np.nan-as-int).")
    return p.parse_args()


def first_existing(hf, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in hf:
            return k
    return None


def proba_to_pred(proba):
    import numpy as np
    p = np.asarray(proba)
    if p.ndim == 1 or (p.ndim == 2 and p.shape[1] == 1):
        return (p.reshape(-1) >= 0.5).astype(int)
    return np.argmax(p, axis=1)


def load_y(h5_path: Path):
    import h5py
    import numpy as np
    try:
        with h5py.File(h5_path, "r") as hf:
            tk = first_existing(hf, CANDIDATE_TRUE_KEYS)
            if tk is None:
                return None
            y_true = np.asarray(hf[tk][()]).reshape(-1)

            pk = first_existing(hf, CANDIDATE_PRED_KEYS)
            if pk is not None:
                y_pred = np.asarray(hf[pk][()]).reshape(-1)
            else:
                qk = first_existing(hf, CANDIDATE_PROBA_KEYS)
                if qk is None:
                    return None
                y_pred = proba_to_pred(np.asarray(hf[qk][()])).reshape(-1)

            if y_true.shape[0] != y_pred.shape[0]:
                return None
            return y_true, y_pred
    except OSError:
        return None


def parse_inputs(raw: List[str]) -> List[Tuple[float, Path]]:
    out: List[Tuple[float, Path]] = []
    for entry in raw:
        if ":" not in entry:
            raise SystemExit(f"--input must be 'MISSING_PCT:DIR', got '{entry}'")
        pct_str, path_str = entry.split(":", 1)
        try:
            pct = float(pct_str)
        except ValueError as e:
            raise SystemExit(f"Invalid missing percent in '{entry}': {e}")
        path = Path(path_str)
        if not path.is_dir():
            raise SystemExit(f"Not a directory: {path}")
        out.append((pct, path))
    return out


# ---------------------------------------------------------------------------
def latest_round_files(metrics_root: Path) -> Dict[Tuple[str, str, str, str], Path]:
    """Return map[(algo, dataset, alpha, ratio)] -> path of highest round."""
    best: Dict[Tuple[str, str, str, str], Tuple[int, Path]] = {}
    for ds_dir in sorted(d for d in metrics_root.iterdir()
                         if d.is_dir() and d.name.lower() != "eval"):
        for h5 in ds_dir.glob("*.h5"):
            m = FILE_RE.match(h5.name)
            if not m:
                continue
            tok = TOKEN_RE.match(m.group("token"))
            if not tok:
                continue
            key = (m.group("algo"),
                   tok.group("dataset"),
                   tok.group("alpha"),
                   tok.group("ratio"))
            r = int(m.group("round"))
            cur = best.get(key)
            if cur is None or cur[0] < r:
                best[key] = (r, h5)
    return {k: v[1] for k, v in best.items()}


# ---------------------------------------------------------------------------
def compute_row(y_true, y_pred, zero_division: int) -> Dict[str, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score
    return dict(
        precision_macro=float(precision_score(
            y_true, y_pred, average="macro", zero_division=zero_division)),
        recall_macro=float(recall_score(
            y_true, y_pred, average="macro", zero_division=zero_division)),
        f1_macro=float(f1_score(
            y_true, y_pred, average="macro", zero_division=zero_division)),
        precision_weighted=float(precision_score(
            y_true, y_pred, average="weighted", zero_division=zero_division)),
        recall_weighted=float(recall_score(
            y_true, y_pred, average="weighted", zero_division=zero_division)),
        f1_weighted=float(f1_score(
            y_true, y_pred, average="weighted", zero_division=zero_division)),
        n_samples=int(len(y_true)),
    )


def collect_long_table(inputs: List[Tuple[float, Path]],
                       zero_division: int):
    import pandas as pd
    rows: List[dict] = []
    for missing_pct, root in inputs:
        files = latest_round_files(root)
        if not files:
            print(f"[WARN] No round files matched under {root}", file=sys.stderr)
            continue
        for (algo, ds, alpha, ratio), h5 in sorted(files.items()):
            yp = load_y(h5)
            if yp is None:
                print(f"[SKIP] {h5}: could not extract y_true/y_pred",
                      file=sys.stderr)
                continue
            y_true, y_pred = yp
            metrics = compute_row(y_true, y_pred, zero_division)
            rows.append(dict(
                missing_pct=missing_pct,
                algorithm=algo,
                dataset=ds,
                alpha=float(alpha),
                ratio=float(ratio),
                round=int(re.search(r"_round_(\d+)\.h5$", h5.name).group(1)),
                file=str(h5),
                **metrics,
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def _format_pct(v: float) -> str:
    import numpy as np
    return "-" if not np.isfinite(v) else f"{v * 100:.2f}"


def _apply_elementwise(df, fn):
    """pandas 1.x and 2.0 use DataFrame.applymap; 2.1+ uses DataFrame.map."""
    if hasattr(df, "map") and callable(getattr(df, "map")):
        try:
            return df.map(fn)
        except (TypeError, ValueError):
            pass
    return df.applymap(fn)


def _safe_to_markdown(df) -> str:
    """to_markdown needs `tabulate`; degrade gracefully if it's missing."""
    try:
        return df.to_markdown()
    except ImportError:
        # Manual fallback: pipe-delimited Markdown without tabulate.
        cols = list(df.columns)
        head = "| algorithm | " + " | ".join(str(c) for c in cols) + " |"
        sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
        body = []
        for idx, row in df.iterrows():
            body.append("| " + str(idx) + " | "
                        + " | ".join(str(v) for v in row.values) + " |")
        return "\n".join([head, sep] + body)


def write_wide_outputs(df, out_dir: Path, avg: str) -> None:
    """One wide table per dataset; rows = Algorithm; columns = (Missing x Alpha x Metric)."""
    metrics = []
    if avg in ("macro", "both"):
        metrics += ["precision_macro", "recall_macro", "f1_macro"]
    if avg in ("weighted", "both"):
        metrics += ["precision_weighted", "recall_weighted", "f1_weighted"]

    for ds, sub in df.groupby("dataset"):
        # row = algo ; col = (miss, alpha, metric)
        sub = sub.copy()
        sub["miss"] = sub["missing_pct"].map(lambda x: f"{x:.0f}%")
        sub["alpha_str"] = sub["alpha"].map(lambda x: f"alpha={x:g}")

        wide = (sub.pivot_table(
                    index="algorithm",
                    columns=["miss", "alpha_str"],
                    values=metrics,
                    aggfunc="mean")
                .swaplevel(0, 1, axis=1)
                .swaplevel(1, 2, axis=1))
        wide = wide.sort_index(axis=1)

        # Pretty-print as percent strings for MD/TeX while keeping CSV numeric.
        ds_safe = ds.replace(" ", "_")
        csv_path = out_dir / f"{ds_safe}_metrics_wide.csv"
        md_path = out_dir / f"{ds_safe}_metrics_wide.md"
        tex_path = out_dir / f"{ds_safe}_metrics_wide.tex"

        wide.to_csv(csv_path)
        # Stringified copy for human-readable outputs
        wide_str = _apply_elementwise(wide, _format_pct)
        md_path.write_text(_safe_to_markdown(wide_str), encoding="utf-8")
        tex_path.write_text(
            wide_str.to_latex(
                caption=(f"Precision / Recall / F1 (\\%) on {ds} for every "
                         f"algorithm, missing-rate, and Dirichlet $\\alpha$."),
                label=f"tab:metrics_{ds_safe}",
                multicolumn=True, multirow=True),
            encoding="utf-8",
        )
        print(f"[OK] {ds:<24}  ->  {csv_path}, {md_path}, {tex_path}")


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    inputs = parse_inputs(args.input)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = collect_long_table(inputs, args.zero_division)
    if df.empty:
        print("No metrics could be computed; check your --input paths.",
              file=sys.stderr)
        sys.exit(1)

    long_path = out_dir / "metrics_table_long.csv"
    df_sorted = df.sort_values(
        ["dataset", "missing_pct", "alpha", "algorithm"]).reset_index(drop=True)
    df_sorted.to_csv(long_path, index=False)
    print(f"\n[OK] long-form table  ->  {long_path}  ({len(df_sorted)} rows)")

    write_wide_outputs(df_sorted, out_dir, args.avg)

    print("\nDone.  Use the WIDE tables for the paper, and the LONG csv for "
          "ad-hoc filtering / re-aggregation.")


if __name__ == "__main__":
    main()
