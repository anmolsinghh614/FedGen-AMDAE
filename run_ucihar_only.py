#!/usr/bin/env python
"""
run_ucihar_only.py
==================

Focused UCI HAR run that produces ONLY the artifacts the paper needs from
UCI HAR for §5.6:

    1.  Accuracy / F1 / Precision / Recall table:
            rows = 5 FL algorithms,  cols = 3 missingness mechanisms
                     (MCAR / MAR / MNAR)
            -> results/ucihar_paper/ucihar_metrics_paper.csv
            -> results/ucihar_paper/ucihar_metrics_paper.md
            -> results/ucihar_paper/ucihar_metrics_paper.tex

    2.  Per-class F1 heatmap, one per missingness mechanism (paper Figs 5-8
        equivalent on real data):
            -> results/ucihar_paper/<mech>/heatmap/...

    3.  Last-round confusion matrix per (mechanism, algorithm)  -> 15 PNGs
            -> results/ucihar_paper/<mech>/confusion/...

    4.  Accuracy + training-loss side-by-side per mechanism (paper Fig 13):
            -> results/ucihar_paper/<mech>/acc_loss.png

Each missingness mechanism writes to its OWN namespaced directory:

    results/ucihar_paper/<mechanism>/{models,metrics,figures}/

so MAR doesn't clobber MCAR and MNAR doesn't clobber MAR (the plain
orchestrator path runs all three into the same `metrics_mr15/` because
the HDF5 filename is keyed only on algorithm, not pattern).

Default config matches paper §5.6:

    alpha = 0.5
    missing_rate = 0.15
    mechanisms = MCAR + MAR + MNAR
    algorithms = FedAvg, FedProx, FedDistill, FedEnsemble, FedGen
    num_glob_iters = 100, local_epochs = 20, num_users = 10, batch_size = 64

Usage examples:

    # Paper-quality run (single GPU, ~1-2 h)
    python run_ucihar_only.py --device cuda

    # Multi-seed run (three seeds; best for journal review)
    python run_ucihar_only.py --device cuda --times 3

    # ~20-min smoke
    python run_ucihar_only.py --device cuda --quick

    # Print the plan, run nothing
    python run_ucihar_only.py --dry_run
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
PY = sys.executable

ALGOS_DEFAULT = ["FedAvg", "FedGen", "FedProx", "FedDistill", "FedEnsemble"]
MECHANISMS_DEFAULT = ["random", "mar", "mnar"]   # 'random' == MCAR
MECH_LABEL = {"random": "MCAR", "mcar": "MCAR", "mar": "MAR", "mnar": "MNAR"}

DATASET_TOKEN = "UCI HAR-alpha0.5-ratio0.5"      # alpha=0.5, sampling ratio=0.5
TOKEN_FS_SAFE = DATASET_TOKEN                     # main.py / plotters use this exact string

PAPER_OUT = ROOT / "results" / "ucihar_paper"


# ---------------------------------------------------------------- helpers
def banner(msg: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def run(cmd, dry: bool = False, cwd: Path = None, allow_fail: bool = False) -> int:
    print(">> " + " ".join(str(c) for c in cmd), f"(cwd={cwd or ROOT})", flush=True)
    if dry:
        return 0
    rc = subprocess.call([str(c) for c in cmd], cwd=str(cwd) if cwd else str(ROOT))
    if rc != 0 and not allow_fail:
        print(f"[WARN] command exited rc={rc}", file=sys.stderr)
    return rc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--alpha", type=float, default=0.5,
                   help="Dirichlet alpha for the UCI HAR Dirichlet split (paper uses 0.5).")
    p.add_argument("--missing_rate", type=float, default=0.15,
                   help="Missing rate (paper uses 0.15).")
    p.add_argument("--mechanisms", nargs="*", default=MECHANISMS_DEFAULT,
                   choices=["random", "mcar", "mar", "mnar",
                            "fixed_intervals", "continuous_periods"],
                   help="Which missingness mechanisms to sweep (default: random mar mnar).")
    p.add_argument("--algorithms", nargs="*", default=ALGOS_DEFAULT,
                   help=f"FL algorithms (default: {ALGOS_DEFAULT}).")

    # Training knobs
    p.add_argument("--num_glob_iters", type=int, default=100)
    p.add_argument("--local_epochs", type=int, default=20)
    p.add_argument("--num_users", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gen_batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--times", type=int, default=1,
                   help="Random seeds (>=3 for paper-quality means/stds).")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    p.add_argument("--quick", action="store_true",
                   help="Tiny smoke run (5 rounds, 1 mechanism only).")
    p.add_argument("--skip_train", action="store_true",
                   help="Don't train; only rebuild tables/plots from existing data.")
    p.add_argument("--skip_plots", action="store_true",
                   help="Don't run the plotting phase (table only).")
    p.add_argument("--dry_run", action="store_true",
                   help="Print every command without executing.")
    p.add_argument("--auto_download", action="store_true",
                   help="If UCI HAR raw is missing, attempt to download via wget+unzip.")
    return p.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    if not args.quick:
        return
    args.num_glob_iters = 5
    args.local_epochs = 2
    args.mechanisms = args.mechanisms[:1]
    print("[quick] num_glob_iters=5 local_epochs=2 mechanisms=", args.mechanisms)


# ---------------------------------------------------------------- pre-flight
def ucihar_raw_present() -> bool:
    raw = ROOT / "data" / "UCI HAR" / "UCI HAR Dataset"
    return (raw / "train").is_dir() and (raw / "test").is_dir()


def maybe_download_ucihar(args: argparse.Namespace) -> None:
    if ucihar_raw_present():
        return
    if not args.auto_download:
        raise SystemExit(
            "\n[ERROR] UCI HAR raw archive not found at\n"
            f"        {ROOT/'data/UCI HAR/UCI HAR Dataset'}\n"
            "        Either pass --auto_download (needs wget+unzip+internet) or\n"
            "        manually place the dataset there. The dataset is at:\n"
            "        https://archive.ics.uci.edu/ml/machine-learning-databases/00240/\n"
            "        UCI%20HAR%20Dataset.zip"
        )
    target = ROOT / "data" / "UCI HAR"
    target.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"[dry] would download UCI HAR archive into {target}")
        return
    rc = run(["bash", "-c",
              "cd 'data/UCI HAR' && "
              "wget -q 'https://archive.ics.uci.edu/ml/machine-learning-databases/"
              "00240/UCI%20HAR%20Dataset.zip' && "
              "unzip -q 'UCI HAR Dataset.zip'"])
    if rc != 0 or not ucihar_raw_present():
        raise SystemExit("[ERROR] auto-download of UCI HAR failed; place it manually.")


def ensure_dirichlet_split(args: argparse.Namespace) -> None:
    """Generate the per-user split required by main.py if it isn't there.

    NOTE: UCI HAR's get_data_dir() in utils/model_utils.py hard-codes the
    path prefix to 'u20-alpha{a}-ratio{r}'. The federation MUST have 20
    users on disk (this is independent of --num_users, which is the number
    sampled per round). Sampling ratio is also hard-coded to 0.5 by the
    DATASET_TOKEN we pass to main.py."""
    n_user_total = 20
    sampling_ratio = 0.5
    split_dir = ROOT / "data" / "UCI HAR" / f"u{n_user_total}-alpha{args.alpha}-ratio{sampling_ratio}"
    if (split_dir / "train").is_dir() and (split_dir / "test").is_dir():
        print(f"[ok] Dirichlet split present at {split_dir}")
        return
    gen = ROOT / "data" / "UCI HAR" / "generate_niid_dirichlet.py"
    if not gen.is_file():
        raise SystemExit(f"[ERROR] missing generator: {gen}")
    # generator flag is singular: --n_user (not --n_users)
    cmd = [PY, str(gen),
           "--n_user", str(n_user_total),
           "--alpha", str(args.alpha),
           "--sampling_ratio", str(sampling_ratio)]
    rc = run(cmd, dry=args.dry_run, cwd=ROOT / "data" / "UCI HAR")
    if rc != 0 and not args.dry_run:
        raise SystemExit(f"[ERROR] failed to generate Dirichlet split (rc={rc})")


def expected_h5(args: argparse.Namespace, algorithm: str, seed: int,
                models_dir: Path) -> Path:
    """Mirror utils.model_utils.get_log_path for resume-skip."""
    name = (f"{DATASET_TOKEN}_{algorithm}_{args.learning_rate}_"
            f"{args.num_users}u_{args.batch_size}b_{args.local_epochs}_{seed}")
    if "FedGen" in algorithm:
        name += "_embed0"
        if int(args.gen_batch_size) != int(args.batch_size):
            name += f"_gb{args.gen_batch_size}"
    return models_dir / f"{name}.h5"


# ---------------------------------------------------------------- training
def train_one_mechanism(args: argparse.Namespace, mechanism: str) -> Path:
    """Train every algorithm on UCI HAR with one missingness mechanism.

    Returns the per-mechanism output root (results/ucihar_paper/<mech>/)."""
    mech_root = PAPER_OUT / mechanism
    models_dir = mech_root / "models"
    metrics_dir = mech_root / "metrics"
    figures_dir = mech_root / "figures"

    for d in (models_dir, metrics_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    banner(f"TRAIN  mechanism = {mechanism.upper()}  ->  {mech_root}")

    for algo in args.algorithms:
        if all(expected_h5(args, algo, s, models_dir).exists()
               for s in range(args.times)):
            print(f"[skip] {algo} {mechanism}  (all {args.times} seed(s) already trained)")
            continue

        # Stage live results/metrics/ aside so per-round dumps don't mix
        # across mechanisms. main.py hard-codes 'results/metrics' as the
        # per-round dump root (utils/metrics_utils.py) -- we move it after
        # each algo finishes.
        stale = ROOT / "results" / "metrics"
        if stale.is_dir() and not args.dry_run:
            print(f"[ERROR] leftover {stale} present; refusing to mix mechanisms.")
            print("        Move it out of the way (e.g. mv results/metrics _stale) "
                  "and re-run.")
            raise SystemExit(1)

        cmd = [PY, "main.py",
               "--dataset", DATASET_TOKEN,
               "--algorithm", algo,
               "--missing_rate", str(args.missing_rate),
               "--missing_pattern", mechanism,
               "--num_glob_iters", str(args.num_glob_iters),
               "--local_epochs", str(args.local_epochs),
               "--num_users", str(args.num_users),
               "--batch_size", str(args.batch_size),
               "--gen_batch_size", str(args.gen_batch_size),
               "--learning_rate", str(args.learning_rate),
               "--times", str(args.times),
               "--device", args.device,
               "--result_path", str(models_dir)]
        rc = run(cmd, dry=args.dry_run, allow_fail=True)
        if rc != 0:
            print(f"[WARN] training failed: {algo} / {mechanism}", file=sys.stderr)
            continue

        # Move per-round metrics dumps into this mechanism's namespace
        live = ROOT / "results" / "metrics"
        if live.is_dir() and not args.dry_run:
            for sub in live.iterdir():
                target = metrics_dir / sub.name
                target.mkdir(parents=True, exist_ok=True)
                if sub.is_dir():
                    for h5 in sub.iterdir():
                        shutil.move(str(h5), str(target / h5.name))
                else:
                    shutil.move(str(sub), str(target / sub.name))
            shutil.rmtree(live, ignore_errors=True)
            print(f"[move] results/metrics -> {metrics_dir}")

    return mech_root


# ---------------------------------------------------------------- table
def compute_paper_table(args: argparse.Namespace) -> None:
    """Build a single table (rows = algos, cols = mechanisms x metric).

    Reads the *highest-round* .h5 from each mechanism's metrics dir and
    computes accuracy, macro/weighted F1, precision, recall via
    sklearn.metrics. Writes csv/md/tex into results/ucihar_paper/."""
    if args.dry_run:
        print("[dry] would compute UCI HAR paper table")
        return

    try:
        import h5py
        import numpy as np
        from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                     recall_score)
    except ImportError as e:
        print(f"[WARN] paper-table phase needs h5py/numpy/scikit-learn ({e})")
        return

    rows = []
    for mech in args.mechanisms:
        token_dir = PAPER_OUT / mech / "metrics" / DATASET_TOKEN
        if not token_dir.is_dir():
            print(f"[WARN] no metrics found for mechanism={mech} at {token_dir}")
            continue
        for algo in args.algorithms:
            # pick highest-numbered round_*.h5 for this algo
            cand = sorted(token_dir.glob(f"{algo}_*round_*.h5"),
                          key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
            if not cand:
                print(f"[WARN] no round HDF5 for {algo} / {mech} in {token_dir}")
                continue
            last = cand[-1]
            try:
                with h5py.File(last, "r") as hf:
                    yt = np.asarray(hf["y_true"][:]).reshape(-1)
                    yp = np.asarray(hf["y_pred"][:]).reshape(-1)
            except Exception as e:
                print(f"[WARN] could not read {last}: {e}")
                continue

            rows.append({
                "Algorithm": algo,
                "Mechanism": MECH_LABEL.get(mech, mech.upper()),
                "Round": int(last.stem.rsplit("_", 1)[-1]),
                "Accuracy": accuracy_score(yt, yp),
                "MacroF1": f1_score(yt, yp, average="macro", zero_division=0),
                "WeightedF1": f1_score(yt, yp, average="weighted", zero_division=0),
                "MacroPrecision": precision_score(yt, yp, average="macro",
                                                  zero_division=0),
                "WeightedPrecision": precision_score(yt, yp, average="weighted",
                                                     zero_division=0),
                "MacroRecall": recall_score(yt, yp, average="macro",
                                            zero_division=0),
                "WeightedRecall": recall_score(yt, yp, average="weighted",
                                               zero_division=0),
            })

    if not rows:
        print("[WARN] no rows to write to paper table; skipping")
        return

    try:
        import pandas as pd
    except ImportError as e:
        print(f"[WARN] paper-table phase needs pandas ({e})")
        return

    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    long_path = PAPER_OUT / "ucihar_metrics_long.csv"
    df.to_csv(long_path, index=False)
    print(f"[ok] wrote {long_path}")

    # Wide form: rows = algorithm, columns = (mechanism, metric)
    metric_cols = ["Accuracy", "MacroF1", "WeightedF1",
                   "MacroPrecision", "WeightedPrecision",
                   "MacroRecall", "WeightedRecall"]
    wide = df.pivot_table(index="Algorithm",
                          columns="Mechanism",
                          values=metric_cols,
                          aggfunc="mean")
    wide_path_csv = PAPER_OUT / "ucihar_metrics_paper.csv"
    wide.to_csv(wide_path_csv, float_format="%.4f")
    print(f"[ok] wrote {wide_path_csv}")

    md_path = PAPER_OUT / "ucihar_metrics_paper.md"
    with open(md_path, "w") as f:
        f.write("# UCI HAR  --  Paper Table (Section 5.6)\n\n")
        f.write(f"alpha = {args.alpha}, missing_rate = {args.missing_rate}, "
                f"num_glob_iters = {args.num_glob_iters}, "
                f"local_epochs = {args.local_epochs}, "
                f"num_users = {args.num_users}, times = {args.times}\n\n")
        try:
            f.write(wide.to_markdown(floatfmt=".4f"))
        except Exception:
            f.write(wide.to_string(float_format=lambda x: f"{x:.4f}"))
        f.write("\n")
    print(f"[ok] wrote {md_path}")

    tex_path = PAPER_OUT / "ucihar_metrics_paper.tex"
    try:
        wide.to_latex(tex_path, float_format="%.4f", multicolumn=True)
        print(f"[ok] wrote {tex_path}")
    except Exception as e:
        print(f"[WARN] LaTeX export failed: {e}")


# ---------------------------------------------------------------- plots
def _have(*scripts: str) -> bool:
    return all((ROOT / s).is_file() for s in scripts)


def make_plots(args: argparse.Namespace) -> None:
    if args.skip_plots:
        print("[skip] --skip_plots given")
        return
    banner("PLOT  per-class F1 heatmap + confusion matrices + acc/loss")

    last_round = max(1, args.num_glob_iters - 1)

    for mech in args.mechanisms:
        mech_root = PAPER_OUT / mech
        metrics_dir = mech_root / "metrics"
        figures_dir = mech_root / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        if not (metrics_dir / DATASET_TOKEN).is_dir() and not args.dry_run:
            print(f"[skip] no metrics for mechanism={mech}; skipping its plots")
            continue

        # --- per-class F1 heatmap (paper Figs 5-8 equivalent on real data) -
        if _have("plot_per_class_f1_heatmap.py"):
            cmd = [PY, "plot_per_class_f1_heatmap.py",
                   "--dataset", DATASET_TOKEN,
                   "--input-root", str(metrics_dir),
                   "--output-root", str(figures_dir / "heatmap_f1"),
                   "--algorithms", *args.algorithms,
                   "--metric", "f1"]
            run(cmd, dry=args.dry_run, allow_fail=True)

        # --- F1-by-round line chart -----------------------------------------
        # NOTE: f1score_all.py auto-discovers datasets and algorithms from
        # the metrics dir; it does NOT accept --datasets / --algorithms.
        if _have("f1score_all.py"):
            cmd = [PY, "f1score_all.py",
                   "--input-root", str(metrics_dir),
                   "--output-root", str(figures_dir / "f1_by_round"),
                   "--rounds", str(last_round)]
            run(cmd, dry=args.dry_run, allow_fail=True)

        # --- last-round confusion matrix per algorithm ----------------------
        # Same auto-discovery convention as f1score_all.py.
        if _have("confusion_matrix_all.py"):
            cmd = [PY, "confusion_matrix_all.py",
                   "--input-root", str(metrics_dir),
                   "--output-root", str(figures_dir / "confusion_matrix"),
                   "--rounds", str(last_round)]
            run(cmd, dry=args.dry_run, allow_fail=True)

        # --- accuracy + training-loss side-by-side (paper Fig 13) -----------
        if _have("plot_experiment_results.py"):
            cmd = [PY, "plot_experiment_results.py",
                   "--dataset", DATASET_TOKEN,
                   "--algorithms",
                   ",".join(args.algorithms),
                   "--num_glob_iters", str(args.num_glob_iters),
                   "--local_epochs", str(args.local_epochs),
                   "--num_users", str(args.num_users),
                   "--batch_size", str(args.batch_size),
                   "--gen_batch_size", str(args.gen_batch_size),
                   "--learning_rate", str(args.learning_rate),
                   "--times", str(args.times),
                   "--result_path", str(mech_root / "models"),
                   "--plot_loss"]
            run(cmd, dry=args.dry_run, allow_fail=True)


# ---------------------------------------------------------------- summary
def final_summary(args: argparse.Namespace) -> None:
    banner("DONE  UCI HAR paper bundle")
    base = PAPER_OUT.relative_to(ROOT) if ROOT in PAPER_OUT.parents else PAPER_OUT
    print(f"\nEverything you need for paper Section 5.6 is under: {base}/\n")
    print("Paper-ready artifacts:")
    print(f"  - {base}/ucihar_metrics_paper.csv      (rows=algorithm,"
          " cols=mechanism x metric)")
    print(f"  - {base}/ucihar_metrics_paper.md       (paste into Markdown / docx)")
    print(f"  - {base}/ucihar_metrics_paper.tex      (paste into LaTeX paper)")
    print(f"  - {base}/ucihar_metrics_long.csv       (one row per (algo, mech) cell)")
    for mech in args.mechanisms:
        m = MECH_LABEL.get(mech, mech.upper())
        print(f"  - {base}/{mech}/figures/heatmap_f1/...    (Per-class F1 heatmap, {m})")
        print(f"  - {base}/{mech}/figures/confusion_matrix/...  (last-round CMs, {m})")
        print(f"  - {base}/{mech}/figures/f1_by_round/...   (F1 vs rounds, {m})")
    print(f"  - results/experiment_summary/acc_loss_*.png  (Fig 13 acc+loss "
          "side-by-side, written by plot_experiment_results.py)")

    print("\nTo paste into the paper:")
    print("  Section 5.6 table  -> ucihar_metrics_paper.tex (or .md)")
    print("  Heatmap figure     -> {mech}/figures/heatmap_f1/UCI HAR-alpha0.5-ratio0.5/"
          "heatmap_f1.png   (one per mechanism)")
    print("  Confusion-matrix figure(s) -> {mech}/figures/confusion_matrix/"
          "UCI HAR-alpha0.5-ratio0.5/confusion_matrix_round_<R>_<ALGO>.png")
    print("  Acc+loss figure    -> results/experiment_summary/"
          "acc_loss_UCI HAR_alpha0.5_miss0.15.png")


# ---------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    apply_quick(args)

    banner(
        f"FedGen-AMDAE  ::  UCI HAR ONLY  "
        f"({'DRY RUN' if args.dry_run else 'LIVE'})\n"
        f"  alpha            = {args.alpha}\n"
        f"  missing_rate     = {args.missing_rate}\n"
        f"  mechanisms       = {[MECH_LABEL.get(m, m.upper()) for m in args.mechanisms]}\n"
        f"  algorithms       = {args.algorithms}\n"
        f"  num_glob_iters   = {args.num_glob_iters}\n"
        f"  local_epochs     = {args.local_epochs}\n"
        f"  num_users        = {args.num_users}\n"
        f"  batch_size       = {args.batch_size}\n"
        f"  times (seeds)    = {args.times}\n"
        f"  device           = {args.device}"
    )

    if not args.skip_train:
        if not args.dry_run:
            maybe_download_ucihar(args)
            ensure_dirichlet_split(args)
        else:
            print("[dry] would check UCI HAR raw + Dirichlet split here")
        for mech in args.mechanisms:
            train_one_mechanism(args, mech)

    compute_paper_table(args)
    make_plots(args)
    final_summary(args)


if __name__ == "__main__":
    main()
