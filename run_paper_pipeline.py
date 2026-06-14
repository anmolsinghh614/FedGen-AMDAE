#!/usr/bin/env python
"""
run_paper_pipeline.py -- One-shot driver that produces every result and
figure required for the paper.

Phases (each can be disabled with --skip_<phase>):

  PHASE 0 -- Emit AMDAE methodology declaration  (goal1)
  PHASE 1 -- Auto-generate per-dataset Dirichlet splits as needed
  PHASE 2 -- Train every (dataset, alpha, missing_rate, algorithm) cell
  PHASE 3 -- Build the paper-ready Precision/Recall/F1 tables (goal3)
  PHASE 4 -- Build paper-ready plots:
                * F1-by-round per algorithm  (f1score_all.py)
                * Last-round confusion matrices  (confusion_matrix_all.py)
                * Accuracy curves + summary text table
                  (plot_experiment_results.py)
  PHASE 5 -- (optional) UCI HAR run via goal2_real_dataset_experiment.py
             when --include_ucihar is given.

Outputs land under:
  results/zero_missing_baseline/    AMDAE declaration + 0% summary
  results/models_mr<RR>/            per-run accuracy/loss HDF5
  results/metrics_mr<RR>/           per-round y_true/y_pred HDF5
  results/tables/                   long + wide F1/Precision/Recall tables
  results/figures/mr<RR>/           per-missing-rate plots
  results/experiment_summary/       paper-style accuracy curves + tables
  results/real_dataset_experiments/ UCI HAR summary

Quick start for a 5-round dry-style sanity check:

  py -3 run_paper_pipeline.py --quick

A full paper run on the default grid:

  py -3 run_paper_pipeline.py --datasets Mnist EMnist \
        --alphas 0.1 1.0 10.0 --missing_rates 0.0 0.1 0.2 \
        --num_glob_iters 100

Re-run only the table + plot phases on already-trained results:

  py -3 run_paper_pipeline.py --skip_train
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent

DATASETS_DEFAULT = ["Mnist", "EMnist"]
ALPHAS_DEFAULT = [0.1, 1.0, 10.0]
MISSING_DEFAULT = [0.0, 0.1, 0.2]
ALGOS_DEFAULT = ["FedAvg", "FedGen", "FedProx", "FedDistill", "FedEnsemble"]

# What sampling-ratio each dataset uses end-to-end in this codebase.
SAMPLING_RATIO = {
    "Mnist": 0.5,
    "EMnist": 0.1,
    "UCI HAR": 0.5,
    "PAMAP2": 0.5,
}

# Per-dataset path patterns the model code expects on disk.
DATASET_SPLIT_DIR_TEMPLATE = {
    "Mnist": "data/Mnist/u20c10-alpha{alpha}-ratio{ratio}",
    "EMnist": "data/EMnist/u20-letters-alpha{alpha}-ratio{ratio}",
    "UCI HAR": "data/UCI HAR/u20-alpha{alpha}-ratio{ratio}",
    "PAMAP2": "data/PAMAP2/u20-alpha{alpha}-ratio{ratio}",
}

# Where to invoke each generator from, and the script name.
# PAMAP2 has no in-tree generator; use goal2_real_dataset_experiment.py first.
DATASET_GENERATORS = {
    "Mnist": ("data/Mnist", "generate_niid_dirichlet.py"),
    "EMnist": ("data/EMnist", "generate_niid_dirichlet.py"),
    "UCI HAR": ("data/UCI HAR", "generate_niid_dirichlet.py"),
}


# -------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--datasets", nargs="+", default=DATASETS_DEFAULT,
                   help="Datasets to sweep. Use 'UCI HAR' (quoted) to add UCI HAR.")
    p.add_argument("--alphas", nargs="+", type=float, default=ALPHAS_DEFAULT,
                   help="Dirichlet alpha values to sweep.")
    p.add_argument("--missing_rates", nargs="+", type=float, default=MISSING_DEFAULT,
                   help="Missing-rate fractions (0.0 .. 1.0) to sweep.")
    p.add_argument("--algorithms", nargs="+", default=ALGOS_DEFAULT)

    p.add_argument("--num_glob_iters", type=int, default=100)
    p.add_argument("--local_epochs", type=int, default=20)
    p.add_argument("--num_users", type=int, default=10)
    p.add_argument("--n_user_split", type=int, default=20,
                   help="Total users in the per-dataset split; should be 20 for "
                        "Mnist/EMnist/UCI HAR because utils/model_utils.py "
                        "hard-codes that prefix.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gen_batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--times", type=int, default=1)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    p.add_argument("--include_ucihar", action="store_true",
                   help="Append 'UCI HAR' to --datasets for the entire sweep.")
    p.add_argument("--include_pamap2", action="store_true",
                   help="Append 'PAMAP2' to --datasets for the entire sweep. "
                        "Requires that you have run goal2_real_dataset_experiment.py "
                        "--dataset_kind pamap2 --prepare_only first.")
    p.add_argument("--missing_pattern", choices=[
        "random", "mcar", "mar", "mnar",
        "fixed_intervals", "continuous_periods",
    ], default="random",
        help="Missing-data mechanism passed to main.py and AM-DAE simulator. "
             "'mar' uses row labels, 'mnar' uses cell magnitudes.")
    p.add_argument("--paper_preset", choices=["none", "ucihar", "ucihar_3mech"],
                   default="none",
                   help="Pre-canned sweeps that match the paper. "
                        "'ucihar' = alpha=0.5, missing=0.15, MCAR; "
                        "'ucihar_3mech' = alpha=0.5, missing=0.15, sweeping "
                        "MCAR/MAR/MNAR (one orchestration pass each).")
    p.add_argument("--quick", action="store_true",
                   help="Tiny sanity-check run: 5 rounds, 1 alpha, 2 missing-rates.")

    p.add_argument("--skip_data_prep", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_table", action="store_true")
    p.add_argument("--skip_plot", action="store_true")

    p.add_argument("--dry_run", action="store_true",
                   help="Print every command without executing it.")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip the 'are you sure' confirmation prompt.")
    return p.parse_args()


def apply_quick(args: argparse.Namespace) -> None:
    args.num_glob_iters = 5
    args.local_epochs = 2
    args.alphas = [args.alphas[0]]
    if len(args.missing_rates) > 2:
        args.missing_rates = args.missing_rates[:2]


def apply_paper_preset(args: argparse.Namespace) -> None:
    """Override the sweep grid to match a published paper section."""
    if args.paper_preset == "none":
        return
    if args.paper_preset.startswith("ucihar"):
        args.datasets = ["UCI HAR"]
        args.include_ucihar = True
        args.alphas = [0.5]
        args.missing_rates = [0.15]
        if args.paper_preset == "ucihar_3mech":
            # Caller is expected to re-run the orchestrator three times,
            # one per mechanism. We only set the *default* mechanism here.
            print("[preset] ucihar_3mech selected: this single invocation runs "
                  "missing_pattern=", args.missing_pattern, "; re-run with "
                  "--missing_pattern mar and --missing_pattern mnar to cover "
                  "all three mechanisms reported in paper Sec.5.6.")
        else:
            args.missing_pattern = "random"


# -------------------------------------------------------------------------
def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78, flush=True)


def run_cmd(cmd: List[str], cwd: Path = None, dry: bool = False) -> int:
    print(">>", " ".join(str(c) for c in cmd), f"(cwd={cwd})" if cwd else "")
    sys.stdout.flush()
    if dry:
        return 0
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None)


# -------------------------------------------------------------------------
def expected_h5(args: argparse.Namespace, token: str, algorithm: str,
                seed: int, mr_int: int) -> Path:
    """Mirror utils.model_utils.get_log_path so we can detect already-done runs."""
    name = (f"{token}_{algorithm}_{args.learning_rate}_{args.num_users}u_"
            f"{args.batch_size}b_{args.local_epochs}_{seed}")
    if "FedGen" in algorithm:
        name += "_embed0"
        if int(args.gen_batch_size) != int(args.batch_size):
            name += f"_gb{args.gen_batch_size}"
    return ROOT / f"results/models_mr{mr_int}" / f"{name}.h5"


def all_seeds_done(args: argparse.Namespace, token: str, algorithm: str,
                   mr_int: int) -> bool:
    """A cell is considered done only when every seed 0..times-1 has its h5."""
    return all(
        expected_h5(args, token, algorithm, seed, mr_int).exists()
        for seed in range(args.times)
    )


def dataset_token(dataset: str, alpha: float) -> str:
    ratio = SAMPLING_RATIO.get(dataset, 0.5)
    return f"{dataset}-alpha{alpha}-ratio{ratio}"


# -------------------------------------------------------------------------
# PHASE 0 -- AMDAE declaration
# -------------------------------------------------------------------------
def phase_declaration(args: argparse.Namespace) -> None:
    banner("PHASE 0  AMDAE methodology declaration")
    rep_dataset = dataset_token(args.datasets[0], args.alphas[0])
    cmd = [sys.executable, "goal1_zero_missing_baseline.py",
           "--dataset", rep_dataset, "--declaration_only"]
    run_cmd(cmd, cwd=ROOT, dry=args.dry_run)


# -------------------------------------------------------------------------
# PHASE 1 -- Data prep
# -------------------------------------------------------------------------
def phase_data_prep(args: argparse.Namespace) -> None:
    banner("PHASE 1  Auto-generate Dirichlet splits if missing")
    for ds in args.datasets:
        ratio = SAMPLING_RATIO.get(ds, 0.5)
        for alpha in args.alphas:
            split_dir = ROOT / DATASET_SPLIT_DIR_TEMPLATE[ds].format(
                alpha=alpha, ratio=ratio)
            train_dir = split_dir / "train"
            if train_dir.is_dir() and any(train_dir.iterdir()):
                print(f"[skip] split exists: {split_dir}")
                continue

            if ds == "PAMAP2":
                # PAMAP2 has no in-tree Dirichlet generator; goal2 owns prep.
                print(f"[gen ] PAMAP2 split needed: {split_dir}")
                cmd = [sys.executable, "goal2_real_dataset_experiment.py",
                       "--dataset_kind", "pamap2",
                       "--alpha", str(alpha),
                       "--sampling_ratio", str(ratio),
                       "--n_user", str(args.n_user_split),
                       "--prepare_only"]
                rc = run_cmd(cmd, cwd=ROOT, dry=args.dry_run)
                if rc != 0:
                    raise SystemExit(
                        f"\nPAMAP2 data preparation failed for alpha={alpha}.\n"
                        f"Network access is required to download the archive; "
                        f"re-run goal2_real_dataset_experiment.py manually if "
                        f"you need to debug.")
                continue

            if ds not in DATASET_GENERATORS:
                print(f"[warn] no generator registered for '{ds}'; skipping")
                continue

            gen_dir, gen_script = DATASET_GENERATORS[ds]
            print(f"[gen ] split needed:  {split_dir}")
            cmd = [sys.executable, gen_script,
                   "--alpha", str(alpha),
                   "--sampling_ratio", str(ratio),
                   "--n_user", str(args.n_user_split)]
            rc = run_cmd(cmd, cwd=ROOT / gen_dir, dry=args.dry_run)
            if rc != 0:
                raise SystemExit(
                    f"\nDataset generation failed for {ds} alpha={alpha}.\n"
                    f"  cwd: {ROOT / gen_dir}\n"
                    f"  cmd: {' '.join(cmd)}\n"
                    f"For 'UCI HAR' make sure the raw dataset is already at "
                    f"data/UCI HAR/data/UCI HAR Dataset/ before retrying.")


# -------------------------------------------------------------------------
# PHASE 2 -- Training
# -------------------------------------------------------------------------
def phase_train(args: argparse.Namespace) -> None:
    banner("PHASE 2  Train every (dataset, alpha, missing_rate, algorithm)")
    for ds in args.datasets:
        for alpha in args.alphas:
            token = dataset_token(ds, alpha)
            for mr in args.missing_rates:
                mr_int = int(round(mr * 100))
                models_dir = ROOT / f"results/models_mr{mr_int}"
                metrics_dir = ROOT / f"results/metrics_mr{mr_int}"
                models_dir.mkdir(parents=True, exist_ok=True)

                # Safety: utils/metrics_utils.py hard-codes results/metrics/.
                # On a *clean* start (no metrics_mr<R> dirs yet), a leftover
                # results/metrics/ is almost certainly from a prior crash;
                # we cannot tell which missing-rate it belongs to, so we
                # refuse to silently mis-attribute it. The user must clear
                # or rename it before resuming.
                stale = ROOT / "results/metrics"
                any_mr_dir = any(
                    (ROOT / f"results/metrics_mr{int(round(m * 100))}").is_dir()
                    for m in args.missing_rates
                )
                if stale.is_dir() and not any_mr_dir:
                    raise SystemExit(
                        f"\n[ERROR] Found leftover {stale} but no results/metrics_mr<RR>/\n"
                        f"        directories exist yet, so the orchestrator cannot tell\n"
                        f"        which missing-rate this data belongs to.\n"
                        f"        Move/rename or delete it manually, then re-run.\n"
                        f"        (Hint: if you know it came from missing_rate=R, run\n"
                        f"        `mv results/metrics results/metrics_mrRR` first.)"
                    )

                for algo in args.algorithms:
                    if all_seeds_done(args, token, algo, mr_int):
                        print(f"[skip] {algo} {token} mr={mr} -- all "
                              f"{args.times} seed(s) already trained")
                        continue
                    cmd = [
                        sys.executable, "main.py",
                        "--dataset", token,
                        "--algorithm", algo,
                        "--missing_rate", str(mr),
                        "--missing_pattern", args.missing_pattern,
                        "--num_glob_iters", str(args.num_glob_iters),
                        "--local_epochs", str(args.local_epochs),
                        "--num_users", str(args.num_users),
                        "--batch_size", str(args.batch_size),
                        "--gen_batch_size", str(args.gen_batch_size),
                        "--learning_rate", str(args.learning_rate),
                        "--times", str(args.times),
                        "--device", args.device,
                        "--result_path", str(models_dir),
                    ]
                    rc = run_cmd(cmd, cwd=ROOT, dry=args.dry_run)
                    if rc != 0:
                        print(f"[WARN] training failed: {token} / {algo} / mr={mr}",
                              file=sys.stderr)

                # End of one missing-rate sweep -- move per-round metrics aside.
                live = ROOT / "results/metrics"
                if live.is_dir() and not args.dry_run:
                    if metrics_dir.is_dir():
                        # Per-file merge: preserve existing data from earlier
                        # (ds, alpha) pairs that already wrote into this same
                        # missing-rate bucket.
                        for sub in live.iterdir():
                            target = metrics_dir / sub.name
                            target.mkdir(parents=True, exist_ok=True)
                            if sub.is_dir():
                                for h5 in sub.iterdir():
                                    shutil.move(str(h5), str(target / h5.name))
                            else:
                                shutil.move(str(sub), str(target / sub.name))
                        shutil.rmtree(live, ignore_errors=True)
                        print(f"[move] merged results/metrics -> {metrics_dir}")
                    else:
                        shutil.move(str(live), str(metrics_dir))
                        print(f"[move] results/metrics -> {metrics_dir}")


# -------------------------------------------------------------------------
# PHASE 3 -- Tables
# -------------------------------------------------------------------------
def phase_tables(args: argparse.Namespace) -> None:
    banner("PHASE 3  Build Precision/Recall/F1 tables (goal3)")
    inputs = []
    for mr in args.missing_rates:
        mr_int = int(round(mr * 100))
        d = ROOT / f"results/metrics_mr{mr_int}"
        if d.is_dir():
            inputs += ["--input", f"{int(round(mr * 100))}:{d}"]
    if not inputs:
        print("[skip] no metrics_mr<rate> directories found; nothing to tabulate.")
        return
    cmd = [sys.executable, "goal3_metrics_table.py", *inputs,
           "--out_dir", str(ROOT / "results/tables")]
    run_cmd(cmd, cwd=ROOT, dry=args.dry_run)


# -------------------------------------------------------------------------
# PHASE 4 -- Plots
# -------------------------------------------------------------------------
def phase_plots(args: argparse.Namespace) -> None:
    banner("PHASE 4  Build paper plots")
    fig_root = ROOT / "results/figures"
    fig_root.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets:
        for alpha in args.alphas:
            token = dataset_token(ds, alpha)
            for mr in args.missing_rates:
                mr_int = int(round(mr * 100))
                metrics_dir = ROOT / f"results/metrics_mr{mr_int}"
                models_dir = ROOT / f"results/models_mr{mr_int}"
                fig_dir = fig_root / f"mr{mr_int}"
                if not metrics_dir.is_dir():
                    print(f"[skip] no metrics_mr{mr_int} -- skipping plots for "
                          f"{token}, mr={mr}")
                    continue

                # f1score_all/confusion_matrix_all scan rounds 1..R (1-indexed),
                # but main.py writes round_0..round_{N-1}. Asking for N-1 lets
                # them see every file that was actually saved (round 0 is still
                # not included on the curve because both tools are 1-indexed).
                scan_rounds = max(1, args.num_glob_iters - 1)

                # F1-vs-round (one figure per dataset, all algorithms)
                cmd = [sys.executable, "f1score_all.py",
                       "--dataset", token,
                       "--rounds", str(scan_rounds),
                       "--input-root", str(metrics_dir),
                       "--output-root", str(fig_dir)]
                run_cmd(cmd, cwd=ROOT, dry=args.dry_run)

                # Last-round confusion matrices (one per algorithm)
                cmd = [sys.executable, "confusion_matrix_all.py",
                       "--dataset", token,
                       "--rounds", str(scan_rounds),
                       "--input-root", str(metrics_dir),
                       "--output-root", str(fig_dir)]
                run_cmd(cmd, cwd=ROOT, dry=args.dry_run)

                # Per-class F1 heatmap (paper Figs 5-8)
                cmd = [sys.executable, "plot_per_class_f1_heatmap.py",
                       "--dataset", token,
                       "--input-root", str(metrics_dir),
                       "--output-root", str(fig_dir / "heatmap_f1")]
                run_cmd(cmd, cwd=ROOT, dry=args.dry_run)

                # Paper-style accuracy curves (+ optional loss panel)
                if models_dir.is_dir():
                    cmd = [sys.executable, "plot_experiment_results.py",
                           "--dataset", token,
                           "--algorithms", ",".join(args.algorithms),
                           "--missing_rate", str(mr),
                           "--result_path", str(models_dir),
                           "--num_glob_iters", str(args.num_glob_iters),
                           "--num_users", str(args.num_users),
                           "--batch_size", str(args.batch_size),
                           "--gen_batch_size", str(args.gen_batch_size),
                           "--local_epochs", str(args.local_epochs),
                           "--learning_rate", str(args.learning_rate),
                           "--times", str(args.times),
                           "--plot_loss"]
                    run_cmd(cmd, cwd=ROOT, dry=args.dry_run)


# -------------------------------------------------------------------------
# PHASE 5 -- UCI HAR analysis (only if user asked)
# -------------------------------------------------------------------------
def phase_real_dataset(args: argparse.Namespace, kind: str, label: str) -> None:
    """Drive goal2_real_dataset_experiment for UCI HAR or PAMAP2."""
    banner(f"PHASE  {label}  full sweep via goal2 (data prep + train)")
    for alpha in args.alphas:
        for mr in args.missing_rates:
            cmd = [sys.executable, "goal2_real_dataset_experiment.py",
                   "--dataset_kind", kind,
                   "--alpha", str(alpha),
                   "--sampling_ratio", str(SAMPLING_RATIO.get(label, 0.5)),
                   "--missing_rate", str(mr),
                   "--n_user", str(args.n_user_split),
                   "--num_glob_iters", str(args.num_glob_iters),
                   "--local_epochs", str(args.local_epochs),
                   "--num_users", str(args.num_users),
                   "--batch_size", str(args.batch_size),
                   "--gen_batch_size", str(args.gen_batch_size),
                   "--learning_rate", str(args.learning_rate),
                   "--times", str(args.times),
                   "--device", args.device,
                   "--algorithms", *args.algorithms,
                   "--skip_generate"]
            run_cmd(cmd, cwd=ROOT, dry=args.dry_run)


def phase_ucihar(args: argparse.Namespace) -> None:
    if not args.include_ucihar or "UCI HAR" not in args.datasets:
        return
    phase_real_dataset(args, "ucihar", "UCI HAR")


def phase_pamap2(args: argparse.Namespace) -> None:
    if not args.include_pamap2 or "PAMAP2" not in args.datasets:
        return
    phase_real_dataset(args, "pamap2", "PAMAP2")


# -------------------------------------------------------------------------
def estimate_grid(args: argparse.Namespace) -> int:
    return (len(args.datasets) * len(args.alphas)
            * len(args.missing_rates) * len(args.algorithms) * args.times)


def confirm(args: argparse.Namespace) -> None:
    n = estimate_grid(args)
    print()
    print(f"  datasets       : {args.datasets}")
    print(f"  alphas         : {args.alphas}")
    print(f"  missing_rates  : {args.missing_rates}")
    print(f"  algorithms     : {args.algorithms}")
    print(f"  num_glob_iters : {args.num_glob_iters}")
    print(f"  local_epochs   : {args.local_epochs}")
    print(f"  num_users/round: {args.num_users}")
    print(f"  device         : {args.device}")
    print(f"  total trainings: {n}")
    print(f"  missing_pattern: {args.missing_pattern}")
    print(f"  paper_preset   : {args.paper_preset}")
    print(f"  include_ucihar : {args.include_ucihar}")
    print(f"  include_pamap2 : {args.include_pamap2}")
    print(f"  skip_train     : {args.skip_train}")
    print(f"  skip_table     : {args.skip_table}")
    print(f"  skip_plot      : {args.skip_plot}")
    print(f"  dry_run        : {args.dry_run}")
    if args.yes or args.dry_run:
        return
    ans = input("\nProceed? [y/N] ").strip().lower()
    if ans not in ("y", "yes"):
        print("Aborted.")
        sys.exit(0)


# -------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    if args.quick:
        apply_quick(args)
    apply_paper_preset(args)
    if args.include_ucihar and "UCI HAR" not in args.datasets:
        args.datasets.append("UCI HAR")
    if args.include_pamap2 and "PAMAP2" not in args.datasets:
        args.datasets.append("PAMAP2")

    confirm(args)

    phase_declaration(args)
    if not args.skip_data_prep:
        phase_data_prep(args)
    if not args.skip_train:
        phase_train(args)
    if not args.skip_table:
        phase_tables(args)
    if not args.skip_plot:
        phase_plots(args)
    phase_ucihar(args)
    phase_pamap2(args)

    banner("DONE")
    print("Artifacts:")
    print("  results/zero_missing_baseline/   AMDAE declaration + 0% summary")
    print("  results/models_mr<RR>/           per-run accuracy/loss HDF5")
    print("  results/metrics_mr<RR>/          per-round y_true/y_pred HDF5")
    print("  results/tables/                  long + wide F1/Precision/Recall")
    print("  results/figures/mr<RR>/          F1-by-round + confusion matrices + heatmap_f1/")
    print("  results/experiment_summary/      paper-style accuracy curves + acc_loss panels")
    if args.include_ucihar:
        print("  results/real_dataset_experiments/  UCI HAR summary")
    if args.include_pamap2:
        print("  results/real_dataset_experiments/  PAMAP2 summary")


if __name__ == "__main__":
    main()
