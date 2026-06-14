#!/usr/bin/env python
"""
produce_paper_results.py
========================

ONE-BUTTON driver that runs every experiment / plot / table needed to fill
the gaps in the FedGen-AMDAE paper. After this finishes you will have:

  1. AM-DAE methodology declaration paragraph     (paste into paper Sec.4.2)
  2. 0%-missing baseline numbers (MNIST + EMNIST) (adds the "0%" column to
                                                   paper Tables 2 & 3)
  3. 10% / 20% missing sweep over alpha={0.1,1,10}(reproduces paper Tables 2&3)
  4. UCI HAR @ alpha=0.5, 15% missing, MCAR/MAR/MNAR (paper Sec.5.6)
  5. PAMAP2 results (new section the paper currently lists as future work)
  6. Per-class F1 / Precision / Recall TABLE       (paper Sec.5.x table)
  7. Per-class F1 heatmaps                         (paper Figs 5-8)
  8. Acc + training-loss side-by-side              (paper Fig 13)

It is a thin wrapper around `run_paper_pipeline.py`, `goal3_metrics_table.py`,
and `goal2_real_dataset_experiment.py`. It just orchestrates them in the
right order with the right flags.

Usage (one shot, full paper rebuild on a GPU machine):
    py -3 produce_paper_results.py

Quick sanity check (~30 min on CPU; tiny grid, will NOT match paper numbers):
    py -3 produce_paper_results.py --quick

Just simulate; print the commands:
    py -3 produce_paper_results.py --dry_run

Skip the real-world datasets (UCI HAR + PAMAP2):
    py -3 produce_paper_results.py --skip_real

You can also pass any extra args through to the orchestrator:
    py -3 produce_paper_results.py --num_glob_iters 50 --times 1
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

ALGOS = ["FedAvg", "FedGen", "FedProx", "FedDistill", "FedEnsemble"]


# ------------------------------------------------------------------ helpers
def banner(msg: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def run(cmd, dry: bool, cwd: Path = None) -> int:
    print(">> " + " ".join(str(c) for c in cmd), f"(cwd={cwd or ROOT})", flush=True)
    if dry:
        return 0
    rc = subprocess.call([str(c) for c in cmd], cwd=str(cwd) if cwd else str(ROOT))
    if rc != 0:
        print(f"[WARN] command exited rc={rc}", file=sys.stderr)
    return rc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Sweep size
    p.add_argument("--quick", action="store_true",
                   help="Tiny smoke-run (5 rounds, 1 alpha, 2 missing rates).")
    p.add_argument("--budget", choices=["full", "fast", "minimal"], default="full",
                   help="Pre-canned time budgets. "
                        "'full'    = paper grid, ~24-72h GPU-h (default). "
                        "'fast'    = times=1, 50 rounds, full grid, ~10-12h. "
                        "'minimal' = times=1, 50 rounds, 1 alpha, 2 miss-rates, ~4-6h.")
    p.add_argument("--num_glob_iters", type=int, default=100,
                   help="Federated rounds per cell (paper uses 100).")
    p.add_argument("--local_epochs", type=int, default=20)
    p.add_argument("--num_users", type=int, default=10,
                   help="Active users per round (paper uses 10/20).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gen_batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--times", type=int, default=3,
                   help="Random seeds (>=3 for paper-quality means/stds).")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    # Stage-1 grid trimming (passthrough to run_paper_pipeline.py)
    p.add_argument("--datasets", nargs="*",
                   default=["Mnist", "EMnist"],
                   help="Stage-1 datasets (default: Mnist EMnist).")
    p.add_argument("--alphas", nargs="*", type=float,
                   default=[0.1, 1.0, 10.0],
                   help="Stage-1 Dirichlet alphas (default: 0.1 1.0 10.0).")
    p.add_argument("--missing_rates", nargs="*", type=float,
                   default=[0.0, 0.1, 0.2],
                   help="Stage-1 missing rates (default: 0.0 0.1 0.2).")
    p.add_argument("--algorithms", nargs="*",
                   default=ALGOS,
                   help=f"FL algorithms to run in every stage (default: {ALGOS}).")

    # Phase toggles
    p.add_argument("--skip_baseline", action="store_true",
                   help="Skip the MNIST/EMNIST sweep over (alpha, missing).")
    p.add_argument("--skip_real", action="store_true",
                   help="Skip UCI HAR + PAMAP2 phases.")
    p.add_argument("--skip_ucihar", action="store_true")
    p.add_argument("--skip_pamap2", action="store_true")
    p.add_argument("--skip_table", action="store_true",
                   help="Skip the goal3 F1/Precision/Recall table phase.")

    # Real-dataset knobs
    p.add_argument("--har_patterns", nargs="*",
                   default=["random", "mar", "mnar"],
                   help="UCI HAR missing mechanisms to run (paper Sec.5.6).")
    p.add_argument("--har_alpha", type=float, default=0.5,
                   help="Dirichlet alpha for UCI HAR (paper uses 0.5).")
    p.add_argument("--har_missing_rate", type=float, default=0.15,
                   help="UCI HAR missing rate (paper uses 0.15).")
    p.add_argument("--pamap2_alphas", nargs="*", type=float,
                   default=[0.1, 1.0],
                   help="Dirichlet alpha values to run on PAMAP2.")
    p.add_argument("--pamap2_missing_rates", nargs="*", type=float,
                   default=[0.1, 0.2],
                   help="Missing-rate values to run on PAMAP2.")

    # Behaviour
    p.add_argument("--dry_run", action="store_true",
                   help="Print every command without executing.")
    p.add_argument("--yes", "-y", action="store_true",
                   help="Skip the orchestrator's confirmation prompt.")
    return p.parse_args()


# ------------------------------------------------------------------ phases
def common_orchestrator_args(args, missing_pattern: str = "random"):
    base = [
        "--num_glob_iters", str(args.num_glob_iters),
        "--local_epochs", str(args.local_epochs),
        "--num_users", str(args.num_users),
        "--batch_size", str(args.batch_size),
        "--gen_batch_size", str(args.gen_batch_size),
        "--learning_rate", str(args.learning_rate),
        "--times", str(args.times),
        "--device", args.device,
        "--missing_pattern", missing_pattern,
    ]
    if args.quick:
        base.append("--quick")
    if args.dry_run:
        base.append("--dry_run")
    if args.yes:
        base.append("-y")
    return base


def phase_baseline_sweep(args) -> None:
    """MNIST + EMNIST x alpha x missing -- the bulk of the paper grid."""
    if args.skip_baseline:
        print("[skip] --skip_baseline given")
        return
    banner("STAGE 1  MNIST/EMNIST sweep (Tables 2-3) + 0% baseline column")
    cmd = [
        PY, "run_paper_pipeline.py",
        "--datasets", *args.datasets,
        "--alphas", *[str(a) for a in args.alphas],
        "--missing_rates", *[str(m) for m in args.missing_rates],
        "--algorithms", *args.algorithms,
        *common_orchestrator_args(args, missing_pattern="random"),
    ]
    run(cmd, args.dry_run)


def phase_ucihar(args) -> None:
    if args.skip_real or args.skip_ucihar:
        print("[skip] UCI HAR phase skipped")
        return
    banner("STAGE 2  UCI HAR (paper Sec.5.6) -- one pass per missing-pattern")
    # The plain 'ucihar' preset force-overrides --missing_pattern back to
    # 'random'. Use 'ucihar_3mech' so the explicit pattern wins.
    for pat in args.har_patterns:
        sub_cmd = [
            PY, "run_paper_pipeline.py",
            "--paper_preset", "ucihar_3mech",
            *common_orchestrator_args(args, missing_pattern=pat),
        ]
        run(sub_cmd, args.dry_run)


def phase_pamap2(args) -> None:
    if args.skip_real or args.skip_pamap2:
        print("[skip] PAMAP2 phase skipped")
        return
    banner("STAGE 3  PAMAP2 (paper Sec.5.6 promotion to actual results)")
    cmd = [
        PY, "run_paper_pipeline.py",
        "--datasets", "PAMAP2",
        "--include_pamap2",
        "--alphas", *[str(a) for a in args.pamap2_alphas],
        "--missing_rates", *[str(m) for m in args.pamap2_missing_rates],
        "--algorithms", *args.algorithms,
        *common_orchestrator_args(args, missing_pattern="random"),
    ]
    run(cmd, args.dry_run)


def phase_metrics_table(args) -> None:
    """Stand-alone goal3 invocation across ALL discovered metrics_mr* dirs.

    The orchestrator already calls goal3 inside each stage; this is an extra
    sweep that picks up *every* missing-rate that has any data on disk, so the
    final long-form table covers everything you ran (MNIST, EMNIST, UCI HAR,
    PAMAP2)."""
    if args.skip_table:
        print("[skip] goal3 metrics-table phase skipped")
        return
    banner("STAGE 4  Master F1 / Precision / Recall table (goal3, all data)")
    inputs = []
    for d in sorted((ROOT / "results").glob("metrics_mr*")):
        if not d.is_dir():
            continue
        try:
            mr_int = int(d.name.replace("metrics_mr", ""))
        except ValueError:
            continue
        inputs += ["--input", f"{mr_int}:{d}"]
    if not inputs:
        if args.dry_run:
            inputs = ["--input", "0:results/metrics_mr0",
                      "--input", "10:results/metrics_mr10",
                      "--input", "20:results/metrics_mr20"]
        else:
            print("[skip] no results/metrics_mr<RR>/ directories on disk yet")
            return
    cmd = [PY, "goal3_metrics_table.py", *inputs,
           "--out_dir", "results/tables", "--avg", "both"]
    run(cmd, args.dry_run)


# ------------------------------------------------------------------ summary
def final_checklist(args) -> None:
    banner("DONE  --  paper-artifacts checklist")

    items = [
        ("AM-DAE declaration paragraph (paste into Sec.4.2)",
         "results/zero_missing_baseline/amdae_declaration.txt"),
        ("0%-missing accuracy summary (per dataset)",
         "results/zero_missing_baseline/<DS>_zero_missing.{txt,csv}"),
        ("Per-run training history HDF5",
         "results/models_mr<RR>/<TOKEN>_<ALGO>_<...>.h5"),
        ("Per-round y_true/y_pred HDF5",
         "results/metrics_mr<RR>/<TOKEN>/<ALGO>_<TOKEN>_round_<i>.h5"),
        ("F1/Precision/Recall WIDE table (paper-ready)",
         "results/tables/<DS>_metrics_wide.{csv,md,tex}"),
        ("F1/Precision/Recall LONG table (one row per cell)",
         "results/tables/metrics_table_long.csv"),
        ("F1-vs-round line chart per dataset",
         "results/figures/mr<RR>/<TOKEN>/f1_by_round.{png,csv}"),
        ("Per-class F1 heatmap (paper Figs 5-8)",
         "results/figures/mr<RR>/heatmap_f1/<TOKEN>/heatmap_f1.{png,csv}"),
        ("Last-round confusion matrix per algorithm (paper Figs 9-12)",
         "results/figures/mr<RR>/<TOKEN>/confusion_matrix_round_<R>_<ALGO>.{png,csv}"),
        ("Accuracy + training-loss side-by-side (paper Fig 13)",
         "results/experiment_summary/acc_loss_<DS>_alpha<a>_miss<m>.png"),
        ("Paper-style accuracy curves (means +/- std)",
         "results/experiment_summary/plot_<DS>_alpha<a>_miss<m>.png"),
        ("Imputation method comparison bar chart",
         "results/comprehensive_imputation_comparison.png"),
    ]
    if not (args.skip_real or args.skip_ucihar):
        items.append(
            ("UCI HAR per-pattern summary",
             "results/real_dataset_experiments/UCI_HAR-alpha0.5-ratio0.5_mr0.15.{txt,csv}"))
    if not (args.skip_real or args.skip_pamap2):
        items.append(
            ("PAMAP2 per-cell summary",
             "results/real_dataset_experiments/PAMAP2-alpha<a>-ratio<r>_mr<m>.{txt,csv}"))

    for title, where in items:
        print(f"  - {title}\n      {where}")

    print()
    print("What you still have to do BY HAND in the paper text:")
    print("  1. Add the 0% missing column to Tables 2 and 3 using")
    print("     results/zero_missing_baseline/<DS>_zero_missing.csv.")
    print("  2. Paste results/zero_missing_baseline/amdae_declaration.txt at")
    print("     the start of Sec.4.2 (formal AM-DAE-as-imputer declaration).")
    print("  3. Add a new sub-section between Sec.5.2 and Sec.5.3 for the")
    print("     F1 / Precision / Recall table from results/tables/.")
    print("  4. Update Sec.5.6:")
    print("       (a) replace the MCAR-only HAR numbers with the actual numbers")
    print("           from results/real_dataset_experiments/UCI_HAR_*.csv,")
    print("       (b) add MAR / MNAR rows from the same files,")
    print("       (c) promote PAMAP2 from 'recommended for future work' to a")
    print("           real results sub-section, using the PAMAP2 csvs.")
    print("  5. Replace Figs 5-8 with the new")
    print("     results/figures/mr*/heatmap_f1/<token>/heatmap_f1.png")
    print("  6. Replace Fig 13 with")
    print("     results/experiment_summary/acc_loss_<DS>_alpha*_miss*.png")


# ------------------------------------------------------------------ main
def apply_budget(args: argparse.Namespace) -> None:
    """Pre-canned wall-clock budgets that compress the sweep without losing
    paper-relevant cells (only seeds and round-count are reduced)."""
    if args.budget == "full":
        return
    if args.budget == "fast":
        # ~10-12 GPU-h: full grid, 1 seed, half rounds.
        if args.times == 3:           # only override defaults
            args.times = 1
        if args.num_glob_iters == 100:
            args.num_glob_iters = 50
        print("[budget=fast] times=1, num_glob_iters=50, full grid kept")
    elif args.budget == "minimal":
        # ~4-6 GPU-h: 1 seed, 50 rounds, 1 alpha (1.0), 2 missing-rates (0.1, 0.2).
        if args.times == 3:
            args.times = 1
        if args.num_glob_iters == 100:
            args.num_glob_iters = 50
        if args.alphas == [0.1, 1.0, 10.0]:
            args.alphas = [1.0]
        if args.missing_rates == [0.0, 0.1, 0.2]:
            args.missing_rates = [0.1, 0.2]
        if args.pamap2_alphas == [0.1, 1.0]:
            args.pamap2_alphas = [1.0]
        if args.pamap2_missing_rates == [0.1, 0.2]:
            args.pamap2_missing_rates = [0.1]
        if args.har_patterns == ["random", "mar", "mnar"]:
            args.har_patterns = ["random"]
        print("[budget=minimal] times=1, num_glob_iters=50, alphas=[1.0], "
              "missing_rates=[0.1,0.2], HAR patterns=[random], PAMAP2 trimmed")


def main() -> None:
    args = parse_args()
    apply_budget(args)

    # Sanity: orchestrator + scripts exist
    for must_exist in [
        "run_paper_pipeline.py",
        "goal1_zero_missing_baseline.py",
        "goal2_real_dataset_experiment.py",
        "goal3_metrics_table.py",
        "plot_per_class_f1_heatmap.py",
        "plot_experiment_results.py",
        "main.py",
    ]:
        if not (ROOT / must_exist).is_file():
            raise SystemExit(
                f"Missing {must_exist} in {ROOT}. Did you clone the full repo?")

    banner(
        f"FedGen-AMDAE  ::  produce_paper_results  ({'DRY RUN' if args.dry_run else 'LIVE'})\n"
        f"  num_glob_iters = {args.num_glob_iters}, local_epochs = {args.local_epochs},\n"
        f"  num_users      = {args.num_users}, batch_size = {args.batch_size},\n"
        f"  times (seeds)  = {args.times}, device = {args.device}\n"
        f"  skip_baseline  = {args.skip_baseline},  skip_table = {args.skip_table}\n"
        f"  skip_real      = {args.skip_real},  skip_ucihar = {args.skip_ucihar},\n"
        f"  skip_pamap2    = {args.skip_pamap2}\n"
        f"  HAR patterns   = {args.har_patterns}, alpha={args.har_alpha}, missing={args.har_missing_rate}\n"
        f"  PAMAP2 alphas  = {args.pamap2_alphas}, missing={args.pamap2_missing_rates}"
    )

    phase_baseline_sweep(args)
    phase_ucihar(args)
    phase_pamap2(args)
    phase_metrics_table(args)
    final_checklist(args)


if __name__ == "__main__":
    main()
