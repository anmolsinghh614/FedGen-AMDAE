#!/usr/bin/env python
"""
run_optionA_sweep.py
====================

Master orchestrator for the Option A paper sweep that produces every cell
the FedGen-AMDAE paper needs:

    Datasets        : EMnist-letters, UCI HAR, PAMAP2
    Heterogeneity a : {0.1, 1, 10}
    Missing rate    : {0.0, 0.10, 0.20}
    Mechanism       : MCAR (random) only -- MAR / MNAR are not in this sweep
    Algorithms      : FedAvg, FedProx, FedDistill, FedEnsemble, FedGen
    Imputer         : AM-DAE (forced via --force_imputer amdae) for the main
                      sweep so every "FedGen-AMDAE" row is unambiguously
                      AM-DAE-imputed; the imputer-ablation phase additionally
                      runs FedGen with {mean, median, zero, none}.

Per-cell namespacing:

    results/optionA/<dataset_short>/alpha<a>_miss<m>/<algo>/
        models/<TOKEN>_<algo>_<lr>_<num_users>u_<bs>b_<le>_<seed>.h5
        metrics/<TOKEN>/<algo>_<TOKEN>_round_<R>.h5

Resume-skip is per (cell, seed): the driver checks expected HDF5 paths
before invoking main.py and skips seeds whose summary file already exists.

Usage:

    # Stage 1 -- single seed across the full grid (~22-30 GPU-h)
    python run_optionA_sweep.py --stage 1 --device cuda

    # Stage 2 -- multi-seed (re-runs only the missing seeds; cheap if
    # Stage 1 already wrote seed 0)
    python run_optionA_sweep.py --stage 2 --device cuda --times 3

    # Imputer ablation only -- FedGen x {amdae, mean, median, zero, none}
    # at the headline cell (alpha=1, missing=10%) per dataset, --times 3
    python run_optionA_sweep.py --ablation --device cuda

    # Smaller subset, e.g. only EMNIST or only one alpha
    python run_optionA_sweep.py --stage 1 --datasets EMnist-letters --alphas 0.1
    python run_optionA_sweep.py --stage 1 --algorithms FedGen FedAvg

    # Print the plan, run nothing
    python run_optionA_sweep.py --dry_run --stage 1
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
PY = sys.executable
OUT_ROOT = ROOT / "results" / "optionA"

# ---------------------------------------------------------------- locked scope
ALPHAS_DEFAULT = [0.1, 1.0, 10.0]
MISSING_RATES_DEFAULT = [0.0, 0.10, 0.20]
ALGOS_DEFAULT = ["FedAvg", "FedProx", "FedDistill", "FedEnsemble", "FedGen"]
DATASETS_DEFAULT = ["EMnist-letters", "UCI HAR", "PAMAP2"]
SAMPLING_RATIO = 0.5
N_USERS_TOTAL = 20

# Per-dataset communication-round budget (matches paper conventions;
# EMNIST gets the FedGen paper's 200, real-data datasets get 100).
ROUNDS = {"EMnist-letters": 200, "UCI HAR": 100, "PAMAP2": 100}

# Headline cell for imputer ablation + headline figures.
HEADLINE_ALPHA = 1.0
HEADLINE_MISS = 0.10
ABLATION_IMPUTERS = ["amdae", "mean", "median", "zero", "none"]


# ---------------------------------------------------------------- helpers
def banner(msg: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def run(cmd: list, dry: bool = False, cwd: Optional[Path] = None,
        allow_fail: bool = False) -> int:
    print(">> " + " ".join(str(c) for c in cmd),
          f"(cwd={cwd or ROOT})", flush=True)
    if dry:
        return 0
    rc = subprocess.call([str(c) for c in cmd],
                         cwd=str(cwd) if cwd else str(ROOT))
    if rc != 0 and not allow_fail:
        print(f"[WARN] command exited rc={rc}", file=sys.stderr)
    return rc


# ---------------------------------------------------------------- adapters
def dataset_short(dataset: str) -> str:
    return {
        "EMnist-letters": "emnist",
        "UCI HAR": "ucihar",
        "PAMAP2": "pamap2",
    }[dataset]


def dataset_token(dataset: str, alpha: float) -> str:
    """Token passed to main.py --dataset (matches utils/model_utils.py)."""
    a = _fmt_alpha(alpha)
    return f"{dataset}-alpha{a}-ratio{SAMPLING_RATIO}"


def _fmt_alpha(alpha: float) -> str:
    """Render alpha consistently with the path conventions used by the
    Dirichlet generators (e.g. 1.0 -> '1.0', 0.1 -> '0.1', 10.0 -> '10.0').
    The generators all store the alpha in the dirname using `str(alpha)`,
    so we mirror that exactly to keep paths consistent."""
    return str(alpha)


def dataset_split_dir(dataset: str, alpha: float) -> Path:
    """Filesystem path where the Dirichlet split lives."""
    a = _fmt_alpha(alpha)
    if dataset == "EMnist-letters":
        return ROOT / "data" / "EMnist" / \
            f"u{N_USERS_TOTAL}-letters-alpha{a}-ratio{SAMPLING_RATIO}"
    if dataset == "UCI HAR":
        return ROOT / "data" / "UCI HAR" / \
            f"u{N_USERS_TOTAL}-alpha{a}-ratio{SAMPLING_RATIO}"
    if dataset == "PAMAP2":
        return ROOT / "data" / "PAMAP2" / \
            f"u{N_USERS_TOTAL}-alpha{a}-ratio{SAMPLING_RATIO}"
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------- pre-flight
def _ucihar_natural_path() -> Path:
    return ROOT / "data" / "UCI HAR" / "UCI HAR Dataset"


def _ucihar_generator_path() -> Path:
    return ROOT / "data" / "UCI HAR" / "data" / "UCI HAR Dataset"


def stage_ucihar_for_generator(dry: bool = False) -> None:
    """The UCI HAR Dirichlet generator hard-codes data_dir='./data'; bridge
    the canonical extract path to the generator's expected path."""
    src = _ucihar_natural_path()
    dst = _ucihar_generator_path()
    if (dst / "train").is_dir() and (dst / "test").is_dir():
        return
    if not (src / "train").is_dir():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"[dry] would symlink {dst} -> {src}")
        return
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst, target_is_directory=True)
    except (OSError, NotImplementedError):
        shutil.copytree(src, dst)


def ensure_split(args: argparse.Namespace, dataset: str, alpha: float) -> bool:
    """Generate the per-(dataset, alpha) Dirichlet split if missing.
    Returns True on success (or already-present), False on generator failure."""
    split_dir = dataset_split_dir(dataset, alpha)
    if (split_dir / "train").is_dir() and (split_dir / "test").is_dir():
        print(f"[ok] split present: {split_dir}")
        return True

    if dataset == "EMnist-letters":
        gen = ROOT / "data" / "EMnist" / "generate_niid_dirichlet.py"
        cwd = ROOT / "data" / "EMnist"
        cmd = [PY, str(gen),
               "--n_user", str(N_USERS_TOTAL),
               "--alpha", str(alpha),
               "--sampling_ratio", str(SAMPLING_RATIO),
               "--split", "letters"]
    elif dataset == "UCI HAR":
        if not _ucihar_natural_path().is_dir() and not args.dry_run:
            print(f"[ERROR] UCI HAR raw data not found at "
                  f"{_ucihar_natural_path()}. Download from "
                  "https://archive.ics.uci.edu/ml/machine-learning-databases/"
                  "00240/UCI%20HAR%20Dataset.zip and unzip into "
                  "data/UCI HAR/.")
            return False
        stage_ucihar_for_generator(dry=args.dry_run)
        gen = ROOT / "data" / "UCI HAR" / "generate_niid_dirichlet.py"
        cwd = ROOT / "data" / "UCI HAR"
        cmd = [PY, str(gen),
               "--n_user", str(N_USERS_TOTAL),
               "--alpha", str(alpha),
               "--sampling_ratio", str(SAMPLING_RATIO)]
    elif dataset == "PAMAP2":
        gen = ROOT / "goal2_real_dataset_experiment.py"
        cwd = ROOT
        cmd = [PY, str(gen),
               "--dataset_kind", "pamap2",
               "--alpha", str(alpha),
               "--sampling_ratio", str(SAMPLING_RATIO),
               "--n_user", str(N_USERS_TOTAL),
               "--prepare_only"]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    rc = run(cmd, dry=args.dry_run, cwd=cwd, allow_fail=True)
    if rc != 0 and not args.dry_run:
        print(f"[ERROR] data-prep failed for {dataset} alpha={alpha} (rc={rc})")
        return False
    return True


# ---------------------------------------------------------------- per-cell I/O
def cell_dir(dataset: str, alpha: float, miss: float,
             algo: str, suffix: str = "") -> Path:
    """Per-cell namespace root. `suffix` allows optional sub-bucketing
    (e.g. 'imputer_ablation/<imputer>') without changing the public layout."""
    base = OUT_ROOT / dataset_short(dataset) / \
        f"alpha{alpha}_miss{miss}" / algo
    if suffix:
        base = base / suffix
    return base


def expected_h5(args: argparse.Namespace, dataset: str, alpha: float,
                algo: str, seed: int, models_dir: Path) -> Path:
    """Mirror utils.model_utils.get_log_path so we know exactly which file
    the server will write."""
    token = dataset_token(dataset, alpha)
    name = (f"{token}_{algo}_{args.learning_rate}_"
            f"{args.num_users}u_{args.batch_size}b_"
            f"{args.local_epochs}_{seed}")
    if "FedGen" in algo:
        name += "_embed0"
        if int(args.gen_batch_size) != int(args.batch_size):
            name += f"_gb{args.gen_batch_size}"
    return models_dir / f"{name}.h5"


def _relocate_per_round_metrics(args: argparse.Namespace,
                                metrics_dir: Path, seed: int) -> None:
    """main.py / utils.metrics_utils write per-round dumps to the static path
    `results/metrics/<TOKEN>/...`. The per-round HDF5 filenames do NOT
    include the seed, so back-to-back seeds for the same cell would
    silently overwrite each other.

    To preserve per-seed F1 (needed for mean +/- std in the paper tables),
    we move each seed's dumps into a seed-specific sub-folder:

        <cell>/metrics/seed_<s>/<TOKEN>/<algo>_<TOKEN>_round_<R>.h5

    paper_table_optionA.py reads these per-seed dumps to compute
    mean +/- std Macro-F1 across seeds.
    """
    live = ROOT / "results" / "metrics"
    if not live.is_dir() or args.dry_run:
        return
    seed_dir = metrics_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    for sub in live.iterdir():
        target = seed_dir / sub.name
        target.mkdir(parents=True, exist_ok=True)
        if sub.is_dir():
            for h5 in sub.iterdir():
                shutil.move(str(h5), str(target / h5.name))
        else:
            shutil.move(str(sub), str(target / sub.name))
    shutil.rmtree(live, ignore_errors=True)


def _check_no_stale_metrics() -> bool:
    """Refuse to start a fresh cell if a previous crash left
    `results/metrics/` lying around -- mixing it into our cell would
    misattribute someone else's per-round dumps."""
    live = ROOT / "results" / "metrics"
    if live.is_dir() and any(live.iterdir()):
        print(f"[ERROR] leftover {live} found; refusing to start a new cell.")
        print("        Move it out of the way (e.g. `mv results/metrics _stale`) "
              "and re-run.")
        return False
    return True


# ---------------------------------------------------------------- training
def train_cell(args: argparse.Namespace, dataset: str, alpha: float,
               miss: float, algo: str, seeds_wanted: int,
               force_imputer: Optional[str] = None,
               cell_suffix: str = "") -> None:
    """Train all `seeds_wanted` seeds for one (dataset, alpha, miss, algo)
    cell. Uses --seed_start so already-completed seeds are not re-run."""
    base = cell_dir(dataset, alpha, miss, algo, suffix=cell_suffix)
    models_dir = base / "models"
    metrics_dir = base / "metrics"
    for d in (models_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    todo_seeds = [s for s in range(seeds_wanted)
                  if not expected_h5(args, dataset, alpha, algo,
                                     s, models_dir).exists()]
    if not todo_seeds:
        print(f"[skip] {dataset_short(dataset):>6} a={alpha} m={miss} "
              f"{algo:>11s}{(' (' + cell_suffix + ')') if cell_suffix else ''} "
              f"-- all {seeds_wanted} seed(s) already trained")
        return

    rounds = ROUNDS[dataset]
    token = dataset_token(dataset, alpha)

    for s in todo_seeds:
        if not args.dry_run and not _check_no_stale_metrics():
            return

        cmd = [PY, "main.py",
               "--dataset", token,
               "--algorithm", algo,
               "--missing_rate", str(miss),
               "--missing_pattern", "random",
               "--num_glob_iters", str(rounds),
               "--local_epochs", str(args.local_epochs),
               "--num_users", str(args.num_users),
               "--batch_size", str(args.batch_size),
               "--gen_batch_size", str(args.gen_batch_size),
               "--learning_rate", str(args.learning_rate),
               "--times", "1",
               "--seed_start", str(s),
               "--device", args.device,
               "--result_path", str(models_dir)]
        if force_imputer:
            cmd += ["--force_imputer", force_imputer]

        rc = run(cmd, dry=args.dry_run, allow_fail=True)
        if rc != 0:
            print(f"[WARN] training failed: {dataset} a={alpha} m={miss} "
                  f"{algo} seed={s} (rc={rc})", file=sys.stderr)
        # Whether the training failed or not, relocate any per-round dumps
        # so they don't pollute the next cell.
        _relocate_per_round_metrics(args, metrics_dir, s)


# ---------------------------------------------------------------- phases
def phase_main_sweep(args: argparse.Namespace) -> None:
    """The (dataset x alpha x missing x algo x seeds) grid."""
    seeds_wanted = args.times if args.stage == 2 else 1

    banner(f"MAIN SWEEP  stage={args.stage}  seeds={seeds_wanted}  "
           f"force_imputer={args.force_imputer or 'auto-pick (AM-DAE)'}")

    for dataset in args.datasets:
        for alpha in args.alphas:
            ok = ensure_split(args, dataset, alpha)
            if not ok and not args.dry_run:
                print(f"[ERROR] skipping all cells for {dataset} alpha={alpha}")
                continue
            for miss in args.missing_rates:
                for algo in args.algorithms:
                    train_cell(args, dataset, alpha, miss, algo,
                               seeds_wanted=seeds_wanted,
                               force_imputer=args.force_imputer)


def phase_imputer_ablation(args: argparse.Namespace) -> None:
    """FedGen x {amdae, mean, median, zero, none} at the headline cell of
    each dataset, --times args.times (default 3). Output is namespaced
    under <cell>/imputer_ablation/<imputer>/{models,metrics}/."""
    seeds_wanted = max(1, args.times)
    banner(f"IMPUTER ABLATION  cell=alpha{HEADLINE_ALPHA}/miss{HEADLINE_MISS}  "
           f"imputers={ABLATION_IMPUTERS}  seeds={seeds_wanted}")

    for dataset in args.datasets:
        ok = ensure_split(args, dataset, HEADLINE_ALPHA)
        if not ok and not args.dry_run:
            print(f"[ERROR] skipping ablation for {dataset}")
            continue
        for imputer in ABLATION_IMPUTERS:
            train_cell(args, dataset, HEADLINE_ALPHA, HEADLINE_MISS,
                       "FedGen", seeds_wanted=seeds_wanted,
                       force_imputer=imputer,
                       cell_suffix=f"imputer_ablation/{imputer}")


# ---------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Scope (defaults match the locked Option A grid).
    p.add_argument("--datasets", nargs="*", default=DATASETS_DEFAULT,
                   choices=DATASETS_DEFAULT,
                   help=f"Subset of datasets to run (default: all 3).")
    p.add_argument("--alphas", nargs="*", type=float, default=ALPHAS_DEFAULT,
                   help=f"alpha values (default: {ALPHAS_DEFAULT}).")
    p.add_argument("--missing_rates", nargs="*", type=float,
                   default=MISSING_RATES_DEFAULT,
                   help=f"Missing rates (default: {MISSING_RATES_DEFAULT}).")
    p.add_argument("--algorithms", nargs="*", default=ALGOS_DEFAULT,
                   choices=ALGOS_DEFAULT,
                   help=f"FL algorithms (default: all 5).")

    # Training knobs (mirror main.py defaults).
    p.add_argument("--local_epochs", type=int, default=20)
    p.add_argument("--num_users", type=int, default=10,
                   help="Sampled users per round (separate from N_USERS_TOTAL=20 "
                        "which controls the size of the on-disk Dirichlet split).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gen_batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--times", type=int, default=3,
                   help="Total seeds wanted in Stage 2 (default 3 for "
                        "journal-grade std-devs). Stage 1 always runs --times 1.")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    # Per-dataset round overrides (defaults are ROUNDS[]).
    p.add_argument("--num_glob_iters_emnist", type=int, default=None,
                   help=f"Override rounds for EMNIST (default {ROUNDS['EMnist-letters']}).")
    p.add_argument("--num_glob_iters_ucihar", type=int, default=None,
                   help=f"Override rounds for UCI HAR (default {ROUNDS['UCI HAR']}).")
    p.add_argument("--num_glob_iters_pamap2", type=int, default=None,
                   help=f"Override rounds for PAMAP2 (default {ROUNDS['PAMAP2']}).")

    # Phase selection.
    p.add_argument("--stage", type=int, default=1, choices=[1, 2],
                   help="1: main sweep with seeds=1 (Stage 1). "
                        "2: main sweep with seeds=times (Stage 2; "
                        "skip-resume already takes care of seed-0).")
    p.add_argument("--ablation", action="store_true",
                   help="Run ONLY the imputer-ablation phase "
                        "(FedGen x 5 imputers at headline cell).")

    # Override the imputer for the main sweep. By default we hard-force
    # 'amdae' so every "FedGen-AMDAE" row is reviewer-bulletproof.
    p.add_argument("--force_imputer", default="amdae",
                   choices=["amdae", "mean", "median", "zero", "none", "auto"],
                   help="Imputer for the MAIN sweep (default amdae). "
                        "'auto' means: don't pass --force_imputer to "
                        "main.py; let the patched composite pick.")

    p.add_argument("--dry_run", action="store_true",
                   help="Print every command without executing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Translate 'auto' to None so main.py's --force_imputer arg is omitted.
    if args.force_imputer == "auto":
        args.force_imputer = None

    # Apply per-dataset round overrides (if any). These mutate the module-
    # level ROUNDS dict, which train_cell() reads.
    if args.num_glob_iters_emnist is not None:
        ROUNDS["EMnist-letters"] = args.num_glob_iters_emnist
    if args.num_glob_iters_ucihar is not None:
        ROUNDS["UCI HAR"] = args.num_glob_iters_ucihar
    if args.num_glob_iters_pamap2 is not None:
        ROUNDS["PAMAP2"] = args.num_glob_iters_pamap2

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    banner(
        f"FedGen-AMDAE  ::  Option A sweep  "
        f"({'DRY RUN' if args.dry_run else 'LIVE'})\n"
        f"  datasets         = {args.datasets}\n"
        f"  alphas           = {args.alphas}\n"
        f"  missing_rates    = {args.missing_rates}\n"
        f"  algorithms       = {args.algorithms}\n"
        f"  rounds           = {ROUNDS}\n"
        f"  local_epochs     = {args.local_epochs}\n"
        f"  num_users (samp) = {args.num_users}\n"
        f"  batch_size       = {args.batch_size}\n"
        f"  times (seeds)    = {args.times}\n"
        f"  stage            = {args.stage}\n"
        f"  ablation only?   = {args.ablation}\n"
        f"  force_imputer    = {args.force_imputer or '(auto)'}\n"
        f"  device           = {args.device}\n"
        f"  out_root         = {OUT_ROOT}"
    )

    if args.ablation:
        phase_imputer_ablation(args)
    else:
        phase_main_sweep(args)

    banner("DONE  Option A sweep")
    print(f"\nResults under: {OUT_ROOT}")
    print("Next steps:")
    print("  python paper_table_optionA.py --input-root results/optionA "
          "--metric Accuracy")
    print("  python paper_table_optionA.py --input-root results/optionA "
          "--metric MacroF1")
    print("  python paper_table_optionA.py --input-root results/optionA "
          "--imputer_ablation")
    print("  python paper_dashboard.py     --input-root results/optionA")


if __name__ == "__main__":
    main()
