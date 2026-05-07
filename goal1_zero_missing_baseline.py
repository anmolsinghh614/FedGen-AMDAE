#!/usr/bin/env python
"""
Goal 1 -- Original-paper baselines at 0% missing data.

What this script does
---------------------
1. Emits a clear "AMDAE Imputation Declaration" paragraph to stdout
   AND saves it to results/zero_missing_baseline/amdae_declaration.txt
   so it can be pasted into the paper / README.
2. Wraps main.py so that every supported FL method
   (FedAvg, FedProx, FedDistill, FedDistill-FL, FedEnsemble, FedGen)
   is run with --missing_rate 0.0, reproducing the original-paper
   conditions on the chosen dataset.
3. Aggregates the resulting accuracies into a single TXT/CSV
   summary so the 0%-missing column of your final table is ready
   to use.

Why a 0%-missing run still uses the AMDAE codepath
---------------------------------------------------
Even though apply_amdae_imputation() short-circuits when
missing_rate <= 0, the codepath is still invoked uniformly across
every server (FedAvg/FedProx/FedDistill/FedEnsemble/FedGen).
This is what the declaration block makes explicit for the paper.

Examples
--------
# Reproduce all 5 methods on EMnist alpha=0.1 (50 rounds, 1 seed):
python goal1_zero_missing_baseline.py \
    --dataset EMnist-alpha0.1-ratio0.1 \
    --num_glob_iters 50 --times 1

# Just emit the methodology paragraph, no training:
python goal1_zero_missing_baseline.py \
    --dataset EMnist-alpha0.1-ratio0.1 --declaration_only
"""
import argparse
import os
import subprocess
import sys
import textwrap
from typing import List

ALGORITHMS_DEFAULT = ["FedAvg", "FedGen", "FedProx", "FedDistill", "FedEnsemble"]

DECLARATION = textwrap.dedent(
    """
    ============================================================
    METHODOLOGY DECLARATION -- IMPUTATION (please cite verbatim)
    ============================================================
    All federated learning algorithms compared in this work
    (FedAvg, FedProx, FedDistill / FedDistill-FL, FedEnsemble,
    and FedGen) share a single, common imputation front-end:
    the Adaptive-Learned Median-Filled Deep Autoencoder (AM-DAE),
    based on:

        Y. Cui et al., "Imputation of Missing Values in Time
        Series Using an Adaptive-Learned Median-Filled Deep
        Autoencoder", IEEE Transactions on Cybernetics, 2023.

    Concretely, every server class -- FedAvg, FedProx,
    FedDistill, FedEnsemble, FedGen -- performs the same two
    pre-training calls:

        data = read_data(args.dataset)
        data = apply_amdae_imputation(
                   data, missing_rate=args.missing_rate)

    so the *same* AM-DAE imputer (with Mean / Median / Zero
    baselines auto-evaluated and the best one selected) repairs
    missing values before any local update or aggregation step.

    When --missing_rate = 0.0, apply_amdae_imputation() returns
    the data unchanged. The runs in this script therefore
    reproduce the baselines reported in the original FedGen /
    FedAvg / FedProx / FedDistill / FedEnsemble papers and
    serve as the "no-missingness" column of every table /
    figure produced downstream.
    ============================================================
    """
).strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True,
                   help="e.g. Mnist-alpha0.1-ratio0.5, EMnist-alpha0.1-ratio0.1, "
                        "'UCI HAR-alpha0.1-ratio0.5'.")
    p.add_argument("--algorithms", nargs="*", default=ALGORITHMS_DEFAULT,
                   help="Subset of algorithms to run.")
    p.add_argument("--num_glob_iters", type=int, default=100)
    p.add_argument("--local_epochs", type=int, default=20)
    p.add_argument("--num_users", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gen_batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--times", type=int, default=1)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--result_path", default="results/models",
                   help="Where main.py writes h5 logs (default: results/models).")
    p.add_argument("--declaration_only", action="store_true",
                   help="Only emit the AMDAE declaration; skip training.")
    p.add_argument("--summary_dir", default="results/zero_missing_baseline",
                   help="Where to write the declaration + accuracy summary.")
    return p.parse_args()


def build_h5_name(args: argparse.Namespace, algorithm: str, seed: int) -> str:
    """Mirrors utils.model_utils.get_log_path so we can locate the run's HDF5 file."""
    name = f"{args.dataset}_{algorithm}_{args.learning_rate}_{args.num_users}u_" \
           f"{args.batch_size}b_{args.local_epochs}_{seed}"
    if "FedGen" in algorithm:
        name += "_embed0"  # default embedding=0 in main.py
        if int(args.gen_batch_size) != int(args.batch_size):
            name += f"_gb{args.gen_batch_size}"
    return name


def run_one(algorithm: str, args: argparse.Namespace) -> int:
    cmd = [
        sys.executable, "main.py",
        "--dataset", args.dataset,
        "--algorithm", algorithm,
        "--missing_rate", "0.0",
        "--num_glob_iters", str(args.num_glob_iters),
        "--local_epochs", str(args.local_epochs),
        "--num_users", str(args.num_users),
        "--batch_size", str(args.batch_size),
        "--gen_batch_size", str(args.gen_batch_size),
        "--learning_rate", str(args.learning_rate),
        "--times", str(args.times),
        "--device", args.device,
        "--result_path", args.result_path,
    ]
    print("\n>>", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def collect_summary(args: argparse.Namespace) -> List[dict]:
    import h5py
    import numpy as np
    rows = []
    for algorithm in args.algorithms:
        for seed in range(args.times):
            base = build_h5_name(args, algorithm, seed)
            path = os.path.join(args.result_path, base + ".h5")
            if not os.path.exists(path):
                rows.append(dict(algorithm=algorithm, seed=seed, found=False,
                                 final_acc=float("nan"), best_acc=float("nan"),
                                 best_round=-1, path=path))
                continue
            try:
                with h5py.File(path, "r") as hf:
                    acc = np.asarray(hf["glob_acc"][:], dtype=float)
            except Exception as exc:                         # pragma: no cover
                rows.append(dict(algorithm=algorithm, seed=seed, found=False,
                                 final_acc=float("nan"), best_acc=float("nan"),
                                 best_round=-1, path=path,
                                 error=repr(exc)))
                continue
            rows.append(dict(
                algorithm=algorithm,
                seed=seed,
                found=True,
                final_acc=float(acc[-1]),
                best_acc=float(np.max(acc)),
                best_round=int(np.argmax(acc)),
                path=path,
            ))
    return rows


def write_summary(rows: List[dict], args: argparse.Namespace) -> None:
    os.makedirs(args.summary_dir, exist_ok=True)
    txt = [
        "0%-MISSING BASELINE -- ORIGINAL-PAPER REPRODUCTION",
        "=" * 70,
        f"Dataset      : {args.dataset}",
        f"Global rounds: {args.num_glob_iters}",
        f"Local epochs : {args.local_epochs}",
        f"Num users    : {args.num_users}",
        f"Batch size   : {args.batch_size}",
        f"Seeds        : {args.times}",
        "-" * 70,
        f"{'Algorithm':<16}{'Seed':>5}  {'Final Acc%':>11}  {'Best Acc%':>11}  {'Best round':>11}",
        "-" * 70,
    ]
    csv_lines = ["algorithm,seed,final_acc,best_acc,best_round,path"]
    for r in rows:
        if r["found"]:
            txt.append(
                f"{r['algorithm']:<16}{r['seed']:>5}"
                f"  {r['final_acc']*100:>10.2f}%"
                f"  {r['best_acc']*100:>10.2f}%"
                f"  {r['best_round']:>11d}"
            )
        else:
            txt.append(f"{r['algorithm']:<16}{r['seed']:>5}  -- file not found --")
        csv_lines.append(
            f"{r['algorithm']},{r['seed']},{r['final_acc']:.6f},"
            f"{r['best_acc']:.6f},{r['best_round']},{r['path']}"
        )
    txt.append("-" * 70)
    summary_text = "\n".join(txt)
    print("\n" + summary_text)

    base = f"{args.dataset}_zero_missing"
    txt_path = os.path.join(args.summary_dir, base + ".txt")
    csv_path = os.path.join(args.summary_dir, base + ".csv")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"\nSummary written to {txt_path}")
    print(f"CSV     written to {csv_path}")


def main() -> None:
    args = parse_args()

    print(DECLARATION)

    os.makedirs(args.summary_dir, exist_ok=True)
    decl_path = os.path.join(args.summary_dir, "amdae_declaration.txt")
    with open(decl_path, "w", encoding="utf-8") as f:
        f.write(DECLARATION + "\n")
    print(f"\nDeclaration written to {decl_path}")

    if args.declaration_only:
        return

    failed: List[str] = []
    for algorithm in args.algorithms:
        if run_one(algorithm, args) != 0:
            failed.append(algorithm)

    rows = collect_summary(args)
    write_summary(rows, args)

    if failed:
        print(f"\nWARNING: the following algorithms exited non-zero: {failed}",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
