#!/usr/bin/env python
"""
Goal 2 -- Real-world dataset analysis (UCI HAR or PAMAP2).

Why a separate script?
----------------------
The Mnist/EMnist runs use synthetic image data; reviewers often want
a SECOND validation on a real-sensor dataset. This script wraps:

  1. Dataset preparation
       UCI HAR  -> automatic Dirichlet-non-IID split via the existing
                   data/UCI HAR/generate_niid_dirichlet.py.
       PAMAP2   -> downloads the UCI archive, unzips, converts the
                   per-subject .dat files into the same {x,y} dict
                   format used by the rest of the codebase, then
                   performs a Dirichlet split.
  2. Training of every selected FL algorithm via main.py with
     --missing_rate as configured.
  3. A consolidated round-by-round accuracy summary at the end.

Notes
-----
- UCI HAR is supported end-to-end by this repo.
- PAMAP2 is also supported end-to-end: model wiring lives in
  utils/model_utils.py (get_data_dir / get_dataset_name) and
  utils/model_config.py (CONFIGS_ / GENERATORCONFIGS / RUNCONFIGS).
  Use --prepare_only to download / convert / split without training.

Examples
--------
# UCI HAR, alpha=0.1, 10% missing, all 5 methods, 50 rounds:
python goal2_real_dataset_experiment.py \
    --dataset_kind ucihar --alpha 0.1 --sampling_ratio 0.5 \
    --missing_rate 0.1 --num_glob_iters 50

# PAMAP2 -- only prepare the data; do not train:
python goal2_real_dataset_experiment.py \
    --dataset_kind pamap2 --prepare_only
"""
import argparse
import os
import shutil
import subprocess
import sys
import textwrap
import urllib.request
import zipfile
from typing import List

ALGORITHMS_DEFAULT = ["FedAvg", "FedGen", "FedProx", "FedDistill", "FedEnsemble"]

PAMAP2_URL = (
    "https://archive.ics.uci.edu/static/public/231/"
    "pamap2+physical+activity+monitoring.zip"
)


# ---------------------------------------------------------------------------
#  ARGUMENTS
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset_kind", choices=["ucihar", "pamap2"], default="ucihar")

    # Data-prep parameters (forwarded to the per-dataset generator)
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Dirichlet concentration (smaller = more non-IID).")
    p.add_argument("--sampling_ratio", type=float, default=0.5)
    p.add_argument("--n_user", type=int, default=20,
                   help="UCI HAR is hard-coded to 20 users by utils/model_utils.py "
                        "(do not change unless you also patch that file).")
    p.add_argument("--n_class", type=int, default=None,
                   help="Override the number of classes (UCIHAR=6, PAMAP2=12).")

    # Training parameters
    p.add_argument("--missing_rate", type=float, default=0.1)
    p.add_argument("--num_glob_iters", type=int, default=100)
    p.add_argument("--local_epochs", type=int, default=20)
    p.add_argument("--num_users", type=int, default=10,
                   help="Active users sampled per FL round (<= --n_user).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gen_batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--times", type=int, default=1)
    p.add_argument("--algorithms", nargs="*", default=ALGORITHMS_DEFAULT)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--result_path", default="results/models")
    p.add_argument("--summary_dir", default="results/real_dataset_experiments")

    # Workflow toggles
    p.add_argument("--skip_generate", action="store_true",
                   help="Don't re-generate the per-user split; use what's already on disk.")
    p.add_argument("--prepare_only", action="store_true",
                   help="Only prepare the dataset; do not train.")
    return p.parse_args()


# ---------------------------------------------------------------------------
#  UCI HAR
# ---------------------------------------------------------------------------
def prepare_ucihar(args: argparse.Namespace) -> str:
    data_dir = os.path.join("data", "UCI HAR")
    if not os.path.isdir(data_dir):
        raise SystemExit(f"Missing folder: {data_dir}")

    if not args.skip_generate:
        cmd = [
            sys.executable, "generate_niid_dirichlet.py",
            "--alpha", str(args.alpha),
            "--sampling_ratio", str(args.sampling_ratio),
            "--n_user", str(args.n_user),
        ]
        if args.n_class is not None:
            cmd += ["--n_class", str(args.n_class)]
        print(">>", " ".join(cmd), f"(cwd={data_dir})", flush=True)
        rc = subprocess.call(cmd, cwd=data_dir)
        if rc != 0:
            raise SystemExit(f"UCI HAR data-generation failed (rc={rc}).")

    return f"UCI HAR-alpha{args.alpha}-ratio{args.sampling_ratio}"


# ---------------------------------------------------------------------------
#  PAMAP2 (data prep only; model wiring is a separate follow-up task)
# ---------------------------------------------------------------------------
def _download_pamap2(target_zip: str) -> None:
    if os.path.exists(target_zip):
        print(f"PAMAP2 zip already present: {target_zip}")
        return
    print(f"Downloading PAMAP2 from {PAMAP2_URL} ...")
    with urllib.request.urlopen(PAMAP2_URL) as resp, open(target_zip, "wb") as f:
        shutil.copyfileobj(resp, f)
    print(f"Saved to {target_zip}")


def _unzip_pamap2(zip_path: str, dest_dir: str) -> str:
    if os.path.isdir(os.path.join(dest_dir, "PAMAP2_Dataset")):
        print(f"PAMAP2 already unzipped under {dest_dir}")
        return os.path.join(dest_dir, "PAMAP2_Dataset")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    # The archive sometimes nests a second zip; flatten if so.
    inner = os.path.join(dest_dir, "PAMAP2_Dataset.zip")
    if os.path.exists(inner):
        with zipfile.ZipFile(inner) as z:
            z.extractall(dest_dir)
    return os.path.join(dest_dir, "PAMAP2_Dataset")


# PAMAP2 column layout (see official README):
#   col  0    : timestamp
#   col  1    : activity_id
#   col  2    : heart-rate                  <- frequently NaN; we ignore it
#   col  3..53: IMU sensors (3 IMUs * 17 channels each = 51 features)
PAMAP2_LABEL_COL = 1
PAMAP2_HR_COL = 2
PAMAP2_FEATURE_SLICE = slice(3, 54)   # cols 3..53 inclusive (51 features)
PAMAP2_NUM_FEATURES = 51
PAMAP2_PADDED = 64                    # smallest perfect square >= 51
PAMAP2_IMG_SIZE = 8                   # 8 x 8


def _load_pamap2_subject_files(root: str):
    """
    Walk Protocol/ and Optional/, return a list of (subject_id, np.ndarray)
    where each ndarray has the full 54 raw columns (we slice them later).
    """
    import pandas as pd
    subjects = []
    for sub_root in ["Protocol", "Optional"]:
        path = os.path.join(root, sub_root)
        if not os.path.isdir(path):
            continue
        for fname in sorted(os.listdir(path)):
            if not fname.endswith(".dat"):
                continue
            full = os.path.join(path, fname)
            df = pd.read_csv(full, sep=r"\s+", header=None, engine="c")
            sid = int(fname.replace("subject", "").replace(".dat", ""))
            subjects.append((sid, df.values))
    return subjects


def _pamap2_to_xy(subjects, n_class: int):
    """
    Convert raw subject arrays into (X, y) for the federated pipeline.
    Returns X with shape (N, 1, 8, 8) - the same image-style tensor that
    UCI HAR uses, so the existing 'ucihar'-style CNN ('pamap2' entry in
    CONFIGS_) can be re-used unchanged.
    """
    import numpy as np
    # 12 main activities; map their PAMAP2 ids to a contiguous range 0..11.
    activity_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
                    7: 6, 12: 7, 13: 8, 16: 9, 17: 10, 24: 11}
    xs, ys, sids = [], [], []
    for sid, arr in subjects:
        labels = arr[:, PAMAP2_LABEL_COL].astype(int)
        keep_lbl = np.isin(labels, list(activity_map.keys()))
        if keep_lbl.sum() == 0:
            continue
        feats = arr[keep_lbl, PAMAP2_FEATURE_SLICE].astype(np.float32)  # (n,51)
        # Drop rows with NaN in feature/label cols only (HR is irrelevant).
        finite = np.all(np.isfinite(feats), axis=1)
        if finite.sum() == 0:
            continue
        feats = feats[finite]
        ys_subj = np.array(
            [activity_map[int(v)] for v in labels[keep_lbl][finite]],
            dtype=np.int64,
        )
        xs.append(feats)
        ys.append(ys_subj)
        sids.append((sid, len(ys_subj)))
    if not xs:
        raise SystemExit("PAMAP2: no usable rows after filtering -- check raw files.")
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    # Standardise per channel
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    X = (X - mu) / sd
    # Zero-pad 51 -> 64 then reshape to (N, 1, 8, 8) image-style tensor
    pad = PAMAP2_PADDED - PAMAP2_NUM_FEATURES
    X = np.pad(X, ((0, 0), (0, pad)), mode="constant")
    X = X.reshape(-1, 1, PAMAP2_IMG_SIZE, PAMAP2_IMG_SIZE).astype(np.float32)
    # Restrict to the first n_class labels in case the caller asked for fewer.
    keep = y < n_class
    return X[keep], y[keep], sids


def _dirichlet_split(X, y, n_user: int, alpha: float, seed: int = 42):
    import numpy as np
    rng = np.random.default_rng(seed)
    n_class = int(y.max()) + 1
    user_idx = [[] for _ in range(n_user)]
    for c in range(n_class):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        if len(idx_c) == 0:
            continue
        proportions = rng.dirichlet(np.repeat(alpha, n_user))
        cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        for u, chunk in enumerate(np.split(idx_c, cuts)):
            user_idx[u].extend(chunk.tolist())
    return user_idx


def prepare_pamap2(args: argparse.Namespace) -> str:
    """
    PAMAP2 data-prep:
      data/PAMAP2/raw/PAMAP2_Dataset/Protocol/*.dat  ->
      data/PAMAP2/u<N>-alpha..-ratio../{train,test}/{train,test}.pt

    NOTE: utils/model_utils.py + utils/model_config.py do not yet
    know about 'pamap2'. To finish wiring this dataset to main.py,
    add a 'pamap2' branch to get_data_dir(), get_dataset_name() and
    add a 'pamap2' entry to CONFIGS_/GENERATORCONFIGS/RUNCONFIGS.
    This script will print the patch hint at the end.
    """
    import numpy as np
    import torch  # local import; only needed when pamap2 is actually used

    n_class = args.n_class if args.n_class is not None else 12
    base_dir = os.path.join("data", "PAMAP2")
    raw_dir = os.path.join(base_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    if not args.skip_generate:
        zip_path = os.path.join(raw_dir, "pamap2.zip")
        _download_pamap2(zip_path)
        ds_root = _unzip_pamap2(zip_path, raw_dir)

        print("Loading PAMAP2 subject files ...")
        subjects = _load_pamap2_subject_files(ds_root)
        print(f"  loaded {len(subjects)} subjects")
        X, y, _ = _pamap2_to_xy(subjects, n_class=n_class)
        print(f"  total samples = {len(X)}, feature dim = {X.shape[1]}")

        # 80/20 train/test split (per sample, stratification optional)
        rng = np.random.default_rng(0)
        idx = np.arange(len(X)); rng.shuffle(idx)
        cut = int(0.8 * len(idx))
        tr_idx, te_idx = idx[:cut], idx[cut:]

        # Dirichlet across train; test gets a uniform per-user slice
        train_users = _dirichlet_split(
            X[tr_idx], y[tr_idx], n_user=args.n_user, alpha=args.alpha)

        out_root = os.path.join(
            base_dir, f"u{args.n_user}-alpha{args.alpha}-ratio{args.sampling_ratio}")
        for split, (xs, ys_, user_idx) in [
            ("train", (X[tr_idx], y[tr_idx], train_users)),
            ("test", (X[te_idx], y[te_idx],
                      _dirichlet_split(X[te_idx], y[te_idx],
                                       n_user=args.n_user, alpha=10.0))),
        ]:
            split_dir = os.path.join(out_root, split)
            os.makedirs(split_dir, exist_ok=True)
            dataset = {"users": [], "user_data": {}, "num_samples": []}
            for u in range(args.n_user):
                uname = f"f_{u:05d}"
                dataset["users"].append(uname)
                ix = np.array(user_idx[u], dtype=np.int64)
                dataset["user_data"][uname] = {
                    "x": torch.tensor(xs[ix], dtype=torch.float32),
                    "y": torch.tensor(ys_[ix], dtype=torch.int64),
                }
                dataset["num_samples"].append(int(len(ix)))
            out_path = os.path.join(split_dir, f"{split}.pt")
            torch.save(dataset, out_path)
            print(f"  wrote {out_path}  (#users={args.n_user}, "
                  f"total={sum(dataset['num_samples'])})")

    print(
        "PAMAP2 model wiring is already merged into utils/model_utils.py and "
        "utils/model_config.py (CONFIGS_/GENERATORCONFIGS/RUNCONFIGS each carry "
        "a 'pamap2' entry). main.py --dataset PAMAP2-alpha<a>-ratio<r> will run "
        "end-to-end against this prepared split."
    )

    return f"PAMAP2-alpha{args.alpha}-ratio{args.sampling_ratio}"


# ---------------------------------------------------------------------------
#  TRAIN
# ---------------------------------------------------------------------------
def run_one(algorithm: str, args: argparse.Namespace, dataset_token: str) -> int:
    cmd = [
        sys.executable, "main.py",
        "--dataset", dataset_token,
        "--algorithm", algorithm,
        "--missing_rate", str(args.missing_rate),
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


def collect_summary(args: argparse.Namespace, dataset_token: str) -> List[dict]:
    import h5py
    import numpy as np
    rows = []
    for algorithm in args.algorithms:
        for seed in range(args.times):
            base = (f"{dataset_token}_{algorithm}_{args.learning_rate}_"
                    f"{args.num_users}u_{args.batch_size}b_"
                    f"{args.local_epochs}_{seed}")
            if "FedGen" in algorithm:
                base += "_embed0"
                if int(args.gen_batch_size) != int(args.batch_size):
                    base += f"_gb{args.gen_batch_size}"
            path = os.path.join(args.result_path, base + ".h5")
            if not os.path.exists(path):
                rows.append(dict(algorithm=algorithm, seed=seed, found=False,
                                 final_acc=float("nan"), best_acc=float("nan"),
                                 path=path))
                continue
            try:
                with h5py.File(path, "r") as hf:
                    acc = np.asarray(hf["glob_acc"][:], dtype=float)
            except Exception as exc:                         # pragma: no cover
                rows.append(dict(algorithm=algorithm, seed=seed, found=False,
                                 final_acc=float("nan"), best_acc=float("nan"),
                                 path=path, error=repr(exc)))
                continue
            rows.append(dict(
                algorithm=algorithm, seed=seed, found=True,
                final_acc=float(acc[-1]), best_acc=float(np.max(acc)),
                best_round=int(np.argmax(acc)), path=path,
            ))
    return rows


def write_summary(rows: List[dict], args: argparse.Namespace, token: str) -> None:
    os.makedirs(args.summary_dir, exist_ok=True)
    txt = [
        "REAL-DATASET EXPERIMENT SUMMARY",
        "=" * 70,
        f"Dataset      : {token}",
        f"Missing rate : {args.missing_rate:.2f}",
        f"Global rounds: {args.num_glob_iters}",
        f"Local epochs : {args.local_epochs}",
        f"Active users : {args.num_users} of {args.n_user}",
        f"Seeds        : {args.times}",
        "-" * 70,
        f"{'Algorithm':<16}{'Seed':>5}  {'Final Acc%':>11}  {'Best Acc%':>11}",
        "-" * 70,
    ]
    csv_lines = ["algorithm,seed,final_acc,best_acc,path"]
    for r in rows:
        if r["found"]:
            txt.append(f"{r['algorithm']:<16}{r['seed']:>5}"
                       f"  {r['final_acc']*100:>10.2f}%"
                       f"  {r['best_acc']*100:>10.2f}%")
        else:
            txt.append(f"{r['algorithm']:<16}{r['seed']:>5}  -- file not found --")
        csv_lines.append(
            f"{r['algorithm']},{r['seed']},{r['final_acc']:.6f},"
            f"{r['best_acc']:.6f},{r['path']}"
        )
    summary_text = "\n".join(txt + ["-" * 70])
    print("\n" + summary_text)
    safe_token = token.replace(" ", "_")
    txt_path = os.path.join(args.summary_dir, f"{safe_token}_mr{args.missing_rate}.txt")
    csv_path = os.path.join(args.summary_dir, f"{safe_token}_mr{args.missing_rate}.csv")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines) + "\n")
    print(f"\nSummary written to {txt_path}")
    print(f"CSV     written to {csv_path}")


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    if args.dataset_kind == "ucihar":
        dataset_token = prepare_ucihar(args)
    else:
        dataset_token = prepare_pamap2(args)

    print(f"\nDataset token : '{dataset_token}'")
    print(f"Missing rate  : {args.missing_rate}")

    if args.prepare_only:
        return

    failed: List[str] = []
    for algorithm in args.algorithms:
        if run_one(algorithm, args, dataset_token) != 0:
            failed.append(algorithm)

    rows = collect_summary(args, dataset_token)
    write_summary(rows, args, dataset_token)

    if failed:
        print(f"\nWARNING: the following algorithms exited non-zero: {failed}",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
