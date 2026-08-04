"""
generate_niid_dirichlet.py  --  CIFAR-10 Dirichlet split
========================================================

Materialise a Dirichlet-partitioned client split of CIFAR-10 that is
byte-compatible with the rest of the codebase (same schema as EMNIST /
Mnist / FedISIC / HAM10000).

Pipeline:

    1. torchvision.datasets.CIFAR10(download=True) grabs the 50k train /
       10k test pool into data/CIFAR10/raw/ (idempotent).
    2. Per-channel z-score normalisation (statistics computed on the
       training pool once, cached alongside the split).
    3. Rearrange by class -> 10 buckets.
    4. Dirichlet allocation with parameter --alpha across --n_user
       clients, mirroring the FedISIC / HAM10000 / EMNIST layout.

Output:

    data/CIFAR10/u{n_user}-alpha{a}-ratio{r}/train/train.pt
    data/CIFAR10/u{n_user}-alpha{a}-ratio{r}/test/test.pt

The .pt files are dictionaries with the same schema every other
dataset uses:

    {
      'users': ['f_00000', 'f_00001', ...],
      'user_data': {'f_00000': {'x': FloatTensor[N, 3, 32, 32],
                                'y': LongTensor[N]}, ...},
      'num_samples': [n0, n1, ...]
    }

Usage:

    python generate_niid_dirichlet.py --n_user 20 --alpha 1.0 \
                                       --sampling_ratio 0.5
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "raw"
N_CLASSES = 10
IMG_SIZE = 32

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------- raw pool
def _load_torchvision_cifar10() -> tuple[np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray]:
    """Return (train_x uint8 [N,3,32,32], train_y int64 [N],
                test_x uint8 [M,3,32,32],  test_y int64 [M]).

    torchvision handles the auto-download + CRC check; we simply
    re-shape (N, 32, 32, 3) -> (N, 3, 32, 32) for CIFAR-style layout.
    """
    from torchvision.datasets import CIFAR10

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    tr = CIFAR10(root=str(RAW_ROOT), train=True,  download=True)
    te = CIFAR10(root=str(RAW_ROOT), train=False, download=True)

    tr_x = np.asarray(tr.data, dtype=np.uint8).transpose(0, 3, 1, 2)
    tr_y = np.asarray(tr.targets, dtype=np.int64)
    te_x = np.asarray(te.data, dtype=np.uint8).transpose(0, 3, 1, 2)
    te_y = np.asarray(te.targets, dtype=np.int64)
    return tr_x, tr_y, te_x, te_y


def _normalise(images_u8: np.ndarray,
               mean: np.ndarray | None = None,
               std:  np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel z-score. If (mean, std) are provided they are
    applied verbatim; otherwise computed from images_u8."""
    x = images_u8.astype(np.float32) / 255.0
    if mean is None:
        mean = x.mean(axis=(0, 2, 3), keepdims=True)
    if std is None:
        std = x.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    x = (x - mean) / std
    return x, mean, std


def rearrange_data_by_class(data: np.ndarray, targets: np.ndarray,
                            n_class: int) -> list[np.ndarray]:
    return [data[targets == i] for i in range(n_class)]


def get_dataset() -> tuple[list[np.ndarray], list[np.ndarray], int, int]:
    """Return (train_by_class, test_by_class, n_train, n_test)."""
    tr_x_u8, tr_y, te_x_u8, te_y = _load_torchvision_cifar10()

    # Compute normalisation on the train pool, apply to both.
    tr_x, mean, std = _normalise(tr_x_u8)
    te_x, _,   _   = _normalise(te_x_u8, mean=mean, std=std)

    train_by_class = rearrange_data_by_class(tr_x, tr_y, N_CLASSES)
    test_by_class  = rearrange_data_by_class(te_x, te_y, N_CLASSES)

    print(f"[cifar10] TRAIN: N={len(tr_x)}  per-class:",
          [len(v) for v in train_by_class])
    print(f"[cifar10] TEST:  N={len(te_x)}  per-class:",
          [len(v) for v in test_by_class])
    return train_by_class, test_by_class, len(tr_x), len(te_x)


# ---------------------------------------------------------------- Dirichlet
def divide_train_data(data, n_sample, SRC_CLASSES, NUM_USERS,
                      min_sample, alpha=1.0, sampling_ratio=0.5):
    """Mirror of Dirichlet allocation used by EMNIST / FedISIC / HAM10000."""
    min_sample = 10
    min_size = 0
    while min_size < min_sample:
        print("Try to find valid data separation")
        idx_batch = [{} for _ in range(NUM_USERS)]
        samples_per_user = [0 for _ in range(NUM_USERS)]
        max_samples_per_user = sampling_ratio * n_sample / NUM_USERS
        for l in SRC_CLASSES:
            idx_l = list(range(len(data[l])))
            np.random.shuffle(idx_l)
            if sampling_ratio < 1:
                samples_for_l = int(min(max_samples_per_user,
                                        int(sampling_ratio * len(data[l]))))
                idx_l = idx_l[:samples_for_l]
                print(l, len(data[l]), len(idx_l))
            proportions = np.random.dirichlet(np.repeat(alpha, NUM_USERS))
            proportions = np.array(
                [p * (n_per_user < max_samples_per_user)
                 for p, n_per_user in zip(proportions, samples_per_user)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_l)) \
                .astype(int)[:-1]
            for u, new_idx in enumerate(np.split(idx_l, proportions)):
                idx_batch[u][l] = new_idx.tolist()
                samples_per_user[u] += len(idx_batch[u][l])
        min_size = min(samples_per_user)

    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    Labels = [set() for _ in range(NUM_USERS)]
    for u, user_idx_batch in enumerate(idx_batch):
        for l, indices in user_idx_batch.items():
            if not indices:
                continue
            X[u] += data[l][indices].tolist()
            y[u] += (l * np.ones(len(indices))).tolist()
            Labels[u].add(l)
    return X, y, Labels, idx_batch, samples_per_user


def divide_test_data(NUM_USERS, SRC_CLASSES, test_data, Labels, unknown_test):
    test_X = [[] for _ in range(NUM_USERS)]
    test_y = [[] for _ in range(NUM_USERS)]
    idx = {l: 0 for l in SRC_CLASSES}
    for user in trange(NUM_USERS):
        user_sampled_labels = SRC_CLASSES if unknown_test else list(Labels[user])
        for l in user_sampled_labels:
            num_samples = max(1, int(len(test_data[l]) / NUM_USERS))
            take = min(num_samples, len(test_data[l]) - idx[l])
            if take <= 0:
                continue
            test_X[user] += test_data[l][idx[l]:idx[l] + take].tolist()
            test_y[user] += (l * np.ones(take)).tolist()
            idx[l] += take
    return test_X, test_y


# ---------------------------------------------------------------- entry
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--format", "-f", default="pt", choices=["pt"])
    p.add_argument("--min_sample", type=int, default=10)
    p.add_argument("--sampling_ratio", type=float, default=0.5)
    p.add_argument("--unknown_test", type=int, default=0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--n_user", type=int, default=20)
    args = p.parse_args()

    NUM_USERS = args.n_user
    print(f"[cifar10] n_user={NUM_USERS}  alpha={args.alpha}  "
          f"sampling_ratio={args.sampling_ratio}")

    path_prefix = f"u{NUM_USERS}-alpha{args.alpha}-ratio{args.sampling_ratio}"

    def process_user_data(mode, data, n_sample, SRC_CLASSES,
                          Labels=None, unknown_test=0):
        if mode == "train":
            X, y, Labels, idx_batch, samples_per_user = divide_train_data(
                data, n_sample, SRC_CLASSES, NUM_USERS,
                args.min_sample, args.alpha, args.sampling_ratio)
        else:
            X, y = divide_test_data(
                NUM_USERS, SRC_CLASSES, data, Labels, unknown_test)

        dataset = {"users": [], "user_data": {}, "num_samples": []}
        for i in range(NUM_USERS):
            uname = "f_{0:05d}".format(i)
            dataset["users"].append(uname)
            dataset["user_data"][uname] = {
                "x": torch.tensor(X[i], dtype=torch.float32),
                "y": torch.tensor(y[i], dtype=torch.int64),
            }
            dataset["num_samples"].append(len(X[i]))
        print(f"{mode.upper()} #sample by user:", dataset["num_samples"])

        out_dir = ROOT / path_prefix / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"{mode}.pt", "wb") as f:
            torch.save(dataset, f)
        print(f"  wrote {out_dir / (mode + '.pt')}")

        if mode == "train":
            return Labels
        return None

    train_by_class, test_by_class, n_train, n_test = get_dataset()
    SRC_CLASSES = list(range(N_CLASSES))
    random.shuffle(SRC_CLASSES)
    Labels = process_user_data("train", train_by_class, n_train, SRC_CLASSES)
    process_user_data("test", test_by_class, n_test, SRC_CLASSES,
                      Labels=Labels, unknown_test=args.unknown_test)
    print("[cifar10] split generation complete.")


if __name__ == "__main__":
    main()
