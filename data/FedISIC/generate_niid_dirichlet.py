"""
generate_niid_dirichlet.py  --  FedISIC (ISIC 2019) Dirichlet split
====================================================================

Materialise a Dirichlet-partitioned client split of the ISIC 2019
training pool that is byte-compatible with the rest of the Run 3
codebase.

Pipeline:

    1. If the raw ISIC 2019 pool is missing, call download_fedisic.py
       to fetch it (S3, no auth). Fully idempotent.
    2. Load ISIC_2019_Training_GroundTruth.csv, parse each image's
       one-hot label into an integer class in {0..7}.
    3. For every JPG in ISIC_2019_Training_Input/, load with PIL,
       resize to 32x32 RGB, and cache the (N, 3, 32, 32) uint8 tensor
       and (N,) int64 labels to
           raw/fedisic_cache_32x32.pt
       so that subsequent alpha values do not re-process 25 331 images.
    4. Global per-channel z-score normalisation. Values are then stored
       as float32 tensors in shape (3, 32, 32) so downstream code sees
       CIFAR-style data.
    5. Chronological 80/20 train/test split *per class* (so every class
       is represented in both partitions, similar to how UCI HAR splits
       per user).
    6. Dirichlet allocation with parameter --alpha across --n_user
       clients, mirroring UCI HAR / EMNIST / WISDM output layout.

Output:

    data/FedISIC/u{n_user}-alpha{a}-ratio{r}/train/train.pt
    data/FedISIC/u{n_user}-alpha{a}-ratio{r}/test/test.pt

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
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import trange, tqdm

ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "raw"
CACHE_PATH = RAW_ROOT / "fedisic_cache_32x32.pt"

LABEL_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
N_CLASSES = len(LABEL_COLS)
IMG_SIZE = 32

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------- raw pool
def _ensure_raw_pool() -> Path:
    """Return the raw pool dir (downloading if needed). Import lazily so
    the module can be imported even in constrained environments."""
    if RAW_ROOT.is_dir() and \
       (RAW_ROOT / "ISIC_2019_Training_GroundTruth.csv").is_file() and \
       (RAW_ROOT / "ISIC_2019_Training_Input").is_dir():
        return RAW_ROOT
    print("[fedisic] raw pool missing, invoking download_fedisic.py ...")
    from download_fedisic import download_fedisic
    download_fedisic(ROOT, metadata_only=False)
    return RAW_ROOT


# ---------------------------------------------------------------- image cache
def _build_cache() -> tuple[np.ndarray, np.ndarray]:
    """Load every ISIC 2019 image, resize to IMG_SIZE, and cache as
    uint8 arrays. Returns (images uint8 [N,3,H,W], labels int64 [N])."""
    if CACHE_PATH.is_file():
        blob = torch.load(CACHE_PATH, map_location="cpu")
        return blob["images"].numpy(), blob["labels"].numpy()

    from PIL import Image
    import csv

    raw = _ensure_raw_pool()
    gt_path = raw / "ISIC_2019_Training_GroundTruth.csv"
    img_dir = raw / "ISIC_2019_Training_Input"

    # Parse (image_id -> class_idx) map from the one-hot ground truth CSV.
    id2class: dict[str, int] = {}
    with open(gt_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Column layout is `image, MEL, NV, BCC, AK, BKL, DF, VASC, SCC[, UNK]`
        col_idx = {name: header.index(name) for name in LABEL_COLS}
        img_col = header.index("image")
        for row in reader:
            if not row or not row[img_col]:
                continue
            probs = [float(row[col_idx[c]]) for c in LABEL_COLS]
            id2class[row[img_col]] = int(np.argmax(probs))
    print(f"[fedisic] parsed {len(id2class)} labels")

    # Load every JPG present in ISIC_2019_Training_Input/, resize to IMG_SIZE,
    # and stash into an ndarray. Missing labels are dropped silently.
    files = sorted(img_dir.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(
            f"No JPG files found under {img_dir}. "
            f"Re-run `python download_fedisic.py` to fetch the pool."
        )

    images = np.zeros((len(files), 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    labels = np.zeros((len(files),), dtype=np.int64)
    kept = 0
    for f in tqdm(files, desc="loading + resizing ISIC 2019"):
        img_id = f.stem
        cls = id2class.get(img_id)
        if cls is None:
            continue
        try:
            with Image.open(f) as img:
                img = img.convert("RGB").resize(
                    (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                arr = np.asarray(img, dtype=np.uint8)  # (H, W, 3)
                # Convert to (3, H, W)
                images[kept] = arr.transpose(2, 0, 1)
                labels[kept] = cls
                kept += 1
        except (OSError, ValueError) as exc:
            print(f"[fedisic] skipping unreadable image {f.name}: {exc}",
                  file=sys.stderr)
            continue
    images = images[:kept]
    labels = labels[:kept]
    print(f"[fedisic] cached {kept} images, class histogram:",
          np.bincount(labels, minlength=N_CLASSES).tolist())

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": torch.from_numpy(images),
                "labels": torch.from_numpy(labels)},
               CACHE_PATH)
    return images, labels


# ---------------------------------------------------------------- normalise
def _normalise(images_u8: np.ndarray) -> np.ndarray:
    """Per-channel z-score, then float32. images_u8 is (N, 3, H, W) uint8."""
    x = images_u8.astype(np.float32) / 255.0  # [0, 1]
    mean = x.mean(axis=(0, 2, 3), keepdims=True)
    std = x.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    x = (x - mean) / std
    return x


# ---------------------------------------------------------------- per-class
def rearrange_data_by_class(data: np.ndarray, targets: np.ndarray,
                            n_class: int) -> list[np.ndarray]:
    return [data[targets == i] for i in range(n_class)]


def get_dataset(mode: str = "train",
                test_ratio: float = 0.20) -> tuple[list[np.ndarray], int, int]:
    """Return (data_by_class, n_samples, n_class) for the requested mode."""
    imgs_u8, labels = _build_cache()
    imgs = _normalise(imgs_u8)

    # Split *per class* so every partition contains every class.
    rng = np.random.RandomState(42)
    train_idx: list[int] = []
    test_idx:  list[int] = []
    for c in range(N_CLASSES):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        cut = max(1, int(round(len(idx_c) * (1.0 - test_ratio))))
        train_idx.extend(idx_c[:cut].tolist())
        test_idx.extend(idx_c[cut:].tolist())
    train_idx = np.array(train_idx, dtype=np.int64)
    test_idx = np.array(test_idx,  dtype=np.int64)

    if mode == "train":
        x, y = imgs[train_idx], labels[train_idx]
    else:
        x, y = imgs[test_idx], labels[test_idx]

    by_class = rearrange_data_by_class(x, y, N_CLASSES)
    print(f"{mode.upper()} SET:  N={len(x)}, per-class:",
          [len(v) for v in by_class])
    return by_class, len(x), N_CLASSES


# ---------------------------------------------------------------- Dirichlet
def divide_train_data(data, n_sample, SRC_CLASSES, NUM_USERS,
                      min_sample, alpha=1.0, sampling_ratio=0.5):
    """Mirror of the Dirichlet allocation in EMNIST / UCI HAR / WISDM."""
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
    print(f"[fedisic] n_user={NUM_USERS}  alpha={args.alpha}  "
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

    train_data, n_train, _ = get_dataset("train")
    test_data,  n_test,  _ = get_dataset("test")
    SRC_CLASSES = list(range(N_CLASSES))
    random.shuffle(SRC_CLASSES)
    Labels = process_user_data("train", train_data, n_train, SRC_CLASSES)
    process_user_data("test", test_data, n_test, SRC_CLASSES,
                      Labels=Labels, unknown_test=args.unknown_test)
    print("[fedisic] split generation complete.")


if __name__ == "__main__":
    main()
