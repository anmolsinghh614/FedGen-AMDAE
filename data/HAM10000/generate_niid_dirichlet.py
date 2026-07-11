"""
generate_niid_dirichlet.py  --  HAM10000 Dirichlet split
=========================================================

Materialise a Dirichlet-partitioned client split of the HAM10000
training pool (ISIC 2018 Task 3) that is byte-compatible with the
rest of the Run 3 codebase.

Pipeline:

    1. If the raw ISIC-2018 pool is missing, invoke download_ham10000.py
       to fetch it from the ISIC public S3 bucket. Idempotent.
    2. Load ISIC2018_Task3_Training_GroundTruth.csv (columns: image,
       MEL, NV, BCC, AKIEC, BKL, DF, VASC) and derive an integer label
       in {0..6} per image.
    3. Load every JPG in ISIC2018_Task3_Training_Input/, resize to
       32x32 RGB, and cache the resulting (N, 3, 32, 32) uint8 tensor
       plus (N,) int64 labels to raw/ham10000_cache_32x32.pt so that
       subsequent alpha values re-use the cache.
    4. Global per-channel z-score normalisation to float32.
    5. Chronological 80/20 train/test split per class.
    6. Dirichlet allocation, mirroring UCI HAR / EMNIST / WISDM /
       FedISIC.

Output:

    data/HAM10000/u{n_user}-alpha{a}-ratio{r}/train/train.pt
    data/HAM10000/u{n_user}-alpha{a}-ratio{r}/test/test.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import trange, tqdm

ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "raw"
CACHE_PATH = RAW_ROOT / "ham10000_cache_32x32.pt"

# HAM10000 ground-truth CSV column order:
# image, MEL, NV, BCC, AKIEC, BKL, DF, VASC
LABEL_COLS = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
N_CLASSES = len(LABEL_COLS)
IMG_SIZE = 32

random.seed(42)
np.random.seed(42)


def _ensure_raw_pool() -> Path:
    csv_path = RAW_ROOT / "ISIC2018_Task3_Training_GroundTruth.csv"
    img_dir = RAW_ROOT / "ISIC2018_Task3_Training_Input"
    if csv_path.is_file() and img_dir.is_dir():
        return RAW_ROOT
    print("[ham10000] raw pool missing, invoking download_ham10000.py ...")
    from download_ham10000 import download_ham10000
    download_ham10000(ROOT, metadata_only=False)
    return RAW_ROOT


def _build_cache() -> tuple[np.ndarray, np.ndarray]:
    if CACHE_PATH.is_file():
        blob = torch.load(CACHE_PATH, map_location="cpu")
        return blob["images"].numpy(), blob["labels"].numpy()

    from PIL import Image
    import csv

    raw = _ensure_raw_pool()
    gt_path = raw / "ISIC2018_Task3_Training_GroundTruth.csv"
    img_dir = raw / "ISIC2018_Task3_Training_Input"

    id2class: dict[str, int] = {}
    with open(gt_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col_idx = {name: header.index(name) for name in LABEL_COLS}
        img_col = header.index("image")
        for row in reader:
            if not row or not row[img_col]:
                continue
            probs = [float(row[col_idx[c]]) for c in LABEL_COLS]
            id2class[row[img_col]] = int(np.argmax(probs))
    print(f"[ham10000] parsed {len(id2class)} labels")

    files = sorted(img_dir.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(
            f"No JPG files found under {img_dir}. "
            f"Re-run `python download_ham10000.py` to fetch the pool."
        )

    images = np.zeros((len(files), 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    labels = np.zeros((len(files),), dtype=np.int64)
    kept = 0
    for f in tqdm(files, desc="loading + resizing HAM10000"):
        img_id = f.stem
        cls = id2class.get(img_id)
        if cls is None:
            continue
        try:
            with Image.open(f) as img:
                img = img.convert("RGB").resize(
                    (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                arr = np.asarray(img, dtype=np.uint8)
                images[kept] = arr.transpose(2, 0, 1)
                labels[kept] = cls
                kept += 1
        except (OSError, ValueError) as exc:
            print(f"[ham10000] skipping unreadable image {f.name}: {exc}",
                  file=sys.stderr)
            continue
    images = images[:kept]
    labels = labels[:kept]
    print(f"[ham10000] cached {kept} images, class histogram:",
          np.bincount(labels, minlength=N_CLASSES).tolist())

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": torch.from_numpy(images),
                "labels": torch.from_numpy(labels)},
               CACHE_PATH)
    return images, labels


def _normalise(images_u8: np.ndarray) -> np.ndarray:
    x = images_u8.astype(np.float32) / 255.0
    mean = x.mean(axis=(0, 2, 3), keepdims=True)
    std = x.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    return (x - mean) / std


def rearrange_data_by_class(data: np.ndarray, targets: np.ndarray,
                            n_class: int) -> list[np.ndarray]:
    return [data[targets == i] for i in range(n_class)]


def get_dataset(mode: str = "train",
                test_ratio: float = 0.20) -> tuple[list[np.ndarray], int, int]:
    imgs_u8, labels = _build_cache()
    imgs = _normalise(imgs_u8)

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


def divide_train_data(data, n_sample, SRC_CLASSES, NUM_USERS,
                      min_sample, alpha=1.0, sampling_ratio=0.5):
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
    print(f"[ham10000] n_user={NUM_USERS}  alpha={args.alpha}  "
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
    print("[ham10000] split generation complete.")


if __name__ == "__main__":
    main()
