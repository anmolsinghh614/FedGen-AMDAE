"""
=============================================================================
WISDM v1.1 (Human Activity Recognition, tri-axial accelerometer)
Federated Non-IID Split (Dirichlet)
=============================================================================

Structure mirrors ``data/UCI HAR/generate_niid_dirichlet.py`` so the rest of
the pipeline (``main.py``, ``model_config.py``, ``run_optionA_sweep.py``,
``paper_*``) treats WISDM identically to UCI HAR:

  * Input samples reshape to a single-channel 24x24 tensor -- Dropping
    WISDM into the slot vacated by PAMAP2 requires zero changes to the
    CNN backbone that the pipeline uses for UCI HAR.
  * Output layout under ``./u<N>-alpha<a>-ratio<r>/{train,test}/*.pt`` is
    a torch-serialised dict compatible with FedGen's ``read_user_data``.

Feature representation
----------------------
* Sampling rate: 20 Hz (WISDM standard).
* Window: **192 samples (9.6 s), 50% overlap**  -> 192 x 3 axes = 576
  scalars per window, reshaped as (1, 24, 24). No zero-padding needed --
  the 24x24 grid holds exactly the flattened raw signal.
* Per-axis z-score normalisation is computed on the combined train+test
  pool (matches the UCI HAR pipeline's global normalisation).
* Chronological 80/20 split PER USER (so test rows come from the tail of
  each subject's recording, avoiding leakage from adjacent windows).

USAGE
    cd data/WISDM
    python generate_niid_dirichlet.py --n_user 20 --alpha 0.1 --sampling_ratio 0.5
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

# Local sibling module (same directory).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from download_wisdm import (  # noqa: E402
    load_wisdm_dataframe,
    ACTIVITY_TO_INT,
    N_CLASSES,
    SAMPLING_HZ,
)

random.seed(42)
np.random.seed(42)

# ---------- Feature-shape constants ------------------------------------------
WINDOW_SAMPLES = 192      # 9.6 s @ 20 Hz -- times 3 axes = 576 = 24*24
WINDOW_STRIDE  = 96       # 50% overlap
NUM_AXES       = 3
IMG_SIZE       = 24
FEATURE_DIM    = WINDOW_SAMPLES * NUM_AXES        # = 576
assert FEATURE_DIM == IMG_SIZE * IMG_SIZE, \
    "WISDM window layout must produce exactly 24*24 features"

TEST_FRACTION = 0.20      # last 20% of each user's timeline is test


# ---------- Data loading + windowing -----------------------------------------
def _window_one_user(user_df) -> tuple:
    """Sliding-window a single subject's rows.

    Returns
    -------
    X : np.ndarray, shape (N_windows, 1, 24, 24), dtype float32
    y : np.ndarray, shape (N_windows,)          , dtype int64
    """
    # WISDM rows are already in acquisition order (ascending timestamp)
    # after we group-and-sort by (user, timestamp). Trust that and slide.
    user_df = user_df.sort_values("timestamp", kind="mergesort")
    xs = user_df["x"].to_numpy(dtype=np.float32)
    ys = user_df["y"].to_numpy(dtype=np.float32)
    zs = user_df["z"].to_numpy(dtype=np.float32)
    labels_int = user_df["activity"].map(ACTIVITY_TO_INT).to_numpy(
        dtype=np.int64)

    n = len(xs)
    if n < WINDOW_SAMPLES:
        return (np.empty((0, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32),
                np.empty((0,), dtype=np.int64))

    X_list = []
    y_list = []
    for start in range(0, n - WINDOW_SAMPLES + 1, WINDOW_STRIDE):
        end = start + WINDOW_SAMPLES
        # A window's label = the modal activity in that window.
        # In practice WISDM windows are highly homogeneous (users perform
        # one activity for many seconds); modal label matches the majority.
        seg_labels = labels_int[start:end]
        vals, counts = np.unique(seg_labels, return_counts=True)
        modal = vals[np.argmax(counts)]
        # Skip windows that span an activity change if the modal class
        # is < 80% of the window (rare; keeps the dataset clean).
        if counts.max() < 0.8 * WINDOW_SAMPLES:
            continue

        seg_x = xs[start:end]
        seg_y = ys[start:end]
        seg_z = zs[start:end]
        # Interleave [x0,y0,z0,x1,y1,z1,...] so each column of the 24x24
        # grid is one axis-triple. Layout is arbitrary but consistent.
        flat = np.empty(FEATURE_DIM, dtype=np.float32)
        flat[0::3] = seg_x
        flat[1::3] = seg_y
        flat[2::3] = seg_z
        X_list.append(flat.reshape(1, IMG_SIZE, IMG_SIZE))
        y_list.append(int(modal))

    if not X_list:
        return (np.empty((0, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32),
                np.empty((0,), dtype=np.int64))
    return (np.stack(X_list, axis=0).astype(np.float32),
            np.array(y_list, dtype=np.int64))


def load_wisdm(data_dir: str = None):
    """Return ``(X_train, y_train, X_test, y_test)`` in the same shape /
    dtype convention as UCI HAR's ``load_uci_har``.

    * Downloads the raw file if missing (via ``download_wisdm``).
    * Windows every subject.
    * Chronological 80/20 split per subject.
    * Global per-axis z-score normalisation fit on all training windows.
    """
    dest_dir = Path(data_dir) if data_dir else _HERE
    df = load_wisdm_dataframe(dest_dir, auto_download=True)

    X_train_parts, y_train_parts = [], []
    X_test_parts,  y_test_parts  = [], []

    users = sorted(df["user"].unique())
    print(f"[WISDM] windowing {len(users)} subjects "
          f"(window={WINDOW_SAMPLES} samples @ {SAMPLING_HZ}Hz, "
          f"stride={WINDOW_STRIDE})")

    for u in users:
        X_u, y_u = _window_one_user(df[df["user"] == u])
        if len(X_u) == 0:
            continue
        # Chronological split -- last TEST_FRACTION of windows go to test.
        n_test = max(1, int(round(TEST_FRACTION * len(X_u))))
        split = len(X_u) - n_test
        X_train_parts.append(X_u[:split])
        y_train_parts.append(y_u[:split])
        X_test_parts.append(X_u[split:])
        y_test_parts.append(y_u[split:])

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    X_test  = np.concatenate(X_test_parts,  axis=0)
    y_test  = np.concatenate(y_test_parts,  axis=0)

    # ------- Global per-axis z-score normalisation (fit on TRAIN only) ---
    # Undo the interleave for stat computation, then rescale in-place.
    #   X shape: (N, 1, 24, 24) -> flatten to (N, 576) -> stride view
    def _stats(X):
        flat = X.reshape(len(X), FEATURE_DIM)
        return (flat[:, 0::3], flat[:, 1::3], flat[:, 2::3])

    xs_train, ys_train, zs_train = _stats(X_train)
    means = np.array([xs_train.mean(), ys_train.mean(), zs_train.mean()],
                     dtype=np.float32)
    stds  = np.array([xs_train.std() + 1e-8,
                      ys_train.std() + 1e-8,
                      zs_train.std() + 1e-8], dtype=np.float32)

    def _apply(X):
        flat = X.reshape(len(X), FEATURE_DIM)
        flat[:, 0::3] = (flat[:, 0::3] - means[0]) / stds[0]
        flat[:, 1::3] = (flat[:, 1::3] - means[1]) / stds[1]
        flat[:, 2::3] = (flat[:, 2::3] - means[2]) / stds[2]
        # Rescale to roughly [-1, 1] using a soft-clip via tanh -- avoids
        # long tails from acceleration spikes dominating the input range.
        return np.tanh(flat.reshape(-1, 1, IMG_SIZE, IMG_SIZE))

    X_train = _apply(X_train).astype(np.float32)
    X_test  = _apply(X_test).astype(np.float32)

    print(f"[WISDM]   TRAIN windows: {len(X_train):,}    "
          f"TEST windows: {len(X_test):,}")
    return X_train, y_train, X_test, y_test


def rearrange_data_by_class(data, targets, n_class):
    return [data[targets == i] for i in range(n_class)]


def get_dataset(mode: str = "train", data_dir=None):
    X_train, y_train, X_test, y_test = load_wisdm(data_dir)
    if mode == "train":
        data, targets = X_train, y_train
    else:
        data, targets = X_test, y_test

    print("Rearrange data by class...")
    data_by_class = rearrange_data_by_class(data, targets, N_CLASSES)
    n_sample = len(data)
    print(f"{mode.upper()} SET:")
    print(f"  Total #samples: {n_sample}. sample shape: {data[0].shape}")
    print(f"  #samples per class:\n {[len(v) for v in data_by_class]}")
    return data_by_class, n_sample, N_CLASSES


# ---------- Dirichlet allocation (mirrors UCI HAR exactly) --------------------
def divide_train_data(data, n_sample, SRC_CLASSES, NUM_USERS, min_sample,
                      alpha=0.5, sampling_ratio=0.5):
    min_size = 0
    max_retries = 100
    current_alpha = alpha
    attempt = 0

    while min_size < min_sample:
        attempt += 1
        if attempt > max_retries:
            print(f"WARNING: Could not satisfy min_sample={min_sample} "
                  f"after {max_retries} attempts. Using best result so far "
                  f"(min_size={min_size}).")
            break
        if attempt % 20 == 0:
            old_alpha = current_alpha
            current_alpha = min(current_alpha * 1.5, 10.0)
            print(f"  [Attempt {attempt}] Increasing alpha "
                  f"{old_alpha:.3f} -> {current_alpha:.3f} for better balance")
        if attempt <= 3 or attempt % 10 == 0:
            print(f"Try to find valid data separation (attempt {attempt}, "
                  f"alpha={current_alpha:.3f})")

        idx_batch = [{} for _ in range(NUM_USERS)]
        samples_per_user = np.zeros(NUM_USERS, dtype=int)
        max_samples_per_user = sampling_ratio * n_sample / NUM_USERS

        for l in SRC_CLASSES:
            n_l = len(data[l])
            idx_l = np.random.permutation(n_l)
            if sampling_ratio < 1:
                samples_for_l = int(min(max_samples_per_user,
                                        int(sampling_ratio * n_l)))
                idx_l = idx_l[:samples_for_l]
                if attempt == 1:
                    print(l, n_l, len(idx_l))

            proportions = np.random.dirichlet(
                np.repeat(current_alpha, NUM_USERS))
            mask = (samples_per_user < max_samples_per_user).astype(float)
            proportions = proportions * mask
            prop_sum = proportions.sum()
            if prop_sum == 0:
                proportions = np.ones(NUM_USERS) / NUM_USERS
            else:
                proportions = proportions / prop_sum

            split_points = (np.cumsum(proportions) * len(idx_l)).astype(int)[:-1]
            split_indices = np.split(idx_l, split_points)
            for u, new_idx in enumerate(split_indices):
                idx_batch[u][l] = new_idx
                samples_per_user[u] += len(new_idx)
        min_size = samples_per_user.min()

    print("Processing users (vectorized)...")

    X = [None] * NUM_USERS
    y = [None] * NUM_USERS
    Labels = [set() for _ in range(NUM_USERS)]

    def _assemble_user(u):
        x_parts, y_parts = [], []
        labels_u = set()
        for l, indices in idx_batch[u].items():
            if len(indices) == 0:
                continue
            x_parts.append(data[l][indices])
            y_parts.append(np.full(len(indices), l, dtype=np.int64))
            labels_u.add(l)
        if x_parts:
            x_u = np.concatenate(x_parts, axis=0)
            y_u = np.concatenate(y_parts, axis=0)
        else:
            sample_shape = data[0].shape[1:]  # (1, 24, 24)
            x_u = np.empty((0,) + sample_shape, dtype=np.float32)
            y_u = np.empty((0,), dtype=np.int64)
        return u, x_u, y_u, labels_u

    n_workers = min(NUM_USERS, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_assemble_user, u)
                   for u in range(NUM_USERS)]
        for f in futures:
            u, x_u, y_u, labels_u = f.result()
            X[u] = x_u
            y[u] = y_u
            Labels[u] = labels_u

    return X, y, Labels, idx_batch, samples_per_user.tolist()


def divide_test_data(NUM_USERS, SRC_CLASSES, test_data, Labels, unknown_test):
    test_X = [None] * NUM_USERS
    test_y = [None] * NUM_USERS
    idx = {l: 0 for l in SRC_CLASSES}

    for user in range(NUM_USERS):
        user_sampled_labels = (SRC_CLASSES if unknown_test
                               else list(Labels[user]))
        x_parts, y_parts = [], []
        for l in user_sampled_labels:
            num_samples = int(len(test_data[l]) / NUM_USERS)
            if num_samples + idx[l] > len(test_data[l]):
                num_samples = max(0, len(test_data[l]) - idx[l])
            x_parts.append(test_data[l][idx[l]:idx[l] + num_samples])
            y_parts.append(np.full(num_samples, l, dtype=np.int64))
            idx[l] += num_samples

        if x_parts:
            test_X[user] = np.concatenate(x_parts, axis=0)
            test_y[user] = np.concatenate(y_parts, axis=0)
        else:
            sample_shape = test_data[0].shape[1:]
            test_X[user] = np.empty((0,) + sample_shape, dtype=np.float32)
            test_y[user] = np.empty((0,), dtype=np.int64)

        assert len(test_X[user]) == len(test_y[user])
    return test_X, test_y


# ---------- Entry point ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", "-f", type=str, default="pt",
                        choices=["pt", "json"])
    parser.add_argument("--n_class", type=int, default=N_CLASSES)
    parser.add_argument("--min_sample", type=int, default=64,
                        help="Min samples per user (>= batch_size=64).")
    parser.add_argument("--sampling_ratio", type=float, default=0.5)
    parser.add_argument("--unknown_test", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_user", type=int, default=20)
    args = parser.parse_args()

    t0 = time.perf_counter()

    print()
    print("=" * 60)
    print("WISDM v1.1 -- Federated Non-IID Split (Dirichlet)")
    print("  window={}s @ {}Hz, stride={}s"
          .format(WINDOW_SAMPLES / SAMPLING_HZ, SAMPLING_HZ,
                  WINDOW_STRIDE / SAMPLING_HZ))
    print("=" * 60)
    print(f"Number of users: {args.n_user}")
    print(f"Number of classes: {args.n_class}")
    print(f"Min # of samples per user: {args.min_sample}")
    print(f"Alpha for Dirichlet: {args.alpha}")
    print(f"Sampling ratio: {args.sampling_ratio}")

    NUM_USERS = args.n_user
    path_prefix = f"u{NUM_USERS}-alpha{args.alpha}-ratio{args.sampling_ratio}"

    def process_user_data(mode, data, n_sample, SRC_CLASSES,
                          Labels=None, unknown_test=0):
        t_start = time.perf_counter()

        if mode == "train":
            X, y, Labels, idx_batch, samples_per_user = divide_train_data(
                data, n_sample, SRC_CLASSES, NUM_USERS,
                args.min_sample, args.alpha, args.sampling_ratio)
        else:
            assert Labels is not None or unknown_test
            X, y = divide_test_data(NUM_USERS, SRC_CLASSES, data,
                                    Labels, unknown_test)

        dataset = {"users": [], "user_data": {}, "num_samples": []}
        for i in range(NUM_USERS):
            uname = "f_{0:05d}".format(i)
            dataset["users"].append(uname)
            dataset["user_data"][uname] = {
                "x": torch.from_numpy(np.ascontiguousarray(X[i])).float(),
                "y": torch.from_numpy(np.ascontiguousarray(y[i])).long()}
            dataset["num_samples"].append(len(X[i]))
        print(f"{mode.upper()} #sample by user:", dataset["num_samples"])

        data_path = f"./{path_prefix}/{mode}"
        os.makedirs(data_path, exist_ok=True)
        out_file = os.path.join(data_path, f"{mode}.{args.format}")
        if args.format == "json":
            raise NotImplementedError("json output not supported (tensor).")
        with open(out_file, "wb") as fh:
            print(f"Dumping {mode} data => {out_file}")
            torch.save(dataset, fh)
        print(f"  {mode.upper()} phase total: {time.perf_counter() - t_start:.3f}s")

        if mode == "train":
            for u in range(NUM_USERS):
                info = ""
                total = 0
                for l in sorted(list(Labels[u])):
                    n_l = len(idx_batch[u][l])
                    total += n_l
                    info += f"c={l},n={n_l}| "
                print(f"{samples_per_user[u]} samples in total")
                print(info)
                print(f"{len(Labels[u])} Labels/ {total} Number of "
                      f"training samples for user [{u}]:")
            return Labels

    print("\nReading + windowing source dataset ...")
    train_data, n_train, SRC_N_CLASS = get_dataset(mode="train")
    test_data,  n_test,  _           = get_dataset(mode="test")

    SRC_CLASSES = list(range(SRC_N_CLASS))
    random.shuffle(SRC_CLASSES)
    print(f"{len(SRC_CLASSES)} labels in total.")

    print("\n--- Processing TRAIN data ---")
    Labels = process_user_data("train", train_data, n_train, SRC_CLASSES)

    print("\n--- Processing TEST data ---")
    process_user_data("test", test_data, n_test, SRC_CLASSES,
                      Labels=Labels, unknown_test=args.unknown_test)

    print("\n" + "=" * 60)
    print("Finish Generating User samples")
    print(f"TOTAL TIME: {time.perf_counter() - t0:.3f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
