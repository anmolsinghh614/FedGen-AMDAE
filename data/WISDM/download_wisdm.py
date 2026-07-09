"""
================================================================================
WISDM v1.1 raw dataset -- auto-download + parse helper
================================================================================
Fetches ``WISDM_ar_latest.tar.gz`` (~11 MB) from Fordham's WISDM Lab, extracts
the raw text file ``WISDM_ar_v1.1_raw.txt`` (~68 MB), and returns a clean
pandas DataFrame with columns ``[user, activity, timestamp, x, y, z]``.

Design notes
------------
* The raw file has a notoriously flaky format: each row ends with a trailing
  ``;``, occasional rows are truncated / missing values, and separators
  sometimes drift. We parse defensively (skip malformed lines).
* Two callable entry points:
    ``download_wisdm(dest_dir)``   -- ensure the raw txt is on disk, download
                                     if needed. Returns the file path.
    ``load_wisdm_dataframe(...)``  -- return a cleaned pandas.DataFrame.
* No side effects at import time.

Standalone usage (once, from the ``data/WISDM/`` directory):
    python download_wisdm.py               # download only
    python download_wisdm.py --smoke_test  # download + parse + summary
================================================================================
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


WISDM_URL = (
    "http://www.cis.fordham.edu/wisdm/includes/datasets/latest/"
    "WISDM_ar_latest.tar.gz"
)
RAW_FILENAME = "WISDM_ar_v1.1_raw.txt"
TARBALL_NAME = "WISDM_ar_latest.tar.gz"

ACTIVITY_TO_INT = {
    "Walking":    0,
    "Jogging":    1,
    "Upstairs":   2,
    "Downstairs": 3,
    "Sitting":    4,
    "Standing":   5,
}
INT_TO_ACTIVITY = {v: k for k, v in ACTIVITY_TO_INT.items()}
N_CLASSES = 6
SAMPLING_HZ = 20


# --------------------------------------------------------------------------- I/O
def _stream_download(url: str, dest: Path, timeout: int = 120) -> None:
    """Stream `url` into `dest` using stdlib. No wget/curl dependency."""
    print(f"[WISDM] downloading\n    {url}\n  -> {dest}", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=timeout) as resp, \
         open(tmp, "wb") as f:
        for chunk in iter(lambda: resp.read(1024 * 256), b""):
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dest)
    size_mb = dest.stat().st_size / 1e6
    print(f"[WISDM] download complete: {size_mb:.1f} MB in "
          f"{time.time() - t0:.1f}s",
          flush=True)


def _extract_raw_txt_from_tarball(tarball: Path, out_dir: Path) -> Path:
    """Extract the WISDM raw txt from the tarball into out_dir. Returns the
    path to the extracted file. The tarball has a top-level folder like
    ``WISDM_ar_v1.1/`` -- we place the raw txt directly under ``out_dir``
    to keep the layout flat (mirrors what other loaders in this project
    expect)."""
    print(f"[WISDM] extracting {RAW_FILENAME} from {tarball} ...", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / RAW_FILENAME
    with tarfile.open(tarball, "r:gz") as tf:
        member = None
        for m in tf.getmembers():
            if m.name.endswith("/" + RAW_FILENAME) or m.name == RAW_FILENAME:
                member = m
                break
        if member is None:
            raise RuntimeError(
                f"{RAW_FILENAME} not found inside {tarball}. "
                f"Archive members: {[m.name for m in tf.getmembers()]}"
            )
        # tar member always uses forward slashes; strip its directory
        # prefix and write directly to `target`.
        src = tf.extractfile(member)
        if src is None:
            raise RuntimeError(f"could not open {member.name} inside tarball")
        with open(target, "wb") as dst:
            dst.write(src.read())
    print(f"[WISDM] extracted -> {target}  "
          f"({target.stat().st_size / 1e6:.1f} MB)", flush=True)
    return target


def download_wisdm(dest_dir: Path, force: bool = False) -> Path:
    """Ensure the raw txt is on disk. Return its path.

    * If ``<dest_dir>/WISDM_ar_v1.1_raw.txt`` already exists and force=False,
      just return the path (idempotent).
    * Otherwise download the tarball into ``<dest_dir>`` and extract the
      raw txt into the same directory.
    """
    dest_dir = Path(dest_dir)
    raw_path = dest_dir / RAW_FILENAME
    if raw_path.is_file() and not force:
        return raw_path

    tarball = dest_dir / TARBALL_NAME
    if not tarball.is_file() or force:
        _stream_download(WISDM_URL, tarball)
    else:
        print(f"[WISDM] tarball already present: {tarball}")

    _extract_raw_txt_from_tarball(tarball, dest_dir)
    if not raw_path.is_file():
        raise RuntimeError(
            f"expected {raw_path} after extract, but it is missing")
    return raw_path


# ------------------------------------------------------------------------ parse
def load_wisdm_dataframe(dest_dir: Path,
                         auto_download: bool = True) -> pd.DataFrame:
    """Return a DataFrame with columns ``[user, activity, timestamp, x, y, z]``.

    Robust to the raw file's known quirks:
      * trailing ``;`` on each line
      * a handful of malformed / truncated lines
      * occasional lines with 5 fields instead of 6

    Rows with missing values or bad numeric parsing are dropped.
    """
    dest_dir = Path(dest_dir)
    raw_path = dest_dir / RAW_FILENAME
    if not raw_path.is_file():
        if not auto_download:
            raise FileNotFoundError(
                f"{raw_path} missing and auto_download=False. Set "
                f"auto_download=True or download manually from {WISDM_URL}."
            )
        raw_path = download_wisdm(dest_dir)

    rows = []
    n_bad = 0
    with open(raw_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip().rstrip(";").rstrip(",")
            if not line:
                continue
            # A handful of rows have two records concatenated on one line;
            # split on ';' just in case.
            for chunk in line.split(";"):
                if not chunk.strip():
                    continue
                parts = chunk.split(",")
                if len(parts) != 6:
                    n_bad += 1
                    continue
                try:
                    u   = int(parts[0])
                    act = parts[1].strip()
                    ts  = int(parts[2])
                    x   = float(parts[3])
                    y   = float(parts[4])
                    z   = float(parts[5])
                except ValueError:
                    n_bad += 1
                    continue
                if act not in ACTIVITY_TO_INT:
                    n_bad += 1
                    continue
                rows.append((u, act, ts, x, y, z))

    if not rows:
        raise RuntimeError(
            f"parsed 0 valid rows from {raw_path}. "
            f"(bad rows skipped: {n_bad})"
        )
    df = pd.DataFrame(rows, columns=["user", "activity", "timestamp",
                                     "x", "y", "z"])
    print(f"[WISDM] parsed {len(df):,} valid rows "
          f"({n_bad:,} bad rows skipped)")
    print(f"[WISDM] {df['user'].nunique()} subjects, "
          f"{df['activity'].nunique()} activities, "
          f"class balance:\n{df['activity'].value_counts().to_string()}")
    return df


def _smoke_test(dest_dir: Path) -> int:
    df = load_wisdm_dataframe(dest_dir)
    print("\n=== SMOKE TEST ===")
    print(f"columns  : {list(df.columns)}")
    print(f"rows     : {len(df):,}")
    print(f"users    : {sorted(df['user'].unique())}")
    print(f"activity : {df['activity'].value_counts().to_dict()}")
    print(f"x range  : [{df['x'].min():.2f}, {df['x'].max():.2f}]")
    print(f"y range  : [{df['y'].min():.2f}, {df['y'].max():.2f}]")
    print(f"z range  : [{df['z'].min():.2f}, {df['z'].max():.2f}]")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="WISDM v1.1 auto-downloader.")
    ap.add_argument("--dest_dir", type=str, default=str(Path(__file__).parent),
                    help="Directory to download / extract into "
                         "(default: this file's directory).")
    ap.add_argument("--force", action="store_true",
                    help="Force re-download even if files exist.")
    ap.add_argument("--smoke_test", action="store_true",
                    help="After download, parse the raw file and print a "
                         "small summary.")
    args = ap.parse_args()

    dest = Path(args.dest_dir).resolve()
    download_wisdm(dest, force=args.force)
    if args.smoke_test:
        return _smoke_test(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
