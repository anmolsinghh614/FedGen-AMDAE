"""
download_ham10000.py
====================

Auto-downloader for HAM10000, the "Human Against Machine with 10 000
training images" dermatoscopic dataset released by Tschandl et al.
(2018) and used as the training pool of the ISIC 2018 Challenge
(Task 3: Lesion Diagnosis).

Rather than pull from Harvard Dataverse (which requires a client that
handles Dataverse's tokenised API), we fetch the exact same images
from the ISIC 2018 challenge public S3 bucket -- these files are
served without authentication under CC BY-NC 4.0 by the ISIC Archive:

    ISIC2018_Task3_Training_Input.zip       ~2.8 GB, 10 015 JPGs
    ISIC2018_Task3_Training_GroundTruth.zip contains a single CSV with
                                            one-hot class labels

The 7 diagnostic classes are:
    MEL, NV, BCC, AKIEC, BKL, DF, VASC

This helper is idempotent: files that are already on disk with a
non-zero size are not re-downloaded.

Usage:
    python download_ham10000.py                 # download everything
    python download_ham10000.py --metadata_only # skip the image zip
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ISIC2018_ROOT = "https://isic-challenge-data.s3.amazonaws.com/2018"

FILES = {
    "images_zip": ("ISIC2018_Task3_Training_Input.zip",
                   ISIC2018_ROOT + "/ISIC2018_Task3_Training_Input.zip"),
    "labels_zip": ("ISIC2018_Task3_Training_GroundTruth.zip",
                   ISIC2018_ROOT + "/ISIC2018_Task3_Training_GroundTruth.zip"),
}


def _stream_download(url: str, dst: Path, chunk_mb: int = 4) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}\n    -> {dst}", flush=True)
    chunk = chunk_mb * 1024 * 1024
    written = 0
    next_ping = 100 * 1024 * 1024
    with urllib.request.urlopen(url, timeout=120) as resp, open(dst, "wb") as out:
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            out.write(buf)
            written += len(buf)
            if written >= next_ping:
                print(f"    ... {written / 1e6:.0f} MB", flush=True)
                next_ping += 100 * 1024 * 1024
    print(f"  done ({written / 1e6:.1f} MB)", flush=True)


def _need_download(dst: Path) -> bool:
    return not (dst.exists() and dst.stat().st_size > 0)


def download_ham10000(dst_root: Path, metadata_only: bool = False) -> Path:
    """Ensure HAM10000 raw pool exists under dst_root/raw/. Returns
    the path to the raw pool directory."""
    raw = dst_root / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    order = ["labels_zip"]
    if not metadata_only:
        order.append("images_zip")

    for key in order:
        name, url = FILES[key]
        dst = raw / name
        if _need_download(dst):
            try:
                _stream_download(url, dst)
            except Exception as exc:
                print(
                    f"[ERROR] Could not download {name} from ISIC S3.\n"
                    f"        Reason: {exc}\n"
                    f"        Manually fetch {url}\n"
                    f"        and place it at {dst}.",
                    file=sys.stderr,
                )
                raise
        else:
            print(f"  already present: {dst.name} "
                  f"({dst.stat().st_size / 1e6:.1f} MB)")

    # Extract labels zip -> the single ground-truth CSV
    lbl_zip = raw / FILES["labels_zip"][0]
    lbl_csv = raw / "ISIC2018_Task3_Training_GroundTruth.csv"
    if not lbl_csv.is_file():
        print(f"  extracting {lbl_zip.name} ...", flush=True)
        with zipfile.ZipFile(lbl_zip) as zf:
            for member in zf.namelist():
                if member.endswith(".csv"):
                    # Flatten in case the zip nests inside a folder.
                    src = zf.open(member)
                    with open(lbl_csv, "wb") as out:
                        shutil.copyfileobj(src, out)
                    print(f"    -> {lbl_csv}")
                    break
        if not lbl_csv.is_file():
            print(f"[WARN] no CSV found inside {lbl_zip.name}",
                  file=sys.stderr)

    # Extract images zip
    if not metadata_only:
        img_zip = raw / FILES["images_zip"][0]
        img_dir = raw / "ISIC2018_Task3_Training_Input"
        if not img_dir.is_dir() or not any(img_dir.glob("*.jpg")):
            print(f"  extracting {img_zip.name} -> {img_dir}/ ...", flush=True)
            with zipfile.ZipFile(img_zip) as zf:
                zf.extractall(raw)
            print(f"  extracted "
                  f"({sum(1 for _ in img_dir.glob('*.jpg'))} images)",
                  flush=True)
        else:
            n = sum(1 for _ in img_dir.glob('*.jpg'))
            print(f"  already extracted: {img_dir.name} ({n} images)")

    return raw


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dst_root", default=str(Path(__file__).resolve().parent))
    p.add_argument("--metadata_only", action="store_true")
    args = p.parse_args()

    dst = Path(args.dst_root).resolve()
    print(f"HAM10000 download target: {dst}")
    raw = download_ham10000(dst, metadata_only=args.metadata_only)
    print(f"OK. Raw pool ready at: {raw}")


if __name__ == "__main__":
    main()
