"""
download_fedisic.py
===================

Auto-downloader for the ISIC 2019 challenge training set, which is the
publicly-available raw pool underlying the Fed-ISIC-2019 federated
benchmark (Flamby).

The ISIC 2019 challenge released three public files on S3 (no auth,
no registration -- they are hosted by the ISIC Archive under
CC BY-NC 4.0):

    ISIC_2019_Training_Input.zip       ~9 GB  raw JPG images (25 331 total)
    ISIC_2019_Training_GroundTruth.csv       one-hot class labels per image
    ISIC_2019_Training_Metadata.csv          per-image metadata (age, sex,
                                             lesion site, imaging site)

The 8 diagnostic classes are:
    MEL, NV, BCC, AK, BKL, DF, VASC, SCC

The imaging site (`lesion_id`s roughly cluster into 6 collecting centres)
provides a natural federation partition; we default to Dirichlet-based
partitioning here for consistency with the rest of the Run 3 sweep, but
`generate_niid_dirichlet.py` also emits a --natural mode that groups by
imaging site.

This helper is idempotent: it does not re-download files that are
already on disk with the expected size (>0 bytes; we do not verify SHA
because ISIC does not publish per-file hashes).

Usage:
    python download_fedisic.py                 # download all three files
    python download_fedisic.py --metadata_only # tiny CSVs only, skip images

The downloaded raw pool lives at::

    data/FedISIC/raw/
        ISIC_2019_Training_Input/*.jpg
        ISIC_2019_Training_GroundTruth.csv
        ISIC_2019_Training_Metadata.csv
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ISIC_ROOT = "https://isic-challenge-data.s3.amazonaws.com/2019"

FILES = {
    "images_zip":  ("ISIC_2019_Training_Input.zip",       ISIC_ROOT + "/ISIC_2019_Training_Input.zip"),
    "labels_csv":  ("ISIC_2019_Training_GroundTruth.csv", ISIC_ROOT + "/ISIC_2019_Training_GroundTruth.csv"),
    "metadata_csv":("ISIC_2019_Training_Metadata.csv",    ISIC_ROOT + "/ISIC_2019_Training_Metadata.csv"),
}


def _stream_download(url: str, dst: Path, chunk_mb: int = 4) -> None:
    """Stream `url` to `dst` using stdlib only. Prints size every 100 MB."""
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


def download_fedisic(dst_root: Path, metadata_only: bool = False) -> Path:
    """Ensure the raw ISIC 2019 pool exists under dst_root/raw/. Returns
    the path to the raw pool directory."""
    raw = dst_root / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    order = ["labels_csv", "metadata_csv"]
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
            print(f"  already present: {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")

    # Extract image zip once (~9 GB expands to ~9 GB; JPEGs are already
    # compressed so no growth). Skip if already extracted.
    if not metadata_only:
        img_zip = raw / FILES["images_zip"][0]
        img_dir = raw / "ISIC_2019_Training_Input"
        if not img_dir.is_dir() or not any(img_dir.glob("*.jpg")):
            print(f"  extracting {img_zip.name} -> {img_dir}/ ...", flush=True)
            with zipfile.ZipFile(img_zip) as zf:
                zf.extractall(raw)
            print(f"  extracted ({sum(1 for _ in img_dir.glob('*.jpg'))} images)",
                  flush=True)
        else:
            n = sum(1 for _ in img_dir.glob('*.jpg'))
            print(f"  already extracted: {img_dir.name} ({n} images)")

    return raw


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dst_root", default=str(Path(__file__).resolve().parent),
                   help="Root of the FedISIC data folder (default: this dir)")
    p.add_argument("--metadata_only", action="store_true",
                   help="Fetch only the two CSVs (tiny), skip the 9 GB image zip")
    args = p.parse_args()

    dst = Path(args.dst_root).resolve()
    print(f"FedISIC download target: {dst}")
    raw = download_fedisic(dst, metadata_only=args.metadata_only)
    print(f"OK. Raw pool ready at: {raw}")


if __name__ == "__main__":
    main()
