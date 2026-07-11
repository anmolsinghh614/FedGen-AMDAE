# HAM10000 -- Run 3 dataset slot

Dermatoscopic skin-lesion classification. HAM10000 ("Human Against
Machine with 10 000 training images", Tschandl et al. 2018) is the
official training pool of the ISIC 2018 Challenge Task 3 (Lesion
Diagnosis) and is one of the standard benchmarks for medical FL.

**Source**: <https://isic-challenge-data.s3.amazonaws.com/2018/> (public,
no auth; identical files that Harvard Dataverse also mirrors under
`doi:10.7910/DVN/DBW86T`)
**License**: CC BY-NC 4.0
**Task**: 7-way classification
**Original resolution**: 600x450; downsampled here to 32x32 RGB

## Classes (7)

| Idx | Code   | Full name |
|-----|--------|-----------|
| 0   | MEL    | Melanoma |
| 1   | NV     | Melanocytic nevus |
| 2   | BCC    | Basal cell carcinoma |
| 3   | AKIEC  | Actinic keratosis / Bowen's disease |
| 4   | BKL    | Benign keratosis |
| 5   | DF     | Dermatofibroma |
| 6   | VASC   | Vascular lesion |

Class balance is heavily skewed (NV ~67 %, MEL ~11 %, BKL ~11 %,
remaining 4 classes ~11 %). Dirichlet partitioning is applied on top
of this natural imbalance.

## Auto-download

```
python download_ham10000.py                 # fetch everything (~2.8 GB)
python download_ham10000.py --metadata_only # tiny CSV only
```

Files land under `data/HAM10000/raw/`. The downloader is idempotent.

## Preprocessing

Each JPG is loaded, converted to RGB, resized to 32x32 with bilinear
interpolation, and cached to `raw/ham10000_cache_32x32.pt` (~30 MB) so
subsequent alpha values reuse the cache. Tensors are per-channel
z-scored and stored as `FloatTensor[N, 3, 32, 32]`, matching FedISIC
and CIFAR-10 conventions for architecture reuse.

## Non-IID split

```
python generate_niid_dirichlet.py --n_user 20 --alpha 1.0 --sampling_ratio 0.5
```

Output layout:

```
data/HAM10000/u{n_user}-alpha{a}-ratio{r}/
    train/train.pt
    test/test.pt
```

Schema (dict keys / tensor shapes) matches FedISIC and every other Run 3
dataset.
