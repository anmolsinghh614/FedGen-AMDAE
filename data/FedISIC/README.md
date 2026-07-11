# FedISIC (ISIC 2019) -- Run 3 dataset slot

Dermoscopy skin-lesion classification. This is the publicly-available
training pool used by the [Flamby](https://github.com/owkin/FLamby)
Fed-ISIC-2019 benchmark; we use Dirichlet-partitioned clients here for
grid consistency with the rest of the Run 3 sweep.

**Source**: <https://challenge.isic-archive.com/data/#2019>
**License**: CC BY-NC 4.0 (attribution + non-commercial)
**Task**: 8-way classification (multi-class, not multi-label)
**Original resolution**: variable, up to 6000x4000; downsampled here to 32x32 RGB

## Classes (8)

| Idx | Code | Full name |
|-----|------|-----------|
| 0   | MEL  | Melanoma |
| 1   | NV   | Melanocytic nevus |
| 2   | BCC  | Basal cell carcinoma |
| 3   | AK   | Actinic keratosis |
| 4   | BKL  | Benign keratosis |
| 5   | DF   | Dermatofibroma |
| 6   | VASC | Vascular lesion |
| 7   | SCC  | Squamous cell carcinoma |

Note that the class distribution is highly skewed (NV ~50%, MEL ~18%,
BCC ~13%, remaining 5 classes ~19%). Dirichlet allocation on top of this
imbalance produces realistic non-IID clients where rare-class users
train on only a handful of samples per class.

## Auto-download

Everything is bootstrapped from the ISIC S3 bucket. No account or API
token is required:

```
python download_fedisic.py                 # fetch all three files (~9 GB)
python download_fedisic.py --metadata_only # tiny CSVs only, skip images
```

The downloader is idempotent and skips files that are already present.

## Preprocessing

Every JPG is loaded, converted to RGB, resized to 32x32 with bilinear
interpolation, and cached to `raw/fedisic_cache_32x32.pt` (~78 MB) so
subsequent alpha values do not re-process 25 331 images. The final
tensors are per-channel z-scored and stored as
`FloatTensor[N, 3, 32, 32]` for downstream code parity with CIFAR-10.

The 32x32 downsample matches the CIFAR-10 convention used in the
original FedGen paper (NeurIPS 2021), so the same small CNN backbone
can be used across datasets without a per-dataset architecture change.

## Non-IID split

```
python generate_niid_dirichlet.py --n_user 20 --alpha 1.0 --sampling_ratio 0.5
```

Output layout (mirrors UCI HAR / WISDM):

```
data/FedISIC/u{n_user}-alpha{a}-ratio{r}/
    train/train.pt
    test/test.pt
```

Each `.pt` file is a dict with `users`, `user_data`, `num_samples` keys,
where `user_data[uid]['x']` is `FloatTensor[N, 3, 32, 32]` and
`user_data[uid]['y']` is `LongTensor[N]` with values in `{0..7}`.
