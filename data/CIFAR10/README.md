# CIFAR-10

The classic CIFAR-10 image classification benchmark, adapted for Dirichlet
non-IID federated splits.

- **Source**: [Krizhevsky, 2009](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Content**: 60 000 32x32 RGB images across 10 classes
  (`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`,
  `ship`, `truck`).
- **Split**: 50 000 train / 10 000 test (torchvision default).
- **Client heterogeneity**: Dirichlet with parameter `alpha`, sampled per
  class across `--n_user` clients (default 20).

## Auto-download

The download is handled transparently by
`torchvision.datasets.CIFAR10(download=True)` and cached to
`data/CIFAR10/raw/`. No API key or manual step is required. The runner
scripts (`run_cifar10_sweep.py`) invoke this generator automatically.

## Manual usage

```bash
cd data/CIFAR10
python generate_niid_dirichlet.py --n_user 20 --alpha 0.1 --sampling_ratio 0.5
python generate_niid_dirichlet.py --n_user 20 --alpha 1.0 --sampling_ratio 0.5
python generate_niid_dirichlet.py --n_user 20 --alpha 10  --sampling_ratio 0.5
```

Each invocation writes:

```
data/CIFAR10/u20-alpha{a}-ratio0.5/train/train.pt
data/CIFAR10/u20-alpha{a}-ratio0.5/test/test.pt
```

Each `.pt` file follows the shared schema:

```python
{
  "users": ["f_00000", "f_00001", ...],
  "user_data": {
      "f_00000": {
          "x": FloatTensor[N, 3, 32, 32],
          "y": LongTensor[N],
      },
      ...
  },
  "num_samples": [n0, n1, ...],
}
```

## Notes

- Normalisation is per-channel z-score computed on the train pool, then
  applied to both train and test (standard CIFAR practice).
- The 32x32x3 shape matches the CNN backbone used by FedISIC and
  HAM10000, so CIFAR-10 slots directly into the shared model pipeline.
- Class distribution in CIFAR-10 is uniform (6 000 train + 1 000 test
  per class), so any heterogeneity in the resulting splits is purely
  from the Dirichlet sampling.
