# WISDM v1.1 -- Option A slot

This directory holds the WISDM (Wireless Sensor Data Mining) v1.1 dataset,
which occupies the "3rd real-world dataset" slot in the Option A paper
sweep (replaces PAMAP2 in the June 2026 revision). It is a small,
low-noise, phone-accelerometer HAR dataset that fits the same 1x24x24 CNN
input shape as UCI HAR, so it drops in with zero model changes.

## Source

Fordham University WISDM Lab.

- Landing page: <https://www.cis.fordham.edu/wisdm/dataset.php>
- Direct tarball: <http://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz>
- Canonical raw file inside the tarball: `WISDM_ar_v1.1_raw.txt`

Citation:

> Kwapisz, J. R., Weiss, G. M., & Moore, S. A. (2011). *Activity Recognition
> using Cell Phone Accelerometers.* SIGKDD Explorations, 12(2), 74-82.

## Shape

- Sampling rate: 20 Hz
- Subjects: 36 (natural client IDs 1..36)
- Activity classes: 6 (Walking, Jogging, Upstairs, Downstairs, Sitting, Standing)
- Raw rows: ~1.1M

Windowing (chosen so the flattened window matches UCI HAR's 24x24 grid):

- Window: 192 samples (9.6 s), 50% overlap (stride 96)
- Per window: 192 samples x 3 axes = **576 scalars, reshaped to (1, 24, 24)**
- Global per-axis z-score normalisation fit on train windows, soft-clipped
  via `tanh` to keep the range bounded (matches UCI HAR's `[-1, 1]` layout).
- Windows spanning an activity change (modal class < 80% of window) are
  dropped so labels stay clean.

Chronological 80/20 train/test split *per subject* (test rows come from
the tail of each subject's timeline; no window-level leakage).

## Auto-download

`download_wisdm.py` handles the download + tar extraction end-to-end. It
is invoked automatically by `run_optionA_sweep.py` when the Dirichlet
generator does not find `WISDM_ar_v1.1_raw.txt`.

Manual usage (rarely needed):

```bash
cd data/WISDM
python download_wisdm.py --smoke_test
```

The smoke test parses the raw file and prints a class-balance summary.

## Non-IID split

`generate_niid_dirichlet.py` mirrors `data/UCI HAR/generate_niid_dirichlet.py`
in every respect that matters:

- CLI:  `--n_user`, `--alpha`, `--sampling_ratio`, `--min_sample`, ...
- Output layout: `./u<N>-alpha<A>-ratio<R>/{train,test}/*.pt` with the
  same `{'users': [...], 'user_data': {uname: {'x': tensor, 'y': tensor}}, 'num_samples': [...]}`
  dict `main.py` expects.

Called by `run_optionA_sweep.py` with `--n_user 20 --sampling_ratio 0.5`
for `alpha in {0.1, 1.0, 10.0}`, matching the paper's locked grid.
