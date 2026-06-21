#!/usr/bin/env python
"""
run_optionA_sweep.py
====================

Master orchestrator for the Option A paper sweep that produces every cell
the FedGen-AMDAE paper needs:

    Datasets        : EMnist-letters, UCI HAR, PAMAP2
    Heterogeneity a : {0.1, 1, 10}
    Missing rate    : {0.0, 0.10, 0.20}
    Mechanism       : MCAR (random) only -- MAR / MNAR are not in this sweep
    Algorithms      : FedAvg, FedProx, FedDistill, FedEnsemble, FedGen
    Imputer         : AM-DAE (forced via --force_imputer amdae) for the main
                      sweep so every "FedGen-AMDAE" row is unambiguously
                      AM-DAE-imputed; the imputer-ablation phase additionally
                      runs FedGen with {mean, median, zero, none}.

Per-cell namespacing:

    results/optionA/<dataset_short>/alpha<a>_miss<m>/<algo>/
        models/<TOKEN>_<algo>_<lr>_<num_users>u_<bs>b_<le>_<seed>.h5
        metrics/<TOKEN>/<algo>_<TOKEN>_round_<R>.h5

Resume-skip is per (cell, seed): the driver checks expected HDF5 paths
before invoking main.py and skips seeds whose summary file already exists.

Usage:

    # Stage 1 -- single seed across the full grid (~22-30 GPU-h)
    python run_optionA_sweep.py --stage 1 --device cuda

    # Stage 2 -- multi-seed (re-runs only the missing seeds; cheap if
    # Stage 1 already wrote seed 0)
    python run_optionA_sweep.py --stage 2 --device cuda --times 3

    # Imputer ablation only -- FedGen x {amdae, mean, median, zero, none}
    # at the headline cell (alpha=1, missing=10%) per dataset, --times 3
    python run_optionA_sweep.py --ablation --device cuda

    # Smaller subset, e.g. only EMNIST or only one alpha
    python run_optionA_sweep.py --stage 1 --datasets EMnist-letters --alphas 0.1
    python run_optionA_sweep.py --stage 1 --algorithms FedGen FedAvg

    # Print the plan, run nothing
    python run_optionA_sweep.py --dry_run --stage 1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent
PY = sys.executable
OUT_ROOT = ROOT / "results" / "optionA"

# ---------------------------------------------------------------- locked scope
ALPHAS_DEFAULT = [0.1, 1.0, 10.0]
MISSING_RATES_DEFAULT = [0.0, 0.10, 0.20]
ALGOS_DEFAULT = ["FedAvg", "FedProx", "FedDistill", "FedEnsemble", "FedGen"]
# Run order: UCI HAR (small, fastest sanity signal) -> EMNIST -> PAMAP2.
# This is the order produced cells are created in. Stage-1 surfaces UCI
# HAR first so any wiring break shows up within the first ~hour.
DATASETS_DEFAULT = ["UCI HAR", "EMnist-letters", "PAMAP2"]
SAMPLING_RATIO = 0.5
N_USERS_TOTAL = 20

# Per-dataset communication-round budget (matches paper conventions;
# EMNIST gets the FedGen paper's 200, real-data datasets get 100).
ROUNDS = {"EMnist-letters": 200, "UCI HAR": 100, "PAMAP2": 100}

# UCI HAR raw archive (the only one of the three that is not bundled with
# torchvision and does not have a download helper inside the project).
UCIHAR_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
              "00240/UCI%20HAR%20Dataset.zip")

# Headline cell for imputer ablation + headline figures.
HEADLINE_ALPHA = 1.0
HEADLINE_MISS = 0.10
ABLATION_IMPUTERS = ["amdae", "mean", "median", "zero", "none"]


# ---------------------------------------------------------------- registry
class MilestoneRegistry:
    """Append-only success/failure log of every pipeline milestone.

    Writes two files alongside the sweep results:

      results/optionA/_status.md     human-readable, grouped by phase
      results/optionA/_status.json   machine-readable, full record list

    The Markdown file is rebuilt from the in-memory record list after
    every log entry, so it is always up-to-date even mid-sweep -- you
    can `tail` it from another shell while the sweep runs.

    At the end of the run, call `summarise()` to print a final banner
    saying "ALL <N> MILESTONES OK" or listing the failures.
    """

    PHASES_ORDER = [
        "bootstrap",
        "split_prep",
        "main_sweep",
        "imputer_ablation",
        "paper_outputs",
    ]
    PHASES_LABEL = {
        "bootstrap":        "Dataset bootstrap (download + extract raw data)",
        "split_prep":       "Dirichlet split generation",
        "main_sweep":       "Main sweep (per-cell, per-seed training)",
        "imputer_ablation": "Imputer ablation (FedGen x 5 imputers)",
        "paper_outputs":    "Paper outputs (tables + dashboards)",
    }

    def __init__(self, out_dir: Path, dry: bool = False) -> None:
        self.dry = dry
        self.records: list[dict] = []
        self.out_dir = out_dir
        self.md_path = out_dir / "_status.md"
        self.json_path = out_dir / "_status.json"
        self.summary_json_path = out_dir / "_status_summary.json"
        self._t_start = time.time()
        self._run_id = datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H-%M-%SZ")
        self._finished = False

    def log(self, phase: str, item: str, status: str,
            message: str = "", duration_s: Optional[float] = None) -> None:
        """Record one milestone. status in {'PASS', 'FAIL', 'SKIP', 'INFO'}."""
        rec = {
            "phase": phase,
            "item": item,
            "status": status,
            "message": message or "",
            "duration_s": (round(float(duration_s), 1)
                           if duration_s is not None else None),
            "ts_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"),
        }
        self.records.append(rec)
        # Always echo to stdout so users see milestones in the live log.
        dur = (f" ({rec['duration_s']:.1f}s)"
               if rec['duration_s'] is not None else "")
        line = f"[{status}] {phase}/{item}{dur}"
        if message:
            line += f"  -- {message}"
        print(line, flush=True)
        # Always flush so users can `tail _status.md` while the sweep
        # is running, even on a dry run (which is useful for previewing
        # what milestones the registry will record).
        self._flush()

    def info(self, phase: str, item: str, message: str = "") -> None:
        self.log(phase, item, "INFO", message)

    def passed(self, phase: str, item: str, message: str = "",
               duration_s: Optional[float] = None) -> None:
        self.log(phase, item, "PASS", message, duration_s)

    def failed(self, phase: str, item: str, message: str = "",
               duration_s: Optional[float] = None) -> None:
        self.log(phase, item, "FAIL", message, duration_s)

    def skipped(self, phase: str, item: str, message: str = "") -> None:
        self.log(phase, item, "SKIP", message)

    def _build_summary(self) -> dict:
        """Build the flat snapshot dict written to _status_summary.json.
        Always reflects the current state (in-progress or finished)."""
        # Per-phase rollup
        phases: dict[str, dict] = {}
        failures: list[dict] = []
        for rec in self.records:
            ph = rec["phase"]
            slot = phases.setdefault(ph, {
                "PASS": 0, "FAIL": 0, "SKIP": 0, "INFO": 0, "items": []
            })
            slot[rec["status"]] = slot.get(rec["status"], 0) + 1
            slot["items"].append({
                "item": rec["item"],
                "status": rec["status"],
                "message": rec["message"],
                "duration_s": rec["duration_s"],
                "ts_utc": rec["ts_utc"],
            })
            if rec["status"] == "FAIL":
                failures.append({
                    "phase": ph,
                    "item": rec["item"],
                    "message": rec["message"],
                    "ts_utc": rec["ts_utc"],
                })

        n_pass = sum(1 for r in self.records if r["status"] == "PASS")
        n_fail = sum(1 for r in self.records if r["status"] == "FAIL")
        n_skip = sum(1 for r in self.records if r["status"] == "SKIP")
        n_info = sum(1 for r in self.records if r["status"] == "INFO")

        if not self.records:
            verdict = "NOT_STARTED"
        elif self._finished and n_fail == 0:
            verdict = "ALL_OK"
        elif self._finished and n_fail > 0:
            verdict = "FAILURES_PRESENT"
        elif n_fail == 0:
            verdict = "IN_PROGRESS_HEALTHY"
        else:
            verdict = "IN_PROGRESS_WITH_FAILURES"

        return {
            "run_id": self._run_id,
            "started_utc": self._run_id,
            "last_update_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"),
            "elapsed_s": round(time.time() - self._t_start, 1),
            "finished": self._finished,
            "verdict": verdict,
            "totals": {
                "TOTAL": len(self.records),
                "PASS": n_pass,
                "FAIL": n_fail,
                "SKIP": n_skip,
                "INFO": n_info,
            },
            "per_phase": {
                ph: {k: phases[ph].get(k, 0)
                     for k in ("PASS", "FAIL", "SKIP", "INFO")}
                for ph in self.PHASES_ORDER if ph in phases
            },
            "failures": failures,
            "phase_details": phases,
        }

    def _flush(self) -> None:
        """Rewrite _status.md, _status.json (full record list), and
        _status_summary.json (flat snapshot) from `self.records`."""
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "run_id": self._run_id,
                    "started": self._run_id,
                    "elapsed_s": round(time.time() - self._t_start, 1),
                    "finished": self._finished,
                    "records": self.records,
                }, f, indent=2)
            with open(self.summary_json_path, "w", encoding="utf-8") as f:
                json.dump(self._build_summary(), f, indent=2)
            self._write_md()
        except OSError as exc:
            print(f"[WARN] could not flush status registry: {exc}",
                  file=sys.stderr)

    def _write_md(self) -> None:
        lines = [
            f"# Option A sweep -- run status",
            f"",
            f"_Run ID: `{self._run_id}`_  ",
            f"_Last update: "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_  ",
            f"_Elapsed: {time.time() - self._t_start:.0f}s_",
            f"",
        ]
        # Per-phase grouping
        by_phase: dict[str, list[dict]] = {p: [] for p in self.PHASES_ORDER}
        for rec in self.records:
            by_phase.setdefault(rec["phase"], []).append(rec)

        for phase in self.PHASES_ORDER:
            recs = by_phase.get(phase, [])
            if not recs:
                continue
            n_pass = sum(1 for r in recs if r["status"] == "PASS")
            n_fail = sum(1 for r in recs if r["status"] == "FAIL")
            n_skip = sum(1 for r in recs if r["status"] == "SKIP")
            n_info = sum(1 for r in recs if r["status"] == "INFO")
            lines.append(f"## {self.PHASES_LABEL[phase]}")
            lines.append(f"")
            lines.append(f"PASS={n_pass}  FAIL={n_fail}  "
                         f"SKIP={n_skip}  INFO={n_info}")
            lines.append(f"")
            for r in recs:
                tick = {"PASS": "[PASS]", "FAIL": "[FAIL]",
                        "SKIP": "[SKIP]", "INFO": "[INFO]"}.get(
                            r["status"], "[----]")
                dur = (f"  ({r['duration_s']:.1f}s)"
                       if r["duration_s"] is not None else "")
                msg = f"  -- {r['message']}" if r["message"] else ""
                lines.append(f"- {tick} `{r['item']}`{dur}{msg}")
            lines.append("")

        # Summary footer
        n_pass = sum(1 for r in self.records if r["status"] == "PASS")
        n_fail = sum(1 for r in self.records if r["status"] == "FAIL")
        n_skip = sum(1 for r in self.records if r["status"] == "SKIP")
        verdict = ("ALL MILESTONES OK" if n_fail == 0
                   else f"{n_fail} FAILURES")
        lines += [
            "## Summary",
            "",
            f"- Total milestones : {len(self.records)}",
            f"- PASS             : {n_pass}",
            f"- FAIL             : {n_fail}",
            f"- SKIP             : {n_skip}",
            f"",
            f"### Verdict: **{verdict}**",
            "",
        ]
        self.md_path.write_text("\n".join(lines), encoding="utf-8")

    def summarise(self) -> bool:
        """Print the end-of-run banner. Returns True if every milestone
        passed (or was deliberately skipped), False if any FAILED."""
        # Mark the registry finished so the JSON verdict flips from
        # IN_PROGRESS_* to ALL_OK / FAILURES_PRESENT, then re-flush so
        # the on-disk files reflect the final state.
        self._finished = True
        self._flush()

        n_pass = sum(1 for r in self.records if r["status"] == "PASS")
        n_fail = sum(1 for r in self.records if r["status"] == "FAIL")
        n_skip = sum(1 for r in self.records if r["status"] == "SKIP")
        bar = "=" * 78
        print(f"\n{bar}")
        print("RUN STATUS SUMMARY")
        print(bar)
        # Per-phase rollup
        by_phase: dict[str, list[dict]] = {p: [] for p in self.PHASES_ORDER}
        for rec in self.records:
            by_phase.setdefault(rec["phase"], []).append(rec)
        for phase in self.PHASES_ORDER:
            recs = by_phase.get(phase, [])
            if not recs:
                continue
            p_pass = sum(1 for r in recs if r["status"] == "PASS")
            p_fail = sum(1 for r in recs if r["status"] == "FAIL")
            p_skip = sum(1 for r in recs if r["status"] == "SKIP")
            label = self.PHASES_LABEL[phase][:48]
            line = f"  {label:<48s}  PASS={p_pass:>3}  FAIL={p_fail:>3}"
            if p_skip:
                line += f"  SKIP={p_skip:>3}"
            print(line)
        print(bar)
        if n_fail == 0:
            print(f"  ALL MILESTONES OK  ({n_pass} PASS, "
                  f"{n_skip} SKIP)")
        else:
            print(f"  {n_fail} FAILURE(S) -- see {self.md_path} for details")
            print()
            print("  Failures:")
            for r in self.records:
                if r["status"] == "FAIL":
                    print(f"    - {r['phase']}/{r['item']}: "
                          f"{r['message'] or '(no message)'}")
        print(bar)
        print(f"  Status report  : {self.md_path}")
        print(f"  JSON record    : {self.json_path}")
        print(f"  JSON summary   : {self.summary_json_path}")
        print(bar)
        return n_fail == 0


# Module-level registry; populated in main() and consumed by every phase.
REGISTRY: Optional[MilestoneRegistry] = None


# ---------------------------------------------------------------- helpers
def banner(msg: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{msg}\n{bar}", flush=True)


def run(cmd: list, dry: bool = False, cwd: Optional[Path] = None,
        allow_fail: bool = False) -> int:
    print(">> " + " ".join(str(c) for c in cmd),
          f"(cwd={cwd or ROOT})", flush=True)
    if dry:
        return 0
    rc = subprocess.call([str(c) for c in cmd],
                         cwd=str(cwd) if cwd else str(ROOT))
    if rc != 0 and not allow_fail:
        print(f"[WARN] command exited rc={rc}", file=sys.stderr)
    return rc


# ---------------------------------------------------------------- adapters
def dataset_short(dataset: str) -> str:
    return {
        "EMnist-letters": "emnist",
        "UCI HAR": "ucihar",
        "PAMAP2": "pamap2",
    }[dataset]


def dataset_token(dataset: str, alpha: float) -> str:
    """Token passed to main.py --dataset (matches utils/model_utils.py)."""
    a = _fmt_alpha(alpha)
    return f"{dataset}-alpha{a}-ratio{SAMPLING_RATIO}"


def _fmt_alpha(alpha: float) -> str:
    """Render alpha consistently with the path conventions used by the
    Dirichlet generators (e.g. 1.0 -> '1.0', 0.1 -> '0.1', 10.0 -> '10.0').
    The generators all store the alpha in the dirname using `str(alpha)`,
    so we mirror that exactly to keep paths consistent."""
    return str(alpha)


def dataset_split_dir(dataset: str, alpha: float) -> Path:
    """Filesystem path where the Dirichlet split lives."""
    a = _fmt_alpha(alpha)
    if dataset == "EMnist-letters":
        return ROOT / "data" / "EMnist" / \
            f"u{N_USERS_TOTAL}-letters-alpha{a}-ratio{SAMPLING_RATIO}"
    if dataset == "UCI HAR":
        return ROOT / "data" / "UCI HAR" / \
            f"u{N_USERS_TOTAL}-alpha{a}-ratio{SAMPLING_RATIO}"
    if dataset == "PAMAP2":
        return ROOT / "data" / "PAMAP2" / \
            f"u{N_USERS_TOTAL}-alpha{a}-ratio{SAMPLING_RATIO}"
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------- pre-flight
def _ucihar_natural_path() -> Path:
    return ROOT / "data" / "UCI HAR" / "UCI HAR Dataset"


def _ucihar_generator_path() -> Path:
    return ROOT / "data" / "UCI HAR" / "data" / "UCI HAR Dataset"


def stage_ucihar_for_generator(dry: bool = False) -> None:
    """The UCI HAR Dirichlet generator hard-codes data_dir='./data'; bridge
    the canonical extract path to the generator's expected path."""
    src = _ucihar_natural_path()
    dst = _ucihar_generator_path()
    if (dst / "train").is_dir() and (dst / "test").is_dir():
        return
    if not (src / "train").is_dir():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry:
        print(f"[dry] would symlink {dst} -> {src}")
        return
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst, target_is_directory=True)
    except (OSError, NotImplementedError):
        shutil.copytree(src, dst)


def _download_ucihar_zip(zip_path: Path) -> None:
    """Stream the official UCI HAR zip into `zip_path` using stdlib only
    (no wget / unzip dependency). Cross-platform; works on Windows GPU
    boxes too."""
    import urllib.request
    print(f"  downloading UCI HAR archive from\n    {UCIHAR_URL}\n  -> {zip_path}",
          flush=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(UCIHAR_URL, timeout=120) as resp, \
         open(zip_path, "wb") as f:
        shutil.copyfileobj(resp, f, length=1024 * 1024)
    print(f"  download complete ({zip_path.stat().st_size / 1e6:.1f} MB)",
          flush=True)


def _ensure_ucihar_raw(args: argparse.Namespace) -> bool:
    """If the UCI HAR raw extract is missing, download + unzip it via
    stdlib (no wget/unzip required). Returns True on success or
    already-present, False when auto-download is disabled and the data
    is missing."""
    import zipfile
    t0 = time.time()
    if (_ucihar_natural_path() / "train").is_dir():
        if REGISTRY is not None:
            REGISTRY.passed("bootstrap", "ucihar",
                            "raw data already present (no download needed)")
        return True

    if not getattr(args, "auto_download", True):
        msg = (f"raw data missing at {_ucihar_natural_path()} and "
               f"--no_auto_download is set")
        print(f"[ERROR] UCI HAR {msg}.\n"
              f"        Manually download {UCIHAR_URL}\n"
              f"        and unzip it into {_ucihar_natural_path().parent}/.",
              file=sys.stderr)
        if REGISTRY is not None:
            REGISTRY.failed("bootstrap", "ucihar", msg)
        return False
    if args.dry_run:
        print(f"[dry] would auto-download + unzip UCI HAR archive into "
              f"{_ucihar_natural_path().parent}/")
        if REGISTRY is not None:
            REGISTRY.info("bootstrap", "ucihar", "DRY RUN")
        return True

    target_dir = _ucihar_natural_path().parent  # data/UCI HAR/
    zip_path = target_dir / "UCI HAR Dataset.zip"

    try:
        if not zip_path.exists():
            _download_ucihar_zip(zip_path)
        else:
            print(f"  UCI HAR zip already present: {zip_path}")
        print(f"  unzipping into {target_dir} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target_dir)
    except Exception as exc:
        msg = f"download/unzip failed: {exc}"
        print(f"[ERROR] UCI HAR auto-download / unzip failed: {exc}\n"
              f"        Manually fetch {UCIHAR_URL}\n"
              f"        and unzip into {target_dir}/.",
              file=sys.stderr)
        if REGISTRY is not None:
            REGISTRY.failed("bootstrap", "ucihar", msg, time.time() - t0)
        return False

    if not (_ucihar_natural_path() / "train").is_dir():
        msg = f"expected dir {_ucihar_natural_path()}/train missing after unzip"
        print(f"[ERROR] After unzip, expected layout is missing:\n"
              f"        {_ucihar_natural_path()}/train was not created.\n"
              f"        Inspect {target_dir} contents.",
              file=sys.stderr)
        if REGISTRY is not None:
            REGISTRY.failed("bootstrap", "ucihar", msg, time.time() - t0)
        return False
    print(f"  UCI HAR raw ready at {_ucihar_natural_path()}")
    if REGISTRY is not None:
        size_mb = (zip_path.stat().st_size / 1e6) if zip_path.exists() else 0
        REGISTRY.passed("bootstrap", "ucihar",
                        f"download + unzip complete ({size_mb:.1f} MB zip)",
                        time.time() - t0)
    return True


def ensure_split(args: argparse.Namespace, dataset: str, alpha: float) -> bool:
    """Generate the per-(dataset, alpha) Dirichlet split if missing.
    Returns True on success (or already-present), False on generator failure."""
    split_dir = dataset_split_dir(dataset, alpha)
    item = f"{dataset_short(dataset)}_alpha{alpha}"
    if (split_dir / "train").is_dir() and (split_dir / "test").is_dir():
        print(f"[ok] split present: {split_dir}")
        if REGISTRY is not None:
            REGISTRY.passed("split_prep", item, "already present")
        return True

    t0 = time.time()

    if dataset == "EMnist-letters":
        gen = ROOT / "data" / "EMnist" / "generate_niid_dirichlet.py"
        cwd = ROOT / "data" / "EMnist"
        cmd = [PY, str(gen),
               "--n_user", str(N_USERS_TOTAL),
               "--alpha", str(alpha),
               "--sampling_ratio", str(SAMPLING_RATIO),
               "--split", "letters"]
    elif dataset == "UCI HAR":
        if not _ensure_ucihar_raw(args):
            return False
        stage_ucihar_for_generator(dry=args.dry_run)
        gen = ROOT / "data" / "UCI HAR" / "generate_niid_dirichlet.py"
        cwd = ROOT / "data" / "UCI HAR"
        cmd = [PY, str(gen),
               "--n_user", str(N_USERS_TOTAL),
               "--alpha", str(alpha),
               "--sampling_ratio", str(SAMPLING_RATIO)]
    elif dataset == "PAMAP2":
        gen = ROOT / "goal2_real_dataset_experiment.py"
        cwd = ROOT
        cmd = [PY, str(gen),
               "--dataset_kind", "pamap2",
               "--alpha", str(alpha),
               "--sampling_ratio", str(SAMPLING_RATIO),
               "--n_user", str(N_USERS_TOTAL),
               "--prepare_only"]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    rc = run(cmd, dry=args.dry_run, cwd=cwd, allow_fail=True)
    dt = time.time() - t0
    if rc != 0 and not args.dry_run:
        print(f"[ERROR] data-prep failed for {dataset} alpha={alpha} (rc={rc})")
        if REGISTRY is not None:
            REGISTRY.failed("split_prep", item,
                            f"generator exited rc={rc}", dt)
        return False
    if REGISTRY is not None and not args.dry_run:
        REGISTRY.passed("split_prep", item,
                        f"generated at {split_dir.name}", dt)
    elif REGISTRY is not None and args.dry_run:
        REGISTRY.info("split_prep", item, "DRY RUN")
    return True


# ---------------------------------------------------------------- per-cell I/O
def cell_dir(dataset: str, alpha: float, miss: float,
             algo: str, suffix: str = "") -> Path:
    """Per-cell namespace root. `suffix` allows optional sub-bucketing
    (e.g. 'imputer_ablation/<imputer>') without changing the public layout."""
    base = OUT_ROOT / dataset_short(dataset) / \
        f"alpha{alpha}_miss{miss}" / algo
    if suffix:
        base = base / suffix
    return base


def expected_h5(args: argparse.Namespace, dataset: str, alpha: float,
                algo: str, seed: int, models_dir: Path) -> Path:
    """Mirror utils.model_utils.get_log_path so we know exactly which file
    the server will write."""
    token = dataset_token(dataset, alpha)
    name = (f"{token}_{algo}_{args.learning_rate}_"
            f"{args.num_users}u_{args.batch_size}b_"
            f"{args.local_epochs}_{seed}")
    if "FedGen" in algo:
        name += "_embed0"
        if int(args.gen_batch_size) != int(args.batch_size):
            name += f"_gb{args.gen_batch_size}"
    return models_dir / f"{name}.h5"


def _relocate_per_round_metrics(args: argparse.Namespace,
                                metrics_dir: Path, seed: int) -> None:
    """main.py / utils.metrics_utils write per-round dumps to the static path
    `results/metrics/<TOKEN>/...`. The per-round HDF5 filenames do NOT
    include the seed, so back-to-back seeds for the same cell would
    silently overwrite each other.

    To preserve per-seed F1 (needed for mean +/- std in the paper tables),
    we move each seed's dumps into a seed-specific sub-folder:

        <cell>/metrics/seed_<s>/<TOKEN>/<algo>_<TOKEN>_round_<R>.h5

    paper_table_optionA.py reads these per-seed dumps to compute
    mean +/- std Macro-F1 across seeds.
    """
    live = ROOT / "results" / "metrics"
    if not live.is_dir() or args.dry_run:
        return
    seed_dir = metrics_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    for sub in live.iterdir():
        target = seed_dir / sub.name
        target.mkdir(parents=True, exist_ok=True)
        if sub.is_dir():
            for h5 in sub.iterdir():
                shutil.move(str(h5), str(target / h5.name))
        else:
            shutil.move(str(sub), str(target / sub.name))
    shutil.rmtree(live, ignore_errors=True)


def _check_no_stale_metrics() -> bool:
    """Refuse to start a fresh cell if a previous crash left
    `results/metrics/` lying around -- mixing it into our cell would
    misattribute someone else's per-round dumps."""
    live = ROOT / "results" / "metrics"
    if live.is_dir() and any(live.iterdir()):
        print(f"[ERROR] leftover {live} found; refusing to start a new cell.")
        print("        Move it out of the way (e.g. `mv results/metrics _stale`) "
              "and re-run.")
        return False
    return True


# ---------------------------------------------------------------- training
def train_cell(args: argparse.Namespace, dataset: str, alpha: float,
               miss: float, algo: str, seeds_wanted: int,
               force_imputer: Optional[str] = None,
               cell_suffix: str = "") -> None:
    """Train all `seeds_wanted` seeds for one (dataset, alpha, miss, algo)
    cell. Uses --seed_start so already-completed seeds are not re-run."""
    base = cell_dir(dataset, alpha, miss, algo, suffix=cell_suffix)
    models_dir = base / "models"
    metrics_dir = base / "metrics"
    for d in (models_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Phase tag is "imputer_ablation" for ablation cells, "main_sweep"
    # otherwise (cell_suffix encodes this).
    phase = ("imputer_ablation" if cell_suffix.startswith("imputer_ablation")
             else "main_sweep")

    todo_seeds = [s for s in range(seeds_wanted)
                  if not expected_h5(args, dataset, alpha, algo,
                                     s, models_dir).exists()]
    cell_id_base = (f"{dataset_short(dataset)}_alpha{alpha}_miss{miss}_{algo}"
                    + (f"_{cell_suffix.replace('/', '_')}"
                       if cell_suffix else ""))
    if not todo_seeds:
        print(f"[skip] {dataset_short(dataset):>6} a={alpha} m={miss} "
              f"{algo:>11s}{(' (' + cell_suffix + ')') if cell_suffix else ''} "
              f"-- all {seeds_wanted} seed(s) already trained")
        if REGISTRY is not None:
            for s in range(seeds_wanted):
                REGISTRY.passed(phase, f"{cell_id_base}_seed{s}",
                                "already trained (resume-skip)")
        return

    rounds = ROUNDS[dataset]
    token = dataset_token(dataset, alpha)

    for s in todo_seeds:
        if not args.dry_run and not _check_no_stale_metrics():
            if REGISTRY is not None:
                REGISTRY.failed(phase, f"{cell_id_base}_seed{s}",
                                "stale results/metrics from previous crash")
            return

        cmd = [PY, "main.py",
               "--dataset", token,
               "--algorithm", algo,
               "--missing_rate", str(miss),
               "--missing_pattern", "random",
               "--num_glob_iters", str(rounds),
               "--local_epochs", str(args.local_epochs),
               "--num_users", str(args.num_users),
               "--batch_size", str(args.batch_size),
               "--gen_batch_size", str(args.gen_batch_size),
               "--learning_rate", str(args.learning_rate),
               "--times", "1",
               "--seed_start", str(s),
               "--device", args.device,
               "--result_path", str(models_dir)]
        if force_imputer:
            cmd += ["--force_imputer", force_imputer]

        t0 = time.time()
        rc = run(cmd, dry=args.dry_run, allow_fail=True)
        dt = time.time() - t0

        # Confirm the expected output file actually appeared. A spurious
        # rc=0 without an HDF5 (e.g. process aborted before save_results)
        # is treated as a failure for paper-status purposes.
        h5 = expected_h5(args, dataset, alpha, algo, s, models_dir)
        produced = (args.dry_run or h5.exists())
        item_id = f"{cell_id_base}_seed{s}"

        if rc != 0:
            print(f"[WARN] training failed: {dataset} a={alpha} m={miss} "
                  f"{algo} seed={s} (rc={rc})", file=sys.stderr)
            if REGISTRY is not None:
                REGISTRY.failed(phase, item_id,
                                f"main.py exited rc={rc}", dt)
        elif not produced:
            if REGISTRY is not None:
                REGISTRY.failed(phase, item_id,
                                f"rc=0 but expected HDF5 missing: {h5.name}",
                                dt)
        else:
            if REGISTRY is not None:
                if args.dry_run:
                    REGISTRY.info(phase, item_id, "DRY RUN")
                else:
                    REGISTRY.passed(phase, item_id,
                                    f"trained, summary={h5.name}", dt)

        # Whether the training failed or not, relocate any per-round dumps
        # so they don't pollute the next cell.
        _relocate_per_round_metrics(args, metrics_dir, s)


# ---------------------------------------------------------------- phases
def phase_main_sweep(args: argparse.Namespace) -> None:
    """The (dataset x alpha x missing x algo x seeds) grid."""
    seeds_wanted = args.times if args.stage == 2 else 1

    banner(f"MAIN SWEEP  stage={args.stage}  seeds={seeds_wanted}  "
           f"force_imputer={args.force_imputer or 'auto-pick (AM-DAE)'}")

    for dataset in args.datasets:
        for alpha in args.alphas:
            ok = ensure_split(args, dataset, alpha)
            if not ok and not args.dry_run:
                print(f"[ERROR] skipping all cells for {dataset} alpha={alpha}")
                continue
            for miss in args.missing_rates:
                for algo in args.algorithms:
                    train_cell(args, dataset, alpha, miss, algo,
                               seeds_wanted=seeds_wanted,
                               force_imputer=args.force_imputer)


def phase_paper_outputs(args: argparse.Namespace) -> None:
    """Run paper_table_optionA.py + paper_dashboard.py to materialise the
    7 tables and 4 dashboard PNGs. Both scripts are smoke-test-safe -- they
    render placeholder cells when seeds are missing."""
    banner("PAPER OUTPUTS  (tables + dashboards)")

    table_cmd = [PY, "paper_table_optionA.py",
                 "--input-root", str(OUT_ROOT),
                 "--output-dir", str(OUT_ROOT / "tables"),
                 "--metric", "all",
                 "--imputer_ablation"]
    t0 = time.time()
    rc = run(table_cmd, dry=args.dry_run, allow_fail=True)
    dt = time.time() - t0
    if args.dry_run:
        if REGISTRY is not None:
            REGISTRY.info("paper_outputs", "paper_table_optionA", "DRY RUN")
    elif rc != 0:
        print("[WARN] paper_table_optionA.py failed (rc={}); see log."
              .format(rc), file=sys.stderr)
        if REGISTRY is not None:
            REGISTRY.failed("paper_outputs", "paper_table_optionA",
                            f"rc={rc}", dt)
    else:
        if REGISTRY is not None:
            REGISTRY.passed("paper_outputs", "paper_table_optionA",
                            f"7 tables -> {OUT_ROOT / 'tables'}", dt)

    dash_cmd = [PY, "paper_dashboard.py",
                "--input-root", str(OUT_ROOT),
                "--output-dir", str(OUT_ROOT / "dashboards"),
                "--headline-alpha", "1.0",
                "--headline-missing", "0.10"]
    t0 = time.time()
    rc = run(dash_cmd, dry=args.dry_run, allow_fail=True)
    dt = time.time() - t0
    if args.dry_run:
        if REGISTRY is not None:
            REGISTRY.info("paper_outputs", "paper_dashboard", "DRY RUN")
    elif rc != 0:
        print("[WARN] paper_dashboard.py failed (rc={}); see log."
              .format(rc), file=sys.stderr)
        if REGISTRY is not None:
            REGISTRY.failed("paper_outputs", "paper_dashboard",
                            f"rc={rc}", dt)
    else:
        if REGISTRY is not None:
            REGISTRY.passed("paper_outputs", "paper_dashboard",
                            f"4 figures -> {OUT_ROOT / 'dashboards'}", dt)


def phase_imputer_ablation(args: argparse.Namespace) -> None:
    """FedGen x {amdae, mean, median, zero, none} at the headline cell of
    each dataset, --times args.times (default 3). Output is namespaced
    under <cell>/imputer_ablation/<imputer>/{models,metrics}/."""
    seeds_wanted = max(1, args.times)
    banner(f"IMPUTER ABLATION  cell=alpha{HEADLINE_ALPHA}/miss{HEADLINE_MISS}  "
           f"imputers={ABLATION_IMPUTERS}  seeds={seeds_wanted}")

    for dataset in args.datasets:
        ok = ensure_split(args, dataset, HEADLINE_ALPHA)
        if not ok and not args.dry_run:
            print(f"[ERROR] skipping ablation for {dataset}")
            continue
        for imputer in ABLATION_IMPUTERS:
            train_cell(args, dataset, HEADLINE_ALPHA, HEADLINE_MISS,
                       "FedGen", seeds_wanted=seeds_wanted,
                       force_imputer=imputer,
                       cell_suffix=f"imputer_ablation/{imputer}")


# ---------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # Scope (defaults match the locked Option A grid).
    p.add_argument("--datasets", nargs="*", default=DATASETS_DEFAULT,
                   choices=DATASETS_DEFAULT,
                   help=f"Subset of datasets to run (default: all 3).")
    p.add_argument("--alphas", nargs="*", type=float, default=ALPHAS_DEFAULT,
                   help=f"alpha values (default: {ALPHAS_DEFAULT}).")
    p.add_argument("--missing_rates", nargs="*", type=float,
                   default=MISSING_RATES_DEFAULT,
                   help=f"Missing rates (default: {MISSING_RATES_DEFAULT}).")
    p.add_argument("--algorithms", nargs="*", default=ALGOS_DEFAULT,
                   choices=ALGOS_DEFAULT,
                   help=f"FL algorithms (default: all 5).")

    # Training knobs (mirror main.py defaults).
    p.add_argument("--local_epochs", type=int, default=20)
    p.add_argument("--num_users", type=int, default=10,
                   help="Sampled users per round (separate from N_USERS_TOTAL=20 "
                        "which controls the size of the on-disk Dirichlet split).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gen_batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--times", type=int, default=3,
                   help="Total seeds wanted in Stage 2 (default 3 for "
                        "journal-grade std-devs). Stage 1 always runs --times 1.")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    # Per-dataset round overrides (defaults are ROUNDS[]).
    p.add_argument("--num_glob_iters_emnist", type=int, default=None,
                   help=f"Override rounds for EMNIST (default {ROUNDS['EMnist-letters']}).")
    p.add_argument("--num_glob_iters_ucihar", type=int, default=None,
                   help=f"Override rounds for UCI HAR (default {ROUNDS['UCI HAR']}).")
    p.add_argument("--num_glob_iters_pamap2", type=int, default=None,
                   help=f"Override rounds for PAMAP2 (default {ROUNDS['PAMAP2']}).")

    # Phase selection.
    p.add_argument("--stage", type=int, default=1, choices=[1, 2],
                   help="1: main sweep with seeds=1 (Stage 1). "
                        "2: main sweep with seeds=times (Stage 2; "
                        "skip-resume already takes care of seed-0).")
    p.add_argument("--ablation", action="store_true",
                   help="Run ONLY the imputer-ablation phase "
                        "(FedGen x 5 imputers at headline cell).")
    p.add_argument("--full_pipeline", action="store_true",
                   help="One-shot: run main sweep (Stage 2 semantics, "
                        "--times seeds per cell), then imputer ablation, "
                        "then build paper tables + dashboards. Equivalent "
                        "to running --stage 2, then --ablation, then "
                        "paper_table_optionA.py, then paper_dashboard.py "
                        "back-to-back. Default --times is 3 (multi-seed); "
                        "pass --times 1 for a single-seed sanity pass.")

    # Override the imputer for the main sweep. By default we hard-force
    # 'amdae' so every "FedGen-AMDAE" row is reviewer-bulletproof.
    p.add_argument("--force_imputer", default="amdae",
                   choices=["amdae", "mean", "median", "zero", "none", "auto"],
                   help="Imputer for the MAIN sweep (default amdae). "
                        "'auto' means: don't pass --force_imputer to "
                        "main.py; let the patched composite pick.")

    p.add_argument("--dry_run", action="store_true",
                   help="Print every command without executing.")
    p.add_argument("--no_auto_download", dest="auto_download",
                   action="store_false",
                   help="Disable automatic download of raw datasets when "
                        "they are missing. EMNIST + PAMAP2 are auto-fetched "
                        "by their data-prep helpers; UCI HAR is auto-fetched "
                        "by this driver. Use this flag to disable UCI HAR "
                        "auto-download (e.g. on a no-internet GPU box).")
    p.set_defaults(auto_download=True)
    return p.parse_args()


def main() -> None:
    global REGISTRY
    args = parse_args()

    # Translate 'auto' to None so main.py's --force_imputer arg is omitted.
    if args.force_imputer == "auto":
        args.force_imputer = None

    # Apply per-dataset round overrides (if any). These mutate the module-
    # level ROUNDS dict, which train_cell() reads.
    if args.num_glob_iters_emnist is not None:
        ROUNDS["EMnist-letters"] = args.num_glob_iters_emnist
    if args.num_glob_iters_ucihar is not None:
        ROUNDS["UCI HAR"] = args.num_glob_iters_ucihar
    if args.num_glob_iters_pamap2 is not None:
        ROUNDS["PAMAP2"] = args.num_glob_iters_pamap2

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Initialise the milestone registry. Every PASS / FAIL through the
    # sweep is appended to results/optionA/_status.{md,json} so the user
    # can confirm at a glance whether everything finished cleanly.
    REGISTRY = MilestoneRegistry(OUT_ROOT, dry=args.dry_run)

    banner(
        f"FedGen-AMDAE  ::  Option A sweep  "
        f"({'DRY RUN' if args.dry_run else 'LIVE'})\n"
        f"  datasets         = {args.datasets}\n"
        f"  alphas           = {args.alphas}\n"
        f"  missing_rates    = {args.missing_rates}\n"
        f"  algorithms       = {args.algorithms}\n"
        f"  rounds           = {ROUNDS}\n"
        f"  local_epochs     = {args.local_epochs}\n"
        f"  num_users (samp) = {args.num_users}\n"
        f"  batch_size       = {args.batch_size}\n"
        f"  times (seeds)    = {args.times}\n"
        f"  stage            = {args.stage}\n"
        f"  ablation only?   = {args.ablation}\n"
        f"  full_pipeline?   = {args.full_pipeline}\n"
        f"  force_imputer    = {args.force_imputer or '(auto)'}\n"
        f"  device           = {args.device}\n"
        f"  auto_download    = {args.auto_download}\n"
        f"  out_root         = {OUT_ROOT}"
    )

    if args.full_pipeline:
        # Stage 2 semantics + ablation + paper outputs, all in one go.
        args.stage = 2
        phase_main_sweep(args)
        phase_imputer_ablation(args)
        phase_paper_outputs(args)
    elif args.ablation:
        phase_imputer_ablation(args)
    else:
        phase_main_sweep(args)

    banner("DONE  Option A sweep")
    print(f"\nResults under: {OUT_ROOT}")
    if args.full_pipeline:
        print(f"Tables       : {OUT_ROOT / 'tables'}")
        print(f"Dashboards   : {OUT_ROOT / 'dashboards'}")
    else:
        print("Next steps (build paper tables + dashboards):")
        print("  python paper_table_optionA.py --metric all --imputer_ablation")
        print("  python paper_dashboard.py     --headline-alpha 1 "
              "--headline-missing 0.10")
        print("Or skip these and re-run with --full_pipeline to chain "
              "everything in one command.")

    # Final milestone summary -- single source of truth for "did everything
    # finish?". Exits 0 on success, 1 if any milestone FAILed so wrapping
    # shell scripts / CI can pick it up via $?.
    all_ok = REGISTRY.summarise()
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
