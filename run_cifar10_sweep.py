#!/usr/bin/env python
"""
run_cifar10_sweep.py
====================

Master orchestrator for the CIFAR-10 paper sweep. This is a single-
dataset counterpart to `run_optionA_sweep.py` and `run_run3_sweep.py`
that keeps CIFAR-10 results isolated in their own `results/cifar10/`
tree.

    Dataset         : CIFAR-10 (10 classes, 32x32 RGB, torchvision auto-DL)
    Heterogeneity a : {0.1, 1, 10}
    Missing rate    : {0.0, 0.10, 0.20}
    Mechanism       : MCAR (random) only
    Algorithms      : FedAvg, FedProx, FedDistill, FedEnsemble, FedGen
    Imputer         : AM-DAE forced for the main sweep; ablation phase
                      additionally runs {mean, median, zero, none}.

Metrics recorded (all computable from stored y_true / y_pred / y_prob):
    Accuracy, Macro F1, Macro Precision, Macro Recall.

Per-cell namespacing (identical layout to Option A / Run 3):

    results/cifar10/cifar10/alpha<a>_miss<m>/<algo>/
        models/<TOKEN>_<algo>_<lr>_<num_users>u_<bs>b_<le>_<seed>.h5
        metrics/seed_<s>/<TOKEN>/<algo>_<TOKEN>_round_<R>.h5

Usage:

    python run_cifar10_sweep.py --full_pipeline --device cuda --times 3

    # Or single-seed sanity pass first (Stage 1, 1 seed per cell)
    python run_cifar10_sweep.py --stage 1 --device cuda --times 1

    # Only imputer ablation at the headline cell (alpha=1, miss=10%)
    python run_cifar10_sweep.py --ablation --device cuda

    # Dry-run: print every command without executing
    python run_cifar10_sweep.py --dry_run --full_pipeline
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
from typing import Optional

ROOT = Path(__file__).resolve().parent
PY = sys.executable
OUT_ROOT = ROOT / "results" / "cifar10"

# ---------------------------------------------------------------- locked scope
ALPHAS_DEFAULT = [0.1, 1.0, 10.0]
MISSING_RATES_DEFAULT = [0.0, 0.10, 0.20]
ALGOS_DEFAULT = ["FedAvg", "FedProx", "FedDistill", "FedEnsemble", "FedGen"]
DATASETS_DEFAULT = ["CIFAR10"]
SAMPLING_RATIO = 0.5
N_USERS_TOTAL = 20

# CIFAR-10 needs the full 200 rounds to converge on a small CNN, matching
# how the medical image slots were tuned in Run 3.
ROUNDS = {"CIFAR10": 200}

# Headline cell for imputer ablation + headline figures.
HEADLINE_ALPHA = 1.0
HEADLINE_MISS = 0.10
ABLATION_IMPUTERS = ["amdae", "mean", "median", "zero", "none"]


# ---------------------------------------------------------------- registry
class MilestoneRegistry:
    """Append-only success/failure log of every pipeline milestone.

    Writes three files alongside the sweep results:

      results/cifar10/_status.md            human-readable, grouped by phase
      results/cifar10/_status.json          machine-readable, full record list
      results/cifar10/_status_summary.json  flat snapshot with verdict / totals
    """

    PHASES_ORDER = [
        "bootstrap",
        "split_prep",
        "main_sweep",
        "imputer_ablation",
        "paper_outputs",
    ]
    PHASES_LABEL = {
        "bootstrap":        "Dataset bootstrap (torchvision download)",
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
        dur = (f" ({rec['duration_s']:.1f}s)"
               if rec['duration_s'] is not None else "")
        line = f"[{status}] {phase}/{item}{dur}"
        if message:
            line += f"  -- {message}"
        print(line, flush=True)
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
            "# CIFAR-10 sweep -- run status",
            "",
            f"_Run ID: `{self._run_id}`_  ",
            f"_Last update: "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}_  ",
            f"_Elapsed: {time.time() - self._t_start:.0f}s_",
            "",
        ]
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
            lines.append("")
            lines.append(f"PASS={n_pass}  FAIL={n_fail}  "
                         f"SKIP={n_skip}  INFO={n_info}")
            lines.append("")
            for r in recs:
                tick = {"PASS": "[PASS]", "FAIL": "[FAIL]",
                        "SKIP": "[SKIP]", "INFO": "[INFO]"}.get(
                            r["status"], "[----]")
                dur = (f"  ({r['duration_s']:.1f}s)"
                       if r["duration_s"] is not None else "")
                msg = f"  -- {r['message']}" if r["message"] else ""
                lines.append(f"- {tick} `{r['item']}`{dur}{msg}")
            lines.append("")

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
            "",
            f"### Verdict: **{verdict}**",
            "",
        ]
        self.md_path.write_text("\n".join(lines), encoding="utf-8")

    def summarise(self) -> bool:
        self._finished = True
        self._flush()

        n_pass = sum(1 for r in self.records if r["status"] == "PASS")
        n_fail = sum(1 for r in self.records if r["status"] == "FAIL")
        n_skip = sum(1 for r in self.records if r["status"] == "SKIP")
        bar = "=" * 78
        print(f"\n{bar}")
        print("CIFAR-10 STATUS SUMMARY")
        print(bar)
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
    return {"CIFAR10": "cifar10"}[dataset]


def _fmt_alpha(alpha: float) -> str:
    return str(alpha)


def dataset_token(dataset: str, alpha: float) -> str:
    """Token passed to main.py --dataset (matches utils/model_utils.py)."""
    a = _fmt_alpha(alpha)
    return f"{dataset}-alpha{a}-ratio{SAMPLING_RATIO}"


def dataset_split_dir(dataset: str, alpha: float) -> Path:
    """Filesystem path where the Dirichlet split lives."""
    a = _fmt_alpha(alpha)
    if dataset == "CIFAR10":
        return ROOT / "data" / "CIFAR10" / \
            f"u{N_USERS_TOTAL}-alpha{a}-ratio{SAMPLING_RATIO}"
    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------- split-prep
def ensure_split(args: argparse.Namespace, dataset: str, alpha: float) -> bool:
    """Generate the per-(dataset, alpha) Dirichlet split if missing.
    CIFAR-10 self-bootstraps its raw data via torchvision.datasets.CIFAR10
    from within generate_niid_dirichlet.py."""
    split_dir = dataset_split_dir(dataset, alpha)
    item = f"{dataset_short(dataset)}_alpha{alpha}"
    if (split_dir / "train").is_dir() and (split_dir / "test").is_dir():
        print(f"[ok] split present: {split_dir}")
        if REGISTRY is not None:
            REGISTRY.passed("split_prep", item, "already present")
        return True

    if not getattr(args, "auto_download", True):
        msg = (f"split missing at {split_dir} and --no_auto_download is set; "
               f"cannot auto-generate")
        print(f"[ERROR] {msg}", file=sys.stderr)
        if REGISTRY is not None:
            REGISTRY.failed("split_prep", item, msg)
        return False

    t0 = time.time()
    gen = ROOT / "data" / "CIFAR10" / "generate_niid_dirichlet.py"
    cwd = ROOT / "data" / "CIFAR10"
    cmd = [PY, str(gen),
           "--n_user", str(N_USERS_TOTAL),
           "--alpha", str(alpha),
           "--sampling_ratio", str(SAMPLING_RATIO)]

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
    base = OUT_ROOT / dataset_short(dataset) / \
        f"alpha{alpha}_miss{miss}" / algo
    if suffix:
        base = base / suffix
    return base


def expected_h5(args: argparse.Namespace, dataset: str, alpha: float,
                algo: str, seed: int, models_dir: Path) -> Path:
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
    """Move `results/metrics/<TOKEN>/*.h5` -> <cell>/metrics/seed_<s>/<TOKEN>/
    so back-to-back seeds don't overwrite each other."""
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


def _archive_stale_live_metrics(reason: str = "startup") -> Optional[Path]:
    """If `results/metrics/` is left over from a prior process, move it
    aside under a timestamped archive dir so the new sweep starts clean."""
    live = ROOT / "results" / "metrics"
    if not live.is_dir():
        return None
    try:
        has_content = any(live.iterdir())
    except OSError:
        has_content = True
    if not has_content:
        try:
            live.rmdir()
        except OSError:
            pass
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f") + "Z"
    target = ROOT / "results" / f"_stale_metrics_{stamp}"
    if target.exists():
        i = 2
        while (ROOT / "results" / f"_stale_metrics_{stamp}_{i}").exists():
            i += 1
        target = ROOT / "results" / f"_stale_metrics_{stamp}_{i}"
    print(f"[{reason}] Found leftover {live}/  --  archiving to {target.name}/ "
          f"so the new sweep starts clean.")
    try:
        shutil.move(str(live), str(target))
        if REGISTRY is not None:
            REGISTRY.info("bootstrap", "stale_metrics_archive",
                          f"moved {live} -> {target.name}")
        return target
    except OSError as exc:
        print(f"[WARN] could not archive stale {live}: {exc}", file=sys.stderr)
        if REGISTRY is not None:
            REGISTRY.failed("bootstrap", "stale_metrics_archive",
                            f"could not move {live}: {exc}")
        return None


def _check_no_stale_metrics() -> bool:
    _archive_stale_live_metrics(reason="pre-cell")
    return True


# ---------------------------------------------------------------- training
def train_cell(args: argparse.Namespace, dataset: str, alpha: float,
               miss: float, algo: str, seeds_wanted: int,
               force_imputer: Optional[str] = None,
               cell_suffix: str = "") -> None:
    """Train all `seeds_wanted` seeds for one cell, resume-skipping any
    seed whose summary HDF5 is already present."""
    base = cell_dir(dataset, alpha, miss, algo, suffix=cell_suffix)
    models_dir = base / "models"
    metrics_dir = base / "metrics"
    for d in (models_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    phase = ("imputer_ablation" if cell_suffix.startswith("imputer_ablation")
             else "main_sweep")

    todo_seeds = [s for s in range(seeds_wanted)
                  if not expected_h5(args, dataset, alpha, algo,
                                     s, models_dir).exists()]
    cell_id_base = (f"{dataset_short(dataset)}_alpha{alpha}_miss{miss}_{algo}"
                    + (f"_{cell_suffix.replace('/', '_')}"
                       if cell_suffix else ""))
    if not todo_seeds:
        print(f"[skip] {dataset_short(dataset):>10} a={alpha} m={miss} "
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
        if not args.dry_run:
            _check_no_stale_metrics()

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

        _relocate_per_round_metrics(args, metrics_dir, s)


# ---------------------------------------------------------------- phases
def phase_main_sweep(args: argparse.Namespace) -> None:
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
    banner("PAPER OUTPUTS  (tables + dashboards)")

    table_cmd = [PY, "paper_table_cifar10.py",
                 "--input-root", str(OUT_ROOT),
                 "--output-dir", str(OUT_ROOT / "tables"),
                 "--metric", "all",
                 "--imputer_ablation"]
    t0 = time.time()
    rc = run(table_cmd, dry=args.dry_run, allow_fail=True)
    dt = time.time() - t0
    if args.dry_run:
        if REGISTRY is not None:
            REGISTRY.info("paper_outputs", "paper_table_cifar10", "DRY RUN")
    elif rc != 0:
        print("[WARN] paper_table_cifar10.py failed (rc={}); see log."
              .format(rc), file=sys.stderr)
        if REGISTRY is not None:
            REGISTRY.failed("paper_outputs", "paper_table_cifar10",
                            f"rc={rc}", dt)
    else:
        if REGISTRY is not None:
            REGISTRY.passed("paper_outputs", "paper_table_cifar10",
                            f"tables -> {OUT_ROOT / 'tables'}", dt)

    dash_cmd = [PY, "paper_dashboard_cifar10.py",
                "--input-root", str(OUT_ROOT),
                "--output-dir", str(OUT_ROOT / "dashboards"),
                "--headline-alpha", "1.0",
                "--headline-missing", "0.10"]
    t0 = time.time()
    rc = run(dash_cmd, dry=args.dry_run, allow_fail=True)
    dt = time.time() - t0
    if args.dry_run:
        if REGISTRY is not None:
            REGISTRY.info("paper_outputs", "paper_dashboard_cifar10", "DRY RUN")
    elif rc != 0:
        print("[WARN] paper_dashboard_cifar10.py failed (rc={}); see log."
              .format(rc), file=sys.stderr)
        if REGISTRY is not None:
            REGISTRY.failed("paper_outputs", "paper_dashboard_cifar10",
                            f"rc={rc}", dt)
    else:
        if REGISTRY is not None:
            REGISTRY.passed("paper_outputs", "paper_dashboard_cifar10",
                            f"figures -> {OUT_ROOT / 'dashboards'}", dt)


def phase_imputer_ablation(args: argparse.Namespace) -> None:
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

    p.add_argument("--datasets", nargs="*", default=DATASETS_DEFAULT,
                   choices=DATASETS_DEFAULT,
                   help="Datasets to run (only CIFAR10 supported here).")
    p.add_argument("--alphas", nargs="*", type=float, default=ALPHAS_DEFAULT,
                   help=f"alpha values (default: {ALPHAS_DEFAULT}).")
    p.add_argument("--missing_rates", nargs="*", type=float,
                   default=MISSING_RATES_DEFAULT,
                   help=f"Missing rates (default: {MISSING_RATES_DEFAULT}).")
    p.add_argument("--algorithms", nargs="*", default=ALGOS_DEFAULT,
                   choices=ALGOS_DEFAULT,
                   help="FL algorithms (default: all 5).")

    p.add_argument("--local_epochs", type=int, default=20)
    p.add_argument("--num_users", type=int, default=10,
                   help="Sampled users per round (separate from N_USERS_TOTAL=20 "
                        "which controls the on-disk Dirichlet split size).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gen_batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=0.01)
    p.add_argument("--times", type=int, default=3,
                   help="Total seeds wanted in Stage 2 (default 3 for "
                        "journal-grade std-devs).")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    p.add_argument("--num_glob_iters_cifar10", type=int, default=None,
                   help=f"Override rounds for CIFAR-10 (default {ROUNDS['CIFAR10']}).")

    p.add_argument("--stage", type=int, default=1, choices=[1, 2],
                   help="1: main sweep with seeds=1 (Stage 1). "
                        "2: main sweep with seeds=times (Stage 2).")
    p.add_argument("--ablation", action="store_true",
                   help="Run ONLY the imputer-ablation phase.")
    p.add_argument("--full_pipeline", action="store_true",
                   help="One-shot: main sweep (Stage 2 semantics) + "
                        "imputer ablation + paper tables + dashboards.")

    p.add_argument("--force_imputer", default="amdae",
                   choices=["amdae", "mean", "median", "zero", "none", "auto"],
                   help="Imputer for the MAIN sweep (default amdae).")

    p.add_argument("--dry_run", action="store_true",
                   help="Print every command without executing.")
    p.add_argument("--no_auto_download", dest="auto_download",
                   action="store_false",
                   help="Disable automatic download of CIFAR-10 via torchvision.")
    p.set_defaults(auto_download=True)
    return p.parse_args()


def main() -> None:
    global REGISTRY
    args = parse_args()

    if args.force_imputer == "auto":
        args.force_imputer = None

    if args.num_glob_iters_cifar10 is not None:
        ROUNDS["CIFAR10"] = args.num_glob_iters_cifar10

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    REGISTRY = MilestoneRegistry(OUT_ROOT, dry=args.dry_run)

    banner(
        f"FedGen-AMDAE  ::  CIFAR-10 sweep  "
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

    if not args.dry_run:
        _archive_stale_live_metrics(reason="startup")

    if args.full_pipeline:
        args.stage = 2
        phase_main_sweep(args)
        phase_imputer_ablation(args)
        phase_paper_outputs(args)
    elif args.ablation:
        phase_imputer_ablation(args)
    else:
        phase_main_sweep(args)

    banner("DONE  CIFAR-10 sweep")
    print(f"\nResults under: {OUT_ROOT}")
    if args.full_pipeline:
        print(f"Tables       : {OUT_ROOT / 'tables'}")
        print(f"Dashboards   : {OUT_ROOT / 'dashboards'}")
    else:
        print("Next steps (build paper tables + dashboards):")
        print("  python paper_table_cifar10.py --metric all --imputer_ablation")
        print("  python paper_dashboard_cifar10.py --headline-alpha 1 "
              "--headline-missing 0.10")

    all_ok = REGISTRY.summarise()
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
