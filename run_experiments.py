#!/usr/bin/env python3
"""
Sequential experiment runner for mpi_nn_train.py

Runs the full parameter grid and ensures every run writes a self-contained
result JSON and a stdout log. Also creates a concise summary CSV and optional
per-run history CSV extracted from the JSON.

Default grid follows the project requirements:
  - Activations: relu, sigmoid, tanh
  - Batch sizes: 32, 64, 128, 256, 512
  - Processes: 1, 2, 4, 8
  - Fixed: iters=5000 (others vary by activation)

Usage example:
  python run_experiments.py \
    --data-dir data_preprocessed_enhanced \
    --results-dir results \
    --procs 1 2 4 8 \
    --activations relu sigmoid tanh \
    --batches 32 64 128 256 512 \
    --iters 5000 --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Utilities
# -----------------------------


def find_mpi_launcher() -> str:
    """Return the preferred MPI launcher executable name.

    Tries mpiexec then mpirun. Raises if neither is found.
    """
    from shutil import which

    for cmd in ("mpiexec", "mpirun"):
        if which(cmd):
            return cmd
    raise RuntimeError("Neither 'mpiexec' nor 'mpirun' found in PATH. Please install MPI.")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def data_dir_ok(data_dir: Path) -> bool:
    req = [
        data_dir / "X_train_scaled.npy",
        data_dir / "X_test_scaled.npy",
        data_dir / "y_train.npy",
        data_dir / "y_test.npy",
    ]
    return all(x.exists() for x in req)


def default_grid() -> Tuple[List[str], List[int], List[int]]:
    activations = ["relu", "sigmoid", "tanh"]
    batches = [32, 64, 128, 256, 512]
    procs = [1, 2, 4, 8]
    return activations, batches, procs


def activation_defaults(act: str) -> Dict[str, object]:
    """Per-activation baseline hyperparameters.

    - tanh:    hidden=512,  lr=1.5e-3, weight_decay=2e-4, lr_decay=0.5, lr_decay_steps=1200
    - relu:    hidden=128,  lr=1e-3,   weight_decay=1e-5, lr_decay=1.0, lr_decay_steps=0
    - sigmoid: hidden=256,  lr=5e-4,   weight_decay=2e-4, lr_decay=1.0, lr_decay_steps=0
    """
    if act == "tanh":
        return {
            "hidden": 512,
            "lr": 1.5e-3,
            "weight_decay": 2e-4,
            "lr_decay": 0.5,
            "lr_decay_steps": 1200,
        }
    if act == "relu":
        return {
            "hidden": 128,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "lr_decay": 1.0,
            "lr_decay_steps": 0,
        }
    if act == "sigmoid":
        return {
            "hidden": 256,
            "lr": 5e-4,
            "weight_decay": 2e-4,
            "lr_decay": 1.0,
            "lr_decay_steps": 0,
        }
    # Fallback
    return {
        "hidden": 128,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "lr_decay": 1.0,
        "lr_decay_steps": 0,
    }


def run_cmd(cmd: List[str], log_path: Path, env: Optional[Dict[str, str]] = None) -> int:
    """Run a command, tee stdout+stderr to a log file, return exit code."""
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write("# CMD: " + shlex.join(cmd) + "\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
        )
        return proc.wait()


def combos(activations: Iterable[str], batches: Iterable[int], procs: Iterable[int]) -> Iterable[Tuple[str, int, int]]:
    for a in activations:
        for b in batches:
            for p in procs:
                yield a, b, p


def combo_key(act: str, batch: int, procs: int, seed: int) -> str:
    return f"act={act}_b={batch}_p={procs}_seed={seed}"


def result_paths(base: Path, act: str, batch: int, procs: int, seed: int) -> Dict[str, Path]:
    tag = combo_key(act, batch, procs, seed)
    return {
        "json": base / "raw" / f"{tag}.json",
        "log": base / "logs" / f"{tag}.log",
        "hist_csv": base / "history" / f"{tag}.csv",
    }


def write_history_csv(json_path: Path, csv_path: Path) -> None:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    history = data.get("history", [])
    ensure_dir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=["iter", "R"])
        w.writeheader()
        for row in history:
            # Only write expected keys; ignore extras if any
            w.writerow({"iter": row.get("iter"), "R": row.get("R")})


def append_summary(summary_csv: Path, json_path: Path) -> None:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cfg = data.get("config", {})
    metrics = data.get("metrics", {})
    row = {
        "activation": cfg.get("activation"),
        "batch": cfg.get("batch_size"),
        "processes": cfg.get("processes"),
        "hidden": cfg.get("n_hidden"),
        "lr": cfg.get("lr"),
        "iters": cfg.get("max_iters"),
        "rmse_train": metrics.get("rmse_train"),
        "rmse_test": metrics.get("rmse_test"),
        "time_total_sec": metrics.get("time_total_sec"),
        "time_compute_sec": metrics.get("time_compute_sec"),
        "time_comm_sec": metrics.get("time_comm_sec"),
        "history_file": str(json_path),
    }

    header = list(row.keys())
    new_file = not summary_csv.exists()
    ensure_dir(summary_csv.parent)
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        w.writerow(row)


# -----------------------------
# Runner
# -----------------------------


@dataclass
class RunConfig:
    data_dir: Path
    results_dir: Path
    activations: List[str]
    batches: List[int]
    procs: List[int]
    hidden: Optional[int]
    lr: Optional[float]
    lr_decay: Optional[float]
    lr_decay_steps: Optional[int]
    iters: int
    eval_every: int
    seed: int
    dtype: str
    standardize_y: bool
    log1p_y: bool
    weight_decay: Optional[float]
    grad_clip: float
    loss_clip: Optional[float]
    resume: bool


def run_one(cfg: RunConfig, launcher: str, act: str, batch: int, procs: int) -> Tuple[int, Path, Path]:
    paths = result_paths(cfg.results_dir, act, batch, procs, cfg.seed)
    json_path, log_path = paths["json"], paths["log"]
    ensure_dir(json_path.parent)
    ensure_dir(log_path.parent)

    # Skip if resume and file already exists
    if cfg.resume and json_path.exists():
        return 0, json_path, log_path

    # Derive effective hyperparameters: per-activation policy overridden by CLI if provided
    policy = activation_defaults(act)
    hidden = cfg.hidden if cfg.hidden is not None else int(policy["hidden"])
    lr = cfg.lr if cfg.lr is not None else float(policy["lr"])
    weight_decay = cfg.weight_decay if cfg.weight_decay is not None else float(policy["weight_decay"])
    lr_decay = cfg.lr_decay if cfg.lr_decay is not None else float(policy["lr_decay"])
    lr_decay_steps = cfg.lr_decay_steps if cfg.lr_decay_steps is not None else int(policy["lr_decay_steps"])
    loss_clip = cfg.loss_clip if cfg.loss_clip is not None else 9.0

    # Build command
    base_cmd = [
        launcher, "-n", str(procs), sys.executable, "mpi_nn_train.py",
        "--data-dir", str(cfg.data_dir),
        "--activation", act,
        "--hidden", str(hidden),
        "--batch", str(batch),
        "--lr", str(lr),
        "--lr-decay", str(lr_decay),
        "--lr-decay-steps", str(lr_decay_steps),
        "--iters", str(cfg.iters),
        "--eval-every", str(cfg.eval_every),
        "--seed", str(cfg.seed),
        "--history-out", str(json_path),
        "--dtype", cfg.dtype,
        "--stats-out", "1",  # flag-like: any non-empty enables extra stats
    ]
    if cfg.standardize_y:
        base_cmd.append("--standardize-y")
    if cfg.log1p_y:
        base_cmd.append("--log1p-y")
    if weight_decay and weight_decay > 0:
        base_cmd += ["--weight-decay", str(weight_decay)]
    if cfg.grad_clip and cfg.grad_clip > 0:
        base_cmd += ["--grad-clip", str(cfg.grad_clip)]
    if loss_clip and loss_clip > 0:
        base_cmd += ["--loss-clip", str(loss_clip)]

    rc = run_cmd(base_cmd, log_path)
    return rc, json_path, log_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sequential runner for mpi_nn_train grid experiments")
    parser.add_argument("--data-dir", default="data_preprocessed_enhanced", help="Directory with preprocessed .npy files")
    parser.add_argument("--results-dir", default="results", help="Output directory for results and logs")
    parser.add_argument("--activations", nargs="*", default=["relu", "sigmoid", "tanh"], help="Activation choices")
    parser.add_argument("--batches", nargs="*", type=int, default=[32, 64, 128, 256, 512], help="Batch sizes")
    parser.add_argument("--procs", nargs="*", type=int, default=[1, 2, 4, 8], help="MPI process counts")
    # None defaults allow per-activation policy to take effect
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr-decay", type=float, default=None)
    parser.add_argument("--lr-decay-steps", type=int, default=None)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--standardize-y", action="store_true", default=True)
    parser.add_argument("--log1p-y", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--loss-clip", type=float, default=9.0)
    parser.add_argument("--resume", action="store_true", help="Skip runs whose JSON already exists")
    parser.add_argument("--write-history-csv", action="store_true", help="Extract and save per-run history CSV")
    parser.add_argument("--summary-csv", default=None, help="Optional path for a summary CSV")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    ensure_dir(results_dir)
    ensure_dir(results_dir / "raw")
    ensure_dir(results_dir / "logs")
    ensure_dir(results_dir / "history")

    if not data_dir_ok(data_dir):
        print(
            f"[runner] Data directory '{data_dir}' missing required .npy files. "
            "Run preprocessing first (see preprocess_nytaxi_enhanced.py).",
            file=sys.stderr,
        )
        return 2

    try:
        launcher = find_mpi_launcher()
    except Exception as e:
        print(f"[runner] {e}", file=sys.stderr)
        return 3

    cfg = RunConfig(
        data_dir=data_dir,
        results_dir=results_dir,
        activations=list(args.activations),
        batches=list(args.batches),
        procs=list(args.procs),
        hidden=args.hidden,
        lr=args.lr,
        lr_decay=args.lr_decay,
        lr_decay_steps=args.lr_decay_steps,
        iters=args.iters,
        eval_every=args.eval_every,
        seed=args.seed,
        dtype=args.dtype,
        standardize_y=bool(args.standardize_y),
        log1p_y=bool(args.log1p_y),
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        loss_clip=args.loss_clip,
        resume=bool(args.resume),
    )

    total = len(cfg.activations) * len(cfg.batches) * len(cfg.procs)
    done = 0
    failures: List[Tuple[str, Path]] = []

    summary_csv = Path(args.summary_csv) if args.summary_csv else (results_dir / "summary" / "rmse_summary.csv")

    for act, batch, procs in combos(cfg.activations, cfg.batches, cfg.procs):
        tag = combo_key(act, batch, procs, cfg.seed)
        print(f"[runner] ({done+1}/{total}) Running {tag} ...", flush=True)
        rc, json_path, log_path = run_one(cfg, launcher, act, batch, procs)
        if rc != 0:
            print(f"[runner] FAILED: {tag} (rc={rc}). Log: {log_path}", file=sys.stderr)
            failures.append((tag, log_path))
        else:
            # Write history CSV if requested
            if args.write_history_csv:
                try:
                    write_history_csv(json_path, result_paths(cfg.results_dir, act, batch, procs, cfg.seed)["hist_csv"])
                except Exception as e:
                    print(f"[runner] Warning: history CSV failed for {tag}: {e}")

            # Append summary
            try:
                append_summary(summary_csv, json_path)
            except Exception as e:
                print(f"[runner] Warning: summary append failed for {tag}: {e}")

        done += 1

    print(f"[runner] Completed {done} runs. Failures: {len(failures)}")
    if failures:
        for tag, logp in failures:
            print(f"  - {tag} -> {logp}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
