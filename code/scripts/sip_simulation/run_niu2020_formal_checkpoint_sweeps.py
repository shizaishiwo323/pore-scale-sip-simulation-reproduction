#!/usr/bin/env python3
"""Run resumable Niu 2020 full-grid checkpoint sweeps."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent

DEFAULT_RESULT_DIR = PROJECT_ROOT / "results" / "niu2020_berea_reproduction_20260625_solver_hardening"
DEFAULT_MECHANISMS = ("interfacial", "pore", "membrane", "all")
DEFAULT_DIRECTIONS = ("x", "y", "z")
DEFAULT_CHECKPOINT_FREQUENCIES = (1e-3, 1e-2, 1e-1, 1.0, 1e3, 1e6, 1e9)


class CheckpointTask(NamedTuple):
    mechanism: str
    direction: str
    frequency_hz: float
    run_name: str
    out_dir: Path
    sweep_results_csv: Path
    command: list[str]


def frequency_tag(frequency_hz: float) -> str:
    if frequency_hz == 1.0:
        return "1"
    if frequency_hz >= 10.0 or frequency_hz < 1.0:
        return f"{frequency_hz:.0e}".replace("+0", "").replace("+", "")
    return f"{frequency_hz:g}"


def run_name_for(mechanism: str, direction: str, frequency_hz: float) -> str:
    return f"chk_{mechanism}_{direction}_{frequency_tag(frequency_hz)}_c128"


def is_successful_sweep(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return False
    if frame.empty or "info" not in frame or "true_residual_passed" not in frame:
        return False
    return bool((frame["info"].astype(int) == 0).all() and frame["true_residual_passed"].astype(bool).all())


def build_tasks(
    *,
    result_dir: Path,
    mechanisms: tuple[str, ...] = DEFAULT_MECHANISMS,
    directions: tuple[str, ...] = DEFAULT_DIRECTIONS,
    frequencies: tuple[float, ...] = DEFAULT_CHECKPOINT_FREQUENCIES,
    python_exe: str = sys.executable,
    rtol: str = "1e-5",
    maxiter: str = "1000",
    residual_every: str = "50",
) -> list[CheckpointTask]:
    tasks: list[CheckpointTask] = []
    sweep_script = PROJECT_ROOT / "code" / "scripts" / "sip_simulation" / "run_ac3d_matrix_free_gpu_sweep.py"
    raw = PROJECT_ROOT / "data" / "Niu 2020data" / "microCT_Berea.raw"
    spectra_dir = result_dir / "source_data" / "formal_inputs" / "component_spectra_checkpoints"
    for mechanism in mechanisms:
        spectra = spectra_dir / f"polarization_spectra_{mechanism}.csv"
        for direction in directions:
            for frequency in frequencies:
                run_name = run_name_for(mechanism, direction, frequency)
                out_dir = result_dir / "simulation_sweeps" / run_name
                command = [
                    python_exe,
                    str(sweep_script),
                    "--raw",
                    str(raw),
                    "--shape",
                    "350",
                    "350",
                    "350",
                    "--pore-label",
                    "1",
                    "--solid-label",
                    "2",
                    "--voxel-size-m",
                    "2.8e-6",
                    "--spectra",
                    str(spectra),
                    "--frequencies",
                    f"{frequency:g}",
                    "--frequency-match-mode",
                    "exact",
                    "--direction",
                    direction,
                    "--dtype",
                    "complex128",
                    "--preconditioner",
                    "fft",
                    "--fft-reference",
                    "pore",
                    "--gauge-mode",
                    "auto",
                    "--rtol",
                    rtol,
                    "--atol",
                    "0",
                    "--maxiter",
                    maxiter,
                    "--residual-every",
                    residual_every,
                    "--progress-every",
                    residual_every,
                    "--out-dir",
                    str(out_dir),
                ]
                tasks.append(
                    CheckpointTask(
                        mechanism=mechanism,
                        direction=direction,
                        frequency_hz=float(frequency),
                        run_name=run_name,
                        out_dir=out_dir,
                        sweep_results_csv=out_dir / "sweep_results.csv",
                        command=command,
                    )
                )
    return tasks


def task_record(task: CheckpointTask, status: str) -> dict[str, object]:
    return {
        "mechanism": task.mechanism,
        "direction": task.direction,
        "frequency_hz": task.frequency_hz,
        "run_name": task.run_name,
        "status": status,
        "sweep_results_csv": str(task.sweep_results_csv),
        "command": task.command,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--mechanisms", nargs="+", default=list(DEFAULT_MECHANISMS))
    parser.add_argument("--directions", nargs="+", default=list(DEFAULT_DIRECTIONS))
    parser.add_argument("--frequencies", nargs="+", type=float, default=list(DEFAULT_CHECKPOINT_FREQUENCIES))
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()

    tasks = build_tasks(
        result_dir=args.result_dir,
        mechanisms=tuple(args.mechanisms),
        directions=tuple(args.directions),
        frequencies=tuple(args.frequencies),
        python_exe=args.python_exe,
    )
    records: list[dict[str, object]] = []
    runnable = 0
    for task in tasks:
        if not args.force and is_successful_sweep(task.sweep_results_csv):
            records.append(task_record(task, "skipped_successful"))
            continue
        if args.max_tasks is not None and runnable >= args.max_tasks:
            records.append(task_record(task, "not_started_max_tasks"))
            continue
        runnable += 1
        if args.dry_run:
            records.append(task_record(task, "dry_run"))
            continue
        completed = subprocess.run(task.command, cwd=PROJECT_ROOT, check=False)
        records.append(task_record(task, "completed" if completed.returncode == 0 else f"failed_{completed.returncode}"))
        if completed.returncode != 0:
            break

    summary = {
        "result_dir": str(args.result_dir),
        "dry_run": bool(args.dry_run),
        "force": bool(args.force),
        "task_count": len(tasks),
        "records": records,
    }
    summary_json = args.summary_json or args.result_dir / "provenance" / "formal_checkpoint_sweep_batch_summary.json"
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
