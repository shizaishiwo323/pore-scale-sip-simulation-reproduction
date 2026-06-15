#!/usr/bin/env python3
"""Run a GPU matrix-free AC3D frequency sweep with warm starts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pore_scale_electrical.ac3d_gpu import (  # noqa: E402
    GPUFaceTypeConductivity,
    cp,
    cupy_available,
    gpu_memory_info,
    parse_gpu_complex_dtype,
    solve_ac3d_matrix_free_gpu_face_types,
    synchronize_gpu,
)
from pore_scale_electrical.ac3d_solver import build_face_type_codes, face_type_conductance_values  # noqa: E402
from pore_scale_electrical.polarization import PolarizationParameters  # noqa: E402


DEFAULT_FREQUENCIES = [1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e8, 1.0e9]


def parse_int3(values: list[str] | None, name: str) -> tuple[int, int, int] | None:
    if values is None:
        return None
    if len(values) != 3:
        raise ValueError(f"{name} expects exactly three integers")
    parsed = tuple(int(v) for v in values)
    if any(v < 0 for v in parsed):
        raise ValueError(f"{name} must be non-negative")
    return parsed


def read_volume(raw_path: Path, shape: tuple[int, int, int], crop_start: tuple[int, int, int] | None, crop_size: tuple[int, int, int] | None) -> np.ndarray:
    volume = np.memmap(raw_path, dtype="<u2", mode="r", shape=shape, order="C")
    if crop_start is None or crop_size is None:
        return np.asarray(volume)
    stop = tuple(s + n for s, n in zip(crop_start, crop_size))
    if any(e > full for e, full in zip(stop, shape)):
        raise ValueError(f"requested subvolume {crop_start}:{stop} exceeds shape {shape}")
    return np.asarray(volume[tuple(slice(s, e) for s, e in zip(crop_start, stop))])


def load_spectra(path: Path) -> pd.DataFrame:
    spectra = pd.read_csv(path)
    required = {"frequency_hz", "apparent_water_sigma_real_s_m", "apparent_water_sigma_imag_s_m"}
    missing = required.difference(spectra.columns)
    if missing:
        raise ValueError(f"spectra file is missing columns: {sorted(missing)}")
    return spectra


def nearest_spectrum_row(spectra: pd.DataFrame, frequency_hz: float) -> pd.Series:
    idx = np.abs(np.log(spectra["frequency_hz"].to_numpy(dtype=float)) - np.log(frequency_hz)).argmin()
    return spectra.iloc[int(idx)]


def phase_conductivities_from_spectrum_row(
    spectrum_row: pd.Series,
    frequency_hz: float,
    params: PolarizationParameters,
) -> tuple[complex, complex]:
    water_sigma = complex(
        float(spectrum_row["apparent_water_sigma_real_s_m"]),
        float(spectrum_row["apparent_water_sigma_imag_s_m"]),
    )
    if {"solid_sigma_real_s_m", "solid_sigma_imag_s_m"}.issubset(spectrum_row.index):
        solid_sigma = complex(
            float(spectrum_row["solid_sigma_real_s_m"]),
            float(spectrum_row["solid_sigma_imag_s_m"]),
        )
    else:
        omega = 2.0 * np.pi * frequency_hz
        solid_sigma = 1j * omega * params.solid_permittivity_f_m
    return water_sigma, solid_sigma


def frequency_tag(frequency_hz: float) -> str:
    return f"{frequency_hz:.6e}".replace("+", "").replace("-", "m").replace(".", "p")


def write_residual_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def write_config(path: Path, config: dict[str, object]) -> None:
    lines: list[str] = []
    for key, value in config.items():
        if isinstance(value, (list, tuple)):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(ROOT / "论文数据" / "microCT_Berea.raw"))
    parser.add_argument("--shape", nargs=3, default=("350", "350", "350"))
    parser.add_argument("--crop-start", nargs=3)
    parser.add_argument("--crop-size", nargs=3)
    parser.add_argument("--pore-label", type=int, default=1)
    parser.add_argument("--solid-label", type=int, default=2)
    parser.add_argument("--voxel-size-m", type=float, default=2.8e-6)
    parser.add_argument("--spectra", default=str(ROOT / "outputs" / "polarization_spectra_from_pnextract.csv"))
    parser.add_argument("--frequencies", nargs="+", type=float, default=DEFAULT_FREQUENCIES)
    parser.add_argument("--frequency-order", choices=["ascending", "descending", "input"], default="ascending")
    parser.add_argument("--direction", choices=["x", "y", "z"], default="x")
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--residual-every", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--preconditioner", choices=["none", "jacobi", "fft"], default="jacobi")
    parser.add_argument("--fft-reference", choices=["mean-face", "mean-abs", "pore"], default="mean-face")
    parser.add_argument("--save-solutions", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "ac3d_gpu_sweep"))
    args = parser.parse_args()

    if not cupy_available():
        raise SystemExit("CuPy/CUDA is not available in this Python environment")

    shape = parse_int3(args.shape, "--shape")
    if shape is None:
        raise ValueError("--shape is required")
    crop_start = parse_int3(args.crop_start, "--crop-start")
    crop_size = parse_int3(args.crop_size, "--crop-size")
    solve_shape = crop_size if crop_size else shape
    dtype = parse_gpu_complex_dtype(args.dtype)
    frequencies = list(args.frequencies)
    if args.frequency_order == "ascending":
        frequencies.sort()
    elif args.frequency_order == "descending":
        frequencies.sort(reverse=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config(
        out_dir / "config.yml",
        {
            "raw": str(Path(args.raw)),
            "shape": shape,
            "crop_start": crop_start,
            "crop_size": crop_size,
            "frequencies_requested_hz": frequencies,
            "direction": args.direction,
            "rtol": args.rtol,
            "atol": args.atol,
            "maxiter": args.maxiter,
            "dtype": args.dtype,
            "preconditioner": args.preconditioner,
            "fft_reference": args.fft_reference,
            "residual_every": args.residual_every,
            "save_solutions": args.save_solutions,
            "gpu_memory_at_start": gpu_memory_info(),
        },
    )

    labels = read_volume(Path(args.raw), shape, crop_start, crop_size)
    cpu_face_types = build_face_type_codes(labels, args.pore_label, args.solid_label)
    gpu_face_types = tuple(cp.asarray(codes, dtype=cp.uint8) for codes in cpu_face_types)
    del labels
    del cpu_face_types

    params = PolarizationParameters()
    spectra = load_spectra(Path(args.spectra))
    previous_solution: np.ndarray | None = None
    summary_rows: list[dict[str, object]] = []
    summary_path = out_dir / "sweep_results.csv"

    for index, requested_frequency in enumerate(frequencies):
        spectrum_row = nearest_spectrum_row(spectra, requested_frequency)
        used_frequency = float(spectrum_row["frequency_hz"])
        frequency_dir = out_dir / f"frequency_{index:03d}_{frequency_tag(used_frequency)}Hz"
        frequency_dir.mkdir(parents=True, exist_ok=True)
        result_path = frequency_dir / "result.json"
        solution_path = frequency_dir / "solution.npy"
        residual_history_path = frequency_dir / "residual_history.csv"

        if args.resume and result_path.exists():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            summary_rows.append(existing)
            if solution_path.exists():
                previous_solution = np.load(solution_path).astype(dtype, copy=False)
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            print(f"skipped existing {used_frequency:g} Hz", flush=True)
            continue

        water_sigma, solid_sigma = phase_conductivities_from_spectrum_row(spectrum_row, used_frequency, params)
        conductance_values = cp.asarray(face_type_conductance_values(water_sigma, solid_sigma, dtype=dtype), dtype=dtype)
        face_data = GPUFaceTypeConductivity(face_types=gpu_face_types, conductance_values=conductance_values, shape=solve_shape, dtype=dtype)
        memory_before = gpu_memory_info()
        start_time = time.perf_counter()
        residual_rows: list[dict[str, float | int]] = []

        def report_progress(iterations: int) -> None:
            if args.progress_every > 0 and iterations % args.progress_every == 0:
                elapsed = time.perf_counter() - start_time
                print(f"{used_frequency:g}Hz krylov_iterations={iterations} elapsed_s={elapsed:.1f}", flush=True)

        def report_residual(iterations: int, residual_norm: float) -> None:
            elapsed = time.perf_counter() - start_time
            residual_rows.append({"iteration": iterations, "elapsed_s": elapsed, "relative_residual_norm": residual_norm})
            write_residual_history(residual_history_path, residual_rows)
            print(f"{used_frequency:g}Hz krylov_iterations={iterations} elapsed_s={elapsed:.1f} relative_residual_norm={residual_norm:.6e}", flush=True)

        result = solve_ac3d_matrix_free_gpu_face_types(
            face_data,
            direction=args.direction,
            voxel_size_m=args.voxel_size_m,
            rtol=args.rtol,
            atol=args.atol,
            maxiter=args.maxiter,
            use_jacobi=args.preconditioner == "jacobi",
            preconditioner=args.preconditioner,
            fft_reference=args.fft_reference,
            return_potential=True,
            iteration_callback=report_progress,
            x0=previous_solution,
            residual_every=args.residual_every,
            residual_callback=report_residual if args.residual_every > 0 else None,
        )
        synchronize_gpu()
        elapsed_total = time.perf_counter() - start_time
        memory_after = gpu_memory_info()

        previous_solution = None if result.potential is None else np.asarray(result.potential, dtype=dtype)
        saved_solution_path = None
        if args.save_solutions and previous_solution is not None:
            np.save(solution_path, previous_solution)
            saved_solution_path = str(solution_path)

        output = {
            "solver": result.solver,
            "requested_frequency_hz": requested_frequency,
            "frequency_hz": used_frequency,
            "direction": args.direction,
            "shape": shape,
            "crop_start": crop_start,
            "solve_shape": solve_shape,
            "n_cells": int(np.prod(solve_shape)),
            "dtype": args.dtype,
            "water_sigma_real_s_m": water_sigma.real,
            "water_sigma_imag_s_m": water_sigma.imag,
            "solid_sigma_real_s_m": solid_sigma.real,
            "solid_sigma_imag_s_m": solid_sigma.imag,
            "effective_sigma_real_s_m": result.effective_conductivity_s_m.real,
            "effective_sigma_imag_s_m": result.effective_conductivity_s_m.imag,
            "mean_current_real_a_m2": result.mean_current_density_a_m2.real,
            "mean_current_imag_a_m2": result.mean_current_density_a_m2.imag,
            "relative_residual_norm": result.residual_norm,
            "iterations": result.iterations,
            "info": result.info,
            "rtol": args.rtol,
            "atol": args.atol,
            "maxiter": args.maxiter,
            "preconditioner": args.preconditioner,
            "fft_reference": args.fft_reference,
            "used_warm_start": index > 0,
            "elapsed_total_s": elapsed_total,
            "elapsed_per_iteration_s": elapsed_total / max(result.iterations, 1),
            "cuda_total_gib": memory_after["cuda_total_bytes"] / 1024**3,
            "cuda_free_before_gib": memory_before["cuda_free_bytes"] / 1024**3,
            "cuda_free_after_gib": memory_after["cuda_free_bytes"] / 1024**3,
            "cuda_used_delta_gib": max(0, memory_before["cuda_free_bytes"] - memory_after["cuda_free_bytes"]) / 1024**3,
            "residual_history_path": str(residual_history_path) if residual_rows else None,
            "solution_path": saved_solution_path,
            "result_path": str(result_path),
        }
        result_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        summary_rows.append(output)
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(json.dumps(output, indent=2, ensure_ascii=False), flush=True)

    print("wrote", summary_path)


if __name__ == "__main__":
    main()
