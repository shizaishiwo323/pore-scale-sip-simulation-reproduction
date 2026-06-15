#!/usr/bin/env python3
"""Run one GPU matrix-free AC3D solve with residual history."""

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
    cupy_available,
    face_type_conductivity_gpu,
    gpu_memory_info,
    parse_gpu_complex_dtype,
    solve_ac3d_matrix_free_gpu_face_types,
    synchronize_gpu,
)
from pore_scale_electrical.polarization import PolarizationParameters  # noqa: E402


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


def nearest_spectrum_row(path: Path, frequency_hz: float) -> pd.Series:
    spectra = pd.read_csv(path)
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


def load_initial_guess(path: Path, expected_shape: tuple[int, int, int], dtype: np.dtype) -> np.ndarray:
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            key = "potential" if "potential" in loaded else "solution" if "solution" in loaded else loaded.files[0]
            array = loaded[key]
        finally:
            loaded.close()
    else:
        array = loaded
    array = np.asarray(array, dtype=dtype)
    if array.size != int(np.prod(expected_shape)):
        raise ValueError(f"x0 has {array.size} entries, expected {int(np.prod(expected_shape))}")
    return array.reshape(expected_shape)


def write_residual_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


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
    parser.add_argument("--frequency", type=float, default=1.0)
    parser.add_argument("--direction", choices=["x", "y", "z"], default="x")
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--residual-every", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--preconditioner", choices=["none", "jacobi", "fft"], default="jacobi")
    parser.add_argument("--fft-reference", choices=["mean-face", "mean-abs", "pore"], default="mean-face")
    parser.add_argument("--x0")
    parser.add_argument("--save-solution", action="store_true")
    parser.add_argument("--solution-path")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "ac3d_gpu_single"))
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    residual_history_path = out_dir / "residual_history.csv"
    result_path = out_dir / "matrix_free_gpu_single_result.json"

    params = PolarizationParameters()
    spectrum_row = nearest_spectrum_row(Path(args.spectra), args.frequency)
    used_frequency = float(spectrum_row["frequency_hz"])
    water_sigma, solid_sigma = phase_conductivities_from_spectrum_row(spectrum_row, used_frequency, params)

    labels = read_volume(Path(args.raw), shape, crop_start, crop_size)
    x0 = load_initial_guess(Path(args.x0), solve_shape, dtype) if args.x0 else None
    memory_before = gpu_memory_info()
    build_start = time.perf_counter()
    face_data = face_type_conductivity_gpu(labels, args.pore_label, args.solid_label, water_sigma, solid_sigma, dtype=dtype)
    build_elapsed = time.perf_counter() - build_start
    del labels

    residual_rows: list[dict[str, float | int]] = []
    solve_start = time.perf_counter()

    def report_progress(iterations: int) -> None:
        if args.progress_every > 0 and iterations % args.progress_every == 0:
            elapsed = time.perf_counter() - solve_start
            print(f"krylov_iterations={iterations} elapsed_s={elapsed:.1f}", flush=True)

    def report_residual(iterations: int, residual_norm: float) -> None:
        elapsed = time.perf_counter() - solve_start
        residual_rows.append({"iteration": iterations, "elapsed_s": elapsed, "relative_residual_norm": residual_norm})
        write_residual_history(residual_history_path, residual_rows)
        print(f"krylov_iterations={iterations} elapsed_s={elapsed:.1f} relative_residual_norm={residual_norm:.6e}", flush=True)

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
        return_potential=args.save_solution,
        iteration_callback=report_progress,
        x0=x0,
        residual_every=args.residual_every,
        residual_callback=report_residual if args.residual_every > 0 else None,
    )
    synchronize_gpu()
    solve_elapsed = time.perf_counter() - solve_start
    memory_after = gpu_memory_info()

    solution_path = None
    if args.save_solution:
        if result.potential is None:
            raise RuntimeError("GPU solver did not return a potential")
        solution_path = Path(args.solution_path) if args.solution_path else out_dir / "solution.npy"
        solution_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(solution_path, np.asarray(result.potential, dtype=dtype))

    output = {
        "solver": result.solver,
        "shape": shape,
        "crop_start": crop_start,
        "solve_shape": solve_shape,
        "n_cells": int(np.prod(solve_shape)),
        "dtype": args.dtype,
        "requested_frequency_hz": args.frequency,
        "frequency_hz": used_frequency,
        "direction": args.direction,
        "rtol": args.rtol,
        "atol": args.atol,
        "maxiter": args.maxiter,
        "preconditioner": args.preconditioner,
        "fft_reference": args.fft_reference,
        "iterations": result.iterations,
        "info": result.info,
        "relative_residual_norm": result.residual_norm,
        "water_sigma_real_s_m": water_sigma.real,
        "water_sigma_imag_s_m": water_sigma.imag,
        "solid_sigma_real_s_m": solid_sigma.real,
        "solid_sigma_imag_s_m": solid_sigma.imag,
        "effective_sigma_real_s_m": result.effective_conductivity_s_m.real,
        "effective_sigma_imag_s_m": result.effective_conductivity_s_m.imag,
        "mean_current_real_a_m2": result.mean_current_density_a_m2.real,
        "mean_current_imag_a_m2": result.mean_current_density_a_m2.imag,
        "build_elapsed_s": build_elapsed,
        "solve_elapsed_s": solve_elapsed,
        "elapsed_per_iteration_s": solve_elapsed / max(result.iterations, 1),
        "cuda_total_gib": memory_after["cuda_total_bytes"] / 1024**3,
        "cuda_free_before_gib": memory_before["cuda_free_bytes"] / 1024**3,
        "cuda_free_after_gib": memory_after["cuda_free_bytes"] / 1024**3,
        "cuda_used_delta_gib": max(0, memory_before["cuda_free_bytes"] - memory_after["cuda_free_bytes"]) / 1024**3,
        "cupy_pool_used_gib": memory_after["cupy_pool_used_bytes"] / 1024**3,
        "x0": args.x0,
        "residual_history_path": str(residual_history_path) if residual_rows else None,
        "solution_path": None if solution_path is None else str(solution_path),
    }
    result_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    print("wrote", result_path)


if __name__ == "__main__":
    main()
