#!/usr/bin/env python3
"""Benchmark the CuPy GPU AC3D prototype on cropped Berea grids."""

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
    cp,
    cupy_available,
    face_type_conductivity_gpu,
    gpu_memory_info,
    solve_ac3d_matrix_free_gpu_face_types,
    timed_gpu_call,
)
from pore_scale_electrical.ac3d_solver import (  # noqa: E402
    face_type_conductivity,
    solve_ac3d_matrix_free_face_types,
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


def parse_complex_dtype(name: str) -> np.dtype:
    if name == "complex64":
        return np.dtype(np.complex64)
    if name == "complex128":
        return np.dtype(np.complex128)
    raise ValueError("dtype must be complex64 or complex128")


def read_center_crop(raw_path: Path, shape: tuple[int, int, int], size: int, crop_start: tuple[int, int, int] | None) -> np.ndarray:
    volume = np.memmap(raw_path, dtype="<u2", mode="r", shape=shape, order="C")
    if size > min(shape):
        raise ValueError(f"crop size {size} exceeds full volume shape {shape}")
    start = crop_start if crop_start is not None else tuple((full - size) // 2 for full in shape)
    stop = tuple(s + size for s in start)
    if any(e > full for e, full in zip(stop, shape)):
        raise ValueError(f"requested crop {start}:{stop} exceeds shape {shape}")
    return np.asarray(volume[tuple(slice(s, e) for s, e in zip(start, stop))])


def nearest_spectrum_row(path: Path, frequency_hz: float) -> pd.Series:
    spectra = pd.read_csv(path)
    idx = np.abs(np.log(spectra["frequency_hz"].to_numpy(dtype=float)) - np.log(frequency_hz)).argmin()
    return spectra.iloc[int(idx)]


def write_plot(csv_path: Path, figure_path: Path) -> None:
    data = pd.read_csv(csv_path)
    if data.empty:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for dtype, group in data.groupby("dtype"):
        gpu = group[group["backend"] == "gpu"]
        if not gpu.empty:
            axes[0].plot(gpu["grid_size"], gpu["elapsed_s"], marker="o", label=f"GPU {dtype}")
        cpu = group[group["backend"] == "cpu"]
        if not cpu.empty:
            axes[0].plot(cpu["grid_size"], cpu["elapsed_s"], marker="s", linestyle="--", label=f"CPU {dtype}")
    axes[0].set_xlabel("grid size")
    axes[0].set_ylabel("elapsed seconds")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    gpu_rows = data[data["backend"] == "gpu"]
    for dtype, group in gpu_rows.groupby("dtype"):
        axes[1].plot(group["grid_size"], group["cuda_peak_used_gib"], marker="o", label=dtype)
    axes[1].set_xlabel("grid size")
    axes[1].set_ylabel("approx GPU memory used GiB")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(ROOT / "论文数据" / "microCT_Berea.raw"))
    parser.add_argument("--shape", nargs=3, default=("350", "350", "350"))
    parser.add_argument("--crop-start", nargs=3)
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--pore-label", type=int, default=1)
    parser.add_argument("--solid-label", type=int, default=2)
    parser.add_argument("--voxel-size-m", type=float, default=2.8e-6)
    parser.add_argument("--spectra", default=str(ROOT / "outputs" / "polarization_spectra_from_pnextract.csv"))
    parser.add_argument("--frequency", type=float, default=1.0)
    parser.add_argument("--direction", choices=["x", "y", "z"], default="x")
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--residual-every", type=int, default=10)
    parser.add_argument("--dtypes", nargs="+", choices=["complex64", "complex128"], default=["complex64", "complex128"])
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "ac3d_benchmarks"))
    parser.add_argument("--compare-cpu", action="store_true", help="also run CPU face-type solver")
    parser.add_argument("--cpu-compare-max-size", type=int, default=64)
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()

    if not cupy_available():
        raise SystemExit("CuPy/CUDA is not available in this Python environment")

    shape = parse_int3(args.shape, "--shape")
    if shape is None:
        raise ValueError("--shape is required")
    crop_start = parse_int3(args.crop_start, "--crop-start")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gpu_benchmark.csv"
    figure_path = ROOT / "figures" / "ac3d_cpu_gpu_benchmark.png"

    params = PolarizationParameters()
    spectrum_row = nearest_spectrum_row(Path(args.spectra), args.frequency)
    used_frequency = float(spectrum_row["frequency_hz"])
    omega = 2.0 * np.pi * used_frequency
    water_sigma = complex(
        float(spectrum_row["apparent_water_sigma_real_s_m"]),
        float(spectrum_row["apparent_water_sigma_imag_s_m"]),
    )
    solid_sigma = 1j * omega * params.solid_permittivity_f_m

    rows: list[dict[str, object]] = []
    for size in args.sizes:
        labels = read_center_crop(Path(args.raw), shape, size, crop_start)
        n_cells = int(labels.size)
        for dtype_name in args.dtypes:
            dtype = parse_complex_dtype(dtype_name)
            cp.get_default_memory_pool().free_all_blocks()
            before = gpu_memory_info()
            build_start = time.perf_counter()
            gpu_face_data = face_type_conductivity_gpu(
                labels,
                args.pore_label,
                args.solid_label,
                water_sigma,
                solid_sigma,
                dtype=dtype,
            )
            build_elapsed = time.perf_counter() - build_start
            result, solve_elapsed = timed_gpu_call(
                lambda: solve_ac3d_matrix_free_gpu_face_types(
                    gpu_face_data,
                    direction=args.direction,
                    voxel_size_m=args.voxel_size_m,
                    rtol=args.rtol,
                    atol=args.atol,
                    maxiter=args.maxiter,
                    residual_every=args.residual_every,
                    return_potential=args.compare_cpu and size <= args.cpu_compare_max_size,
                )
            )
            after = gpu_memory_info()
            gpu_row: dict[str, object] = {
                "backend": "gpu",
                "grid_size": size,
                "n_cells": n_cells,
                "dtype": dtype_name,
                "frequency_hz": used_frequency,
                "rtol": args.rtol,
                "atol": args.atol,
                "maxiter": args.maxiter,
                "iterations": result.iterations,
                "info": result.info,
                "relative_residual_norm": result.residual_norm,
                "effective_sigma_real_s_m": result.effective_conductivity_s_m.real,
                "effective_sigma_imag_s_m": result.effective_conductivity_s_m.imag,
                "build_elapsed_s": build_elapsed,
                "elapsed_s": solve_elapsed,
                "elapsed_per_iteration_s": solve_elapsed / max(result.iterations, 1),
                "cuda_total_gib": after["cuda_total_bytes"] / 1024**3,
                "cuda_free_before_gib": before["cuda_free_bytes"] / 1024**3,
                "cuda_free_after_gib": after["cuda_free_bytes"] / 1024**3,
                "cuda_peak_used_gib": max(0, before["cuda_free_bytes"] - after["cuda_free_bytes"]) / 1024**3,
                "cupy_pool_used_gib": after["cupy_pool_used_bytes"] / 1024**3,
                "cpu_effective_sigma_abs_error": np.nan,
                "cpu_potential_relative_error": np.nan,
            }
            rows.append(gpu_row)
            print(json.dumps(gpu_row, indent=2, ensure_ascii=False), flush=True)

            if args.compare_cpu and size <= args.cpu_compare_max_size:
                cpu_face_data = face_type_conductivity(
                    labels,
                    args.pore_label,
                    args.solid_label,
                    water_sigma,
                    solid_sigma,
                    dtype=dtype,
                )
                cpu_start = time.perf_counter()
                cpu_result = solve_ac3d_matrix_free_face_types(
                    cpu_face_data,
                    direction=args.direction,
                    voxel_size_m=args.voxel_size_m,
                    rtol=args.rtol,
                    atol=args.atol,
                    maxiter=args.maxiter,
                    residual_every=args.residual_every,
                    return_potential=result.potential is not None,
                )
                cpu_elapsed = time.perf_counter() - cpu_start
                sigma_error = abs(cpu_result.effective_conductivity_s_m - result.effective_conductivity_s_m)
                potential_error = np.nan
                if result.potential is not None and cpu_result.potential is not None:
                    denominator = max(float(np.linalg.norm(cpu_result.potential.ravel())), np.finfo(float).eps)
                    potential_error = float(np.linalg.norm(cpu_result.potential.ravel() - result.potential.ravel()) / denominator)
                gpu_row["cpu_effective_sigma_abs_error"] = sigma_error
                gpu_row["cpu_potential_relative_error"] = potential_error
                cpu_row = {
                    "backend": "cpu",
                    "grid_size": size,
                    "n_cells": n_cells,
                    "dtype": dtype_name,
                    "frequency_hz": used_frequency,
                    "rtol": args.rtol,
                    "atol": args.atol,
                    "maxiter": args.maxiter,
                    "iterations": cpu_result.iterations,
                    "info": cpu_result.info,
                    "relative_residual_norm": cpu_result.residual_norm,
                    "effective_sigma_real_s_m": cpu_result.effective_conductivity_s_m.real,
                    "effective_sigma_imag_s_m": cpu_result.effective_conductivity_s_m.imag,
                    "build_elapsed_s": np.nan,
                    "elapsed_s": cpu_elapsed,
                    "elapsed_per_iteration_s": cpu_elapsed / max(cpu_result.iterations, 1),
                    "cuda_total_gib": np.nan,
                    "cuda_free_before_gib": np.nan,
                    "cuda_free_after_gib": np.nan,
                    "cuda_peak_used_gib": np.nan,
                    "cupy_pool_used_gib": np.nan,
                    "cpu_effective_sigma_abs_error": 0.0,
                    "cpu_potential_relative_error": 0.0,
                }
                rows.append(cpu_row)
                print(json.dumps(cpu_row, indent=2, ensure_ascii=False), flush=True)
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    if not args.skip_plot:
        write_plot(csv_path, figure_path)
    print(f"wrote {csv_path}")
    if not args.skip_plot:
        print(f"wrote {figure_path}")


if __name__ == "__main__":
    main()
