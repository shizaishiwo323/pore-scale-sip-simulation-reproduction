#!/usr/bin/env python3
"""Run 3-D SIP mechanism sweeps and plot against PSIP data.

The original defaults target sample 89, but all paths, labels, and output
prefixes are configurable so the same workflow can be reused for other plugs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "code"
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_polarization_spectra import compute_spectra, load_pnextract  # noqa: E402
from make_polarization_component_spectra import make_component_spectrum  # noqa: E402
from pore_scale_electrical.ac3d_gpu import (  # noqa: E402
    GPUFaceTypeConductivity,
    cp,
    cupy_available,
    gpu_memory_info,
    parse_gpu_complex_dtype,
    solve_ac3d_bicgstab_compact_gpu_face_types,
    solve_ac3d_cocg_gpu_face_types,
    solve_ac3d_red_black_sor_gpu_face_types,
    solve_ac3d_matrix_free_gpu_face_types,
    solve_ac3d_weighted_jacobi_gpu_face_types,
    synchronize_gpu,
)
from pore_scale_electrical.ac3d_solver import build_face_type_codes, face_type_conductance_values  # noqa: E402
from pore_scale_electrical.polarization import PolarizationParameters  # noqa: E402


MECHANISM_TO_COMPONENT = {
    "maxwell": "interfacial",
    "pore": "pore",
    "membrane": "membrane",
    "all": "all",
}
MECHANISM_LABELS = {
    "maxwell": "Maxwell",
    "pore": "Pore",
    "membrane": "Membrane",
    "all": "Total",
}
MECHANISM_COLORS = {
    "maxwell": "#2a6f97",
    "pore": "#b56576",
    "membrane": "#6a994e",
    "all": "#222222",
}
MHD_DTYPES = {
    "MET_UCHAR": np.dtype("uint8"),
    "MET_CHAR": np.dtype("int8"),
    "MET_USHORT": np.dtype("<u2"),
    "MET_SHORT": np.dtype("<i2"),
    "MET_UINT": np.dtype("<u4"),
    "MET_INT": np.dtype("<i4"),
    "MET_FLOAT": np.dtype("<f4"),
    "MET_DOUBLE": np.dtype("<f8"),
}


def frequency_tag(frequency_hz: float) -> str:
    return f"{frequency_hz:.6e}".replace("+", "").replace("-", "m").replace(".", "p")


def parse_psip_csv(path: Path) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header: list[str] | None = None
        for row in reader:
            if not row:
                continue
            if row[0] == "X_Value" and "Frequency[Hz]" in row:
                header = row
                continue
            if header is None or len(row) < len(header):
                continue
            record = dict(zip(header, row))
            if not record.get("Frequency[Hz]"):
                continue
            frequency = float(record["Frequency[Hz]"])
            magnitude = float(record["Magnitude[ratio]"])
            phase = float(record["Phase_Shift[rad]"])
            rows.append(
                {
                    "frequency_hz": frequency,
                    "magnitude_ratio": magnitude,
                    "phase_shift_rad": phase,
                    "experiment_real_raw": magnitude * np.cos(phase),
                    "experiment_imag_raw": magnitude * np.sin(phase),
                }
            )
    if not rows:
        raise ValueError(f"no PSIP data rows found in {path}")
    return pd.DataFrame(rows).sort_values("frequency_hz").reset_index(drop=True)


def add_bulk_conductivity_from_psip_impedance(
    experiment: pd.DataFrame,
    *,
    current_resistor_ohm: float,
    sample_length_cm: float,
    sample_diameter_cm: float,
) -> pd.DataFrame:
    out = experiment.copy()
    length_m = sample_length_cm / 100.0
    radius_m = (sample_diameter_cm / 100.0) / 2.0
    area_m2 = np.pi * radius_m**2
    geometry_factor_m_inv = length_m / area_m2
    impedance = current_resistor_ohm * out["magnitude_ratio"].to_numpy(dtype=float) * np.exp(
        1j * out["phase_shift_rad"].to_numpy(dtype=float)
    )
    admittance = 1.0 / impedance
    conductivity = admittance * geometry_factor_m_inv
    out["experiment_admittance_real_s"] = admittance.real
    out["experiment_admittance_imag_s"] = admittance.imag
    out["experiment_sigma_real_s_m"] = conductivity.real
    out["experiment_sigma_imag_s_m"] = conductivity.imag
    out["current_resistor_ohm"] = current_resistor_ohm
    out["sample_length_cm"] = sample_length_cm
    out["sample_diameter_cm"] = sample_diameter_cm
    out["geometry_factor_m_inv"] = geometry_factor_m_inv
    return out


def build_polarization_parameters(
    *,
    water_conductivity_s_m: float,
    surface_conductance_s: float,
    membrane_polarizability: float,
    solid_relative_permittivity: float,
    dynamic_pore_size_m: float,
    diffusion_coefficient_m2_s: float | None = None,
    water_relative_permittivity: float | None = None,
) -> PolarizationParameters:
    params = PolarizationParameters()
    updates: dict[str, float] = {
        "water_conductivity_s_m": water_conductivity_s_m,
        "surface_conductance_s": surface_conductance_s,
        "membrane_polarizability": membrane_polarizability,
        "solid_relative_permittivity": solid_relative_permittivity,
        "dynamic_pore_size_m": dynamic_pore_size_m,
    }
    if diffusion_coefficient_m2_s is not None:
        updates["diffusion_coefficient_m2_s"] = diffusion_coefficient_m2_s
    if water_relative_permittivity is not None:
        updates["water_relative_permittivity"] = water_relative_permittivity
    return replace(params, **updates)


def parse_mhd(path: Path) -> dict[str, object]:
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    dims_xyz = tuple(int(value) for value in values["DimSize"].split())
    element_size_um_xyz = tuple(float(value) for value in values["ElementSize"].split())
    element_type = values["ElementType"]
    if element_type not in MHD_DTYPES:
        raise ValueError(f"unsupported MHD ElementType: {element_type}")
    raw_path = Path(values["ElementDataFile"])
    if not raw_path.is_absolute():
        raw_path = Path(path).parent / raw_path
    return {
        "path": Path(path),
        "raw_path": raw_path,
        "shape_zyx": (dims_xyz[2], dims_xyz[1], dims_xyz[0]),
        "dims_xyz": dims_xyz,
        "element_size_um_xyz": element_size_um_xyz,
        "voxel_size_m": element_size_um_xyz[0] * 1.0e-6,
        "dtype": MHD_DTYPES[element_type],
    }


def load_label_volume(mhd_info: dict[str, object]) -> np.ndarray:
    shape = mhd_info["shape_zyx"]
    dtype = mhd_info["dtype"]
    raw_path = mhd_info["raw_path"]
    volume = np.memmap(raw_path, dtype=dtype, mode="r", shape=shape, order="C")
    return np.asarray(volume)


def phase_conductivities(row: pd.Series, frequency_hz: float) -> tuple[complex, complex]:
    water_sigma = complex(float(row["apparent_water_sigma_real_s_m"]), float(row["apparent_water_sigma_imag_s_m"]))
    solid_sigma = complex(float(row["solid_sigma_real_s_m"]), float(row["solid_sigma_imag_s_m"]))
    return water_sigma, solid_sigma


def nearest_spectrum_row(spectra: pd.DataFrame, frequency_hz: float) -> pd.Series:
    values = spectra["frequency_hz"].to_numpy(dtype=float)
    idx = int(np.abs(np.log(values) - np.log(frequency_hz)).argmin())
    return spectra.iloc[idx]


def can_reuse_solution_as_warm_start(*, info: int, residual_norm: float, rtol: float) -> bool:
    return int(info) == 0 and float(residual_norm) <= float(rtol)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def make_component_spectra_files(
    *,
    output_prefix: str,
    frequencies_hz: np.ndarray,
    network_dir: Path,
    out_dir: Path,
    mode: str,
    params: PolarizationParameters,
    solid_conductivity_s_m: float,
    solid_background_components: set[str],
    pore_radius_scale: float = 1.0,
    membrane_length_scale: float = 1.0,
    membrane_zdc_scale: float = 1.0,
) -> tuple[Path, dict[str, Path], dict[str, object]]:
    pores, throats = load_pnextract(network_dir, params)
    base, base_metadata = compute_spectra(
        frequencies_hz,
        pores,
        throats,
        params,
        figure5_zdc_ohm=None,
        pore_radius_scale=pore_radius_scale,
        membrane_length_scale=membrane_length_scale,
        membrane_zdc_scale=membrane_zdc_scale,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = out_dir / f"{output_prefix}_polarization_spectra_base.csv"
    base.to_csv(base_path, index=False)

    component_paths: dict[str, Path] = {}
    for mechanism, component in MECHANISM_TO_COMPONENT.items():
        frame = make_component_spectrum(base, component, params, mode)
        frame["mechanism"] = mechanism
        if mechanism in solid_background_components:
            frame["solid_sigma_real_s_m"] = solid_conductivity_s_m
        path = out_dir / f"{output_prefix}_component_spectra_{mechanism}.csv"
        frame.to_csv(path, index=False)
        component_paths[mechanism] = path

    metadata = {
        "network_dir": str(network_dir),
        "base_spectra": str(base_path),
        "component_mode": mode,
        "frequencies_hz": frequencies_hz.tolist(),
        "base_metadata": base_metadata,
        "solid_conductivity_s_m": solid_conductivity_s_m,
        "solid_background_components": sorted(solid_background_components),
        "component_paths": {key: str(value) for key, value in component_paths.items()},
    }
    write_json(out_dir / f"{output_prefix}_component_spectra_metadata.json", metadata)
    return base_path, component_paths, metadata


def run_component_sweep(
    *,
    labels: np.ndarray,
    mechanism: str,
    spectra_path: Path,
    frequencies_hz: list[float],
    voxel_size_m: float,
    out_dir: Path,
    pore_label: int,
    solid_label: int,
    direction: str,
    rtol: float,
    atol: float,
    maxiter: int,
    dtype_name: str,
    preconditioner: str,
    fft_reference: str,
    progress_every: int,
    residual_every: int,
    iteration_sleep_s: float,
    zero_solid_regularization_s_m: float,
    resume: bool,
    solver_backend: str,
    jacobi_omega: float,
    sor_omega: float,
    max_new_results: int | None = None,
    accept_residual_le: float | None = None,
) -> pd.DataFrame:
    dtype = parse_gpu_complex_dtype(dtype_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    cpu_face_types = build_face_type_codes(labels, pore_label, solid_label)
    gpu_face_types = tuple(cp.asarray(codes, dtype=cp.uint8) for codes in cpu_face_types)
    del cpu_face_types

    spectra = pd.read_csv(spectra_path)
    previous_solution: np.ndarray | None = None
    rows: list[dict[str, object]] = []
    summary_path = out_dir / "sweep_results.csv"
    new_results = 0

    for index, requested_frequency in enumerate(frequencies_hz):
        spectrum_row = nearest_spectrum_row(spectra, requested_frequency)
        used_frequency = float(spectrum_row["frequency_hz"])
        frequency_dir = out_dir / f"frequency_{index:03d}_{frequency_tag(used_frequency)}Hz"
        result_path = frequency_dir / "result.json"
        residual_history_path = frequency_dir / "residual_history.csv"
        frequency_dir.mkdir(parents=True, exist_ok=True)

        if resume and result_path.exists():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            existing["mechanism"] = mechanism
            rows.append(existing)
            pd.DataFrame(rows).to_csv(summary_path, index=False)
            print(f"[{mechanism}] skipped existing {used_frequency:g} Hz", flush=True)
            continue

        if max_new_results is not None and new_results >= max_new_results:
            print(f"[{mechanism}] reached --max-new-results={max_new_results}; stopping sweep chunk", flush=True)
            break

        water_sigma, solid_sigma = phase_conductivities(spectrum_row, used_frequency)
        original_solid_sigma = solid_sigma
        solid_regularization_applied = False
        if zero_solid_regularization_s_m > 0.0 and abs(solid_sigma) == 0.0:
            solid_sigma = complex(zero_solid_regularization_s_m, 0.0)
            solid_regularization_applied = True
        conductance_values = cp.asarray(face_type_conductance_values(water_sigma, solid_sigma, dtype=dtype), dtype=dtype)
        face_data = GPUFaceTypeConductivity(
            face_types=gpu_face_types,
            conductance_values=conductance_values,
            shape=labels.shape,
            dtype=dtype,
        )
        residual_rows: list[dict[str, float | int]] = []
        start_time = time.perf_counter()
        memory_before = gpu_memory_info()

        def report_progress(iterations: int) -> None:
            if iteration_sleep_s > 0:
                time.sleep(iteration_sleep_s)
            if progress_every > 0 and iterations % progress_every == 0:
                elapsed = time.perf_counter() - start_time
                print(f"[{mechanism}] {used_frequency:g}Hz iterations={iterations} elapsed_s={elapsed:.1f}", flush=True)

        def report_residual(iterations: int, residual_norm: float) -> None:
            elapsed = time.perf_counter() - start_time
            residual_rows.append({"iteration": iterations, "elapsed_s": elapsed, "relative_residual_norm": residual_norm})
            if residual_every > 0:
                pd.DataFrame(residual_rows).to_csv(residual_history_path, index=False)

        if solver_backend == "gpu-bicgstab-compact":
            result = solve_ac3d_bicgstab_compact_gpu_face_types(
                face_data,
                direction=direction,
                voxel_size_m=voxel_size_m,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
                return_potential=False,
                iteration_callback=report_progress,
                residual_every=residual_every,
                residual_callback=report_residual if residual_every > 0 else None,
            )
        elif solver_backend == "gpu-cocg-lowmem":
            result = solve_ac3d_cocg_gpu_face_types(
                face_data,
                direction=direction,
                voxel_size_m=voxel_size_m,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
                return_potential=False,
                iteration_callback=report_progress,
                residual_every=residual_every,
                residual_callback=report_residual if residual_every > 0 else None,
            )
        elif solver_backend == "gpu-rb-sor-lowmem":
            result = solve_ac3d_red_black_sor_gpu_face_types(
                face_data,
                direction=direction,
                voxel_size_m=voxel_size_m,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
                omega=sor_omega,
                return_potential=False,
                iteration_callback=report_progress,
                residual_every=residual_every,
                residual_callback=report_residual if residual_every > 0 else None,
            )
        elif solver_backend == "gpu-jacobi-lowmem":
            result = solve_ac3d_weighted_jacobi_gpu_face_types(
                face_data,
                direction=direction,
                voxel_size_m=voxel_size_m,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
                omega=jacobi_omega,
                return_potential=False,
                iteration_callback=report_progress,
                residual_every=residual_every,
                residual_callback=report_residual if residual_every > 0 else None,
            )
        else:
            result = solve_ac3d_matrix_free_gpu_face_types(
                face_data,
                direction=direction,
                voxel_size_m=voxel_size_m,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
                use_jacobi=preconditioner == "jacobi",
                preconditioner=preconditioner,
                fft_reference=fft_reference,
                return_potential=True,
                iteration_callback=report_progress,
                x0=previous_solution,
                residual_every=residual_every,
                residual_callback=report_residual if residual_every > 0 else None,
            )
        synchronize_gpu()
        elapsed_total = time.perf_counter() - start_time
        memory_after = gpu_memory_info()
        if (
            solver_backend not in {"gpu-jacobi-lowmem", "gpu-rb-sor-lowmem", "gpu-cocg-lowmem", "gpu-bicgstab-compact"}
            and result.potential is not None
            and can_reuse_solution_as_warm_start(info=result.info, residual_norm=result.residual_norm, rtol=rtol)
        ):
            previous_solution = np.asarray(result.potential, dtype=dtype)
        else:
            previous_solution = None
        omega = 2.0 * np.pi * used_frequency
        output = {
            "mechanism": mechanism,
            "mechanism_label": MECHANISM_LABELS[mechanism],
            "solver": result.solver,
            "requested_frequency_hz": requested_frequency,
            "frequency_hz": used_frequency,
            "direction": direction,
            "shape": list(labels.shape),
            "n_cells": int(np.prod(labels.shape)),
            "voxel_size_m": voxel_size_m,
            "dtype": dtype_name,
            "water_sigma_real_s_m": float(water_sigma.real),
            "water_sigma_imag_s_m": float(water_sigma.imag),
            "solid_sigma_real_s_m": float(solid_sigma.real),
            "solid_sigma_imag_s_m": float(solid_sigma.imag),
            "original_solid_sigma_real_s_m": float(original_solid_sigma.real),
            "original_solid_sigma_imag_s_m": float(original_solid_sigma.imag),
            "zero_solid_regularization_s_m": float(zero_solid_regularization_s_m),
            "solid_regularization_applied": solid_regularization_applied,
            "effective_sigma_real_s_m": float(result.effective_conductivity_s_m.real),
            "effective_sigma_imag_s_m": float(result.effective_conductivity_s_m.imag),
            "real_relative_permittivity": float(result.effective_conductivity_s_m.imag / (omega * 8.8541878128e-12)),
            "mean_current_real_a_m2": float(result.mean_current_density_a_m2.real),
            "mean_current_imag_a_m2": float(result.mean_current_density_a_m2.imag),
            "relative_residual_norm": float(result.residual_norm),
            "iterations": int(result.iterations),
            "info": int(result.info),
            "rtol": rtol,
            "atol": atol,
            "maxiter": maxiter,
            "preconditioner": preconditioner,
            "fft_reference": fft_reference,
            "used_warm_start": solver_backend not in {"gpu-jacobi-lowmem", "gpu-rb-sor-lowmem", "gpu-cocg-lowmem", "gpu-bicgstab-compact"} and index > 0,
            "elapsed_total_s": elapsed_total,
            "cuda_total_gib": memory_after["cuda_total_bytes"] / 1024**3,
            "cuda_free_before_gib": memory_before["cuda_free_bytes"] / 1024**3,
            "cuda_free_after_gib": memory_after["cuda_free_bytes"] / 1024**3,
            "residual_history_path": str(residual_history_path) if residual_rows else None,
            "result_path": str(result_path),
        }
        strict_converged = can_reuse_solution_as_warm_start(
            info=result.info,
            residual_norm=result.residual_norm,
            rtol=rtol,
        )
        accepted_nonconverged_result = (
            accept_residual_le is not None
            and int(result.info) == int(maxiter)
            and float(result.residual_norm) <= float(accept_residual_le)
        )
        output["converged_strict"] = strict_converged
        output["accepted_nonconverged_result"] = accepted_nonconverged_result
        output["accept_residual_le"] = accept_residual_le
        output["converged"] = strict_converged or accepted_nonconverged_result
        if not output["converged"]:
            failed_path = frequency_dir / "failed_result.json"
            output["result_path"] = str(failed_path)
            write_json(failed_path, output)
            raise RuntimeError(
                f"{mechanism} {used_frequency:g} Hz did not converge: "
                f"info={result.info} residual={result.residual_norm:.3e} rtol={rtol:.3e}. "
                f"Failed solve metadata written to {failed_path}."
            )
        write_json(result_path, output)
        rows.append(output)
        new_results += 1
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(
            f"[{mechanism}] {used_frequency:g} Hz "
            f"sigma=({output['effective_sigma_real_s_m']:.6e}, {output['effective_sigma_imag_s_m']:.6e}) "
            f"res={result.residual_norm:.3e} iter={result.iterations}",
            flush=True,
        )

    return pd.DataFrame(rows).sort_values("frequency_hz").reset_index(drop=True)


def make_pnm_formation_factor_sweep(
    *,
    mechanism: str,
    spectra_path: Path,
    frequencies_hz: list[float],
    formation_factor: float,
    direction: str,
    out_dir: Path,
) -> pd.DataFrame:
    """Upscale apparent water spectra with a measured formation factor.

    This is a low-memory mechanism-resolved fallback for full-resolution PNM
    inputs. It does not replace an AC3D field solve; metadata records the
    solver as ``pnm_formation_factor`` so the two routes stay distinct.
    """

    if formation_factor <= 0:
        raise ValueError("formation_factor must be positive")
    spectra = pd.read_csv(spectra_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    summary_path = out_dir / "sweep_results.csv"
    for index, requested_frequency in enumerate(frequencies_hz):
        spectrum_row = nearest_spectrum_row(spectra, requested_frequency)
        used_frequency = float(spectrum_row["frequency_hz"])
        water_sigma, solid_sigma = phase_conductivities(spectrum_row, used_frequency)
        effective = water_sigma / formation_factor
        omega = 2.0 * np.pi * used_frequency
        output = {
            "mechanism": mechanism,
            "mechanism_label": MECHANISM_LABELS[mechanism],
            "solver": "pnm_formation_factor",
            "requested_frequency_hz": requested_frequency,
            "frequency_hz": used_frequency,
            "direction": direction,
            "formation_factor": formation_factor,
            "water_sigma_real_s_m": float(water_sigma.real),
            "water_sigma_imag_s_m": float(water_sigma.imag),
            "solid_sigma_real_s_m": float(solid_sigma.real),
            "solid_sigma_imag_s_m": float(solid_sigma.imag),
            "effective_sigma_real_s_m": float(effective.real),
            "effective_sigma_imag_s_m": float(effective.imag),
            "real_relative_permittivity": float(effective.imag / (omega * 8.8541878128e-12)),
            "mean_current_real_a_m2": float(effective.real),
            "mean_current_imag_a_m2": float(effective.imag),
            "relative_residual_norm": np.nan,
            "iterations": 0,
            "info": 0,
            "rtol": np.nan,
            "atol": np.nan,
            "maxiter": 0,
            "preconditioner": "not_applicable",
            "fft_reference": "not_applicable",
            "used_warm_start": False,
            "elapsed_total_s": 0.0,
            "cuda_total_gib": np.nan,
            "cuda_free_before_gib": np.nan,
            "cuda_free_after_gib": np.nan,
            "residual_history_path": None,
            "result_path": str(out_dir / f"frequency_{index:03d}_{frequency_tag(used_frequency)}Hz" / "result.json"),
        }
        result_path = Path(output["result_path"])
        write_json(result_path, output)
        rows.append(output)
    frame = pd.DataFrame(rows).sort_values("frequency_hz").reset_index(drop=True)
    frame.to_csv(summary_path, index=False)
    return frame


def scale_experiment_to_simulation(experiment: pd.DataFrame, simulation: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    exp = experiment.sort_values("frequency_hz").reset_index(drop=True).copy()
    all_sim = simulation[simulation["mechanism"] == "all"].sort_values("frequency_hz").reset_index(drop=True)
    if all_sim.empty:
        raise ValueError("simulation must contain mechanism == 'all' rows")
    anchor_frequency = float(exp.iloc[0]["frequency_hz"])
    idx = int(np.abs(np.log(all_sim["frequency_hz"].to_numpy(dtype=float)) - np.log(anchor_frequency)).argmin())
    sim_anchor = float(all_sim.iloc[idx]["effective_sigma_real_s_m"])
    exp_anchor = float(exp.iloc[0]["experiment_real_raw"])
    if exp_anchor == 0:
        raise ValueError("lowest-frequency experimental real response is zero")
    scale = sim_anchor / exp_anchor
    exp["experiment_real_scaled_s_m"] = exp["experiment_real_raw"] * scale
    exp["experiment_imag_scaled_s_m"] = exp["experiment_imag_raw"] * scale
    return exp, float(scale)


def score_trend(simulation: pd.DataFrame, experiment: pd.DataFrame, mechanism: str = "all") -> dict[str, float | int]:
    sim = simulation[simulation["mechanism"] == mechanism].sort_values("frequency_hz").reset_index(drop=True)
    exp = experiment.sort_values("frequency_hz").reset_index(drop=True)
    if sim.empty:
        raise ValueError(f"simulation has no mechanism == {mechanism!r} rows")
    if len(sim) != len(exp):
        raise ValueError(f"simulation and experiment frequency counts differ: {len(sim)} vs {len(exp)}")
    sim_freq = sim["frequency_hz"].to_numpy(dtype=float)
    exp_freq = exp["frequency_hz"].to_numpy(dtype=float)
    if not np.allclose(sim_freq, exp_freq, rtol=1.0e-8, atol=1.0e-12):
        raise ValueError("simulation and experiment frequencies do not match")

    sim_real = sim["effective_sigma_real_s_m"].to_numpy(dtype=float)
    sim_imag = sim["effective_sigma_imag_s_m"].to_numpy(dtype=float)
    exp_real = exp["experiment_sigma_real_s_m"].to_numpy(dtype=float)
    exp_imag = exp["experiment_sigma_imag_s_m"].to_numpy(dtype=float)

    sim_peak_idx = int(np.nanargmax(sim_imag))
    exp_peak_idx = int(np.nanargmax(exp_imag))
    real_increases = bool(sim_real[-1] > sim_real[0])
    imag_has_internal_peak = bool(0 < sim_peak_idx < len(sim_imag) - 1 and sim_imag[sim_peak_idx] > sim_imag[0] and sim_imag[sim_peak_idx] > sim_imag[-1])
    imag_peak_drop_ratio = (sim_imag[sim_peak_idx] - sim_imag[-1]) / max(abs(sim_imag[sim_peak_idx]), 1.0e-30)
    imag_peak_rise_ratio = (sim_imag[sim_peak_idx] - sim_imag[0]) / max(abs(sim_imag[sim_peak_idx]), 1.0e-30)
    real_log_rmse = float(np.sqrt(np.mean((np.log10(sim_real) - np.log10(exp_real)) ** 2)))
    imag_rmse = float(np.sqrt(np.mean((sim_imag - exp_imag) ** 2)))
    peak_log_frequency_error = abs(np.log10(sim_freq[sim_peak_idx]) - np.log10(exp_freq[exp_peak_idx]))

    return {
        "real_increases": int(real_increases),
        "imag_has_internal_peak": int(imag_has_internal_peak),
        "trend_pass": int(real_increases and imag_has_internal_peak),
        "real_gain_ratio": float(sim_real[-1] / sim_real[0] - 1.0),
        "experiment_real_gain_ratio": float(exp_real[-1] / exp_real[0] - 1.0),
        "imag_peak_hz": float(sim_freq[sim_peak_idx]),
        "experiment_imag_peak_hz": float(exp_freq[exp_peak_idx]),
        "imag_peak_value_s_m": float(sim_imag[sim_peak_idx]),
        "imag_first_value_s_m": float(sim_imag[0]),
        "imag_last_value_s_m": float(sim_imag[-1]),
        "imag_peak_rise_ratio": float(imag_peak_rise_ratio),
        "imag_peak_drop_ratio": float(imag_peak_drop_ratio),
        "real_log_rmse": real_log_rmse,
        "imag_rmse_s_m": imag_rmse,
        "peak_log_frequency_error": float(peak_log_frequency_error),
    }


def score_sample89_trend(simulation: pd.DataFrame, experiment: pd.DataFrame) -> dict[str, float | int]:
    """Backward-compatible sample 89 trend score helper used by tests."""
    return score_trend(simulation, experiment, mechanism="all")


def experiment_plot_columns(experiment: pd.DataFrame, experiment_label: str) -> tuple[str, str, str]:
    if {"experiment_sigma_real_s_m", "experiment_sigma_imag_s_m"}.issubset(experiment.columns):
        return "experiment_sigma_real_s_m", "experiment_sigma_imag_s_m", experiment_label
    return "experiment_real_scaled_s_m", "experiment_imag_scaled_s_m", f"{experiment_label} scaled"


def positive_log10_series(frame: pd.DataFrame, x_col: str, y_col: str) -> tuple[pd.Series, np.ndarray]:
    values = frame[y_col].to_numpy(dtype=float)
    mask = np.isfinite(values) & (values > 0.0)
    return frame.loc[mask, x_col], np.log10(values[mask])


def positive_series(frame: pd.DataFrame, x_col: str, y_col: str) -> tuple[pd.Series, pd.Series]:
    values = frame[y_col].to_numpy(dtype=float)
    mask = np.isfinite(values) & (values > 0.0)
    return frame.loc[mask, x_col], frame.loc[mask, y_col]


def widen_log10_ylim(ax: plt.Axes, values: list[np.ndarray], *, min_span: float = 0.75, pad_fraction: float = 0.12) -> None:
    finite = np.concatenate([np.asarray(item, dtype=float)[np.isfinite(item)] for item in values if len(item)])
    if finite.size == 0:
        return
    low = float(finite.min())
    high = float(finite.max())
    span = high - low
    if span < min_span:
        center = 0.5 * (low + high)
        low = center - 0.5 * min_span
        high = center + 0.5 * min_span
        span = min_span
    pad = max(span * pad_fraction, 0.02)
    ax.set_ylim(low - pad, high + pad)


def plot_comparison(experiment: pd.DataFrame, simulation: pd.DataFrame, output_png: Path, *, sample_label: str, experiment_label: str) -> None:
    exp_real_col, exp_imag_col, exp_label = experiment_plot_columns(experiment, experiment_label)
    experiment_mask = experiment[exp_imag_col].to_numpy(dtype=float) > 0.0
    experiment_plot = experiment.loc[experiment_mask].copy()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), constrained_layout=True)

    exp_real_x, exp_real_y = positive_series(experiment_plot, "frequency_hz", exp_real_col)
    exp_imag_x, exp_imag_y = positive_series(experiment_plot, "frequency_hz", exp_imag_col)
    for mechanism in ("all", "maxwell", "pore", "membrane"):
        frame = simulation[simulation["mechanism"] == mechanism].sort_values("frequency_hz")
        if frame.empty:
            continue
        label = MECHANISM_LABELS[mechanism]
        color = MECHANISM_COLORS[mechanism]
        linewidth = {"all": 2.4, "membrane": 2.0}.get(mechanism, 1.25)
        alpha = 0.72 if mechanism == "all" else 0.98
        zorder = 2 if mechanism == "all" else 4
        linestyle = "--" if mechanism == "membrane" else "-"
        sim_real_x, sim_real_y = positive_series(frame, "frequency_hz", "effective_sigma_real_s_m")
        sim_imag_x, sim_imag_y = positive_series(frame, "frequency_hz", "effective_sigma_imag_s_m")
        axes[0].loglog(
            sim_real_x,
            sim_real_y,
            linestyle,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
            label=label,
        )
        axes[1].loglog(
            sim_imag_x,
            sim_imag_y,
            linestyle,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
            label=label,
        )
    axes[0].loglog(
        exp_real_x,
        exp_real_y,
        "o",
        color="#111111",
        markersize=3.2,
        zorder=5,
        label=f"{exp_label}, imag > 0",
    )
    axes[1].loglog(
        exp_imag_x,
        exp_imag_y,
        "o",
        color="#111111",
        markersize=3.2,
        zorder=5,
        label=f"{exp_label}, imag > 0",
    )

    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel(r"$\sigma'$ (S/m)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel(r"$\sigma''$ (S/m)")
    for ax in axes:
        ax.set_xlim(1.0e-4, 1.0e5)
    axes[0].set_ylim(1.0e-4, 1.0e-1)
    axes[1].set_ylim(1.0e-7, 1.0e-3)
    for label, ax in zip(["(a)", "(b)"], axes):
        ax.grid(True, which="both", alpha=0.18, linewidth=0.55)
        handles, labels = ax.get_legend_handles_labels()
        order = [labels.index(item) for item in [f"{exp_label}, imag > 0", "Maxwell", "Pore", "Membrane", "Total"] if item in labels]
        ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=7, frameon=False)
        ax.text(-0.13, 1.04, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")
    fig.suptitle(f"{sample_label}", fontsize=10, y=1.04)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=240)
    plt.close(fig)


def plot_raw_experiment(experiment: pd.DataFrame, output_png: Path, *, experiment_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    axes[0].semilogx(experiment["frequency_hz"], experiment["magnitude_ratio"], "o-", color="#375a7f")
    axes[1].semilogx(experiment["frequency_hz"], experiment["phase_shift_rad"], "o-", color="#8a4f7d")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Magnitude (ratio)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase shift (rad)")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
    fig.suptitle(f"{experiment_label} raw PSIP response")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220)
    plt.close(fig)


def mechanism_meaning_for_mode(field_solve_mode: str) -> dict[str, str]:
    if field_solve_mode == "pnm-formation-factor":
        return {
            "maxwell": "PNM formation-factor dielectric proxy for the interfacial/Maxwell-only spectrum. It upscales the apparent water high-frequency dielectric term by the measured formation factor and does not solve the heterogeneous solid-water AC3D field.",
            "pore": "Niu paper-mode pore-only spectrum: water phase is sigma_w + Delta sigma_pore*, solid phase is zero; the resulting apparent water spectrum is upscaled by the measured formation factor. This is not Pore minus Maxwell.",
            "membrane": "Niu paper-mode membrane-only spectrum: water phase is sigma_w + Delta sigma_membrane*, solid phase is zero; the resulting apparent water spectrum is upscaled by the measured formation factor. This is not Membrane minus Maxwell.",
            "all": "Niu paper-mode combined spectrum: pore and membrane perturbations are combined with high-frequency dielectric terms and upscaled by the measured formation factor. This is a quick proxy, not a full AC3D field solve.",
        }
    return {
        "maxwell": "AC3D two-phase water/solid contrast with water/solid high-frequency permittivity and no pore or membrane increment.",
        "pore": "Niu paper-mode pore-only AC3D field solve: water phase is sigma_w + Delta sigma_pore*, solid phase is zero. This is not Pore minus Maxwell.",
        "membrane": "Niu paper-mode membrane-only AC3D field solve: water phase is sigma_w + Delta sigma_membrane*, solid phase is zero. This is not Membrane minus Maxwell.",
        "all": "Combined pore plus membrane increment with water/solid high-frequency permittivity in the AC3D field solve.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mhd",
        default=str(ROOT / "results" / "pnextract_inputs" / "sample_89_Grainstone_89seged" / "sample_89_Grainstone_89seged_pnextract.mhd"),
    )
    parser.add_argument(
        "--network-dir",
        default=str(ROOT / "results" / "pnextract" / "sample_89_Grainstone_89seged" / "network_parsed"),
    )
    parser.add_argument(
        "--experiment-csv",
        default=str(ROOT / "data_inventory" / "ct_backed_samples_raw_copy_20260605" / "sample_89_Grainstone" / "SIP" / "LKC-89.csv"),
    )
    parser.add_argument(
        "--segmented-volume",
        default=str(ROOT / "data_inventory" / "ct_backed_samples_raw_copy_20260605" / "sample_89_Grainstone" / "CT_slices" / "89seged.tiff"),
    )
    parser.add_argument("--components", nargs="+", choices=list(MECHANISM_TO_COMPONENT), default=["maxwell", "pore", "membrane", "all"])
    parser.add_argument("--sample-label", default="Sample 89 Grainstone")
    parser.add_argument("--experiment-label", default="LKC-89 experiment")
    parser.add_argument("--output-prefix", default="sample89_3d_sip")
    parser.add_argument(
        "--field-solve-mode",
        choices=["gpu-ac3d", "gpu-jacobi-lowmem", "gpu-rb-sor-lowmem", "gpu-cocg-lowmem", "gpu-bicgstab-compact", "pnm-formation-factor"],
        default="gpu-ac3d",
        help="gpu-ac3d runs the full field solve; pnm-formation-factor only upscales full-resolution PNM spectra by measured F.",
    )
    parser.add_argument("--formation-factor", type=float, default=None)
    parser.add_argument("--jacobi-omega", type=float, default=0.65)
    parser.add_argument("--sor-omega", type=float, default=1.35)
    parser.add_argument("--component-mode", choices=["paper", "legacy"], default="paper")
    parser.add_argument("--experiment-mode", choices=["impedance", "scaled-raw"], default="impedance")
    parser.add_argument("--current-resistor-ohm", type=float, default=10000.0)
    parser.add_argument("--sample-length-cm", type=float, default=5.01)
    parser.add_argument("--sample-diameter-cm", type=float, default=2.54)
    parser.add_argument("--water-conductivity-s-m", type=float, default=0.12)
    parser.add_argument("--solid-conductivity-s-m", type=float, default=0.0024)
    parser.add_argument("--surface-conductance-s", type=float, default=1.3e-9)
    parser.add_argument("--membrane-polarizability", type=float, default=0.01)
    parser.add_argument("--solid-relative-permittivity", type=float, default=7.0)
    parser.add_argument("--water-relative-permittivity", type=float, default=80.0)
    parser.add_argument("--dynamic-pore-size-m", type=float, default=2.7e-6)
    parser.add_argument("--diffusion-coefficient-m2-s", type=float, default=1.3e-9)
    parser.add_argument("--pore-radius-scale", type=float, default=1.0)
    parser.add_argument("--membrane-length-scale", type=float, default=1.0)
    parser.add_argument("--membrane-zdc-scale", type=float, default=1.0)
    parser.add_argument(
        "--solid-background-components",
        nargs="+",
        choices=list(MECHANISM_TO_COMPONENT),
        default=["maxwell", "all"],
        help="Mechanism spectra that should include the solid-phase effective real conductivity.",
    )
    parser.add_argument("--frequencies", nargs="+", type=float)
    parser.add_argument("--direction", choices=["x", "y", "z"], default="x")
    parser.add_argument("--pore-label", type=int, default=0)
    parser.add_argument("--solid-label", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument(
        "--accept-residual-le",
        type=float,
        default=None,
        help=(
            "Optional practical acceptance threshold for relative residual norm when a solve stops exactly at "
            "--maxiter. This does not mark the solve as strictly converged; metadata records the relaxed acceptance."
        ),
    )
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64")
    parser.add_argument("--preconditioner", choices=["none", "jacobi", "fft"], default="fft")
    parser.add_argument("--fft-reference", choices=["mean-face", "mean-abs", "pore"], default="mean-face")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--residual-every", type=int, default=0)
    parser.add_argument(
        "--iteration-sleep-s",
        type=float,
        default=0.0,
        help="Sleep this many seconds after each solver iteration to reduce average GPU duty cycle.",
    )
    parser.add_argument(
        "--zero-solid-regularization-s-m",
        type=float,
        default=0.0,
        help=(
            "Optional numerical floor applied only when a component spectrum has exactly zero solid conductivity. "
            "This removes null modes in pore/membrane single-mechanism solves; result metadata records the original zero value."
        ),
    )
    parser.add_argument(
        "--max-new-results",
        type=int,
        default=None,
        help="Stop each requested mechanism after this many newly computed frequencies. Existing resumed frequencies do not count.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "sample89_3d_sip_lkc89_v1"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not cupy_available():
        raise SystemExit("CuPy/CUDA is not available in this Python environment")

    out_dir = Path(args.out_dir)
    spectra_dir = out_dir / "spectra"
    sweeps_dir = out_dir / "sweeps"
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    experiment = parse_psip_csv(Path(args.experiment_csv))
    if args.frequencies:
        requested = np.asarray(args.frequencies, dtype=float)
        keep_rows = []
        for frequency in requested:
            idx = int(np.abs(np.log(experiment["frequency_hz"].to_numpy(dtype=float)) - np.log(frequency)).argmin())
            keep_rows.append(experiment.iloc[idx])
        experiment = pd.DataFrame(keep_rows).drop_duplicates(subset=["frequency_hz"]).sort_values("frequency_hz").reset_index(drop=True)
    frequencies = experiment["frequency_hz"].to_numpy(dtype=float)
    experiment_absolute = add_bulk_conductivity_from_psip_impedance(
        experiment,
        current_resistor_ohm=args.current_resistor_ohm,
        sample_length_cm=args.sample_length_cm,
        sample_diameter_cm=args.sample_diameter_cm,
    )
    parsed_experiment_path = out_dir / f"{args.output_prefix}_experiment_parsed.csv"
    experiment_bulk_path = out_dir / f"{args.output_prefix}_experiment_bulk_conductivity_from_impedance.csv"
    experiment.to_csv(parsed_experiment_path, index=False)
    experiment_absolute.to_csv(experiment_bulk_path, index=False)
    plot_raw_experiment(experiment_absolute, figures_dir / f"{args.output_prefix}_raw_psip_response.png", experiment_label=args.experiment_label)

    mhd_info = parse_mhd(Path(args.mhd))
    labels = load_label_volume(mhd_info)
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = {str(int(value)): int(count) for value, count in zip(unique, counts)}

    params = build_polarization_parameters(
        water_conductivity_s_m=args.water_conductivity_s_m,
        surface_conductance_s=args.surface_conductance_s,
        membrane_polarizability=args.membrane_polarizability,
        solid_relative_permittivity=args.solid_relative_permittivity,
        dynamic_pore_size_m=args.dynamic_pore_size_m,
        diffusion_coefficient_m2_s=args.diffusion_coefficient_m2_s,
        water_relative_permittivity=args.water_relative_permittivity,
    )

    _, component_paths, spectra_metadata = make_component_spectra_files(
        output_prefix=args.output_prefix,
        frequencies_hz=frequencies,
        network_dir=Path(args.network_dir),
        out_dir=spectra_dir,
        mode=args.component_mode,
        params=params,
        solid_conductivity_s_m=args.solid_conductivity_s_m,
        solid_background_components=set(args.solid_background_components),
        pore_radius_scale=args.pore_radius_scale,
        membrane_length_scale=args.membrane_length_scale,
        membrane_zdc_scale=args.membrane_zdc_scale,
    )

    sweep_frames: list[pd.DataFrame] = []
    if args.field_solve_mode == "pnm-formation-factor":
        if args.formation_factor is None:
            raise SystemExit("--formation-factor is required when --field-solve-mode pnm-formation-factor")
        for mechanism in args.components:
            frame = make_pnm_formation_factor_sweep(
                mechanism=mechanism,
                spectra_path=component_paths[mechanism],
                frequencies_hz=frequencies.tolist(),
                formation_factor=args.formation_factor,
                direction=args.direction,
                out_dir=sweeps_dir / mechanism,
            )
            sweep_frames.append(frame)
    else:
        for mechanism in args.components:
            frame = run_component_sweep(
                labels=labels,
                mechanism=mechanism,
                spectra_path=component_paths[mechanism],
                frequencies_hz=frequencies.tolist(),
                voxel_size_m=float(mhd_info["voxel_size_m"]),
                out_dir=sweeps_dir / mechanism,
                pore_label=args.pore_label,
                solid_label=args.solid_label,
                direction=args.direction,
                rtol=args.rtol,
                atol=args.atol,
                maxiter=args.maxiter,
                dtype_name=args.dtype,
                preconditioner=args.preconditioner,
                fft_reference=args.fft_reference,
                progress_every=args.progress_every,
                residual_every=args.residual_every,
                iteration_sleep_s=args.iteration_sleep_s,
                zero_solid_regularization_s_m=args.zero_solid_regularization_s_m,
                resume=args.resume,
                solver_backend=args.field_solve_mode,
                jacobi_omega=args.jacobi_omega,
                sor_omega=args.sor_omega,
                max_new_results=args.max_new_results,
                accept_residual_le=args.accept_residual_le,
            )
            sweep_frames.append(frame)

    simulation = pd.concat(sweep_frames, ignore_index=True).sort_values(["mechanism", "frequency_hz"]).reset_index(drop=True)
    simulation_path = out_dir / f"{args.output_prefix}_mechanism_sweeps.csv"
    simulation.to_csv(simulation_path, index=False)
    scaled_experiment_path = out_dir / f"{args.output_prefix}_experiment_scaled_to_total_low_frequency.csv"
    if "all" in set(simulation["mechanism"]):
        scaled_experiment, experiment_scale = scale_experiment_to_simulation(experiment, simulation)
        scaled_experiment.to_csv(scaled_experiment_path, index=False)
        all_sim = simulation[simulation["mechanism"] == "all"].sort_values("frequency_hz").reset_index(drop=True)
        if len(all_sim) == len(experiment_absolute):
            trend_score = score_trend(simulation, experiment_absolute)
        else:
            trend_score = {
                "trend_pass": 0,
                "reason": (
                    "trend score skipped for partial mechanism chunk: "
                    f"all_count={len(all_sim)} experiment_count={len(experiment_absolute)}"
                ),
            }
    else:
        scaled_experiment = experiment.copy()
        experiment_scale = float("nan")
        trend_score = {
            "trend_pass": 0,
            "reason": "trend score requires mechanism == 'all'",
        }
    comparison_experiment = experiment_absolute if args.experiment_mode == "impedance" or "all" not in set(simulation["mechanism"]) else scaled_experiment
    comparison_png = figures_dir / f"{args.output_prefix}_real_imag_comparison.png"
    plot_comparison(comparison_experiment, simulation, comparison_png, sample_label=args.sample_label, experiment_label=args.experiment_label)

    metadata = {
        "description": f"{args.sample_label} 3-D AC SIP comparison against {args.experiment_label} PSIP data.",
        "sample_label": args.sample_label,
        "experiment_label": args.experiment_label,
        "output_prefix": args.output_prefix,
        "output_dir": str(out_dir),
        "segmented_volume_full_resolution": str(Path(args.segmented_volume)),
        "field_solve_volume_mhd": str(Path(args.mhd)),
        "field_solve_volume_raw": str(mhd_info["raw_path"]),
        "mhd_info": {
            "shape_zyx": list(mhd_info["shape_zyx"]),
            "dims_xyz": list(mhd_info["dims_xyz"]),
            "element_size_um_xyz": list(mhd_info["element_size_um_xyz"]),
            "voxel_size_m": mhd_info["voxel_size_m"],
            "dtype": str(mhd_info["dtype"]),
            "label_counts": label_counts,
        },
        "network_dir": str(Path(args.network_dir)),
        "experiment_csv": str(Path(args.experiment_csv)),
        "experiment_frequency_range_hz": [float(frequencies.min()), float(frequencies.max())],
        "n_experiment_frequencies": int(len(frequencies)),
        "components": args.components,
        "component_meaning": mechanism_meaning_for_mode(args.field_solve_mode),
        "experiment_scaling": {
            "reason": "Retained for comparison; impedance mode uses Current Resistor and core geometry to estimate absolute bulk conductivity.",
            "scale_to_s_m": experiment_scale,
            "anchor": "lowest-frequency experimental real response matched to the all-mechanism simulated real conductivity.",
            "phase_convention": "experiment_imag_scaled_s_m preserves Magnitude * sin(Phase_Shift) sign from the CSV.",
        },
        "experiment_impedance_conversion": {
            "enabled_for_plot": args.experiment_mode == "impedance",
            "current_resistor_ohm": args.current_resistor_ohm,
            "sample_length_cm": args.sample_length_cm,
            "sample_diameter_cm": args.sample_diameter_cm,
            "formula": "sigma* = (1 / (current_resistor_ohm * Magnitude[ratio] * exp(i phase))) * length_m / area_m2",
            "output_csv": str(experiment_bulk_path),
        },
        "parameter_overrides": {
            "water_conductivity_s_m": args.water_conductivity_s_m,
            "solid_conductivity_s_m": args.solid_conductivity_s_m,
            "surface_conductance_s": args.surface_conductance_s,
            "membrane_polarizability": args.membrane_polarizability,
            "solid_relative_permittivity": args.solid_relative_permittivity,
            "water_relative_permittivity": args.water_relative_permittivity,
            "dynamic_pore_size_m": args.dynamic_pore_size_m,
            "diffusion_coefficient_m2_s": args.diffusion_coefficient_m2_s,
            "pore_radius_scale": args.pore_radius_scale,
            "membrane_length_scale": args.membrane_length_scale,
            "membrane_zdc_scale": args.membrane_zdc_scale,
            "zero_solid_regularization_s_m": args.zero_solid_regularization_s_m,
            "solid_background_components": args.solid_background_components,
        },
        "trend_score": trend_score,
        "solver": {
            "field_solve_mode": args.field_solve_mode,
            "formation_factor": args.formation_factor,
            "jacobi_omega": args.jacobi_omega,
            "sor_omega": args.sor_omega,
            "direction": args.direction,
            "rtol": args.rtol,
            "atol": args.atol,
            "maxiter": args.maxiter,
            "accept_residual_le": args.accept_residual_le,
            "dtype": args.dtype,
            "preconditioner": args.preconditioner,
            "fft_reference": args.fft_reference,
            "zero_solid_regularization_s_m": args.zero_solid_regularization_s_m,
            "max_new_results": args.max_new_results,
        },
        "spectra_metadata": spectra_metadata,
        "outputs": {
            "parsed_experiment_csv": str(parsed_experiment_path),
            "experiment_bulk_conductivity_csv": str(experiment_bulk_path),
            "scaled_experiment_csv": str(scaled_experiment_path),
            "simulation_csv": str(simulation_path),
            "comparison_png": str(comparison_png),
            "raw_experiment_png": str(figures_dir / f"{args.output_prefix}_raw_psip_response.png"),
        },
    }
    write_json(out_dir / "run_metadata.json", metadata)
    print(json.dumps(metadata["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
