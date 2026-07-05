#!/usr/bin/env python3
"""Compute project-extracted field-weighted dynamic pore size for AC3D volumes."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pore_scale_electrical.ac3d_active_domain import periodic_active_component_anchor_indices  # noqa: E402
from pore_scale_electrical.ac3d_solver import (  # noqa: E402
    AC3DIterativeResult,
    face_conductivity_arrays_from_types,
    face_type_conductivity,
    jacobi_inverse_diagonal_faces,
    matrix_free_matvec_faces,
    matrix_free_rhs_faces,
    mean_current_density_faces,
)
from pore_scale_electrical.dynamic_pore_size import (  # noqa: E402
    compute_dynamic_pore_size_from_fields,
    fields_from_periodic_potential,
    physical_direction_to_array_axis,
)


def parse_int3(values: list[int] | None, name: str) -> tuple[int, int, int] | None:
    if values is None:
        return None
    if len(values) != 3:
        raise ValueError(f"{name} expects exactly three integers")
    parsed = tuple(int(v) for v in values)
    if any(v < 0 for v in parsed):
        raise ValueError(f"{name} must be non-negative")
    return parsed


def validate_run_mode(args: argparse.Namespace) -> None:
    if args.run_mode != "formal":
        return
    if args.crop_start is not None or args.crop_size is not None:
        raise ValueError("run-mode=formal requires uncropped full-volume input")
    if list(args.directions) != ["x", "y", "z"]:
        raise ValueError("run-mode=formal requires directions x y z in that order")
    if args.axis_order is None:
        raise ValueError("run-mode=formal requires explicit --axis-order")
    if args.pore_label is None or args.solid_label is None:
        raise ValueError("run-mode=formal requires explicit --pore-label and --solid-label")


def read_volume(
    raw_path: Path,
    dtype: str,
    shape: tuple[int, int, int],
    crop_start: tuple[int, int, int] | None,
    crop_size: tuple[int, int, int] | None,
) -> np.ndarray:
    volume = np.memmap(raw_path, dtype=np.dtype(dtype), mode="r", shape=shape, order="C")
    if crop_start is None and crop_size is None:
        return np.asarray(volume)
    if crop_start is None or crop_size is None:
        raise ValueError("--crop-start and --crop-size must be provided together")
    stop = tuple(s + n for s, n in zip(crop_start, crop_size, strict=True))
    if any(e > full for e, full in zip(stop, shape, strict=True)):
        raise ValueError(f"requested crop {crop_start}:{stop} exceeds shape {shape}")
    return np.asarray(volume[tuple(slice(s, e) for s, e in zip(crop_start, stop, strict=True))])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT.parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def _bicgstab(
    operator: spla.LinearOperator,
    rhs: np.ndarray,
    *,
    rtol: float,
    maxiter: int | None,
    preconditioner: spla.LinearOperator | None,
    callback: Any,
) -> tuple[np.ndarray, int]:
    kwargs: dict[str, Any] = {"maxiter": maxiter, "M": preconditioner, "callback": callback}
    if "rtol" in inspect.signature(spla.bicgstab).parameters:
        kwargs["rtol"] = rtol
        kwargs["atol"] = 0.0
    else:
        kwargs["tol"] = rtol
    return spla.bicgstab(operator, rhs, **kwargs)


def solve_cpu_active_domain(
    labels: np.ndarray,
    *,
    pore_label: int,
    solid_label: int,
    direction_axis: int,
    voxel_size_m: float,
    dtype: np.dtype,
    rtol: float,
    maxiter: int | None,
    use_jacobi: bool,
) -> AC3DIterativeResult:
    face_data = face_type_conductivity(
        labels,
        pore_label,
        solid_label,
        water_conductivity_s_m=1.0 + 0.0j,
        solid_conductivity_s_m=0.0 + 0.0j,
        dtype=dtype,
    )
    face_arrays = face_conductivity_arrays_from_types(face_data)
    diagonal = np.zeros(face_data.shape, dtype=dtype)
    for axis, g in enumerate(face_arrays):
        diagonal += np.asarray(g) + np.roll(np.asarray(g), 1, axis=axis)
    active_mask = np.abs(diagonal) > np.finfo(float).tiny
    identity_mask = ~active_mask.ravel(order="C")
    gauge_indices = np.asarray(periodic_active_component_anchor_indices(active_mask), dtype=np.int64)
    rhs = matrix_free_rhs_faces(face_arrays, direction_axis, 1.0, voxel_size_m)
    rhs[identity_mask] = 0.0
    if gauge_indices.size:
        rhs[gauge_indices] = 0.0
    rhs_norm = max(float(np.linalg.norm(rhs)), np.finfo(float).eps)

    def matvec(vector: np.ndarray) -> np.ndarray:
        out = matrix_free_matvec_faces(face_arrays, vector)
        out[identity_mask] = vector[identity_mask]
        if gauge_indices.size:
            out[gauge_indices] = vector[gauge_indices]
        return out

    operator = spla.LinearOperator((rhs.size, rhs.size), matvec=matvec, dtype=dtype)
    preconditioner = None
    if use_jacobi:
        inverse_diagonal = jacobi_inverse_diagonal_faces(face_arrays)
        inverse_diagonal[identity_mask] = 1.0
        if gauge_indices.size:
            inverse_diagonal[gauge_indices] = 1.0
        preconditioner = spla.LinearOperator((rhs.size, rhs.size), matvec=lambda vector: inverse_diagonal * vector, dtype=dtype)

    iterations = 0
    residual_history: list[tuple[int, float]] = []

    def callback(xk: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1
        if iterations % 10 == 0:
            residual_history.append((iterations, float(np.linalg.norm(matvec(xk) - rhs) / rhs_norm)))

    solution, info = _bicgstab(operator, rhs, rtol=rtol, maxiter=maxiter, preconditioner=preconditioner, callback=callback)
    residual_norm = float(np.linalg.norm(matvec(solution) - rhs) / rhs_norm)
    if not residual_history or residual_history[-1][0] != iterations:
        residual_history.append((iterations, residual_norm))
    potential = solution.reshape(face_data.shape)
    mean_current = mean_current_density_faces(face_arrays, potential, direction_axis, 1.0, voxel_size_m)
    return AC3DIterativeResult(
        direction=str(direction_axis),
        effective_conductivity_s_m=complex(mean_current),
        mean_current_density_a_m2=complex(mean_current),
        field_strength_v_m=1.0,
        residual_norm=residual_norm,
        potential=potential,
        solver="matrix_free_bicgstab_active_domain_jacobi" if use_jacobi else "matrix_free_bicgstab_active_domain",
        iterations=iterations,
        info=int(info),
        residual_history=tuple(residual_history),
    )


def solve_gpu_active_domain(
    labels: np.ndarray,
    *,
    pore_label: int,
    solid_label: int,
    direction_axis: int,
    voxel_size_m: float,
    solver_dtype: str,
    preconditioner: str,
    fft_reference: str,
    gauge_mode: str,
    rtol: float,
    maxiter: int | None,
) -> AC3DIterativeResult:
    from pore_scale_electrical.ac3d_gpu import (  # noqa: PLC0415
        face_type_conductivity_gpu,
        parse_gpu_complex_dtype,
        solve_ac3d_matrix_free_gpu_face_types,
        synchronize_gpu,
    )

    dtype = parse_gpu_complex_dtype(solver_dtype)
    face_data = face_type_conductivity_gpu(labels, pore_label, solid_label, 1.0 + 0.0j, 0.0 + 0.0j, dtype=dtype)
    result = solve_ac3d_matrix_free_gpu_face_types(
        face_data,
        direction=direction_axis,
        field_strength_v_m=1.0,
        voxel_size_m=voxel_size_m,
        rtol=rtol,
        maxiter=maxiter,
        use_jacobi=preconditioner == "jacobi",
        preconditioner=preconditioner,
        fft_reference=fft_reference,
        gauge_mode=gauge_mode,
        return_potential=True,
        residual_every=10,
    )
    synchronize_gpu()
    return result


def write_config(path: Path, args: argparse.Namespace) -> None:
    lines = [
        "# 动态孔径计算配置；正式结果包中用于人工复核参数来源。",
        f"raw: {args.raw}",
        f"shape: {list(args.shape)}",
        f"dtype: {args.dtype}",
        f"axis_order: {args.axis_order}",
        f"pore_label: {args.pore_label}",
        f"solid_label: {args.solid_label}",
        f"voxel_size_m: {args.voxel_size_m}",
        f"directions: {list(args.directions)}",
        f"backend: {args.backend}",
        f"preconditioner: {args.preconditioner}",
        f"fft_reference: {args.fft_reference}",
        f"gauge_mode: {args.gauge_mode}",
        f"solver_dtype: {args.solver_dtype}",
        f"rtol: {args.rtol}",
        f"maxiter: {args.maxiter}",
        f"run_mode: {args.run_mode}",
        f"crop_start: {args.crop_start}",
        f"crop_size: {args.crop_size}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_manifest(
    *,
    directional_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    physical_to_array_axis: dict[str, int],
    input_sha256: str,
    git_commit: str | None,
) -> dict[str, Any]:
    valid_rows = [row for row in directional_rows if row.get("direction_valid", True)]
    total_volume = float(sum(float(row["volume_integral_v2_m"]) for row in valid_rows))
    total_surface = float(sum(float(row["surface_integral_v2"]) for row in valid_rows))
    lambda_iso = float(2.0 * total_volume / total_surface) if total_surface > 0 else None
    total_pore_volume = float(sum(float(row["pore_volume_m3"]) for row in valid_rows))
    total_faces_area = float(
        sum(float(row["interface_face_count"]) * float(row["voxel_surface_area_m2"]) for row in valid_rows)
    )
    lambda_uniform_proxy = float(2.0 * total_pore_volume / total_faces_area) if total_faces_area > 0 else None
    formal = args.run_mode == "formal"
    manifest: dict[str, Any] = {
        "method": "johnson_niu_field_weighted_dynamic_pore_size_voxel_face_v1",
        "lambda_iso_m": lambda_iso,
        "lambda_uniform_proxy_m": lambda_uniform_proxy,
        "paper_reference_lambda_m": 2.7e-6,
        "paper_reference_used_as_project_value": False,
        "formal_dynamic_pore_size": formal,
        "dynamic_pore_size_use": "formal_project_extracted" if formal else "diagnostic_only_not_formal",
        "axis_order": args.axis_order,
        "physical_to_array_axis": physical_to_array_axis,
        "voxel_size_m": float(args.voxel_size_m),
        "pore_label": int(args.pore_label),
        "solid_label": int(args.solid_label),
        "surface_estimator": "six_connected_voxel_faces_pore_side_tangential_field",
        "direction_aggregation": "ratio_of_summed_volume_and_surface_integrals",
        "direction_aggregation_status": "project_rule_to_match_scalar_Niu_framework",
        "input_raw": str(args.raw),
        "input_shape": list(args.shape),
        "input_dtype": getattr(args, "dtype", None),
        "input_sha256": input_sha256,
        "git_commit": git_commit,
        "run_mode": args.run_mode,
        "crop_start": args.crop_start,
        "crop_size": args.crop_size,
        "backend": args.backend,
        "preconditioner": args.preconditioner,
        "fft_reference": args.fft_reference,
        "gauge_mode": args.gauge_mode,
        "solver_dtype": args.solver_dtype,
        "rtol": float(args.rtol),
        "maxiter": args.maxiter,
        "directions": list(args.directions),
    }
    for row in directional_rows:
        direction = row["physical_direction"]
        manifest[f"lambda_{direction}_m"] = row.get("dynamic_pore_size_m")
    if directional_rows:
        manifest["pore_voxel_count"] = int(directional_rows[0]["pore_voxel_count"])
        manifest["interface_face_count"] = int(directional_rows[0]["interface_face_count"])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--shape", nargs=3, type=int, required=True)
    parser.add_argument("--dtype", default="<u2")
    parser.add_argument("--axis-order", default=None)
    parser.add_argument("--pore-label", type=int, default=1)
    parser.add_argument("--solid-label", type=int, default=2)
    parser.add_argument("--voxel-size-m", type=float, required=True)
    parser.add_argument("--directions", nargs="+", choices=["x", "y", "z"], default=["x", "y", "z"])
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--preconditioner", choices=["none", "jacobi", "fft"], default="jacobi")
    parser.add_argument("--fft-reference", choices=["pore", "mean-face", "mean-abs"], default="pore")
    parser.add_argument("--gauge-mode", choices=["active-domain", "auto", "single-cell"], default="active-domain")
    parser.add_argument("--solver-dtype", choices=["complex64", "complex128"], default="complex128")
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--crop-start", nargs=3, type=int)
    parser.add_argument("--crop-size", nargs=3, type=int)
    parser.add_argument("--run-mode", choices=["smoke", "diagnostic", "formal"], default="smoke")
    parser.add_argument("--save-potential", action="store_true")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.axis_order is None:
        args.axis_order = "zyx" if args.run_mode != "formal" else None
    validate_run_mode(args)

    shape = parse_int3(args.shape, "--shape")
    crop_start = parse_int3(args.crop_start, "--crop-start")
    crop_size = parse_int3(args.crop_size, "--crop-size")
    if shape is None:
        raise ValueError("--shape is required")
    labels = read_volume(Path(args.raw), args.dtype, shape, crop_start, crop_size)
    valid_labels = (labels == args.pore_label) | (labels == args.solid_label)
    if not np.all(valid_labels):
        bad = np.unique(labels[~valid_labels])
        raise ValueError(f"unexpected labels in volume: {bad[:10]}")
    pore_mask = labels == args.pore_label

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "solver").mkdir(exist_ok=True)
    (out_dir / "provenance").mkdir(exist_ok=True)
    write_config(out_dir / "config.yml", args)

    physical_to_array_axis = {direction: physical_direction_to_array_axis(direction, args.axis_order) for direction in args.directions}
    directional_rows: list[dict[str, Any]] = []
    for direction in args.directions:
        axis = physical_to_array_axis[direction]
        direction_dir = out_dir / "solver" / direction
        direction_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        if args.backend == "gpu":
            result = solve_gpu_active_domain(
                labels,
                pore_label=args.pore_label,
                solid_label=args.solid_label,
                direction_axis=axis,
                voxel_size_m=args.voxel_size_m,
                solver_dtype=args.solver_dtype,
                preconditioner=args.preconditioner,
                fft_reference=args.fft_reference,
                gauge_mode=args.gauge_mode,
                rtol=args.rtol,
                maxiter=args.maxiter,
            )
        else:
            result = solve_cpu_active_domain(
                labels,
                pore_label=args.pore_label,
                solid_label=args.solid_label,
                direction_axis=axis,
                voxel_size_m=args.voxel_size_m,
                dtype=np.dtype(args.solver_dtype),
                rtol=args.rtol,
                maxiter=args.maxiter,
                use_jacobi=args.preconditioner == "jacobi",
            )
        if result.potential is None:
            raise RuntimeError("solver did not return potential")
        fields = fields_from_periodic_potential(
            result.potential,
            physical_direction=direction,  # type: ignore[arg-type]
            axis_order=args.axis_order,
            field_strength_v_m=1.0,
            voxel_size_m=args.voxel_size_m,
            pore_mask=pore_mask,
        )
        integral = compute_dynamic_pore_size_from_fields(pore_mask, fields, args.voxel_size_m)
        elapsed_s = time.perf_counter() - start
        cell_volume = float(np.prod(labels.shape) * args.voxel_size_m**3)
        identity_volume_integral = float(cell_volume * result.effective_conductivity_s_m.real)
        energy_identity_relative_error = abs(integral.volume_integral_v2_m - identity_volume_integral) / max(
            abs(identity_volume_integral), np.finfo(float).eps
        )
        row = {
            "physical_direction": direction,
            "array_axis": axis,
            "volume_integral_v2_m": integral.volume_integral_v2_m,
            "surface_integral_v2": integral.surface_integral_v2,
            "dynamic_pore_size_m": integral.dynamic_pore_size_m,
            "effective_conductivity_s_m": result.effective_conductivity_s_m.real,
            "residual_norm": result.residual_norm,
            "iterations": result.iterations,
            "info": result.info,
            "interface_face_count": integral.interface_face_count,
            "pore_voxel_count": integral.pore_voxel_count,
            "pore_volume_m3": integral.pore_volume_m3,
            "voxel_surface_area_m2": integral.voxel_surface_area_m2,
            "uniform_field_proxy_m": integral.uniform_field_proxy_m,
            "energy_identity_relative_error": energy_identity_relative_error,
            "direction_valid": True,
            "elapsed_s": elapsed_s,
        }
        directional_rows.append(row)
        pd.DataFrame(
            [{"iteration": iteration, "relative_residual_norm": residual} for iteration, residual in result.residual_history]
        ).to_csv(direction_dir / "residual_history.csv", index=False)
        if args.save_potential:
            np.save(direction_dir / "potential.npy", result.potential)
        (direction_dir / "result.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")

    directional_csv = out_dir / "directional_integrals.csv"
    pd.DataFrame(directional_rows).to_csv(directional_csv, index=False)
    input_hash = sha256_file(Path(args.raw))
    manifest = build_manifest(
        directional_rows=directional_rows,
        args=args,
        physical_to_array_axis=physical_to_array_axis,
        input_sha256=input_hash,
        git_commit=git_commit(),
    )
    (out_dir / "input_manifest.json").write_text(
        json.dumps(
            {
                "raw": args.raw,
                "shape": list(shape),
                "dtype": args.dtype,
                "crop_start": crop_start,
                "crop_size": crop_size,
                "sha256": input_hash,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest_path = out_dir / "dynamic_pore_size.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "provenance" / "dynamic_pore_size_provenance.md").write_text(
        "\n".join(
            [
                "# Dynamic pore size provenance",
                "",
                "本目录记录 project-extracted microCT Laplace field solve 得到的动态孔径。",
                "Niu 2020 Table 1 的 2.7 um 仅作为 paper reference，不作为 project value。",
                "",
                f"- manifest: {manifest_path}",
                f"- directional integrals: {directional_csv}",
                f"- run_mode: {args.run_mode}",
                f"- backend: {args.backend}",
                f"- surface_estimator: {manifest['surface_estimator']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("wrote", manifest_path)


if __name__ == "__main__":
    main()
