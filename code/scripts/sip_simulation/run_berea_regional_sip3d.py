#!/usr/bin/env python3
"""Regional (region-resolved) multi-mechanism SIP simulation on the Niu 2020 Berea micro-CT.

Pipeline (all outputs under ``results/<run>/``; finished stages are reused on rerun):

1. pnextract on the full 350^3 segmentation with ``write_elements`` / ``write_throats``
   (pore-body voxel labels and throat-interface voxels).
2. Region map: pnextract pore bodies + throat segments (``ac3d_regional.build_region_map``).
3. dc pore-only GPU solve -> fields -> field-weighted ``Lambda_k`` for every region.
4. Mechanism sweeps (interfacial / pore_regional / membrane_regional / all_regional) at
   10 frequencies picked from the experimental frequency set (2 mHz - 860 MHz).

Usage::

    python code/scripts/sip_simulation/run_berea_regional_sip3d.py \
        --config configs/sip_simulation/berea_regional_sip3d_config.py \
        --output-dir results/berea_regional_sip_pnextract_10freq_20261014
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "scripts" / "pore_network"))

from parse_pnextract_network import read_link1, read_link2, read_node1, read_node2  # noqa: E402
from pore_scale_electrical.ac3d_regional import (  # noqa: E402
    build_region_map,
    read_throat_voxels,
    read_velems,
    regional_dynamic_pore_sizes,
    solve_voxel_conductivity_gpu,
)
from pore_scale_electrical.ac3d_gpu import gpu_memory_info  # noqa: E402
from pore_scale_electrical.dynamic_pore_size import fields_from_periodic_potential  # noqa: E402
from pore_scale_electrical_2d.polarization import (  # noqa: E402
    RegionSpec,
    SIP2DParameters,
    membrane_increments,
    phase_values,
    pore_increments,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sip_simulation" / "berea_regional_sip3d_config.py"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "berea_regional_sip_pnextract_10freq_20261014"
TITLE = "microCT_Berea"


def load_config(path: Path) -> dict:
    spec = importlib.util.spec_from_file_location("berea_regional_sip3d_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return copy.deepcopy(module.CONFIG)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def build_parameters(config: dict) -> SIP2DParameters:
    e, p = config["electrical"], config["polarization"]
    return SIP2DParameters(
        water_conductivity_s_m=e["water_conductivity_s_m"],
        solid_conductivity_s_m=e["solid_conductivity_s_m"],
        water_relative_permittivity=e["water_relative_permittivity"],
        solid_relative_permittivity=e["solid_relative_permittivity"],
        epsilon_0_f_m=e["epsilon_0_f_m"],
        surface_conductance_s=p["surface_conductance_s"],
        stern_diffusion_m2_s=p["stern_diffusion_m2_s"],
        membrane_diffusion_m2_s=p["membrane_diffusion_m2_s"],
        membrane_polarizability=p["membrane_polarizability"],
    )


# --------------------------------------------------------------------------- experiment / frequencies


def load_experiment(config: dict) -> pd.DataFrame:
    real = pd.read_excel(PROJECT_ROOT / config["frequencies"]["experiment_real_workbook"], header=None)
    imag = pd.read_excel(PROJECT_ROOT / config["frequencies"]["experiment_imag_workbook"], header=None)
    r = real.iloc[2:, 2:4].dropna().astype(float).to_numpy()
    i = imag.iloc[2:, 0:3].dropna().astype(float).to_numpy()
    if not np.allclose(r[:, 0], i[:, 0]):
        raise ValueError("Figure7 and Figure8 experimental frequencies differ")
    return pd.DataFrame(
        {
            "frequency_hz": r[:, 0],
            "sigma_real_s_m": r[:, 1],
            "sigma_imag_s_m": i[:, 1],
            "real_relative_permittivity": i[:, 2],
        }
    )


def select_frequencies(experiment_hz: np.ndarray, n_points: int) -> np.ndarray:
    """Experimental frequencies nearest (in log) to ``n_points`` log-uniform targets."""

    sorted_hz = np.sort(np.asarray(experiment_hz, dtype=float))
    log_f = np.log10(sorted_hz)
    targets = np.linspace(log_f[0], log_f[-1], n_points)
    picked: list[int] = []
    for target in targets:
        order = np.argsort(np.abs(log_f - target))
        picked.append(int(next(j for j in order if j not in picked)))
    return sorted_hz[sorted(picked)]  # exact experimental values (no 10**log10 round trip)


# --------------------------------------------------------------------------- stage 1: pnextract


def run_pnextract(config: dict, out_dir: Path, pore_mask: np.ndarray) -> Path:
    input_dir = out_dir / "pore_network" / "pnextract_input"
    prefix = input_dir / TITLE
    if Path(f"{prefix}_VElems.raw").exists() and Path(f"{prefix}_link2.dat").exists():
        print("[pnextract] reuse existing extraction", flush=True)
        return prefix
    input_dir.mkdir(parents=True, exist_ok=True)
    raw = input_dir / f"{TITLE}_pore0_solid1.raw"
    np.where(pore_mask, 0, 1).astype(np.uint8).tofile(raw)
    nz, ny, nx = pore_mask.shape
    voxel_um = config["input"]["voxel_size_m"] * 1e6
    lines = [
        "ObjectType = Image", "NDims = 3", "ElementType = MET_UCHAR", "ElementByteOrderMSB = False",
        f"DimSize = {nx} {ny} {nz}", f"ElementSize = {voxel_um:g} {voxel_um:g} {voxel_um:g}", "Offset = 0 0 0",
        f"ElementDataFile = {raw.name}", "", f"title {TITLE}", *config["pnextract"]["keywords"], "",
    ]
    (input_dir / f"{TITLE}.mhd").write_text("\n".join(lines), encoding="utf-8")
    exe = PROJECT_ROOT / config["pnextract"]["exe"]
    with (input_dir / "pnextract_stdout.log").open("w", encoding="utf-8") as log:
        subprocess.run([str(exe), f"{TITLE}.mhd"], cwd=input_dir, stdout=log, stderr=subprocess.STDOUT, check=True)
    return prefix


# --------------------------------------------------------------------------- stage 2-3: regions and Lambda_k


def load_network(prefix: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pores = read_node1(Path(f"{prefix}_node1.dat")).merge(read_node2(Path(f"{prefix}_node2.dat")), on="pore_id")
    throats = read_link1(Path(f"{prefix}_link1.dat")).merge(read_link2(Path(f"{prefix}_link2.dat")), on="throat_id")
    return pores.sort_values("pore_id").reset_index(drop=True), throats.sort_values("throat_id").reset_index(drop=True)


def region_stage(config: dict, out_dir: Path, pore_mask: np.ndarray, prefix: Path, pores, throats):
    cache = out_dir / "source_data" / "region_map.npz"
    if cache.exists():
        print("[regions] reuse cached region map", flush=True)
        data = np.load(cache, allow_pickle=False)
        return data["labels"], json.loads(str(data["stats"]))
    velems = read_velems(prefix, pore_mask.shape)
    throat_voxels = read_throat_voxels(prefix, pore_mask.shape)
    if np.any((throat_voxels > 0) & ~pore_mask):
        raise ValueError("shift-corrected throat voxels fall on solid; check the pnextract writer offset")
    if not config["regions"]["include_boundary_throats"]:
        boundary = throats.loc[(throats.pore1_id <= 0) | (throats.pore2_id <= 0), "throat_id"].to_numpy()
        throat_voxels[np.isin(throat_voxels, boundary)] = 0
    scale = 2.0 * config["regions"]["throat_half_extent_fraction_of_length"]
    region = build_region_map(
        pore_mask,
        velems,
        throat_voxels,
        throat_pore1=throats.pore1_id.to_numpy(),
        throat_pore2=throats.pore2_id.to_numpy(),
        throat_length_m=scale * throats.throat_length_m.to_numpy(),
        voxel_size_m=config["input"]["voxel_size_m"],
        n_pores=len(pores),
    )
    stats = dict(region.stats, n_pores=region.n_pores, n_throats=region.n_throats)
    np.savez_compressed(cache, labels=region.labels, stats=json.dumps(stats))
    return region.labels, stats


def dc_stage(config: dict, out_dir: Path, pore_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    run_dir = out_dir / "simulation_sweeps" / "dc_pore_only"
    run_dir.mkdir(parents=True, exist_ok=True)
    potential_path = run_dir / "potential_real.npy"
    result_path = run_dir / "result.json"
    if potential_path.exists() and result_path.exists():
        print("[dc] reuse dc pore-only potential", flush=True)
        return np.load(potential_path), json.loads(result_path.read_text(encoding="utf-8"))
    sigma_w = config["electrical"]["water_conductivity_s_m"]
    solver = config["solver"]
    print("[dc] pore-only dc solve ...", flush=True)
    result = solve_voxel_conductivity_gpu(
        np.where(pore_mask, sigma_w, 0.0),
        drive_axis=solver["drive_axis"],
        voxel_size_m=config["input"]["voxel_size_m"],
        fft_reference_s_m=sigma_w,
        rtol=solver["dc_rtol"],
        maxiter=solver["maxiter"],
        residual_every=solver["residual_every"],
    )
    potential = result.potential.real.astype(np.float64)
    np.save(potential_path, potential)
    summary = {
        "sigma_eff_s_m": result.effective_conductivity_s_m.real,
        "sigma_eff_energy_form_s_m": result.energy_form_conductivity_s_m.real,
        "formation_factor": sigma_w / result.effective_conductivity_s_m.real,
        "iterations": result.iterations,
        "info": result.info,
        "true_residual_norm": result.true_residual_norm,
        "true_residual_passed": result.true_residual_passed,
        "rtol": solver["dc_rtol"],
        "elapsed_s": result.elapsed_s,
        "potential_file": potential_path.name,
    }
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(result.residual_history, columns=["iteration", "true_relative_residual"]).to_csv(
        run_dir / "residual_history.csv", index=False
    )
    print(f"[dc] sigma_eff = {summary['sigma_eff_s_m']:.6e} S/m, F = {summary['formation_factor']:.3f}, "
          f"{result.iterations} it, {result.elapsed_s:.0f} s", flush=True)
    return potential, summary


def lambda_stage(config: dict, out_dir: Path, labels: np.ndarray, potential: np.ndarray, pores, throats) -> pd.DataFrame:
    table_path = out_dir / "source_data" / "region_table.csv"
    if table_path.exists():
        print("[lambda] reuse region table", flush=True)
        return pd.read_csv(table_path)
    voxel = config["input"]["voxel_size_m"]
    pore_mask = labels > 0
    fields = fields_from_periodic_potential(
        potential, physical_direction="x", axis_order="xyz", field_strength_v_m=1.0, voxel_size_m=voxel, pore_mask=pore_mask
    )
    n_pores, n_throats = len(pores), len(throats)
    lam = regional_dynamic_pore_sizes(labels, fields, voxel, n_pores + n_throats)
    del fields
    label = np.arange(1, n_pores + n_throats + 1)
    kind = np.where(label <= n_pores, "pore_body", "throat")
    table = pd.DataFrame(
        {
            "region_label": label,
            "kind": kind,
            "pnextract_id": np.where(label <= n_pores, label, label - n_pores),
            "voxel_count": lam.voxel_count[1:].astype(int),
            "volume_m3": lam.voxel_count[1:] * voxel**3,
            "wall_face_count": lam.wall_face_count[1:].astype(int),
            "field_energy_volume_integral_v2_m": lam.volume_integral_v2_m[1:],
            "wall_tangential_energy_integral_v2": lam.surface_integral_v2[1:],
            "lambda_dynamic_m": lam.dynamic_pore_size_m[1:],
            "lambda_geometric_2V_over_S_m": lam.geometric_size_m[1:],
        }
    )
    total_v = lam.volume_integral_v2_m[1:].sum()
    total_s = lam.surface_integral_v2[1:].sum()
    table["field_energy_fraction"] = table.field_energy_volume_integral_v2_m / total_v
    table["wall_energy_fraction"] = table.wall_tangential_energy_integral_v2 / total_s
    mean_energy = table.field_energy_volume_integral_v2_m / table.volume_m3.where(table.volume_m3 > 0)
    reliable = (table.wall_tangential_energy_integral_v2 > 0) & (
        mean_energy > config["regions"]["min_mean_field_energy_fraction"]
    )
    has_wall = table.wall_face_count > 0
    table["lambda_used_m"] = np.where(reliable, table.lambda_dynamic_m, np.where(has_wall, table.lambda_geometric_2V_over_S_m, np.nan))
    table["lambda_source"] = np.where(reliable, "field_weighted", np.where(has_wall, "geometric_low_field", "no_wall_no_stern"))
    radius = np.concatenate([pores.pore_radius_m.to_numpy(), throats.throat_radius_m.to_numpy()])
    table["relaxation_radius_m"] = radius
    table["membrane_length_m"] = np.concatenate([np.full(n_pores, np.nan), throats.throat_length_m.to_numpy()])
    table["pore1_id"] = np.concatenate([np.full(n_pores, -999), throats.pore1_id.to_numpy()])
    table["pore2_id"] = np.concatenate([np.full(n_pores, -999), throats.pore2_id.to_numpy()])
    table["pnextract_volume_m3"] = np.concatenate([pores.pore_volume_m3.to_numpy(), throats.throat_volume_m3.to_numpy()])
    table.attrs["global_lambda_m"] = lam.global_dynamic_pore_size_m
    table.to_csv(table_path, index=False)
    (out_dir / "source_data" / "global_dynamic_pore_size.json").write_text(
        json.dumps({"global_lambda_m": lam.global_dynamic_pore_size_m, "definition": "2 int_V|E|^2 / int_S|E_t|^2, dc pore-only field"}, indent=2),
        encoding="utf-8",
    )
    return table


def region_specs(table: pd.DataFrame, config: dict) -> dict[int, RegionSpec]:
    throat_stern = bool(config["regions"]["throat_stern_polarization"])
    specs = {}
    for row in table.itertuples(index=False):
        if row.voxel_count == 0:
            continue
        is_throat = row.kind == "throat"
        stern = (not is_throat or throat_stern) and row.lambda_source != "no_wall_no_stern"
        specs[int(row.region_label)] = RegionSpec(
            label=int(row.region_label),
            name=f"{row.kind}_{int(row.pnextract_id)}",
            kind=row.kind,
            relaxation_radius_m=float(row.relaxation_radius_m),
            dynamic_pore_size_m=float(row.lambda_used_m) if stern else 1.0,
            membrane_length_m=float(row.membrane_length_m) if is_throat else None,
            stern_polarization=stern,
        )
    return specs


# --------------------------------------------------------------------------- stage 4: sweeps


def run_sweeps(config, params, specs, labels, frequencies, dc_potential, out_dir: Path, only_runs=None) -> dict[str, pd.DataFrame]:
    solver = config["solver"]
    voxel = config["input"]["voxel_size_m"]
    pore_inc = pore_increments(frequencies, specs, params, pore_mode="regional")
    membrane_inc = membrane_increments(frequencies, specs, params)
    n_labels = int(labels.max()) + 1
    omega = 2.0 * np.pi * frequencies
    frames = {}
    for run in config["runs"]:
        if only_runs and run["name"] not in only_runs:
            continue
        mechanism = run["mechanism"]
        solid, water = phase_values(mechanism, frequencies, specs, params, pore=pore_inc, membrane=membrane_inc)
        with_dielectric = mechanism in ("interfacial", "all")
        water_base = params.water_conductivity_s_m + (1j * omega * params.water_permittivity_f_m if with_dielectric else 0.0 * omega)
        sweep_dir = out_dir / "simulation_sweeps" / run["name"]
        sweep_dir.mkdir(parents=True, exist_ok=True)
        (sweep_dir / "config.json").write_text(
            json.dumps(
                {
                    "run": run,
                    "pore_mode": "regional" if mechanism in ("pore", "all") else "n/a",
                    "frequencies_hz": frequencies.tolist(),
                    "solver": "GPU BiCGSTAB + FFT-Poisson preconditioner on per-voxel conductivity (ac3d_regional.solve_voxel_conductivity_gpu)",
                    "rtol": solver["rtol"],
                    "rtol_dielectric_dominated": solver["rtol_dielectric_dominated"],
                    "dielectric_ratio_threshold": solver["dielectric_ratio_threshold"],
                    "maxiter": solver["maxiter"],
                    "drive_axis": solver["drive_axis"],
                    "shape_zyx": list(labels.shape),
                    "voxel_size_m": voxel,
                    "warm_start": solver["warm_start"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        rows = []
        x0 = dc_potential if solver["warm_start"] else None
        for index, frequency in enumerate(frequencies):
            freq_dir = sweep_dir / f"frequency_{index:03d}_{frequency:.6e}Hz".replace("+", "").replace(".", "p")
            freq_dir.mkdir(parents=True, exist_ok=True)
            result_path = freq_dir / "result.json"
            if result_path.exists():
                rows.append(json.loads(result_path.read_text(encoding="utf-8")))
                x0 = None
                print(f"[{run['name']}] f={frequency:.4g} Hz reuse", flush=True)
                continue
            lut = np.empty(n_labels, dtype=np.complex128)
            lut[0] = solid[index]
            lut[1:] = water_base[index]
            for label, value in water.items():
                lut[label] = value[index]
            sigma = lut[labels]
            dielectric_ratio = omega[index] * params.water_permittivity_f_m / params.water_conductivity_s_m
            rtol = (
                solver["rtol_dielectric_dominated"]
                if with_dielectric and dielectric_ratio > solver["dielectric_ratio_threshold"]
                else solver["rtol"]
            )
            memory_before = gpu_memory_info()
            result = solve_voxel_conductivity_gpu(
                sigma,
                drive_axis=solver["drive_axis"],
                voxel_size_m=voxel,
                fft_reference_s_m=complex(water_base[index]),
                rtol=rtol,
                maxiter=solver["maxiter"],
                x0=x0,
                residual_every=solver["residual_every"],
            )
            del sigma
            used_warm = x0 is not None
            x0 = result.potential if solver["warm_start"] else None
            s, se = result.effective_conductivity_s_m, result.energy_form_conductivity_s_m
            row = {
                "frequency_hz": float(frequency),
                "sigma_real_s_m": s.real,
                "sigma_imag_s_m": s.imag,
                "phase_mrad": 1e3 * float(np.angle(s)),
                "energy_form_sigma_real_s_m": se.real,
                "energy_form_sigma_imag_s_m": se.imag,
                "imag_relative_difference_mean_current_vs_energy": abs(s.imag - se.imag) / max(abs(se.imag), np.finfo(float).tiny),
                "effective_relative_permittivity": s.imag / (omega[index] * params.epsilon_0_f_m),
                "mechanism": mechanism,
                "pore_mode": "regional" if mechanism in ("pore", "all") else "n/a",
                "solid_sigma_real_s_m": complex(solid[index]).real,
                "solid_sigma_imag_s_m": complex(solid[index]).imag,
                "water_base_sigma_real_s_m": complex(water_base[index]).real,
                "water_base_sigma_imag_s_m": complex(water_base[index]).imag,
                "true_residual_norm": result.true_residual_norm,
                "true_residual_passed": result.true_residual_passed,
                "iterations": result.iterations,
                "info": result.info,
                "rtol": rtol,
                "water_dielectric_ratio": dielectric_ratio,
                "used_warm_start": used_warm,
                "elapsed_s": result.elapsed_s,
                "gpu_free_before_gib": memory_before.get("cuda_free_bytes", 0) / 2**30,
            }
            result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            pd.DataFrame(result.residual_history, columns=["iteration", "true_relative_residual"]).to_csv(
                freq_dir / "residual_history.csv", index=False
            )
            rows.append(row)
            print(
                f"[{run['name']}] f={frequency:.4g} Hz sigma*={s.real:.6e}{s.imag:+.4e}j "
                f"(energy {se.imag:+.4e}j) it={result.iterations} res={result.true_residual_norm:.1e} "
                f"{result.elapsed_s:.0f}s",
                flush=True,
            )
        frame = pd.DataFrame(rows)
        frame.to_csv(sweep_dir / "sweep_results.csv", index=False)
        frames[run["name"]] = frame
    return frames


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--runs", nargs="*", help="subset of run names (default: all)")
    parser.add_argument("--max-frequencies", type=int, help="diagnostic: only the first N frequencies")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    out_dir = Path(args.output_dir)
    out_dir = out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir
    config = load_config(config_path)
    for sub in ("configs", "source_data", "simulation_sweeps", "provenance"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, out_dir / "configs" / config_path.name)
    (out_dir / "configs" / "resolved_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    started = time.perf_counter()

    tiff_path = PROJECT_ROOT / config["input"]["tiff_path"]
    volume = tifffile.imread(tiff_path)
    values = set(np.unique(volume).tolist())
    if values != {config["input"]["pore_value"], config["input"]["solid_value"]}:
        raise ValueError(f"unexpected TIFF values {values}")
    pore_mask = volume == config["input"]["pore_value"]
    del volume

    prefix = run_pnextract(config, out_dir, pore_mask)
    pores, throats = load_network(prefix)
    labels, region_stats = region_stage(config, out_dir, pore_mask, prefix, pores, throats)
    print(f"[regions] {json.dumps(region_stats)}", flush=True)
    potential, dc_summary = dc_stage(config, out_dir, pore_mask)
    table = lambda_stage(config, out_dir, labels, potential, pores, throats)
    specs = region_specs(table, config)

    experiment = load_experiment(config)
    experiment.to_csv(out_dir / "source_data" / "niu2020_experiment_sigma_real_imag.csv", index=False)
    frequencies = select_frequencies(experiment.frequency_hz.to_numpy(), config["frequencies"]["n_points"])
    if args.max_frequencies:
        frequencies = frequencies[: args.max_frequencies]
    print(f"[frequencies] {np.array2string(frequencies, precision=4)}", flush=True)

    params = build_parameters(config)
    frames = run_sweeps(config, params, specs, labels, frequencies, potential, out_dir, only_runs=args.runs)

    manifest = {
        "status": "complete" if len(frames) == len(config["runs"]) and not args.max_frequencies else "partial",
        "model_dimension": "3D (full 350^3 CT, 3-D field solve); regional upscaling is a project extension of Niu 2020",
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "config": str(config_path.relative_to(PROJECT_ROOT)),
        "input_tiff": str(tiff_path.relative_to(PROJECT_ROOT)),
        "input_tiff_sha256": sha256(tiff_path),
        "pnextract_prefix": str(prefix.relative_to(PROJECT_ROOT)),
        "git_commit": git_commit(PROJECT_ROOT),
        "porosity": float(pore_mask.mean()),
        "region_stats": region_stats,
        "dc_pore_only": dc_summary,
        "regions_used": len(specs),
        "regions_stern_on": int(sum(s.stern_polarization for s in specs.values())),
        "lambda_source_counts": table.lambda_source.value_counts().to_dict(),
        "frequencies_hz": frequencies.tolist(),
        "runs": {name: str((out_dir / "simulation_sweeps" / name / "sweep_results.csv").relative_to(PROJECT_ROOT)) for name in frames},
        "elapsed_total_s": time.perf_counter() - started,
    }
    (out_dir / "provenance" / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("status", "regions_used", "lambda_source_counts", "elapsed_total_s")}, indent=2))


if __name__ == "__main__":
    main()
