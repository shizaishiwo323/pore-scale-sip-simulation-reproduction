#!/usr/bin/env python3
"""Two-scale regional SIP simulation of LKC sample 16 (oomoldic grainstone, rev336, 1.92 um).

* CT-resolved pores: regional (pnextract pore bodies + throat segments) Stern polarization and
  Titov membrane polarization, as in ``run_berea_regional_sip3d.py``.
* CT "solid": calcite with unresolved microporosity -> conductive, polarizable matrix.
  - dc conductivity: the single calibrated quantity (target: measured formation factor);
  - Stern polarization of the micropores: surface-to-volume distribution from the sample's NMR
    T2 spectrum (rho2 from the CT/NMR resolution boundary);
  - membrane polarization of the ion-selective matrix bridges between moldic pores: Titov
    impedance with L = local thickness of the CT solid phase (per voxel class) and
    eta = Du / (1 + Du) from the micropore Dukhin number.
* Electrochemical parameters come from calcite literature (see the config), not from Niu (2020).

Stages (outputs under ``results/<run>/``; finished stages are reused):
  1. experiment table (LKC-16.csv) and frequency selection;
  2. pnextract (must already exist, see provenance) -> region map; matrix bridge classes;
  3. dc matrix calibration (GPU) -> dc potential, Lambda_k, linear-response sensitivities;
  4. NMR micropore model; linear-response screening of the literature parameter grid;
  5. full 3-D AC sweeps (7 runs) for the central scenario (+ ``all`` for the best grid point).
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pore_scale_electrical.ac3d_regional import (  # noqa: E402
    local_thickness_gpu,
    solve_voxel_conductivity_gpu,
    voxel_sensitivities,
)
from pore_scale_electrical_2d.polarization import (  # noqa: E402
    SIP2DParameters,
    membrane_conductivity_increment,
    pore_increments,
)
from run_berea_regional_sip3d import (  # noqa: E402
    git_commit,
    lambda_stage,
    load_config,
    load_network,
    region_specs,
    region_stage,
    select_frequencies,
    sha256,
)
from run_sample89_3d_sip_comparison import add_bulk_conductivity_from_psip_impedance, parse_psip_csv  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sip_simulation" / "sample16_regional_sip3d_config.py"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "sample16_regional_sip_pnextract_20261014"
# run name -> parts switched on (dielectric, macro EDL, micro EDL, throat membrane, matrix membrane)
MECHANISMS = {
    "interfacial": {"dielectric"},
    "edl_macro_regional": {"macro"},
    "edl_micro_matrix": {"micro"},
    "membrane_regional": {"membrane"},
    "membrane_matrix": {"matrix_membrane"},
    "all": {"dielectric", "macro", "micro", "membrane", "matrix_membrane"},
    "all_no_matrix_membrane": {"dielectric", "macro", "micro", "membrane"},
}


# --------------------------------------------------------------------------- experiment


def load_experiment(config: dict, out_dir: Path) -> pd.DataFrame:
    e = config["experiment"]
    raw = parse_psip_csv(PROJECT_ROOT / e["psip_csv"])
    table = add_bulk_conductivity_from_psip_impedance(
        raw,
        current_resistor_ohm=e["current_resistor_ohm"],
        sample_length_cm=e["sample_length_cm"],
        sample_diameter_cm=e["sample_diameter_cm"],
    )
    table["phase_mrad"] = 1e3 * np.arctan2(table.experiment_sigma_imag_s_m, table.experiment_sigma_real_s_m)
    table["reliable_band"] = table.frequency_hz <= e["reliable_max_frequency_hz"]
    table.to_csv(out_dir / "source_data" / "lkc16_experiment_sigma.csv", index=False)
    return table


# --------------------------------------------------------------------------- porosity / matrix


def porosity_split(config: dict, pore_mask: np.ndarray) -> dict:
    phi_ct = float(pore_mask.mean())
    phi_he = config["porosity"]["helium"]
    phi_mu = (phi_he - phi_ct) / (1.0 - phi_ct)
    e = config["electrical"]
    eps_m = ((1.0 - phi_mu) * np.sqrt(e["calcite_relative_permittivity"]) + phi_mu * np.sqrt(e["water_relative_permittivity"])) ** 2
    return {"phi_ct": phi_ct, "phi_helium": phi_he, "phi_nmr": config["porosity"]["nmr"],
            "matrix_microporosity": phi_mu, "matrix_relative_permittivity_crim": float(eps_m)}


def matrix_classes_stage(config: dict, out_dir: Path, labels: np.ndarray, n_regions: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Extended label map: matrix voxels get one label per integer local-thickness class."""

    cache = out_dir / "source_data" / "matrix_bridge_classes.npz"
    table_path = out_dir / "source_data" / "matrix_bridge_classes.csv"
    if cache.exists() and table_path.exists():
        print("[matrix] reuse bridge classes", flush=True)
        return np.load(cache)["labels_ext"], pd.read_csv(table_path)
    matrix = labels == 0
    thickness = local_thickness_gpu(matrix)
    values, inverse, counts = np.unique(thickness[matrix], return_inverse=True, return_counts=True)
    labels_ext = labels.astype(np.int32).copy()
    labels_ext[matrix] = n_regions + 1 + inverse.astype(np.int32)
    voxel = config["input"]["voxel_size_m"]
    table = pd.DataFrame({
        "class_label": n_regions + 1 + np.arange(values.size), "local_thickness_voxels": values,
        "bridge_length_m": values * voxel, "voxel_count": counts,
    })
    np.savez_compressed(cache, labels_ext=labels_ext)
    table.to_csv(table_path, index=False)
    return labels_ext, table


def calibrate_matrix(config: dict, out_dir: Path, pore_mask: np.ndarray) -> tuple[float, np.ndarray, list[dict]]:
    run_dir = out_dir / "simulation_sweeps" / "dc_matrix_calibration"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path, potential_path = run_dir / "calibration.json", run_dir / "potential_real.npy"
    if result_path.exists() and potential_path.exists():
        print("[dc] reuse matrix calibration", flush=True)
        records = json.loads(result_path.read_text(encoding="utf-8"))
        return records[-1]["sigma_m_s_m"], np.load(potential_path), records
    sigma_w = config["electrical"]["water_conductivity_s_m"]
    m = config["matrix"]
    solver = config["solver"]
    voxel = config["input"]["voxel_size_m"]
    target = sigma_w / m["formation_factor"]
    sigma_m = m["initial_sigma_m_s_m"]
    records, potential = [], None
    for attempt in range(m["max_calibration_solves"]):
        sigma = np.where(pore_mask, sigma_w, sigma_m)
        result = solve_voxel_conductivity_gpu(
            sigma, drive_axis=solver["drive_axis"], voxel_size_m=voxel, fft_reference_s_m=float(sigma.mean()),
            rtol=solver["dc_rtol"], maxiter=solver["maxiter"], x0=potential, residual_every=solver["residual_every"],
        )
        potential = result.potential.real.copy()
        sigma_eff = result.effective_conductivity_s_m.real
        sens = voxel_sensitivities(sigma, potential, drive_axis=solver["drive_axis"], voxel_size_m=voxel)
        d_matrix = float(sens[~pore_mask].sum() / sens.size)
        records.append({
            "attempt": attempt, "sigma_m_s_m": sigma_m, "sigma_eff_s_m": sigma_eff,
            "sigma_eff_energy_form_s_m": result.energy_form_conductivity_s_m.real, "target_s_m": target,
            "apparent_formation_factor": sigma_w / sigma_eff, "dsigma_eff_dsigma_m": d_matrix,
            "iterations": result.iterations, "true_residual_norm": result.true_residual_norm,
            "true_residual_passed": result.true_residual_passed, "elapsed_s": result.elapsed_s,
        })
        print(f"[dc] sigma_m={sigma_m:.5e} -> sigma_eff={sigma_eff:.5e} (target {target:.5e}) "
              f"it={result.iterations} {result.elapsed_s:.0f}s", flush=True)
        if abs(sigma_eff / target - 1.0) < m["calibration_rtol"]:
            break
        sigma_m = sigma_m + (target - sigma_eff) / d_matrix
    np.save(potential_path, potential)
    result_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records[-1]["sigma_m_s_m"], potential, records


def sensitivity_stage(config: dict, out_dir: Path, labels_ext: np.ndarray, sigma_dc: np.ndarray, potential: np.ndarray) -> np.ndarray:
    path = out_dir / "source_data" / "linear_response_sensitivity_per_label.csv"
    if path.exists():
        return pd.read_csv(path).sensitivity_sum_v2_m2.to_numpy()
    sens = voxel_sensitivities(sigma_dc, potential, drive_axis=config["solver"]["drive_axis"], voxel_size_m=config["input"]["voxel_size_m"])
    per_label = np.bincount(labels_ext.ravel(), weights=sens.ravel(), minlength=int(labels_ext.max()) + 1)
    pd.DataFrame({"label": np.arange(per_label.size), "sensitivity_sum_v2_m2": per_label}).to_csv(path, index=False)
    return per_label


# --------------------------------------------------------------------------- NMR micropores


def nmr_micropores(config: dict, phi_ct: float, rho2_choice) -> dict:
    n = config["nmr"]
    spec = pd.read_csv(PROJECT_ROOT / n["spectrum_csv"], header=None).astype(float).to_numpy()
    t2_s = spec[:, 0] * 1e-3
    amp = np.clip(spec[:, 1], 0.0, None)
    cum = np.cumsum(amp) / amp.sum()
    micro_fraction = 1.0 - phi_ct / config["porosity"]["nmr"]
    t2_cut = float(10 ** np.interp(micro_fraction, cum, np.log10(t2_s)))
    a = n["shape_factor_a"]
    rho2_boundary = n["ct_resolution_radius_m"] / (a * t2_cut)
    rho2 = rho2_boundary if rho2_choice == "ct_boundary" else float(rho2_choice) * 1e-6
    micro = t2_s < t2_cut
    weights = np.where(micro, amp, 0.0) / amp[micro].sum()
    reliable = t2_s >= n["min_reliable_t2_echo_multiple"] * n["echo_time_s"]
    use = micro & reliable
    surface = 1.0 / (rho2 * t2_s[use])
    return {
        "rho2_m_s": rho2, "rho2_ct_boundary_m_s": rho2_boundary, "t2_cut_s": t2_cut, "micro_fraction_of_nmr": micro_fraction,
        "t2_s": t2_s[use], "weight": weights[use], "surface_to_volume_m": surface,
        "radius_m": a * rho2 * t2_s[use], "excluded_short_t2_weight": float(weights[micro & ~reliable].sum()),
        "mean_surface_to_volume_m": float(np.sum(weights[use] * surface) / weights[use].sum()),
    }


def matrix_edl_increment(freq: np.ndarray, micro: dict, stern_s: float, diffusion: float, sigma_m: float, sigma_w: float) -> np.ndarray:
    """(sigma_m / sigma_w) sum_j w_j Sigma_S (S/V)_j i w tau_j / (1 + i w tau_j): Archie-scaled micropore Stern increment."""

    omega = 2.0 * np.pi * np.atleast_1d(freq)[:, None]
    tau = micro["radius_m"][None, :] ** 2 / (2.0 * diffusion)
    iwt = 1j * omega * tau
    water_increment = (micro["weight"] * stern_s * micro["surface_to_volume_m"])[None, :] * iwt / (1.0 + iwt)
    return (sigma_m / sigma_w) * water_increment.sum(axis=1)


def dukhin_eta(diffuse_s: float, surface_to_volume_m: float, sigma_w: float) -> float:
    du = diffuse_s * surface_to_volume_m / sigma_w
    return du / (1.0 + du)


# --------------------------------------------------------------------------- increments


def polarization_parameters(config: dict, stern_s: float, diffusion: float) -> SIP2DParameters:
    e, p = config["electrical"], config["polarization"]
    return SIP2DParameters(
        water_conductivity_s_m=e["water_conductivity_s_m"], solid_conductivity_s_m=0.0,
        water_relative_permittivity=e["water_relative_permittivity"], solid_relative_permittivity=e["calcite_relative_permittivity"],
        epsilon_0_f_m=e["epsilon_0_f_m"], surface_conductance_s=stern_s, stern_diffusion_m2_s=diffusion,
        membrane_diffusion_m2_s=p["membrane_water_diffusion_m2_s"], membrane_polarizability=0.0,
    )


def scenario_increments(freq, config, specs, table, classes, micro, sigma_m, stern_s, diffusion, diffuse_s, eps_m) -> dict:
    params = polarization_parameters(config, stern_s, diffusion)
    sigma_w = params.water_conductivity_s_m
    omega = 2.0 * np.pi * np.atleast_1d(freq)
    radius = table.set_index("region_label").relaxation_radius_m
    membrane = {}
    for label, spec in specs.items():
        if spec.membrane_length_m is None:
            continue
        eta = dukhin_eta(diffuse_s, 2.0 / radius[label], sigma_w)
        membrane[label] = membrane_conductivity_increment(freq, spec.membrane_length_m, replace(params, membrane_polarizability=eta))
    eta_m = dukhin_eta(diffuse_s, micro["mean_surface_to_volume_m"], sigma_w)
    matrix_params = replace(params, water_conductivity_s_m=sigma_m, membrane_polarizability=eta_m)
    matrix_membrane = {} if not config["polarization"]["matrix_membrane"] else {
        int(row.class_label): membrane_conductivity_increment(freq, row.bridge_length_m, matrix_params)
        for row in classes.itertuples(index=False)
    }
    return {
        "macro": pore_increments(freq, specs, params, pore_mode="regional"),
        "membrane": membrane,
        "matrix_membrane": matrix_membrane,
        "matrix_eta": eta_m,
        "micro": matrix_edl_increment(freq, micro, stern_s, diffusion, sigma_m, sigma_w),
        "water_dielectric": 1j * omega * params.water_permittivity_f_m,
        "matrix_dielectric": 1j * omega * eps_m * params.epsilon_0_f_m,
    }


def linear_response(inc: dict, per_label: np.ndarray, class_labels: np.ndarray, n_voxels: int, sigma_dc: float) -> dict:
    """sigma_eff(w) per mechanism from the dc sensitivities (first order in the increments)."""

    s_matrix = per_label[class_labels].sum()
    s_water = per_label.sum() - s_matrix
    parts = {
        "dielectric": (inc["water_dielectric"] * s_water + inc["matrix_dielectric"] * s_matrix) / n_voxels,
        "macro": sum(value * per_label[label] for label, value in inc["macro"].items()) / n_voxels,
        "micro": inc["micro"] * s_matrix / n_voxels,
        "membrane": sum(value * per_label[label] for label, value in inc["membrane"].items()) / n_voxels,
        "matrix_membrane": sum((value * per_label[label] for label, value in inc["matrix_membrane"].items()), 0.0) / n_voxels,
    }
    return {name: sigma_dc + sum(parts[p] for p in on) for name, on in MECHANISMS.items()}


def misfit(model_f, model_sigma, experiment: pd.DataFrame, thesis: dict) -> dict:
    band = experiment[experiment.reliable_band & (experiment.experiment_sigma_imag_s_m > 0)]
    sim_imag = np.interp(np.log10(band.frequency_hz), np.log10(model_f), model_sigma.imag)
    at1 = np.interp(0.0, np.log10(model_f), model_sigma.imag)
    ratio = np.clip(sim_imag, 1e-30, None) / band.experiment_sigma_imag_s_m
    return {
        "rms_log10_sigma_imag_vs_csv": float(np.sqrt(np.mean(np.log10(ratio) ** 2))),
        "max_abs_log10_sigma_imag_vs_csv": float(np.max(np.abs(np.log10(ratio)))),
        "median_ratio_sigma_imag_vs_csv": float(np.median(ratio)),
        "sigma_imag_1hz_ratio_vs_thesis": float(at1 / thesis["sigma_imag_s_m"]),
        "sigma_real_1hz_ratio_vs_thesis": float(np.interp(0.0, np.log10(model_f), model_sigma.real) / thesis["sigma_real_s_m"]),
    }


# --------------------------------------------------------------------------- full sweeps


def build_lut(on: set, index: int, inc: dict, sigma_m: float, sigma_w: float, n_labels: int, class_labels: np.ndarray,
              background: dict | None = None) -> np.ndarray:
    """Per-label conductivity at one frequency. ``background`` = frequency-independent increments present
    in every run (``{"water": {label: value}, "matrix": value}``, e.g. dc surface conduction)."""

    base = sigma_w + (inc["water_dielectric"][index] if "dielectric" in on else 0.0)
    lut = np.full(n_labels, base, dtype=np.complex128)
    matrix = sigma_m + (inc["matrix_dielectric"][index] if "dielectric" in on else 0.0) + (inc["micro"][index] if "micro" in on else 0.0)
    lut[class_labels] = matrix
    if "matrix_membrane" in on:
        for label, value in inc["matrix_membrane"].items():
            lut[label] += value[index]
    for part in ("macro", "membrane"):
        if part in on:
            for label, value in inc[part].items():
                lut[label] += value[index]
    if background is not None:
        lut[class_labels] += background["matrix"]
        for label, value in background["water"].items():
            lut[label] += value
    return lut


def run_sweeps(config, out_dir, labels_ext, class_labels, frequencies, inc, sigma_m, potential, runs: dict, scenario_meta: dict,
               background: dict | None = None, sweep_root: Path | None = None) -> dict:
    solver = config["solver"]
    voxel = config["input"]["voxel_size_m"]
    sigma_w = config["electrical"]["water_conductivity_s_m"]
    n_labels = int(labels_ext.max()) + 1
    frames = {}
    for run_name, on in runs.items():
        sweep_dir = (sweep_root or out_dir / "simulation_sweeps") / run_name
        sweep_dir.mkdir(parents=True, exist_ok=True)
        (sweep_dir / "config.json").write_text(json.dumps({
            "parts": sorted(on), "scenario": scenario_meta, "frequencies_hz": frequencies.tolist(),
            "matrix_sigma_dc_s_m": sigma_m, "matrix_membrane_eta": inc["matrix_eta"],
            "solver": "ac3d_regional.solve_voxel_conductivity_gpu (COCG, fully conducting medium)",
            "rtol": solver["rtol"], "drive_axis": solver["drive_axis"], "shape_zyx": list(labels_ext.shape), "voxel_size_m": voxel,
        }, indent=2, default=float), encoding="utf-8")
        rows, x0 = [], potential if solver["warm_start"] else None
        for index, frequency in enumerate(frequencies):
            freq_dir = sweep_dir / f"frequency_{index:03d}_{frequency:.6e}Hz".replace("+", "").replace(".", "p")
            freq_dir.mkdir(parents=True, exist_ok=True)
            result_path = freq_dir / "result.json"
            if result_path.exists():
                rows.append(json.loads(result_path.read_text(encoding="utf-8")))
                x0 = None
                continue
            lut = build_lut(on, index, inc, sigma_m, sigma_w, n_labels, class_labels, background)
            sigma = lut[labels_ext]
            result = solve_voxel_conductivity_gpu(
                sigma, drive_axis=solver["drive_axis"], voxel_size_m=voxel, fft_reference_s_m=complex(sigma.mean()),
                rtol=solver["rtol"], maxiter=solver["maxiter"], x0=x0, residual_every=solver["residual_every"],
            )
            del sigma
            x0 = result.potential if solver["warm_start"] else None
            s, se = result.effective_conductivity_s_m, result.energy_form_conductivity_s_m
            row = {
                "frequency_hz": float(frequency), "sigma_real_s_m": s.real, "sigma_imag_s_m": s.imag,
                "phase_mrad": 1e3 * float(np.angle(s)), "energy_form_sigma_real_s_m": se.real, "energy_form_sigma_imag_s_m": se.imag,
                "imag_relative_difference_mean_current_vs_energy": abs(s.imag - se.imag) / max(abs(se.imag), np.finfo(float).tiny),
                "parts": "+".join(sorted(on)), "true_residual_norm": result.true_residual_norm,
                "true_residual_passed": result.true_residual_passed, "iterations": result.iterations, "info": result.info,
                "rtol": solver["rtol"], "elapsed_s": result.elapsed_s,
            }
            result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            pd.DataFrame(result.residual_history, columns=["iteration", "true_relative_residual"]).to_csv(freq_dir / "residual_history.csv", index=False)
            rows.append(row)
            print(f"[{run_name}] f={frequency:.4g} Hz sigma*={s.real:.6e}{s.imag:+.4e}j (energy {se.imag:+.4e}j) "
                  f"it={result.iterations} res={result.true_residual_norm:.1e} {result.elapsed_s:.0f}s", flush=True)
        frame = pd.DataFrame(rows)
        frame.to_csv(sweep_dir / "sweep_results.csv", index=False)
        frames[run_name] = frame
    return frames


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--skip-sweeps", action="store_true", help="stop after the linear-response screening")
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

    experiment = load_experiment(config, out_dir)
    f_cfg = config["frequencies"]
    band = experiment[(experiment.frequency_hz >= f_cfg["min_hz"]) & (experiment.frequency_hz <= f_cfg["max_hz"])]
    frequencies = select_frequencies(band.frequency_hz.to_numpy(), f_cfg["n_points"])
    print(f"[frequencies] {np.array2string(frequencies, precision=4)}", flush=True)

    tiff_path = PROJECT_ROOT / config["input"]["tiff_path"]
    volume = tifffile.imread(tiff_path)
    if set(np.unique(volume).tolist()) != {config["input"]["pore_value"], config["input"]["solid_value"]}:
        raise ValueError("unexpected TIFF labels")
    pore_mask = volume == config["input"]["pore_value"]
    del volume
    prefix = out_dir / "pore_network" / "pnextract_input" / config["input"]["title"]
    if not Path(f"{prefix}_VElems.raw").exists():
        raise FileNotFoundError(f"run pnextract first (see provenance): {prefix}_VElems.raw")
    pores, throats = load_network(prefix)
    labels, region_stats = region_stage(config, out_dir, pore_mask, prefix, pores, throats)
    n_regions = len(pores) + len(throats)
    labels_ext, classes = matrix_classes_stage(config, out_dir, labels, n_regions)
    class_labels = classes.class_label.to_numpy()
    print(f"[regions] {json.dumps(region_stats)}; matrix bridge classes: {len(classes)}", flush=True)
    porosity = porosity_split(config, pore_mask)

    sigma_w = config["electrical"]["water_conductivity_s_m"]
    sigma_m, potential, calibration = calibrate_matrix(config, out_dir, pore_mask)
    sigma_dc = calibration[-1]["sigma_eff_s_m"]
    porosity["matrix_sigma_dc_s_m"] = sigma_m
    porosity["implied_micro_cementation_exponent"] = float(np.log(sigma_w / sigma_m) / np.log(1.0 / porosity["matrix_microporosity"]))
    table = lambda_stage(config, out_dir, labels, potential, pores, throats)
    specs = region_specs(table, config)
    per_label = sensitivity_stage(config, out_dir, labels_ext, np.where(pore_mask, sigma_w, sigma_m), potential)
    n_voxels = labels.size
    print(f"[dc] sigma_m={sigma_m:.4e}, sigma_eff={sigma_dc:.4e}, m_mu={porosity['implied_micro_cementation_exponent']:.2f}, "
          f"matrix share of sum(sigma*sensitivity)={sigma_m * per_label[class_labels].sum() / (sigma_dc * n_voxels):.3f}", flush=True)

    # ---- linear-response screening of the literature grid
    pol = config["polarization"]
    eps_m = porosity["matrix_relative_permittivity_crim"]
    dense = np.logspace(np.log10(0.05), np.log10(2.0e4), f_cfg["dense_points"])
    rows, spectra = [], []
    grid = itertools.product(pol["stern_conductance_s"], pol["stern_diffusion_m2_s"], pol["rho2_um_s"], pol["diffuse_layer_conductance_s"])
    for stern_s, diffusion, rho2_choice, diffuse_s in grid:
        micro = nmr_micropores(config, porosity["phi_ct"], rho2_choice)
        inc = scenario_increments(dense, config, specs, table, classes, micro, sigma_m, stern_s, diffusion, diffuse_s, eps_m)
        lr = linear_response(inc, per_label, class_labels, n_voxels, sigma_dc)
        scenario = f"S{stern_s * 1e9:.1f}nS_D{diffusion:.2e}_rho2{rho2_choice}_Sd{diffuse_s * 1e9:.2f}nS"
        rows.append({"scenario": scenario, "stern_conductance_s": stern_s, "stern_diffusion_m2_s": diffusion,
                     "rho2_m_s": micro["rho2_m_s"], "rho2_choice": str(rho2_choice), "diffuse_layer_conductance_s": diffuse_s,
                     "matrix_eta": inc["matrix_eta"],
                     **misfit(dense, lr["all"], experiment, config["experiment"]["thesis_point"])})
        for name, values in lr.items():
            spectra.append(pd.DataFrame({"scenario": scenario, "mechanism": name, "frequency_hz": dense,
                                         "sigma_real_s_m": values.real, "sigma_imag_s_m": values.imag}))
    screening = pd.DataFrame(rows).sort_values("rms_log10_sigma_imag_vs_csv").reset_index(drop=True)
    screening.to_csv(out_dir / "source_data" / "linear_response_scenario_screening.csv", index=False)
    pd.concat(spectra).to_csv(out_dir / "source_data" / "linear_response_spectra.csv", index=False)
    print(screening.head(8).to_string(index=False), flush=True)

    central = pol["central"]
    micro_c = nmr_micropores(config, porosity["phi_ct"], central["rho2_um_s"])
    pd.DataFrame({k: micro_c[k] for k in ("t2_s", "weight", "surface_to_volume_m", "radius_m")}).to_csv(
        out_dir / "source_data" / "nmr_micropore_model_central.csv", index=False)
    inc_c = scenario_increments(frequencies, config, specs, table, classes, micro_c, sigma_m, central["stern_conductance_s"],
                                central["stern_diffusion_m2_s"], central["diffuse_layer_conductance_s"], eps_m)
    meta = {"porosity": porosity, "dc_calibration": calibration, "region_stats": region_stats,
            "matrix_bridge_classes": int(len(classes)), "central_matrix_membrane_eta": inc_c["matrix_eta"],
            "nmr_central": {k: v for k, v in micro_c.items() if not isinstance(v, np.ndarray)},
            "frequencies_hz": frequencies.tolist(), "screening_best": screening.iloc[0].to_dict()}
    (out_dir / "provenance" / "model_summary.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    if args.skip_sweeps:
        return

    # ---- full 3-D sweeps
    frames = run_sweeps(config, out_dir, labels_ext, class_labels, frequencies, inc_c, sigma_m, potential,
                        {f"central_{name}": on for name, on in MECHANISMS.items()}, {"name": "central", **central})
    best = screening.iloc[0]
    best_choice = best.rho2_choice if best.rho2_choice == "ct_boundary" else float(best.rho2_choice)
    best_key = (best.stern_conductance_s, best.stern_diffusion_m2_s, str(best_choice), best.diffuse_layer_conductance_s)
    central_key = (central["stern_conductance_s"], central["stern_diffusion_m2_s"], str(central["rho2_um_s"]), central["diffuse_layer_conductance_s"])
    same_numbers = np.allclose([best_key[0], best_key[1], best_key[3]], [central_key[0], central_key[1], central_key[3]], rtol=1e-9, atol=0.0)
    if not (same_numbers and best_key[2] == central_key[2]):
        micro_b = nmr_micropores(config, porosity["phi_ct"], best_choice)
        inc_b = scenario_increments(frequencies, config, specs, table, classes, micro_b, sigma_m, best.stern_conductance_s,
                                    best.stern_diffusion_m2_s, best.diffuse_layer_conductance_s, eps_m)
        frames.update(run_sweeps(config, out_dir, labels_ext, class_labels, frequencies, inc_b, sigma_m, potential,
                                 {"best_all": MECHANISMS["all"]}, {"name": "best_grid_point", "scenario": best.scenario}))
    manifest = {
        "status": "complete", "model_dimension": "3D field solve on 336^3 CT REV (1.92 um); two-scale regional model",
        "command": " ".join([Path(sys.executable).name, *sys.argv]), "config": str(config_path.relative_to(PROJECT_ROOT)),
        "input_tiff": str(tiff_path.relative_to(PROJECT_ROOT)), "input_tiff_sha256": sha256(tiff_path), "git_commit": git_commit(PROJECT_ROOT),
        **meta, "runs": {name: str((out_dir / "simulation_sweeps" / name / "sweep_results.csv").relative_to(PROJECT_ROOT)) for name in frames},
        "elapsed_total_s": time.perf_counter() - started,
    }
    (out_dir / "provenance" / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
