#!/usr/bin/env python3
"""Second-round variants of the sample-16 two-scale regional SIP model (see the config ``variants``).

* Stern dc conduction: Stern + diffuse layers also conduct at dc, Sigma_DC = Sigma_S + Sigma_d, added as a
  frequency-independent background (regional 2 Sigma_DC / Lambda_k in CT regions, Archie-scaled
  Sigma_DC * sum_j w_j (S/V)_j in the matrix) to every mechanism run; the polarization terms are unchanged.
* Brine: ``thesis`` keeps sigma_w = 0.12 S/m; ``csv`` infers sigma_w from LKC-16.csv itself by matching the model
  sigma'(0.1 Hz) (full dc solve + linear-response polarization real part) to the file's sigma'(0.1 Hz).
  Bulk conductivities scale together (F is salinity independent), so the cached dc field and the
  linear-response sensitivities of ``run_sample16_regional_sip3d.py`` stay valid.

Outputs: ``simulation_sweeps/<variant>/<run>/``, ``source_data/<variant>_*.csv``, ``provenance/<variant>_summary.json``.

Usage::

    python code/scripts/sip_simulation/run_sample16_regional_variants.py --variant sternDC_brine_thesis
    python code/scripts/sip_simulation/run_sample16_regional_variants.py --variant sternDC_brine_csv
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pore_scale_electrical.ac3d_regional import solve_voxel_conductivity_gpu  # noqa: E402
from run_berea_regional_sip3d import lambda_stage, load_config, load_network, region_specs, region_stage, select_frequencies  # noqa: E402
from run_sample16_regional_sip3d import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    MECHANISMS,
    calibrate_matrix,
    linear_response,
    load_experiment,
    matrix_classes_stage,
    misfit,
    nmr_micropores,
    porosity_split,
    run_sweeps,
    scenario_increments,
    sensitivity_stage,
)


def surface_background(specs: dict, micro: dict, sigma_m: float, sigma_w: float, sigma_dc_s: float) -> dict:
    """Frequency-independent dc surface-conduction increments (same geometric conversion as the Stern terms)."""

    water = {label: 2.0 * sigma_dc_s / spec.dynamic_pore_size_m for label, spec in specs.items() if spec.stern_polarization}
    matrix = (sigma_m / sigma_w) * sigma_dc_s * float(np.sum(micro["weight"] * micro["surface_to_volume_m"]))
    return {"water": water, "matrix": matrix}


def background_linear(bg: dict, per_label: np.ndarray, class_labels: np.ndarray, n_voxels: int) -> float:
    return (sum(value * per_label[label] for label, value in bg["water"].items()) + bg["matrix"] * per_label[class_labels].sum()) / n_voxels


def dc_with_background(config, labels_ext, class_labels, sigma_w, sigma_m, bg, x0) -> tuple[float, np.ndarray, dict]:
    lut = np.full(int(labels_ext.max()) + 1, sigma_w)
    lut[class_labels] = sigma_m + bg["matrix"]
    for label, value in bg["water"].items():
        lut[label] += value
    sigma = lut[labels_ext]
    solver = config["solver"]
    result = solve_voxel_conductivity_gpu(
        sigma, drive_axis=solver["drive_axis"], voxel_size_m=config["input"]["voxel_size_m"], fft_reference_s_m=float(sigma.mean()),
        rtol=solver["dc_rtol"], maxiter=solver["maxiter"], x0=x0, residual_every=solver["residual_every"],
    )
    record = {"sigma_w_s_m": sigma_w, "sigma_m_s_m": sigma_m, "sigma_eff_dc_s_m": result.effective_conductivity_s_m.real,
              "iterations": result.iterations, "true_residual_passed": result.true_residual_passed, "elapsed_s": result.elapsed_s}
    return result.effective_conductivity_s_m.real, result.potential.real, record


def real_misfit(model_f, model_sigma, experiment: pd.DataFrame, thesis: dict) -> dict:
    band = experiment[experiment.reliable_band]
    sim = np.interp(np.log10(band.frequency_hz), np.log10(model_f), model_sigma.real)
    return {
        "rms_log10_sigma_real_vs_csv": float(np.sqrt(np.mean(np.log10(sim / band.experiment_sigma_real_s_m) ** 2))),
        "sigma_real_1hz_ratio_vs_thesis_point": float(np.interp(0.0, np.log10(model_f), model_sigma.real) / thesis["sigma_real_s_m"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    started = time.perf_counter()
    config = load_config(Path(args.config).resolve())
    variant = config["variants"][args.variant]
    out_dir = Path(args.output_dir)
    out_dir = out_dir if out_dir.is_absolute() else PROJECT_ROOT / out_dir

    # ---- cached stages of the base run
    experiment = load_experiment(config, out_dir)
    f_cfg = config["frequencies"]
    band = experiment[(experiment.frequency_hz >= f_cfg["min_hz"]) & (experiment.frequency_hz <= f_cfg["max_hz"])]
    frequencies = select_frequencies(band.frequency_hz.to_numpy(), f_cfg["n_points"])
    pore_mask = tifffile.imread(PROJECT_ROOT / config["input"]["tiff_path"]) == config["input"]["pore_value"]
    prefix = out_dir / "pore_network" / "pnextract_input" / config["input"]["title"]
    pores, throats = load_network(prefix)
    labels, _ = region_stage(config, out_dir, pore_mask, prefix, pores, throats)
    labels_ext, classes = matrix_classes_stage(config, out_dir, labels, len(pores) + len(throats))
    class_labels = classes.class_label.to_numpy()
    porosity = porosity_split(config, pore_mask)
    sigma_w0 = config["electrical"]["water_conductivity_s_m"]
    sigma_m0, potential, calibration = calibrate_matrix(config, out_dir, pore_mask)
    sigma_dc0 = calibration[-1]["sigma_eff_s_m"]
    table = lambda_stage(config, out_dir, labels, potential, pores, throats)
    specs = region_specs(table, config)
    per_label = sensitivity_stage(config, out_dir, labels_ext, np.where(pore_mask, sigma_w0, sigma_m0), potential)
    n_voxels = labels.size
    eps_m = porosity["matrix_relative_permittivity_crim"]
    pol = config["polarization"]
    central = pol["central"]
    micro_c = nmr_micropores(config, porosity["phi_ct"], central["rho2_um_s"])
    use_dc = variant["stern_dc_conduction"]

    def sigma_dc_surface(stern_s, diffuse_s):
        return (stern_s + diffuse_s) if use_dc else 0.0

    # ---- brine conductivity
    dc_records = []
    scale = 1.0
    if variant["brine"] == "csv":
        target = float(experiment.sort_values("frequency_hz").experiment_sigma_real_s_m.iloc[0])
        f_low = float(experiment.frequency_hz.min())
        bg_c = surface_background(specs, micro_c, sigma_m0, sigma_w0, sigma_dc_surface(central["stern_conductance_s"], central["diffuse_layer_conductance_s"]))

        def pol_real(s):
            cfg = copy.deepcopy(config)
            cfg["electrical"]["water_conductivity_s_m"] = s * sigma_w0
            inc = scenario_increments(np.array([f_low]), cfg, specs, table, classes, micro_c, s * sigma_m0, central["stern_conductance_s"],
                                      central["stern_diffusion_m2_s"], central["diffuse_layer_conductance_s"], eps_m)
            return float((linear_response(inc, per_label, class_labels, n_voxels, 0.0)["all"].real)[0])

        scale = (target - background_linear(bg_c, per_label, class_labels, n_voxels) - pol_real(1.0)) / sigma_dc0
        history = []
        x0 = potential
        for _ in range(4):
            dc, x0, record = dc_with_background(config, labels_ext, class_labels, scale * sigma_w0, scale * sigma_m0, bg_c, x0)
            model = dc + pol_real(scale)
            record.update({"scale": scale, "model_sigma_real_at_f_low": model, "target": target})
            dc_records.append(record)
            history.append((scale, model))
            print(f"[brine] scale={scale:.5f} sigma_w={scale * sigma_w0:.5f} S/m -> sigma'({f_low:g} Hz)={model:.5e} (target {target:.5e})", flush=True)
            if abs(model / target - 1.0) < 1e-3:
                break
            if len(history) == 1:
                scale = scale * target / model
            else:
                (s1, m1), (s2, m2) = history[-2], history[-1]
                scale = s2 + (target - m2) * (s2 - s1) / (m2 - m1)
    sigma_w, sigma_m, sigma_dc_bulk = scale * sigma_w0, scale * sigma_m0, scale * sigma_dc0
    config_v = copy.deepcopy(config)
    config_v["electrical"]["water_conductivity_s_m"] = sigma_w
    print(f"[{args.variant}] sigma_w={sigma_w:.5f} S/m, sigma_m={sigma_m:.4e} S/m, bulk sigma_eff={sigma_dc_bulk:.4e}", flush=True)

    # ---- linear-response screening (dc surface background + polarization, around the scaled bulk dc state)
    dense = np.logspace(np.log10(0.05), np.log10(2.0e4), f_cfg["dense_points"])
    thesis = config["experiment"]["thesis_point"]
    rows, spectra = [], []
    grid = itertools.product(pol["stern_conductance_s"], pol["stern_diffusion_m2_s"], pol["rho2_um_s"], pol["diffuse_layer_conductance_s"])
    for stern_s, diffusion, rho2_choice, diffuse_s in grid:
        micro = nmr_micropores(config, porosity["phi_ct"], rho2_choice)
        inc = scenario_increments(dense, config_v, specs, table, classes, micro, sigma_m, stern_s, diffusion, diffuse_s, eps_m)
        surface = background_linear(surface_background(specs, micro, sigma_m, sigma_w, sigma_dc_surface(stern_s, diffuse_s)), per_label, class_labels, n_voxels)
        lr = {name: value + surface for name, value in linear_response(inc, per_label, class_labels, n_voxels, sigma_dc_bulk).items()}
        scenario = f"S{stern_s * 1e9:.1f}nS_D{diffusion:.2e}_rho2{rho2_choice}_Sd{diffuse_s * 1e9:.2f}nS"
        rows.append({"scenario": scenario, "stern_conductance_s": stern_s, "stern_diffusion_m2_s": diffusion, "rho2_choice": str(rho2_choice),
                     "diffuse_layer_conductance_s": diffuse_s, "matrix_eta": inc["matrix_eta"], "dc_surface_sigma_s_m": surface,
                     **misfit(dense, lr["all"], experiment, thesis), **real_misfit(dense, lr["all"], experiment, thesis)})
        for name in ("all", "all_no_matrix_membrane"):
            spectra.append(pd.DataFrame({"scenario": scenario, "mechanism": name, "frequency_hz": dense,
                                         "sigma_real_s_m": lr[name].real, "sigma_imag_s_m": lr[name].imag}))
    screening = pd.DataFrame(rows).sort_values("rms_log10_sigma_imag_vs_csv").reset_index(drop=True)
    screening.to_csv(out_dir / "source_data" / f"{args.variant}_linear_response_scenario_screening.csv", index=False)
    pd.concat(spectra).to_csv(out_dir / "source_data" / f"{args.variant}_linear_response_spectra.csv", index=False)
    print(screening.head(5).to_string(index=False), flush=True)

    # ---- full 3-D sweeps: central (all mechanisms) + best grid point (all)
    root = out_dir / "simulation_sweeps" / args.variant
    sets = {"central": (central["stern_conductance_s"], central["stern_diffusion_m2_s"], central["rho2_um_s"], central["diffuse_layer_conductance_s"])}
    best = screening.iloc[0]
    best_choice = best.rho2_choice if best.rho2_choice == "ct_boundary" else float(best.rho2_choice)
    sets["best"] = (best.stern_conductance_s, best.stern_diffusion_m2_s, best_choice, best.diffuse_layer_conductance_s)
    frames = {}
    for set_name, (stern_s, diffusion, rho2_choice, diffuse_s) in sets.items():
        micro = nmr_micropores(config, porosity["phi_ct"], rho2_choice)
        inc = scenario_increments(frequencies, config_v, specs, table, classes, micro, sigma_m, stern_s, diffusion, diffuse_s, eps_m)
        bg = surface_background(specs, micro, sigma_m, sigma_w, sigma_dc_surface(stern_s, diffuse_s))
        runs = {f"central_{name}": on for name, on in MECHANISMS.items()} if set_name == "central" else {"best_all": MECHANISMS["all"]}
        frames.update(run_sweeps(config_v, out_dir, labels_ext, class_labels, frequencies, inc, sigma_m, potential, runs,
                                 {"name": set_name, "variant": args.variant, "stern_conductance_s": stern_s, "stern_diffusion_m2_s": diffusion,
                                  "rho2_choice": str(rho2_choice), "diffuse_layer_conductance_s": diffuse_s,
                                  "dc_surface_conductance_s": sigma_dc_surface(stern_s, diffuse_s), "matrix_dc_surface_increment_s_m": bg["matrix"]},
                                 background=bg, sweep_root=root))
    summary = {
        "variant": args.variant, "definition": variant, "sigma_w_s_m": sigma_w, "brine_scale_vs_thesis": scale,
        "sigma_m_s_m": sigma_m, "bulk_sigma_eff_s_m": sigma_dc_bulk, "brine_calibration_dc_solves": dc_records,
        "frequencies_hz": frequencies.tolist(), "screening_best": best.to_dict(),
        "central_screening": screening[screening.scenario == "S1.8nS_D5.10e-10_rho2ct_boundary_Sd0.30nS"].iloc[0].to_dict(),
        "runs": {name: str((root / name / "sweep_results.csv").relative_to(PROJECT_ROOT)) for name in frames},
        "elapsed_s": time.perf_counter() - started,
    }
    (out_dir / "provenance" / f"{args.variant}_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
