"""2D SIP of the 20 throat cases of the NMR throat sweep (w x L), same electrochemistry as C3 (Niu et al., 2020).

Purpose: test whether SIP observables carry the throat geometry that controls NMR diffusive coupling.
  - dc phase: pore-only dc solve per case -> formation factor F, global/regional dynamic pore size Lambda,
    field-energy share of the throats, mesh throat length (throat area / w).
  - spectra phase: pore_regional, all_regional and all_niu sweeps (1e-3-1e5 Hz) per case, then a Debye
    decomposition in the lab band -> peak frequency, peak radius r = sqrt(2 D tau), amplitude route
    Lambda_amp = 2 Sigma_S / (F Delta sigma).

Geometry, grid and region handling are reused from run_ideal_two_pore_channel_sip2d.py. Model dimension: 2D.

Example (from the SIP repo root):
    python code/scripts/sip_simulation/run_throat_sweep_sip2d.py --phase dc --output-dir results/throat_sweep_sip2d_20261014
    python code/scripts/sip_simulation/run_throat_sweep_sip2d.py --phase spectra --output-dir results/throat_sweep_sip2d_20261014
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_ideal_two_pore_channel_sip2d as base  # noqa: E402

from pore_scale_electrical_2d import TwoPoreChain2D, relaxation_decomposition  # noqa: E402

DEFAULT_CONFIG = base.PROJECT_ROOT / "configs" / "sip_simulation" / "throat_sweep_niu2020_sip2d_config.py"
THROAT_LABELS = (3, 4)


def case_name(width_m: float, length_m: float) -> str:
    return f"w{width_m * 1e6:g}_L{length_m * 1e6:g}"


def build_case(config: dict, width_m: float, length_m: float, coarse_step_m: float | None = None):
    g, gr = config["geometry"], config["grid"]
    step = coarse_step_m or gr["coarse_step_m"]
    chain = TwoPoreChain2D(g["big_radius_m"], g["small_radius_m"], width_m, length_m, g["solid_margin_m"])
    # At least throat_cells across the throat, never coarser than the background step.
    throat_cells = max(int(gr["throat_cells"]), int(math.ceil(width_m / step - 1e-9)))
    grid = chain.grid(step, throat_cells, gr["growth"])
    return chain, grid, chain.region_map(grid)


def dc_row(chain, grid, labels, params) -> tuple[dict, object, dict]:
    dc, _, global_lambda, regional, table = base.dc_field_analysis(chain, grid, labels, params.water_conductivity_s_m)
    total = table.loc[table["label"] == 0].iloc[0]
    throat = table[table["label"].isin(THROAT_LABELS)]
    throat_area = float(throat["pore_area_m2"].sum())
    w = chain.throat_width_m
    row = {
        "width_um": w * 1e6,
        "length_um": chain.throat_length_m * 1e6,
        "throat_length_mesh_um": 0.5 * throat_area / w * 1e6,  # two throats per period
        "period_um": chain.period_m * 1e6,
        "height_um": chain.height_m * 1e6,
        "grid_ny": grid.shape[0],
        "grid_nx": grid.shape[1],
        "sigma_dc_s_m": dc.effective_conductivity_s_m.real,
        "formation_factor": params.water_conductivity_s_m / dc.effective_conductivity_s_m.real,
        "dc_relative_residual": dc.residual_norm,
        "lambda_global_um": global_lambda.dynamic_pore_size_m * 1e6,
        "lambda_big_um": regional[1].dynamic_pore_size_m * 1e6,
        "lambda_small_um": regional[2].dynamic_pore_size_m * 1e6,
        "lambda_throat_um": regional[3].dynamic_pore_size_m * 1e6,
        "throat_area_fraction": throat_area / float(total["pore_area_m2"]),
        "throat_field_energy_fraction": float(throat["area_field_energy_v2"].sum()) / float(total["area_field_energy_v2"]),
        "throat_wall_energy_fraction": float(throat["wall_tangential_energy_v2_per_m"].sum()) / float(total["wall_tangential_energy_v2_per_m"]),
        "big_field_energy_fraction": float(table.loc[table["label"] == 1, "area_field_energy_v2"].iloc[0]) / float(total["area_field_energy_v2"]),
        "small_field_energy_fraction": float(table.loc[table["label"] == 2, "area_field_energy_v2"].iloc[0]) / float(total["area_field_energy_v2"]),
    }
    return row, global_lambda, regional


def lab_band_analysis(frame: pd.DataFrame, config: dict, params) -> dict:
    lo, hi = config["sweep"]["lab_band_hz"]
    sel = (frame["frequency_hz"] >= lo * (1 - 1e-9)) & (frame["frequency_hz"] <= hi * (1 + 1e-9))
    f = frame.loc[sel, "frequency_hz"].to_numpy()
    sigma = frame.loc[sel, "sigma_real_s_m"].to_numpy() + 1j * frame.loc[sel, "sigma_imag_s_m"].to_numpy()
    f0 = np.logspace(math.log10(lo) - 1, math.log10(hi) + 1, 10 * int(round(math.log10(hi / lo) + 2)) + 1)
    tau = 1.0 / (2.0 * np.pi * f0)
    fit = relaxation_decomposition(f, sigma, {"debye": (tau, "debye")}, config["debye"]["regularization"])
    amp = fit.amplitudes_s_m["debye"]
    d = config["debye"]["radius_diffusion_m2_s"]
    radius_um = np.sqrt(2.0 * tau * d) * 1e6
    imag_pol = fit.prediction.imag - 2.0 * np.pi * f * fit.epsilon_inf_f_m
    f_peak = float(f[np.argmax(imag_pol)])
    delta = float(amp.sum())
    formation = params.water_conductivity_s_m / fit.sigma_0_s_m
    peaks = sorted(base.local_peaks(radius_um, amp), key=lambda p: -p["amplitude_fraction"])
    return {
        "f_peak_hz": f_peak,
        "r_peak_um": math.sqrt(d / (math.pi * f_peak)) * 1e6,
        "dominant_debye_radius_um": peaks[0]["radius_um"] if peaks else math.nan,
        "dominant_debye_fraction": peaks[0]["amplitude_fraction"] if peaks else math.nan,
        "n_debye_peaks": len(peaks),
        "debye_peaks": peaks,
        "sigma0_s_m": fit.sigma_0_s_m,
        "delta_sigma_s_m": delta,
        "formation_factor_fit": formation,
        "lambda_amp_um": 2.0 * params.surface_conductance_s / (formation * delta) * 1e6,
        "imag_rmse_over_max": fit.imag_rmse_s_m / float(np.max(np.abs(sigma.imag))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=base.PROJECT_ROOT / "results" / "throat_sweep_sip2d_20261014")
    parser.add_argument("--phase", choices=("dc", "spectra"), required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--grid-check", action="store_true", help="dc phase: also solve the extreme cases on a 2x finer grid")
    args = parser.parse_args()

    config = base.load_config(args.config)
    if args.workers is not None:
        config["sweep"]["workers"] = args.workers
    params = base.build_parameters(config)
    out = args.output_dir.resolve()
    for sub in ("configs", "cases", "source_data", "provenance"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out / "configs" / args.config.name)
    (out / "configs" / "resolved_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    sw = config["sweep"]
    cases = [(w, L) for w in sw["widths_m"] for L in sw["lengths_m"]]
    t0 = time.perf_counter()

    if args.phase == "dc":
        rows = []
        for w, L in cases:
            chain, grid, labels = build_case(config, w, L)
            row, _, _ = dc_row(chain, grid, labels, params)
            rows.append({"case": case_name(w, L), **row})
            print(f"{case_name(w, L)}: grid {grid.shape}, F = {row['formation_factor']:.2f}, Lambda = {row['lambda_global_um']:.3f} um, "
                  f"throat energy {100 * row['throat_field_energy_fraction']:.1f}% ({time.perf_counter() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(out / "source_data" / "dc_summary.csv", index=False)
        if args.grid_check:
            checks = []
            for w, L in [(0.1e-6, 2e-6), (0.1e-6, 30e-6), (4e-6, 2e-6), (1e-6, 15e-6)]:
                for step in (config["grid"]["coarse_step_m"], 0.5 * config["grid"]["coarse_step_m"]):
                    chain, grid, labels = build_case(config, w, L, step)
                    row, _, _ = dc_row(chain, grid, labels, params)
                    checks.append({"case": case_name(w, L), "coarse_step_um": step * 1e6, **row})
                    print(f"  grid check {case_name(w, L)} step {step*1e6:.4f} um: F = {row['formation_factor']:.3f}, Lambda = {row['lambda_global_um']:.4f}", flush=True)
            pd.DataFrame(checks).to_csv(out / "source_data" / "dc_grid_check.csv", index=False)
    else:
        frequencies = np.logspace(
            math.log10(sw["frequency_min_hz"]), math.log10(sw["frequency_max_hz"]),
            int(round(sw["points_per_decade"] * math.log10(sw["frequency_max_hz"] / sw["frequency_min_hz"]))) + 1,
        )
        rows = []
        for w, L in cases:
            name = case_name(w, L)
            case_dir = out / "cases" / name
            done = all((case_dir / "simulation_sweeps" / r["name"] / "sweep_results.csv").exists() for r in sw["runs"])
            chain, grid, labels = build_case(config, w, L)
            dc, global_lambda, regional = dc_row(chain, grid, labels, params)
            if done:  # resume: reuse finished cases
                frames = {r["name"]: pd.read_csv(case_dir / "simulation_sweeps" / r["name"] / "sweep_results.csv") for r in sw["runs"]}
            else:
                regions = base.region_specs(chain, regional, config)
                frames = base.run_sweeps(config, params, regions, labels, grid, frequencies, global_lambda, case_dir)
            for run_name, frame in frames.items():
                res = lab_band_analysis(frame, config, params)
                rows.append({"case": name, "run": run_name, "width_um": w * 1e6, "length_um": L * 1e6,
                             "formation_factor_dc": dc["formation_factor"], "lambda_global_um": dc["lambda_global_um"],
                             "max_relative_residual": float(frame["relative_residual"].max()),
                             **{k: v for k, v in res.items() if k != "debye_peaks"},
                             "debye_peaks_json": json.dumps(res["debye_peaks"])})
            print(f"{name}: spectra done ({time.perf_counter() - t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(out / "source_data" / "spectra_summary.csv", index=False)

    manifest_path = out / "provenance" / f"run_manifest_{args.phase}.json"
    script = Path(__file__).resolve()
    manifest = {
        "status": "complete",
        "phase": args.phase,
        "model_dimension": "2D (per unit depth); periodic chain B - throat - S - throat - B along x",
        "command": " ".join([Path(sys.executable).name, *sys.argv]),
        "cases_um": [[w * 1e6, L * 1e6] for w, L in cases],
        "parameters": {**config["electrical"], **config["polarization"]},
        "grid": config["grid"],
        "seconds": time.perf_counter() - t0,
        "provenance": {
            "python": sys.version,
            "script": str(script),
            "script_sha256": base.sha256(script),
            "config": str(args.config.resolve()),
            "config_sha256": base.sha256(args.config),
            "solver_package_sha256": {p.name: base.sha256(p) for p in sorted((base.PROJECT_ROOT / "code" / "src" / "pore_scale_electrical_2d").glob("*.py"))},
            "git_commit": base.git_commit(base.PROJECT_ROOT),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({"status": "complete", "phase": args.phase, "output_dir": str(out), "seconds": round(manifest["seconds"])}), flush=True)


if __name__ == "__main__":
    main()
