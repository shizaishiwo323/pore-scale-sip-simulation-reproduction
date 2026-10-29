"""First-order (small-increment) check of the regional 3-D SIP sweeps.

For a small water-phase increment the effective conductivity changes by
``d sigma_eff = (1/V) sum_k d sigma_k(w) int_Vk |E_dc|^2 dV / E0^2`` (dc pore-only field),
and with ``d sigma_k = 2 C*(r_k) / Lambda_k`` this is ``(1/V) sum_k C*(r_k) int_Sk |E_t|^2 dS``.
The region integrals are already in ``source_data/region_table.csv``, so this gives a continuous
spectrum without new field solves. It is a QA reference for the full GPU solves (which include the
field redistribution the increments cause); it is not a simulation result and is not drawn in the
main comparison figure.

Writes ``source_data/first_order_check.csv`` (dense spectrum) and ``first_order_check_at_solved_points.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pore_scale_electrical_2d.polarization import membrane_conductivity_increment, stern_conductance  # noqa: E402
from run_berea_regional_sip3d import DEFAULT_OUTPUT, build_parameters, load_config  # noqa: E402


def first_order(frequency_hz: np.ndarray, table: pd.DataFrame, params, total_volume_m3: float, throat_stern: bool) -> dict:
    used = table[table.voxel_count > 0]
    pore = np.zeros(frequency_hz.shape, dtype=complex)
    membrane = np.zeros(frequency_hz.shape, dtype=complex)
    for row in used.itertuples(index=False):
        weight = row.field_energy_volume_integral_v2_m / total_volume_m3
        stern_on = row.lambda_source != "no_wall_no_stern" and (row.kind == "pore_body" or throat_stern)
        if stern_on:
            pore += 2.0 * stern_conductance(frequency_hz, row.relaxation_radius_m, params) / row.lambda_used_m * weight
        if row.kind == "throat":
            membrane += membrane_conductivity_increment(frequency_hz, row.membrane_length_m, params) * weight
    return {"pore": pore, "membrane": membrane}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    run = Path(args.run)
    run = run if run.is_absolute() else PROJECT_ROOT / run
    config = load_config(next((run / "configs").glob("berea_regional_sip3d_config.py")))
    params = build_parameters(config)
    table = pd.read_csv(run / "source_data" / "region_table.csv")
    dc = json.loads((run / "simulation_sweeps" / "dc_pore_only" / "result.json").read_text(encoding="utf-8"))
    shape = json.loads((run / "pore_network" / "pnextract_input" / "microCT_Berea_prepare_summary.json").read_text(encoding="utf-8"))["shape_zyx"]
    total_volume = float(np.prod(shape)) * config["input"]["voxel_size_m"] ** 3
    throat_stern = bool(config["regions"]["throat_stern_polarization"])

    dense = np.logspace(-3, 9, 241)
    est = first_order(dense, table, params, total_volume, throat_stern)
    pd.DataFrame(
        {
            "frequency_hz": dense,
            "pore_regional_sigma_real_s_m": dc["sigma_eff_s_m"] + est["pore"].real,
            "pore_regional_sigma_imag_s_m": est["pore"].imag,
            "membrane_regional_sigma_real_s_m": dc["sigma_eff_s_m"] + est["membrane"].real,
            "membrane_regional_sigma_imag_s_m": est["membrane"].imag,
        }
    ).to_csv(run / "source_data" / "first_order_check.csv", index=False)

    rows = []
    for name, key in (("pore_regional", "pore"), ("membrane_regional", "membrane")):
        sweep = run / "simulation_sweeps" / name / "sweep_results.csv"
        if not sweep.exists():
            continue
        frame = pd.read_csv(sweep)
        f = frame.frequency_hz.to_numpy()
        point = first_order(f, table, params, total_volume, throat_stern)[key]
        for fi, solved_imag, solved_real, p in zip(f, frame.sigma_imag_s_m, frame.sigma_real_s_m, point):
            rows.append(
                {
                    "run": name,
                    "frequency_hz": fi,
                    "solved_sigma_imag_s_m": solved_imag,
                    "first_order_sigma_imag_s_m": p.imag,
                    "imag_ratio_solved_over_first_order": solved_imag / p.imag if p.imag else np.nan,
                    "solved_delta_sigma_real_s_m": solved_real - dc["sigma_eff_s_m"],
                    "first_order_delta_sigma_real_s_m": p.real,
                }
            )
    check = pd.DataFrame(rows)
    check.to_csv(run / "source_data" / "first_order_check_at_solved_points.csv", index=False)
    if not check.empty:
        print(check.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    peak = dense[np.argmax(est["pore"].imag)]
    print(f"first-order EDL sigma'' peak: {est['pore'].imag.max():.3e} S/m at {peak:.3g} Hz; "
          f"membrane peak {est['membrane'].imag.max():.3e} S/m at {dense[np.argmax(est['membrane'].imag)]:.3g} Hz")


if __name__ == "__main__":
    main()
