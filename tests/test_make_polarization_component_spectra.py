from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "sip_simulation" / "make_polarization_component_spectra.py"


def load_module():
    spec = importlib.util.spec_from_file_location("make_polarization_component_spectra", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_make_diagnostic_all_spectrum_replaces_only_membrane_increment():
    module = load_module()
    params = module.PolarizationParameters()
    base = pd.DataFrame(
        {
            "frequency_hz": [1.0, 10.0],
            "omega_rad_s": [2.0 * np.pi, 20.0 * np.pi],
            "delta_sigma_pore_real_s_m": [1.0e-4, 2.0e-4],
            "delta_sigma_pore_imag_s_m": [3.0e-5, 4.0e-5],
            "delta_sigma_membrane_real_s_m": [9.0e-4, 9.0e-4],
            "delta_sigma_membrane_imag_s_m": [9.0e-5, 9.0e-5],
        }
    )
    diagnostic_membrane = pd.DataFrame(
        {
            "frequency_hz": [1.0, 10.0],
            "omega_rad_s": [2.0 * np.pi, 20.0 * np.pi],
            "delta_sigma_membrane_real_s_m": [5.0e-4, 6.0e-4],
            "delta_sigma_membrane_imag_s_m": [7.0e-5, 8.0e-5],
            "membrane_geometry_mode": ["diagnostic_two_length_active_passive_volume_area"] * 2,
            "passive_area_source": ["volume_over_passive_length_area"] * 2,
            "passive_length_source": ["pore1_plus_pore2"] * 2,
            "passive_weight_source": ["volume_over_passive_length"] * 2,
            "passive_zdc_source": ["volume_over_passive_length_area"] * 2,
            "passive_branch_alpha": [0.0163545, 0.0163545],
            "passive_branch_alpha_median": [0.016, 0.016],
            "passive_branch_alpha_mean": [0.02, 0.02],
            "passive_branch_alpha_min": [0.001, 0.001],
            "passive_branch_alpha_max": [0.1, 0.1],
            "passive_branch_alpha_source": ["titov_geometry_factor_median_transport_number_difference"] * 2,
            "passive_transport_number_difference": [0.022, 0.022],
            "passive_titov_geometry_factor_median": [0.337, 0.337],
            "passive_anion_transport_number_reference": [0.5, 0.5],
            "active_anion_transport_number_inferred": [0.478, 0.478],
            "neutral_passive_mobility_contrast_inferred": [1.092, 1.092],
            "edl_debye_length_m": [5.216e-9, 5.216e-9],
            "edl_thickness_multiplier": [38.35, 38.35],
            "edl_selectivity": [1.0, 1.0],
            "passive_transport_number_edl_limited_fraction": [0.0034, 0.0034],
            "edl_selection_rule": ["smallest_effective_thickness_within_rmse_tolerance"] * 2,
            "edl_selected_peak_normalized_rmse": [0.02534, 0.02534],
        }
    )

    spectrum = module.make_diagnostic_all_spectrum(base, diagnostic_membrane, params)

    assert set(spectrum["component"]) == {"all_diagnostic_volume_area"}
    assert set(spectrum["membrane_geometry_mode"]) == {"diagnostic_two_length_active_passive_volume_area"}
    assert set(spectrum["passive_area_source"]) == {"volume_over_passive_length_area"}
    assert set(spectrum["passive_length_source"]) == {"pore1_plus_pore2"}
    assert set(spectrum["passive_weight_source"]) == {"volume_over_passive_length"}
    assert set(spectrum["passive_zdc_source"]) == {"volume_over_passive_length_area"}
    assert np.allclose(spectrum["passive_branch_alpha"], 0.0163545)
    assert np.allclose(spectrum["passive_branch_alpha_median"], 0.016)
    assert np.allclose(spectrum["passive_branch_alpha_mean"], 0.02)
    assert set(spectrum["passive_branch_alpha_source"]) == {"titov_geometry_factor_median_transport_number_difference"}
    assert np.allclose(spectrum["passive_transport_number_difference"], 0.022)
    assert np.allclose(spectrum["passive_titov_geometry_factor_median"], 0.337)
    assert np.allclose(spectrum["active_anion_transport_number_inferred"], 0.478)
    assert np.allclose(spectrum["neutral_passive_mobility_contrast_inferred"], 1.092)
    assert np.allclose(spectrum["edl_debye_length_m"], 5.216e-9)
    assert np.allclose(spectrum["edl_thickness_multiplier"], 38.35)
    assert np.allclose(spectrum["edl_selectivity"], 1.0)
    assert np.allclose(spectrum["passive_transport_number_edl_limited_fraction"], 0.0034)
    assert set(spectrum["edl_selection_rule"]) == {"smallest_effective_thickness_within_rmse_tolerance"}
    assert np.allclose(spectrum["edl_selected_peak_normalized_rmse"], 0.02534)
    assert np.allclose(
        spectrum["delta_sigma_component_real_s_m"],
        [6.0e-4, 8.0e-4],
    )
    assert np.allclose(
        spectrum["delta_sigma_component_imag_s_m"],
        [1.0e-4, 1.2e-4],
    )
    assert np.allclose(
        spectrum["apparent_water_sigma_real_s_m"],
        params.water_conductivity_s_m + np.array([6.0e-4, 8.0e-4]),
    )
    assert np.allclose(
        spectrum["apparent_water_sigma_imag_s_m"],
        base["omega_rad_s"].to_numpy(dtype=float) * params.water_permittivity_f_m
        + np.array([1.0e-4, 1.2e-4]),
    )
    assert np.allclose(
        spectrum["solid_sigma_imag_s_m"],
        base["omega_rad_s"].to_numpy(dtype=float) * params.solid_permittivity_f_m,
    )


def test_main_writes_diagnostic_all_spectrum_when_membrane_component_is_supplied(tmp_path):
    module = load_module()
    base_csv = tmp_path / "base.csv"
    pd.DataFrame(
        {
            "frequency_hz": [1.0],
            "omega_rad_s": [2.0 * np.pi],
            "delta_sigma_pore_real_s_m": [1.0e-4],
            "delta_sigma_pore_imag_s_m": [3.0e-5],
            "delta_sigma_membrane_real_s_m": [9.0e-4],
            "delta_sigma_membrane_imag_s_m": [9.0e-5],
            "delta_sigma_total_real_s_m": [1.0e-3],
            "delta_sigma_total_imag_s_m": [1.2e-4],
        }
    ).to_csv(base_csv, index=False)
    diagnostic_csv = tmp_path / "diagnostic_membrane.csv"
    pd.DataFrame(
        {
            "frequency_hz": [1.0],
            "omega_rad_s": [2.0 * np.pi],
            "delta_sigma_membrane_real_s_m": [5.0e-4],
            "delta_sigma_membrane_imag_s_m": [7.0e-5],
            "membrane_geometry_mode": ["diagnostic_two_length_active_passive_volume_area"],
            "passive_area_source": ["volume_over_passive_length_area"],
            "passive_length_source": ["pore1_plus_pore2"],
            "passive_weight_source": ["volume_over_passive_length"],
            "passive_zdc_source": ["volume_over_passive_length_area"],
            "passive_branch_alpha": [0.0163545],
            "passive_branch_alpha_median": [0.016],
            "passive_branch_alpha_mean": [0.02],
            "passive_branch_alpha_min": [0.001],
            "passive_branch_alpha_max": [0.1],
            "passive_branch_alpha_source": ["titov_geometry_factor_median_transport_number_difference"],
            "passive_transport_number_difference": [0.022],
            "passive_titov_geometry_factor_median": [0.337],
            "passive_anion_transport_number_reference": [0.5],
            "active_anion_transport_number_inferred": [0.478],
            "neutral_passive_mobility_contrast_inferred": [1.092],
            "edl_debye_length_m": [5.216e-9],
            "edl_thickness_multiplier": [38.35],
            "edl_selectivity": [1.0],
            "passive_transport_number_edl_limited_fraction": [0.0034],
            "edl_selection_rule": ["smallest_effective_thickness_within_rmse_tolerance"],
            "edl_selected_peak_normalized_rmse": [0.02534],
        }
    ).to_csv(diagnostic_csv, index=False)
    out_dir = tmp_path / "components"

    module.main(
        [
            "--input",
            str(base_csv),
            "--out-dir",
            str(out_dir),
            "--components",
            "all",
            "--diagnostic-membrane-component",
            str(diagnostic_csv),
        ]
    )

    spectrum_path = out_dir / "polarization_spectra_all_diagnostic_volume_area.csv"
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    spectrum = pd.read_csv(spectrum_path)
    assert spectrum_path.exists()
    assert metadata["all_diagnostic_volume_area"] == str(spectrum_path)
    assert spectrum["component"].unique().tolist() == ["all_diagnostic_volume_area"]
    assert spectrum["passive_area_source"].unique().tolist() == ["volume_over_passive_length_area"]
    assert metadata["diagnostic_membrane_geometry_mode"] == "diagnostic_two_length_active_passive_volume_area"
    assert metadata["diagnostic_membrane_passive_area_source"] == "volume_over_passive_length_area"
    assert metadata["diagnostic_membrane_passive_branch_alpha_median"] == 0.016
    assert metadata["diagnostic_membrane_passive_branch_alpha_mean"] == 0.02
    assert metadata["diagnostic_membrane_passive_branch_alpha_source"] == "titov_geometry_factor_median_transport_number_difference"
    assert metadata["diagnostic_membrane_passive_transport_number_difference"] == 0.022
    assert metadata["diagnostic_membrane_active_anion_transport_number_inferred"] == 0.478
    assert metadata["diagnostic_membrane_neutral_passive_mobility_contrast_inferred"] == 1.092
    assert metadata["diagnostic_membrane_edl_debye_length_m"] == 5.216e-9
    assert metadata["diagnostic_membrane_edl_thickness_multiplier"] == 38.35
    assert metadata["diagnostic_membrane_edl_selectivity"] == 1.0
    assert metadata["diagnostic_membrane_passive_transport_number_edl_limited_fraction"] == 0.0034
    assert metadata["diagnostic_membrane_edl_selection_rule"] == "smallest_effective_thickness_within_rmse_tolerance"
    assert metadata["diagnostic_membrane_edl_selected_peak_normalized_rmse"] == 0.02534
