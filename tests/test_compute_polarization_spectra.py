import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "code" / "scripts" / "sip_simulation" / "compute_polarization_spectra.py"
sys.path.insert(0, str(ROOT / "code" / "src"))
sys.path.insert(0, str(ROOT / "code" / "scripts" / "sip_simulation"))

from compute_polarization_spectra import compute_spectra  # noqa: E402
import compute_polarization_spectra as spectra_module  # noqa: E402
from pore_scale_electrical.polarization import PolarizationParameters  # noqa: E402


def test_compute_spectra_rejects_membrane_geometry_scales():
    params = PolarizationParameters(diffusion_coefficient_m2_s=1.0e-9)
    pores = pd.DataFrame({"radius_m": [1.0e-6], "weight": [1.0]})
    throats = pd.DataFrame({"length_m": [10.0e-6], "weight": [1.0], "zdc_ohm": [100.0]})

    with pytest.raises(ValueError, match="Re-extract the pore network"):
        compute_spectra(
            np.array([1.0]),
            pores,
            throats,
            params,
            figure5_zdc_ohm=None,
            membrane_length_scale=0.25,
        )

    with pytest.raises(ValueError, match="geometry-derived resistance"):
        compute_spectra(
            np.array([1.0]),
            pores,
            throats,
            params,
            figure5_zdc_ohm=None,
            membrane_zdc_scale=4.0,
        )


def test_membrane_uses_extracted_geometry_without_adjustment():
    params = PolarizationParameters(diffusion_coefficient_m2_s=1.0e-9)
    pores = pd.DataFrame({"radius_m": [1.0e-6], "weight": [1.0]})
    throats = pd.DataFrame({"length_m": [20.0e-6], "weight": [1.0], "zdc_ohm": [100.0]})
    frequencies = np.logspace(-1, 8, 901)

    base, metadata = compute_spectra(frequencies, pores, throats, params, None)

    base_peak = base.loc[base["delta_sigma_membrane_imag_s_m"].idxmax(), "frequency_hz"]
    assert base_peak > 0
    assert metadata["membrane_length_scale"] == 1.0
    assert metadata["membrane_zdc_scale"] == 1.0
    assert metadata["membrane_effective_length_m"]["min"] == 20.0e-6
    assert metadata["spectrum_quantity"] == "water_phase_polarization_increment"
    assert metadata["effective_conductivity_requires_full_grid_solve"] is True


def test_membrane_weight_mode_can_shift_peak_without_geometry_scaling():
    params = PolarizationParameters(diffusion_coefficient_m2_s=1.3e-9)
    pores = pd.DataFrame({"radius_m": [1.0e-6], "weight": [1.0]})
    throats = pd.DataFrame(
        {
            "length_m": [0.14e-6, 20.0e-6],
            "weight": [1.0, 1000.0],
            "zdc_ohm": [100.0, 100.0],
        }
    )
    frequencies = np.logspace(0, 6, 241)

    volume, volume_metadata = compute_spectra(frequencies, pores, throats, params, None)
    length_density, length_density_metadata = compute_spectra(
        frequencies,
        pores,
        throats,
        params,
        None,
        membrane_weight_mode="volume_over_length2",
    )

    volume_peak = float(volume.loc[volume["delta_sigma_membrane_imag_s_m"].idxmax(), "frequency_hz"])
    length_density_peak = float(
        length_density.loc[length_density["delta_sigma_membrane_imag_s_m"].idxmax(), "frequency_hz"]
    )

    assert volume_peak < 100.0
    assert length_density_peak > 1.0e4
    assert volume_metadata["membrane_weight_mode"] == "volume"
    assert length_density_metadata["membrane_weight_mode"] == "volume_over_length2"
    assert length_density_metadata["membrane_length_scale"] == 1.0
    assert length_density_metadata["membrane_zdc_scale"] == 1.0


def test_volume_linear_density_is_explicit_alias_for_volume_over_length():
    params = PolarizationParameters(diffusion_coefficient_m2_s=1.3e-9)
    pores = pd.DataFrame({"radius_m": [1.0e-6], "weight": [1.0]})
    throats = pd.DataFrame(
        {
            "length_m": [0.14e-6, 20.0e-6],
            "weight": [1.0, 1000.0],
            "zdc_ohm": [100.0, 100.0],
        }
    )
    frequencies = np.logspace(0, 6, 61)

    alias, alias_metadata = compute_spectra(
        frequencies, pores, throats, params, None, membrane_weight_mode="volume_linear_density"
    )
    legacy_name, legacy_metadata = compute_spectra(
        frequencies, pores, throats, params, None, membrane_weight_mode="volume_over_length"
    )

    assert np.allclose(
        alias["delta_sigma_membrane_imag_s_m"],
        legacy_name["delta_sigma_membrane_imag_s_m"],
    )
    assert alias_metadata["membrane_weight_mode"] == "volume_density_renormalized_diagnostic"
    assert alias_metadata["membrane_weight_mode_requested"] == "volume_linear_density"
    assert legacy_metadata["membrane_weight_mode"] == "volume_over_length"


def test_density_renormalized_alias_documents_diagnostic_weighting():
    params = PolarizationParameters(diffusion_coefficient_m2_s=1.3e-9)
    pores = pd.DataFrame({"radius_m": [1.0e-6], "weight": [1.0]})
    throats = pd.DataFrame(
        {
            "length_m": [0.14e-6, 20.0e-6],
            "weight": [1.0, 1000.0],
            "zdc_ohm": [100.0, 100.0],
        }
    )
    frequencies = np.logspace(0, 6, 61)

    diagnostic, metadata = compute_spectra(
        frequencies, pores, throats, params, None, membrane_weight_mode="volume_density_renormalized_diagnostic"
    )
    alias, _ = compute_spectra(
        frequencies, pores, throats, params, None, membrane_weight_mode="volume_linear_density"
    )

    assert np.allclose(diagnostic["delta_sigma_membrane_imag_s_m"], alias["delta_sigma_membrane_imag_s_m"])
    assert metadata["membrane_weight_mode"] == "volume_density_renormalized_diagnostic"
    assert metadata["membrane_weight_mode_requested"] == "volume_density_renormalized_diagnostic"
    assert "diagnostic" in metadata["membrane_weight_mode_note"]


def test_zdc_length_mode_uses_geometry_columns_without_scaling():
    params = PolarizationParameters(diffusion_coefficient_m2_s=1.3e-9)
    pores = pd.DataFrame({"radius_m": [1.0e-6], "weight": [1.0]})
    throats = pd.DataFrame(
        {
            "length_m": [0.14e-6],
            "weight": [1.0],
            "zdc_ohm": [100.0],
            "zdc_center_to_center_ohm": [1000.0],
            "zdc_conduit_ohm": [2000.0],
        }
    )
    frequencies = np.array([4.64158883361278e4])

    current, current_metadata = compute_spectra(frequencies, pores, throats, params, None)
    center, center_metadata = compute_spectra(
        frequencies, pores, throats, params, None, membrane_zdc_length_mode="center_to_center"
    )
    conduit, conduit_metadata = compute_spectra(
        frequencies, pores, throats, params, None, membrane_zdc_length_mode="conduit"
    )

    current_imag = float(current["delta_sigma_membrane_imag_s_m"].iloc[0])
    center_imag = float(center["delta_sigma_membrane_imag_s_m"].iloc[0])
    conduit_imag = float(conduit["delta_sigma_membrane_imag_s_m"].iloc[0])

    assert center_imag < current_imag
    assert conduit_imag < center_imag
    assert current_metadata["membrane_zdc_length_mode"] == "throat"
    assert center_metadata["membrane_zdc_length_mode"] == "center_to_center"
    assert conduit_metadata["membrane_zdc_length_mode"] == "conduit"
    assert conduit_metadata["membrane_zdc_scale"] == 1.0


def test_explicit_membrane_relaxation_length_controls_peak_without_scaling():
    params = PolarizationParameters(diffusion_coefficient_m2_s=1.3e-9)
    pores = pd.DataFrame({"radius_m": [1.0e-6], "weight": [1.0]})
    base_throats = pd.DataFrame({"length_m": [20.0e-6], "weight": [1.0], "zdc_ohm": [100.0]})
    explicit_throats = pd.DataFrame(
        {
            "length_m": [20.0e-6],
            "membrane_relaxation_length_m": [0.2e-6],
            "weight": [1.0],
            "zdc_ohm": [100.0],
        }
    )
    frequencies = np.logspace(1, 8, 701)

    base, _base_metadata = compute_spectra(frequencies, pores, base_throats, params, None)
    explicit, metadata = compute_spectra(frequencies, pores, explicit_throats, params, None)

    base_peak = float(base.loc[base["delta_sigma_membrane_imag_s_m"].idxmax(), "frequency_hz"])
    explicit_peak = float(explicit.loc[explicit["delta_sigma_membrane_imag_s_m"].idxmax(), "frequency_hz"])

    assert explicit_peak > base_peak * 1000.0
    assert metadata["membrane_geometry_mode"] == "explicit_membrane_geometry_columns"
    assert metadata["membrane_relaxation_length_m"]["min"] == 0.2e-6
    assert metadata["membrane_length_scale"] == 1.0


def test_load_pnextract_uses_explicit_active_area_for_zdc(tmp_path: Path):
    network_dir = tmp_path / "network"
    network_dir.mkdir()
    params = PolarizationParameters(water_conductivity_s_m=0.05)
    pd.DataFrame(
        {
            "pore_id": [1, 2],
            "pore_radius_m": [1.0e-6, 2.0e-6],
            "pore_volume_m3": [1.0, 2.0],
        }
    ).to_csv(network_dir / "pores.csv", index=False)
    pd.DataFrame(
        {
            "throat_id": [1],
            "pore1_id": [1],
            "pore2_id": [2],
            "throat_length_m": [20.0e-6],
            "membrane_relaxation_length_m": [0.2e-6],
            "membrane_zdc_length_m": [2.0e-6],
            "membrane_active_area_m2": [4.0e-14],
            "throat_radius_m": [10.0e-6],
            "throat_shape_factor": [0.05],
            "throat_volume_m3": [1.0],
        }
    ).to_csv(network_dir / "throats.csv", index=False)

    _pores, throats = spectra_module.load_pnextract(network_dir, params)

    assert throats["length_m"].iloc[0] == 20.0e-6
    assert throats["membrane_relaxation_length_m"].iloc[0] == 0.2e-6
    assert throats["membrane_zdc_length_m"].iloc[0] == 2.0e-6
    assert throats["membrane_active_area_m2"].iloc[0] == 4.0e-14
    assert throats["zdc_ohm"].iloc[0] == pytest.approx(1.0e9)
    assert throats.attrs["membrane_zdc_geometry_source"] == "explicit_membrane_active_area_m2"


def test_compute_spectra_rejects_pore_radius_scale():
    params = PolarizationParameters(diffusion_coefficient_m2_s=1.0e-9)
    pores = pd.DataFrame({"radius_m": [1.0e-6], "weight": [1.0]})
    throats = pd.DataFrame({"length_m": [10.0e-6], "weight": [1.0], "zdc_ohm": [100.0]})

    with pytest.raises(ValueError, match="pore_radius_scale"):
        compute_spectra(np.array([1.0]), pores, throats, params, None, pore_radius_scale=2.0)


def test_compute_spectra_rejects_unknown_membrane_weight_mode_even_without_zdc():
    params = PolarizationParameters(diffusion_coefficient_m2_s=1.0e-9)
    pores = pd.DataFrame({"radius_m": [1.0e-6], "weight": [1.0]})
    throats = pd.DataFrame({"length_m": [10.0e-6], "weight": [1.0]})

    with pytest.raises(ValueError, match="membrane_weight_mode"):
        compute_spectra(np.array([1.0]), pores, throats, params, None, membrane_weight_mode="bad_mode")


def test_load_pnextract_excludes_boundary_throats_by_default(tmp_path: Path):
    network_dir = tmp_path / "network"
    network_dir.mkdir()
    pd.DataFrame(
        {
            "pore_id": [1, 2, 3],
            "pore_radius_m": [1.0e-6, 2.0e-6, 3.0e-6],
            "pore_volume_m3": [1.0, 2.0, 3.0],
        }
    ).to_csv(network_dir / "pores.csv", index=False)
    pd.DataFrame(
        {
            "throat_id": [1, 2, 3],
            "pore1_id": [-1, 1, 2],
            "pore2_id": [1, 2, 3],
            "throat_length_m": [0.5e-6, 1.0e-6, 2.0e-6],
            "throat_radius_m": [10.0e-6, 0.2e-6, 0.3e-6],
            "throat_shape_factor": [0.05, 0.05, 0.05],
            "throat_volume_m3": [100.0, 1.0, 2.0],
        }
    ).to_csv(network_dir / "throats.csv", index=False)

    pores, throats = spectra_module.load_pnextract(network_dir, PolarizationParameters())
    spectra, metadata = compute_spectra(np.array([1.0]), pores, throats, PolarizationParameters(), None)

    assert len(throats) == 2
    assert throats["weight"].to_list() == [1.0, 2.0]
    assert throats.attrs["pnextract_excluded_boundary_throat_count"] == 1
    assert metadata["pnextract_excluded_boundary_throat_count"] == 1
    assert metadata["membrane_geometry_mode"] == "single_throat_geometry"
    assert spectra["delta_sigma_membrane_imag_s_m"].iloc[0] > 0


def test_figure5_cli_default_points_to_project_data(tmp_path):
    assert spectra_module.DEFAULT_FIGURE5_PATH.exists()

    out = tmp_path / "figure5_spectra.csv"
    metadata = tmp_path / "figure5_spectra.metadata.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--source",
            "figure5",
            "--out",
            str(out),
            "--metadata-out",
            str(metadata),
            "--figure5-zdc-ohm",
            "1.0",
        ],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert out.exists()
    assert metadata.exists()


def test_cli_accepts_explicit_frequencies(tmp_path):
    out = tmp_path / "figure5_selected_frequencies.csv"
    metadata = tmp_path / "figure5_selected_frequencies.metadata.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--source",
            "figure5",
            "--out",
            str(out),
            "--metadata-out",
            str(metadata),
            "--figure5-zdc-ohm",
            "1.0",
            "--frequencies",
            "42169.65034285822",
            "46415.8883361278",
        ],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    frame = pd.read_csv(out)
    assert frame["frequency_hz"].to_list() == [42169.65034285822, 46415.8883361278]
