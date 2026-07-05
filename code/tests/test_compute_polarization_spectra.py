from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "sip_simulation"
sys.path.insert(0, str(SCRIPT_DIR))


def test_pore_radius_scale_is_rejected_in_production_spectra():
    from compute_polarization_spectra import compute_spectra
    from pore_scale_electrical.polarization import PolarizationParameters

    frequency_hz = np.logspace(-1, 3, 80)
    pores = pd.DataFrame({"radius_m": [10.0e-6], "weight": [1.0]})
    throats = pd.DataFrame({"length_m": [1.0e-6], "weight": [1.0], "zdc_ohm": [1.0e6]})
    params = PolarizationParameters(surface_conductance_s=1.0e-10, diffusion_coefficient_m2_s=1.0e-9)

    with pytest.raises(ValueError, match="pore_radius_scale"):
        compute_spectra(
            frequency_hz,
            pores,
            throats,
            params,
            figure5_zdc_ohm=None,
            pore_radius_scale=0.5,
        )


def test_project_extracted_mode_requires_dynamic_pore_size_manifest():
    from compute_polarization_spectra import resolve_polarization_parameters

    with pytest.raises(ValueError, match="dynamic pore size manifest"):
        resolve_polarization_parameters("project-extracted", None)


def test_project_extracted_mode_uses_manifest_lambda(tmp_path):
    from compute_polarization_spectra import resolve_polarization_parameters

    manifest = tmp_path / "dynamic_pore_size.json"
    manifest.write_text(
        '{"lambda_iso_m": 4.2e-6, "paper_reference_lambda_m": 2.7e-6, '
        '"paper_reference_used_as_project_value": false}',
        encoding="utf-8",
    )

    params, metadata = resolve_polarization_parameters("project-extracted", manifest)

    assert params.dynamic_pore_size_m == pytest.approx(4.2e-6)
    assert metadata["dynamic_pore_size_source"] == "project_extracted_microct_laplace_field"
    assert metadata["paper_reference_used_as_project_value"] is False
