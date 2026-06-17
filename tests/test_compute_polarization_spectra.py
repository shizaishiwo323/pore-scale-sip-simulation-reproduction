import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))
sys.path.insert(0, str(ROOT / "code" / "scripts" / "sip_simulation"))

from compute_polarization_spectra import compute_spectra  # noqa: E402
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


def test_compute_spectra_rejects_pore_radius_scale():
    params = PolarizationParameters(diffusion_coefficient_m2_s=1.0e-9)
    pores = pd.DataFrame({"radius_m": [1.0e-6], "weight": [1.0]})
    throats = pd.DataFrame({"length_m": [10.0e-6], "weight": [1.0], "zdc_ohm": [100.0]})

    with pytest.raises(ValueError, match="pore_radius_scale"):
        compute_spectra(np.array([1.0]), pores, throats, params, None, pore_radius_scale=2.0)
