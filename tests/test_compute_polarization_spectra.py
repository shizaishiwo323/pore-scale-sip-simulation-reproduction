import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "src"))
sys.path.insert(0, str(ROOT / "code" / "scripts" / "sip_simulation"))

from compute_polarization_spectra import compute_spectra  # noqa: E402
from pore_scale_electrical.polarization import PolarizationParameters  # noqa: E402


def test_compute_spectra_records_membrane_geometry_scales():
    params = PolarizationParameters(diffusion_coefficient_m2_s=1.0e-9)
    pores = pd.DataFrame({"radius_m": [1.0e-6], "weight": [1.0]})
    throats = pd.DataFrame({"length_m": [10.0e-6], "weight": [1.0], "zdc_ohm": [100.0]})

    spectra, metadata = compute_spectra(
        np.array([1.0]),
        pores,
        throats,
        params,
        figure5_zdc_ohm=None,
        membrane_length_scale=0.25,
        membrane_zdc_scale=4.0,
    )

    assert metadata["membrane_length_scale"] == 0.25
    assert metadata["membrane_zdc_scale"] == 4.0
    assert metadata["membrane_effective_length_m"]["min"] == 2.5e-6
    assert metadata["membrane_effective_zdc_ohm"]["min"] == 400.0
    assert spectra["membrane_effective_length_m_mean"].iloc[0] == 2.5e-6


def test_membrane_length_scale_shifts_peak_by_inverse_square():
    params = PolarizationParameters(diffusion_coefficient_m2_s=1.0e-9)
    pores = pd.DataFrame({"radius_m": [1.0e-6], "weight": [1.0]})
    throats = pd.DataFrame({"length_m": [20.0e-6], "weight": [1.0], "zdc_ohm": [100.0]})
    frequencies = np.logspace(-1, 8, 901)

    base, _ = compute_spectra(frequencies, pores, throats, params, None)
    scaled, _ = compute_spectra(
        frequencies,
        pores,
        throats,
        params,
        None,
        membrane_length_scale=0.1,
    )

    base_peak = base.loc[base["delta_sigma_membrane_imag_s_m"].idxmax(), "frequency_hz"]
    scaled_peak = scaled.loc[scaled["delta_sigma_membrane_imag_s_m"].idxmax(), "frequency_hz"]
    assert np.isclose(scaled_peak / base_peak, 100.0, rtol=0.06)
