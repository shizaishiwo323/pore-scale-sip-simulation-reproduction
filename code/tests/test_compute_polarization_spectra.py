from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "sip_simulation"
sys.path.insert(0, str(SCRIPT_DIR))


def test_pore_radius_scale_shifts_pore_polarization_peak_and_is_recorded():
    from compute_polarization_spectra import compute_spectra
    from pore_scale_electrical.polarization import PolarizationParameters

    frequency_hz = np.logspace(-1, 3, 80)
    pores = pd.DataFrame({"radius_m": [10.0e-6], "weight": [1.0]})
    throats = pd.DataFrame({"length_m": [1.0e-6], "weight": [1.0], "zdc_ohm": [1.0e6]})
    params = PolarizationParameters(surface_conductance_s=1.0e-10, diffusion_coefficient_m2_s=1.0e-9)

    unscaled, _ = compute_spectra(frequency_hz, pores, throats, params, figure5_zdc_ohm=None)
    scaled, metadata = compute_spectra(
        frequency_hz,
        pores,
        throats,
        params,
        figure5_zdc_ohm=None,
        pore_radius_scale=0.5,
    )

    unscaled_peak = float(unscaled.loc[unscaled["delta_sigma_pore_imag_s_m"].idxmax(), "frequency_hz"])
    scaled_peak = float(scaled.loc[scaled["delta_sigma_pore_imag_s_m"].idxmax(), "frequency_hz"])

    assert scaled_peak > unscaled_peak
    assert metadata["pore_radius_scale"] == 0.5
    assert metadata["pore_effective_radius_m"]["mean"] == 5.0e-6
