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
