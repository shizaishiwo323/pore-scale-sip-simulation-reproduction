from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "sip_simulation" / "verify_niu2020_figure7_style_provenance.py"


def load_module():
    spec = importlib.util.spec_from_file_location("verify_niu2020_figure7_style_provenance", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_verify_component_source_requires_exact_sweep_values(tmp_path):
    module = load_module()
    module.MECHANISMS = ["all"]
    sweep = tmp_path / "all.csv"
    frequency = 1.0
    imag = 2.0e-5
    pd.DataFrame(
        {
            "frequency_hz": [frequency],
            "effective_sigma_real_s_m": [0.003],
            "effective_sigma_imag_s_m": [imag],
        }
    ).to_csv(sweep, index=False)
    omega = 2.0 * np.pi * frequency
    source = pd.DataFrame(
        {
            "dataset": ["our_corrected_all"],
            "frequency_hz": [frequency],
            "real_conductivity_s_m": [0.003],
            "imaginary_conductivity_s_m": [imag],
            "imaginary_conductivity_magnitude_s_m": [imag],
            "real_relative_permittivity": [imag / (omega * module.EPSILON0_F_M)],
            "real_relative_permittivity_magnitude": [imag / (omega * module.EPSILON0_F_M)],
        }
    )

    records = module.verify_component_source(source, {"all": sweep})

    assert records == [{"dataset": "our_corrected_all", "source_csv": str(sweep), "rows": 1, "matches_sweep_csv": True}]
    source.loc[0, "imaginary_conductivity_s_m"] = imag * 2.0
    with pytest.raises(AssertionError):
        module.verify_component_source(source, {"all": sweep})
