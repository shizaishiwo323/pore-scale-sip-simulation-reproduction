from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "sip_simulation" / "run_ac3d_matrix_free_gpu_sweep.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_ac3d_matrix_free_gpu_sweep", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_formal_frequency_row_requires_exact_match():
    module = load_module()
    spectra = pd.DataFrame(
        {
            "frequency_hz": [1.0, 10.0],
            "apparent_water_sigma_real_s_m": [0.1, 0.2],
            "apparent_water_sigma_imag_s_m": [0.01, 0.02],
        }
    )

    row = module.select_spectrum_row(spectra, 10.0, frequency_match_mode="exact")

    assert float(row["frequency_hz"]) == 10.0
    with pytest.raises(ValueError, match="exactly match"):
        module.select_spectrum_row(spectra, 5.0, frequency_match_mode="exact")


def test_diagnostic_frequency_row_records_nearest_mode():
    module = load_module()
    spectra = pd.DataFrame(
        {
            "frequency_hz": [1.0, 10.0],
            "apparent_water_sigma_real_s_m": [0.1, 0.2],
            "apparent_water_sigma_imag_s_m": [0.01, 0.02],
        }
    )

    row = module.select_spectrum_row(spectra, 5.0, frequency_match_mode="nearest")

    assert float(row["frequency_hz"]) == 10.0


def test_warm_start_reuse_requires_successful_true_residual():
    module = load_module()

    assert module.can_reuse_solution_as_warm_start(info=0, true_residual_passed=True)
    assert not module.can_reuse_solution_as_warm_start(info=800, true_residual_passed=False)
    assert not module.can_reuse_solution_as_warm_start(info=0, true_residual_passed=False)
    assert not module.can_reuse_solution_as_warm_start(info=-20, true_residual_passed=False)


def test_write_residual_history_creates_parent_directories(tmp_path):
    module = load_module()
    out = tmp_path / "nested" / "frequency_000" / "residual_history.csv"

    module.write_residual_history(out, [{"iteration": 1, "relative_residual_norm": 0.1}])

    assert out.exists()
