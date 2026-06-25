from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "sip_simulation" / "verify_ac3d_precision_checkpoints.py"


def load_module():
    spec = importlib.util.spec_from_file_location("verify_ac3d_precision_checkpoints", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_precision_checkpoint_report_computes_relative_errors(tmp_path):
    module = load_module()
    c64 = tmp_path / "complex64.csv"
    c128 = tmp_path / "complex128.csv"
    pd.DataFrame(
        {
            "frequency_hz": [1e-3, 1.0],
            "effective_sigma_real_s_m": [1.0, 2.0],
            "effective_sigma_imag_s_m": [0.1, 0.2],
            "true_residual_norm": [1e-6, 1e-6],
            "true_residual_passed": [True, True],
        }
    ).to_csv(c64, index=False)
    pd.DataFrame(
        {
            "frequency_hz": [1e-3, 1.0],
            "effective_sigma_real_s_m": [1.01, 2.0],
            "effective_sigma_imag_s_m": [0.101, 0.22],
            "true_residual_norm": [1e-8, 1e-8],
            "true_residual_passed": [True, True],
        }
    ).to_csv(c128, index=False)

    report = module.compare_precision_checkpoints(c64, c128, checkpoint_frequencies=[1e-3, 1.0])

    assert report["checkpoint_frequencies_hz"] == [1e-3, 1.0]
    assert report["passed"] is False
    assert report["max_relative_error_real"] == 0.009900990099
    assert report["max_relative_error_imag"] == 0.090909090909


def test_precision_checkpoint_requires_requested_frequencies(tmp_path):
    module = load_module()
    c64 = tmp_path / "complex64.csv"
    c128 = tmp_path / "complex128.csv"
    pd.DataFrame(
        {"frequency_hz": [1.0], "effective_sigma_real_s_m": [1.0], "effective_sigma_imag_s_m": [0.1]}
    ).to_csv(c64, index=False)
    pd.DataFrame(
        {"frequency_hz": [1.0], "effective_sigma_real_s_m": [1.0], "effective_sigma_imag_s_m": [0.1]}
    ).to_csv(c128, index=False)

    try:
        module.compare_precision_checkpoints(c64, c128, checkpoint_frequencies=[1e-3])
    except ValueError as exc:
        assert "missing checkpoint frequencies" in str(exc)
    else:
        raise AssertionError("expected missing checkpoint frequencies to fail")
