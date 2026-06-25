from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "sip_simulation" / "summarize_ac3d_directional_sweeps.py"


def load_module():
    spec = importlib.util.spec_from_file_location("summarize_ac3d_directional_sweeps", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_sweep(path: Path, direction: str, real_values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "frequency_hz": [1.0, 10.0],
            "requested_frequency_hz": [1.0, 10.0],
            "direction": [direction, direction],
            "effective_sigma_real_s_m": real_values,
            "effective_sigma_imag_s_m": [value * 0.1 for value in real_values],
            "true_residual_norm": [1e-8, 2e-8],
            "true_residual_passed": [True, True],
            "frequency_match_mode": ["exact", "exact"],
        }
    ).to_csv(path, index=False)


def test_summarizes_directional_mean_and_anisotropy(tmp_path):
    module = load_module()
    x = tmp_path / "x" / "sweep_results.csv"
    y = tmp_path / "y" / "sweep_results.csv"
    z = tmp_path / "z" / "sweep_results.csv"
    write_sweep(x, "x", [1.0, 2.0])
    write_sweep(y, "y", [2.0, 4.0])
    write_sweep(z, "z", [3.0, 6.0])

    summary = module.summarize_directional_sweeps({"x": x, "y": y, "z": z})

    assert summary["directions"] == ["x", "y", "z"]
    frame = summary["frame"]
    assert np.allclose(frame["directional_mean_real_s_m"], [2.0, 4.0])
    assert np.allclose(frame["anisotropy_ratio_real"], [3.0, 3.0])
    assert frame["all_true_residual_passed"].tolist() == [True, True]


def test_directional_summary_rejects_mismatched_frequencies(tmp_path):
    module = load_module()
    x = tmp_path / "x" / "sweep_results.csv"
    y = tmp_path / "y" / "sweep_results.csv"
    z = tmp_path / "z" / "sweep_results.csv"
    write_sweep(x, "x", [1.0, 2.0])
    write_sweep(y, "y", [2.0, 4.0])
    write_sweep(z, "z", [3.0, 6.0])
    frame = pd.read_csv(z)
    frame.loc[1, "frequency_hz"] = 11.0
    frame.to_csv(z, index=False)

    try:
        module.summarize_directional_sweeps({"x": x, "y": y, "z": z})
    except ValueError as exc:
        assert "same exact frequency grid" in str(exc)
    else:
        raise AssertionError("expected mismatched frequency grids to fail")
