from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "pore_network" / "diagnose_short_throat_deficit.py"


def load_module():
    spec = importlib.util.spec_from_file_location("diagnose_short_throat_deficit", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bin_deficit_reports_missing_short_volume_without_double_counting():
    module = load_module()
    centers = np.array([0.5e-6, 5.0e-6, 20.0e-6])
    paper = np.array([0.20, 0.30, 0.50])
    candidate = np.array([0.05, 0.10, 0.85])

    deficit = module.bin_deficit_table(centers, paper, candidate)

    assert np.isclose(deficit["positive_deficit_fraction"].sum(), 0.35)
    assert np.isclose(deficit.loc[0, "positive_deficit_fraction"], 0.15)
    assert np.isclose(deficit.loc[1, "positive_deficit_fraction"], 0.20)
    assert np.isclose(deficit.loc[2, "excess_fraction"], 0.35)


def test_reweight_to_paper_bins_preserves_total_weight_and_matches_reachable_bins():
    module = load_module()
    centers = np.array([0.5e-6, 5.0e-6, 20.0e-6])
    lengths = np.array([0.5e-6, 5.0e-6, 20.0e-6, 20.0e-6])
    weights = np.array([1.0, 1.0, 4.0, 4.0])
    paper = np.array([0.20, 0.30, 0.50])

    adjusted, details = module.reweight_to_paper_bins(lengths, weights, centers, paper)

    assert np.isclose(adjusted.sum(), weights.sum())
    adjusted_dist = module.distribution_on_centers(lengths, adjusted, centers)
    assert np.allclose(adjusted_dist, paper)
    assert details["unreachable_paper_fraction"] == 0.0


def test_relaxation_frequency_uses_niu_membrane_tau_definition():
    module = load_module()
    lengths = np.array([10e-6])
    freq = module.relaxation_frequency_hz(lengths, diffusion_coefficient_m2_s=1.3e-9)
    expected = 1.0 / (2.0 * np.pi * (lengths[0] ** 2 / (4.0 * 1.3e-9)))
    assert np.isclose(freq[0], expected)


def test_membrane_proxy_metrics_handles_invalid_throats_before_conductance_fractions():
    module = load_module()
    params = module.PolarizationParameters()
    lengths = np.array([0.0, 0.5e-6, 20e-6])
    weights = np.array([1.0, 2.0, 8.0])
    zdc = np.array([1.0, 2.0, 4.0])

    metrics = module.membrane_proxy_metrics("case", lengths, weights, zdc, params)

    assert metrics["valid_throats"] == 2
    assert np.isfinite(metrics["conductance_proxy_fraction_lt_1um"])
    assert metrics["conductance_proxy_fraction_lt_1um"] > 0.0
