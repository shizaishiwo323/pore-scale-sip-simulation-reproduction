from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "pore_network" / "audit_throat_length_definitions.py"


def load_module():
    spec = importlib.util.spec_from_file_location("audit_throat_length_definitions", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compute_length_definitions_adds_export_center_overlap_alpha_and_volume_lengths():
    module = load_module()
    pores = pd.DataFrame(
        {
            "pore_id": [1, 2],
            "pore_radius_m": [2.0e-6, 1.0e-6],
            "pore_volume_m3": [1.0e-15, 1.0e-15],
        }
    )
    throats = pd.DataFrame(
        {
            "throat_id": [10],
            "pore1_id": [1],
            "pore2_id": [2],
            "pore_center_to_center_length_m": [4.0e-6],
            "pore1_length_m": [1.0e-6],
            "pore2_length_m": [1.0e-6],
            "throat_length_m": [2.0e-6],
            "throat_radius_m": [1.0e-6],
            "throat_volume_m3": [np.pi * 1.0e-12 * 3.0e-6],
        }
    )

    definitions = module.compute_length_definitions(pores, throats, alpha_values=[0.5, 1.0], epsilon_m=1.0e-12)

    assert list(definitions["throat_id"]) == [10]
    assert definitions.loc[0, "exported_m"] == 2.0e-6
    assert definitions.loc[0, "center_m"] == 4.0e-6
    assert definitions.loc[0, "center_minus_rp1_rp2_m"] == 1.0e-6
    assert np.isclose(definitions.loc[0, "center_minus_alpha_0p5_radii_m"], 2.5e-6)
    assert np.isclose(definitions.loc[0, "center_minus_alpha_1_radii_m"], 1.0e-6)
    assert np.isclose(definitions.loc[0, "volume_over_circular_area_m"], 3.0e-6)


def test_evaluate_length_definitions_reports_short_tail_and_l1_against_paper():
    module = load_module()
    paper = pd.DataFrame(
        {
            "center_m": [0.5e-6, 5.0e-6, 20.0e-6],
            "relative_volume": [0.25, 0.0, 0.75],
        }
    )
    definitions = pd.DataFrame(
        {
            "throat_id": [1, 2],
            "throat_volume_m3": [1.0, 3.0],
            "exported_m": [0.5e-6, 20.0e-6],
            "center_m": [5.0e-6, 20.0e-6],
        }
    )

    summary, comparison = module.evaluate_length_definitions(paper, definitions, ["exported_m", "center_m"])

    exported = summary.set_index("definition").loc["exported_m"]
    center = summary.set_index("definition").loc["center_m"]
    assert exported["candidate_lt_1um_fraction"] == 0.25
    assert exported["candidate_lt_5um_fraction"] == 0.25
    assert exported["candidate_lt_10um_fraction"] == 0.25
    assert center["candidate_lt_1um_fraction"] == 0.0
    assert center["candidate_lt_5um_fraction"] == 0.0
    assert center["candidate_lt_10um_fraction"] == 0.25
    assert exported["l1_distance"] < center["l1_distance"]
    assert set(comparison["definition"]) == {"exported_m", "center_m"}
