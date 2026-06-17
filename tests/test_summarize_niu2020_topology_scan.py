from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "pore_network" / "summarize_niu2020_topology_scan.py"


def load_module():
    spec = importlib.util.spec_from_file_location("summarize_niu2020_topology_scan", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_topology_summary_identifies_short_tail_failure(tmp_path: Path) -> None:
    module = load_module()
    scan_csv = tmp_path / "scan_metrics.csv"
    pd.DataFrame(
        [
            {
                "candidate": "a",
                "status": "ok",
                "pore_l1": 0.2,
                "throat_l1": 0.4,
                "objective_score": 2.0,
                "pnextract_lines": "[]",
                "paper_throat_volume_fraction_lt_1um": 0.05,
                "paper_throat_volume_fraction_lt_5um": 0.10,
                "paper_throat_volume_fraction_lt_10um": 0.20,
                "candidate_throat_volume_fraction_lt_1um": 0.001,
                "candidate_throat_volume_fraction_lt_5um": 0.02,
                "candidate_throat_volume_fraction_lt_10um": 0.10,
            },
            {
                "candidate": "b",
                "status": "ok",
                "pore_l1": 0.3,
                "throat_l1": 0.5,
                "objective_score": 1.0,
                "pnextract_lines": "[]",
                "paper_throat_volume_fraction_lt_1um": 0.05,
                "paper_throat_volume_fraction_lt_5um": 0.10,
                "paper_throat_volume_fraction_lt_10um": 0.20,
                "candidate_throat_volume_fraction_lt_1um": 0.002,
                "candidate_throat_volume_fraction_lt_5um": 0.03,
                "candidate_throat_volume_fraction_lt_10um": 0.11,
            },
        ]
    ).to_csv(scan_csv, index=False)

    top, summary = module.summarize(module.read_scan(scan_csv))

    assert summary["successful_candidates"] == 2
    assert summary["best_objective_candidate"] == "b"
    assert summary["best_throat_l1_candidate"] == "a"
    assert summary["topology_parameters_recover_niu_short_tail"] is False
    assert top["basis"].str.contains("max_lt_1um").any()
