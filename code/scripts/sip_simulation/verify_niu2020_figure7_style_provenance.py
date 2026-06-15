#!/usr/bin/env python3
"""Verify data provenance for the corrected Niu 2020 Figure 7-style plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EPSILON0_F_M = 8.8541878128e-12
MECHANISMS = ["all", "pore", "membrane", "interfacial"]
PAPER_SIMULATION_BLOCKS = {
    "paper_simulation_all": [3, 4, 5],
    "paper_pore": [6, 7, 8],
    "paper_membrane": [9, 10, 11],
    "paper_interfacial": [12, 13, 14],
}


def read_numeric_block(path: Path, cols: list[int], names: list[str]) -> pd.DataFrame:
    table = pd.read_excel(path, sheet_name=0, header=None)
    block = table.iloc[2:, cols].copy()
    block.columns = names
    for name in names:
        block[name] = pd.to_numeric(block[name], errors="coerce")
    return block.dropna(subset=[names[0]]).reset_index(drop=True)


def read_paper_block(path: Path, cols: list[int]) -> pd.DataFrame:
    return read_numeric_block(path, cols, ["frequency_hz", "imaginary_conductivity_s_m", "real_relative_permittivity"])


def expected_source_from_sweep(path: Path, mechanism: str) -> pd.DataFrame:
    sweep = pd.read_csv(path)
    omega = 2.0 * np.pi * sweep["frequency_hz"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "dataset": f"our_corrected_{mechanism}",
            "frequency_hz": sweep["frequency_hz"],
            "real_conductivity_s_m": sweep["effective_sigma_real_s_m"],
            "imaginary_conductivity_s_m": sweep["effective_sigma_imag_s_m"],
            "imaginary_conductivity_magnitude_s_m": sweep["effective_sigma_imag_s_m"].abs(),
            "real_relative_permittivity": sweep["effective_sigma_imag_s_m"].to_numpy(dtype=float) / (omega * EPSILON0_F_M),
            "real_relative_permittivity_magnitude": np.abs(sweep["effective_sigma_imag_s_m"].to_numpy(dtype=float) / (omega * EPSILON0_F_M)),
        }
    )


def assert_close_series(left: pd.Series, right: pd.Series, label: str) -> None:
    if not np.allclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), rtol=1.0e-12, atol=1.0e-30):
        raise AssertionError(f"source data does not match sweep output for {label}")


def verify_component_source(source: pd.DataFrame, paths: dict[str, Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for mechanism in MECHANISMS:
        dataset = f"our_corrected_{mechanism}"
        plotted = source.loc[source["dataset"] == dataset].sort_values("frequency_hz").reset_index(drop=True)
        expected = expected_source_from_sweep(paths[mechanism], mechanism).sort_values("frequency_hz").reset_index(drop=True)
        if len(plotted) != len(expected):
            raise AssertionError(f"{dataset} has {len(plotted)} plotted rows but {len(expected)} sweep rows")
        for column in [
            "frequency_hz",
            "real_conductivity_s_m",
            "imaginary_conductivity_s_m",
            "imaginary_conductivity_magnitude_s_m",
            "real_relative_permittivity",
            "real_relative_permittivity_magnitude",
        ]:
            assert_close_series(plotted[column], expected[column], f"{dataset}:{column}")
        records.append(
            {
                "dataset": dataset,
                "source_csv": str(paths[mechanism]),
                "rows": int(len(plotted)),
                "matches_sweep_csv": True,
            }
        )
    return records


def verify_experiment_source(source: pd.DataFrame, figure8: Path) -> dict[str, object]:
    plotted = source.loc[source["dataset"] == "experiment"].sort_values("frequency_hz").reset_index(drop=True)
    expected = read_paper_block(figure8, [0, 1, 2]).sort_values("frequency_hz").reset_index(drop=True)
    if len(plotted) != len(expected):
        raise AssertionError(f"experiment has {len(plotted)} plotted rows but {len(expected)} Figure8 experiment rows")
    for column in ["frequency_hz", "imaginary_conductivity_s_m", "real_relative_permittivity"]:
        assert_close_series(plotted[column], expected[column], f"experiment:{column}")
    return {"dataset": "experiment", "source": str(figure8), "columns": [0, 1, 2], "rows": int(len(plotted))}


def compare_against_paper_simulation_columns(source: pd.DataFrame, figure8: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    plotted_sim = source.loc[source["dataset"].str.startswith("our_corrected_", na=False)].copy()
    plotted_counts = plotted_sim.groupby("dataset").size().to_dict()
    for name, cols in PAPER_SIMULATION_BLOCKS.items():
        paper = read_paper_block(figure8, cols)
        records.append(
            {
                "paper_block": name,
                "paper_columns": cols,
                "paper_rows": int(len(paper)),
                "our_component_rows": {key: int(value) for key, value in plotted_counts.items()},
                "used_as_plotted_simulation_source": False,
                "reason": "Corrected plotted datasets match local full350 AC3D sweep CSVs; paper simulation blocks have separate columns and row counts.",
            }
        )
    return records


def write_markdown(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# Niu 2020 Corrected Figure 7-Style Provenance Verification",
        "",
        "This report verifies that the plotted simulation curves come from this project's AC3D `sweep_results.csv` files.",
        "The Niu 2020 paper simulation columns in `Figure8.xlsx` are not used as plotted simulation curves.",
        "",
        f"- Status: `{report['status']}`",
        f"- Source data: `{report['source_data_csv']}`",
        f"- Figure8 workbook: `{report['figure8_xlsx']}`",
        "",
        "## Plotted Simulation Curves",
        "",
        "| dataset | rows | source CSV | matches source CSV |",
        "|---|---:|---|---|",
    ]
    for record in report["component_sources"]:
        lines.append(
            f"| {record['dataset']} | {record['rows']} | `{record['source_csv']}` | {record['matches_sweep_csv']} |"
        )
    lines.extend(
        [
            "",
            "## Experiment Scatter",
            "",
            f"- Source columns: `{report['experiment_source']['columns']}` from `Figure8.xlsx`.",
            f"- Rows: `{report['experiment_source']['rows']}`.",
            "",
            "## Paper Simulation Columns",
            "",
            "The following blocks were inspected only to document that they are separate from the plotted corrected AC3D curves.",
            "",
            "| paper block | columns | rows | used as plotted simulation source |",
            "|---|---|---:|---|",
        ]
    )
    for record in report["paper_simulation_column_check"]:
        lines.append(
            f"| {record['paper_block']} | `{record['paper_columns']}` | {record['paper_rows']} | {record['used_as_plotted_simulation_source']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-data-csv",
        default=str(PROJECT_ROOT / "results" / "source_data" / "niu2020_figure7_style_corrected_reproduction_source_data.csv"),
    )
    parser.add_argument("--figure8", default=str(PROJECT_ROOT / "data" / "Niu 2020data" / "Figure8.xlsx"))
    parser.add_argument("--all-csv", default=str(PROJECT_ROOT / "results" / "niu2020_berea_full350_all_scaled_membrane_fft_x" / "sweep_results.csv"))
    parser.add_argument("--pore-csv", default=str(PROJECT_ROOT / "results" / "niu2020_berea_full350_pore_fft_x" / "sweep_results.csv"))
    parser.add_argument(
        "--membrane-csv",
        default=str(PROJECT_ROOT / "results" / "niu2020_berea_full350_membrane_scaled_fft_x" / "sweep_results.csv"),
    )
    parser.add_argument(
        "--interfacial-csv",
        default=str(PROJECT_ROOT / "results" / "niu2020_berea_full350_interfacial_precision_merged" / "sweep_results.csv"),
    )
    parser.add_argument("--out-json", default=str(PROJECT_ROOT / "results" / "niu2020" / "niu2020_figure7_style_corrected_provenance.json"))
    parser.add_argument("--out-md", default=str(PROJECT_ROOT / "results" / "niu2020" / "niu2020_figure7_style_corrected_provenance.md"))
    args = parser.parse_args()

    source_path = Path(args.source_data_csv)
    figure8 = Path(args.figure8)
    source = pd.read_csv(source_path)
    paths = {
        "all": Path(args.all_csv),
        "pore": Path(args.pore_csv),
        "membrane": Path(args.membrane_csv),
        "interfacial": Path(args.interfacial_csv),
    }
    report = {
        "status": "verified",
        "source_data_csv": str(source_path),
        "figure8_xlsx": str(figure8),
        "experiment_source": verify_experiment_source(source, figure8),
        "component_sources": verify_component_source(source, paths),
        "paper_simulation_column_check": compare_against_paper_simulation_columns(source, figure8),
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(Path(args.out_md), report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {out_json}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()
