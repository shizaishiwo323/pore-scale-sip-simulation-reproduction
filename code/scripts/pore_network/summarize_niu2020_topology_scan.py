#!/usr/bin/env python3
"""Summarize Niu 2020 pnextract topology-parameter scan evidence.

This script does not rerun pnextract. It audits an existing scan_metrics.csv
and writes a compact evidence bundle focused on whether medial-surface/topology
parameters can recover the short-throat tail in Niu Figure5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCAN = PROJECT_ROOT / "results" / "niu2020_berea_pnextract_multparam_scan_v2" / "scan_metrics.csv"
DEFAULT_OUT = PROJECT_ROOT / "results" / "niu2020_berea_topology_scan_audit_v1"

PAPER_COLS = {
    "lt_1um": "paper_throat_volume_fraction_lt_1um",
    "lt_5um": "paper_throat_volume_fraction_lt_5um",
    "lt_10um": "paper_throat_volume_fraction_lt_10um",
}
CANDIDATE_COLS = {
    "lt_1um": "candidate_throat_volume_fraction_lt_1um",
    "lt_5um": "candidate_throat_volume_fraction_lt_5um",
    "lt_10um": "candidate_throat_volume_fraction_lt_10um",
}


def read_scan(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        "candidate",
        "status",
        "pore_l1",
        "throat_l1",
        "objective_score",
        "pnextract_lines",
        *PAPER_COLS.values(),
        *CANDIDATE_COLS.values(),
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"missing required columns in {path}: {missing}")
    return df[df["status"].eq("ok")].copy()


def row_summary(row: pd.Series, basis: str) -> dict[str, object]:
    item: dict[str, object] = {
        "basis": basis,
        "candidate": row["candidate"],
        "pore_l1": float(row["pore_l1"]),
        "throat_l1": float(row["throat_l1"]),
        "objective_score": float(row["objective_score"]),
        "pnextract_lines": row["pnextract_lines"],
    }
    for key in ("lt_1um", "lt_5um", "lt_10um"):
        paper = float(row[PAPER_COLS[key]])
        candidate = float(row[CANDIDATE_COLS[key]])
        item[f"paper_{key}"] = paper
        item[f"candidate_{key}"] = candidate
        item[f"coverage_{key}"] = candidate / paper if paper > 0 else float("nan")
        item[f"gap_{key}"] = paper - candidate
    for optional in ("pore_mean_ratio", "throat_mean_ratio", "n_pores", "n_throats"):
        if optional in row:
            item[optional] = float(row[optional])
    return item


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.append(row_summary(df.sort_values("objective_score").iloc[0], "best_objective"))
    rows.append(row_summary(df.sort_values("throat_l1").iloc[0], "best_throat_l1"))
    rows.append(row_summary(df.sort_values("pore_l1").iloc[0], "best_pore_l1"))
    for key, column in CANDIDATE_COLS.items():
        rows.append(row_summary(df.sort_values(column, ascending=False).iloc[0], f"max_{key}"))

    top = pd.DataFrame(rows).drop_duplicates(subset=["basis", "candidate"]).reset_index(drop=True)
    paper = {key: float(df[column].iloc[0]) for key, column in PAPER_COLS.items()}
    maxima = {
        key: {
            "candidate": str(df.sort_values(column, ascending=False).iloc[0]["candidate"]),
            "value": float(df[column].max()),
            "coverage": float(df[column].max() / paper[key]),
            "gap": float(paper[key] - df[column].max()),
        }
        for key, column in CANDIDATE_COLS.items()
    }
    summary = {
        "successful_candidates": int(len(df)),
        "paper_short_throat_fractions": paper,
        "max_candidate_short_throat_fractions": maxima,
        "best_objective_candidate": str(df.sort_values("objective_score").iloc[0]["candidate"]),
        "best_throat_l1_candidate": str(df.sort_values("throat_l1").iloc[0]["candidate"]),
        "topology_parameters_recover_niu_short_tail": bool(
            maxima["lt_1um"]["coverage"] >= 0.8
            and maxima["lt_5um"]["coverage"] >= 0.8
            and maxima["lt_10um"]["coverage"] >= 0.8
        ),
    }
    return top, summary


def write_readme(out_dir: Path, scan_csv: Path, top: pd.DataFrame, summary: dict[str, object]) -> None:
    paper = summary["paper_short_throat_fractions"]
    maxima = summary["max_candidate_short_throat_fractions"]
    lines = [
        "# Niu 2020 Berea pnextract topology scan audit",
        "",
        "This audit re-summarizes the existing full-Berea multi-parameter pnextract scan from the viewpoint of short-throat recovery. It does not rerun pnextract.",
        "",
        f"- Source scan: `{scan_csv}`",
        f"- Successful candidates: `{summary['successful_candidates']}`",
        f"- Best objective candidate: `{summary['best_objective_candidate']}`",
        f"- Best throat-L1 candidate: `{summary['best_throat_l1_candidate']}`",
        "",
        "## Short-throat recovery limits",
        "",
        "| threshold | Niu Figure5 | best candidate | coverage | missing fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (("lt_1um", "<1 um"), ("lt_5um", "<5 um"), ("lt_10um", "<10 um")):
        max_item = maxima[key]
        lines.append(
            f"| {label} | {paper[key]:.6g} | {max_item['value']:.6g} | "
            f"{max_item['coverage']:.2%} | {max_item['gap']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The scanned medial-surface/topology parameter recipes improve the <10 um tail only partially, but none of them recovers the sub-micron throat fraction in Niu Figure5.",
            "",
            "This supports the current working interpretation: ordinary pnextract threshold tuning is not enough; the remaining mismatch most likely sits in the internal pore/throat construction, pore merging/pruning rules, preprocessing details not captured by simple morphology operations, or Niu's effective throat-resistance definition.",
            "",
            "## Selected rows",
            "",
            "`topology_scan_selected_candidates.csv` lists candidates selected by best objective, best throat L1, best pore L1, and maximum short-throat fractions.",
        ]
    )
    out_dir.joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-csv", type=Path, default=DEFAULT_SCAN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    df = read_scan(args.scan_csv)
    top, summary = summarize(df)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    top.to_csv(args.out_dir / "topology_scan_selected_candidates.csv", index=False)
    (args.out_dir / "topology_scan_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_readme(args.out_dir, args.scan_csv, top, summary)
    print(json.dumps({"out_dir": str(args.out_dir), **summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
