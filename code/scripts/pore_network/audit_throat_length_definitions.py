#!/usr/bin/env python3
"""Audit alternative throat-length definitions against Niu 2020 Figure5."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT / "scripts" / "sip_simulation"))

from scan_pnextract_for_niu2020_membrane import (  # noqa: E402
    cumulative_fraction,
    log_edges,
    normalize,
    read_figure5,
    weighted_distribution,
    weighted_stats,
)


DEFAULT_NETWORK_DIR = (
    PROJECT_ROOT
    / "results"
    / "niu2020_berea_dong_blunt_beta_floor_scan_v2"
    / "recommended_figure5_candidate_beta_0p625_floor_0p25"
)
DEFAULT_FIGURE5 = PROJECT_ROOT / "data" / "Niu 2020data" / "Figure5.xlsx"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "niu2020_berea_throat_length_definition_audit_v1"


def alpha_tag(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "p").replace("-", "m")


def positive_or_nan(values: pd.Series | np.ndarray, epsilon_m: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.where(np.isfinite(arr) & (arr > 0.0), arr, np.nan)
    out = np.where(np.isfinite(out), np.maximum(out, epsilon_m), np.nan)
    return out


def merge_pore_radii(pores: pd.DataFrame, throats: pd.DataFrame) -> pd.DataFrame:
    radii = pores[["pore_id", "pore_radius_m"]].copy()
    out = throats.copy()
    out = out.merge(radii.rename(columns={"pore_id": "pore1_id", "pore_radius_m": "pore1_radius_m"}), on="pore1_id", how="left")
    out = out.merge(radii.rename(columns={"pore_id": "pore2_id", "pore_radius_m": "pore2_radius_m"}), on="pore2_id", how="left")
    return out


def compute_dong_blunt_partition_length(
    throat_data: pd.DataFrame,
    *,
    beta: float,
    epsilon_m: float,
) -> np.ndarray:
    """Approximate Dong-Blunt length from center length, radii, and throat radius.

    The original pnextract throat-center distances before pore/throat partition
    are not exported in Statoil link files. This diagnostic partitions the
    center-to-center distance in proportion to pore radii, then applies the
    Dong-Blunt radius-ratio pore/throat split.
    """

    center = throat_data["pore_center_to_center_length_m"].to_numpy(dtype=float)
    rt = throat_data["throat_radius_m"].to_numpy(dtype=float)
    rp1 = throat_data["pore1_radius_m"].to_numpy(dtype=float)
    rp2 = throat_data["pore2_radius_m"].to_numpy(dtype=float)
    denom = rp1 + rp2
    valid = np.isfinite(center) & np.isfinite(rt) & np.isfinite(rp1) & np.isfinite(rp2) & (denom > 0) & (rp1 > 0) & (rp2 > 0)
    out = np.full(len(throat_data), np.nan, dtype=float)
    lpt1 = np.zeros_like(center)
    lpt2 = np.zeros_like(center)
    lpt1[valid] = center[valid] * rp1[valid] / denom[valid]
    lpt2[valid] = center[valid] - lpt1[valid]
    lp1 = lpt1 * (1.0 - beta * rt / rp1)
    lp2 = lpt2 * (1.0 - beta * rt / rp2)
    lp1 = np.minimum(np.maximum(lp1, 0.0), lpt1)
    lp2 = np.minimum(np.maximum(lp2, 0.0), lpt2)
    length = center - lp1 - lp2
    out[valid] = np.maximum(length[valid], epsilon_m)
    return out


def compute_length_definitions(
    pores: pd.DataFrame,
    throats: pd.DataFrame,
    *,
    alpha_values: list[float] | tuple[float, ...] = (0.5, 0.67, 1.0),
    beta: float = 0.625,
    epsilon_m: float = 1.0e-12,
) -> pd.DataFrame:
    data = merge_pore_radii(pores, throats)
    out = pd.DataFrame(
        {
            "throat_id": data["throat_id"].to_numpy(dtype=int),
            "pore1_id": data["pore1_id"].to_numpy(dtype=int),
            "pore2_id": data["pore2_id"].to_numpy(dtype=int),
            "throat_volume_m3": data["throat_volume_m3"].to_numpy(dtype=float),
            "throat_radius_m": data["throat_radius_m"].to_numpy(dtype=float),
            "exported_m": positive_or_nan(data["throat_length_m"], epsilon_m),
            "center_m": positive_or_nan(data["pore_center_to_center_length_m"], epsilon_m),
        }
    )

    center = data["pore_center_to_center_length_m"].to_numpy(dtype=float)
    rp1 = data["pore1_radius_m"].to_numpy(dtype=float)
    rp2 = data["pore2_radius_m"].to_numpy(dtype=float)
    both_internal = np.isfinite(rp1) & np.isfinite(rp2)
    center_minus = np.full(len(data), np.nan, dtype=float)
    center_minus[both_internal] = center[both_internal] - rp1[both_internal] - rp2[both_internal]
    out["center_minus_rp1_rp2_m"] = positive_or_nan(center_minus, epsilon_m)

    for alpha in alpha_values:
        values = np.full(len(data), np.nan, dtype=float)
        values[both_internal] = center[both_internal] - alpha * (rp1[both_internal] + rp2[both_internal])
        out[f"center_minus_alpha_{alpha_tag(alpha)}_radii_m"] = positive_or_nan(values, epsilon_m)

    circular_area = math.pi * np.square(data["throat_radius_m"].to_numpy(dtype=float))
    volume_length = data["throat_volume_m3"].to_numpy(dtype=float) / circular_area
    out["volume_over_circular_area_m"] = positive_or_nan(volume_length, epsilon_m)
    out[f"dong_blunt_beta_{alpha_tag(beta)}_partition_m"] = compute_dong_blunt_partition_length(data, beta=beta, epsilon_m=epsilon_m)
    return out


def evaluate_length_definitions(
    paper_throat: pd.DataFrame,
    definitions: pd.DataFrame,
    definition_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    centers = paper_throat["center_m"].to_numpy(dtype=float)
    paper_y = normalize(paper_throat["relative_volume"].to_numpy(dtype=float))
    weights = definitions["throat_volume_m3"].to_numpy(dtype=float)
    paper_stats = weighted_stats(centers, paper_y)

    rows: list[dict[str, float | int | str]] = []
    comparisons: list[pd.DataFrame] = []
    for column in definition_columns:
        values = definitions[column].to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(weights) & (values > 0) & (weights > 0)
        y = weighted_distribution(values, weights, centers)
        stats = weighted_stats(values, weights)
        row: dict[str, float | int | str] = {
            "definition": column,
            "valid_throats": int(valid.sum()),
            "total_throats": int(len(definitions)),
            "valid_volume_fraction": float(weights[valid].sum() / weights[np.isfinite(weights) & (weights > 0)].sum()),
            "l1_distance": float(np.abs(y - paper_y).sum()),
            "mean_m": stats["mean_m"],
            "mean_um": stats["mean_m"] * 1.0e6,
            "mean_ratio_to_paper": stats["mean_m"] / paper_stats["mean_m"],
            "median_um": stats["median_m"] * 1.0e6,
            "min_um": stats["min_m"] * 1.0e6,
            "q10_um": stats["q10_m"] * 1.0e6,
            "q90_um": stats["q90_m"] * 1.0e6,
            "max_um": stats["max_m"] * 1.0e6,
        }
        for threshold_um in (1.0, 5.0, 10.0):
            threshold_m = threshold_um * 1.0e-6
            row[f"paper_lt_{threshold_um:g}um_fraction"] = cumulative_fraction(centers, paper_y, threshold_m)
            row[f"candidate_lt_{threshold_um:g}um_fraction"] = cumulative_fraction(values, weights, threshold_m)
        rows.append(row)
        comparisons.append(
            pd.DataFrame(
                {
                    "definition": column,
                    "bin_center_m": centers,
                    "paper_relative_volume": paper_y,
                    "candidate_relative_volume": y,
                }
            )
        )

    return pd.DataFrame(rows), pd.concat(comparisons, ignore_index=True)


def plot_comparison(comparison: pd.DataFrame, summary: pd.DataFrame, out_path: Path) -> None:
    top = summary.sort_values("l1_distance").head(6)["definition"].tolist()
    paper = comparison[comparison["definition"] == top[0]][["bin_center_m", "paper_relative_volume"]].copy()
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    ax.plot(paper["bin_center_m"] * 1.0e6, paper["paper_relative_volume"], "o-", color="black", label="Niu Figure5")
    for definition in top:
        subset = comparison[comparison["definition"] == definition]
        ax.plot(subset["bin_center_m"] * 1.0e6, subset["candidate_relative_volume"], "s--", label=definition)
    ax.set_xscale("log")
    ax.set_xlabel("Pore throat length (um)")
    ax.set_ylabel("Relative volume")
    ax.set_title("Alternative throat-length definitions vs Niu Figure5")
    ax.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_readme(out_dir: Path, network_dir: Path, figure5: Path, summary: pd.DataFrame) -> None:
    best = summary.sort_values("l1_distance").iloc[0]
    exported = summary[summary["definition"] == "exported_m"].iloc[0]
    lines = [
        "# Niu 2020 Throat-Length Definition Audit",
        "",
        "This diagnostic recomputes the Niu Figure5 pore-throat length distribution from one parsed pnextract network using multiple possible length definitions.",
        "",
        f"- Network directory: `{network_dir}`",
        f"- Figure5 workbook: `{figure5}`",
        "- Input network CSVs are read only; this run does not edit the original CT, paper data, or vendored pnextract source.",
        "",
        "## Key result",
        "",
        f"- Best definition by throat L1: `{best['definition']}` with L1 `{best['l1_distance']:.6g}`.",
        f"- Exported pnextract length L1: `{exported['l1_distance']:.6g}`.",
        f"- Best `<1/<5/<10 um`: `{best['candidate_lt_1um_fraction']*100:.4g}% / {best['candidate_lt_5um_fraction']*100:.4g}% / {best['candidate_lt_10um_fraction']*100:.4g}%`.",
        "- If the best short-throat fractions remain far below Niu, the mismatch is not only a link2 throat-length field-definition problem.",
        "",
        "## Outputs",
        "",
        "- `length_definition_metrics.csv`: per-definition L1, mean, short-throat fractions, valid-throat counts.",
        "- `length_definition_distribution_comparison.csv`: binned Figure5-vs-candidate source data.",
        "- `length_definition_audit_summary.json`: machine-readable summary and input provenance.",
        "- `figures/throat_length_definition_comparison.png`: comparison plot for the best definitions.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-dir", default=str(DEFAULT_NETWORK_DIR))
    parser.add_argument("--figure5", default=str(DEFAULT_FIGURE5))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--alpha-values", nargs="*", type=float, default=[0.5, 0.67, 1.0])
    parser.add_argument("--beta", type=float, default=0.625)
    parser.add_argument("--epsilon-m", type=float, default=1.0e-12)
    args = parser.parse_args()

    network_dir = Path(args.network_dir)
    figure5 = Path(args.figure5)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    pores = pd.read_csv(network_dir / "pores.csv")
    throats = pd.read_csv(network_dir / "throats.csv")
    _, paper_throat = read_figure5(figure5)
    definitions = compute_length_definitions(
        pores,
        throats,
        alpha_values=args.alpha_values,
        beta=args.beta,
        epsilon_m=args.epsilon_m,
    )
    definition_columns = [column for column in definitions.columns if column.endswith("_m") and column not in {"throat_volume_m3", "throat_radius_m"}]
    summary, comparison = evaluate_length_definitions(paper_throat, definitions, definition_columns)
    summary = summary.sort_values(["l1_distance", "definition"]).reset_index(drop=True)

    definitions.to_csv(out_dir / "throat_length_definitions.csv", index=False)
    summary.to_csv(out_dir / "length_definition_metrics.csv", index=False)
    comparison.to_csv(out_dir / "length_definition_distribution_comparison.csv", index=False)
    plot_comparison(comparison, summary, out_dir / "figures" / "throat_length_definition_comparison.png")

    payload = {
        "network_dir": str(network_dir),
        "figure5": str(figure5),
        "n_pores": int(len(pores)),
        "n_throats": int(len(throats)),
        "alpha_values": [float(v) for v in args.alpha_values],
        "beta": float(args.beta),
        "epsilon_m": float(args.epsilon_m),
        "paper_short_throat_fraction_percent": {
            "<1um": float(summary.iloc[0]["paper_lt_1um_fraction"] * 100.0),
            "<5um": float(summary.iloc[0]["paper_lt_5um_fraction"] * 100.0),
            "<10um": float(summary.iloc[0]["paper_lt_10um_fraction"] * 100.0),
        },
        "best_definition": summary.iloc[0].to_dict(),
    }
    (out_dir / "length_definition_audit_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(out_dir, network_dir, figure5, summary)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
