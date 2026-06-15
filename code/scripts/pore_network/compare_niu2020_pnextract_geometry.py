#!/usr/bin/env python3
"""Compare our pnextract Berea network geometry against Niu 2020 figure data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
NIU_TABLE1 = {
    "core_voxels_per_side": 350.0,
    "voxel_size_um": 2.8,
    "core_edge_length_um": 350.0 * 2.8,
    "porosity_percent": 20.2,
    "mean_pore_size_um": 35.0,
}


def centers_to_log_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float)
    centers = centers[np.isfinite(centers) & (centers > 0)]
    if centers.size < 2:
        raise ValueError("at least two positive bin centers are required")
    centers = np.sort(centers)
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = np.sqrt(centers[:-1] * centers[1:])
    edges[0] = centers[0] ** 2 / edges[1]
    edges[-1] = centers[-1] ** 2 / edges[-2]
    return edges


def weighted_histogram(values: pd.Series, weights: pd.Series, edges: np.ndarray) -> np.ndarray:
    values_np = values.to_numpy(dtype=float)
    weights_np = weights.to_numpy(dtype=float)
    mask = np.isfinite(values_np) & np.isfinite(weights_np) & (values_np > 0) & (weights_np > 0)
    hist, _ = np.histogram(values_np[mask], bins=edges, weights=weights_np[mask])
    total = hist.sum()
    if total <= 0:
        raise ValueError("weighted histogram is empty")
    return hist / total


def normalize_fraction(values: pd.Series) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    arr = np.nan_to_num(arr, nan=0.0)
    total = arr.sum()
    if total <= 0:
        raise ValueError("relative-volume column sums to zero")
    return arr / total


def weighted_mean_um(values_um: pd.Series, weights: pd.Series) -> float:
    values_np = values_um.to_numpy(dtype=float)
    weights_np = weights.to_numpy(dtype=float)
    mask = np.isfinite(values_np) & np.isfinite(weights_np) & (values_np > 0) & (weights_np > 0)
    return float(np.average(values_np[mask], weights=weights_np[mask]))


def read_niu_geometry(figure5_xlsx: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(figure5_xlsx, sheet_name=0)
    pore = pd.DataFrame(
        {
            "size_um": pd.to_numeric(raw["pore node size (m)"], errors="coerce") * 1.0e6,
            "paper_relative_volume": pd.to_numeric(raw["pore node relaive volume"], errors="coerce"),
        }
    ).dropna()
    throat = pd.DataFrame(
        {
            "length_um": pd.to_numeric(raw["pore throat length (m)"], errors="coerce") * 1.0e6,
            "paper_relative_volume": pd.to_numeric(raw["Pore throat relative volume"], errors="coerce"),
        }
    ).dropna()
    pore["paper_relative_volume"] = normalize_fraction(pore["paper_relative_volume"])
    throat["paper_relative_volume"] = normalize_fraction(throat["paper_relative_volume"])
    return pore, throat


def build_comparison(
    paper_pore: pd.DataFrame,
    paper_throat: pd.DataFrame,
    our_pores: pd.DataFrame,
    our_throats: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pore_edges = centers_to_log_edges(paper_pore["size_um"].to_numpy(dtype=float))
    throat_edges = centers_to_log_edges(paper_throat["length_um"].to_numpy(dtype=float))

    pore = paper_pore.rename(columns={"size_um": "bin_center_um"}).copy()
    pore["our_relative_volume"] = weighted_histogram(
        our_pores["pore_radius_m"] * 1.0e6,
        our_pores["pore_volume_m3"],
        pore_edges,
    )
    pore["quantity"] = "pore_node_size"
    pore["unit"] = "um"

    throat = paper_throat.rename(columns={"length_um": "bin_center_um"}).copy()
    throat["our_relative_volume"] = weighted_histogram(
        our_throats["throat_length_m"] * 1.0e6,
        our_throats["throat_volume_m3"],
        throat_edges,
    )
    throat["quantity"] = "pore_throat_length"
    throat["unit"] = "um"
    return pore, throat


def build_metric_table(
    paper_pore: pd.DataFrame,
    paper_throat: pd.DataFrame,
    our_pores: pd.DataFrame,
    our_throats: pd.DataFrame,
    prepare_summary: dict[str, object],
    network_summary: dict[str, object],
) -> pd.DataFrame:
    paper_pore_mean = float(np.sum(paper_pore["size_um"] * paper_pore["paper_relative_volume"]))
    paper_throat_mean = float(np.sum(paper_throat["length_um"] * paper_throat["paper_relative_volume"]))
    our_pore_mean = weighted_mean_um(our_pores["pore_radius_m"] * 1.0e6, our_pores["pore_volume_m3"])
    our_throat_mean = weighted_mean_um(our_throats["throat_length_m"] * 1.0e6, our_throats["throat_volume_m3"])
    our_voxel_size = float(prepare_summary["effective_voxel_size_um"])
    our_core_edge = float(prepare_summary["shape_zyx"][0]) * our_voxel_size
    rows = [
        {
            "metric": "core_edge_length",
            "paper_value": NIU_TABLE1["core_edge_length_um"],
            "our_value": our_core_edge,
            "unit": "um",
            "paper_source": "Niu 2020 Figure 3 caption: 350^3 voxels, 2.8 um voxel",
            "our_source": "pnextract prepare summary shape_zyx and effective_voxel_size_um",
        },
        {
            "metric": "voxel_size",
            "paper_value": NIU_TABLE1["voxel_size_um"],
            "our_value": our_voxel_size,
            "unit": "um",
            "paper_source": "Niu 2020 Figure 3 caption",
            "our_source": "pnextract prepare summary effective_voxel_size_um",
        },
        {
            "metric": "porosity",
            "paper_value": NIU_TABLE1["porosity_percent"],
            "our_value": float(prepare_summary["porosity"]) * 100.0,
            "unit": "%",
            "paper_source": "Niu 2020 Table 1 bulk petrophysical property",
            "our_source": "count(pore voxels) / total voxels in segmented microCT_Berea",
        },
        {
            "metric": "pore_node_size_volume_weighted_mean",
            "paper_value": paper_pore_mean,
            "our_value": our_pore_mean,
            "unit": "um",
            "paper_source": "Niu 2020 Figure 4 data in data/Niu 2020data/Figure5.xlsx",
            "our_source": "pore_radius_m weighted by pore_volume_m3",
        },
        {
            "metric": "pore_throat_length_volume_weighted_mean",
            "paper_value": paper_throat_mean,
            "our_value": our_throat_mean,
            "unit": "um",
            "paper_source": "Niu 2020 Figure 4 data in data/Niu 2020data/Figure5.xlsx",
            "our_source": "throat_length_m weighted by throat_volume_m3",
        },
        {
            "metric": "pore_node_count",
            "paper_value": np.nan,
            "our_value": float(network_summary["n_pores"]),
            "unit": "count",
            "paper_source": "not reported in Niu 2020 Figure 4/Table 1",
            "our_source": "parsed pnextract network_summary.json",
        },
        {
            "metric": "pore_throat_count",
            "paper_value": np.nan,
            "our_value": float(network_summary["n_throats"]),
            "unit": "count",
            "paper_source": "not reported in Niu 2020 Figure 4/Table 1",
            "our_source": "parsed pnextract network_summary.json",
        },
    ]
    table = pd.DataFrame(rows)
    table["ours_minus_paper"] = table["our_value"] - table["paper_value"]
    table["ours_over_paper"] = table["our_value"] / table["paper_value"]
    return table


def plot_comparison(pore: pd.DataFrame, throat: pd.DataFrame, metrics: pd.DataFrame, out_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 10,
            "axes.linewidth": 0.9,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.semilogx(pore["bin_center_um"], pore["paper_relative_volume"], "o-", color="black", lw=1.6, ms=4, label="Niu 2020 Figure 4")
    ax.semilogx(pore["bin_center_um"], pore["our_relative_volume"], "s--", color="#0072B2", lw=1.4, ms=3.5, label="Our pnextract")
    ax.axvline(NIU_TABLE1["mean_pore_size_um"], color="#D55E00", ls=":", lw=1.4, label="Niu Table 1 mean pore size")
    ax.set_xlabel("Pore node size / radius, um")
    ax.set_ylabel("Relative volume fraction")
    ax.set_title("(a) Pore node size distribution")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.semilogx(
        throat["bin_center_um"],
        throat["paper_relative_volume"],
        "o-",
        color="black",
        lw=1.6,
        ms=4,
        label="Niu 2020 Figure 4",
    )
    ax.semilogx(
        throat["bin_center_um"],
        throat["our_relative_volume"],
        "s--",
        color="#009E73",
        lw=1.4,
        ms=3.5,
        label="Our pnextract",
    )
    ax.set_xlabel("Pore throat length, um")
    ax.set_ylabel("Relative volume fraction")
    ax.set_title("(b) Pore throat length distribution")
    ax.legend(frameon=False)

    scalar_names = [
        "core_edge_length",
        "porosity",
        "pore_node_size_volume_weighted_mean",
        "pore_throat_length_volume_weighted_mean",
    ]
    subset = metrics.loc[metrics["metric"].isin(scalar_names)].copy()
    labels = ["Core edge\n(um)", "Porosity\n(%)", "Pore size\nVW mean (um)", "Throat length\nVW mean (um)"]
    x = np.arange(len(subset))
    width = 0.36
    ax = axes[1, 0]
    ratio = subset["ours_over_paper"].to_numpy(dtype=float)
    bars = ax.bar(x, ratio, 0.55, color=["#999999", "#E69F00", "#0072B2", "#009E73"])
    ax.axhline(1.0, color="black", lw=1.0, ls="--")
    for bar, value in zip(bars, ratio):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.2f}x", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(1.65, float(np.nanmax(ratio)) * 1.22))
    ax.set_ylabel("Our value / Niu 2020 value")
    ax.set_title("(c) Scalar geometry ratios")

    ax = axes[1, 1]
    ax.axis("off")
    summary = metrics.set_index("metric")
    lines = [
        "(d) Main check",
        "",
        f"Grid/voxel: {summary.loc['core_edge_length','our_value']:.0f} um vs {summary.loc['core_edge_length','paper_value']:.0f} um",
        f"Porosity: {summary.loc['porosity','our_value']:.2f}% vs {summary.loc['porosity','paper_value']:.2f}%",
        f"Pore VW mean: {summary.loc['pore_node_size_volume_weighted_mean','our_value']:.2f} vs {summary.loc['pore_node_size_volume_weighted_mean','paper_value']:.2f} um",
        f"Throat VW mean: {summary.loc['pore_throat_length_volume_weighted_mean','our_value']:.2f} vs {summary.loc['pore_throat_length_volume_weighted_mean','paper_value']:.2f} um",
        "",
        f"Our parsed network: {summary.loc['pore_node_count','our_value']:.0f} pores, {summary.loc['pore_throat_count','our_value']:.0f} throats",
        "",
        "Note: Niu Table 1 reports mean pore size = 35 um;",
        "Figure 4 distribution gives a separate volume-weighted mean.",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {".png": {"dpi": 350}, ".svg": {}, ".pdf": {}}.items():
        fig.savefig(out_base.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_summary(path: Path, metrics: pd.DataFrame, distribution_csv: Path) -> None:
    lines = [
        "# Niu 2020 Pore Network Geometry Comparison",
        "",
        "Paper geometry data are read from `data/Niu 2020data/Figure5.xlsx`, whose headers contain the Figure 4 pore-node and pore-throat relative-volume distributions.",
        "Our network data are read from the parsed full-resolution pnextract output for `microCT_Berea`.",
        "",
        f"- Distribution source data: `{distribution_csv}`",
        "",
        "| metric | Niu 2020 | ours | unit | ours/paper |",
        "|---|---:|---:|---|---:|",
    ]
    for row in metrics.itertuples(index=False):
        paper = "" if pd.isna(row.paper_value) else f"{row.paper_value:.6g}"
        ratio = "" if pd.isna(row.ours_over_paper) else f"{row.ours_over_paper:.3f}"
        lines.append(f"| {row.metric} | {paper} | {row.our_value:.6g} | {row.unit} | {ratio} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure5-xlsx", default=str(ROOT / "data" / "Niu 2020data" / "Figure5.xlsx"))
    parser.add_argument("--pores-csv", default=str(ROOT / "results" / "pnextract" / "niu2020_berea_fullres" / "network_parsed" / "pores.csv"))
    parser.add_argument("--throats-csv", default=str(ROOT / "results" / "pnextract" / "niu2020_berea_fullres" / "network_parsed" / "throats.csv"))
    parser.add_argument(
        "--network-summary",
        default=str(ROOT / "results" / "pnextract" / "niu2020_berea_fullres" / "network_parsed" / "network_summary.json"),
    )
    parser.add_argument(
        "--prepare-summary",
        default=str(ROOT / "results" / "pnextract_inputs" / "niu2020_berea_fullres" / "niu2020_berea_fullres_pnextract_prepare_summary.json"),
    )
    parser.add_argument(
        "--out-base",
        default=str(ROOT / "figures" / "niu2020" / "niu2020_pore_network_geometry_comparison"),
    )
    parser.add_argument(
        "--distribution-csv",
        default=str(ROOT / "results" / "source_data" / "niu2020_pore_network_geometry_distribution_comparison.csv"),
    )
    parser.add_argument(
        "--metrics-csv",
        default=str(ROOT / "results" / "source_data" / "niu2020_pore_network_geometry_metric_comparison.csv"),
    )
    parser.add_argument(
        "--summary-md",
        default=str(ROOT / "results" / "niu2020" / "niu2020_pore_network_geometry_comparison_summary.md"),
    )
    args = parser.parse_args()

    paper_pore, paper_throat = read_niu_geometry(Path(args.figure5_xlsx))
    our_pores = pd.read_csv(args.pores_csv)
    our_throats = pd.read_csv(args.throats_csv)
    network_summary = json.loads(Path(args.network_summary).read_text(encoding="utf-8"))
    prepare_summary = json.loads(Path(args.prepare_summary).read_text(encoding="utf-8"))

    pore_cmp, throat_cmp = build_comparison(paper_pore, paper_throat, our_pores, our_throats)
    distribution = pd.concat([pore_cmp, throat_cmp], ignore_index=True, sort=False)
    metrics = build_metric_table(paper_pore, paper_throat, our_pores, our_throats, prepare_summary, network_summary)

    distribution_path = Path(args.distribution_csv)
    distribution_path.parent.mkdir(parents=True, exist_ok=True)
    distribution.to_csv(distribution_path, index=False)
    metrics_path = Path(args.metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    plot_comparison(pore_cmp, throat_cmp, metrics, Path(args.out_base))
    write_summary(Path(args.summary_md), metrics, distribution_path)
    print(f"wrote {Path(args.out_base).with_suffix('.png')}")
    print(f"wrote {distribution_path}")
    print(f"wrote {metrics_path}")
    print(f"wrote {args.summary_md}")


if __name__ == "__main__":
    main()
