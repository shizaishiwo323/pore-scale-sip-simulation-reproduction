#!/usr/bin/env python3
"""Summarize CT-core physical and pore-network properties for the reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-audit", default=str(PROJECT_ROOT / "results" / "data_audit" / "microct_raw_audit" / "raw_audit_summary.json"))
    parser.add_argument("--network-summary", default=str(PROJECT_ROOT / "results" / "pnextract" / "rawdata_pnextract_rerun_20260527" / "network_parsed" / "network_summary.json"))
    parser.add_argument("--pores-csv", default=str(PROJECT_ROOT / "results" / "pnextract" / "rawdata_pnextract_rerun_20260527" / "network_parsed" / "pores.csv"))
    parser.add_argument("--throats-csv", default=str(PROJECT_ROOT / "results" / "pnextract" / "rawdata_pnextract_rerun_20260527" / "network_parsed" / "throats.csv"))
    parser.add_argument("--voxel-size-m", type=float, default=2.8e-6)
    parser.add_argument("--out-json", default=str(PROJECT_ROOT / "results" / "source_data" / "ct_core_physical_properties.json"))
    parser.add_argument("--out-md", default=str(PROJECT_ROOT / "docs" / "notes" / "ct_core_physical_properties.md"))
    args = parser.parse_args()

    raw_audit = json.loads(Path(args.raw_audit).read_text(encoding="utf-8"))
    network_summary = json.loads(Path(args.network_summary).read_text(encoding="utf-8"))
    pores = pd.read_csv(args.pores_csv)
    throats = pd.read_csv(args.throats_csv)

    shape = np.array(raw_audit["shape"], dtype=float)
    physical_size_m = shape * args.voxel_size_m
    bulk_volume_m3 = float(np.prod(physical_size_m))
    porosity = float(raw_audit["porosity_from_pore_label"])
    pore_volume_m3 = bulk_volume_m3 * porosity

    summary = {
        "raw_volume": {
            "shape_voxels": raw_audit["shape"],
            "dtype": raw_audit["dtype"],
            "pore_label": raw_audit["pore_label"],
            "solid_label": raw_audit["solid_label"],
            "porosity_from_voxel_count": porosity,
            "voxel_size_m": args.voxel_size_m,
            "physical_size_m": physical_size_m.tolist(),
            "bulk_volume_m3": bulk_volume_m3,
            "pore_volume_from_voxels_m3": pore_volume_m3,
        },
        "pnextract_network": {
            **network_summary,
            "pore_radius_median_m": float(pores["pore_radius_m"].median()),
            "throat_radius_median_m": float(throats["throat_radius_m"].median()),
            "throat_length_median_m": float(throats["throat_length_m"].median()),
        },
        "algorithm_statement": {
            "network_extractor": "pnextract",
            "algorithm_family": "maximal-ball pore-network extraction",
            "local_source": r"C:\Users\imgw\Documents\Codex\论文复现\pore-scale-simulation-reproduction\pnextract\src\pnm\pnextract\README.md",
            "important_distinction": "pnextract creates a pore-throat ball-and-stick network for geometry and polarization statistics; the AC3D effective-property solve uses the original two-phase voxel grid.",
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# CT Core Physical Properties",
        "",
        "## Raw Voxel Core",
        "",
        f"- Shape: `{raw_audit['shape'][0]} x {raw_audit['shape'][1]} x {raw_audit['shape'][2]}` voxels.",
        f"- Dtype: `{raw_audit['dtype']}`.",
        f"- Label mapping used here: `{raw_audit['pore_label']}` = pore/water, `{raw_audit['solid_label']}` = solid.",
        f"- Porosity from voxel counts: `{porosity:.10f}`.",
        f"- Voxel size used for physical scaling: `{args.voxel_size_m:.6e} m`.",
        f"- Physical side lengths: `{physical_size_m[0]:.6e}, {physical_size_m[1]:.6e}, {physical_size_m[2]:.6e} m`.",
        f"- Bulk volume: `{bulk_volume_m3:.6e} m^3`.",
        "",
        "## pnextract Network",
        "",
        f"- Pores: `{network_summary['n_pores']}`.",
        f"- Throats: `{network_summary['n_throats']}`.",
        f"- Volume-weighted mean pore radius: `{network_summary['pore_radius_volume_weighted_mean_m']:.6e} m`.",
        f"- Volume-weighted mean throat length: `{network_summary['throat_length_volume_weighted_mean_m']:.6e} m`.",
        "",
        "## Algorithm Note",
        "",
        "- The pnextract README identifies the extractor as a rewrite of the Dong and Blunt (2009) maximal-ball network extraction algorithm.",
        "- Therefore the extracted pore network is a pore-throat, ball-and-stick representation based on maximal balls.",
        "- The AC3D electrical simulation in this project is not solved on that network; it is solved on the original two-phase voxel grid, while the pnextract network supplies pore/throat statistics for polarization spectra.",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
