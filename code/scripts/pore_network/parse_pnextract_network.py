#!/usr/bin/env python3
"""Parse pnextract Statoil-style node/link files into CSV summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_node1(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        first = True
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if first:
                first = False
                continue
            if len(parts) < 4:
                continue
            rows.append(
                {
                    "pore_id": int(parts[0]),
                    "pore_center_x_m": float(parts[1]),
                    "pore_center_y_m": float(parts[2]),
                    "pore_center_z_m": float(parts[3]),
                }
            )
    return pd.DataFrame(rows)


def read_node2(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            rows.append(
                {
                    "pore_id": int(parts[0]),
                    "pore_volume_m3": float(parts[1]),
                    "pore_radius_m": float(parts[2]),
                    "pore_shape_factor": float(parts[3]),
                    "pore_clay_volume": float(parts[4]),
                }
            )
    return pd.DataFrame(rows)


def read_link1(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        first = True
        for line in f:
            if first:
                first = False
                continue
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            rows.append(
                {
                    "throat_id": int(parts[0]),
                    "pore1_id": int(parts[1]),
                    "pore2_id": int(parts[2]),
                    "throat_radius_m": float(parts[3]),
                    "throat_shape_factor": float(parts[4]),
                    "pore_center_to_center_length_m": float(parts[5]),
                }
            )
    return pd.DataFrame(rows)


def read_link2(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            rows.append(
                {
                    "throat_id": int(parts[0]),
                    "pore1_id_link2": int(parts[1]),
                    "pore2_id_link2": int(parts[2]),
                    "pore1_length_m": float(parts[3]),
                    "pore2_length_m": float(parts[4]),
                    "throat_length_m": float(parts[5]),
                    "throat_volume_m3": float(parts[6]),
                    "throat_clay_volume": float(parts[7]),
                }
            )
    return pd.DataFrame(rows)


def infer_voxel_size_m(throats: pd.DataFrame, diagnostics: pd.DataFrame) -> float | None:
    """Infer voxel size from link2 meter lengths and diagnostics voxel lengths."""
    if "length_voxels" not in diagnostics.columns:
        return None
    merged = throats[["throat_id", "throat_length_m"]].merge(
        diagnostics[["split_throat_id", "length_voxels"]],
        left_on="throat_id",
        right_on="split_throat_id",
        how="inner",
    )
    if merged.empty:
        return None
    length_voxels = merged["length_voxels"].to_numpy(dtype=float)
    length_m = merged["throat_length_m"].to_numpy(dtype=float)
    valid = np.isfinite(length_voxels) & np.isfinite(length_m) & (length_voxels > 0) & (length_m > 0)
    if not valid.any():
        return None
    return float(np.median(length_m[valid] / length_voxels[valid]))


def merge_contact_split_dong_blunt_diagnostics(
    throats: pd.DataFrame,
    diagnostics_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Attach explicit membrane geometry fields exported by the v11 contact-split patch."""
    if not diagnostics_path.exists():
        return throats, {
            "contact_patch_dong_blunt_diagnostics_merged": False,
            "contact_patch_dong_blunt_diagnostics_csv": None,
            "membrane_geometry_mode": "single_throat_geometry",
        }

    diagnostics = pd.read_csv(diagnostics_path)
    if "split_throat_id" not in diagnostics.columns:
        raise ValueError(f"{diagnostics_path} must include split_throat_id")

    voxel_size_m = infer_voxel_size_m(throats, diagnostics)
    if voxel_size_m is None:
        raise ValueError(
            f"Could not infer voxel size for {diagnostics_path}; diagnostics length_voxels "
            "must match parsed throat_length_m."
        )

    rename = {
        "split_throat_id": "throat_id",
        "original_throat_id": "contact_original_throat_id",
        "patch_index": "contact_patch_index",
        "patch_count": "contact_patch_count",
        "face_count": "contact_face_count",
        "length_voxels": "contact_length_voxels",
        "radius_voxels": "contact_radius_voxels",
        "radius_for_length_voxels": "contact_radius_for_length_voxels",
        "radius_unclamped_voxels": "contact_radius_unclamped_voxels",
        "max_radius_from_neighbor_pores_voxels": "contact_max_radius_from_neighbor_pores_voxels",
        "contact_active_aperture_radius_voxels": "contact_active_aperture_radius_voxels",
        "contact_active_aperture_all_contact_splits": "contact_active_aperture_all_contact_splits",
        "active_aperture_limited": "contact_active_aperture_limited",
        "shape_factor": "contact_shape_factor",
        "volume_voxels3": "contact_volume_voxels3",
        "raw_contact_gap_length_voxels": "contact_raw_contact_gap_length_voxels",
        "raw_dong_blunt_length_voxels": "contact_raw_dong_blunt_length_voxels",
        "raw_contact_thickness_length_voxels": "contact_raw_contact_thickness_length_voxels",
        "original_area_over_length_voxels": "contact_original_area_over_length_voxels",
        "alpha": "contact_dong_blunt_alpha",
        "min_effective_throat_length_voxels": "contact_min_effective_throat_length_voxels",
        "contact_gap_bound_threshold_voxels": "contact_gap_bound_threshold_voxels",
    }
    keep_columns = [column for column in rename if column in diagnostics.columns]
    diagnostics = diagnostics[keep_columns].rename(columns=rename)
    diagnostics = diagnostics.drop(columns=[column for column in ["pore1_id", "pore2_id"] if column in diagnostics])

    merged = throats.merge(diagnostics, on="throat_id", how="left")
    matched = merged["contact_length_voxels"].notna() if "contact_length_voxels" in merged else pd.Series(False)

    length_m = merged["contact_length_voxels"].astype(float) * voxel_size_m
    radius_m = merged["contact_radius_voxels"].astype(float) * voxel_size_m
    shape_factor = merged.get("contact_shape_factor", merged["throat_shape_factor"]).astype(float)
    active_area_m2 = radius_m.pow(2) / (4.0 * shape_factor)

    merged.loc[matched, "membrane_relaxation_length_m"] = length_m[matched]
    merged.loc[matched, "membrane_zdc_length_m"] = length_m[matched]
    merged.loc[matched, "membrane_active_radius_m"] = radius_m[matched]
    merged.loc[matched, "membrane_active_area_m2"] = active_area_m2[matched]
    merged.loc[matched, "membrane_geometry_source"] = "contact_patch_dong_blunt_diagnostics"

    for column in [
        "contact_radius_for_length_voxels",
        "contact_radius_unclamped_voxels",
        "contact_max_radius_from_neighbor_pores_voxels",
        "contact_raw_contact_gap_length_voxels",
        "contact_raw_dong_blunt_length_voxels",
        "contact_raw_contact_thickness_length_voxels",
        "contact_min_effective_throat_length_voxels",
        "contact_gap_bound_threshold_voxels",
    ]:
        if column in merged.columns:
            out_column = column.removesuffix("_voxels") + "_m"
            merged.loc[matched, out_column] = merged.loc[matched, column].astype(float) * voxel_size_m

    return merged, {
        "contact_patch_dong_blunt_diagnostics_merged": True,
        "contact_patch_dong_blunt_diagnostics_csv": str(diagnostics_path),
        "contact_patch_dong_blunt_diagnostics_rows": int(len(diagnostics)),
        "contact_patch_dong_blunt_matched_throats": int(matched.sum()),
        "contact_patch_dong_blunt_inferred_voxel_size_m": voxel_size_m,
        "membrane_geometry_mode": "explicit_contact_split_diagnostics",
    }


def parse_network(prefix: Path, outdir: Path) -> dict[str, object]:
    prefix = Path(prefix)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pores = read_node2(prefix.with_name(prefix.name + "_node2.dat"))
    node1_path = prefix.with_name(prefix.name + "_node1.dat")
    if node1_path.exists():
        centers = read_node1(node1_path)
        pores = pores.merge(centers, on="pore_id", how="left")
    link1 = read_link1(prefix.with_name(prefix.name + "_link1.dat"))
    link2 = read_link2(prefix.with_name(prefix.name + "_link2.dat"))
    throats = link1.merge(link2, on="throat_id", how="inner")
    diagnostics_path = prefix.with_name(prefix.name + "_contact_patch_dong_blunt_diagnostics.csv")
    throats, diagnostics_summary = merge_contact_split_dong_blunt_diagnostics(throats, diagnostics_path)

    pores.to_csv(outdir / "pores.csv", index=False)
    throats.to_csv(outdir / "throats.csv", index=False)

    summary = {
        "prefix": str(prefix),
        "n_pores": int(len(pores)),
        "n_throats": int(len(throats)),
        "pore_radius_volume_weighted_mean_m": float(np.average(pores["pore_radius_m"], weights=pores["pore_volume_m3"])),
        "throat_length_volume_weighted_mean_m": float(np.average(throats["throat_length_m"], weights=throats["throat_volume_m3"])),
        "pore_radius_min_m": float(pores["pore_radius_m"].min()),
        "pore_radius_max_m": float(pores["pore_radius_m"].max()),
        "throat_length_min_m": float(throats["throat_length_m"].min()),
        "throat_length_max_m": float(throats["throat_length_m"].max()),
        **diagnostics_summary,
    }
    (outdir / "network_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True, help="pnextract output prefix, e.g. path/Berea350_full")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    summary = parse_network(prefix=Path(args.prefix), outdir=Path(args.outdir))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

