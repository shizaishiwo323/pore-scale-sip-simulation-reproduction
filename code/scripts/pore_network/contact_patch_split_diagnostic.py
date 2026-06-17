#!/usr/bin/env python3
"""Diagnose contact-patch splitting for pnextract pore-pair throats.

pnextract exports one throat for each unique pore pair. This diagnostic starts
from a pore-label volume, splits disconnected contact-face patches for each
pore pair, and compares the resulting experimental throat-length distribution
against the aggregated pnextract throats. It is intentionally diagnostic and
does not claim to reconstruct Niu's hidden network.
"""

import argparse
import gzip
import json
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT / "scripts" / "sip_simulation"))

from scan_pnextract_for_niu2020_membrane import read_figure5, weighted_distribution  # noqa: E402


@dataclass(frozen=True)
class PoreGeometry:
    pore_id: int
    center_voxels: np.ndarray
    radius_voxels: float


@dataclass(frozen=True)
class AggregatedThroat:
    pore1_id: int
    pore2_id: int
    length_voxels: float
    radius_voxels: float
    volume_voxels3: float


@dataclass(frozen=True)
class ContactPatch:
    pore1_id: int
    pore2_id: int
    patch_index: int
    axis: int
    face_count: int
    centroid_voxels: np.ndarray


def parse_mhd(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def element_type_to_dtype(element_type: str) -> np.dtype:
    mapping = {
        "MET_CHAR": np.int8,
        "MET_UCHAR": np.uint8,
        "MET_SHORT": np.int16,
        "MET_USHORT": np.uint16,
        "MET_INT": np.int32,
        "MET_UINT": np.uint32,
        "MET_FLOAT": np.float32,
        "MET_DOUBLE": np.float64,
    }
    if element_type not in mapping:
        raise ValueError(f"unsupported MHD ElementType: {element_type}")
    return np.dtype(mapping[element_type])


def read_mhd_volume(path: Path) -> tuple[np.ndarray, dict[str, str]]:
    meta = parse_mhd(path)
    dims = tuple(int(v) for v in meta["DimSize"].split())
    dtype = element_type_to_dtype(meta["ElementType"])
    raw_path = Path(meta["ElementDataFile"])
    if not raw_path.is_absolute():
        raw_path = path.parent / raw_path
    if meta.get("CompressedData", "False").lower() == "true" or raw_path.suffix == ".gz":
        with gzip.open(raw_path, "rb") as handle:
            data = np.frombuffer(handle.read(), dtype=dtype).copy()
    else:
        data = np.fromfile(raw_path, dtype=dtype)
    expected = int(np.prod(dims))
    if data.size != expected:
        raise ValueError(f"{raw_path} has {data.size} values, expected {expected}")
    return data.reshape(dims, order="F"), meta


def normalize_pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def doubled_face_midpoint(axis: int, i: int, j: int, k: int) -> tuple[int, int, int]:
    if axis == 0:
        return (2 * i + 1, 2 * j, 2 * k)
    if axis == 1:
        return (2 * i, 2 * j + 1, 2 * k)
    return (2 * i, 2 * j, 2 * k + 1)


def collect_contact_faces(labels: np.ndarray, pore_ids: set[int] | None = None) -> dict[tuple[int, int], list[tuple[int, tuple[int, int, int]]]]:
    labels = np.asarray(labels)
    faces: dict[tuple[int, int], list[tuple[int, tuple[int, int, int]]]] = defaultdict(list)
    for axis in (0, 1, 2):
        slicer_a = [slice(None)] * 3
        slicer_b = [slice(None)] * 3
        slicer_a[axis] = slice(0, labels.shape[axis] - 1)
        slicer_b[axis] = slice(1, labels.shape[axis])
        a = labels[tuple(slicer_a)]
        b = labels[tuple(slicer_b)]
        mask = (a != b) & (a >= 0) & (b >= 0)
        if pore_ids is not None:
            mask &= np.isin(a, list(pore_ids)) & np.isin(b, list(pore_ids))
        coords = np.argwhere(mask)
        for coord in coords:
            i, j, k = (int(coord[0]), int(coord[1]), int(coord[2]))
            p1 = int(a[i, j, k])
            p2 = int(b[i, j, k])
            faces[normalize_pair(p1, p2)].append((axis, doubled_face_midpoint(axis, i, j, k)))
    return faces


def connected_components(face_midpoints: list[tuple[int, tuple[int, int, int]]]) -> list[list[tuple[int, tuple[int, int, int]]]]:
    remaining = set(range(len(face_midpoints)))
    components: list[list[tuple[int, tuple[int, int, int]]]] = []
    points = [midpoint for _, midpoint in face_midpoints]
    point_to_indices: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, point in enumerate(points):
        point_to_indices[point].append(index)
    same_axis_offsets = []
    for delta in (-2, 2):
        same_axis_offsets.extend([(delta, 0, 0), (0, delta, 0), (0, 0, delta)])
    cross_axis_offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if abs(dx) + abs(dy) + abs(dz) == 2:
                    cross_axis_offsets.append((dx, dy, dz))
    while remaining:
        start = remaining.pop()
        queue: deque[int] = deque([start])
        component = [face_midpoints[start]]
        while queue:
            current = queue.popleft()
            cx, cy, cz = points[current]
            current_axis = face_midpoints[current][0]
            neighbor_candidates: set[int] = set()
            for dx, dy, dz in same_axis_offsets:
                neighbor_candidates.update(point_to_indices.get((cx + dx, cy + dy, cz + dz), []))
            for dx, dy, dz in cross_axis_offsets:
                neighbor_candidates.update(point_to_indices.get((cx + dx, cy + dy, cz + dz), []))
            for other in list(neighbor_candidates & remaining):
                other_axis = face_midpoints[other][0]
                same_axis_edge_neighbor = current_axis == other_axis
                cross_axis_edge_neighbor = current_axis != other_axis
                if not (same_axis_edge_neighbor or cross_axis_edge_neighbor):
                    continue
                remaining.remove(other)
                queue.append(other)
                component.append(face_midpoints[other])
        components.append(component)
    return components


def find_contact_patches(labels: np.ndarray, pore_ids: set[int] | None = None) -> list[ContactPatch]:
    patches: list[ContactPatch] = []
    for pair, faces in collect_contact_faces(labels, pore_ids=pore_ids).items():
        for patch_index, component in enumerate(connected_components(faces), start=1):
            midpoints = np.asarray([midpoint for _, midpoint in component], dtype=float) / 2.0
            axes = [axis for axis, _ in component]
            axis = max(set(axes), key=axes.count)
            patches.append(
                ContactPatch(
                    pore1_id=pair[0],
                    pore2_id=pair[1],
                    patch_index=patch_index,
                    axis=axis,
                    face_count=len(component),
                    centroid_voxels=midpoints.mean(axis=0),
                )
            )
    return patches


def pore_geometries_from_csv(path: Path, voxel_size_m: float) -> dict[int, PoreGeometry]:
    pores = pd.read_csv(path)
    required = {"pore_id", "pore_radius_m", "pore_center_x_m", "pore_center_y_m", "pore_center_z_m"}
    missing = required.difference(pores.columns)
    if missing:
        raise ValueError(f"missing pore columns in {path}: {sorted(missing)}")
    out: dict[int, PoreGeometry] = {}
    for row in pores.itertuples(index=False):
        pore_id = int(row.pore_id)
        out[pore_id] = PoreGeometry(
            pore_id=pore_id,
            center_voxels=np.array(
                [row.pore_center_x_m / voxel_size_m, row.pore_center_y_m / voxel_size_m, row.pore_center_z_m / voxel_size_m],
                dtype=float,
            ),
            radius_voxels=float(row.pore_radius_m / voxel_size_m),
        )
    return out


def throats_from_csv(path: Path, voxel_size_m: float) -> dict[tuple[int, int], AggregatedThroat]:
    throats = pd.read_csv(path)
    required = {"pore1_id", "pore2_id", "throat_length_m", "throat_radius_m", "throat_volume_m3"}
    missing = required.difference(throats.columns)
    if missing:
        raise ValueError(f"missing throat columns in {path}: {sorted(missing)}")
    out: dict[tuple[int, int], AggregatedThroat] = {}
    for row in throats.itertuples(index=False):
        p1 = int(row.pore1_id)
        p2 = int(row.pore2_id)
        if p1 < 1 or p2 < 1:
            continue
        pair = normalize_pair(p1, p2)
        out[pair] = AggregatedThroat(
            pore1_id=pair[0],
            pore2_id=pair[1],
            length_voxels=float(row.throat_length_m / voxel_size_m),
            radius_voxels=float(row.throat_radius_m / voxel_size_m),
            volume_voxels3=float(row.throat_volume_m3 / (voxel_size_m**3)),
        )
    return out


def summarize_patch_split(
    patches: Iterable[ContactPatch],
    *,
    pores: dict[int, PoreGeometry],
    throats: dict[tuple[int, int], AggregatedThroat],
    voxel_size_um: float,
    epsilon_voxels: float = 1.0e-6,
) -> list[dict[str, float | int]]:
    by_pair: dict[tuple[int, int], list[ContactPatch]] = defaultdict(list)
    for patch in patches:
        pair = normalize_pair(patch.pore1_id, patch.pore2_id)
        if pair in throats and pair[0] in pores and pair[1] in pores:
            by_pair[pair].append(patch)

    rows: list[dict[str, float | int]] = []
    for pair, pair_patches in sorted(by_pair.items()):
        throat = throats[pair]
        total_faces = sum(patch.face_count for patch in pair_patches)
        if total_faces <= 0:
            continue
        p1 = pores[pair[0]]
        p2 = pores[pair[1]]
        for patch in pair_patches:
            share = patch.face_count / total_faces
            d1 = float(np.linalg.norm(patch.centroid_voxels - p1.center_voxels))
            d2 = float(np.linalg.norm(patch.centroid_voxels - p2.center_voxels))
            split_length_voxels = max(d1 + d2 - p1.radius_voxels - p2.radius_voxels, epsilon_voxels)
            patch_radius_voxels = math.sqrt(max(patch.face_count, 1) / math.pi)
            rows.append(
                {
                    "pore1_id": pair[0],
                    "pore2_id": pair[1],
                    "patch_index": patch.patch_index,
                    "patch_count_for_pair": len(pair_patches),
                    "patch_face_count": patch.face_count,
                    "patch_area_voxels2": float(patch.face_count),
                    "patch_centroid_x_voxels": float(patch.centroid_voxels[0]),
                    "patch_centroid_y_voxels": float(patch.centroid_voxels[1]),
                    "patch_centroid_z_voxels": float(patch.centroid_voxels[2]),
                    "aggregated_length_voxels": throat.length_voxels,
                    "aggregated_length_um": throat.length_voxels * voxel_size_um,
                    "split_length_voxels": split_length_voxels,
                    "split_length_um": split_length_voxels * voxel_size_um,
                    "aggregated_radius_voxels": throat.radius_voxels,
                    "patch_equivalent_radius_voxels": patch_radius_voxels,
                    "split_volume_voxels3": throat.volume_voxels3 * share,
                    "split_volume_um3": throat.volume_voxels3 * share * voxel_size_um**3,
                }
            )
    return rows


def fraction_lt(values: np.ndarray, weights: np.ndarray, threshold_um: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (values > 0) & (weights > 0)
    if not mask.any():
        return 0.0
    denom = float(weights[mask].sum())
    return float(weights[mask & (values < threshold_um)].sum() / denom)


def write_outputs(rows: list[dict[str, float | int]], out_dir: Path, figure5_xlsx: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    split = pd.DataFrame(rows)
    split.to_csv(out_dir / "contact_patch_split_throats.csv", index=False)
    _, paper_throat = read_figure5(figure5_xlsx)
    paper_centers_um = paper_throat["center_m"].to_numpy(dtype=float) * 1.0e6
    paper_dist = paper_throat["relative_volume"].to_numpy(dtype=float)

    if split.empty:
        split_dist = np.zeros_like(paper_dist)
        metrics = {"n_split_throats": 0}
    else:
        split_lengths = split["split_length_um"].to_numpy(dtype=float)
        split_weights = split["split_volume_um3"].to_numpy(dtype=float)
        split_dist = weighted_distribution(split_lengths * 1.0e-6, split_weights, paper_throat["center_m"].to_numpy(dtype=float))
        metrics = {
            "n_split_throats": int(len(split)),
            "n_aggregated_pairs_with_patches": int(split[["pore1_id", "pore2_id"]].drop_duplicates().shape[0]),
            "n_pairs_with_multiple_patches": int(split.loc[split["patch_count_for_pair"] > 1, ["pore1_id", "pore2_id"]].drop_duplicates().shape[0]),
            "split_to_aggregated_throat_ratio": float(len(split) / max(1, split[["pore1_id", "pore2_id"]].drop_duplicates().shape[0])),
        }
        for threshold in (1.0, 5.0, 10.0):
            metrics[f"split_lt_{threshold:g}um_fraction"] = fraction_lt(split_lengths, split_weights, threshold)

    comparison = pd.DataFrame(
        {
            "length_center_um": paper_centers_um,
            "niu_figure5_relative_volume": paper_dist,
            "contact_patch_split_relative_volume": split_dist,
        }
    )
    comparison.to_csv(out_dir / "contact_patch_split_figure5_comparison.csv", index=False)
    for threshold in (1.0, 5.0, 10.0):
        metrics[f"paper_lt_{threshold:g}um_fraction"] = float(
            paper_dist[paper_centers_um < threshold].sum()
        )
    (out_dir / "contact_patch_split_summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--velems-mhd", required=True, type=Path)
    parser.add_argument("--pores-csv", required=True, type=Path)
    parser.add_argument("--throats-csv", required=True, type=Path)
    parser.add_argument("--voxel-size-um", type=float, default=2.8)
    parser.add_argument("--figure5-xlsx", type=Path, default=PROJECT_ROOT / "data" / "Niu 2020data" / "Figure5.xlsx")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "results" / "niu2020_berea_contact_patch_split_diagnostic_v1")
    parser.add_argument("--pnextract-label-offset", type=int, default=-1, help="Add this offset to positive VElems labels before matching parsed pore ids")
    args = parser.parse_args(argv)

    labels, meta = read_mhd_volume(args.velems_mhd)
    labels = labels.astype(np.int32, copy=False)
    if args.pnextract_label_offset:
        mapped = labels.copy()
        positive = mapped >= 0
        mapped[positive] += args.pnextract_label_offset
        labels = mapped

    voxel_size_m = args.voxel_size_um * 1.0e-6
    pores = pore_geometries_from_csv(args.pores_csv, voxel_size_m)
    throats = throats_from_csv(args.throats_csv, voxel_size_m)
    patches = find_contact_patches(labels, pore_ids=set(pores))
    rows = summarize_patch_split(patches, pores=pores, throats=throats, voxel_size_um=args.voxel_size_um)
    metrics = write_outputs(rows, args.out_dir, args.figure5_xlsx)
    metadata = {
        "velems_mhd": str(args.velems_mhd),
        "pores_csv": str(args.pores_csv),
        "throats_csv": str(args.throats_csv),
        "voxel_size_um": float(args.voxel_size_um),
        "pnextract_label_offset": int(args.pnextract_label_offset),
        "mhd_metadata": meta,
        "metrics": metrics,
        "interpretation": "diagnostic contact-patch split; not a formal pnextract network",
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
