#!/usr/bin/env python3
"""Convert a 3-D NetCDF phase field to a binary TIFF stack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.io import netcdf_file


def parse_labels(text: str) -> list[int]:
    labels = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not labels:
        raise argparse.ArgumentTypeError("expected at least one integer label")
    return labels


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    variable: str,
    pore_labels: list[int],
    solid_labels: list[int],
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    allowed = np.asarray(pore_labels + solid_labels)
    counts: dict[int, int] = {label: 0 for label in allowed.tolist()}

    with netcdf_file(input_path, "r", mmap=True) as dataset:
        if variable not in dataset.variables:
            raise KeyError(f"NetCDF variable not found: {variable}")
        phase = dataset.variables[variable]
        if len(phase.shape) != 3:
            raise ValueError(f"expected a 3-D phase field, got shape {phase.shape}")
        shape_zyx = tuple(int(value) for value in phase.shape)
        dimensions = [str(value) for value in phase.dimensions]
        voxel_size_xyz_um = [float(value) for value in np.asarray(dataset.voxel_size_xyz).ravel()]
        voxel_unit = dataset.voxel_unit.decode() if isinstance(dataset.voxel_unit, bytes) else str(dataset.voxel_unit)

        with tifffile.TiffWriter(output_path, bigtiff=True) as writer:
            for z_index in range(shape_zyx[0]):
                source = np.asarray(phase[z_index]).copy()
                values, value_counts = np.unique(source, return_counts=True)
                unexpected = values[~np.isin(values, allowed)]
                if unexpected.size:
                    raise ValueError(f"unexpected labels in z slice {z_index}: {unexpected.tolist()}")
                for value, count in zip(values, value_counts):
                    counts[int(value)] += int(count)
                binary = np.where(np.isin(source, pore_labels), 0, 255).astype(np.uint8)
                writer.write(binary, photometric="minisblack", compression="zlib", metadata=None)
        del source, phase
        dataset.variables.clear()

    pore_voxels = sum(counts[label] for label in pore_labels)
    total_voxels = int(np.prod(shape_zyx))
    manifest = {
        "input_netcdf": str(input_path.resolve()),
        "input_sha256": sha256(input_path),
        "output_tiff": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "variable": variable,
        "dimensions_zyx": dimensions,
        "shape_zyx": list(shape_zyx),
        "voxel_size_xyz_um": voxel_size_xyz_um,
        "voxel_unit": voxel_unit,
        "source_label_counts": {str(label): counts[label] for label in sorted(counts)},
        "mapping": {
            "pore_source_labels": pore_labels,
            "solid_source_labels": solid_labels,
            "output_pore_value": 0,
            "output_solid_value": 255,
        },
        "pore_voxels": pore_voxels,
        "solid_voxels": total_voxels - pore_voxels,
        "total_voxels": total_voxels,
        "macro_porosity": pore_voxels / total_voxels,
        "scientific_scope": "Only configured source labels are pore space; all configured mineral/clay labels are solid.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variable", default="phase")
    parser.add_argument("--pore-labels", type=parse_labels, default=[1])
    parser.add_argument("--solid-labels", type=parse_labels, default=[2, 3, 4, 5])
    args = parser.parse_args()
    print(
        json.dumps(
            convert(
                args.input,
                args.output,
                args.manifest,
                variable=args.variable,
                pore_labels=args.pore_labels,
                solid_labels=args.solid_labels,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
