#!/usr/bin/env python3
"""Convert a labeled two-phase raw volume into pnextract's pore-0/solid-1 input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_shape(values: list[str]) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError("--shape expects exactly three integers")
    shape = tuple(int(v) for v in values)
    if any(v <= 0 for v in shape):
        raise ValueError("--shape values must be positive")
    return shape


def write_mhd(
    path: Path,
    raw_name: str,
    shape: tuple[int, int, int],
    voxel_size_um: float | None,
    title: str,
    write_elements: bool,
    write_vtk_network: bool,
) -> None:
    element_size = "1 1 1" if voxel_size_um is None else f"{voxel_size_um:g} {voxel_size_um:g} {voxel_size_um:g}"
    text = "\n".join(
        [
            "ObjectType = Image",
            "NDims = 3",
            "ElementType = MET_UCHAR",
            "ElementByteOrderMSB = False",
            f"DimSize = {shape[0]} {shape[1]} {shape[2]}",
            f"ElementSize = {element_size}",
            "Offset = 0 0 0",
            f"ElementDataFile = {raw_name}",
            "",
            f"title {title}",
            f"write_elements {'true' if write_elements else 'false'}",
            f"write_vtkNetwork {'true' if write_vtk_network else 'false'}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(PROJECT_ROOT / "data" / "microCT_Berea.raw"))
    parser.add_argument("--shape", nargs=3, default=["350", "350", "350"])
    parser.add_argument("--dtype", default="<u2")
    parser.add_argument("--pore-label", type=int, default=1)
    parser.add_argument("--solid-label", type=int, default=2)
    parser.add_argument("--voxel-size-um", type=float, default=None)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "pnextract_inputs" / "berea_from_raw"))
    parser.add_argument("--title", default="Berea_from_raw")
    parser.add_argument("--write-elements", action="store_true")
    parser.add_argument("--write-vtk-network", action="store_true")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    shape = parse_shape(args.shape)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_raw = out_dir / f"{args.title}_pore0_solid1.raw"
    out_mhd = out_dir / f"{args.title}.mhd"
    src = np.memmap(raw_path, dtype=np.dtype(args.dtype), mode="r", shape=shape, order="C")
    valid = (src == args.pore_label) | (src == args.solid_label)
    if not bool(np.all(valid)):
        bad = np.unique(np.asarray(src[~valid]))
        raise ValueError(f"unexpected labels in raw volume: {bad[:10].tolist()}")

    dst = np.memmap(out_raw, dtype="u1", mode="w+", shape=shape, order="C")
    dst[:] = np.where(src == args.pore_label, 0, 1).astype("u1")
    dst.flush()

    write_mhd(
        out_mhd,
        out_raw.name,
        shape,
        args.voxel_size_um,
        args.title,
        args.write_elements,
        args.write_vtk_network,
    )
    values, counts = np.unique(dst, return_counts=True)
    summary = {
        "source_raw": str(raw_path),
        "shape": list(shape),
        "source_dtype": args.dtype,
        "pore_label_source": args.pore_label,
        "solid_label_source": args.solid_label,
        "pnextract_mapping": {"0": "pore/water", "1": "solid"},
        "voxel_size_um_in_mhd": args.voxel_size_um,
        "out_raw": str(out_raw),
        "out_mhd": str(out_mhd),
        "out_raw_size_bytes": out_raw.stat().st_size,
        "mapped_counts": {str(int(v)): int(c) for v, c in zip(values, counts)},
    }
    (out_dir / f"{args.title}_prepare_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
