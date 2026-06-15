#!/usr/bin/env python3
"""Prepare the Niu 2020 Berea TIFF volume as an AC3D label RAW/MHD input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    values = set(np.unique(frame).astype(int).tolist())
    if values.issubset({1, 2}):
        labels = frame.astype("<u2", copy=False)
    elif values.issubset({256, 512}):
        labels = (frame.astype(np.uint16) // np.uint16(256)).astype("<u2", copy=False)
    else:
        raise ValueError(f"unexpected TIFF labels {sorted(values)}; expected 1/2 or 256/512")
    return labels


def load_tiff_labels(path: Path) -> tuple[np.ndarray, dict[str, object]]:
    image = Image.open(path)
    n_frames = int(getattr(image, "n_frames", 1))
    frames: list[np.ndarray] = []
    raw_value_counts: dict[str, int] = {}
    normalized_value_counts: dict[str, int] = {}
    for index in range(n_frames):
        image.seek(index)
        frame = np.asarray(image)
        values, counts = np.unique(frame, return_counts=True)
        for value, count in zip(values, counts):
            raw_value_counts[str(int(value))] = raw_value_counts.get(str(int(value)), 0) + int(count)
        labels = normalize_frame(frame)
        values, counts = np.unique(labels, return_counts=True)
        for value, count in zip(values, counts):
            normalized_value_counts[str(int(value))] = normalized_value_counts.get(str(int(value)), 0) + int(count)
        frames.append(labels)
    volume = np.stack(frames, axis=0).astype("<u2", copy=False)
    info = {
        "format": image.format,
        "mode": image.mode,
        "frame_size_xy": list(image.size),
        "n_frames": n_frames,
        "shape_zyx": list(volume.shape),
        "raw_tiff_value_counts": raw_value_counts,
        "normalized_label_counts": normalized_value_counts,
    }
    return volume, info


def write_mhd(path: Path, raw_path: Path, shape_zyx: tuple[int, int, int], voxel_size_um: float) -> None:
    z, y, x = shape_zyx
    text = "\n".join(
        [
            "ObjectType = Image",
            "NDims = 3",
            "BinaryData = True",
            "BinaryDataByteOrderMSB = False",
            "CompressedData = False",
            f"DimSize = {x} {y} {z}",
            f"ElementSpacing = {voxel_size_um} {voxel_size_um} {voxel_size_um}",
            "ElementType = MET_USHORT",
            f"ElementDataFile = {raw_path.name}",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def compare_optional_raw(volume: np.ndarray, raw_path: Path | None) -> dict[str, object] | None:
    if raw_path is None or not raw_path.exists():
        return None
    reference = np.memmap(raw_path, dtype="<u2", mode="r", shape=volume.shape, order="C")
    matches = bool(np.array_equal(volume, np.asarray(reference)))
    return {
        "reference_raw": str(raw_path),
        "reference_raw_shape_zyx": list(reference.shape),
        "matches_reference_raw_exactly": matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiff", default=str(PROJECT_ROOT / "data" / "Niu 2020data" / "microCT_Berea.tiff"))
    parser.add_argument("--reference-raw", default=str(PROJECT_ROOT / "data" / "Niu 2020data" / "microCT_Berea.raw"))
    parser.add_argument("--voxel-size-um", type=float, default=2.8)
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "results" / "niu2020_berea_tiff_full350_input"),
    )
    args = parser.parse_args()

    tiff_path = Path(args.tiff)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out = out_dir / "microCT_Berea_from_tiff_labels_le_uint16.raw"
    mhd_out = out_dir / "microCT_Berea_from_tiff_labels_le_uint16.mhd"
    metadata_out = out_dir / "microCT_Berea_from_tiff_labels_le_uint16_metadata.json"

    volume, tiff_info = load_tiff_labels(tiff_path)
    raw_out.write_bytes(volume.tobytes(order="C"))
    write_mhd(mhd_out, raw_out, tuple(int(v) for v in volume.shape), args.voxel_size_um)
    metadata = {
        "source_tiff": str(tiff_path),
        "output_raw": str(raw_out),
        "output_mhd": str(mhd_out),
        "voxel_size_um": args.voxel_size_um,
        "pore_label": 1,
        "solid_label": 2,
        "tiff_normalization": "labels 256/512 are divided by 256 to recover Niu labels 1/2",
        "tiff_info": tiff_info,
        "reference_check": compare_optional_raw(volume, Path(args.reference_raw) if args.reference_raw else None),
    }
    metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {raw_out}")
    print(f"wrote {mhd_out}")
    print(f"wrote {metadata_out}")


if __name__ == "__main__":
    main()
