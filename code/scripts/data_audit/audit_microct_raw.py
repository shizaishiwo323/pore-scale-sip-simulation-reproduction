#!/usr/bin/env python3
"""Audit a two-phase micro-CT raw volume without modifying the input file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_shape(values: list[str] | None) -> tuple[int, int, int] | None:
    if values is None:
        return None
    if len(values) != 3:
        raise ValueError("--shape expects exactly three integers")
    shape = tuple(int(v) for v in values)
    if any(v <= 0 for v in shape):
        raise ValueError("--shape values must be positive")
    return shape


def infer_cube_shape(raw_path: Path, dtype: np.dtype) -> tuple[int, int, int]:
    n_values, remainder = divmod(raw_path.stat().st_size, dtype.itemsize)
    if remainder:
        raise ValueError(f"{raw_path} size is not divisible by dtype itemsize {dtype.itemsize}")
    cube = round(n_values ** (1.0 / 3.0))
    if cube**3 != n_values:
        raise ValueError(
            f"cannot infer cubic shape from {raw_path.stat().st_size} bytes and dtype {dtype}; pass --shape"
        )
    return (cube, cube, cube)


def face_contacts(mask: np.ndarray) -> dict[str, int]:
    return {
        "x0": int(mask[0, :, :].any()),
        "x1": int(mask[-1, :, :].any()),
        "y0": int(mask[:, 0, :].any()),
        "y1": int(mask[:, -1, :].any()),
        "z0": int(mask[:, :, 0].any()),
        "z1": int(mask[:, :, -1].any()),
    }


def largest_component_fraction(mask: np.ndarray) -> tuple[int, int, float, dict[str, int]]:
    structure = ndimage.generate_binary_structure(3, 1)
    labels, n_labels = ndimage.label(mask, structure=structure)
    if n_labels == 0:
        return 0, 0, 0.0, {key: 0 for key in ("x0", "x1", "y0", "y1", "z0", "z1")}
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest_label = int(counts.argmax())
    largest_count = int(counts[largest_label])
    return n_labels, largest_count, largest_count / float(mask.sum()), face_contacts(labels == largest_label)


def write_slice_panel(volume: np.ndarray, pore_label: int, solid_label: int, out_dir: Path) -> None:
    mid = tuple(n // 2 for n in volume.shape)
    panels = [
        ("x_mid", volume[mid[0], :, :]),
        ("y_mid", volume[:, mid[1], :]),
        ("z_mid", volume[:, :, mid[2]]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.8), constrained_layout=True)
    for col, (name, slc) in enumerate(panels):
        axes[0, col].imshow(slc == pore_label, cmap="Blues", interpolation="nearest")
        axes[0, col].set_title(f"{name}: pore label {pore_label}", fontsize=9)
        axes[1, col].imshow(slc == solid_label, cmap="gray", interpolation="nearest")
        axes[1, col].set_title(f"{name}: solid label {solid_label}", fontsize=9)
        for row in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.savefig(out_dir / "center_slice_phase_audit.png", dpi=220)
    plt.close(fig)


def periodic_face_counts(volume: np.ndarray, pore_label: int, solid_label: int) -> pd.DataFrame:
    rows = []
    for axis, name in enumerate(("x", "y", "z")):
        neighbor = np.roll(volume, -1, axis=axis)
        pore_pore = int(np.count_nonzero((volume == pore_label) & (neighbor == pore_label)))
        solid_solid = int(np.count_nonzero((volume == solid_label) & (neighbor == solid_label)))
        pore_solid = int(volume.size - pore_pore - solid_solid)
        rows.append(
            {
                "axis": name,
                "pore_pore_faces": pore_pore,
                "solid_solid_faces": solid_solid,
                "pore_solid_faces": pore_solid,
                "pore_solid_fraction": pore_solid / volume.size,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(PROJECT_ROOT / "data" / "microCT_Berea.raw"))
    parser.add_argument("--shape", nargs=3, help="Optional 3-D shape. If omitted, infer a cubic shape from file size.")
    parser.add_argument("--dtype", default="<u2")
    parser.add_argument("--pore-label", type=int)
    parser.add_argument("--solid-label", type=int)
    parser.add_argument("--component-check", action="store_true")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "results" / "data_audit" / "microct_raw_audit"))
    args = parser.parse_args()

    raw_path = Path(args.raw)
    dtype = np.dtype(args.dtype)
    shape = parse_shape(args.shape) or infer_cube_shape(raw_path, dtype)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    volume = np.memmap(raw_path, dtype=dtype, mode="r", shape=shape, order="C")
    values, counts = np.unique(volume, return_counts=True)
    order = np.argsort(counts)
    pore_label = args.pore_label if args.pore_label is not None else int(values[order[0]])
    solid_label = args.solid_label if args.solid_label is not None else int(values[order[-1]])

    label_rows = []
    for value, count in zip(values, counts):
        row: dict[str, object] = {
            "label": int(value),
            "voxel_count": int(count),
            "volume_fraction": float(count / volume.size),
        }
        if args.component_check:
            n_components, largest_count, largest_fraction, contacts = largest_component_fraction(np.asarray(volume == value))
            row.update(
                {
                    "component_count_6_connected": int(n_components),
                    "largest_component_voxels": int(largest_count),
                    "largest_component_fraction_of_label": float(largest_fraction),
                    **{f"largest_component_touches_{k}": v for k, v in contacts.items()},
                }
            )
        label_rows.append(row)

    labels = pd.DataFrame(label_rows)
    faces = periodic_face_counts(volume, pore_label, solid_label)
    labels.to_csv(out_dir / "raw_label_counts.csv", index=False)
    faces.to_csv(out_dir / "raw_periodic_face_type_counts.csv", index=False)
    write_slice_panel(volume, pore_label, solid_label, out_dir)

    summary = {
        "raw_path": str(raw_path),
        "raw_size_bytes": raw_path.stat().st_size,
        "dtype": dtype.str,
        "shape": list(shape),
        "n_voxels": int(volume.size),
        "pore_label": pore_label,
        "solid_label": solid_label,
        "porosity_from_pore_label": float(labels.loc[labels["label"] == pore_label, "volume_fraction"].iloc[0]),
        "label_policy": "default pore label is the minority label; override with --pore-label if needed",
        "component_check": bool(args.component_check),
        "outputs": {
            "label_counts_csv": str(out_dir / "raw_label_counts.csv"),
            "periodic_face_counts_csv": str(out_dir / "raw_periodic_face_type_counts.csv"),
            "slice_panel_png": str(out_dir / "center_slice_phase_audit.png"),
        },
    }
    (out_dir / "raw_audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
