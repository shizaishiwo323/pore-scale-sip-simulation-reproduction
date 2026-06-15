#!/usr/bin/env python3
"""Create a 3-D visualization of a micro-CT digital rock subvolume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_int3(values: list[str], name: str) -> tuple[int, int, int]:
    if len(values) != 3:
        raise ValueError(f"{name} expects exactly three integers")
    parsed = tuple(int(v) for v in values)
    if any(v < 0 for v in parsed):
        raise ValueError(f"{name} values must be non-negative")
    return parsed


def set_axes_equal(ax: plt.Axes, shape: tuple[int, int, int]) -> None:
    max_range = max(shape)
    centers = [n / 2.0 for n in shape]
    for setter, center in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centers):
        setter(center - max_range / 2.0, center + max_range / 2.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(PROJECT_ROOT / "data" / "microCT_Berea.raw"))
    parser.add_argument("--shape", nargs=3, default=["350", "350", "350"])
    parser.add_argument("--dtype", default="<u2")
    parser.add_argument("--pore-label", type=int, default=1)
    parser.add_argument("--solid-label", type=int, default=2)
    parser.add_argument("--crop-start", nargs=3, default=["115", "115", "115"])
    parser.add_argument("--crop-size", nargs=3, default=["120", "120", "120"])
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--phase", choices=["pore", "solid"], default="pore")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "figures" / "digital_rock_berea_3d_pore_crop.png"))
    parser.add_argument("--html-out", default="")
    parser.add_argument("--metadata-out", default=str(PROJECT_ROOT / "results" / "source_data" / "digital_rock_berea_3d_pore_crop_metadata.json"))
    args = parser.parse_args()

    shape = parse_int3(args.shape, "--shape")
    crop_start = parse_int3(args.crop_start, "--crop-start")
    crop_size = parse_int3(args.crop_size, "--crop-size")
    stop = tuple(s + n for s, n in zip(crop_start, crop_size))
    if any(e > full for e, full in zip(stop, shape)):
        raise ValueError(f"crop {crop_start}:{stop} exceeds volume shape {shape}")
    if args.downsample <= 0:
        raise ValueError("--downsample must be positive")

    volume = np.memmap(Path(args.raw), dtype=np.dtype(args.dtype), mode="r", shape=shape, order="C")
    crop = np.asarray(volume[tuple(slice(s, e) for s, e in zip(crop_start, stop))])
    label = args.pore_label if args.phase == "pore" else args.solid_label
    mask = crop == label
    if args.downsample > 1:
        mask = mask[:: args.downsample, :: args.downsample, :: args.downsample]

    field = mask.astype(np.float32)
    if field.min() == field.max():
        raise ValueError("selected crop contains only one phase; choose a larger or different crop")
    verts, faces, _normals, _values = measure.marching_cubes(field, level=0.5, spacing=(args.downsample,) * 3)

    fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    mesh = Poly3DCollection(verts[faces], alpha=0.42, linewidths=0.02)
    mesh.set_facecolor("#1F77B4" if args.phase == "pore" else "#FFFF00")
    mesh.set_edgecolor((0.95, 0.95, 0.95, 0.08))
    ax.add_collection3d(mesh)
    ax.view_init(elev=24, azim=38)
    ax.set_title(f"Berea digital rock: {args.phase} phase, {crop_size[0]}^3-voxel crop", fontsize=10, pad=8)
    set_axes_equal(ax, mask.shape)
    ax.set_axis_off()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260)
    plt.close(fig)

    html_out = Path(args.html_out) if args.html_out else None
    if html_out:
        import pyvista as pv

        faces_pv = np.column_stack(
            [np.full(len(faces), 3, dtype=np.int64), faces.astype(np.int64)]
        ).ravel()
        mesh_pv = pv.PolyData(verts, faces_pv)
        html_out.parent.mkdir(parents=True, exist_ok=True)
        plotter = pv.Plotter(off_screen=True, window_size=(1300, 950))
        plotter.set_background("white")
        plotter.add_mesh(
            mesh_pv,
            color="#1F77B4" if args.phase == "pore" else "#FFFF00",
            opacity=0.62 if args.phase == "pore" else 0.42,
            smooth_shading=True,
            ambient=0.35,
            diffuse=0.72,
            specular=0.34,
            specular_power=24,
        )
        plotter.add_bounding_box(color="#666666", line_width=1.0)
        plotter.camera_position = "iso"
        plotter.export_html(str(html_out))
        plotter.close()

    metadata = {
        "raw": str(Path(args.raw)),
        "shape": list(shape),
        "dtype": args.dtype,
        "phase": args.phase,
        "label": label,
        "crop_start": list(crop_start),
        "crop_size": list(crop_size),
        "downsample": args.downsample,
        "rendered_mask_shape": list(mask.shape),
        "phase_fraction_in_crop": float(np.mean(crop == label)),
        "mesh_vertices": int(len(verts)),
        "mesh_faces": int(len(faces)),
        "output_png": str(out),
        "output_html": str(html_out) if html_out else None,
    }
    metadata_out = Path(args.metadata_out)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
