#!/usr/bin/env python3
"""Render a sphere-and-tube style pore-network view for the Berea micro-CT data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def percentile_scale(values: np.ndarray, out_min: float, out_max: float) -> np.ndarray:
    lo, hi = np.percentile(values, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full_like(values, (out_min + out_max) / 2.0, dtype=float)
    scaled = (values - lo) / (hi - lo)
    return out_min + np.clip(scaled, 0.0, 1.0) * (out_max - out_min)


def set_axes_equal(ax: plt.Axes, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(c - radius, c + radius)


def build_internal_segments(pores: pd.DataFrame, throats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    coords = pores.set_index("pore_id")[["pore_center_x_m", "pore_center_y_m", "pore_center_z_m"]]
    internal = throats[(throats["pore1_id"] > 0) & (throats["pore2_id"] > 0)].copy()
    internal = internal[
        internal["pore1_id"].isin(coords.index) & internal["pore2_id"].isin(coords.index)
    ]
    p1 = coords.loc[internal["pore1_id"].to_numpy()].to_numpy(dtype=float)
    p2 = coords.loc[internal["pore2_id"].to_numpy()].to_numpy(dtype=float)
    return np.stack([p1, p2], axis=1), internal["throat_radius_m"].to_numpy(dtype=float)


def style_box_axes(ax: plt.Axes) -> None:
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.tick_params(axis="both", which="major", labelsize=9, pad=0, colors="#2F2F2F")
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.xaxis.pane.set_edgecolor("#222222")
    ax.yaxis.pane.set_edgecolor("#222222")
    ax.zaxis.pane.set_edgecolor("#222222")
    ax.grid(True, color="#D8D8D8", linewidth=0.35)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pores",
        default=str(
            PROJECT_ROOT
            / "results"
            / "pnextract"
            / "rawdata_pnextract_rerun_20260527"
            / "network_parsed"
            / "pores.csv"
        ),
    )
    parser.add_argument(
        "--throats",
        default=str(
            PROJECT_ROOT
            / "results"
            / "pnextract"
            / "rawdata_pnextract_rerun_20260527"
            / "network_parsed"
            / "throats.csv"
        ),
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "figures" / "berea_pore_network_75deg_pnextract.png"),
    )
    parser.add_argument(
        "--metadata-out",
        default=str(PROJECT_ROOT / "results" / "source_data" / "berea_pore_network_75deg_pnextract_metadata.json"),
    )
    parser.add_argument("--elev", type=float, default=18.0)
    parser.add_argument("--azim", type=float, default=75.0)
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--style", choices=["depth", "james", "james_glossy"], default="depth")
    parser.add_argument("--voxel-size-m", type=float, default=2.8e-6)
    args = parser.parse_args()

    pores = pd.read_csv(args.pores)
    throats = pd.read_csv(args.throats)
    xyz = pores[["pore_center_x_m", "pore_center_y_m", "pore_center_z_m"]].to_numpy(dtype=float)
    radii = pores["pore_radius_m"].to_numpy(dtype=float)
    segments, throat_radii = build_internal_segments(pores, throats)
    if args.style in {"james", "james_glossy"}:
        xyz = xyz / args.voxel_size_m
        segments = segments / args.voxel_size_m

    node_sizes = (
        percentile_scale(radii, 9.0, 135.0)
        if args.style in {"james", "james_glossy"}
        else percentile_scale(radii, 10.0, 190.0)
    )
    line_widths = (
        percentile_scale(throat_radii, 0.35, 3.2)
        if args.style in {"james", "james_glossy"}
        else percentile_scale(throat_radii, 0.18, 2.1)
    )

    fig = plt.figure(figsize=(8.6, 7.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    lc = LineCollection(
        segments[:, :, :2],
        colors=(0.0, 0.05, 0.95, 0.82)
        if args.style in {"james", "james_glossy"}
        else (0.16, 0.18, 0.20, 0.24),
        linewidths=line_widths,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_collection3d(lc, zs=segments[:, :, 2], zdir="z")

    depth_color = (xyz[:, 2] - xyz[:, 2].min()) / max(np.ptp(xyz[:, 2]), 1e-30)
    if args.style in {"james", "james_glossy"}:
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=node_sizes,
            c="#D90000" if args.style == "james_glossy" else "#E50000",
            edgecolors="#7A0000" if args.style == "james_glossy" else "#9A0000",
            linewidths=0.24,
            alpha=0.96,
            depthshade=True,
        )
        if args.style == "james_glossy":
            # A small light-facing offset creates visible specular dots while
            # keeping the render fast enough for thousands of pore bodies.
            center = xyz.mean(axis=0)
            span = np.max(np.ptp(xyz, axis=0))
            light_offset = np.array([-0.012, -0.018, 0.022]) * span
            front_bias = 0.16 * (xyz - center)
            highlight_xyz = xyz + light_offset + front_bias * (node_sizes / node_sizes.max())[:, None]
            ax.scatter(
                highlight_xyz[:, 0],
                highlight_xyz[:, 1],
                highlight_xyz[:, 2],
                s=np.maximum(node_sizes * 0.20, 2.0),
                c="white",
                edgecolors="none",
                alpha=0.82,
                depthshade=False,
            )
            ax.scatter(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                s=node_sizes * 0.48,
                c="#FF3B20",
                edgecolors="none",
                alpha=0.18,
                depthshade=False,
            )
    else:
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=node_sizes,
            c=depth_color,
            cmap="turbo",
            edgecolors=(0.04, 0.04, 0.04, 0.35),
            linewidths=0.22,
            alpha=0.92,
            depthshade=True,
        )

    set_axes_equal(ax, xyz)
    ax.view_init(elev=args.elev, azim=args.azim)
    if args.style in {"james", "james_glossy"}:
        style_box_axes(ax)
    else:
        ax.set_axis_off()
        ax.grid(False)
    ax.set_proj_type("persp", focal_length=0.95)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    metadata = {
        "input_pores_csv": str(Path(args.pores)),
        "input_throats_csv": str(Path(args.throats)),
        "output_png": str(out),
        "view_elev_degrees": args.elev,
        "view_azim_degrees": args.azim,
        "style": args.style,
        "n_pores_rendered": int(len(pores)),
        "n_throats_total": int(len(throats)),
        "n_internal_throats_rendered": int(len(segments)),
        "pore_radius_min_m": float(np.min(radii)),
        "pore_radius_max_m": float(np.max(radii)),
        "throat_radius_min_m": float(np.min(throat_radii)) if len(throat_radii) else None,
        "throat_radius_max_m": float(np.max(throat_radii)) if len(throat_radii) else None,
        "note": "Boundary throats with negative pore ids are omitted from the rendering.",
    }
    metadata_out = Path(args.metadata_out)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
