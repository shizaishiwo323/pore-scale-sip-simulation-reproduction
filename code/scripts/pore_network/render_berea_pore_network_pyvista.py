#!/usr/bin/env python3
"""Render the Berea pore network with true 3-D specular lighting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_network(pores_path: Path, throats_path: Path, voxel_size_m: float) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    pores = pd.read_csv(pores_path)
    throats = pd.read_csv(throats_path)
    coords_m = pores[["pore_center_x_m", "pore_center_y_m", "pore_center_z_m"]].to_numpy(dtype=float)
    coords_vox = coords_m / voxel_size_m
    pores = pores.copy()
    pores[["x_vox", "y_vox", "z_vox"]] = coords_vox
    pores["radius_vox"] = pores["pore_radius_m"].to_numpy(dtype=float) / voxel_size_m
    throats = throats.copy()
    throats["radius_vox"] = throats["throat_radius_m"].to_numpy(dtype=float) / voxel_size_m
    return pores, throats, coords_vox


def make_spheres(pores: pd.DataFrame, sphere_resolution: int, radius_scale: float) -> pv.PolyData:
    points = pores[["x_vox", "y_vox", "z_vox"]].to_numpy(dtype=float)
    cloud = pv.PolyData(points)
    cloud["glyph_scale"] = pores["radius_vox"].to_numpy(dtype=float) * radius_scale
    sphere = pv.Sphere(radius=1.0, theta_resolution=sphere_resolution, phi_resolution=sphere_resolution)
    return cloud.glyph(scale="glyph_scale", geom=sphere, orient=False)


def make_tubes(
    pores: pd.DataFrame,
    throats: pd.DataFrame,
    tube_resolution: int,
    tube_radius_scale: float,
    min_tube_radius_vox: float,
) -> tuple[pv.PolyData, int]:
    coords = pores.set_index("pore_id")[["x_vox", "y_vox", "z_vox"]]
    internal = throats[(throats["pore1_id"] > 0) & (throats["pore2_id"] > 0)].copy()
    internal = internal[
        internal["pore1_id"].isin(coords.index) & internal["pore2_id"].isin(coords.index)
    ]

    cylinders: list[pv.PolyData] = []
    for row in internal.itertuples(index=False):
        p1 = coords.loc[row.pore1_id].to_numpy(dtype=float)
        p2 = coords.loc[row.pore2_id].to_numpy(dtype=float)
        vector = p2 - p1
        length = float(np.linalg.norm(vector))
        if length <= 1e-9:
            continue
        radius = max(float(row.radius_vox) * tube_radius_scale, min_tube_radius_vox)
        cylinders.append(
            pv.Cylinder(
                center=(p1 + p2) / 2.0,
                direction=vector / length,
                radius=radius,
                height=length,
                resolution=tube_resolution,
            )
        )

    if not cylinders:
        return pv.PolyData(), 0
    return pv.MultiBlock(cylinders).combine(), len(cylinders)


def make_plotter(window_size: tuple[int, int]) -> pv.Plotter:
    pv.global_theme.background = "white"
    pv.global_theme.smooth_shading = True
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")
    plotter.remove_all_lights()
    plotter.add_light(pv.Light(position=(-500, -650, 900), focal_point=(175, 175, 175), intensity=1.55))
    plotter.add_light(pv.Light(position=(550, 250, 450), focal_point=(175, 175, 175), intensity=0.82))
    plotter.add_light(pv.Light(position=(175, 175, -300), focal_point=(175, 175, 175), intensity=0.45))
    return plotter


def camera_from_elev_azim(
    center: np.ndarray,
    extent: float,
    elev_deg: float,
    azim_deg: float,
    distance_scale: float,
) -> list[tuple[float, float, float]]:
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    distance = extent * distance_scale
    direction = np.array(
        [
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
        ],
        dtype=float,
    )
    position = center + direction * distance
    return [tuple(position), tuple(center), (0.0, 0.0, 1.0)]


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
    parser.add_argument("--out", default=str(PROJECT_ROOT / "figures" / "berea_pore_network_75deg_specular.png"))
    parser.add_argument(
        "--metadata-out",
        default=str(PROJECT_ROOT / "results" / "source_data" / "berea_pore_network_75deg_specular_metadata.json"),
    )
    parser.add_argument("--voxel-size-m", type=float, default=2.8e-6)
    parser.add_argument("--sphere-resolution", type=int, default=32)
    parser.add_argument("--tube-resolution", type=int, default=16)
    parser.add_argument("--sphere-radius-scale", type=float, default=1.05)
    parser.add_argument("--tube-radius-scale", type=float, default=0.70)
    parser.add_argument("--min-tube-radius-vox", type=float, default=0.55)
    parser.add_argument("--window-size", nargs=2, type=int, default=[1800, 1450])
    parser.add_argument("--zoom", type=float, default=1.05)
    parser.add_argument("--hide-bounds", action="store_true")
    parser.add_argument("--elev", type=float, default=18.0)
    parser.add_argument("--azim", type=float, default=75.0)
    parser.add_argument("--camera-distance-scale", type=float, default=2.65)
    parser.add_argument("--sphere-color", default="#ff0000")
    parser.add_argument("--tube-color", default="#003dff")
    args = parser.parse_args()

    pores_path = Path(args.pores)
    throats_path = Path(args.throats)
    pores, throats, coords_vox = load_network(pores_path, throats_path, args.voxel_size_m)
    spheres = make_spheres(pores, args.sphere_resolution, args.sphere_radius_scale)
    tubes, n_tubes_rendered = make_tubes(
        pores,
        throats,
        args.tube_resolution,
        args.tube_radius_scale,
        args.min_tube_radius_vox,
    )

    plotter = make_plotter(tuple(args.window_size))
    if tubes.n_points:
        plotter.add_mesh(
            tubes,
            color=args.tube_color,
            smooth_shading=True,
            ambient=0.42,
            diffuse=0.72,
            specular=0.42,
            specular_power=28,
        )
    plotter.add_mesh(
        spheres,
        color=args.sphere_color,
        smooth_shading=True,
        ambient=0.38,
        diffuse=0.76,
        specular=0.64,
        specular_power=46,
    )
    if not args.hide_bounds:
        plotter.show_bounds(
            grid="back",
            location="outer",
            color="#333333",
            font_size=18,
            n_xlabels=5,
            n_ylabels=5,
            n_zlabels=5,
        )
    center = coords_vox.mean(axis=0)
    extent = float(np.max(np.ptp(coords_vox, axis=0)))
    plotter.camera_position = camera_from_elev_azim(
        center=center,
        extent=extent,
        elev_deg=args.elev,
        azim_deg=args.azim,
        distance_scale=args.camera_distance_scale,
    )
    plotter.camera.zoom(args.zoom)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(out), transparent_background=False)
    plotter.close()

    metadata = {
        "input_pores_csv": str(pores_path),
        "input_throats_csv": str(throats_path),
        "output_png": str(out),
        "renderer": "pyvista/vtk",
        "material_parameters": {
            "sphere": {"ambient": 0.38, "diffuse": 0.76, "specular": 0.64, "specular_power": 46, "smooth_shading": True},
            "tube": {"ambient": 0.42, "diffuse": 0.72, "specular": 0.42, "specular_power": 28, "smooth_shading": True},
        },
        "zoom": args.zoom,
        "elev_degrees": args.elev,
        "azim_degrees": args.azim,
        "camera_distance_scale": args.camera_distance_scale,
        "hide_bounds": bool(args.hide_bounds),
        "sphere_color": args.sphere_color,
        "tube_color": args.tube_color,
        "n_pores_rendered": int(len(pores)),
        "n_throats_total": int(len(throats)),
        "n_internal_throats_rendered": int(n_tubes_rendered),
        "coordinate_units": "voxel",
        "voxel_size_m": args.voxel_size_m,
        "pore_radius_min_vox": float(pores["radius_vox"].min()),
        "pore_radius_max_vox": float(pores["radius_vox"].max()),
        "throat_radius_min_vox": float(throats["radius_vox"].min()),
        "throat_radius_max_vox": float(throats["radius_vox"].max()),
    }
    metadata_out = Path(args.metadata_out)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
