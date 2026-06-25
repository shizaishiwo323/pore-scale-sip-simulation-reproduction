#!/usr/bin/env python3
"""Export ParaView-ready pore-network sources and a red-sphere/blue-tube style script."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pyvista as pv


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _verts_for_points(n_points: int) -> np.ndarray:
    verts = np.empty((int(n_points), 2), dtype=np.int64)
    verts[:, 0] = 1
    verts[:, 1] = np.arange(int(n_points), dtype=np.int64)
    return verts


def _lines_for_edges(edges: list[tuple[int, int]]) -> np.ndarray:
    lines = np.empty((len(edges), 3), dtype=np.int64)
    lines[:, 0] = 2
    for row_index, (pore1, pore2) in enumerate(edges):
        lines[row_index, 1] = int(pore1)
        lines[row_index, 2] = int(pore2)
    return lines


def build_paraview_sources(
    pores: pd.DataFrame,
    throats: pd.DataFrame,
    *,
    voxel_size_m: float,
    pore_radius_scale: float,
    throat_radius_scale: float,
    min_throat_radius_vox: float,
    max_throat_radius_vox: float | None = None,
    throat_radius_mode: str = "effective_radius",
    throat_diagnostics: pd.DataFrame | None = None,
) -> tuple[pv.PolyData, pv.PolyData, dict[str, object]]:
    pores = pores.copy()
    coords_vox = pores[["pore_center_x_m", "pore_center_y_m", "pore_center_z_m"]].to_numpy(dtype=float) / voxel_size_m
    pore_radius_vox = pores["pore_radius_m"].to_numpy(dtype=float) / voxel_size_m * pore_radius_scale
    id_to_index = {int(pore_id): index for index, pore_id in enumerate(pores["pore_id"].astype(int))}

    edges: list[tuple[int, int]] = []
    throat_radius_vox: list[float] = []
    internal = throats[(throats["pore1_id"] > 0) & (throats["pore2_id"] > 0)].copy()
    if throat_radius_mode != "effective_radius" and throat_diagnostics is None:
        raise ValueError(f"{throat_radius_mode} requires throat_diagnostics")
    if throat_diagnostics is not None:
        diagnostics = throat_diagnostics.copy()
        if "split_throat_id" not in diagnostics.columns:
            raise ValueError("throat diagnostics CSV must include split_throat_id")
        diagnostic_columns = [
            column
            for column in [
                "split_throat_id",
                "length_voxels",
                "volume_voxels3",
                "radius_unclamped_voxels",
                "radius_voxels",
            ]
            if column in diagnostics.columns
        ]
        internal = internal.merge(
            diagnostics[diagnostic_columns],
            left_on="throat_id",
            right_on="split_throat_id",
            how="left",
            suffixes=("", "_diagnostic"),
        )
    clipped_to_min = 0
    clipped_to_max = 0
    raw_radius_voxels: list[float] = []
    for row in internal.itertuples(index=False):
        pore1_id = int(row.pore1_id)
        pore2_id = int(row.pore2_id)
        if pore1_id not in id_to_index or pore2_id not in id_to_index:
            continue
        edges.append((id_to_index[pore1_id], id_to_index[pore2_id]))
        if throat_radius_mode == "effective_radius":
            raw_radius = float(row.throat_radius_m) / voxel_size_m
        elif throat_radius_mode == "diagnostic_unclamped_radius":
            raw_radius = float(getattr(row, "radius_unclamped_voxels", np.nan))
        elif throat_radius_mode == "diagnostic_volume_length_area":
            volume_voxels3 = float(getattr(row, "volume_voxels3", np.nan))
            length_voxels = float(getattr(row, "length_voxels", np.nan))
            raw_radius = float(np.sqrt(max(volume_voxels3 / max(length_voxels, 1e-12), 0.0) / np.pi))
        else:
            raise ValueError(f"unknown throat_radius_mode: {throat_radius_mode}")
        if not np.isfinite(raw_radius) or raw_radius <= 0:
            raw_radius = float(row.throat_radius_m) / voxel_size_m
        radius = raw_radius * throat_radius_scale
        if radius < min_throat_radius_vox:
            radius = min_throat_radius_vox
            clipped_to_min += 1
        if max_throat_radius_vox is not None and radius > max_throat_radius_vox:
            radius = max_throat_radius_vox
            clipped_to_max += 1
        raw_radius_voxels.append(float(raw_radius))
        throat_radius_vox.append(float(radius))

    pores_mesh = pv.PolyData(coords_vox)
    pores_mesh.verts = _verts_for_points(len(coords_vox))
    pores_mesh.point_data["pore_id"] = pores["pore_id"].to_numpy(dtype=np.int64)
    pores_mesh.point_data["pore_radius_vox"] = pore_radius_vox
    pores_mesh.point_data["pore_rgb"] = np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (len(coords_vox), 1))
    pores_mesh.set_active_scalars("pore_radius_vox", preference="point")

    throats_mesh = pv.PolyData(coords_vox, lines=_lines_for_edges(edges).ravel())
    throats_mesh.cell_data["throat_radius_vox"] = np.asarray(throat_radius_vox, dtype=float)
    throats_mesh.cell_data["throat_rgb"] = np.tile(np.array([[0, 56, 255]], dtype=np.uint8), (len(edges), 1))
    throats_mesh.set_active_scalars("throat_radius_vox", preference="cell")

    counts = {
        "pores": int(len(pores)),
        "throats_total": int(len(throats)),
        "throats_rendered": int(len(edges)),
        "throat_radius_mode": throat_radius_mode,
        "throat_radius_scale": float(throat_radius_scale),
        "min_throat_radius_vox": float(min_throat_radius_vox),
        "max_throat_radius_vox": float(max_throat_radius_vox) if max_throat_radius_vox is not None else None,
        "throat_radius_stats_vox": {
            "raw_min": float(np.min(raw_radius_voxels)) if raw_radius_voxels else None,
            "raw_median": float(np.median(raw_radius_voxels)) if raw_radius_voxels else None,
            "raw_p95": float(np.percentile(raw_radius_voxels, 95)) if raw_radius_voxels else None,
            "raw_max": float(np.max(raw_radius_voxels)) if raw_radius_voxels else None,
            "visual_min": float(np.min(throat_radius_vox)) if throat_radius_vox else None,
            "visual_median": float(np.median(throat_radius_vox)) if throat_radius_vox else None,
            "visual_p95": float(np.percentile(throat_radius_vox, 95)) if throat_radius_vox else None,
            "visual_max": float(np.max(throat_radius_vox)) if throat_radius_vox else None,
            "visual_unique_rounded_0p001": int(len(np.unique(np.round(throat_radius_vox, 3))))
            if throat_radius_vox
            else 0,
            "clipped_to_min_count": int(clipped_to_min),
            "clipped_to_max_count": int(clipped_to_max),
        },
    }
    return pores_mesh, throats_mesh, counts


def paraview_style_script(
    *,
    pores_vtp: Path,
    throats_vtp: Path,
    state_out: Path,
    screenshot_out: Path,
) -> str:
    return dedent(
        f"""\
        # Run inside ParaView: Tools > Python Shell > Run Script...
        from paraview.simple import *

        def try_set(obj, name, value):
            try:
                setattr(obj, name, value)
            except Exception:
                pass

        pores = XMLPolyDataReader(registrationName='red pore spheres source', FileName=[r'{pores_vtp}'])
        throats = XMLPolyDataReader(registrationName='blue throat tubes source', FileName=[r'{throats_vtp}'])

        view = GetActiveViewOrCreate('RenderView')
        view.Background = [1.0, 1.0, 1.0]
        view.OrientationAxesVisibility = 1
        try:
            view.AxesGrid = 'GridAxes3DActor'
            view.AxesGrid.Visibility = 1
            view.AxesGrid.XTitle = 'X Axis'
            view.AxesGrid.YTitle = 'Y Axis'
            view.AxesGrid.ZTitle = 'Z Axis'
            view.AxesGrid.GridColor = [0.0, 0.0, 0.0]
            view.AxesGrid.XTitleColor = [0.0, 0.0, 0.0]
            view.AxesGrid.YTitleColor = [0.0, 0.0, 0.0]
            view.AxesGrid.ZTitleColor = [0.0, 0.0, 0.0]
            view.AxesGrid.XLabelColor = [0.0, 0.0, 0.0]
            view.AxesGrid.YLabelColor = [0.0, 0.0, 0.0]
            view.AxesGrid.ZLabelColor = [0.0, 0.0, 0.0]
        except Exception:
            pass

        throat_tubes = Tube(registrationName='blue pore throats', Input=throats)
        throat_tubes.Scalars = ['CELLS', 'throat_radius_vox']
        throat_tubes.Vectors = [None, '']
        throat_tubes.Radius = 1.0
        throat_tubes.NumberofSides = 16
        throat_tubes.VaryRadius = 'By Absolute Scalar'

        pore_spheres = Glyph(registrationName='red pore nodes', Input=pores, GlyphType='Sphere')
        pore_spheres.OrientationArray = [None, '']
        pore_spheres.ScaleArray = ['POINTS', 'pore_radius_vox']
        pore_spheres.ScaleFactor = 1.0
        pore_spheres.GlyphMode = 'All Points'
        try_set(pore_spheres.GlyphType, 'ThetaResolution', 24)
        try_set(pore_spheres.GlyphType, 'PhiResolution', 24)

        throat_display = Show(throat_tubes, view, 'GeometryRepresentation')
        throat_display.Representation = 'Surface'
        throat_display.DiffuseColor = [0.0, 0.22, 1.0]
        throat_display.AmbientColor = [0.0, 0.22, 1.0]
        try_set(throat_display, 'Specular', 0.42)
        try_set(throat_display, 'SpecularPower', 28.0)

        pore_display = Show(pore_spheres, view, 'GeometryRepresentation')
        pore_display.Representation = 'Surface'
        pore_display.DiffuseColor = [1.0, 0.0, 0.0]
        pore_display.AmbientColor = [1.0, 0.0, 0.0]
        try_set(pore_display, 'Specular', 0.64)
        try_set(pore_display, 'SpecularPower', 46.0)

        Hide(pores, view)
        Hide(throats, view)
        view.ResetCamera(False)
        camera = view.GetActiveCamera()
        camera.Elevation(18)
        camera.Azimuth(75)
        camera.Zoom(0.92)
        Render()

        try:
            SaveState(r'{state_out}')
        except Exception as exc:
            print('SaveState skipped:', exc)
        try:
            SaveScreenshot(r'{screenshot_out}', view, ImageResolution=[1800, 1450])
        except Exception as exc:
            print('SaveScreenshot skipped:', exc)
        """
    )


def bake_single_file_geometry(
    pores_mesh: pv.PolyData,
    throats_mesh: pv.PolyData,
    *,
    sphere_resolution: int,
    tube_sides: int,
) -> pv.PolyData:
    sphere = pv.Sphere(radius=1.0, theta_resolution=int(sphere_resolution), phi_resolution=int(sphere_resolution))
    spheres = pores_mesh.glyph(scale="pore_radius_vox", geom=sphere, orient=False)
    spheres.cell_data["style_rgb"] = np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (spheres.n_cells, 1))
    spheres.cell_data["component_id"] = np.full(spheres.n_cells, 1, dtype=np.uint8)

    tubes = throats_mesh.tube(
        scalars="throat_radius_vox",
        capping=True,
        n_sides=int(tube_sides),
        absolute=True,
        preference="cell",
    )
    tubes.cell_data["style_rgb"] = np.tile(np.array([[0, 56, 255]], dtype=np.uint8), (tubes.n_cells, 1))
    tubes.cell_data["component_id"] = np.full(tubes.n_cells, 2, dtype=np.uint8)

    baked = tubes.merge(spheres, merge_points=False)
    baked.set_active_scalars("style_rgb", preference="cell")
    return baked


def readme_text(metadata: dict) -> str:
    return "\n".join(
        [
            "# ParaView Pore Network Style Package",
            "",
            "This package is for viewing a pnextract pore-network export in ParaView.",
            "",
            "## Open",
            "",
            "1. Open ParaView.",
            "2. Use `File > Open...` and select the direct-open baked VTP listed below.",
            "3. Set coloring to `style_rgb` if ParaView does not pick up the red/blue cell colors automatically.",
            "",
            "The Python style script and lightweight VTP sources are retained as an editable fallback for very large scenes or custom rendering.",
            "",
            "## Files",
            "",
            f"- Pores source: `{metadata['pores_vtp']}`",
            f"- Throats source: `{metadata['throats_vtp']}`",
            f"- Direct-open baked VTP: `{metadata.get('direct_open_vtp')}`",
            f"- ParaView script: `{metadata['paraview_style_script']}`",
            f"- State output target: `{metadata['paraview_state_target']}`",
            f"- Preview screenshot target: `{metadata['preview_screenshot_target']}`",
            "",
            "## Counts",
            "",
            f"- Pores: `{metadata['pores']}`",
            f"- Total throats: `{metadata['throats_total']}`",
            f"- Internal throats rendered: `{metadata['throats_rendered']}`",
            "",
            "The source files store coordinates in voxel units. Boundary throats with negative pore ids are not rendered as tubes.",
        ]
    )


def export_package(
    pores: pd.DataFrame,
    throats: pd.DataFrame,
    *,
    out_dir: Path,
    prefix: str,
    source_pores_csv: Path,
    source_throats_csv: Path,
    voxel_size_m: float,
    pore_radius_scale: float,
    throat_radius_scale: float,
    min_throat_radius_vox: float,
    max_throat_radius_vox: float | None = None,
    throat_radius_mode: str = "effective_radius",
    throat_diagnostics_csv: Path | None = None,
    write_baked: bool = False,
    baked_sphere_resolution: int = 12,
    baked_tube_sides: int = 8,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    throat_diagnostics = pd.read_csv(throat_diagnostics_csv) if throat_diagnostics_csv else None
    pores_mesh, throats_mesh, counts = build_paraview_sources(
        pores,
        throats,
        voxel_size_m=voxel_size_m,
        pore_radius_scale=pore_radius_scale,
        throat_radius_scale=throat_radius_scale,
        min_throat_radius_vox=min_throat_radius_vox,
        max_throat_radius_vox=max_throat_radius_vox,
        throat_radius_mode=throat_radius_mode,
        throat_diagnostics=throat_diagnostics,
    )

    pores_vtp = out_dir / f"{prefix}_paraview_pores_points.vtp"
    throats_vtp = out_dir / f"{prefix}_paraview_throats_lines.vtp"
    script_path = out_dir / "open_paraview_red_sphere_blue_tube_scene.py"
    state_out = out_dir / f"{prefix}_red_sphere_blue_tube_scene.pvsm"
    screenshot_out = out_dir / f"{prefix}_red_sphere_blue_tube_preview.png"
    metadata_json = out_dir / f"{prefix}_paraview_style_metadata.json"
    readme_path = out_dir / "README_paraview.md"
    direct_open_vtp = out_dir / f"{prefix}_red_sphere_blue_tube_direct_open.vtp"

    pores_mesh.save(pores_vtp, binary=True)
    throats_mesh.save(throats_vtp, binary=True)
    script_path.write_text(
        paraview_style_script(pores_vtp=pores_vtp, throats_vtp=throats_vtp, state_out=state_out, screenshot_out=screenshot_out),
        encoding="utf-8",
    )

    metadata = {
        "renderer": "paraview-glyph-tube-style",
        "pores_vtp": str(pores_vtp),
        "throats_vtp": str(throats_vtp),
        "paraview_style_script": str(script_path),
        "direct_open_vtp": str(direct_open_vtp) if write_baked else None,
        "direct_open_note": (
            "Open this one VTP directly in ParaView. It contains pre-baked red pore sphere surfaces "
            "and blue variable-radius throat tube surfaces with cell RGB colors."
            if write_baked
            else None
        ),
        "baked_single_file_vtu": None,
        "baked_single_file_note": None,
        "paraview_state_target": str(state_out),
        "preview_screenshot_target": str(screenshot_out),
        "readme": str(readme_path),
        "metadata_json": str(metadata_json),
        "input_pores_csv": str(source_pores_csv),
        "input_throats_csv": str(source_throats_csv),
        "input_throat_diagnostics_csv": str(throat_diagnostics_csv) if throat_diagnostics_csv else None,
        "coordinate_units": "voxel",
        "voxel_size_m": float(voxel_size_m),
        "pore_radius_scale": float(pore_radius_scale),
        "throat_radius_scale": float(throat_radius_scale),
        "min_throat_radius_vox": float(min_throat_radius_vox),
        "max_throat_radius_vox": float(max_throat_radius_vox) if max_throat_radius_vox is not None else None,
        "throat_radius_mode": throat_radius_mode,
        **counts,
        "notes": [
            "The VTP files are lightweight sources, not pre-baked sphere/tube triangle meshes.",
            "Run the ParaView Python script to reproduce red pore spheres, blue throat tubes, white background, and grid axes.",
            "When direct_open_vtp is present, it can be opened directly as a single ParaView file.",
            "The script attempts to save a PVSM state and PNG preview when run inside ParaView.",
        ],
    }
    if write_baked:
        baked = bake_single_file_geometry(
            pores_mesh,
            throats_mesh,
            sphere_resolution=baked_sphere_resolution,
            tube_sides=baked_tube_sides,
        )
        baked.save(direct_open_vtp, binary=True)
        metadata["direct_open_size_bytes"] = int(direct_open_vtp.stat().st_size)
        metadata["baked_sphere_resolution"] = int(baked_sphere_resolution)
        metadata["baked_tube_sides"] = int(baked_tube_sides)
        metadata["baked_points"] = int(baked.n_points)
        metadata["baked_cells"] = int(baked.n_cells)
    metadata_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    readme_path.write_text(readme_text(metadata), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pores", required=True)
    parser.add_argument("--throats", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="berea_pnextract")
    parser.add_argument("--voxel-size-m", type=float, default=2.8e-6)
    parser.add_argument("--pore-radius-scale", type=float, default=1.05)
    parser.add_argument("--throat-radius-scale", type=float, default=0.7)
    parser.add_argument("--min-throat-radius-vox", type=float, default=0.55)
    parser.add_argument("--max-throat-radius-vox", type=float)
    parser.add_argument(
        "--throat-radius-mode",
        choices=["effective_radius", "diagnostic_unclamped_radius", "diagnostic_volume_length_area"],
        default="effective_radius",
    )
    parser.add_argument("--throat-diagnostics-csv")
    parser.add_argument("--write-baked", action="store_true")
    parser.add_argument("--baked-sphere-resolution", type=int, default=12)
    parser.add_argument("--baked-tube-sides", type=int, default=8)
    args = parser.parse_args()

    pores_path = Path(args.pores)
    throats_path = Path(args.throats)
    metadata = export_package(
        pd.read_csv(pores_path),
        pd.read_csv(throats_path),
        out_dir=Path(args.out_dir),
        prefix=args.prefix,
        source_pores_csv=pores_path,
        source_throats_csv=throats_path,
        voxel_size_m=args.voxel_size_m,
        pore_radius_scale=args.pore_radius_scale,
        throat_radius_scale=args.throat_radius_scale,
        min_throat_radius_vox=args.min_throat_radius_vox,
        max_throat_radius_vox=args.max_throat_radius_vox,
        throat_radius_mode=args.throat_radius_mode,
        throat_diagnostics_csv=Path(args.throat_diagnostics_csv) if args.throat_diagnostics_csv else None,
        write_baked=bool(args.write_baked),
        baked_sphere_resolution=args.baked_sphere_resolution,
        baked_tube_sides=args.baked_tube_sides,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
