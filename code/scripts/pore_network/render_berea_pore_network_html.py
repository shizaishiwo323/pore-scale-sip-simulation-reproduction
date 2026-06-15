#!/usr/bin/env python3
"""Export an interactive PyVista/VTK HTML view of the Berea pore network."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MATERIALS = {
    "sphere": {
        "ambient": 0.38,
        "diffuse": 0.76,
        "specular": 0.64,
        "specular_power": 46,
        "smooth_shading": True,
    },
    "tube": {
        "ambient": 0.42,
        "diffuse": 0.72,
        "specular": 0.42,
        "specular_power": 28,
        "smooth_shading": True,
    },
}


def build_scene_payload(
    pores: pd.DataFrame,
    throats: pd.DataFrame,
    *,
    voxel_size_m: float,
    sphere_radius_scale: float,
    tube_radius_scale: float,
    min_tube_radius_vox: float,
    sphere_color: str = "#ff0000",
    tube_color: str = "#004cff",
    elev_deg: float = 18.0,
    azim_deg: float = 75.0,
    camera_distance_scale: float = 3.05,
    zoom: float = 0.92,
) -> dict:
    """Convert pnextract tables to voxel-unit geometry for PyVista export."""
    pores = pores.copy()
    coords_vox = pores[["pore_center_x_m", "pore_center_y_m", "pore_center_z_m"]].to_numpy(dtype=float) / voxel_size_m
    radii_vox = pores["pore_radius_m"].to_numpy(dtype=float) / voxel_size_m * sphere_radius_scale
    pores[["x_vox", "y_vox", "z_vox"]] = coords_vox
    pores["radius_vox"] = radii_vox

    coords = pores.set_index("pore_id")[["x_vox", "y_vox", "z_vox"]]
    internal = throats[(throats["pore1_id"] > 0) & (throats["pore2_id"] > 0)].copy()
    internal = internal[internal["pore1_id"].isin(coords.index) & internal["pore2_id"].isin(coords.index)]

    segments: list[list[float]] = []
    for row in internal.itertuples(index=False):
        p1 = coords.loc[row.pore1_id].to_numpy(dtype=float)
        p2 = coords.loc[row.pore2_id].to_numpy(dtype=float)
        if float(np.linalg.norm(p2 - p1)) <= 1e-9:
            continue
        radius_vox = max(float(row.throat_radius_m) / voxel_size_m * tube_radius_scale, min_tube_radius_vox)
        segments.append([*p1.tolist(), *p2.tolist(), radius_vox])

    mins = coords_vox.min(axis=0)
    maxs = coords_vox.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = float(np.max(maxs - mins))
    return {
        "pores": np.round(np.column_stack([coords_vox, radii_vox]), 6).tolist(),
        "segments": np.round(np.asarray(segments, dtype=float), 6).tolist() if segments else [],
        "bounds": {
            "min": np.round(mins, 6).tolist(),
            "max": np.round(maxs, 6).tolist(),
            "center": np.round(center, 6).tolist(),
            "extent": extent,
        },
        "counts": {
            "pores": int(len(pores)),
            "throats_total": int(len(throats)),
            "throats_rendered": int(len(segments)),
        },
        "style": {
            "sphereColor": sphere_color,
            "tubeColor": tube_color,
            "background": "#ffffff",
        },
        "camera": {
            "elevDeg": float(elev_deg),
            "azimDeg": float(azim_deg),
            "distanceScale": float(camera_distance_scale),
            "zoom": float(zoom),
        },
        "metadata": {
            "coordinate_units": "voxel",
            "voxel_size_m": float(voxel_size_m),
            "sphere_radius_scale": float(sphere_radius_scale),
            "tube_radius_scale": float(tube_radius_scale),
            "min_tube_radius_vox": float(min_tube_radius_vox),
            "note": "Boundary throats with negative pore ids are omitted from the rendering.",
        },
    }


def load_scene_payload(
    pores_path: Path,
    throats_path: Path,
    *,
    voxel_size_m: float,
    sphere_radius_scale: float,
    tube_radius_scale: float,
    min_tube_radius_vox: float,
    sphere_color: str,
    tube_color: str,
    elev_deg: float,
    azim_deg: float,
    camera_distance_scale: float,
    zoom: float,
) -> dict:
    pores = pd.read_csv(pores_path)
    throats = pd.read_csv(throats_path)
    payload = build_scene_payload(
        pores,
        throats,
        voxel_size_m=voxel_size_m,
        sphere_radius_scale=sphere_radius_scale,
        tube_radius_scale=tube_radius_scale,
        min_tube_radius_vox=min_tube_radius_vox,
        sphere_color=sphere_color,
        tube_color=tube_color,
        elev_deg=elev_deg,
        azim_deg=azim_deg,
        camera_distance_scale=camera_distance_scale,
        zoom=zoom,
    )
    payload["metadata"]["input_pores_csv"] = str(pores_path)
    payload["metadata"]["input_throats_csv"] = str(throats_path)
    return payload


def camera_from_elev_azim(center: np.ndarray, extent: float, elev_deg: float, azim_deg: float, distance_scale: float):
    elev = np.deg2rad(elev_deg)
    azim = np.deg2rad(azim_deg)
    direction = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)], dtype=float)
    position = center + direction * extent * distance_scale
    return [tuple(position), tuple(center), (0.0, 0.0, 1.0)]


def make_spheres(pores_xyzr: np.ndarray, sphere_resolution: int) -> pv.PolyData:
    points = pores_xyzr[:, :3]
    cloud = pv.PolyData(points)
    cloud["glyph_scale"] = pores_xyzr[:, 3]
    sphere = pv.Sphere(radius=1.0, theta_resolution=sphere_resolution, phi_resolution=sphere_resolution)
    return cloud.glyph(scale="glyph_scale", geom=sphere, orient=False)


def make_tubes(segments: np.ndarray, tube_resolution: int) -> pv.PolyData:
    cylinders: list[pv.PolyData] = []
    for x1, y1, z1, x2, y2, z2, radius in segments:
        p1 = np.array([x1, y1, z1], dtype=float)
        p2 = np.array([x2, y2, z2], dtype=float)
        vector = p2 - p1
        length = float(np.linalg.norm(vector))
        if length <= 1e-9:
            continue
        cylinders.append(
            pv.Cylinder(
                center=(p1 + p2) / 2.0,
                direction=vector / length,
                radius=float(radius),
                height=length,
                resolution=tube_resolution,
            )
        )
    return pv.MultiBlock(cylinders).combine() if cylinders else pv.PolyData()


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


def build_pyvista_plotter(
    payload: dict,
    *,
    hide_bounds: bool,
    window_size: tuple[int, int],
    sphere_resolution: int = 32,
    tube_resolution: int = 16,
) -> tuple[pv.Plotter, dict]:
    pores = np.asarray(payload["pores"], dtype=float)
    segments = np.asarray(payload["segments"], dtype=float) if payload["segments"] else np.empty((0, 7), dtype=float)
    plotter = make_plotter(window_size)

    actor_metadata: dict = {}
    tubes = make_tubes(segments, tube_resolution=tube_resolution)
    if tubes.n_points:
        actor_metadata["tube_actor"] = plotter.add_mesh(tubes, color=payload["style"]["tubeColor"], **MATERIALS["tube"])

    spheres = make_spheres(pores, sphere_resolution=sphere_resolution)
    actor_metadata["sphere_actor"] = plotter.add_mesh(spheres, color=payload["style"]["sphereColor"], **MATERIALS["sphere"])

    if not hide_bounds:
        plotter.show_bounds(
            grid="back",
            location="outer",
            color="#333333",
            font_size=18,
            n_xlabels=5,
            n_ylabels=5,
            n_zlabels=5,
        )

    center = np.asarray(payload["bounds"]["center"], dtype=float)
    plotter.camera_position = camera_from_elev_azim(
        center=center,
        extent=float(payload["bounds"]["extent"]),
        elev_deg=float(payload["camera"]["elevDeg"]),
        azim_deg=float(payload["camera"]["azimDeg"]),
        distance_scale=float(payload["camera"]["distanceScale"]),
    )
    plotter.camera.zoom(float(payload["camera"]["zoom"]))
    return plotter, actor_metadata


def export_html(plotter: pv.Plotter, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.export_html(out)


def ensure_vtk_global_alias(html_path: Path) -> None:
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    old = "window.global = window.global || {};"
    new = "var global = window.global = window.global || {};"
    if new in text:
        return
    if old in text:
        text = text.replace(old, new, 1)
    else:
        text = text.replace("<body>", f"<body>\n<script>{new}</script>", 1)
    html_path.write_text(text, encoding="utf-8")


def inject_vtk_wheel_zoom_support(html_path: Path) -> None:
    script = """
<div id="pore-network-wheel-zoom-support" data-vtk-zoom-panel style="
  position: fixed;
  left: 18px;
  bottom: 18px;
  z-index: 9999;
  display: flex;
  gap: 6px;
  padding: 7px;
  border-radius: 8px;
  border: 1px solid rgba(0,0,0,0.16);
  background: rgba(255,255,255,0.88);
  box-shadow: 0 8px 24px rgba(0,0,0,0.16);
  font-family: Arial, Helvetica, sans-serif;
">
  <button type="button" data-vtk-zoom="in" title="Zoom in" style="width:32px;height:30px;border:1px solid rgba(0,0,0,0.22);border-radius:6px;background:#fff;cursor:pointer;font-size:18px;line-height:1;">+</button>
  <button type="button" data-vtk-zoom="out" title="Zoom out" style="width:32px;height:30px;border:1px solid rgba(0,0,0,0.22);border-radius:6px;background:#fff;cursor:pointer;font-size:18px;line-height:1;">-</button>
  <button type="button" data-vtk-zoom="reset" title="Reset view" style="height:30px;border:1px solid rgba(0,0,0,0.22);border-radius:6px;background:#fff;cursor:pointer;font-size:12px;">Reset</button>
</div>
<script>
(function() {
  let displayScale = 1.0;
  let zoomClicks = 0;
  function getPanel() {
    return document.getElementById('pore-network-wheel-zoom-support');
  }
  function getVtkViewElement() {
    return document.querySelector('#vtk-root canvas') || document.querySelector('#vtk-root .content') || document.querySelector('#vtk-root');
  }
  function applyVtkDisplayScale() {
    const root = document.querySelector('#vtk-root');
    const view = getVtkViewElement();
    if (!root || !view) return false;
    root.style.overflow = 'hidden';
    view.style.transform = 'scale(' + displayScale.toFixed(4) + ')';
    view.style.transformOrigin = '50% 50%';
    view.style.transition = 'transform 120ms ease-out';
    view.dataset.vtkDisplayScale = displayScale.toFixed(4);
    const panel = getPanel();
    if (panel) {
      panel.dataset.vtkDisplayScale = displayScale.toFixed(4);
      panel.dataset.vtkZoomClicks = String(zoomClicks);
    }
    return true;
  }
  function zoomDisplay(factor) {
    displayScale = Math.max(0.35, Math.min(4.0, displayScale * factor));
    zoomClicks += 1;
    return applyVtkDisplayScale();
  }
  function resetDisplay() {
    displayScale = 1.0;
    return applyVtkDisplayScale();
  }
  function retryUntilReady(action, tries) {
    if (action()) return;
    if (tries > 0) window.setTimeout(function() { retryUntilReady(action, tries - 1); }, 250);
  }
  function bindZoom() {
    const target = document.querySelector('.content') || document.querySelector('#vtk-root') || document.body;
    if (!target) return;
    if (target.dataset.vtkWheelZoomBound !== '1') {
      target.dataset.vtkWheelZoomBound = '1';
      target.style.touchAction = 'none';
      target.addEventListener('wheel', function(event) {
        event.preventDefault();
        const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
        retryUntilReady(function() { return zoomDisplay(factor); }, 20);
      }, { passive: false });
    }
    document.querySelectorAll('[data-vtk-zoom]').forEach(function(button) {
      if (button.dataset.vtkZoomBound === '1') return;
      button.dataset.vtkZoomBound = '1';
      button.addEventListener('click', function() {
        const action = button.getAttribute('data-vtk-zoom');
        if (action === 'in') retryUntilReady(function() { return zoomDisplay(1.2); }, 20);
        else if (action === 'out') retryUntilReady(function() { return zoomDisplay(1 / 1.2); }, 20);
        else retryUntilReady(resetDisplay, 20);
      });
    });
    const panel = getPanel();
    if (panel) panel.dataset.vtkZoomReady = '1';
    retryUntilReady(applyVtkDisplayScale, 20);
  }
  window.poreNetworkZoomDisplay = zoomDisplay;
  window.setTimeout(bindZoom, 250);
  window.setTimeout(bindZoom, 1000);
})();
</script>
"""
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    if "pore-network-wheel-zoom-support" in text:
        return
    if "</body>" in text:
        text = text.replace("</body>", script + "\n</body>", 1)
    else:
        text += script
    html_path.write_text(text, encoding="utf-8")


def porosity_stats_from_volume(volume: np.ndarray, *, pore_value: int = 0, solid_value: int | None = None) -> dict:
    pore_voxels = int(np.count_nonzero(volume == pore_value))
    total_voxels = int(volume.size)
    solid_voxels = int(np.count_nonzero(volume == solid_value)) if solid_value is not None else None
    other_voxels = None if solid_voxels is None else int(total_voxels - pore_voxels - solid_voxels)
    return {
        "shape_zyx": [int(v) for v in volume.shape],
        "dtype": str(volume.dtype),
        "pore_value": int(pore_value),
        "solid_value": None if solid_value is None else int(solid_value),
        "pore_voxels": pore_voxels,
        "solid_voxels": solid_voxels,
        "other_voxels": other_voxels,
        "total_voxels": total_voxels,
        "porosity": float(pore_voxels / total_voxels) if total_voxels else 0.0,
    }


def load_porosity_stats(volume_path: Path, *, pore_value: int = 0, solid_value: int | None = None) -> dict:
    volume = tifffile.imread(volume_path)
    stats = porosity_stats_from_volume(volume, pore_value=pore_value, solid_value=solid_value)
    stats["segmented_volume"] = str(volume_path)
    stats["calculation"] = "count(volume == pore_value) / volume.size"
    return stats


def inject_porosity_overlay(html_path: Path, stats: dict) -> None:
    porosity_percent = float(stats["porosity"]) * 100.0
    total_voxels = int(stats["total_voxels"])
    pore_voxels = int(stats["pore_voxels"])
    overlay = (
        f"Porosity: {porosity_percent:.2f}%"
        f"<br><span>{pore_voxels:,} pore px / {total_voxels:,} total px</span>"
    )
    panel = f"""
<div id="pore-network-porosity-overlay" style="
  position: fixed;
  top: 16px;
  right: 16px;
  z-index: 9999;
  padding: 10px 12px;
  border-radius: 8px;
  background: rgba(255,255,255,0.88);
  color: #111827;
  font-family: Arial, sans-serif;
  font-size: 13px;
  line-height: 1.35;
  box-shadow: 0 8px 24px rgba(15,23,42,0.16);
  pointer-events: none;
">
  <strong>{overlay}</strong>
</div>
"""
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    if "pore-network-porosity-overlay" in text:
        return
    if "</body>" in text:
        text = text.replace("</body>", panel + "\n</body>", 1)
    else:
        text += panel
    html_path.write_text(text, encoding="utf-8")


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
        default=str(
            PROJECT_ROOT
            / "figures"
            / "berea_pore_network"
            / "berea_pore_network_75deg_specular_vivid_full_interactive.html"
        ),
    )
    parser.add_argument(
        "--metadata-out",
        default=str(
            PROJECT_ROOT
            / "results"
            / "source_data"
            / "berea_pore_network_75deg_specular_vivid_full_interactive_metadata.json"
        ),
    )
    parser.add_argument("--voxel-size-m", type=float, default=2.8e-6)
    parser.add_argument("--sphere-resolution", type=int, default=32)
    parser.add_argument("--tube-resolution", type=int, default=16)
    parser.add_argument("--sphere-radius-scale", type=float, default=1.05)
    parser.add_argument("--tube-radius-scale", type=float, default=0.70)
    parser.add_argument("--min-tube-radius-vox", type=float, default=0.55)
    parser.add_argument("--window-size", nargs=2, type=int, default=[1800, 1450])
    parser.add_argument("--elev", type=float, default=18.0)
    parser.add_argument("--azim", type=float, default=75.0)
    parser.add_argument("--camera-distance-scale", type=float, default=3.05)
    parser.add_argument("--zoom", type=float, default=0.92)
    parser.add_argument("--sphere-color", default="#ff0000")
    parser.add_argument("--tube-color", default="#004cff")
    parser.add_argument("--hide-bounds", action="store_true")
    parser.add_argument(
        "--segmented-volume",
        help="Optional segmented 3-D TIFF/volume used only to compute pixel-count porosity for the HTML overlay.",
    )
    parser.add_argument("--pore-value", type=int, default=0)
    parser.add_argument("--solid-value", type=int, default=255)
    args = parser.parse_args()

    pores_path = Path(args.pores)
    throats_path = Path(args.throats)
    payload = load_scene_payload(
        pores_path,
        throats_path,
        voxel_size_m=args.voxel_size_m,
        sphere_radius_scale=args.sphere_radius_scale,
        tube_radius_scale=args.tube_radius_scale,
        min_tube_radius_vox=args.min_tube_radius_vox,
        sphere_color=args.sphere_color,
        tube_color=args.tube_color,
        elev_deg=args.elev,
        azim_deg=args.azim,
        camera_distance_scale=args.camera_distance_scale,
        zoom=args.zoom,
    )

    plotter, _actor_metadata = build_pyvista_plotter(
        payload,
        hide_bounds=bool(args.hide_bounds),
        window_size=tuple(args.window_size),
        sphere_resolution=args.sphere_resolution,
        tube_resolution=args.tube_resolution,
    )
    out = Path(args.out)
    try:
        export_html(plotter, out)
    finally:
        plotter.close()
    ensure_vtk_global_alias(out)
    inject_vtk_wheel_zoom_support(out)

    porosity_stats = None
    if args.segmented_volume:
        porosity_stats = load_porosity_stats(
            Path(args.segmented_volume),
            pore_value=args.pore_value,
            solid_value=args.solid_value,
        )
        inject_porosity_overlay(out, porosity_stats)

    metadata = {
        "output_html": str(out),
        "renderer": "pyvista/vtk single-file html",
        "material_parameters": MATERIALS,
        **payload["metadata"],
        **payload["counts"],
        "style": payload["style"],
        "camera": payload["camera"],
        "html_export_dependency": "pyvista[jupyter] / trame-vtk",
        "porosity_from_voxel_count": porosity_stats,
    }
    metadata_out = Path(args.metadata_out)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
