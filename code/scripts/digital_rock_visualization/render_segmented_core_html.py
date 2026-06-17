#!/usr/bin/env python3
"""Shared helpers for segmented-core PyVista/VTK HTML exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import tifffile
from skimage import measure


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data_inventory"
    / "ct_backed_samples_raw_copy_20260605"
    / "sample_89_Grainstone"
    / "CT_slices"
    / "89seged.tiff"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "segmented_cores" / "sample_89_segmented_core.html"
DEFAULT_METADATA = PROJECT_ROOT / "results" / "segmented_cores" / "sample_89_segmented_core_metadata.json"
DEFAULT_COMPONENT_COLORS = {0: "#1f77b4", 1: "#1f77b4", 2: "#f0c419", 128: "#9b59b6", 255: "#f0c419", 256: "#1f77b4", 512: "#f0c419"}
DEFAULT_VOXEL_OPACITY = {0: 0.22, 1: 0.22, 2: 0.42, 128: 0.34, 255: 0.42, 256: 0.22, 512: 0.42}
SOLID_MATERIAL = {"specular": 0.32, "specular_power": 24.0}


def load_tiff_volume(path: Path) -> np.ndarray:
    volume = tifffile.imread(path)
    if volume.ndim != 3:
        raise ValueError(f"expected a 3-D TIFF stack, got shape {volume.shape}")
    return np.asarray(volume)


def component_label(value: int) -> str:
    if int(value) in {0, 1, 256}:
        return "pore"
    if int(value) in {2, 255, 512}:
        return "solid"
    return f"component {int(value)}"


def binary_volume_stats(volume: np.ndarray, *, solid_value: int) -> dict:
    values, counts = np.unique(volume, return_counts=True)
    total = int(volume.size)
    solid = int(np.count_nonzero(volume == solid_value))
    pore_value = 0 if 0 in values else int(values[np.argmin(counts)])
    pore = int(np.count_nonzero(volume != solid_value))
    return {
        "shape_zyx": [int(v) for v in volume.shape],
        "solid_value": int(solid_value),
        "pore_value": int(pore_value),
        "solid_voxels": solid,
        "pore_voxels": pore,
        "total_voxels": total,
        "porosity": float(pore / total) if total else 0.0,
        "components": [
            {"value": int(v), "label": component_label(int(v)), "voxels": int(c), "fraction": float(c / total) if total else 0.0}
            for v, c in zip(values, counts)
        ],
    }


def _sample_component(volume: np.ndarray, *, component_value: int, downsample: int) -> np.ndarray:
    if downsample < 1:
        raise ValueError("downsample must be >= 1")
    return (volume[::downsample, ::downsample, ::downsample] == component_value)


def segmented_surface_mesh(
    volume: np.ndarray,
    *,
    component_value: int,
    downsample: int = 1,
    voxel_spacing_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> pv.PolyData:
    mask = _sample_component(volume, component_value=component_value, downsample=downsample).astype(np.float32)
    if mask.min() == mask.max():
        raise ValueError("selected component is empty or fills the whole sampled volume")
    spacing_zyx = tuple(float(v) * float(downsample) for v in voxel_spacing_um[::-1])
    verts, faces, _normals, _values = measure.marching_cubes(mask, level=0.5, spacing=spacing_zyx)
    verts_xyz = verts[:, [2, 1, 0]]
    faces_pv = np.column_stack([np.full(len(faces), 3, dtype=np.int64), faces.astype(np.int64)]).ravel()
    return pv.PolyData(verts_xyz, faces_pv)


def segmented_voxel_volume(
    volume: np.ndarray,
    *,
    component_value: int,
    downsample: int = 1,
    voxel_spacing_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> pv.ImageData:
    mask = _sample_component(volume, component_value=component_value, downsample=downsample).astype(np.uint8)
    sampled_xyz = np.transpose(mask, (2, 1, 0))
    sx, sy, sz = (float(value) * float(downsample) for value in voxel_spacing_um)
    grid = pv.ImageData(dimensions=sampled_xyz.shape, spacing=(sx, sy, sz))
    grid.point_data["component_mask"] = sampled_xyz.ravel(order="F")
    return grid


def make_plotter(window_size: tuple[int, int]) -> pv.Plotter:
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")
    return plotter


def build_segmented_core_plotter(mesh: pv.PolyData, *, window_size: tuple[int, int], hide_bounds: bool) -> tuple[pv.Plotter, object]:
    plotter = make_plotter(window_size)
    actor = plotter.add_mesh(
        mesh,
        color="#f0c419",
        opacity=0.62,
        smooth_shading=True,
        ambient=0.35,
        diffuse=0.72,
        specular=SOLID_MATERIAL["specular"],
        specular_power=SOLID_MATERIAL["specular_power"],
    )
    if not hide_bounds:
        plotter.add_bounding_box(color="#666666", line_width=1.0)
    plotter.camera_position = "iso"
    return plotter, actor


def export_html(plotter: pv.Plotter, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.export_html(str(out))


def _inject_once(html_path: Path, marker: str, snippet: str) -> None:
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    if marker in text:
        return
    if "</body>" in text:
        text = text.replace("</body>", snippet + "\n</body>", 1)
    else:
        text += snippet
    html_path.write_text(text, encoding="utf-8")


def inject_porosity_overlay(html_path: Path, stats: dict) -> None:
    porosity = float(stats.get("porosity", 0.0)) * 100.0
    snippet = f"""
<div id="segmented-core-porosity-overlay" style="position: fixed; right: 18px; top: 18px; z-index: 9999; background: rgba(255,255,255,0.88); padding: 8px 10px; border: 1px solid rgba(0,0,0,0.18); font-family: Arial, sans-serif; font-size: 13px;">
  <strong>Porosity</strong><br>{porosity:.2f}%
</div>"""
    _inject_once(html_path, "segmented-core-porosity-overlay", snippet)


def ensure_vtk_global_alias(html_path: Path) -> None:
    snippet = "<script>var global = window.global = window.global || {}; window.segmentedCoreActors = window.segmentedCoreActors || []; window.segmentedCoreVolumes = window.segmentedCoreVolumes || [];</script>"
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    if "var global = window.global = window.global || {};" not in text:
        html_path.write_text(snippet + "\n" + text, encoding="utf-8")


def inject_vtk_wheel_zoom_support(html_path: Path) -> None:
    snippet = """
<script id="segmented-core-wheel-zoom-support">
(function(){ window.vtkDisplayScale = window.vtkDisplayScale || 1; window.vtkZoomClicks = window.vtkZoomClicks || 0; function applyVtkDisplayScale(){ document.body.style.transformOrigin='50% 50%'; document.body.style.transform = 'scale(' + window.vtkDisplayScale + ')'; } window.addEventListener('wheel', function(e){ if(!e.ctrlKey) return; e.preventDefault(); window.vtkZoomClicks += e.deltaY < 0 ? 1 : -1; window.vtkDisplayScale = Math.max(0.4, Math.min(2.5, window.vtkDisplayScale + (e.deltaY < 0 ? 0.05 : -0.05))); applyVtkDisplayScale(); }, {passive:false}); window.applyVtkDisplayScale = applyVtkDisplayScale; })();
</script>"""
    _inject_once(html_path, "segmented-core-wheel-zoom-support", snippet)


def component_specs_from_values(values: list[int] | np.ndarray, *, visible_mode: str = "solid") -> list[dict]:
    specs = []
    for i, value in enumerate(int(v) for v in sorted(values)):
        specs.append(
            {
                "value": value,
                "label": component_label(value),
                "color": DEFAULT_COMPONENT_COLORS.get(value, "#8a8a8a"),
                "visible": visible_mode == "all" or (visible_mode == "solid" and component_label(value) == "solid") or (visible_mode == "pore" and component_label(value) == "pore") or (visible_mode == "first" and i == 0),
                "opacity": DEFAULT_VOXEL_OPACITY.get(value, 0.34),
            }
        )
    return specs


def inject_component_controls(html_path: Path, components: list[dict]) -> None:
    rows = "\n".join(
        f'<label><input class="segmented-core-component-visible" data-component-index="{i}" type="checkbox" {"checked" if item.get("visible") else ""}> {item["label"]} <input type="color" value="{item["color"]}"> <span>Surface mesh + Voxel volume</span></label>'
        for i, item in enumerate(components)
    )
    snippet = f"""
<div id="segmented-core-component-controls">{rows}<button id="segmented-core-apply-components">Apply</button><span>Waiting for VTK objects</span></div>
<script>window.segmentedCoreActors = window.segmentedCoreActors || []; window.segmentedCoreVolumes = window.segmentedCoreVolumes || []; window.segmentedCoreApplyControls = function(){{ window.segmentedCoreActors.forEach(function(item){{ if(item.setVisibility)item.setVisibility(true); if(item.getProperty&&item.getProperty().setColor)item.getProperty().setColor(1,1,1); }}); window.segmentedCoreVolumes.forEach(function(item){{ if(item.getMapper){{ var p=item.getProperty&&item.getProperty(); if(p&&p.getRGBTransferFunction)p.getRGBTransferFunction(); }} }}); }}; if (typeof global !== 'undefined') {{ window.segmentedCoreRenderWindow = global.renderWindow; }}</script>"""
    _inject_once(html_path, "segmented-core-component-controls", snippet)


def rendering_modes(*, include_surface: bool, include_voxel: bool) -> list[str]:
    return [name for enabled, name in ((include_surface, "surface_mesh"), (include_voxel, "voxel_volume")) if enabled]


def build_multi_component_plotter(
    volume: np.ndarray,
    *,
    component_specs: list[dict],
    downsample: int,
    smooth_iterations: int,
    voxel_spacing_um: tuple[float, float, float],
    window_size: tuple[int, int],
    hide_bounds: bool,
    include_surface: bool = True,
    include_voxel: bool = True,
    voxel_visible_mode: str = "none",
) -> tuple[pv.Plotter, list[object], list[dict]]:
    plotter = make_plotter(window_size)
    actors = []
    metadata = []
    for idx, spec in enumerate(component_specs):
        value = int(spec["value"])
        item = {"value": value, "surface_skipped": not include_surface, "voxel_actor_index": None, "voxel_points": 0, "voxel_visible": False, "voxel_sample_distance_um": 0.0}
        if include_surface:
            try:
                mesh = segmented_surface_mesh(volume, component_value=value, downsample=downsample, voxel_spacing_um=voxel_spacing_um)
                actor = plotter.add_mesh(mesh, color=spec.get("color", "#8a8a8a"), opacity=0.55, smooth_shading=True)
                actors.append(actor)
            except ValueError:
                item["surface_skipped"] = True
        if include_voxel:
            grid = segmented_voxel_volume(volume, component_value=value, downsample=downsample, voxel_spacing_um=voxel_spacing_um)
            actor = plotter.add_volume(grid, scalars="component_mask", opacity=[0.0, float(spec.get("opacity", 0.34))], show_scalar_bar=False)
            visible = voxel_visible_mode == "all" or (voxel_visible_mode == "solid" and component_label(value) == "solid") or (voxel_visible_mode == "pore" and component_label(value) == "pore")
            actor.SetVisibility(bool(visible))
            item.update({"voxel_actor_index": idx, "voxel_points": int(grid.n_points), "voxel_visible": bool(visible), "voxel_sample_distance_um": float(max(grid.spacing))})
        metadata.append(item)
    if not hide_bounds:
        plotter.add_bounding_box()
    plotter.camera_position = "iso"
    return plotter, actors, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--metadata-out", default=str(DEFAULT_METADATA))
    parser.add_argument("--solid-value", type=int, default=255)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--voxel-size-um", type=float, default=1.7)
    parser.add_argument("--window-size", nargs=2, type=int, default=[1200, 900])
    parser.add_argument("--components", nargs="*", type=int, default=None)
    parser.add_argument("--initial-visible", choices=["pore", "solid", "all", "first"], default="solid")
    parser.add_argument("--hide-bounds", action="store_true")
    args = parser.parse_args()

    volume = load_tiff_volume(Path(args.input))
    stats = binary_volume_stats(volume, solid_value=args.solid_value)
    components = component_specs_from_values(args.components if args.components is not None else [item["value"] for item in stats["components"]], visible_mode=args.initial_visible)
    plotter, _actors, render_metadata = build_multi_component_plotter(
        volume,
        component_specs=components,
        downsample=args.downsample,
        smooth_iterations=0,
        voxel_spacing_um=(args.voxel_size_um, args.voxel_size_um, args.voxel_size_um),
        window_size=tuple(args.window_size),
        hide_bounds=bool(args.hide_bounds),
        include_surface=True,
        include_voxel=True,
        voxel_visible_mode=args.initial_visible,
    )
    out = Path(args.out)
    try:
        export_html(plotter, out)
    finally:
        plotter.close()
    ensure_vtk_global_alias(out)
    inject_vtk_wheel_zoom_support(out)
    inject_porosity_overlay(out, stats)
    inject_component_controls(out, components)
    metadata = {"input_tiff": str(Path(args.input)), "output_html": str(out), "renderer": "pyvista/vtk single-file html", "downsample": int(args.downsample), **stats, "component_controls": components, "render_metadata": render_metadata}
    metadata_out = Path(args.metadata_out)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
