#!/usr/bin/env python3
"""Export an interactive PyVista/VTK HTML view of a segmented CT core."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pyvista as pv
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data_inventory"
    / "ct_backed_samples_raw_copy_20260605"
    / "sample_89_Grainstone"
    / "CT_slices"
    / "89seged.tiff"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "figures" / "segmented_cores" / "sample_89_Grainstone_89seged_solid255_pore0_interactive.html"
DEFAULT_METADATA = (
    PROJECT_ROOT
    / "results"
    / "source_data"
    / "sample_89_Grainstone_89seged_solid255_pore0_interactive_metadata.json"
)

SOLID_MATERIAL = {
    "ambient": 0.38,
    "diffuse": 0.76,
    "specular": 0.64,
    "specular_power": 46,
    "smooth_shading": True,
}
DEFAULT_COMPONENT_COLORS = {
    0: "#0077ff",
    255: "#ffff00",
}
FALLBACK_COLORS = ["#f0c419", "#30b36b", "#9b59b6", "#ff7f0e", "#17becf", "#8c564b"]
DEFAULT_VOXEL_OPACITY = {
    0: 0.46,
    255: 0.30,
}


def load_tiff_volume(path: Path) -> np.ndarray:
    volume = tifffile.imread(path)
    volume = np.squeeze(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D TIFF stack, got shape {volume.shape}")
    return volume


def binary_volume_stats(volume: np.ndarray, *, solid_value: int) -> dict:
    values, counts = np.unique(volume, return_counts=True)
    component_counts = {int(value): int(count) for value, count in zip(values, counts)}
    solid_voxels = int(component_counts.get(int(solid_value), 0))
    pore_voxels = int(component_counts.get(0, 0))
    return {
        "shape_zyx": [int(v) for v in volume.shape],
        "dtype": str(volume.dtype),
        "solid_value": int(solid_value),
        "pore_value": 0,
        "solid_voxels": solid_voxels,
        "pore_voxels": pore_voxels,
        "other_voxels": int(volume.size - solid_voxels - pore_voxels),
        "solid_fraction": float(solid_voxels / volume.size),
        "porosity": float(pore_voxels / volume.size),
        "components": [
            {"value": int(value), "voxels": int(count), "fraction": float(count / volume.size)}
            for value, count in zip(values, counts)
        ],
    }


def component_label(value: int) -> str:
    if value == 0:
        return "pore"
    if value == 255:
        return "solid"
    return f"component {value}"


def component_specs_from_values(values: list[int] | np.ndarray, *, visible_mode: str = "pore") -> list[dict]:
    specs: list[dict] = []
    sorted_values = [int(v) for v in sorted(values)]
    fallback_i = 0
    for index, value in enumerate(sorted_values):
        color = DEFAULT_COMPONENT_COLORS.get(value)
        if color is None:
            color = FALLBACK_COLORS[fallback_i % len(FALLBACK_COLORS)]
            fallback_i += 1
        visible = (
            visible_mode == "all"
            or (visible_mode == "pore" and value == 0)
            or (visible_mode == "solid" and value == 255)
            or (visible_mode == "first" and index == 0)
        )
        specs.append({"value": value, "label": component_label(value), "color": color, "visible": bool(visible)})
    return specs


def mode_matches_component(mode: str, *, value: int, index: int) -> bool:
    return (
        mode == "all"
        or (mode == "pore" and value == 0)
        or (mode == "solid" and value == 255)
        or (mode == "first" and index == 0)
    )


def rendering_modes(*, include_surface: bool, include_voxel: bool) -> list[str]:
    modes: list[str] = []
    if include_surface:
        modes.append("surface_mesh")
    if include_voxel:
        modes.append("voxel_volume")
    return modes


def segmented_surface_mesh(
    volume: np.ndarray,
    *,
    component_value: int,
    downsample: int,
    smooth_iterations: int = 12,
    voxel_spacing_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> pv.PolyData:
    if downsample < 1:
        raise ValueError("downsample must be >= 1")
    phase_zyx = (volume[::downsample, ::downsample, ::downsample] == component_value).astype(np.float32)
    if not np.any(phase_zyx):
        raise ValueError(f"No voxels equal component_value={component_value}")
    if np.all(phase_zyx):
        raise ValueError("The downsampled volume is entirely this component; no interface can be contoured")

    phase_xyz = np.transpose(phase_zyx, (2, 1, 0))
    nx, ny, nz = phase_xyz.shape
    sx, sy, sz = (float(v) * float(downsample) for v in voxel_spacing_um)
    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(sx, sy, sz))
    grid.point_data["component"] = phase_xyz.ravel(order="F")
    mesh = grid.contour([0.5], scalars="component").clean()
    if smooth_iterations > 0 and mesh.n_points:
        mesh = mesh.smooth(n_iter=smooth_iterations, relaxation_factor=0.08, boundary_smoothing=False)
    return mesh


def segmented_voxel_volume(
    volume: np.ndarray,
    *,
    component_value: int,
    downsample: int,
    voxel_spacing_um: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> pv.ImageData:
    if downsample < 1:
        raise ValueError("downsample must be >= 1")
    phase_zyx = (volume[::downsample, ::downsample, ::downsample] == component_value).astype(np.uint8)
    if not np.any(phase_zyx):
        raise ValueError(f"No voxels equal component_value={component_value}")

    phase_xyz = np.transpose(phase_zyx, (2, 1, 0))
    nx, ny, nz = phase_xyz.shape
    sx, sy, sz = (float(v) * float(downsample) for v in voxel_spacing_um)
    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(sx, sy, sz))
    grid.point_data["component_mask"] = phase_xyz.ravel(order="F")
    return grid


def make_plotter(window_size: tuple[int, int]) -> pv.Plotter:
    pv.global_theme.background = "white"
    pv.global_theme.smooth_shading = True
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")
    plotter.remove_all_lights()
    plotter.add_light(pv.Light(position=(-900, -850, 1200), focal_point=(0, 0, 0), intensity=1.45))
    plotter.add_light(pv.Light(position=(700, 350, 650), focal_point=(0, 0, 0), intensity=0.76))
    plotter.add_light(pv.Light(position=(0, 0, -700), focal_point=(0, 0, 0), intensity=0.32))
    return plotter


def build_segmented_core_plotter(
    mesh: pv.PolyData,
    *,
    window_size: tuple[int, int],
    hide_bounds: bool,
    color: str = "#ffff00",
) -> tuple[pv.Plotter, object]:
    plotter = make_plotter(window_size)
    actor = plotter.add_mesh(mesh, color=color, **SOLID_MATERIAL)
    if not hide_bounds:
        plotter.show_bounds(
            grid="back",
            location="outer",
            color="#333333",
            font_size=16,
            n_xlabels=5,
            n_ylabels=5,
            n_zlabels=5,
        )
    center = np.array(mesh.center, dtype=float)
    extent = float(max(mesh.length, 1.0))
    position = center + np.array([0.38, 1.42, 0.48], dtype=float) * extent
    plotter.camera_position = [tuple(position), tuple(center), (0.0, 0.0, 1.0)]
    plotter.camera.zoom(0.92)
    return plotter, actor


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
    if not include_surface and not include_voxel:
        raise ValueError("At least one of include_surface or include_voxel must be True")

    plotter = make_plotter(window_size)
    actors: list[object] = []
    volume_actors: list[object] = []
    metadata: list[dict] = []
    last_dataset: pv.DataSet | None = None
    for component_index, spec in enumerate(component_specs):
        value = int(spec["value"])
        item_metadata = {
            **spec,
            "actor_index": None,
            "surface_actor_index": None,
            "voxel_actor_index": None,
            "skipped": False,
            "mesh_points": 0,
            "mesh_cells": 0,
            "surface_skipped": True,
            "voxel_points": 0,
            "voxel_cells": 0,
            "voxel_opacity": float(DEFAULT_VOXEL_OPACITY.get(value, 0.36)),
            "voxel_sample_distance_um": 0.0,
            "voxel_visible": False,
            "voxel_skipped": True,
        }
        if include_surface:
            try:
                mesh = segmented_surface_mesh(
                    volume,
                    component_value=value,
                    downsample=downsample,
                    smooth_iterations=smooth_iterations,
                    voxel_spacing_um=voxel_spacing_um,
                )
            except ValueError:
                item_metadata.update({"mesh_points": 0, "mesh_cells": 0, "surface_skipped": True})
            else:
                actor = plotter.add_mesh(mesh, color=spec["color"], **SOLID_MATERIAL)
                # Keep every component actor serializable; initial visibility is applied in the HTML after VTK.js loads.
                actor.SetVisibility(True)
                actor_index = len(actors)
                actors.append(actor)
                item_metadata.update(
                    {
                        "actor_index": int(actor_index),
                        "surface_actor_index": int(actor_index),
                        "mesh_points": int(mesh.n_points),
                        "mesh_cells": int(mesh.n_cells),
                        "surface_skipped": False,
                    }
                )
                last_dataset = mesh

        if include_voxel:
            try:
                voxel_grid = segmented_voxel_volume(
                    volume,
                    component_value=value,
                    downsample=downsample,
                    voxel_spacing_um=voxel_spacing_um,
                )
            except ValueError:
                item_metadata.update({"voxel_points": 0, "voxel_cells": 0, "voxel_skipped": True})
            else:
                opacity = float(DEFAULT_VOXEL_OPACITY.get(value, 0.36))
                volume_actor = plotter.add_volume(
                    voxel_grid,
                    scalars="component_mask",
                    clim=(0, 1),
                    opacity=[0.0, opacity],
                    cmap=["#000000", spec["color"]],
                    opacity_unit_distance=max(voxel_grid.spacing),
                    shade=True,
                    show_scalar_bar=False,
                )
                mapper = volume_actor.GetMapper()
                sample_distance = float(max(voxel_grid.spacing))
                if mapper is not None and hasattr(mapper, "SetSampleDistance"):
                    mapper.SetSampleDistance(sample_distance)
                volume_actor.SetVisibility(True)
                voxel_actor_index = len(volume_actors)
                voxel_visible = mode_matches_component(voxel_visible_mode, value=value, index=component_index)
                volume_actors.append(volume_actor)
                item_metadata.update(
                    {
                        "voxel_actor_index": int(voxel_actor_index),
                        "voxel_points": int(voxel_grid.n_points),
                        "voxel_cells": int(voxel_grid.n_cells),
                        "voxel_opacity": opacity,
                        "voxel_sample_distance_um": sample_distance,
                        "voxel_visible": bool(voxel_visible),
                        "voxel_skipped": False,
                    }
                )
                last_dataset = voxel_grid

        if item_metadata.get("surface_skipped", True) and item_metadata.get("voxel_skipped", True):
            item_metadata["skipped"] = True
        metadata.append(item_metadata)

    if not actors and not volume_actors:
        raise ValueError("No component meshes or voxel volumes could be generated")

    if not hide_bounds:
        plotter.show_bounds(
            grid="back",
            location="outer",
            color="#333333",
            font_size=16,
            n_xlabels=5,
            n_ylabels=5,
            n_zlabels=5,
        )

    bounds_dataset = last_dataset if last_dataset is not None else pv.PolyData()
    center = np.array(bounds_dataset.center, dtype=float)
    extent = float(max(bounds_dataset.length, 1.0))
    position = center + np.array([0.38, 1.42, 0.48], dtype=float) * extent
    plotter.camera_position = [tuple(position), tuple(center), (0.0, 0.0, 1.0)]
    plotter.camera.zoom(0.92)
    return plotter, actors, metadata


def export_html(plotter: pv.Plotter, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    plotter.export_html(out)


def ensure_vtk_global_alias(html_path: Path) -> None:
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    old_alias = "window.global = window.global || {};"
    new_alias = "var global = window.global = window.global || {};"
    if new_alias not in text and old_alias in text:
        text = text.replace(old_alias, new_alias, 1)
    elif new_alias not in text and "<body>" in text:
        text = text.replace("<body>", f"<body>\n<script>{new_alias}</script>", 1)
    elif new_alias not in text:
        text = f"<script>{new_alias}</script>\n{text}"

    shared_hook = "getRenderWindow(),r=HA.getSynchronizerContext(),o=HA.decorate(n);"
    hooked_shared = (
        "getRenderWindow(),r=(window.global=window.global||{},"
        "window.global.renderWindow=n,window.segmentedCoreRenderWindow=n,"
        "HA.getSynchronizerContext()),o=HA.decorate(n);"
    )
    exact_hook = "if(global.renderWindow=n,t.fileURL||t.url){"
    safe_exact = "if(t.fileURL||t.url){"
    fallback_hook = "global.renderWindow=n,t.fileURL||t.url"
    hooked_fallback = "(window.global=window.global||{}).renderWindow=n,window.segmentedCoreRenderWindow=n,t.fileURL||t.url"
    sync_call = "o.synchronize(e.scene),o.render()"
    hooked_sync_call = (
        "Promise.resolve(o.synchronize(e.scene)).then(()=>{const t=[],n=[];"
        "(function s(e){e&&e.type==='vtkOpenGLActor'&&t.push(r.getInstance(e.id));"
        "e&&e.type==='vtkVolume'&&n.push(r.getInstance(e.id));"
        "((e&&e.dependencies)||[]).forEach(s)})(e.scene);"
        "window.segmentedCoreActors=t.filter(Boolean),window.segmentedCoreVolumes=n.filter(Boolean),"
        "window.segmentedCoreVolumes.forEach(e=>e&&e.setVisibility&&e.setVisibility(false)),o.render()})"
    )
    if "window.segmentedCoreRenderWindow=n" not in text and shared_hook in text:
        text = text.replace(shared_hook, hooked_shared, 1)
    elif "window.segmentedCoreRenderWindow=n" not in text and exact_hook in text:
        text = text.replace(exact_hook, safe_exact, 1)
    elif "window.segmentedCoreRenderWindow=n" not in text and fallback_hook in text:
        text = text.replace(fallback_hook, hooked_fallback, 1)
    if exact_hook in text:
        text = text.replace(exact_hook, safe_exact, 1)
    if "window.segmentedCoreActors" not in text and sync_call in text:
        text = text.replace(sync_call, hooked_sync_call, 1)
    html_path.write_text(text, encoding="utf-8")


def inject_vtk_wheel_zoom_support(html_path: Path) -> None:
    script = """
<div id="segmented-core-wheel-zoom-support" data-vtk-zoom-panel style="
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
    return document.getElementById('segmented-core-wheel-zoom-support');
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
  window.segmentedCoreZoomDisplay = zoomDisplay;
  window.setTimeout(bindZoom, 250);
  window.setTimeout(bindZoom, 1000);
})();
</script>
"""
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    if "segmented-core-wheel-zoom-support" in text:
        return
    if "</body>" in text:
        text = text.replace("</body>", script + "\n</body>", 1)
    else:
        text += script
    html_path.write_text(text, encoding="utf-8")


def inject_porosity_overlay(html_path: Path, stats: dict) -> None:
    porosity_percent = float(stats["porosity"]) * 100.0
    label = (
        f"Porosity: {porosity_percent:.2f}%"
        f"<br><span>{int(stats['pore_voxels']):,} pore px / "
        f"{int(stats['pore_voxels']) + int(stats['solid_voxels']):,} total px</span>"
    )
    overlay = f"""
<div id="segmented-core-porosity-overlay" style="
  position: fixed;
  right: 18px;
  top: 18px;
  z-index: 9999;
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid rgba(0,0,0,0.16);
  background: rgba(255,255,255,0.88);
  box-shadow: 0 8px 24px rgba(0,0,0,0.16);
  color: #222;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 14px;
  line-height: 1.35;
  pointer-events: none;
">
  <strong>{html.escape(label, quote=False).replace('&lt;br&gt;', '<br>').replace('&lt;span&gt;', '<span>').replace('&lt;/span&gt;', '</span>')}</strong>
</div>
"""
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    if "segmented-core-porosity-overlay" in text:
        return
    if "</body>" in text:
        text = text.replace("</body>", overlay + "\n</body>", 1)
    else:
        text += overlay
    html_path.write_text(text, encoding="utf-8")


def inject_component_controls(html_path: Path, components: list[dict]) -> None:
    components = [spec for spec in components if not spec.get("skipped", False)]
    has_surface = any(not spec.get("surface_skipped", False) for spec in components)
    has_voxel = any(not spec.get("voxel_skipped", False) for spec in components)
    if has_surface and has_voxel:
        rendering_label = "Surface mesh + voxel volume"
    elif has_voxel:
        rendering_label = "Voxel volume"
    elif has_surface:
        rendering_label = "Surface mesh"
    else:
        rendering_label = "No renderable components"
    rows = []
    for index, spec in enumerate(components):
        surface_checked = " checked" if spec.get("visible", True) else ""
        voxel_checked = " checked" if spec.get("voxel_visible", False) else ""
        surface_disabled = " disabled" if spec.get("surface_skipped", False) else ""
        voxel_disabled = " disabled" if spec.get("voxel_skipped", False) else ""
        label = html.escape(str(spec["label"]))
        color = html.escape(str(spec["color"]))
        value = int(spec["value"])
        rows.append(
            f"""
  <label class="segmented-core-component-row">
    <input type="checkbox" data-component-index="{index}" class="segmented-core-component-visible"{surface_checked}{surface_disabled} title="Surface mesh">
    <input type="checkbox" data-component-index="{index}" class="segmented-core-component-voxel-visible"{voxel_checked}{voxel_disabled} title="Voxel volume">
    <span>{label} ({value})</span>
    <input type="color" data-component-index="{index}" class="segmented-core-component-color" value="{color}">
  </label>"""
        )
    panel = f"""
<div id="segmented-core-component-controls" style="
  position: fixed;
  right: 18px;
  top: 92px;
  z-index: 9999;
  min-width: 220px;
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid rgba(0,0,0,0.16);
  background: rgba(255,255,255,0.88);
  box-shadow: 0 8px 24px rgba(0,0,0,0.16);
  color: #222;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 13px;
  line-height: 1.35;
">
  <strong>Rendering</strong>
  <div id="segmented-core-rendering-label" style="margin-top:6px;color:#555;font-size:12px;">
    {html.escape(rendering_label)}
  </div>
  <div class="segmented-core-component-header">
    <span>Surface mesh</span>
    <span>Voxel volume</span>
    <span>Component</span>
    <span>Color</span>
  </div>
  <div style="display:flex;flex-direction:column;gap:6px;margin-top:8px;">{''.join(rows)}
  </div>
  <button id="segmented-core-apply-components" type="button" style="
    margin-top: 10px;
    width: 100%;
    height: 30px;
    border: 1px solid rgba(0,0,0,0.22);
    border-radius: 6px;
    background: #ffffff;
    color: #111;
    font-size: 13px;
    cursor: pointer;
  ">Apply</button>
  <div id="segmented-core-apply-status" style="margin-top:6px;color:#555;font-size:12px;"></div>
</div>
<style>
  .segmented-core-component-row {{
    display: grid;
    grid-template-columns: 52px 52px 1fr 34px;
    align-items: center;
    gap: 7px;
  }}
  .segmented-core-component-header {{
    display: grid;
    grid-template-columns: 52px 52px 1fr 34px;
    gap: 7px;
    align-items: end;
    margin-top: 9px;
    color: #555;
    font-size: 11px;
  }}
  .segmented-core-component-row input[type="color"] {{
    width: 32px;
    height: 24px;
    padding: 0;
    border: 1px solid rgba(0,0,0,0.18);
    background: transparent;
  }}
</style>
<script>
(function() {{
  const specs = {json.dumps(components, ensure_ascii=False)};
  function hexToRgb(hex) {{
    const clean = hex.replace('#', '');
    const value = parseInt(clean, 16);
    return [((value >> 16) & 255) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255];
  }}
  function getActors() {{
    if (Array.isArray(window.segmentedCoreActors) && window.segmentedCoreActors.length) {{
      return window.segmentedCoreActors.filter((item) => item && item.getProperty);
    }}
    const bareGlobal = (typeof global !== 'undefined') ? global : null;
    const safeGlobalThis = (typeof globalThis !== 'undefined') ? globalThis : null;
    const globalObject = bareGlobal || window.global || (safeGlobalThis && safeGlobalThis.global) || {{}};
    const rw = window.segmentedCoreRenderWindow || globalObject.renderWindow || window.renderWindow;
    if (!rw || !rw.getRenderers) return [];
    const renderers = rw.getRenderers();
    const renderer = renderers && renderers[0];
    if (!renderer) return [];
    let actors = [];
    if (renderer.getActors) actors = renderer.getActors();
    else if (renderer.getViewProps) actors = renderer.getViewProps().filter((item) => item && item.getProperty);
    return Array.from(actors || []).filter((item) => item && item.getProperty && item.getMapper);
  }}
  function getVolumes() {{
    if (Array.isArray(window.segmentedCoreVolumes) && window.segmentedCoreVolumes.length) {{
      return window.segmentedCoreVolumes.filter((item) => item && item.getProperty);
    }}
    const bareGlobal = (typeof global !== 'undefined') ? global : null;
    const safeGlobalThis = (typeof globalThis !== 'undefined') ? globalThis : null;
    const globalObject = bareGlobal || window.global || (safeGlobalThis && safeGlobalThis.global) || {{}};
    const rw = window.segmentedCoreRenderWindow || globalObject.renderWindow || window.renderWindow;
    if (!rw || !rw.getRenderers) return [];
    const renderers = rw.getRenderers();
    const renderer = renderers && renderers[0];
    if (!renderer) return [];
    let volumes = [];
    if (renderer.getVolumes) volumes = renderer.getVolumes();
    else if (renderer.getViewProps) volumes = renderer.getViewProps().filter((item) => item && item.getProperty && !item.getMapper);
    return Array.from(volumes || []).filter((item) => item && item.getProperty);
  }}
  function applyVolumeColor(volume, rgb) {{
    const property = volume && volume.getProperty && volume.getProperty();
    if (!property || !property.getRGBTransferFunction) return;
    const transfer = property.getRGBTransferFunction(0);
    if (!transfer) return;
    if (transfer.removeAllPoints) transfer.removeAllPoints();
    if (transfer.addRGBPoint) {{
      transfer.addRGBPoint(0, 0, 0, 0);
      transfer.addRGBPoint(1, rgb[0], rgb[1], rgb[2]);
    }}
  }}
  function applyControls() {{
    const actors = getActors();
    const volumes = getVolumes();
    const neededActors = specs.filter((spec) => !spec.surface_skipped).length;
    const neededVolumes = specs.filter((spec) => !spec.voxel_skipped).length;
    if (actors.length < neededActors || volumes.length < neededVolumes) {{
      const status = document.getElementById('segmented-core-apply-status');
      if (status) status.textContent = 'Waiting for VTK objects: surface ' + actors.length + '/' + neededActors + ', voxel ' + volumes.length + '/' + neededVolumes;
      window.setTimeout(applyControls, 250);
      return;
    }}
    specs.forEach((spec, index) => {{
      const actorIndex = Number.isInteger(spec.actor_index) ? spec.actor_index : index;
      const volumeIndex = Number.isInteger(spec.voxel_actor_index) ? spec.voxel_actor_index : index;
      const actor = actors[actorIndex];
      const volume = volumes[volumeIndex];
      const visibleInput = document.querySelector('.segmented-core-component-visible[data-component-index="' + index + '"]');
      const voxelVisibleInput = document.querySelector('.segmented-core-component-voxel-visible[data-component-index="' + index + '"]');
      const colorInput = document.querySelector('.segmented-core-component-color[data-component-index="' + index + '"]');
      if (!colorInput) return;
      const rgb = hexToRgb(colorInput.value);
      if (actor && visibleInput) {{
        if (actor.setVisibility) actor.setVisibility(visibleInput.checked);
        const property = actor.getProperty && actor.getProperty();
        if (property && property.setColor) property.setColor(rgb);
      }}
      if (volume && voxelVisibleInput) {{
        if (volume.setVisibility) volume.setVisibility(voxelVisibleInput.checked);
        applyVolumeColor(volume, rgb);
      }}
    }});
    const bareGlobal = (typeof global !== 'undefined') ? global : null;
    const safeGlobalThis = (typeof globalThis !== 'undefined') ? globalThis : null;
    const globalObject = bareGlobal || window.global || (safeGlobalThis && safeGlobalThis.global) || {{}};
    const rw = window.segmentedCoreRenderWindow || globalObject.renderWindow || window.renderWindow;
    if (rw && rw.render) rw.render();
    const status = document.getElementById('segmented-core-apply-status');
    if (status) status.textContent = 'Applied';
  }}
  window.segmentedCoreApplyControls = applyControls;
  const applyButton = document.getElementById('segmented-core-apply-components');
  if (applyButton) applyButton.addEventListener('click', applyControls);
  window.setTimeout(applyControls, 500);
}})();
</script>
"""
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    if "segmented-core-component-controls" in text:
        return
    if "</body>" in text:
        text = text.replace("</body>", panel + "\n</body>", 1)
    else:
        text += panel
    html_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--metadata-out", default=str(DEFAULT_METADATA))
    parser.add_argument("--solid-value", type=int, default=255)
    parser.add_argument("--downsample", type=int, default=12)
    parser.add_argument("--voxel-size-um", type=float, default=1.7)
    parser.add_argument("--smooth-iterations", type=int, default=12)
    parser.add_argument("--window-size", nargs=2, type=int, default=[1800, 1450])
    parser.add_argument("--color", default="#ffff00")
    parser.add_argument("--components", nargs="*", type=int, default=None)
    parser.add_argument("--initial-visible", choices=["pore", "solid", "all", "first"], default="pore")
    parser.add_argument("--voxel-initial-visible", choices=["none", "pore", "solid", "all", "first"], default="none")
    parser.add_argument("--surface-mode", choices=["mesh", "none"], default="mesh")
    parser.add_argument("--voxel-mode", choices=["volume", "none"], default="volume")
    parser.add_argument("--hide-bounds", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    volume = load_tiff_volume(input_path)
    stats = binary_volume_stats(volume, solid_value=args.solid_value)
    component_values = args.components if args.components is not None else [item["value"] for item in stats["components"]]
    component_specs = component_specs_from_values(component_values, visible_mode=args.initial_visible)
    for spec in component_specs:
        if int(spec["value"]) == int(args.solid_value):
            spec["color"] = args.color
    include_surface = args.surface_mode == "mesh"
    include_voxel = args.voxel_mode == "volume"

    plotter, _actors, component_metadata = build_multi_component_plotter(
        volume,
        component_specs=component_specs,
        downsample=args.downsample,
        smooth_iterations=args.smooth_iterations,
        voxel_spacing_um=(args.voxel_size_um, args.voxel_size_um, args.voxel_size_um),
        window_size=tuple(args.window_size),
        hide_bounds=bool(args.hide_bounds),
        include_surface=include_surface,
        include_voxel=include_voxel,
        voxel_visible_mode=args.voxel_initial_visible,
    )
    out = Path(args.out)
    try:
        export_html(plotter, out)
    finally:
        plotter.close()
    ensure_vtk_global_alias(out)
    inject_vtk_wheel_zoom_support(out)
    inject_porosity_overlay(out, stats)
    inject_component_controls(out, component_metadata)

    metadata = {
        "input_tiff": str(input_path),
        "output_html": str(out),
        "renderer": "pyvista/vtk segmented-core single-file html",
        "rendering_modes": rendering_modes(include_surface=include_surface, include_voxel=include_voxel),
        "segmentation_convention": {"solid": int(args.solid_value), "pore": 0},
        "component_values": [int(v) for v in component_values],
        "initial_visible": args.initial_visible,
        "voxel_initial_visible": args.voxel_initial_visible,
        "surface_mode": args.surface_mode,
        "voxel_mode": args.voxel_mode,
        "downsample": int(args.downsample),
        "pixel_size_y_um": float(args.voxel_size_um),
        "voxel_spacing_um_xyz": [float(args.voxel_size_um), float(args.voxel_size_um), float(args.voxel_size_um)],
        "voxel_size_note": "Using supplied pixel_size_y_um as isotropic voxel spacing for x/y/z unless separate axis spacings are provided later.",
        "smooth_iterations": int(args.smooth_iterations),
        "material_parameters": SOLID_MATERIAL,
        "color": args.color,
        "component_meshes": component_metadata,
        "mesh_points": int(sum(item["mesh_points"] for item in component_metadata)),
        "mesh_cells": int(sum(item["mesh_cells"] for item in component_metadata)),
        "voxel_points": int(sum(item["voxel_points"] for item in component_metadata)),
        "voxel_cells": int(sum(item["voxel_cells"] for item in component_metadata)),
        **stats,
    }
    metadata_out = Path(args.metadata_out)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
