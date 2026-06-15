#!/usr/bin/env python3
"""Export a Fiji 3D Viewer-like interactive HTML volume rendering."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_segmented_core_html import (  # noqa: E402
    DEFAULT_COMPONENT_COLORS,
    DEFAULT_INPUT,
    DEFAULT_VOXEL_OPACITY,
    binary_volume_stats,
    component_label,
    ensure_vtk_global_alias,
    export_html,
    inject_porosity_overlay,
    inject_vtk_wheel_zoom_support,
    load_tiff_volume,
    make_plotter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "figures"
    / "segmented_cores"
    / "sample_89_Grainstone_89seged_fiji3d_viewer_style_volume_interactive.html"
)
DEFAULT_METADATA = (
    PROJECT_ROOT
    / "results"
    / "source_data"
    / "sample_89_Grainstone_89seged_fiji3d_viewer_style_volume_interactive_metadata.json"
)
FIJI3D_PLUGIN_REFERENCE = Path.home() / "Fiji.app" / "plugins" / "3D_Viewer-5.0.0.jar"


def fiji3d_label_volume(
    volume: np.ndarray,
    *,
    downsample: int,
    voxel_spacing_um: tuple[float, float, float],
) -> pv.ImageData:
    if downsample < 1:
        raise ValueError("downsample must be >= 1")
    sampled_zyx = volume[::downsample, ::downsample, ::downsample].astype(np.uint8, copy=False)
    sampled_xyz = np.transpose(sampled_zyx, (2, 1, 0))
    nx, ny, nz = sampled_xyz.shape
    sx, sy, sz = (float(value) * float(downsample) for value in voxel_spacing_um)
    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=(sx, sy, sz))
    grid.point_data["segmentation_value"] = sampled_xyz.ravel(order="F")
    return grid


def fiji3d_component_specs(values: list[int] | np.ndarray, *, visible_mode: str = "solid") -> list[dict]:
    specs: list[dict] = []
    fallback_colors = ["#f0c419", "#30b36b", "#9b59b6", "#ff7f0e", "#17becf", "#8c564b"]
    fallback_i = 0
    for index, value in enumerate(int(v) for v in sorted(values)):
        color = DEFAULT_COMPONENT_COLORS.get(value)
        if color is None:
            color = fallback_colors[fallback_i % len(fallback_colors)]
            fallback_i += 1
        visible = (
            visible_mode == "all"
            or (visible_mode == "pore" and value == 0)
            or (visible_mode == "solid" and value == 255)
            or (visible_mode == "first" and index == 0)
        )
        specs.append(
            {
                "value": value,
                "label": component_label(value),
                "color": color,
                "visible": bool(visible),
                "opacity": float(DEFAULT_VOXEL_OPACITY.get(value, 0.34)),
                "threshold_min": value,
                "threshold_max": value,
            }
        )
    return specs


def _make_endpoint_luts(component_specs: list[dict]) -> tuple[list[str], list[float]]:
    by_value = {int(spec["value"]): spec for spec in component_specs}
    values = sorted(by_value)
    if not values:
        raise ValueError("No component values were provided")
    colors = [str(by_value[value]["color"]) for value in values]
    opacities = [float(by_value[value]["opacity"]) if by_value[value].get("visible", False) else 0.0 for value in values]
    return colors, opacities


def _hex_to_rgb01(color: str) -> tuple[float, float, float]:
    clean = color.lstrip("#")
    return (
        int(clean[0:2], 16) / 255.0,
        int(clean[2:4], 16) / 255.0,
        int(clean[4:6], 16) / 255.0,
    )


def apply_fiji3d_threshold_transfer_functions(volume_actor: object, component_specs: list[dict]) -> None:
    prop = volume_actor.GetProperty()
    if prop is None:
        return
    rgb_transfer = prop.GetRGBTransferFunction()
    opacity_transfer = prop.GetScalarOpacity()
    if rgb_transfer is not None and hasattr(rgb_transfer, "RemoveAllPoints"):
        rgb_transfer.RemoveAllPoints()
    if opacity_transfer is not None and hasattr(opacity_transfer, "RemoveAllPoints"):
        opacity_transfer.RemoveAllPoints()

    for spec in sorted(component_specs, key=lambda item: int(item["value"])):
        value = float(spec["value"])
        opacity = float(spec.get("opacity", 0.34)) if spec.get("visible", False) else 0.0
        rgb = _hex_to_rgb01(str(spec["color"]))
        if rgb_transfer is not None and hasattr(rgb_transfer, "AddRGBPoint"):
            rgb_transfer.AddRGBPoint(value, *rgb)
        if opacity_transfer is not None and hasattr(opacity_transfer, "AddPoint"):
            if value > 0:
                opacity_transfer.AddPoint(value - 0.5, 0.0)
            opacity_transfer.AddPoint(value, opacity)
            opacity_transfer.AddPoint(value + 0.5, 0.0)


def build_fiji3d_plotter(
    volume: np.ndarray,
    *,
    component_specs: list[dict],
    downsample: int,
    voxel_spacing_um: tuple[float, float, float],
    window_size: tuple[int, int],
    hide_bounds: bool,
    interpolation: str = "linear",
) -> tuple[pv.Plotter, object, dict]:
    grid = fiji3d_label_volume(volume, downsample=downsample, voxel_spacing_um=voxel_spacing_um)
    colors, opacities = _make_endpoint_luts(component_specs)

    plotter = make_plotter(window_size)
    volume_actor = plotter.add_volume(
        grid,
        scalars="segmentation_value",
        clim=(min(spec["value"] for spec in component_specs), max(spec["value"] for spec in component_specs)),
        cmap=colors,
        opacity=opacities,
        opacity_unit_distance=max(grid.spacing),
        shade=True,
        ambient=0.34,
        diffuse=0.72,
        specular=0.18,
        show_scalar_bar=False,
    )
    volume_actor.SetVisibility(True)
    prop = volume_actor.GetProperty()
    if prop is not None:
        if interpolation == "nearest" and hasattr(prop, "SetInterpolationTypeToNearest"):
            prop.SetInterpolationTypeToNearest()
        elif hasattr(prop, "SetInterpolationTypeToLinear"):
            prop.SetInterpolationTypeToLinear()
    apply_fiji3d_threshold_transfer_functions(volume_actor, component_specs)
    mapper = volume_actor.GetMapper()
    sample_distance = float(max(grid.spacing) * 0.55)
    if mapper is not None and hasattr(mapper, "SetSampleDistance"):
        mapper.SetSampleDistance(sample_distance)

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

    center = np.array(grid.center, dtype=float)
    extent = float(max(grid.length, 1.0))
    position = center + np.array([0.38, 1.42, 0.48], dtype=float) * extent
    plotter.camera_position = [tuple(position), tuple(center), (0.0, 0.0, 1.0)]
    plotter.camera.zoom(0.92)

    metadata = {
        "rendering_style": "fiji_3d_viewer_like_volume",
        "rendering_modes": ["fiji_style_label_volume"],
        "fiji_3d_viewer_modes_referenced": ["VOLUME", "ORTHO", "SURFACE", "MULTIORTHO"],
        "fiji_3d_viewer_features_referenced": ["setThreshold", "setTransparency", "setColor", "saturatedVolumeRendering"],
        "volume_points": int(grid.n_points),
        "volume_cells": int(grid.n_cells),
        "mesh_points": 0,
        "mesh_cells": 0,
        "volume_sample_distance_um": sample_distance,
        "volume_interpolation": interpolation,
        "slice_controls": {
            "enabled": True,
            "axes": ["x", "y", "z"],
            "realtime": False,
            "mode": "vtk_volume_clipping_plane",
            "applies_to": "vtk_volume_mapper_clipping_planes",
        },
        "component_controls": component_specs,
    }
    return plotter, volume_actor, metadata


def _fiji3d_slice_panel_css_top() -> int:
    return 92


def inject_fiji3d_controls(html_path: Path, components: list[dict]) -> None:
    rows = []
    for index, spec in enumerate(components):
        checked = " checked" if spec.get("visible", False) else ""
        label = html.escape(str(spec["label"]))
        color = html.escape(str(spec["color"]))
        opacity = float(spec.get("opacity", 0.34))
        value = int(spec["value"])
        rows.append(
            f"""
  <label class="segmented-core-fiji3d-row">
    <input type="checkbox" data-component-index="{index}" class="segmented-core-fiji3d-visible"{checked} title="Phase visible">
    <span>{label} ({value})</span>
    <input type="color" data-component-index="{index}" class="segmented-core-fiji3d-color" value="{color}">
    <input type="range" data-component-index="{index}" class="segmented-core-fiji3d-opacity" min="0" max="1" step="0.01" value="{opacity:.2f}" title="Opacity">
  </label>"""
        )

    panel = f"""
<div id="segmented-core-fiji3d-controls" style="
  position: fixed;
  right: 18px;
  top: 470px;
  z-index: 9999;
  min-width: 310px;
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid rgba(0,0,0,0.16);
  background: rgba(255,255,255,0.9);
  box-shadow: 0 8px 24px rgba(0,0,0,0.16);
  color: #222;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 13px;
  line-height: 1.35;
">
  <strong>Fiji 3D Viewer style</strong>
  <div style="margin-top:6px;color:#555;font-size:12px;">Volume rendering with LUT, Threshold, Transparency, and Color controls</div>
  <div class="segmented-core-fiji3d-header">
    <span>Phase</span>
    <span>Color</span>
    <span>Opacity</span>
  </div>
  <div style="display:flex;flex-direction:column;gap:7px;margin-top:8px;">{''.join(rows)}
  </div>
  <button id="segmented-core-fiji3d-apply" type="button" style="
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
  <div id="segmented-core-fiji3d-status" style="margin-top:6px;color:#555;font-size:12px;"></div>
</div>
<style>
  .segmented-core-fiji3d-row {{
    display: grid;
    grid-template-columns: 22px 1fr 34px 90px;
    align-items: center;
    gap: 8px;
  }}
  .segmented-core-fiji3d-header {{
    display: grid;
    grid-template-columns: calc(22px + 1fr) 34px 90px;
    gap: 8px;
    align-items: end;
    margin-top: 9px;
    color: #555;
    font-size: 11px;
  }}
  .segmented-core-fiji3d-row input[type="color"] {{
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
  function getVolume() {{
    if (Array.isArray(window.segmentedCoreVolumes) && window.segmentedCoreVolumes.length) {{
      return window.segmentedCoreVolumes.find((item) => item && item.getProperty) || null;
    }}
    const bareGlobal = (typeof global !== 'undefined') ? global : null;
    const safeGlobalThis = (typeof globalThis !== 'undefined') ? globalThis : null;
    const globalObject = bareGlobal || window.global || (safeGlobalThis && safeGlobalThis.global) || {{}};
    const rw = window.segmentedCoreRenderWindow || globalObject.renderWindow || window.renderWindow;
    if (!rw || !rw.getRenderers) return null;
    const renderers = rw.getRenderers();
    const renderer = renderers && renderers[0];
    if (!renderer) return null;
    if (renderer.getVolumes) {{
      const volumes = Array.from(renderer.getVolumes() || []);
      return volumes.find((item) => item && item.getProperty) || null;
    }}
    if (renderer.getViewProps) {{
      const props = Array.from(renderer.getViewProps() || []);
      return props.find((item) => item && item.getProperty && !item.getMapper) || null;
    }}
    return null;
  }}
  function setStatus(text) {{
    const status = document.getElementById('segmented-core-fiji3d-status');
    if (status) status.textContent = text;
  }}
  function applyControls() {{
    const volume = getVolume();
    if (!volume) {{
      setStatus('Waiting for VTK volume');
      window.setTimeout(applyControls, 300);
      return;
    }}
    if (volume.setVisibility) volume.setVisibility(true);
    const property = volume.getProperty && volume.getProperty();
    if (!property) {{
      setStatus('Waiting for VTK volume property');
      window.setTimeout(applyControls, 300);
      return;
    }}
    const rgbTransfer = property.getRGBTransferFunction && property.getRGBTransferFunction(0);
    const opacityTransfer = property.getScalarOpacity && property.getScalarOpacity(0);
    if (rgbTransfer && rgbTransfer.removeAllPoints) rgbTransfer.removeAllPoints();
    if (opacityTransfer && opacityTransfer.removeAllPoints) opacityTransfer.removeAllPoints();
    specs.forEach((spec, index) => {{
      const visibleInput = document.querySelector('.segmented-core-fiji3d-visible[data-component-index="' + index + '"]');
      const colorInput = document.querySelector('.segmented-core-fiji3d-color[data-component-index="' + index + '"]');
      const opacityInput = document.querySelector('.segmented-core-fiji3d-opacity[data-component-index="' + index + '"]');
      if (!colorInput || !opacityInput || !visibleInput) return;
      const rgb = hexToRgb(colorInput.value);
      const scalarValue = Number(spec.value);
      const opacity = visibleInput.checked ? Number(opacityInput.value) : 0;
      if (rgbTransfer && rgbTransfer.addRGBPoint) rgbTransfer.addRGBPoint(scalarValue, rgb[0], rgb[1], rgb[2]);
      if (opacityTransfer && opacityTransfer.addPoint) {{
        if (scalarValue > 0) opacityTransfer.addPoint(scalarValue - 0.5, 0);
        opacityTransfer.addPoint(scalarValue, opacity);
        opacityTransfer.addPoint(scalarValue + 0.5, 0);
      }}
    }});
    const bareGlobal = (typeof global !== 'undefined') ? global : null;
    const safeGlobalThis = (typeof globalThis !== 'undefined') ? globalThis : null;
    const globalObject = bareGlobal || window.global || (safeGlobalThis && safeGlobalThis.global) || {{}};
    const rw = window.segmentedCoreRenderWindow || globalObject.renderWindow || window.renderWindow;
    if (rw && rw.render) rw.render();
    setStatus('Applied');
  }}
  window.segmentedCoreApplyFiji3DControls = applyControls;
  const button = document.getElementById('segmented-core-fiji3d-apply');
  if (button) button.addEventListener('click', applyControls);
  window.setTimeout(applyControls, 500);
  window.setTimeout(applyControls, 1500);
}})();
</script>
"""
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    if "segmented-core-fiji3d-controls" in text:
        return
    if "</body>" in text:
        text = text.replace("</body>", panel + "\n</body>", 1)
    else:
        text += panel
    html_path.write_text(text, encoding="utf-8")


def inject_fiji3d_slice_controls(html_path: Path, components: list[dict]) -> None:
    panel = f"""
<div id="segmented-core-fiji3d-slice-controls" style="
  position: fixed;
  right: 18px;
  top: {_fiji3d_slice_panel_css_top()}px;
  z-index: 10000;
  width: 310px;
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid rgba(0,0,0,0.16);
  background: rgba(255,255,255,0.92);
  box-shadow: 0 8px 24px rgba(0,0,0,0.16);
  color: #222;
  font-family: Arial, Helvetica, sans-serif;
  font-size: 13px;
  line-height: 1.35;
">
  <strong>3D Clip</strong>
  <div style="margin-top:6px;color:#555;font-size:12px;">Choose a cutting plane and side, then Apply.</div>
  <label style="display:grid;grid-template-columns:62px 1fr;align-items:center;gap:8px;margin-top:9px;">
    <span>Axis</span>
    <select id="segmented-core-fiji3d-slice-axis" style="height:28px;border:1px solid rgba(0,0,0,0.22);border-radius:6px;background:#fff;">
      <option value="z">Cut Z plane</option>
      <option value="y">Cut Y plane</option>
      <option value="x">Cut X plane</option>
    </select>
  </label>
  <label style="display:grid;grid-template-columns:62px 1fr 58px;align-items:center;gap:8px;margin-top:8px;">
    <span>Index</span>
    <input id="segmented-core-fiji3d-slice-slider" type="range" min="0" max="0" step="1" value="0">
    <input id="segmented-core-fiji3d-slice-index" type="number" min="0" max="0" step="1" value="0" style="height:26px;border:1px solid rgba(0,0,0,0.22);border-radius:6px;padding:0 5px;">
  </label>
  <label style="display:grid;grid-template-columns:62px 1fr;align-items:center;gap:8px;margin-top:8px;">
    <span>Keep</span>
    <select id="segmented-core-fiji3d-slice-side" style="height:28px;border:1px solid rgba(0,0,0,0.22);border-radius:6px;background:#fff;">
      <option value="positive">Positive side</option>
      <option value="negative">Negative side</option>
    </select>
  </label>
  <button id="segmented-core-fiji3d-slice-apply" type="button" style="
    margin-top: 9px;
    width: 100%;
    height: 30px;
    border: 1px solid rgba(0,0,0,0.22);
    border-radius: 6px;
    background: #ffffff;
    color: #111;
    font-size: 13px;
    cursor: pointer;
  ">Apply</button>
  <button id="segmented-core-fiji3d-slice-reset" type="button" style="
    margin-top: 7px;
    width: 100%;
    height: 28px;
    border: 1px solid rgba(0,0,0,0.18);
    border-radius: 6px;
    background: #f7f7f7;
    color: #333;
    font-size: 13px;
    cursor: pointer;
  ">Show full volume</button>
  <div id="segmented-core-fiji3d-slice-status" style="margin-top:6px;color:#555;font-size:12px;"></div>
</div>
<script>
(function() {{
  const specs = {json.dumps(components, ensure_ascii=False)};
  function getVolume() {{
    if (Array.isArray(window.segmentedCoreVolumes) && window.segmentedCoreVolumes.length) {{
      return window.segmentedCoreVolumes.find((item) => item && item.getMapper) || null;
    }}
    const bareGlobal = (typeof global !== 'undefined') ? global : null;
    const safeGlobalThis = (typeof globalThis !== 'undefined') ? globalThis : null;
    const globalObject = bareGlobal || window.global || (safeGlobalThis && safeGlobalThis.global) || {{}};
    const rw = window.segmentedCoreRenderWindow || globalObject.renderWindow || window.renderWindow;
    if (!rw || !rw.getRenderers) return null;
    const renderers = rw.getRenderers();
    const renderer = renderers && renderers[0];
    if (!renderer) return null;
    if (renderer.getVolumes) {{
      const volumes = Array.from(renderer.getVolumes() || []);
      return volumes.find((item) => item && item.getMapper) || null;
    }}
    if (renderer.getViewProps) {{
      const props = Array.from(renderer.getViewProps() || []);
      return props.find((item) => item && item.getMapper && item.getProperty && !item.getProperty().setColor) || null;
    }}
    return null;
  }}
  function getVolumePayload() {{
    const volume = getVolume();
    const mapper = volume && volume.getMapper && volume.getMapper();
    const image = mapper && mapper.getInputData && mapper.getInputData();
    const dims = image && image.getDimensions && image.getDimensions();
    const bounds = image && image.getBounds && image.getBounds();
    if (!volume || !mapper || !image || !dims || dims.length < 3) return null;
    return {{
      volume,
      mapper,
      image,
      dims: [Number(dims[0]), Number(dims[1]), Number(dims[2])],
      bounds: bounds && bounds.length >= 6 ? bounds.map(Number) : null,
    }};
  }}
  function setSliceStatus(text) {{
    const status = document.getElementById('segmented-core-fiji3d-slice-status');
    if (status) status.textContent = text;
  }}
  function syncAxisLimits() {{
    const axisInput = document.getElementById('segmented-core-fiji3d-slice-axis');
    const slider = document.getElementById('segmented-core-fiji3d-slice-slider');
    const number = document.getElementById('segmented-core-fiji3d-slice-index');
    const payload = getVolumePayload();
    if (!axisInput || !slider || !number || !payload) return false;
    const dims = payload.dims;
    const axis = axisInput.value;
    const maxIndex = axis === 'x' ? dims[0] - 1 : axis === 'y' ? dims[1] - 1 : dims[2] - 1;
    slider.max = String(maxIndex);
    number.max = String(maxIndex);
    if (Number(slider.value) > maxIndex) slider.value = String(Math.floor(maxIndex / 2));
    if (Number(number.value) > maxIndex) number.value = slider.value;
    return true;
  }}
  function renderScene() {{
    const bareGlobal = (typeof global !== 'undefined') ? global : null;
    const safeGlobalThis = (typeof globalThis !== 'undefined') ? globalThis : null;
    const globalObject = bareGlobal || window.global || (safeGlobalThis && safeGlobalThis.global) || {{}};
    const rw = window.segmentedCoreRenderWindow || globalObject.renderWindow || window.renderWindow;
    if (rw && rw.render) rw.render();
  }}
  function makePlane(origin, normal) {{
    let mtime = Date.now();
    const planeOrigin = origin.slice();
    const planeNormal = normal.slice();
    return {{
      isA: function(name) {{ return name === 'vtkPlane'; }},
      getOrigin: function() {{ return planeOrigin.slice(); }},
      getNormal: function() {{ return planeNormal.slice(); }},
      getMTime: function() {{ return mtime; }},
      modified: function() {{ mtime = Date.now(); }},
    }};
  }}
  function setMapperClipPlane(payload, origin, normal) {{
    const plane = makePlane(origin, normal);
    if (payload.mapper.removeAllClippingPlanes) payload.mapper.removeAllClippingPlanes();
    if (payload.mapper.addClippingPlane) payload.mapper.addClippingPlane(plane);
    else if (payload.mapper.setClippingPlanes) payload.mapper.setClippingPlanes([plane]);
    if (payload.mapper.modified) payload.mapper.modified();
    if (payload.volume.modified) payload.volume.modified();
    window.segmentedCoreCurrentClipPlane = {{
      origin: origin.slice(),
      normal: normal.slice(),
      appliedAt: Date.now(),
    }};
  }}
  function refreshCurrentClipPlane() {{
    const state = window.segmentedCoreCurrentClipPlane;
    if (!state) return false;
    const payload = getVolumePayload();
    if (!payload) return false;
    const planes = payload.mapper.getClippingPlanes ? payload.mapper.getClippingPlanes() : [];
    const current = planes && planes[0];
    const origin = current && current.getOrigin ? current.getOrigin() : null;
    const normal = current && current.getNormal ? current.getNormal() : null;
    const sameOrigin = origin && origin.length >= 3 && origin.every((value, index) => Math.abs(value - state.origin[index]) < 1e-9);
    const sameNormal = normal && normal.length >= 3 && normal.every((value, index) => Math.abs(value - state.normal[index]) < 1e-9);
    if (!current || !sameOrigin || !sameNormal) {{
      setMapperClipPlane(payload, state.origin, state.normal);
      return true;
    }}
    return false;
  }}
  function clearClip() {{
    const payload = getVolumePayload();
    if (!payload) {{
      setSliceStatus('Waiting for VTK volume mapper');
      return;
    }}
    installClipPersistenceHook();
    if (payload.mapper.removeAllClippingPlanes) payload.mapper.removeAllClippingPlanes();
    window.segmentedCoreCurrentClipPlane = null;
    if (payload.mapper.modified) payload.mapper.modified();
    if (payload.volume.modified) payload.volume.modified();
    renderScene();
    setSliceStatus('Full volume');
  }}
  function applyClip() {{
    const payload = getVolumePayload();
    if (!payload) {{
      setSliceStatus('Waiting for VTK volume mapper');
      window.setTimeout(applyClip, 350);
      return;
    }}
    const axisInput = document.getElementById('segmented-core-fiji3d-slice-axis');
    const slider = document.getElementById('segmented-core-fiji3d-slice-slider');
    const number = document.getElementById('segmented-core-fiji3d-slice-index');
    const sideInput = document.getElementById('segmented-core-fiji3d-slice-side');
    if (!axisInput || !slider || !number || !sideInput) return;
    syncAxisLimits();
    const axis = axisInput.value;
    const dims = payload.dims;
    const bounds = payload.bounds;
    const nx = dims[0], ny = dims[1], nz = dims[2];
    const maxIndex = axis === 'x' ? nx - 1 : axis === 'y' ? ny - 1 : nz - 1;
    const sliceIndex = Math.max(0, Math.min(maxIndex, Math.round(Number(number.value || slider.value || 0))));
    slider.value = String(sliceIndex);
    number.value = String(sliceIndex);
    const axisIndex = axis === 'x' ? 0 : axis === 'y' ? 1 : 2;
    const low = bounds ? bounds[axisIndex * 2] : 0;
    const high = bounds ? bounds[axisIndex * 2 + 1] : maxIndex;
    const coordinate = maxIndex > 0 ? low + (high - low) * (sliceIndex / maxIndex) : low;
    const origin = [0, 0, 0];
    if (bounds) {{
      origin[0] = (bounds[0] + bounds[1]) * 0.5;
      origin[1] = (bounds[2] + bounds[3]) * 0.5;
      origin[2] = (bounds[4] + bounds[5]) * 0.5;
    }}
    origin[axisIndex] = coordinate;
    const sign = sideInput.value === 'negative' ? -1 : 1;
    const normal = [0, 0, 0];
    normal[axisIndex] = sign;
    installClipPersistenceHook();
    setMapperClipPlane(payload, origin, normal);
    renderScene();
    const sideLabel = sideInput.value === 'negative' ? 'negative' : 'positive';
    setSliceStatus('3D clipped at ' + axis.toUpperCase() + ' index ' + sliceIndex + ' / ' + maxIndex + ', keeping ' + sideLabel + ' side');
  }}
  function installClipPersistenceHook() {{
    const bareGlobal = (typeof global !== 'undefined') ? global : null;
    const safeGlobalThis = (typeof globalThis !== 'undefined') ? globalThis : null;
    const globalObject = bareGlobal || window.global || (safeGlobalThis && safeGlobalThis.global) || {{}};
    const rw = window.segmentedCoreRenderWindow || globalObject.renderWindow || window.renderWindow;
    if (!rw || !rw.render) return false;
    if (window.segmentedCoreClipPersistenceInstalled) return true;
    const originalRender = rw.render.bind(rw);
    let insideRender = false;
    rw.render = function() {{
      if (!insideRender) refreshCurrentClipPlane();
      insideRender = true;
      try {{
        return originalRender();
      }} finally {{
        insideRender = false;
      }}
    }};
    window.segmentedCoreOriginalRender = originalRender;
    window.segmentedCoreClipPersistenceInstalled = true;
    rw.segmentedCoreClipPersistenceInstalled = true;
    return true;
  }}
  function bindSliceControls() {{
    const axisInput = document.getElementById('segmented-core-fiji3d-slice-axis');
    const slider = document.getElementById('segmented-core-fiji3d-slice-slider');
    const number = document.getElementById('segmented-core-fiji3d-slice-index');
    const button = document.getElementById('segmented-core-fiji3d-slice-apply');
    const resetButton = document.getElementById('segmented-core-fiji3d-slice-reset');
    if (!axisInput || !slider || !number || !button || !resetButton) return;
    axisInput.addEventListener('change', function() {{
      if (syncAxisLimits()) {{
        const maxIndex = Number(slider.max || 0);
        slider.value = String(Math.floor(maxIndex / 2));
        number.value = slider.value;
      }}
    }});
    slider.addEventListener('input', function() {{ number.value = slider.value; }});
    number.addEventListener('input', function() {{ slider.value = number.value; }});
    button.addEventListener('click', applyClip);
    resetButton.addEventListener('click', clearClip);
    if (syncAxisLimits()) {{
      const maxIndex = Number(slider.max || 0);
      slider.value = String(Math.floor(maxIndex / 2));
      number.value = slider.value;
      setSliceStatus('Ready: choose a plane and click Apply');
    }}
    installClipPersistenceHook();
  }}
  window.segmentedCoreApplyFiji3DClip = applyClip;
  window.segmentedCoreClearFiji3DClip = clearClip;
  window.segmentedCoreRefreshFiji3DClip = refreshCurrentClipPlane;
  window.setTimeout(bindSliceControls, 600);
  window.setTimeout(bindSliceControls, 1800);
  window.setInterval(installClipPersistenceHook, 1200);
}})();
</script>
"""
    text = html_path.read_text(encoding="utf-8", errors="ignore")
    if "segmented-core-fiji3d-slice-controls" in text:
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
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--voxel-size-um", type=float, default=1.7)
    parser.add_argument("--window-size", nargs=2, type=int, default=[1800, 1450])
    parser.add_argument("--components", nargs="*", type=int, default=None)
    parser.add_argument("--initial-visible", choices=["pore", "solid", "all", "first"], default="pore")
    parser.add_argument("--interpolation", choices=["linear", "nearest"], default="linear")
    parser.add_argument("--hide-bounds", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    volume = load_tiff_volume(input_path)
    stats = binary_volume_stats(volume, solid_value=args.solid_value)
    component_values = args.components if args.components is not None else [item["value"] for item in stats["components"]]
    component_specs = fiji3d_component_specs(component_values, visible_mode=args.initial_visible)

    plotter, _volume_actor, render_metadata = build_fiji3d_plotter(
        volume,
        component_specs=component_specs,
        downsample=args.downsample,
        voxel_spacing_um=(args.voxel_size_um, args.voxel_size_um, args.voxel_size_um),
        window_size=tuple(args.window_size),
        hide_bounds=bool(args.hide_bounds),
        interpolation=args.interpolation,
    )
    out = Path(args.out)
    try:
        export_html(plotter, out)
    finally:
        plotter.close()
    ensure_vtk_global_alias(out)
    inject_vtk_wheel_zoom_support(out)
    inject_porosity_overlay(out, stats)
    inject_fiji3d_slice_controls(out, component_specs)
    inject_fiji3d_controls(out, component_specs)

    metadata = {
        "input_tiff": str(input_path),
        "output_html": str(out),
        "renderer": "pyvista/vtk fiji-3d-viewer-like label volume single-file html",
        "fiji_3d_viewer_plugin_reference": str(FIJI3D_PLUGIN_REFERENCE),
        "segmentation_convention": {"solid": int(args.solid_value), "pore": 0},
        "component_values": [int(v) for v in component_values],
        "initial_visible": args.initial_visible,
        "downsample": int(args.downsample),
        "pixel_size_y_um": float(args.voxel_size_um),
        "voxel_spacing_um_xyz": [float(args.voxel_size_um), float(args.voxel_size_um), float(args.voxel_size_um)],
        "voxel_size_note": "Using supplied pixel_size_y_um as isotropic voxel spacing for x/y/z unless separate axis spacings are provided later.",
        **stats,
        **render_metadata,
    }
    metadata_out = Path(args.metadata_out)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
