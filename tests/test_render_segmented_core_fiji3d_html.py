from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "digital_rock_visualization" / "render_segmented_core_fiji3d_html.py"


def load_module():
    spec = importlib.util.spec_from_file_location("render_segmented_core_fiji3d_html", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tiny_segmented_volume() -> np.ndarray:
    volume = np.zeros((8, 8, 8), dtype=np.uint8)
    volume[2:6, 2:6, 2:6] = 255
    volume[3:5, 3:5, 3:5] = 0
    return volume


def test_fiji3d_label_volume_preserves_segmentation_values():
    module = load_module()

    grid = module.fiji3d_label_volume(tiny_segmented_volume(), downsample=2, voxel_spacing_um=(1.7, 1.7, 1.7))

    assert grid.dimensions == (4, 4, 4)
    assert grid.spacing == (3.4, 3.4, 3.4)
    assert "segmentation_value" in grid.point_data
    assert set(np.unique(grid.point_data["segmentation_value"]).tolist()) == {0, 255}


def test_fiji3d_components_default_to_solid_visible():
    module = load_module()

    components = module.fiji3d_component_specs([0, 255], visible_mode="solid")

    assert components[0]["label"] == "pore"
    assert components[0]["visible"] is False
    assert components[1]["label"] == "solid"
    assert components[1]["visible"] is True
    assert components[1]["opacity"] > 0


def test_fiji3d_plotter_uses_single_volume_actor_and_no_surface_mesh():
    module = load_module()
    components = module.fiji3d_component_specs([0, 255], visible_mode="solid")

    plotter, volume_actor, metadata = module.build_fiji3d_plotter(
        tiny_segmented_volume(),
        component_specs=components,
        downsample=1,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        window_size=(320, 240),
        hide_bounds=True,
    )

    try:
        assert volume_actor is not None
        assert metadata["rendering_style"] == "fiji_3d_viewer_like_volume"
        assert metadata["rendering_modes"] == ["fiji_style_label_volume"]
        assert metadata["volume_points"] > 0
        assert metadata["mesh_points"] == 0
        assert metadata["slice_controls"]["enabled"] is True
        assert metadata["slice_controls"]["realtime"] is False
        assert metadata["slice_controls"]["axes"] == ["x", "y", "z"]
        assert metadata["slice_controls"]["mode"] == "vtk_volume_clipping_plane"
        assert metadata["slice_controls"]["applies_to"] == "vtk_volume_mapper_clipping_planes"
        assert metadata["component_controls"][0]["value"] == 0
        assert metadata["component_controls"][1]["value"] == 255
    finally:
        plotter.close()


def test_fiji3d_plugin_reference_points_to_user_fiji_app():
    module = load_module()

    assert str(module.FIJI3D_PLUGIN_REFERENCE).endswith("Fiji.app\\plugins\\3D_Viewer-5.0.0.jar")
    assert "Documents\\Codex\\Fiji.app" not in str(module.FIJI3D_PLUGIN_REFERENCE)


def test_fiji3d_controls_are_injected_with_phase_color_opacity_and_threshold(tmp_path):
    module = load_module()
    out = tmp_path / "fiji3d.html"
    out.write_text("<html><body><div id=\"app\"></div></body></html>", encoding="utf-8")
    components = module.fiji3d_component_specs([0, 255], visible_mode="solid")

    module.inject_fiji3d_controls(out, components)

    html = out.read_text(encoding="utf-8")
    assert "segmented-core-fiji3d-controls" in html
    assert "Fiji 3D Viewer style" in html
    assert "Phase" in html
    assert "Opacity" in html
    assert "Threshold" in html
    assert "pore (0)" in html
    assert "solid (255)" in html
    assert "type=\"color\"" in html
    assert "type=\"range\"" in html
    assert "getScalarOpacity" in html
    assert "getRGBTransferFunction" in html
    assert "addRGBPoint" in html
    assert "addPoint" in html
    assert "segmentedCoreApplyFiji3DControls" in html
    assert ">Apply<" in html


def test_fiji3d_clip_controls_are_injected_with_axis_slider_and_apply(tmp_path):
    module = load_module()
    out = tmp_path / "fiji3d.html"
    out.write_text("<html><body><div id=\"app\"></div></body></html>", encoding="utf-8")
    components = module.fiji3d_component_specs([0, 255], visible_mode="pore")

    module.inject_fiji3d_slice_controls(out, components)

    html = out.read_text(encoding="utf-8")
    assert "segmented-core-fiji3d-slice-controls" in html
    assert "3D Clip" in html
    assert "Axis" in html
    assert "Cut X plane" in html
    assert "Cut Y plane" in html
    assert "Cut Z plane" in html
    assert "Positive side" in html
    assert "Negative side" in html
    assert "segmented-core-fiji3d-slice-slider" in html
    assert "segmented-core-fiji3d-slice-apply" in html
    assert "segmented-core-fiji3d-slice-reset" in html
    assert ">Apply<" in html
    assert "Show full volume" in html
    assert "getInputData" in html
    assert "getDimensions" in html
    assert "getBounds" in html
    assert "makePlane" in html
    assert "isA: function(name) { return name === 'vtkPlane'; }" in html
    assert "getOrigin: function() { return planeOrigin.slice(); }" in html
    assert "getNormal: function() { return planeNormal.slice(); }" in html
    assert "window.segmentedCoreCurrentClipPlane" in html
    assert "refreshCurrentClipPlane" in html
    assert "installClipPersistenceHook" in html
    assert "window.segmentedCoreClipPersistenceInstalled" in html
    assert "window.segmentedCoreOriginalRender" in html
    assert "rw.render = function()" in html
    assert "removeAllClippingPlanes" in html
    assert "addClippingPlane" in html
    assert "window.segmentedCoreApplyFiji3DClip" in html
    assert "window.segmentedCoreClearFiji3DClip" in html
    assert "window.segmentedCoreRefreshFiji3DClip" in html
    assert "drawSlice" not in html
    assert "segmented-core-fiji3d-slice-canvas" not in html
    assert "slider.addEventListener('input', function() { number.value = slider.value; });" in html
    assert "number.addEventListener('input', function() { slider.value = number.value; });" in html
    assert "button.addEventListener('click', applyClip);" in html
    assert "resetButton.addEventListener('click', clearClip);" in html
    assert "slider.addEventListener('input', applyClip" not in html
    assert "number.addEventListener('input', applyClip" not in html


def test_fiji3d_main_exports_html_with_slice_and_phase_controls(tmp_path, monkeypatch):
    module = load_module()
    out = tmp_path / "fiji3d.html"
    metadata_out = tmp_path / "fiji3d_metadata.json"

    class FakePlotter:
        def close(self):
            pass

    def fake_export_html(_plotter, html_path):
        Path(html_path).write_text("<html><body><div id=\"app\"></div></body></html>", encoding="utf-8")

    def fake_build_fiji3d_plotter(*_args, **_kwargs):
        return (
            FakePlotter(),
            object(),
            {
                "rendering_style": "fiji_3d_viewer_like_volume",
                "rendering_modes": ["fiji_style_label_volume"],
                "volume_points": 512,
                "mesh_points": 0,
                "slice_controls": {
                    "enabled": True,
                    "axes": ["x", "y", "z"],
                    "realtime": False,
                    "mode": "vtk_volume_clipping_plane",
                    "applies_to": "vtk_volume_mapper_clipping_planes",
                },
                "component_controls": module.fiji3d_component_specs([0, 255], visible_mode="pore"),
            },
        )

    monkeypatch.setattr(module, "load_tiff_volume", lambda _path: tiny_segmented_volume())
    monkeypatch.setattr(module, "build_fiji3d_plotter", fake_build_fiji3d_plotter)
    monkeypatch.setattr(module, "export_html", fake_export_html)
    monkeypatch.setattr(module, "ensure_vtk_global_alias", lambda _path: None)
    monkeypatch.setattr(module, "inject_vtk_wheel_zoom_support", lambda _path: None)
    monkeypatch.setattr(module, "inject_porosity_overlay", lambda _path, _stats: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_segmented_core_fiji3d_html.py",
            "--input",
            str(tmp_path / "input.tiff"),
            "--out",
            str(out),
            "--metadata-out",
            str(metadata_out),
            "--initial-visible",
            "pore",
            "--hide-bounds",
        ],
    )

    module.main()

    html = out.read_text(encoding="utf-8")
    metadata = json.loads(metadata_out.read_text(encoding="utf-8"))
    assert "segmented-core-fiji3d-slice-controls" in html
    assert "segmented-core-fiji3d-controls" in html
    assert "segmented-core-fiji3d-slice-apply" in html
    assert "segmented-core-fiji3d-apply" in html
    assert "window.segmentedCoreApplyFiji3DClip" in html
    assert "segmentedCoreApplyFiji3DControls" in html
    assert metadata["slice_controls"]["enabled"] is True
    assert metadata["slice_controls"]["realtime"] is False
    assert metadata["slice_controls"]["mode"] == "vtk_volume_clipping_plane"
