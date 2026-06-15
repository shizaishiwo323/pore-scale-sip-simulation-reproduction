from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "digital_rock_visualization" / "render_segmented_core_html.py"


def load_module():
    spec = importlib.util.spec_from_file_location("render_segmented_core_html", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def tiny_segmented_volume() -> np.ndarray:
    volume = np.zeros((8, 8, 8), dtype=np.uint8)
    volume[2:6, 2:6, 2:6] = 255
    volume[3:5, 3:5, 3:5] = 0
    volume[0:2, 0:2, 0:2] = 128
    return volume


def test_binary_volume_stats_treat_255_as_solid_and_0_as_pore():
    module = load_module()
    stats = module.binary_volume_stats(tiny_segmented_volume(), solid_value=255)

    assert stats["shape_zyx"] == [8, 8, 8]
    assert stats["solid_value"] == 255
    assert stats["pore_value"] == 0
    assert stats["solid_voxels"] == 56
    assert stats["pore_voxels"] == 448
    assert stats["porosity"] == 448 / 512
    assert stats["components"][0]["value"] == 0
    assert stats["components"][1]["value"] == 128
    assert stats["components"][2]["value"] == 255


def test_segmented_surface_is_created_from_solid_phase():
    module = load_module()
    mesh = module.segmented_surface_mesh(tiny_segmented_volume(), component_value=255, downsample=1)

    assert mesh.n_points > 0
    assert mesh.n_cells > 0


def test_segmented_voxel_volume_is_created_from_solid_phase():
    module = load_module()
    grid = module.segmented_voxel_volume(tiny_segmented_volume(), component_value=255, downsample=2)

    assert grid.n_points > 0
    assert grid.n_cells > 0
    assert "component_mask" in grid.point_data
    assert grid.point_data["component_mask"].max() == 1


def test_plotter_uses_pyvista_specular_material_for_segmented_core():
    module = load_module()
    mesh = module.segmented_surface_mesh(tiny_segmented_volume(), component_value=255, downsample=1)
    plotter, actor = module.build_segmented_core_plotter(mesh, window_size=(320, 240), hide_bounds=True)

    try:
        prop = actor.GetProperty()
        assert prop.GetSpecular() == module.SOLID_MATERIAL["specular"]
        assert prop.GetSpecularPower() == module.SOLID_MATERIAL["specular_power"]
    finally:
        plotter.close()


def test_exported_segmented_core_html_is_vtk_single_file(tmp_path):
    module = load_module()
    mesh = module.segmented_surface_mesh(tiny_segmented_volume(), component_value=255, downsample=1)
    plotter, _actor = module.build_segmented_core_plotter(mesh, window_size=(320, 240), hide_bounds=True)
    out = tmp_path / "segmented_core.html"

    try:
        module.export_html(plotter, out)
    finally:
        plotter.close()

    html = out.read_text(encoding="utf-8", errors="ignore")
    assert "vtk" in html.lower()
    assert "<script src=" not in html
    assert "Plotly.newPlot" not in html


def test_porosity_overlay_is_injected_in_top_right(tmp_path):
    module = load_module()
    out = tmp_path / "segmented_core.html"
    out.write_text("<html><body><div id=\"app\"></div></body></html>", encoding="utf-8")
    stats = module.binary_volume_stats(tiny_segmented_volume(), solid_value=255)

    module.inject_porosity_overlay(out, stats)

    html = out.read_text(encoding="utf-8")
    assert "segmented-core-porosity-overlay" in html
    assert "Porosity" in html
    assert "87.50%" in html
    assert "position: fixed" in html
    assert "right: 18px" in html
    assert "top: 18px" in html


def test_component_controls_are_injected_with_visibility_and_color_inputs(tmp_path):
    module = load_module()
    out = tmp_path / "segmented_core.html"
    out.write_text("<html><body><div id=\"app\"></div></body></html>", encoding="utf-8")
    components = [
        {"value": 0, "label": "pore", "color": "#0077ff", "visible": True},
        {"value": 128, "label": "component 128", "color": "#f0c419", "visible": False},
        {"value": 255, "label": "solid", "color": "#ffff00", "visible": False},
    ]

    module.inject_component_controls(out, components)

    html = out.read_text(encoding="utf-8")
    assert "segmented-core-component-controls" in html
    assert "data-component-index=\"0\"" in html
    assert "data-component-index=\"1\"" in html
    assert "type=\"color\"" in html
    assert "Surface mesh" in html
    assert "Voxel volume" in html
    assert "segmented-core-component-voxel-visible" in html
    assert "segmented-core-apply-components" in html
    assert ">Apply<" in html
    assert "setVisibility" in html
    assert "setColor" in html
    assert "window.segmentedCoreApplyControls" in html
    assert "window.segmentedCoreRenderWindow" in html
    assert "window.segmentedCoreActors" in html
    assert "window.segmentedCoreVolumes" in html
    assert "getRGBTransferFunction" in html
    assert "typeof global !== 'undefined'" in html
    assert "item.getMapper" in html
    assert "renderer.getVolumes" in html
    assert "renderer.getViewProps" in html
    assert "Waiting for VTK objects" in html


def test_component_controls_label_voxel_only_outputs(tmp_path):
    module = load_module()
    out = tmp_path / "segmented_core.html"
    out.write_text("<html><body><div id=\"app\"></div></body></html>", encoding="utf-8")
    components = [
        {
            "value": 255,
            "label": "solid",
            "color": "#ffff00",
            "visible": True,
            "surface_skipped": True,
            "voxel_skipped": False,
            "voxel_visible": True,
        },
    ]

    module.inject_component_controls(out, components)

    html = out.read_text(encoding="utf-8")
    assert "Voxel volume" in html
    assert "Surface mesh + voxel volume" not in html
    assert "class=\"segmented-core-component-visible\" checked disabled" in html


def test_vtk_global_alias_is_declared_for_offline_loader(tmp_path):
    module = load_module()
    out = tmp_path / "segmented_core.html"
    out.write_text(
        "<html><body><script>window.global = window.global || {};</script></body></html>",
        encoding="utf-8",
    )

    module.ensure_vtk_global_alias(out)

    html = out.read_text(encoding="utf-8")
    assert "var global = window.global = window.global || {};" in html


def test_vtk_render_window_is_hooked_for_component_controls(tmp_path):
    module = load_module()
    out = tmp_path / "segmented_core.html"
    out.write_text(
        "getRenderWindow(),r=HA.getSynchronizerContext(),o=HA.decorate(n);"
        "if(global.renderWindow=n,t.fileURL||t.url){",
        encoding="utf-8",
    )

    module.ensure_vtk_global_alias(out)

    html = out.read_text(encoding="utf-8")
    assert "window.segmentedCoreRenderWindow=n" in html
    assert "window.global.renderWindow=n" in html
    assert "if(global.renderWindow=n,t.fileURL||t.url){" not in html
    assert "if(t.fileURL||t.url){" in html


def test_vtk_scene_actor_instances_are_registered_after_synchronize(tmp_path):
    module = load_module()
    out = tmp_path / "segmented_core.html"
    out.write_text("o.synchronize(e.scene),o.render()", encoding="utf-8")

    module.ensure_vtk_global_alias(out)

    html = out.read_text(encoding="utf-8")
    assert "window.segmentedCoreActors" in html
    assert "window.segmentedCoreVolumes" in html
    assert "r.getInstance(e.id)" in html
    assert "vtkOpenGLActor" in html
    assert "vtkVolume" in html
    assert "setVisibility(false)" in html


def test_mouse_wheel_zoom_support_is_injected(tmp_path):
    module = load_module()
    out = tmp_path / "segmented_core.html"
    out.write_text("<html><body><div id=\"vtk-root\"></div></body></html>", encoding="utf-8")

    module.inject_vtk_wheel_zoom_support(out)

    html = out.read_text(encoding="utf-8")
    assert "segmented-core-wheel-zoom-support" in html
    assert "addEventListener('wheel'" in html
    assert "applyVtkDisplayScale" in html
    assert "style.transform = 'scale('" in html
    assert "vtkDisplayScale" in html
    assert "vtkZoomClicks" in html
    assert "Zoom 1" not in html


def test_component_specs_can_default_to_solid_only():
    module = load_module()

    specs = module.component_specs_from_values([0, 255], visible_mode="solid")

    assert specs[0]["value"] == 0
    assert specs[0]["visible"] is False
    assert specs[1]["value"] == 255
    assert specs[1]["visible"] is True


def test_segmented_surface_uses_physical_voxel_spacing():
    module = load_module()

    mesh = module.segmented_surface_mesh(
        tiny_segmented_volume(),
        component_value=255,
        downsample=2,
        voxel_spacing_um=(1.7, 1.7, 1.7),
    )

    assert mesh.bounds[1] > 1.7


def test_multi_component_plotter_creates_one_actor_per_component():
    module = load_module()
    volume = tiny_segmented_volume()
    components = module.component_specs_from_values([0, 128, 255])
    components[1]["visible"] = False
    plotter, actors, metadata = module.build_multi_component_plotter(
        volume,
        component_specs=components,
        downsample=1,
        smooth_iterations=0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        window_size=(320, 240),
        hide_bounds=True,
    )

    try:
        assert [item["value"] for item in metadata] == [0, 128, 255]
        assert len(actors) == 3
        assert all(actor.GetVisibility() for actor in actors)
        assert all(item["voxel_actor_index"] is not None for item in metadata)
        assert all(item["voxel_points"] > 0 for item in metadata)
        assert all(item["voxel_visible"] is False for item in metadata)
        assert all(item["voxel_sample_distance_um"] > 0 for item in metadata)
    finally:
        plotter.close()


def test_multi_component_plotter_can_export_voxel_volume_without_surface_mesh():
    module = load_module()
    volume = tiny_segmented_volume()
    components = module.component_specs_from_values([0, 255], visible_mode="solid")

    plotter, actors, metadata = module.build_multi_component_plotter(
        volume,
        component_specs=components,
        downsample=1,
        smooth_iterations=0,
        voxel_spacing_um=(1.0, 1.0, 1.0),
        window_size=(320, 240),
        hide_bounds=True,
        include_surface=False,
        include_voxel=True,
        voxel_visible_mode="solid",
    )

    try:
        assert actors == []
        assert [item["value"] for item in metadata] == [0, 255]
        assert all(item["surface_skipped"] is True for item in metadata)
        assert all(item["voxel_actor_index"] is not None for item in metadata)
        assert metadata[0]["voxel_visible"] is False
        assert metadata[1]["voxel_visible"] is True
    finally:
        plotter.close()


def test_rendering_modes_reflect_enabled_surface_and_voxel_outputs():
    module = load_module()

    assert module.rendering_modes(include_surface=True, include_voxel=True) == ["surface_mesh", "voxel_volume"]
    assert module.rendering_modes(include_surface=False, include_voxel=True) == ["voxel_volume"]
    assert module.rendering_modes(include_surface=True, include_voxel=False) == ["surface_mesh"]
