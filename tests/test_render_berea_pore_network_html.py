from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "pore_network" / "render_berea_pore_network_html.py"


def load_module():
    spec = importlib.util.spec_from_file_location("render_berea_pore_network_html", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    pores = pd.DataFrame(
        {
            "pore_id": [1, 2, 3],
            "pore_center_x_m": [0.0, 2.8e-6, 5.6e-6],
            "pore_center_y_m": [0.0, 0.0, 2.8e-6],
            "pore_center_z_m": [0.0, 2.8e-6, 2.8e-6],
            "pore_radius_m": [2.8e-6, 5.6e-6, 8.4e-6],
        }
    )
    throats = pd.DataFrame(
        {
            "pore1_id": [1, 2, -1],
            "pore2_id": [2, 3, 1],
            "throat_radius_m": [1.4e-6, 5.6e-6, 5.6e-6],
        }
    )
    return pores, throats


def test_scene_payload_uses_voxel_units_and_internal_throats_only():
    module = load_module()
    pores, throats = sample_tables()

    payload = module.build_scene_payload(
        pores,
        throats,
        voxel_size_m=2.8e-6,
        sphere_radius_scale=1.0,
        tube_radius_scale=0.5,
        min_tube_radius_vox=0.75,
    )

    assert payload["counts"] == {"pores": 3, "throats_total": 3, "throats_rendered": 2}
    assert payload["pores"][1] == [1.0, 0.0, 1.0, 2.0]
    assert payload["segments"][0] == [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.75]
    assert payload["segments"][1][-1] == 1.0


def test_pyvista_materials_match_static_specular_png_style():
    module = load_module()

    assert module.MATERIALS["sphere"] == {
        "ambient": 0.38,
        "diffuse": 0.76,
        "specular": 0.64,
        "specular_power": 46,
        "smooth_shading": True,
    }
    assert module.MATERIALS["tube"] == {
        "ambient": 0.42,
        "diffuse": 0.72,
        "specular": 0.42,
        "specular_power": 28,
        "smooth_shading": True,
    }


def test_pyvista_plotter_uses_real_actor_lighting_not_marker_highlights():
    module = load_module()
    pores, throats = sample_tables()
    payload = module.build_scene_payload(
        pores,
        throats,
        voxel_size_m=2.8e-6,
        sphere_radius_scale=1.0,
        tube_radius_scale=0.5,
        min_tube_radius_vox=0.75,
    )

    plotter, actor_metadata = module.build_pyvista_plotter(payload, hide_bounds=True, window_size=(320, 240))

    try:
        sphere_property = actor_metadata["sphere_actor"].GetProperty()
        tube_property = actor_metadata["tube_actor"].GetProperty()
        assert sphere_property.GetSpecular() == module.MATERIALS["sphere"]["specular"]
        assert sphere_property.GetSpecularPower() == module.MATERIALS["sphere"]["specular_power"]
        assert tube_property.GetSpecular() == module.MATERIALS["tube"]["specular"]
        assert tube_property.GetSpecularPower() == module.MATERIALS["tube"]["specular_power"]
        assert "highlight" not in actor_metadata
    finally:
        plotter.close()


def test_exported_html_is_vtk_single_file_with_no_plotly_marker_workaround(tmp_path):
    module = load_module()
    pores, throats = sample_tables()
    payload = module.build_scene_payload(
        pores,
        throats,
        voxel_size_m=2.8e-6,
        sphere_radius_scale=1.0,
        tube_radius_scale=0.5,
        min_tube_radius_vox=0.75,
    )
    plotter, _actor_metadata = module.build_pyvista_plotter(payload, hide_bounds=True, window_size=(320, 240))
    out = tmp_path / "scene.html"

    try:
        module.export_html(plotter, out)
    finally:
        plotter.close()

    html = out.read_text(encoding="utf-8", errors="ignore")
    assert "vtk" in html.lower()
    assert "<script src=" not in html
    assert "Plotly.newPlot" not in html
    assert "pores specular highlight" not in html


def test_porosity_overlay_is_injected_in_top_right(tmp_path):
    module = load_module()
    out = tmp_path / "network.html"
    out.write_text("<html><body><div id=\"app\"></div></body></html>", encoding="utf-8")
    stats = {
        "porosity": 0.08279859909405538,
        "pore_voxels": 31235076,
        "total_voxels": 377241600,
        "pore_value": 0,
    }

    module.inject_porosity_overlay(out, stats)

    html = out.read_text(encoding="utf-8")
    assert "pore-network-porosity-overlay" in html
    assert "Porosity: 8.28%" in html
    assert "31,235,076 pore px / 377,241,600 total px" in html
    assert "top: 16px" in html
    assert "right: 16px" in html


def test_mouse_wheel_zoom_support_is_injected(tmp_path):
    module = load_module()
    out = tmp_path / "network.html"
    out.write_text("<html><body><div id=\"vtk-root\"></div></body></html>", encoding="utf-8")

    module.inject_vtk_wheel_zoom_support(out)

    html = out.read_text(encoding="utf-8")
    assert "pore-network-wheel-zoom-support" in html
    assert "addEventListener('wheel'" in html
    assert "applyVtkDisplayScale" in html
    assert "style.transform = 'scale('" in html
    assert "vtkDisplayScale" in html
    assert "vtkZoomClicks" in html
    assert "Zoom 1" not in html


def test_porosity_stats_are_computed_from_voxel_values():
    module = load_module()
    volume = np.array(
        [
            [[0, 255], [0, 128]],
            [[255, 255], [0, 0]],
        ],
        dtype=np.uint8,
    )

    stats = module.porosity_stats_from_volume(volume, pore_value=0, solid_value=255)

    assert stats["pore_value"] == 0
    assert stats["solid_value"] == 255
    assert stats["pore_voxels"] == 4
    assert stats["solid_voxels"] == 3
    assert stats["other_voxels"] == 1
    assert stats["total_voxels"] == 8
    assert stats["porosity"] == 0.5
