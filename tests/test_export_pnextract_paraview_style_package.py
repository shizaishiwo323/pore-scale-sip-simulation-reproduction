from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "pore_network" / "export_pnextract_paraview_style_package.py"


def load_module():
    spec = importlib.util.spec_from_file_location("export_pnextract_paraview_style_package", SCRIPT_PATH)
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


def test_builds_separate_paraview_sources_with_radius_arrays():
    module = load_module()
    pores, throats = sample_tables()

    pores_mesh, throats_mesh, counts = module.build_paraview_sources(
        pores,
        throats,
        voxel_size_m=2.8e-6,
        pore_radius_scale=1.05,
        throat_radius_scale=0.7,
        min_throat_radius_vox=0.55,
    )

    assert counts == {"pores": 3, "throats_total": 3, "throats_rendered": 2}
    assert pores_mesh.n_points == 3
    assert pores_mesh.n_verts == 3
    assert np.allclose(pores_mesh.point_data["pore_radius_vox"], [1.05, 2.1, 3.15])
    assert throats_mesh.n_lines == 2
    assert throats_mesh.cell_data["throat_radius_vox"].tolist() == [0.55, 1.4]


def test_exports_vtp_sources_and_paraview_style_script(tmp_path):
    module = load_module()
    pores, throats = sample_tables()

    metadata = module.export_package(
        pores,
        throats,
        out_dir=tmp_path,
        prefix="sample",
        source_pores_csv=Path("pores.csv"),
        source_throats_csv=Path("throats.csv"),
        voxel_size_m=2.8e-6,
        pore_radius_scale=1.05,
        throat_radius_scale=0.7,
        min_throat_radius_vox=0.55,
    )

    pores_path = Path(metadata["pores_vtp"])
    throats_path = Path(metadata["throats_vtp"])
    script_path = Path(metadata["paraview_style_script"])
    assert pores_path.exists()
    assert throats_path.exists()
    assert script_path.exists()
    assert pv.read(pores_path).point_data["pore_radius_vox"].size == 3
    assert pv.read(throats_path).cell_data["throat_radius_vox"].size == 2
    script = script_path.read_text(encoding="utf-8")
    assert "GlyphType='Sphere'" in script
    assert "Tube(" in script
    assert "DiffuseColor = [1.0, 0.0, 0.0]" in script
    assert "DiffuseColor = [0.0, 0.22, 1.0]" in script
    assert "GridAxes3DActor" in script
    assert json.loads(Path(metadata["metadata_json"]).read_text(encoding="utf-8"))["renderer"] == "paraview-glyph-tube-style"


def test_exports_single_baked_vtp_for_direct_paraview_open(tmp_path):
    module = load_module()
    pores, throats = sample_tables()

    metadata = module.export_package(
        pores,
        throats,
        out_dir=tmp_path,
        prefix="sample",
        source_pores_csv=Path("pores.csv"),
        source_throats_csv=Path("throats.csv"),
        voxel_size_m=2.8e-6,
        pore_radius_scale=1.05,
        throat_radius_scale=0.7,
        min_throat_radius_vox=0.55,
        write_baked=True,
        baked_sphere_resolution=8,
        baked_tube_sides=6,
    )

    baked_path = Path(metadata["baked_single_file_vtu"])
    assert baked_path.exists()
    baked = pv.read(baked_path)
    assert baked.n_points > 0
    assert baked.n_cells > 0
    assert "style_rgb" in baked.cell_data
    colors = {tuple(row) for row in baked.cell_data["style_rgb"].tolist()}
    assert (255, 0, 0) in colors
    assert (0, 56, 255) in colors
    assert metadata["baked_single_file_note"].startswith("Open this one VTU directly")
