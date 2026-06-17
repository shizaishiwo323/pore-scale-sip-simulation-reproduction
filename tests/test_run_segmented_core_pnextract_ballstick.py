from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "pore_network" / "run_segmented_core_pnextract_ballstick.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_segmented_core_pnextract_ballstick", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_maps_segmented_volume_to_pnextract_pore0_solid1():
    module = load_module()
    volume = np.array(
        [
            [[0, 255, 128], [0, 255, 255]],
            [[255, 0, 128], [0, 0, 255]],
        ],
        dtype=np.uint8,
    )

    mapped, stats = module.map_segmented_volume_to_pnextract(volume, pore_values=[0])

    assert mapped.dtype == np.uint8
    assert mapped.tolist() == [
        [[0, 1, 1], [0, 1, 1]],
        [[1, 0, 1], [0, 0, 1]],
    ]
    assert stats["pore_voxels"] == 5
    assert stats["solid_voxels"] == 7
    assert stats["total_voxels"] == 12
    assert stats["porosity"] == 5 / 12
    assert stats["source_values"] == {"0": 5, "128": 2, "255": 5}


def test_infers_minority_component_as_pore_for_two_class_segmentation():
    module = load_module()
    volume = np.array(
        [
            [[10, 10, 10], [10, 42, 10]],
            [[10, 10, 42], [10, 10, 10]],
        ],
        dtype=np.uint16,
    )

    inference = module.infer_two_class_pore_solid_values(volume)

    assert inference == {
        "pore_values": [42],
        "solid_value": 10,
        "mode": "auto_minority_pore_majority_solid",
    }


def test_writes_mhd_with_xyz_dims_for_zyx_numpy_volume(tmp_path):
    module = load_module()
    raw_path = tmp_path / "sample_pore0_solid1.raw"
    raw_path.write_bytes(b"\x00\x01")
    mhd_path = tmp_path / "sample.mhd"

    module.write_pnextract_mhd(
        mhd_path,
        raw_path,
        shape_zyx=(2, 3, 4),
        voxel_size_um=2.5,
        title="sample_89_network",
    )

    text = mhd_path.read_text(encoding="utf-8")
    assert "DimSize = 4 3 2" in text
    assert "ElementSize = 2.5 2.5 2.5" in text
    assert "ElementDataFile = sample_pore0_solid1.raw" in text
    assert "title sample_89_network" in text
    assert "write_cnm true" in text
    assert "write_vtkNetwork true" in text


def test_writes_mhd_with_extra_pnextract_lines(tmp_path):
    module = load_module()
    raw_path = tmp_path / "sample_pore0_solid1.raw"
    raw_path.write_bytes(b"\x00\x01")
    mhd_path = tmp_path / "sample.mhd"

    module.write_pnextract_mhd(
        mhd_path,
        raw_path,
        shape_zyx=(2, 3, 4),
        voxel_size_um=2.8,
        title="explicit_pnextract_parameter_case",
        pnextract_lines=[
            "minRPore 1.75",
            "write_cnm true",
        ],
    )

    text = mhd_path.read_text(encoding="utf-8")
    assert "minRPore 1.75" in text
    assert "write_cnm true" in text


def test_build_render_command_passes_segmented_volume_for_porosity_overlay(tmp_path):
    module = load_module()
    cmd = module.build_render_command(
        python_exe=Path("python"),
        pores_csv=tmp_path / "pores.csv",
        throats_csv=tmp_path / "throats.csv",
        html_out=tmp_path / "network.html",
        metadata_out=tmp_path / "metadata.json",
        segmented_volume=tmp_path / "89seged.tiff",
        voxel_size_m=2.5e-6,
        pore_value=0,
        solid_value=255,
    )

    assert "code\\scripts\\pore_network\\render_berea_pore_network_html.py" in " ".join(str(part) for part in cmd)
    assert "--segmented-volume" in cmd
    assert str(tmp_path / "89seged.tiff") in cmd
    assert "--pore-value" in cmd
    assert "0" in cmd
    assert "--solid-value" in cmd
    assert "255" in cmd


def test_run_command_replaces_undecodable_subprocess_output(tmp_path):
    module = load_module()

    result = module.run_command(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'\\x8d')"],
        cwd=tmp_path,
    )

    assert result.returncode == 0
    assert result.stdout


def test_default_downsample_is_full_resolution():
    module = load_module()

    assert module.DEFAULT_DOWNSAMPLE == 1


def test_writes_pore_throat_distribution_png_and_metadata_in_um(tmp_path):
    module = load_module()
    pores_csv = tmp_path / "pores.csv"
    throats_csv = tmp_path / "throats.csv"
    figure_out = tmp_path / "pore_throat_distribution.png"
    metadata_out = tmp_path / "pore_throat_distribution_metadata.json"
    pd.DataFrame(
        {
            "pore_id": [1, 2, 3, 4],
            "pore_radius_m": [1.0e-6, 2.0e-6, 4.0e-6, 8.0e-6],
            "pore_volume_m3": [1.0, 1.0, 1.0, 1.0],
        }
    ).to_csv(pores_csv, index=False)
    pd.DataFrame(
        {
            "throat_id": [1, 2, 3],
            "throat_length_m": [2.0e-6, 6.0e-6, 18.0e-6],
            "throat_volume_m3": [1.0, 1.0, 1.0],
        }
    ).to_csv(throats_csv, index=False)

    summary = module.write_pore_throat_distribution_figure(
        pores_csv=pores_csv,
        throats_csv=throats_csv,
        figure_out=figure_out,
        metadata_out=metadata_out,
        bins=3,
    )

    assert figure_out.exists()
    assert figure_out.stat().st_size > 0
    assert summary["units"] == "um"
    assert summary["pore_node_size"]["fraction_kind"] == "frequency_fraction"
    assert summary["pore_throat_length"]["fraction_kind"] == "frequency_fraction"
    assert np.isclose(sum(summary["pore_node_size"]["hist_fraction"]), 1.0)
    assert np.isclose(sum(summary["pore_throat_length"]["hist_fraction"]), 1.0)
    metadata = json.loads(metadata_out.read_text(encoding="utf-8"))
    assert metadata["figure_png"] == str(figure_out)
    assert metadata["pore_node_size"]["size_unit"] == "um"
    assert metadata["pore_throat_length"]["size_unit"] == "um"


def test_annotates_render_metadata_with_downsample_and_voxel_assumption(tmp_path):
    module = load_module()
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text('{"renderer": "pyvista/vtk single-file html"}', encoding="utf-8")

    module.annotate_render_metadata(
        metadata_path,
        {
            "downsample": 4,
            "source_voxel_size_um": 1.0,
            "effective_voxel_size_um": 4.0,
            "note": "test note",
        },
    )

    text = metadata_path.read_text(encoding="utf-8")
    assert '"pnextract_extraction"' in text
    assert '"downsample": 4' in text
    assert '"source_voxel_size_um": 1.0' in text
    assert '"effective_voxel_size_um": 4.0' in text
