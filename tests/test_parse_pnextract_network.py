from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "pore_network" / "parse_pnextract_network.py"


def load_module():
    spec = importlib.util.spec_from_file_location("parse_pnextract_network", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_minimal_network(prefix: Path) -> None:
    prefix.with_name(prefix.name + "_node1.dat").write_text(
        "\n".join(
            [
                "2",
                "1 0.0 0.0 0.0",
                "2 1.0e-5 0.0 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prefix.with_name(prefix.name + "_node2.dat").write_text(
        "\n".join(
            [
                "1 1.0e-15 2.0e-6 0.048 0.0",
                "2 2.0e-15 3.0e-6 0.048 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prefix.with_name(prefix.name + "_link1.dat").write_text(
        "\n".join(
            [
                "1",
                "1 1 2 1.0e-6 0.048 1.0e-5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prefix.with_name(prefix.name + "_link2.dat").write_text(
        "1 1 2 2.0e-6 3.0e-6 4.0e-6 8.0e-18 0.0\n",
        encoding="utf-8",
    )


def test_contact_split_dong_blunt_diagnostics_become_explicit_membrane_geometry(tmp_path: Path):
    module = load_module()
    prefix = tmp_path / "sample"
    outdir = tmp_path / "parsed"
    write_minimal_network(prefix)
    prefix.with_name(prefix.name + "_contact_patch_dong_blunt_diagnostics.csv").write_text(
        "\n".join(
            [
                "split_throat_id,original_throat_id,pore1_id,pore2_id,patch_index,patch_count,face_count,"
                "length_voxels,radius_voxels,radius_for_length_voxels,radius_unclamped_voxels,"
                "max_radius_from_neighbor_pores_voxels,contact_active_aperture_radius_voxels,"
                "contact_active_aperture_all_contact_splits,active_aperture_limited,shape_factor,"
                "volume_voxels3,raw_contact_gap_length_voxels,raw_dong_blunt_length_voxels,"
                "raw_contact_thickness_length_voxels,original_area_over_length_voxels,alpha,"
                "min_effective_throat_length_voxels,contact_gap_bound_threshold_voxels",
                "1,7,1,2,0,1,3,2.0,0.5,0.75,1.1,2.5,0.17,1,1,0.048,4.0,0.03,2.0,0.05,1.2,0.6,0.05,-1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    module.parse_network(prefix=prefix, outdir=outdir)

    throats = pd.read_csv(outdir / "throats.csv")
    summary = json.loads((outdir / "network_summary.json").read_text(encoding="utf-8"))
    row = throats.iloc[0]
    expected_dx = 2.0e-6
    expected_radius = 0.5 * expected_dx
    expected_area = expected_radius**2 / (4.0 * 0.048)

    assert np.isclose(row["membrane_relaxation_length_m"], 4.0e-6)
    assert np.isclose(row["membrane_zdc_length_m"], 4.0e-6)
    assert np.isclose(row["membrane_active_radius_m"], expected_radius)
    assert np.isclose(row["membrane_active_area_m2"], expected_area)
    assert row["membrane_geometry_source"] == "contact_patch_dong_blunt_diagnostics"
    assert row["contact_active_aperture_limited"] == 1
    assert np.isclose(row["contact_raw_contact_gap_length_m"], 0.03 * expected_dx)
    assert summary["contact_patch_dong_blunt_diagnostics_merged"] is True
    assert summary["membrane_geometry_mode"] == "explicit_contact_split_diagnostics"


def test_parse_network_without_diagnostics_keeps_single_throat_geometry_mode(tmp_path: Path):
    module = load_module()
    prefix = tmp_path / "sample"
    outdir = tmp_path / "parsed"
    write_minimal_network(prefix)

    module.parse_network(prefix=prefix, outdir=outdir)

    throats = pd.read_csv(outdir / "throats.csv")
    summary = json.loads((outdir / "network_summary.json").read_text(encoding="utf-8"))

    assert "membrane_relaxation_length_m" not in throats.columns
    assert summary["contact_patch_dong_blunt_diagnostics_merged"] is False
    assert summary["membrane_geometry_mode"] == "single_throat_geometry"
