from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "sip_simulation"
sys.path.insert(0, str(SCRIPT_DIR))


def test_formal_run_requires_full_xyz_uncropped_explicit_labels_and_axis_order():
    from compute_dynamic_pore_size import validate_run_mode

    args = Namespace(
        run_mode="formal",
        crop_start=None,
        crop_size=None,
        directions=["x", "z"],
        axis_order="zyx",
        pore_label=1,
        solid_label=2,
    )

    with pytest.raises(ValueError, match="directions"):
        validate_run_mode(args)


def test_smoke_run_marks_manifest_as_not_formal():
    from compute_dynamic_pore_size import build_manifest

    manifest = build_manifest(
        directional_rows=[
            {
                "physical_direction": "x",
                "dynamic_pore_size_m": 2.0,
                "volume_integral_v2_m": 2.0,
                "surface_integral_v2": 2.0,
                "uniform_field_proxy_m": 1.0,
                "interface_face_count": 2,
                "pore_voxel_count": 8,
                "pore_volume_m3": 8.0,
                "voxel_surface_area_m2": 1.0,
                "effective_conductivity_s_m": 1.0,
                "residual_norm": 0.0,
                "direction_valid": True,
            }
        ],
        args=Namespace(
            run_mode="smoke",
            axis_order="zyx",
            voxel_size_m=1.0,
            pore_label=1,
            solid_label=2,
            raw="input.raw",
            shape=[2, 2, 2],
            crop_start=[0, 0, 0],
            crop_size=[2, 2, 2],
            backend="cpu",
            preconditioner="jacobi",
            fft_reference="pore",
            gauge_mode="active-domain",
            solver_dtype="complex128",
            rtol=1.0e-8,
            maxiter=20,
            directions=["x"],
        ),
        physical_to_array_axis={"x": 2},
        input_sha256="abc",
        git_commit="deadbeef",
    )

    assert manifest["formal_dynamic_pore_size"] is False
    assert manifest["dynamic_pore_size_use"] == "diagnostic_only_not_formal"
