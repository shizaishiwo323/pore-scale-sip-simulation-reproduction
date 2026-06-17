from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "pore_network" / "scan_berea_preprocess_sensitivity.py"


def load_module():
    spec = importlib.util.spec_from_file_location("scan_berea_preprocess_sensitivity", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_apply_preprocess_variant_does_not_mutate_input_and_dilates_pore():
    module = load_module()
    volume = np.ones((3, 3, 3), dtype=np.uint8)
    volume[1, 1, 1] = 0
    original = volume.copy()

    out, metadata = module.apply_preprocess_variant(volume, "pore_dilate_1")

    assert np.array_equal(volume, original)
    assert int(np.count_nonzero(out == 0)) > 1
    assert metadata["variant"] == "pore_dilate_1"
    assert metadata["operation"] == "binary_dilation_on_pore_mask"


def test_apply_preprocess_variant_remove_small_pores_removes_isolated_component():
    module = load_module()
    volume = np.ones((5, 5, 5), dtype=np.uint8)
    volume[0, 0, 0] = 0
    volume[2:4, 2:4, 2:4] = 0

    out, metadata = module.apply_preprocess_variant(volume, "remove_small_pore_components_4")

    assert int(np.count_nonzero(out == 0)) == 8
    assert out[0, 0, 0] == 1
    assert metadata["removed_components"] == 1
    assert metadata["removed_pore_voxels"] == 1


def test_write_variant_inputs_creates_raw_mhd_and_preserves_source_path(tmp_path):
    module = load_module()
    volume = np.ones((2, 3, 4), dtype=np.uint8)
    volume[0, 0, 0] = 0
    source = tmp_path / "source.tiff"
    source.write_bytes(b"sentinel")

    summary = module.write_variant_inputs(
        volume,
        tmp_path / "out",
        source_path=source,
        variant="identity",
        title="case_identity",
        voxel_size_um=2.8,
        pnextract_lines=["minRPore 1.75"],
        metadata={"operation": "none"},
    )

    assert source.read_bytes() == b"sentinel"
    assert Path(summary["raw_path"]).exists()
    assert Path(summary["mhd_path"]).exists()
    assert Path(summary["metadata_path"]).exists()
    assert summary["porosity"] == 1 / 24
    assert "minRPore 1.75" in Path(summary["mhd_path"]).read_text(encoding="utf-8")


def test_defaults_use_original_vendor_pnextract_without_extra_parameter_lines():
    module = load_module()

    assert module.DEFAULT_PNEXTRACT_EXE == PROJECT_ROOT / "code" / "vendor" / "pnextract" / "bin" / "pnextract.exe"
    assert module.DEFAULT_PNEXTRACT_LINES == []
