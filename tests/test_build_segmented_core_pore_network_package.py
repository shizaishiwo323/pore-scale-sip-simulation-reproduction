from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "pore_network" / "build_segmented_core_pore_network_package.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_segmented_core_pore_network_package", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_input_and_output_root_match_sample16_request():
    module = load_module()

    assert module.DEFAULT_INPUT_TIFF == (
        PROJECT_ROOT
        / "data_inventory"
        / "ct_backed_samples_raw_copy_20260605"
        / "sample_16_Grainstone"
        / "CT_slices"
        / "1-CTseg"
        / "9-16small340.tif"
    )
    assert module.DEFAULT_OUTPUT_ROOT == PROJECT_ROOT / "results" / "pore_network"
    assert module.DEFAULT_PNEXTRACT_EXE == PROJECT_ROOT / "code" / "vendor" / "pnextract" / "bin" / "pnextract.exe"


def test_default_run_name_uses_only_tif_stem():
    module = load_module()

    run_name = module.default_run_name(module.DEFAULT_INPUT_TIFF)

    assert run_name == "9-16small340"


def test_build_result_paths_places_everything_under_one_result_subdir(tmp_path):
    module = load_module()

    paths = module.build_result_paths(
        output_root=tmp_path / "results" / "pore_network",
        run_name="sample_16_run",
        sample_id="9-16small340",
    )

    result_dir = tmp_path / "results" / "pore_network" / "sample_16_run"
    assert paths.result_dir == result_dir
    assert paths.binary_tiff == result_dir / "segmented_core" / "9-16small340_solid255_pore0.tiff"
    assert paths.binary_raw == result_dir / "segmented_core" / "9-16small340_solid255_pore0.raw"
    assert paths.digital_rock_html == result_dir / "digital_rock" / "9-16small340_fiji3d_volume_interactive.html"
    assert paths.pore_network_html == result_dir / "pore_network" / "9-16small340_pnextract_ballstick_interactive.html"
    assert paths.distribution_png == result_dir / "pore_network" / "9-16small340_pore_throat_frequency_distribution.png"
    assert paths.run_summary == result_dir / "run_summary.json"

    for path in [
        paths.binary_tiff,
        paths.binary_raw,
        paths.digital_rock_html,
        paths.pore_network_html,
        paths.distribution_png,
        paths.run_summary,
    ]:
        assert result_dir in path.parents or path == result_dir


def test_write_binary_core_maps_auto_minority_to_pore0_and_solid255(tmp_path):
    module = load_module()
    input_tiff = tmp_path / "segmented_float_labels.tif"
    output_tiff = tmp_path / "binary" / "sample_solid255_pore0.tiff"
    output_raw = tmp_path / "binary" / "sample_solid255_pore0.raw"
    metadata_out = tmp_path / "source_data" / "sample_solid255_pore0_remap_metadata.json"
    volume = np.array(
        [
            [[1, 1], [1, 2], [1, 1]],
            [[1, 1], [2, 1], [1, 1]],
        ],
        dtype=np.float32,
    )
    tifffile.imwrite(input_tiff, volume)

    summary = module.write_binary_core(
        input_tiff=input_tiff,
        output_tiff=output_tiff,
        output_raw=output_raw,
        metadata_out=metadata_out,
        pore_values=module.AUTO_PORE_VALUES,
        solid_value=255,
    )

    mapped = tifffile.imread(output_tiff)
    assert mapped.dtype == np.uint8
    assert mapped.tolist() == [
        [[255, 255], [255, 0], [255, 255]],
        [[255, 255], [0, 255], [255, 255]],
    ]
    assert output_raw.read_bytes() == mapped.tobytes(order="C")
    assert summary["pore_values"] == [2]
    assert summary["solid_source_values"] == [1]
    assert summary["pore_voxels"] == 2
    assert summary["solid_voxels"] == 10
    assert summary["porosity"] == 2 / 12
    metadata = json.loads(metadata_out.read_text(encoding="utf-8"))
    assert metadata["mapping"] == {"2": 0, "1": 255}
    assert metadata["output_convention"] == {"pore": 0, "solid": 255}


def test_build_child_commands_use_binary_core_and_package_outputs(tmp_path):
    module = load_module()
    paths = module.build_result_paths(
        output_root=tmp_path / "results" / "pore_network",
        run_name="sample_16_run",
        sample_id="9-16small340",
    )

    digital_cmd = module.build_digital_rock_command(
        python_exe=Path("python.exe"),
        paths=paths,
        voxel_size_um=1.7,
        digital_rock_downsample=2,
    )
    network_cmd = module.build_pore_network_command(
        python_exe=Path("python.exe"),
        paths=paths,
        title="9-16small340_pnextract",
        voxel_size_um=1.7,
        network_downsample=4,
        distribution_bins=24,
        pnextract_exe=None,
    )

    digital_joined = " ".join(str(part) for part in digital_cmd)
    network_joined = " ".join(str(part) for part in network_cmd)
    assert "render_segmented_core_fiji3d_html.py" in digital_joined
    assert str(paths.binary_tiff) in digital_cmd
    assert str(paths.digital_rock_html) in digital_cmd
    assert str(paths.digital_rock_metadata) in digital_cmd
    assert "--components" in digital_cmd
    assert "0" in digital_cmd
    assert "255" in digital_cmd

    assert "run_segmented_core_pnextract_ballstick.py" in network_joined
    assert str(paths.binary_tiff) in network_cmd
    assert str(paths.pore_network_html) in network_cmd
    assert str(paths.pore_network_metadata) in network_cmd
    assert str(paths.distribution_png) in network_cmd
    assert str(paths.distribution_metadata) in network_cmd
    assert "--pore-values" in network_cmd
    assert "0" in network_cmd
    assert "--distribution-bins" in network_cmd
    assert "24" in network_cmd


def test_resolve_pnextract_exe_uses_bundled_original_pnextract_when_present():
    module = load_module()

    resolved = module.resolve_pnextract_exe(None)

    assert resolved == module.DEFAULT_PNEXTRACT_EXE
