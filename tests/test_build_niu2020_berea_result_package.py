from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "sip_simulation" / "build_niu2020_berea_result_package.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_niu2020_berea_result_package", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_package_dir_is_under_results():
    module = load_module()

    assert module.DEFAULT_RESULT_DIR == PROJECT_ROOT / "results" / "niu2020_berea_reproduction_20260617_original_pnextract_defaults"


def test_writes_chinese_parameter_config_inside_result_dir(tmp_path):
    module = load_module()
    out = tmp_path / "configs" / "niu2020_berea_parameters_zh.py"

    module.write_chinese_parameter_config(out)

    text = out.read_text(encoding="utf-8")
    assert "Niu 2020 Berea AC3D/SIP 复现参数" in text
    assert "体素边长" in text
    assert "动态孔径" in text
    assert "水相电导率" in text
    assert "Section 4.2" in text
    assert "NIU2020_BEREA_CONFIG" in text
    assert "不要把 Figure8.xlsx 的 Simulation 列当成本项目模拟结果" in text
    assert text == module.CONFIG_SOURCE.read_text(encoding="utf-8")


def test_build_visualization_commands_use_required_scripts(tmp_path):
    module = load_module()
    binary = module.remap_niu_tiff_to_notebook_binary(
        tmp_path / "input.tiff",
        tmp_path / "segmented_core",
        sample_id="test",
        input_volume=np.asarray([[[256, 512], [512, 512]]], dtype=np.uint16),
        voxel_size_um=2.8,
    )
    commands = module.build_visualization_commands(result_dir=tmp_path, python_exe=Path("python.exe"), binary=binary)
    joined = "\n".join(" ".join(str(part) for part in command) for command in commands)

    assert "visualize_digital_rock_3d.py" not in joined
    assert "render_segmented_core_fiji3d_html.py" in joined
    assert str(binary["binary_raw"]) not in joined
    assert "--pore-label 0" not in joined
    assert "--solid-label 255" not in joined
    assert str(tmp_path / "digital_rock" / "berea_digital_rock_pore_full.png") not in joined
    assert str(tmp_path / "digital_rock" / "berea_fiji3d_volume_interactive.html") in joined
    assert str(binary["binary_tiff"]) in joined
    assert "--solid-value 255" in joined
    assert "--components 0 255" in joined


def test_remaps_niu_tiff_to_notebook_binary(tmp_path):
    module = load_module()
    out_dir = tmp_path / "segmented_core"
    volume = np.asarray([[[256, 512], [256, 512]], [[512, 512], [256, 512]]], dtype=np.uint16)

    summary = module.remap_niu_tiff_to_notebook_binary(
        tmp_path / "microCT_Berea.tiff",
        out_dir,
        sample_id="niu2020_berea",
        input_volume=volume,
        voxel_size_um=2.8,
    )

    binary = tifffile.imread(summary["binary_tiff"])
    metadata = json.loads(Path(summary["metadata_json"]).read_text(encoding="utf-8"))
    assert set(np.unique(binary).tolist()) == {0, 255}
    assert int(np.count_nonzero(binary == 0)) == 3
    assert int(np.count_nonzero(binary == 255)) == 5
    assert metadata["pore_output_value"] == 0
    assert metadata["solid_output_value"] == 255
    assert metadata["solid_source_values"] == [512]
    assert metadata["pore_source_values"] == [256]


def test_build_sip_plot_command_writes_inside_result_dir(tmp_path):
    module = load_module()

    command = module.build_sip_plot_command(result_dir=tmp_path, python_exe=Path("python.exe"))
    joined = " ".join(str(part) for part in command)

    assert "plot_niu2020_conductivity_mechanism_comparison.py" in joined
    assert str(tmp_path / "figures" / "niu2020_conductivity_mechanism_comparison") in joined
    assert str(tmp_path / "source_data" / "niu2020_conductivity_mechanism_comparison_source_data.csv") in joined
    assert str(tmp_path / "provenance" / "niu2020_conductivity_mechanism_comparison_provenance.md") in joined


def test_build_pnextract_command_matches_notebook_pipeline(tmp_path):
    module = load_module()
    binary = {
        "binary_tiff": str(tmp_path / "segmented_core" / "niu2020_berea_solid255_pore0.tiff"),
    }

    command = module.build_pnextract_ballstick_command(result_dir=tmp_path, python_exe=Path("python.exe"), binary=binary)
    joined = " ".join(str(part) for part in command)

    assert "run_segmented_core_pnextract_ballstick.py" in joined
    assert str(tmp_path / "segmented_core" / "niu2020_berea_solid255_pore0.tiff") in joined
    assert "--pore-values 0" in joined
    assert "--solid-value 255" in joined
    assert str(tmp_path / "pore_network" / "berea_pnextract_ballstick_interactive.html") in joined
    assert str(tmp_path / "pore_network" / "pnextract") in joined
    assert str(tmp_path / "pore_network" / "berea_pnextract_pore_throat_frequency_distribution.png") in joined
    assert "pnextract.exe" in joined
    assert "pnextract_contact_split_v2.exe" not in joined
    assert "--pnextract-line" not in joined


def test_vendor_default_parameter_file_appends_no_parameters():
    module = load_module()

    lines = module.load_pnextract_parameter_lines(module.DEFAULT_PNEXTRACT_PARAMETER_FILE)

    assert lines == []


def test_network_html_requires_coordinate_columns(tmp_path):
    module = load_module()
    pores = tmp_path / "pores.csv"
    throats = tmp_path / "throats.csv"
    pd.DataFrame(
        {
            "pore_id": [1, 2],
            "pore_radius_m": [1e-6, 2e-6],
            "pore_center_x_m": [0.0, 1e-5],
            "pore_center_y_m": [0.0, 0.0],
            "pore_center_z_m": [0.0, 0.0],
        }
    ).to_csv(pores, index=False)
    pd.DataFrame({"pore1_id": [1], "pore2_id": [2], "throat_radius_m": [5e-7]}).to_csv(throats, index=False)

    assert module.network_has_renderable_coordinates(pores, throats) is True

    pd.DataFrame({"pore_id": [1], "pore_radius_m": [1e-6]}).to_csv(pores, index=False)

    assert module.network_has_renderable_coordinates(pores, throats) is False


def test_writes_figure4_style_distribution_from_geometry(tmp_path):
    module = load_module()
    figure5 = tmp_path / "Figure5.xlsx"
    pd.DataFrame(
        {
            "pore node size (m)": [1e-6, 2e-6, 4e-6],
            "pore node relative volume": [0.1, 0.2, 0.7],
            "pore throat length (m)": [2e-7, 1e-6, 5e-6],
            "pore throat relative volume": [0.3, 0.4, 0.3],
        }
    ).to_excel(figure5, index=False)
    out = tmp_path / "figures" / "figure4.png"
    source = tmp_path / "source_data" / "figure4.csv"

    summary = module.write_figure4_style_distribution(figure5, out, source)

    assert out.stat().st_size > 0
    assert source.stat().st_size > 0
    assert summary["figure_png"] == str(out)
    assert summary["source_data_csv"] == str(source)
