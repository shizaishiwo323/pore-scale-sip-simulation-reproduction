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
    assert "true_residual_norm" in text
    assert "gauge_mode = auto" in text
    assert "frequency_match_mode = exact" in text
    assert "complex128 checkpoint" in text
    assert "complex128 trusted full-grid solve" in text
    assert "fft_reference = pore" in text
    assert "formal_acceptance_rtol = 1e-5" in text
    assert "sigma_xx" in text
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


def test_writes_diagnostic_membrane_model_summary_with_provenance(tmp_path):
    module = load_module()
    component_csv = tmp_path / "dual_length_membrane_volume_area_component_spectrum.csv"
    pd.DataFrame(
        {
            "frequency_hz": [46415.8883361278],
            "membrane_geometry_mode": ["diagnostic_two_length_active_passive_volume_area"],
            "passive_area_source": ["volume_over_passive_length_area"],
            "passive_length_source": ["pore1_plus_pore2"],
            "passive_weight_source": ["volume_over_passive_length"],
            "passive_zdc_source": ["volume_over_passive_length_area"],
            "passive_branch_alpha": [0.0163545307975375],
        }
    ).to_csv(component_csv, index=False)
    fullgrid_summary = tmp_path / "volume_area_fullgrid_summary.json"
    fullgrid_summary.write_text(
        json.dumps(
            {
                "metrics": {
                    "volume_area_dual_length": {
                        "peak_normalized_rmse": 0.025100752414070043,
                        "candidate_peak_frequency_hz": 46415.8883361278,
                        "ratio_at_paper_peak": 0.9880903129604636,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "provenance" / "diagnostic_membrane_model_summary.json"

    summary = module.write_diagnostic_membrane_model_summary(
        component_csv=component_csv,
        fullgrid_summary_json=fullgrid_summary,
        out=out,
    )

    written = json.loads(out.read_text(encoding="utf-8"))
    assert summary == written
    assert written["status"] == "diagnostic_not_default_niu2020_parameter"
    assert written["membrane_geometry_mode"] == "diagnostic_two_length_active_passive_volume_area"
    assert written["passive_area_source"] == "volume_over_passive_length_area"
    assert written["passive_branch_alpha"] == 0.0163545307975375
    assert written["fullgrid_metrics"]["peak_normalized_rmse"] == 0.025100752414070043
    assert any(
        item["doi"] == "10.1016/S0926-9851(02)00168-4"
        and item["model_role"] == "transport-number and geometry-factor chargeability"
        for item in written["literature_basis_structured"]
    )
    assert any(
        item["doi"] == "10.1190/GEO2012-0548.1"
        and item["model_role"] == "SNP/LNP two-time-scale interpretation"
        for item in written["literature_basis_structured"]
    )
    assert "not a hidden length_scale" in written["interpretation"]


def test_writes_best_edl_membrane_model_summary_with_edl_provenance(tmp_path):
    module = load_module()
    component_csv = tmp_path / "dual_length_membrane_volume_area_best_edl_candidate_component_spectrum.csv"
    pd.DataFrame(
        {
            "frequency_hz": [46415.8883361278],
            "membrane_geometry_mode": [
                "diagnostic_two_length_active_passive_volume_area_transport_number_edl_limited"
            ],
            "passive_area_source": ["volume_over_passive_length_area"],
            "passive_length_source": ["pore1_plus_pore2"],
            "passive_weight_source": ["volume_over_passive_length"],
            "passive_zdc_source": ["volume_over_passive_length_area"],
            "passive_branch_alpha": [1.0],
            "passive_branch_alpha_source": [
                "titov_geometry_factor_per_throat_edl_limited_transport_number_difference"
            ],
            "passive_transport_number_edl_limited_fraction": [0.003425],
            "edl_debye_length_m": [5.2156291980139835e-09],
            "edl_thickness_multiplier": [38.346284294166544],
            "edl_selectivity": [1.0],
            "maximum_transport_number_difference": [0.5],
            "passive_branch_alpha_median": [0.0163545307975375],
            "edl_selection_rule": ["smallest_effective_thickness_within_rmse_tolerance"],
            "edl_selection_rmse_tolerance": [0.0005],
            "edl_selected_peak_normalized_rmse": [0.0253345105946186],
        }
    ).to_csv(component_csv, index=False)
    fullgrid_summary = tmp_path / "best_edl_fullgrid_summary.json"
    fullgrid_summary.write_text(
        json.dumps(
            {
                "peak_normalized_rmse": 0.02610046129914571,
                "candidate_peak_frequency_hz": 46415.8883361278,
                "ratio_at_paper_peak": 0.9879461152547286,
                "all_solver_info_zero": True,
                "max_relative_residual_norm": 9.767548060581188e-06,
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "provenance" / "diagnostic_membrane_model_summary.json"

    summary = module.write_diagnostic_membrane_model_summary(
        component_csv=component_csv,
        fullgrid_summary_json=fullgrid_summary,
        out=out,
    )

    assert summary["membrane_geometry_mode"].endswith("transport_number_edl_limited")
    assert summary["passive_branch_alpha_source"] == (
        "titov_geometry_factor_per_throat_edl_limited_transport_number_difference"
    )
    assert summary["edl_debye_length_m"] == 5.2156291980139835e-09
    assert summary["edl_thickness_multiplier"] == 38.346284294166544
    assert summary["maximum_transport_number_difference"] == 0.5
    assert summary["passive_transport_number_edl_limited_fraction"] == 0.003425
    assert summary["edl_selection_rule"] == "smallest_effective_thickness_within_rmse_tolerance"
    assert summary["fullgrid_metrics"]["peak_normalized_rmse"] == 0.02610046129914571
    assert summary["fullgrid_metrics"]["all_solver_info_zero"] is True
    assert "EDL-limited" in summary["interpretation"]


def test_manifest_readme_mentions_diagnostic_membrane_model_when_recorded(tmp_path):
    module = load_module()
    records = {
        "diagnostic_membrane_model": {
            "status": "diagnostic_not_default_niu2020_parameter",
            "membrane_geometry_mode": "diagnostic_two_length_active_passive_volume_area_transport_number_edl_limited",
            "passive_area_source": "volume_over_passive_length_area",
            "passive_branch_alpha": 1.0,
            "edl_debye_length_m": 5.2156291980139835e-09,
            "edl_thickness_multiplier": 38.346284294166544,
            "passive_transport_number_edl_limited_fraction": 0.003425,
        },
        "best_edl_membrane_verification": {
            "report_json": "provenance/verify_niu2020_best_edl_membrane_reproduction_report.json",
            "passed": True,
            "criteria": {
                "max_peak_normalized_rmse": 0.03,
                "min_ratio_at_paper_peak": 0.95,
                "max_ratio_at_paper_peak": 1.05,
            },
        },
        "best_edl_membrane_parameter_config": {
            "config_py": "configs/niu2020_best_edl_membrane_parameters_zh.py",
            "membrane_geometry_mode": "diagnostic_two_length_active_passive_volume_area_transport_number_edl_limited",
        },
        "best_edl_membrane_model_card": "provenance/niu2020_best_edl_membrane_model_card.md",
        "best_edl_membrane_package_verification": {
            "report_json": "provenance/verify_niu2020_best_edl_membrane_package_report.json",
            "passed": True,
            "scope": (
                "Package-level audit for the diagnostic best-EDL membrane branch, "
                "not a full solution of the total measured SIP spectrum."
            ),
        },
    }

    module.write_manifest(tmp_path, records)

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "diagnostic_two_length_active_passive_volume_area_transport_number_edl_limited" in readme
    assert "volume_over_passive_length_area" in readme
    assert "EDL-limited" in readme
    assert "Debye length" in readme
    assert "Best-EDL membrane verification" in readme
    assert "verify_niu2020_best_edl_membrane_reproduction_report.json" in readme
    assert "peak_normalized_rmse <= 0.03" in readme
    assert "niu2020_best_edl_membrane_parameters_zh.py" in readme
    assert "Best-EDL membrane model card" in readme
    assert "niu2020_best_edl_membrane_model_card.md" in readme
    assert "Best-EDL package verification" in readme
    assert "verify_niu2020_best_edl_membrane_package_report.json" in readme
    assert "not a full solution of the total measured SIP spectrum" in readme
    assert "不是 Niu 2020 已公开给出的默认参数" in readme


def test_manifest_readme_mentions_missing_formal_sweep_plan(tmp_path):
    module = load_module()
    records = {
        "formal_sweep_reproduction_plan": {
            "status": "formal_sweep_results_missing",
            "plan_json": "provenance/formal_fullgrid_sweep_reproduction_plan.json",
            "readme_md": "provenance/formal_fullgrid_sweep_reproduction_plan.md",
        }
    }

    module.write_manifest(tmp_path, records)

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "Formal full-grid sweep status" in readme
    assert "formal_sweep_results_missing" in readme
    assert "formal_fullgrid_sweep_reproduction_plan.md" in readme


def test_writes_best_edl_membrane_parameter_config_with_chinese_provenance(tmp_path):
    module = load_module()
    diagnostic_summary = {
        "status": "diagnostic_not_default_niu2020_parameter",
        "membrane_geometry_mode": "diagnostic_two_length_active_passive_volume_area_transport_number_edl_limited",
        "passive_branch_alpha_source": "titov_geometry_factor_per_throat_edl_limited_transport_number_difference",
        "passive_transport_number_edl_limited_fraction": 0.0034246575342465,
        "edl_debye_length_m": 5.2156291980139835e-09,
        "edl_thickness_multiplier": 38.346284294166544,
        "edl_selectivity": 1.0,
        "maximum_transport_number_difference": 0.5,
        "edl_selection_rule": "smallest_effective_thickness_within_rmse_tolerance",
        "edl_selection_rmse_tolerance": 0.0005,
        "edl_selected_peak_normalized_rmse": 0.0253345105946186,
        "component_spectrum_csv": "source_data/dual_length_membrane_diagnostic/component.csv",
        "fullgrid_summary_json": "source_data/membrane_summary.json",
        "literature_basis_structured": [
            {"doi": "10.1190/1.1438659", "citation_key": "Marshall_Madden_1959"},
            {"doi": "10.1016/S0926-9851(02)00168-4", "citation_key": "Titov_2002"},
            {"doi": "10.1190/GEO2012-0548.1", "citation_key": "Buecker_Hoerdt_2013b"},
        ],
    }
    verification = {
        "report_json": "provenance/verify_niu2020_best_edl_membrane_reproduction_report.json",
        "passed": True,
        "criteria": {
            "max_peak_normalized_rmse": 0.03,
            "min_ratio_at_paper_peak": 0.95,
            "max_ratio_at_paper_peak": 1.05,
        },
    }
    out = tmp_path / "configs" / "niu2020_best_edl_membrane_parameters_zh.py"

    config = module.write_best_edl_membrane_parameter_config(
        out,
        diagnostic_summary=diagnostic_summary,
        verification=verification,
    )

    text = out.read_text(encoding="utf-8")
    assert config["config_py"] == str(out)
    assert "Niu 2020 Berea best-EDL 膜极化诊断参数" in text
    assert "diagnostic_two_length_active_passive_volume_area_transport_number_edl_limited" in text
    assert "edl_debye_length_m" in text
    assert "5.2156291980139835e-09" in text
    assert "edl_thickness_multiplier" in text
    assert "38.346284294166544" in text
    assert "maximum_transport_number_difference" in text
    assert "0.5" in text
    assert "smallest_effective_thickness_within_rmse_tolerance" in text
    assert "不是 length_scale" in text
    assert "不是 zdc_scale" in text
    assert "10.1016/S0926-9851(02)00168-4" in text
    assert "10.1190/GEO2012-0548.1" in text
    assert "verify_niu2020_best_edl_membrane_reproduction_report.json" in text
    assert "NIU2020_BEST_EDL_MEMBRANE_CONFIG" in text


def test_writes_missing_sweep_reproduction_plan(tmp_path):
    module = load_module()
    missing = {
        "all": tmp_path / "simulation_sweeps" / "all" / "sweep_results.csv",
        "pore": tmp_path / "simulation_sweeps" / "pore" / "sweep_results.csv",
    }

    summary = module.write_missing_sweep_reproduction_plan(tmp_path, missing)

    plan = Path(summary["plan_json"])
    readme = Path(summary["readme_md"])
    assert plan.exists()
    assert readme.exists()
    text = readme.read_text(encoding="utf-8")
    assert "sweep_results.csv 缺失" in text
    assert "--frequency-match-mode exact" in text
    assert "--gauge-mode auto" in text
    assert "--dtype complex128" in text
    assert "--fft-reference pore" in text
    assert "--rtol 1e-5" in text
    assert "summarize_ac3d_directional_sweeps.py" in text
    assert "verify_ac3d_precision_checkpoints.py" in text
