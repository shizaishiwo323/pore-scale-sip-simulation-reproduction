from __future__ import annotations

import importlib.util
from pathlib import Path

import json
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "sip_simulation" / "plot_niu2020_conductivity_mechanism_comparison.py"


def load_module():
    spec = importlib.util.spec_from_file_location("plot_niu2020_conductivity_mechanism_comparison", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collect_source_data_separates_experiment_and_local_sweeps():
    module = load_module()
    real_experiment = pd.DataFrame({"frequency_hz": [1.0], "real_conductivity_s_m": [2.5e-3]})
    imag_experiment = pd.DataFrame({"frequency_hz": [1.0], "imaginary_conductivity_s_m": [2.0e-5]})
    components = pd.DataFrame(
        {
            "frequency_hz": [1.0, 1.0],
            "mechanism": ["all", "pore"],
            "effective_sigma_real_s_m": [2.7e-3, 2.6e-3],
            "effective_sigma_imag_s_m": [3.0e-5, 1.0e-5],
            "source_csv": ["local_all.csv", "local_pore.csv"],
        }
    )

    source = module.collect_source_data(real_experiment, imag_experiment, components)

    assert set(source["dataset"]) == {"experiment_real", "experiment_imag", "simulation_all", "simulation_pore"}
    sim = source[source["dataset"].str.startswith("simulation_", na=False)]
    assert set(sim["source_csv"]) == {"local_all.csv", "local_pore.csv"}
    assert source["paper_simulation_columns_used"].dropna().unique().tolist() == [False]


def test_collect_source_data_appends_diagnostic_membrane_candidate_without_relabeling_default():
    module = load_module()
    real_experiment = pd.DataFrame({"frequency_hz": [1.0], "real_conductivity_s_m": [2.5e-3]})
    imag_experiment = pd.DataFrame({"frequency_hz": [1.0], "imaginary_conductivity_s_m": [2.0e-5]})
    components = pd.DataFrame(
        {
            "frequency_hz": [1.0],
            "mechanism": ["membrane"],
            "effective_sigma_real_s_m": [2.7e-3],
            "effective_sigma_imag_s_m": [3.0e-5],
            "source_csv": ["default_membrane.csv"],
        }
    )
    diagnostic = pd.DataFrame(
        {
            "frequency_hz": [1.0],
            "effective_sigma_real_s_m": [2.8e-3],
            "effective_sigma_imag_s_m": [3.4e-5],
            "source_csv": ["diagnostic_volume_area.csv"],
            "diagnostic_status": ["diagnostic_not_default_niu2020_parameter"],
            "membrane_geometry_mode": ["diagnostic_two_length_active_passive_volume_area"],
            "passive_area_source": ["volume_over_passive_length_area"],
            "passive_branch_alpha": [0.0163545307975375],
        }
    )

    source = module.collect_source_data(real_experiment, imag_experiment, components, diagnostic_membrane=diagnostic)

    assert "simulation_membrane" in set(source["dataset"])
    assert "diagnostic_membrane_volume_area" in set(source["dataset"])
    row = source.loc[source["dataset"] == "diagnostic_membrane_volume_area"].iloc[0]
    assert row["diagnostic_status"] == "diagnostic_not_default_niu2020_parameter"
    assert row["membrane_geometry_mode"] == "diagnostic_two_length_active_passive_volume_area"
    assert row["passive_area_source"] == "volume_over_passive_length_area"
    assert row["paper_simulation_columns_used"] == False


def test_collect_source_data_derives_diagnostic_all_by_replacing_membrane_branch():
    module = load_module()
    real_experiment = pd.DataFrame({"frequency_hz": [1.0], "real_conductivity_s_m": [2.5e-3]})
    imag_experiment = pd.DataFrame({"frequency_hz": [1.0], "imaginary_conductivity_s_m": [2.0e-5]})
    components = pd.DataFrame(
        {
            "frequency_hz": [1.0, 1.0],
            "mechanism": ["all", "membrane"],
            "effective_sigma_real_s_m": [10.0, 3.0],
            "effective_sigma_imag_s_m": [8.0, 2.0],
            "source_csv": ["all.csv", "membrane.csv"],
        }
    )
    diagnostic = pd.DataFrame(
        {
            "frequency_hz": [1.0],
            "effective_sigma_real_s_m": [4.0],
            "effective_sigma_imag_s_m": [5.0],
            "source_csv": ["diagnostic_volume_area.csv"],
            "diagnostic_status": ["diagnostic_not_default_niu2020_parameter"],
            "membrane_geometry_mode": ["diagnostic_two_length_active_passive_volume_area"],
            "passive_area_source": ["volume_over_passive_length_area"],
            "passive_branch_alpha": [0.0163545307975375],
        }
    )

    source = module.collect_source_data(real_experiment, imag_experiment, components, diagnostic_membrane=diagnostic)

    row = source.loc[source["dataset"] == "diagnostic_all_volume_area"].iloc[0]
    assert row["mechanism"] == "all"
    assert row["real_conductivity_s_m"] == 11.0
    assert row["imaginary_conductivity_s_m"] == 11.0
    assert row["diagnostic_status"] == "diagnostic_not_default_niu2020_parameter"
    assert row["membrane_geometry_mode"] == "diagnostic_two_length_active_passive_volume_area"


def test_collect_source_data_preserves_edl_transport_provenance_on_diagnostic_rows():
    module = load_module()
    real_experiment = pd.DataFrame({"frequency_hz": [1.0], "real_conductivity_s_m": [2.5e-3]})
    imag_experiment = pd.DataFrame({"frequency_hz": [1.0], "imaginary_conductivity_s_m": [2.0e-5]})
    components = pd.DataFrame(
        {
            "frequency_hz": [1.0, 1.0],
            "mechanism": ["all", "membrane"],
            "effective_sigma_real_s_m": [10.0, 3.0],
            "effective_sigma_imag_s_m": [8.0, 2.0],
            "source_csv": ["all.csv", "membrane.csv"],
        }
    )
    diagnostic = pd.DataFrame(
        {
            "frequency_hz": [1.0],
            "effective_sigma_real_s_m": [4.0],
            "effective_sigma_imag_s_m": [5.0],
            "source_csv": ["diagnostic_volume_area.csv"],
            "diagnostic_status": ["diagnostic_not_default_niu2020_parameter"],
            "membrane_geometry_mode": [
                "diagnostic_two_length_active_passive_volume_area_transport_number_edl_limited"
            ],
            "passive_area_source": ["volume_over_passive_length_area"],
            "passive_branch_alpha": [1.0],
            "edl_debye_length_m": [5.216e-9],
            "edl_thickness_multiplier": [38.35],
            "edl_selectivity": [1.0],
            "maximum_transport_number_difference": [0.5],
            "passive_transport_number_edl_limited_fraction": [0.0034],
            "edl_selection_rule": ["smallest_effective_thickness_within_rmse_tolerance"],
            "edl_selected_peak_normalized_rmse": [0.02534],
        }
    )

    source = module.collect_source_data(real_experiment, imag_experiment, components, diagnostic_membrane=diagnostic)

    for dataset in ["diagnostic_membrane_volume_area", "diagnostic_all_volume_area"]:
        row = source.loc[source["dataset"] == dataset].iloc[0]
        assert row["edl_debye_length_m"] == 5.216e-9
        assert row["edl_thickness_multiplier"] == 38.35
        assert row["edl_selectivity"] == 1.0
        assert row["maximum_transport_number_difference"] == 0.5
        assert row["passive_transport_number_edl_limited_fraction"] == 0.0034
        assert row["edl_selection_rule"] == "smallest_effective_thickness_within_rmse_tolerance"
        assert row["edl_selected_peak_normalized_rmse"] == 0.02534


def test_collect_source_data_prefers_true_diagnostic_all_fullgrid_over_algebraic_replacement():
    module = load_module()
    real_experiment = pd.DataFrame({"frequency_hz": [1.0], "real_conductivity_s_m": [2.5e-3]})
    imag_experiment = pd.DataFrame({"frequency_hz": [1.0], "imaginary_conductivity_s_m": [2.0e-5]})
    components = pd.DataFrame(
        {
            "frequency_hz": [1.0, 1.0],
            "mechanism": ["all", "membrane"],
            "effective_sigma_real_s_m": [10.0, 3.0],
            "effective_sigma_imag_s_m": [8.0, 2.0],
            "source_csv": ["all.csv", "membrane.csv"],
        }
    )
    diagnostic_membrane = pd.DataFrame(
        {
            "frequency_hz": [1.0],
            "effective_sigma_real_s_m": [4.0],
            "effective_sigma_imag_s_m": [5.0],
            "source_csv": ["diagnostic_membrane.csv"],
            "diagnostic_status": ["diagnostic_not_default_niu2020_parameter"],
            "membrane_geometry_mode": [
                "diagnostic_two_length_active_passive_volume_area_transport_number_edl_limited"
            ],
            "passive_area_source": ["volume_over_passive_length_area"],
            "passive_branch_alpha": [1.0],
            "edl_debye_length_m": [5.216e-9],
        }
    )
    diagnostic_all_fullgrid = pd.DataFrame(
        {
            "frequency_hz": [1.0],
            "effective_sigma_real_s_m": [40.0],
            "effective_sigma_imag_s_m": [42.0],
            "source_csv": ["diagnostic_all_fullgrid.csv"],
            "diagnostic_status": ["diagnostic_not_default_niu2020_parameter"],
            "membrane_geometry_mode": [
                "diagnostic_two_length_active_passive_volume_area_transport_number_edl_limited"
            ],
            "passive_area_source": ["volume_over_passive_length_area"],
            "passive_branch_alpha": [1.0],
            "edl_debye_length_m": [5.216e-9],
        }
    )

    source = module.collect_source_data(
        real_experiment,
        imag_experiment,
        components,
        diagnostic_membrane=diagnostic_membrane,
        diagnostic_all_fullgrid=diagnostic_all_fullgrid,
    )

    row = source.loc[source["dataset"] == "diagnostic_all_volume_area"].iloc[0]
    assert row["real_conductivity_s_m"] == 40.0
    assert row["imaginary_conductivity_s_m"] == 42.0
    assert row["source_csv"] == "diagnostic_all_fullgrid.csv"
    assert row["edl_debye_length_m"] == 5.216e-9


def test_load_diagnostic_membrane_candidate_combines_fullgrid_with_model_summary(tmp_path):
    module = load_module()
    fullgrid = tmp_path / "diagnostic_sweep.csv"
    pd.DataFrame(
        {
            "frequency_hz": [1.0, 10.0],
            "effective_sigma_real_s_m": [2.8e-3, 2.9e-3],
            "effective_sigma_imag_s_m": [3.4e-5, 4.4e-5],
        }
    ).to_csv(fullgrid, index=False)
    summary = tmp_path / "diagnostic_membrane_model_summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "diagnostic_not_default_niu2020_parameter",
                "membrane_geometry_mode": "diagnostic_two_length_active_passive_volume_area",
                "passive_area_source": "volume_over_passive_length_area",
                "passive_branch_alpha": 0.0163545307975375,
            }
        ),
        encoding="utf-8",
    )

    diagnostic = module.load_diagnostic_membrane_candidate(fullgrid, summary)

    assert diagnostic["source_csv"].unique().tolist() == [str(fullgrid)]
    assert diagnostic["diagnostic_status"].unique().tolist() == ["diagnostic_not_default_niu2020_parameter"]
    assert diagnostic["membrane_geometry_mode"].unique().tolist() == ["diagnostic_two_length_active_passive_volume_area"]
    assert diagnostic["passive_area_source"].unique().tolist() == ["volume_over_passive_length_area"]
    assert diagnostic["passive_branch_alpha"].unique().tolist() == [0.0163545307975375]


def test_load_diagnostic_membrane_candidate_preserves_edl_transport_provenance(tmp_path):
    module = load_module()
    fullgrid = tmp_path / "diagnostic_sweep.csv"
    pd.DataFrame(
        {
            "frequency_hz": [1.0],
            "effective_sigma_real_s_m": [2.8e-3],
            "effective_sigma_imag_s_m": [3.4e-5],
        }
    ).to_csv(fullgrid, index=False)
    summary = tmp_path / "diagnostic_membrane_model_summary.json"
    summary.write_text(
        json.dumps(
            {
                "status": "diagnostic_not_default_niu2020_parameter",
                "membrane_geometry_mode": "diagnostic_two_length_active_passive_volume_area_transport_number_edl_limited",
                "passive_area_source": "volume_over_passive_length_area",
                "passive_branch_alpha": 1.0,
                "edl_debye_length_m": 5.216e-9,
                "edl_thickness_multiplier": 38.35,
                "edl_selectivity": 1.0,
                "maximum_transport_number_difference": 0.5,
                "passive_transport_number_edl_limited_fraction": 0.0034,
                "edl_selection_rule": "smallest_effective_thickness_within_rmse_tolerance",
                "edl_selected_peak_normalized_rmse": 0.02534,
            }
        ),
        encoding="utf-8",
    )

    diagnostic = module.load_diagnostic_membrane_candidate(fullgrid, summary)

    assert diagnostic["edl_debye_length_m"].unique().tolist() == [5.216e-9]
    assert diagnostic["edl_thickness_multiplier"].unique().tolist() == [38.35]
    assert diagnostic["edl_selectivity"].unique().tolist() == [1.0]
    assert diagnostic["maximum_transport_number_difference"].unique().tolist() == [0.5]
    assert diagnostic["passive_transport_number_edl_limited_fraction"].unique().tolist() == [0.0034]
    assert diagnostic["edl_selection_rule"].unique().tolist() == [
        "smallest_effective_thickness_within_rmse_tolerance"
    ]
    assert diagnostic["edl_selected_peak_normalized_rmse"].unique().tolist() == [0.02534]


def test_load_diagnostic_membrane_candidate_accepts_component_metadata_prefixes(tmp_path):
    module = load_module()
    fullgrid = tmp_path / "best_edl_membrane_sweep.csv"
    pd.DataFrame(
        {
            "frequency_hz": [1.0],
            "effective_sigma_real_s_m": [2.8e-3],
            "effective_sigma_imag_s_m": [3.4e-5],
        }
    ).to_csv(fullgrid, index=False)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "diagnostic_membrane_geometry_mode": (
                    "diagnostic_two_length_active_passive_volume_area_transport_number_edl_limited"
                ),
                "diagnostic_membrane_passive_area_source": "volume_over_passive_length_area",
                "diagnostic_membrane_passive_branch_alpha": 1.0,
                "diagnostic_membrane_edl_debye_length_m": 5.216e-9,
                "diagnostic_membrane_edl_thickness_multiplier": 38.35,
                "diagnostic_membrane_edl_selectivity": 1.0,
                "diagnostic_membrane_maximum_transport_number_difference": 0.5,
                "diagnostic_membrane_passive_transport_number_edl_limited_fraction": 0.0034,
                "diagnostic_membrane_edl_selection_rule": "smallest_effective_thickness_within_rmse_tolerance",
                "diagnostic_membrane_edl_selected_peak_normalized_rmse": 0.02534,
            }
        ),
        encoding="utf-8",
    )

    diagnostic = module.load_diagnostic_membrane_candidate(fullgrid, metadata)

    assert diagnostic["diagnostic_status"].unique().tolist() == ["diagnostic_not_default_niu2020_parameter"]
    assert diagnostic["membrane_geometry_mode"].unique().tolist() == [
        "diagnostic_two_length_active_passive_volume_area_transport_number_edl_limited"
    ]
    assert diagnostic["passive_area_source"].unique().tolist() == ["volume_over_passive_length_area"]
    assert diagnostic["passive_branch_alpha"].unique().tolist() == [1.0]
    assert diagnostic["edl_debye_length_m"].unique().tolist() == [5.216e-9]
    assert diagnostic["edl_thickness_multiplier"].unique().tolist() == [38.35]
    assert diagnostic["edl_selection_rule"].unique().tolist() == [
        "smallest_effective_thickness_within_rmse_tolerance"
    ]


def test_summarize_source_data_reports_diagnostic_membrane_match_metrics():
    module = load_module()
    source = pd.DataFrame(
        {
            "dataset": [
                "experiment_imag",
                "experiment_imag",
                "diagnostic_membrane_volume_area",
                "diagnostic_membrane_volume_area",
            ],
            "frequency_hz": [1.0, 10.0, 1.0, 10.0],
            "imaginary_conductivity_s_m": [2.0e-5, 4.0e-5, 1.8e-5, 4.4e-5],
            "paper_simulation_columns_used": [False, False, False, False],
        }
    )

    summary = module.summarize_source_data(source)

    metrics = summary["diagnostic_membrane_volume_area_vs_experiment_imag"]
    assert summary["paper_simulation_columns_used_any"] is False
    assert metrics["common_frequency_count"] == 2
    assert metrics["paper_peak_frequency_hz"] == 10.0
    assert metrics["candidate_peak_frequency_hz"] == 10.0
    assert np.isclose(metrics["ratio_at_paper_peak"], 1.1)
    assert np.isclose(metrics["peak_normalized_rmse"], np.sqrt((0.05**2 + 0.1**2) / 2.0))


def test_summarize_source_data_reports_diagnostic_all_match_metrics():
    module = load_module()
    source = pd.DataFrame(
        {
            "dataset": [
                "experiment_imag",
                "experiment_imag",
                "diagnostic_all_volume_area",
                "diagnostic_all_volume_area",
            ],
            "frequency_hz": [1.0, 10.0, 1.0, 10.0],
            "imaginary_conductivity_s_m": [2.0e-5, 4.0e-5, 2.0e-5, 3.6e-5],
            "paper_simulation_columns_used": [False, False, False, False],
        }
    )

    summary = module.summarize_source_data(source)

    metrics = summary["diagnostic_all_volume_area_vs_experiment_imag"]
    assert metrics["common_frequency_count"] == 2
    assert metrics["candidate_peak_frequency_hz"] == 10.0
    assert np.isclose(metrics["ratio_at_paper_peak"], 0.9)


def test_summarize_source_data_reports_simulation_all_match_metrics():
    module = load_module()
    source = pd.DataFrame(
        {
            "dataset": ["experiment_imag", "experiment_imag", "simulation_all", "simulation_all"],
            "frequency_hz": [1.0, 10.0, 1.0, 10.0],
            "imaginary_conductivity_s_m": [2.0e-5, 4.0e-5, 1.8e-5, 4.0e-5],
            "paper_simulation_columns_used": [False, False, False, False],
        }
    )

    summary = module.summarize_source_data(source)

    metrics = summary["simulation_all_vs_experiment_imag"]
    assert metrics["common_frequency_count"] == 2
    assert metrics["candidate_peak_frequency_hz"] == 10.0
    assert np.isclose(metrics["ratio_at_paper_peak"], 1.0)


def test_frequency_axis_extends_to_1e9_hz():
    module = load_module()

    assert module.FREQUENCY_X_LIMITS == (1.0e-4, 1.0e9)
    assert module.SIGMA_IMAG_Y_LIMITS == (1.0e-7, 1.0e1)
    assert module.DEFAULT_RESULT_DIR.name == "niu2020_berea_reproduction_20260617_original_pnextract_defaults"


def test_default_sweep_paths_are_inside_project_results():
    module = load_module()

    for mechanism, path in module.DEFAULT_SWEEPS.items():
        assert mechanism in {"all", "pore", "membrane", "interfacial"}
        assert module.PROJECT_ROOT in path.parents
        assert "outputs" not in path.parts
        assert path.parts[path.parts.index("results") + 1] == module.DEFAULT_RESULT_DIR.name
        assert "simulation_sweeps" in path.parts
        assert path.parts[path.parts.index("simulation_sweeps") + 1].startswith("niu2020_berea_full350_")


def test_real_experiment_reader_includes_high_frequency_points():
    module = load_module()

    real = module.read_real_conductivity_experiment(PROJECT_ROOT / "data" / "Niu 2020data")

    assert len(real) == 85
    assert real["frequency_hz"].max() > 8.0e8
    assert real["real_conductivity_s_m"].max() > 4.0e-2


def test_make_figure_writes_png_svg_pdf(tmp_path):
    module = load_module()
    real_experiment = pd.DataFrame(
        {"frequency_hz": [1.0e-3, 1.0, 1.0e3], "real_conductivity_s_m": [2.5e-3, 2.6e-3, 2.8e-3]}
    )
    imag_experiment = pd.DataFrame(
        {"frequency_hz": [1.0e-3, 1.0, 1.0e3], "imaginary_conductivity_s_m": [2.0e-6, 5.0e-6, 5.0e-5]}
    )
    rows = []
    for mechanism, scale in [("all", 1.0), ("pore", 0.25), ("membrane", 0.75), ("interfacial", 0.1)]:
        for freq in real_experiment["frequency_hz"]:
            rows.append(
                {
                    "mechanism": mechanism,
                    "frequency_hz": freq,
                    "effective_sigma_real_s_m": 2.6e-3,
                    "effective_sigma_imag_s_m": 4.0e-5 * scale,
                }
            )
    components = pd.DataFrame(rows)
    output_base = tmp_path / "niu2020_conductivity"

    module.make_figure(real_experiment, imag_experiment, components, output_base, title="Berea")

    assert output_base.with_suffix(".png").stat().st_size > 0
    assert output_base.with_suffix(".svg").stat().st_size > 0
    assert output_base.with_suffix(".pdf").stat().st_size > 0


def test_make_figure_can_overlay_diagnostic_membrane_candidate(tmp_path):
    module = load_module()
    real_experiment = pd.DataFrame(
        {"frequency_hz": [1.0e-3, 1.0, 1.0e3], "real_conductivity_s_m": [2.5e-3, 2.6e-3, 2.8e-3]}
    )
    imag_experiment = pd.DataFrame(
        {"frequency_hz": [1.0e-3, 1.0, 1.0e3], "imaginary_conductivity_s_m": [2.0e-6, 5.0e-6, 5.0e-5]}
    )
    rows = []
    for mechanism, scale in [("all", 1.0), ("pore", 0.25), ("membrane", 0.75), ("interfacial", 0.1)]:
        for freq in real_experiment["frequency_hz"]:
            rows.append(
                {
                    "mechanism": mechanism,
                    "frequency_hz": freq,
                    "effective_sigma_real_s_m": 2.6e-3,
                    "effective_sigma_imag_s_m": 4.0e-5 * scale,
                }
            )
    components = pd.DataFrame(rows)
    diagnostic = pd.DataFrame(
        {
            "dataset": [
                "diagnostic_membrane_volume_area",
                "diagnostic_membrane_volume_area",
                "diagnostic_membrane_volume_area",
            ],
            "frequency_hz": [1.0e-3, 1.0, 1.0e3],
            "effective_sigma_real_s_m": [2.5e-3, 2.6e-3, 2.7e-3],
            "effective_sigma_imag_s_m": [3.0e-6, 6.0e-6, 4.8e-5],
        }
    )
    diagnostic_all = pd.DataFrame(
        {
            "dataset": ["diagnostic_all_volume_area", "diagnostic_all_volume_area", "diagnostic_all_volume_area"],
            "frequency_hz": [1.0e-3, 1.0, 1.0e3],
            "real_conductivity_s_m": [2.6e-3, 2.7e-3, 2.9e-3],
            "imaginary_conductivity_s_m": [4.0e-6, 7.0e-6, 5.2e-5],
        }
    )
    output_base = tmp_path / "niu2020_conductivity_with_diagnostic"

    module.make_figure(
        real_experiment,
        imag_experiment,
        components,
        output_base,
        title="Berea",
        diagnostic_membrane=diagnostic,
        diagnostic_all=diagnostic_all,
    )

    assert output_base.with_suffix(".png").stat().st_size > 0
    assert output_base.with_suffix(".svg").stat().st_size > 0
    assert output_base.with_suffix(".pdf").stat().st_size > 0


def test_write_provenance_documents_internal_sweeps_and_input_limits(tmp_path):
    module = load_module()
    sweep_dir = module.DEFAULT_RESULT_DIR / "simulation_sweeps"
    component_paths = {
        "all": sweep_dir / "niu2020_berea_full350_all_original_pnextract_fft_x" / "sweep_results.csv",
        "pore": sweep_dir / "niu2020_berea_full350_pore_fft_x" / "sweep_results.csv",
        "membrane": sweep_dir / "niu2020_berea_full350_membrane_original_pnextract_fft_x" / "sweep_results.csv",
        "interfacial": sweep_dir / "niu2020_berea_full350_interfacial_precision_merged" / "sweep_results.csv",
    }
    out = tmp_path / "provenance.md"

    module.write_provenance(
        out,
        module.PROJECT_ROOT / "data" / "Niu 2020data",
        component_paths,
        tmp_path / "source.csv",
    )

    text = out.read_text(encoding="utf-8")
    assert "microCT_Berea.raw" in text
    assert "polarization_spectra_from_pnextract.csv" in text
    assert "No post-extraction pore/throat geometry" in text
    retired_length_scale = "membrane_length_" + "scale=0.0446683592150963"
    assert retired_length_scale not in text
    assert "Figure8 paper simulation/component columns are not used" in text
    assert "C:\\Users\\imgw\\Documents\\Codex\\SIP模拟\\outputs" not in text


def test_write_provenance_documents_optional_diagnostic_membrane_candidate(tmp_path):
    module = load_module()
    sweep_dir = module.DEFAULT_RESULT_DIR / "simulation_sweeps"
    component_paths = {
        "all": sweep_dir / "all" / "sweep_results.csv",
        "pore": sweep_dir / "pore" / "sweep_results.csv",
        "membrane": sweep_dir / "membrane" / "sweep_results.csv",
        "interfacial": sweep_dir / "interfacial" / "sweep_results.csv",
    }
    diagnostic = {
        "fullgrid_csv": tmp_path / "diagnostic_sweep.csv",
        "summary_json": tmp_path / "diagnostic_membrane_model_summary.json",
        "status": "diagnostic_not_default_niu2020_parameter",
        "membrane_geometry_mode": "diagnostic_two_length_active_passive_volume_area",
    }
    out = tmp_path / "provenance.md"

    module.write_provenance(
        out,
        module.PROJECT_ROOT / "data" / "Niu 2020data",
        component_paths,
        tmp_path / "source.csv",
        diagnostic_membrane=diagnostic,
    )

    text = out.read_text(encoding="utf-8")
    assert "Diagnostic Membrane Candidate" in text
    assert "diagnostic_two_length_active_passive_volume_area" in text
    assert "diagnostic_not_default_niu2020_parameter" in text
    assert "not a hidden `length_scale` or `zdc_scale`" in text


def test_write_provenance_documents_edl_limited_transport_number_details(tmp_path):
    module = load_module()
    sweep_dir = module.DEFAULT_RESULT_DIR / "simulation_sweeps"
    component_paths = {
        "all": sweep_dir / "all" / "sweep_results.csv",
        "pore": sweep_dir / "pore" / "sweep_results.csv",
        "membrane": sweep_dir / "membrane" / "sweep_results.csv",
        "interfacial": sweep_dir / "interfacial" / "sweep_results.csv",
    }
    diagnostic = {
        "fullgrid_csv": tmp_path / "diagnostic_sweep.csv",
        "summary_json": tmp_path / "diagnostic_membrane_model_summary.json",
        "status": "diagnostic_not_default_niu2020_parameter",
        "membrane_geometry_mode": "diagnostic_two_length_active_passive_volume_area_transport_number_edl_limited",
        "edl_debye_length_m": 5.216e-9,
        "edl_thickness_multiplier": 38.35,
        "edl_selectivity": 1.0,
        "maximum_transport_number_difference": 0.5,
        "passive_transport_number_edl_limited_fraction": 0.0034,
        "edl_selection_rule": "smallest_effective_thickness_within_rmse_tolerance",
        "edl_selected_peak_normalized_rmse": 0.02534,
    }
    out = tmp_path / "provenance.md"

    module.write_provenance(
        out,
        module.PROJECT_ROOT / "data" / "Niu 2020data",
        component_paths,
        tmp_path / "source.csv",
        diagnostic_membrane=diagnostic,
    )

    text = out.read_text(encoding="utf-8")
    assert "EDL-limited transport-number parameters" in text
    assert "edl_debye_length_m" in text
    assert "5.216e-09" in text
    assert "edl_thickness_multiplier" in text
    assert "38.35" in text
    assert "smallest_effective_thickness_within_rmse_tolerance" in text


def test_write_provenance_documents_membrane_verification_report(tmp_path):
    module = load_module()
    sweep_dir = module.DEFAULT_RESULT_DIR / "simulation_sweeps"
    component_paths = {
        "all": sweep_dir / "all" / "sweep_results.csv",
        "pore": sweep_dir / "pore" / "sweep_results.csv",
        "membrane": sweep_dir / "membrane" / "sweep_results.csv",
        "interfacial": sweep_dir / "interfacial" / "sweep_results.csv",
    }
    diagnostic = {
        "fullgrid_csv": tmp_path / "diagnostic_sweep.csv",
        "summary_json": tmp_path / "diagnostic_membrane_model_summary.json",
        "status": "diagnostic_not_default_niu2020_parameter",
        "membrane_geometry_mode": "diagnostic_two_length_active_passive_volume_area_transport_number_edl_limited",
    }
    verification = {
        "report_json": tmp_path / "verify_report.json",
        "passed": True,
        "failed_checks": [],
        "criteria": {
            "max_peak_normalized_rmse": 0.03,
            "min_ratio_at_paper_peak": 0.95,
            "max_ratio_at_paper_peak": 1.05,
            "residual_tolerance": 1.0e-5,
        },
        "checks": {
            "peak_normalized_rmse_within_threshold": {"value": 0.0261004613},
            "ratio_at_paper_peak_within_threshold": {"value": 0.9879461153},
            "sweep_converged": {"max_relative_residual_norm": 9.864e-6},
        },
    }
    out = tmp_path / "provenance.md"

    module.write_provenance(
        out,
        module.PROJECT_ROOT / "data" / "Niu 2020data",
        component_paths,
        tmp_path / "source.csv",
        diagnostic_membrane=diagnostic,
        diagnostic_membrane_verification=verification,
    )

    text = out.read_text(encoding="utf-8")
    assert "Diagnostic Membrane Verification" in text
    assert "verify_report.json" in text
    assert "Passed" in text
    assert "0.0261004613" in text
    assert "0.9879461153" in text
    assert "9.864e-06" in text


def test_main_can_include_optional_diagnostic_membrane_candidate(tmp_path):
    module = load_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    figure7 = pd.DataFrame(np.nan, index=range(3), columns=range(4))
    figure7.iloc[2, 2] = 1.0
    figure7.iloc[2, 3] = 2.5e-3
    figure7.to_excel(data_dir / "Figure7.xlsx", index=False, header=False)
    figure8 = pd.DataFrame(np.nan, index=range(3), columns=range(2))
    figure8.iloc[2, 0] = 1.0
    figure8.iloc[2, 1] = 2.0e-5
    figure8.to_excel(data_dir / "Figure8.xlsx", index=False, header=False)

    sweep_paths = {}
    for mechanism in ["all", "pore", "membrane", "interfacial"]:
        path = tmp_path / f"{mechanism}.csv"
        pd.DataFrame(
            {
                "frequency_hz": [1.0],
                "effective_sigma_real_s_m": [2.7e-3],
                "effective_sigma_imag_s_m": [3.0e-5],
            }
        ).to_csv(path, index=False)
        sweep_paths[mechanism] = path
    diagnostic_fullgrid = tmp_path / "diagnostic_volume_area.csv"
    pd.DataFrame(
        {
            "frequency_hz": [1.0],
            "effective_sigma_real_s_m": [2.8e-3],
            "effective_sigma_imag_s_m": [3.4e-5],
        }
    ).to_csv(diagnostic_fullgrid, index=False)
    diagnostic_summary = tmp_path / "diagnostic_membrane_model_summary.json"
    diagnostic_summary.write_text(
        json.dumps(
            {
                "status": "diagnostic_not_default_niu2020_parameter",
                "membrane_geometry_mode": "diagnostic_two_length_active_passive_volume_area",
                "passive_area_source": "volume_over_passive_length_area",
                "passive_branch_alpha": 0.0163545307975375,
            }
        ),
        encoding="utf-8",
    )
    source_csv = tmp_path / "source_data.csv"
    provenance_md = tmp_path / "provenance.md"
    summary_json = tmp_path / "summary.json"

    module.main(
        [
            "--data-dir",
            str(data_dir),
            "--all-csv",
            str(sweep_paths["all"]),
            "--pore-csv",
            str(sweep_paths["pore"]),
            "--membrane-csv",
            str(sweep_paths["membrane"]),
            "--interfacial-csv",
            str(sweep_paths["interfacial"]),
            "--diagnostic-membrane-fullgrid-csv",
            str(diagnostic_fullgrid),
            "--diagnostic-membrane-summary-json",
            str(diagnostic_summary),
            "--figure-base",
            str(tmp_path / "figure"),
            "--source-data-csv",
            str(source_csv),
            "--provenance-md",
            str(provenance_md),
            "--summary-json",
            str(summary_json),
        ]
    )

    source = pd.read_csv(source_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert "diagnostic_membrane_volume_area" in set(source["dataset"])
    assert summary["diagnostic_membrane_volume_area_vs_experiment_imag"]["common_frequency_count"] == 1
    assert summary["diagnostic_membrane_volume_area_vs_experiment_imag"]["ratio_at_paper_peak"] == 1.7
    assert "diagnostic_two_length_active_passive_volume_area" in provenance_md.read_text(encoding="utf-8")


def test_main_can_include_true_diagnostic_all_fullgrid_candidate(tmp_path):
    module = load_module()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    figure7 = pd.DataFrame(np.nan, index=range(3), columns=range(4))
    figure7.iloc[2, 2] = 1.0
    figure7.iloc[2, 3] = 2.5e-3
    figure7.to_excel(data_dir / "Figure7.xlsx", index=False, header=False)
    figure8 = pd.DataFrame(np.nan, index=range(3), columns=range(2))
    figure8.iloc[2, 0] = 1.0
    figure8.iloc[2, 1] = 2.0e-5
    figure8.to_excel(data_dir / "Figure8.xlsx", index=False, header=False)

    sweep_paths = {}
    for mechanism in ["all", "pore", "membrane", "interfacial"]:
        path = tmp_path / f"{mechanism}.csv"
        pd.DataFrame(
            {
                "frequency_hz": [1.0],
                "effective_sigma_real_s_m": [2.7e-3],
                "effective_sigma_imag_s_m": [3.0e-5],
            }
        ).to_csv(path, index=False)
        sweep_paths[mechanism] = path
    diagnostic_all = tmp_path / "diagnostic_all_fullgrid.csv"
    pd.DataFrame(
        {
            "frequency_hz": [1.0],
            "effective_sigma_real_s_m": [3.2e-3],
            "effective_sigma_imag_s_m": [1.6e-5],
        }
    ).to_csv(diagnostic_all, index=False)
    diagnostic_metadata = tmp_path / "metadata.json"
    diagnostic_metadata.write_text(
        json.dumps(
            {
                "diagnostic_membrane_geometry_mode": (
                    "diagnostic_two_length_active_passive_volume_area_transport_number_edl_limited"
                ),
                "diagnostic_membrane_passive_area_source": "volume_over_passive_length_area",
                "diagnostic_membrane_passive_branch_alpha": 1.0,
                "diagnostic_membrane_edl_debye_length_m": 5.216e-9,
            }
        ),
        encoding="utf-8",
    )
    source_csv = tmp_path / "source_data.csv"
    provenance_md = tmp_path / "provenance.md"
    summary_json = tmp_path / "summary.json"

    module.main(
        [
            "--data-dir",
            str(data_dir),
            "--all-csv",
            str(sweep_paths["all"]),
            "--pore-csv",
            str(sweep_paths["pore"]),
            "--membrane-csv",
            str(sweep_paths["membrane"]),
            "--interfacial-csv",
            str(sweep_paths["interfacial"]),
            "--diagnostic-all-fullgrid-csv",
            str(diagnostic_all),
            "--diagnostic-all-summary-json",
            str(diagnostic_metadata),
            "--figure-base",
            str(tmp_path / "figure"),
            "--source-data-csv",
            str(source_csv),
            "--provenance-md",
            str(provenance_md),
            "--summary-json",
            str(summary_json),
        ]
    )

    source = pd.read_csv(source_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    row = source.loc[source["dataset"] == "diagnostic_all_volume_area"].iloc[0]
    assert row["imaginary_conductivity_s_m"] == 1.6e-5
    assert row["source_csv"] == str(diagnostic_all)
    assert row["edl_debye_length_m"] == 5.216e-9
    assert np.isclose(summary["diagnostic_all_volume_area_vs_experiment_imag"]["ratio_at_paper_peak"], 0.8)
    assert "Diagnostic All Candidate" in provenance_md.read_text(encoding="utf-8")
