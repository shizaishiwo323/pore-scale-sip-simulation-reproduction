from __future__ import annotations

import importlib.util
from pathlib import Path

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
