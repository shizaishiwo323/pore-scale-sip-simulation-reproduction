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
