from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "sip_simulation" / "plot_niu2020_figure7_style_corrected.py"


def load_module():
    spec = importlib.util.spec_from_file_location("plot_niu2020_figure7_style_corrected", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collect_source_data_marks_corrected_scaled_membrane():
    module = load_module()
    experiment = pd.DataFrame(
        {
            "frequency_hz": [1.0],
            "imaginary_conductivity_s_m": [2.0e-5],
            "real_relative_permittivity": [1.0e5],
        }
    )
    components = pd.DataFrame(
        {
            "frequency_hz": [1.0, 1.0, 1.0],
            "mechanism": ["all", "membrane", "interfacial"],
            "effective_sigma_imag_s_m": [3.0e-5, 1.0e-5, 5.0e-7],
            "real_relative_permittivity": [2.0e5, 5.0e4, 32.5],
        }
    )

    source = module.collect_source_data(experiment, components)

    assert set(source["dataset"]) == {
        "experiment",
        "our_corrected_all",
        "our_corrected_membrane",
        "our_corrected_interfacial",
    }
    assert source["correction"].dropna().unique().tolist() == ["membrane_length_zdc_scaled"]
    assert set(source["component_correction"].dropna()) == {
        "includes_scaled_membrane",
        "length_zdc_scaled",
        "precision_merged_low_frequency",
    }


def test_make_figure_writes_all_formats(tmp_path):
    module = load_module()
    experiment = pd.DataFrame(
        {
            "frequency_hz": [1.0e-3, 1.0, 1.0e3],
            "imaginary_conductivity_s_m": [1.0e-5, 2.0e-5, 3.0e-5],
            "real_relative_permittivity": [1.0e8, 1.0e5, 1.0e2],
        }
    )
    rows = []
    for mechanism, scale in [("all", 1.0), ("pore", 0.5), ("membrane", 0.25), ("interfacial", 0.75)]:
        for freq, eps, imag in zip(
            experiment["frequency_hz"],
            experiment["real_relative_permittivity"],
            experiment["imaginary_conductivity_s_m"],
        ):
            rows.append(
                {
                    "mechanism": mechanism,
                    "frequency_hz": freq,
                    "real_relative_permittivity": eps * scale,
                    "effective_sigma_imag_s_m": imag * scale,
                }
            )
    components = pd.DataFrame(rows)
    output_base = tmp_path / "figure7_style"

    module.make_figure(experiment, components, output_base)

    assert output_base.with_suffix(".png").stat().st_size > 0
    assert output_base.with_suffix(".svg").stat().st_size > 0
    assert output_base.with_suffix(".pdf").stat().st_size > 0
