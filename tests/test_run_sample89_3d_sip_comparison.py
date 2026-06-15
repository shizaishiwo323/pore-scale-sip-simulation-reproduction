from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "sip_simulation" / "run_sample89_3d_sip_comparison.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_sample89_3d_sip_comparison", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_psip_csv_extracts_complex_response(tmp_path):
    module = load_module()
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "\n".join(
            [
                "OE PSIP Measurement,",
                "***End_Of_Header***,",
                "X_Value,AO_sampling_rate,Frequency[Hz],Magnitude[ratio],Phase_Shift[rad],TimeStamp[Sec],Comment",
                ",100000,10.0,2.0,0.5,1,",
                ",100000,1.0,4.0,-0.25,2,",
            ]
        ),
        encoding="utf-8",
    )

    frame = module.parse_psip_csv(csv_path)

    assert frame["frequency_hz"].tolist() == [1.0, 10.0]
    assert frame["magnitude_ratio"].tolist() == [4.0, 2.0]
    np.testing.assert_allclose(frame["experiment_real_raw"], [4.0 * np.cos(-0.25), 2.0 * np.cos(0.5)])
    np.testing.assert_allclose(frame["experiment_imag_raw"], [4.0 * np.sin(-0.25), 2.0 * np.sin(0.5)])


def test_parse_mhd_returns_zyx_shape_and_raw_path(tmp_path):
    module = load_module()
    raw_path = tmp_path / "volume.raw"
    raw_path.write_bytes(b"\x00\x01")
    mhd_path = tmp_path / "volume.mhd"
    mhd_path.write_text(
        "\n".join(
            [
                "ObjectType = Image",
                "ElementType = MET_UCHAR",
                "DimSize = 4 3 2",
                "ElementSize = 6.8 6.8 6.8",
                "ElementDataFile = volume.raw",
            ]
        ),
        encoding="utf-8",
    )

    info = module.parse_mhd(mhd_path)

    assert info["shape_zyx"] == (2, 3, 4)
    assert info["element_size_um_xyz"] == (6.8, 6.8, 6.8)
    assert info["dtype"] == np.dtype("uint8")
    assert info["raw_path"] == raw_path


def test_scale_experiment_to_simulation_uses_lowest_frequency_real_anchor():
    module = load_module()
    experiment = module.pd.DataFrame(
        {
            "frequency_hz": [1.0, 10.0],
            "experiment_real_raw": [2.0, 4.0],
            "experiment_imag_raw": [-0.5, -1.0],
        }
    )
    simulation = module.pd.DataFrame(
        {
            "frequency_hz": [1.0, 10.0],
            "mechanism": ["all", "all"],
            "effective_sigma_real_s_m": [0.1, 0.2],
        }
    )

    scaled, scale = module.scale_experiment_to_simulation(experiment, simulation)

    assert scale == 0.05
    assert scaled["experiment_real_scaled_s_m"].tolist() == [0.1, 0.2]
    assert scaled["experiment_imag_scaled_s_m"].tolist() == [-0.025, -0.05]


def test_only_converged_solutions_are_used_as_warm_starts():
    module = load_module()

    assert module.can_reuse_solution_as_warm_start(info=0, residual_norm=9.0e-6, rtol=1.0e-5)
    assert not module.can_reuse_solution_as_warm_start(info=1000, residual_norm=9.0e-6, rtol=1.0e-5)
    assert not module.can_reuse_solution_as_warm_start(info=0, residual_norm=1.1e-5, rtol=1.0e-5)


def test_convert_psip_impedance_ratio_to_bulk_conductivity():
    module = load_module()
    experiment = module.pd.DataFrame(
        {
            "frequency_hz": [1.0],
            "magnitude_ratio": [2.0],
            "phase_shift_rad": [0.0],
            "experiment_real_raw": [2.0],
            "experiment_imag_raw": [0.0],
        }
    )

    converted = module.add_bulk_conductivity_from_psip_impedance(
        experiment,
        current_resistor_ohm=10_000.0,
        sample_length_cm=5.0,
        sample_diameter_cm=2.0,
    )

    expected = (1.0 / (10_000.0 * 2.0)) * (0.05 / (np.pi * 0.01**2))
    np.testing.assert_allclose(converted["experiment_sigma_real_s_m"], [expected])
    np.testing.assert_allclose(converted["experiment_sigma_imag_s_m"], [0.0], atol=1e-15)


def test_build_polarization_parameters_applies_sample_overrides():
    module = load_module()

    params = module.build_polarization_parameters(
        water_conductivity_s_m=0.12,
        surface_conductance_s=2.0e-9,
        membrane_polarizability=0.03,
        solid_relative_permittivity=8.5,
        dynamic_pore_size_m=5.0e-6,
    )

    assert params.water_conductivity_s_m == 0.12
    assert params.surface_conductance_s == 2.0e-9
    assert params.membrane_polarizability == 0.03
    assert params.solid_relative_permittivity == 8.5
    assert params.dynamic_pore_size_m == 5.0e-6


def test_score_trend_identifies_internal_imaginary_peak():
    module = load_module()
    simulation = module.pd.DataFrame(
        {
            "mechanism": ["all", "all", "all"],
            "frequency_hz": [1.0, 10.0, 100.0],
            "effective_sigma_real_s_m": [1.0, 1.4, 2.0],
            "effective_sigma_imag_s_m": [0.2, 0.8, 0.3],
        }
    )
    experiment = module.pd.DataFrame(
        {
            "frequency_hz": [1.0, 10.0, 100.0],
            "experiment_sigma_real_s_m": [1.0, 1.5, 2.2],
            "experiment_sigma_imag_s_m": [0.1, 0.9, 0.2],
        }
    )

    score = module.score_sample89_trend(simulation, experiment)

    assert score["real_increases"] == 1
    assert score["imag_has_internal_peak"] == 1
    assert score["imag_peak_hz"] == 10.0
    assert score["trend_pass"] == 1


def test_score_trend_rejects_high_frequency_imaginary_peak():
    module = load_module()
    simulation = module.pd.DataFrame(
        {
            "mechanism": ["all", "all", "all"],
            "frequency_hz": [1.0, 10.0, 100.0],
            "effective_sigma_real_s_m": [1.0, 1.4, 2.0],
            "effective_sigma_imag_s_m": [0.2, 0.4, 0.8],
        }
    )
    experiment = module.pd.DataFrame(
        {
            "frequency_hz": [1.0, 10.0, 100.0],
            "experiment_sigma_real_s_m": [1.0, 1.5, 2.2],
            "experiment_sigma_imag_s_m": [0.1, 0.9, 0.2],
        }
    )

    score = module.score_sample89_trend(simulation, experiment)

    assert score["real_increases"] == 1
    assert score["imag_has_internal_peak"] == 0
    assert score["trend_pass"] == 0
