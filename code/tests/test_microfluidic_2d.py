from pathlib import Path

import numpy as np

from pore_scale_electrical.microfluidic_2d import (
    CALCITE_LABEL,
    FARADAY_C_PER_EQ,
    WATER_LABEL,
    MicrofluidicCalibration,
    cec_meq_g_to_c_kg,
    default_frequencies_hz,
    geometry_metrics,
    mechanism_phase_conductivities,
    paper_validation_drivers_from_tables,
    segment_microfluidic_image,
    schwarz_interface_conductivity_delta,
    si03_solution_conductivity_at_times,
    waxman_smits_water_increment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "dissolution_results-Da_40.4424_Pe_4.1640_L_0.1200_square"
    / "interface_images"
    / "timestep_0001.png"
)


def test_segment_microfluidic_image_excludes_white_canvas_and_keeps_calcite():
    segmented = segment_microfluidic_image(IMAGE_PATH, calibration=MicrofluidicCalibration())

    assert segmented.bbox_xyxy == (193, 217, 1344, 646)
    assert segmented.labels.shape == (430, 1152)
    assert segmented.red_pixels == 395956
    assert segmented.yellow_pixels == 99404
    assert segmented.active_pixels == 495360
    assert segmented.unknown_pixels_in_crop == 0
    assert not np.any(segmented.labels == 0)
    assert np.count_nonzero(segmented.labels == WATER_LABEL) == segmented.red_pixels
    assert np.count_nonzero(segmented.labels == CALCITE_LABEL) == segmented.yellow_pixels
    assert np.isclose(segmented.dx_m, 4.0e-3 / 1152)
    assert np.isclose(segmented.dy_m, 1.5e-3 / 430)


def test_geometry_metrics_reports_interface_and_pore_statistics():
    segmented = segment_microfluidic_image(IMAGE_PATH)
    metrics = geometry_metrics(segmented)

    assert metrics.width_px == 1152
    assert metrics.height_px == 430
    assert np.isclose(metrics.calcite_area_fraction, segmented.yellow_pixels / segmented.active_pixels)
    assert metrics.interface_length_m > 0
    assert metrics.pore_radius_count > 0
    assert metrics.skeleton_length_m > 0


def test_mechanism_phase_conductivities_cover_all_switches():
    segmented = segment_microfluidic_image(IMAGE_PATH)
    metrics = geometry_metrics(segmented)
    frequencies = default_frequencies_hz()
    radii = np.array([metrics.pore_radius_median_m])
    lengths = np.array([metrics.throat_length_mean_m])

    for mechanism in ["maxwell", "pore", "membrane", "grain", "interface", "all"]:
        spectra = mechanism_phase_conductivities(
            frequencies,
            mechanism=mechanism,
            metrics=metrics,
            pore_radii_m=radii,
            throat_lengths_m=lengths,
            interface_delta_conductivity_s_m=0.02 + 0.003j,
        )
        assert len(spectra) == len(frequencies)
        assert set(
            [
                "frequency_hz",
                "mechanism",
                "water_sigma_real_s_m",
                "water_sigma_imag_s_m",
                "solid_sigma_real_s_m",
                "solid_sigma_imag_s_m",
            ]
        ).issubset(spectra.columns)


def test_cec_meq_g_to_c_kg_uses_faraday_conversion():
    values = cec_meq_g_to_c_kg(np.array([0.0, 1.0, 0.015]))

    assert np.isclose(values[0], 0.0)
    assert np.isclose(values[1], FARADAY_C_PER_EQ)
    assert np.isclose(values[2], 0.015 * FARADAY_C_PER_EQ)


def test_paper_validation_drivers_interpolate_sigma_w_and_cec():
    si01 = {
        "Time (h)": [0.0, 1.0],
        "Porosity": [0.7, 0.8],
        "Water saturation": [1.0, 0.5],
    }
    si02 = {
        "Time (h)": [0.0, 1.0],
        "CEC (mEq/g)": [0.01, 0.03],
    }
    si03 = {
        "Time (h)": [0.0, 0.5],
        "Water conductivity (S/m)": [0.01, 0.2],
    }

    drivers = paper_validation_drivers_from_tables(
        np.array([0.25, 1.0]),
        si01=si01,
        si02=si02,
        si03=si03,
        sigma_hcl_s_m=0.44,
    )

    assert np.allclose(drivers["sigma_w_s_m"], [0.105, 0.44])
    assert np.allclose(drivers["cec_meq_g"], [0.015, 0.03])
    assert np.all(drivers["qv_c_m3"] > 0.0)


def test_si03_solution_conductivity_driver_interpolates_and_extends():
    si03 = {
        "Time (h)": [0.0, 0.5],
        "Water conductivity (S/m)": [0.01, 0.2],
    }

    drivers = si03_solution_conductivity_at_times(
        np.array([0.25, 1.0]),
        si03=si03,
        sigma_hcl_s_m=0.44,
        end_time_h=1.0,
    )

    assert np.allclose(drivers, [0.105, 0.44])


def test_schwarz_interface_delta_peaks_when_omega_tau_is_one():
    freq = np.array([2.5])
    diffusion = 1.3e-9
    radius = np.sqrt(2.0 * diffusion / (2.0 * np.pi * freq[0]))
    delta = schwarz_interface_conductivity_delta(
        freq,
        interface_density_1_m=np.array([1000.0]),
        sigma_s_s=1.0e-5,
        characteristic_length_m=radius,
        diffusion_coefficient_m2_s=diffusion,
    )

    assert np.isclose(delta[0].real, 5.0e-3, rtol=1e-6)
    assert np.isclose(delta[0].imag, 5.0e-3, rtol=1e-6)


def test_schwarz_interface_delta_uses_local_length_not_calcite_body_radius():
    freq = np.array([2.5])
    diffusion = 1.3e-9
    density = np.array([878.0])
    sigma_s = 1.0e-5

    local = schwarz_interface_conductivity_delta(
        freq,
        interface_density_1_m=density,
        sigma_s_s=sigma_s,
        characteristic_length_m=1.3e-5,
        diffusion_coefficient_m2_s=diffusion,
    )
    body = schwarz_interface_conductivity_delta(
        freq,
        interface_density_1_m=density,
        sigma_s_s=sigma_s,
        characteristic_length_m=6.2e-4,
        diffusion_coefficient_m2_s=diffusion,
    )

    assert abs(local[0].imag) > 100.0 * abs(body[0].imag)


def test_interface_mechanism_adds_waxman_smits_delta_to_water_phase():
    segmented = segment_microfluidic_image(IMAGE_PATH)
    metrics = geometry_metrics(segmented)
    frequencies = np.array([2.5])
    delta = waxman_smits_water_increment(
        phi=np.array([0.75]),
        sw=np.array([1.0]),
        cec_meq_g=np.array([0.01]),
    )[0]

    spectra = mechanism_phase_conductivities(
        frequencies,
        mechanism="interface",
        metrics=metrics,
        pore_radii_m=np.array([]),
        throat_lengths_m=np.array([]),
        water_conductivity_s_m=0.1,
        interface_delta_conductivity_s_m=delta,
    )

    assert np.isclose(spectra.loc[0, "water_sigma_real_s_m"], 0.1 + delta.real)
    assert np.isclose(spectra.loc[0, "water_sigma_imag_s_m"], delta.imag)
