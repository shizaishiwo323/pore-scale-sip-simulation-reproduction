from pathlib import Path

import numpy as np

from pore_scale_electrical.microfluidic_2d import (
    CALCITE_LABEL,
    WATER_LABEL,
    MicrofluidicCalibration,
    default_frequencies_hz,
    geometry_metrics,
    mechanism_phase_conductivities,
    segment_microfluidic_image,
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

    for mechanism in ["maxwell", "pore", "membrane", "grain", "all"]:
        spectra = mechanism_phase_conductivities(
            frequencies,
            mechanism=mechanism,
            metrics=metrics,
            pore_radii_m=radii,
            throat_lengths_m=lengths,
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
