"""Image preprocessing and v1 mechanism helpers for 2-D microfluidic SIP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

from pore_scale_electrical.polarization import (
    EPSILON_0,
    PolarizationParameters,
    apparent_water_conductivity,
    membrane_polarization_conductance,
    pore_polarization_conductance,
    upscale_conductance_to_water_conductivity,
)


BACKGROUND_LABEL = np.uint8(0)
WATER_LABEL = np.uint8(1)
CALCITE_LABEL = np.uint8(2)
FARADAY_C_PER_EQ = 96485.33212


@dataclass(frozen=True)
class MicrofluidicCalibration:
    """Physical size of the active red/yellow image domain."""

    width_m: float = 4.0e-3
    height_m: float = 1.5e-3


@dataclass(frozen=True)
class SegmentedMicrofluidicImage:
    """Cropped two-phase image and derived masks."""

    source_path: Path
    labels: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    dx_m: float
    dy_m: float
    red_pixels: int
    yellow_pixels: int
    active_pixels: int
    unknown_pixels_in_crop: int

    @property
    def water_mask(self) -> np.ndarray:
        return self.labels == WATER_LABEL

    @property
    def calcite_mask(self) -> np.ndarray:
        return self.labels == CALCITE_LABEL


@dataclass(frozen=True)
class GeometryMetrics2D:
    """Frame-level geometry metrics from the cropped label image."""

    width_px: int
    height_px: int
    dx_m: float
    dy_m: float
    water_area_m2: float
    calcite_area_m2: float
    active_area_m2: float
    calcite_area_fraction: float
    interface_length_m: float
    calcite_equivalent_radius_m: float
    pore_radius_mean_m: float
    pore_radius_median_m: float
    pore_radius_count: int
    skeleton_length_m: float
    throat_count: int
    throat_length_mean_m: float


def classify_red_yellow_white(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify the simple rendered masks used by the dissolution image series."""

    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("expected an RGB image array")
    red = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 80) & (arr[:, :, 2] < 80)
    yellow = (arr[:, :, 0] > 200) & (arr[:, :, 1] > 200) & (arr[:, :, 2] < 80)
    white = (arr[:, :, 0] > 220) & (arr[:, :, 1] > 220) & (arr[:, :, 2] > 220)
    return red, yellow, white


def active_bbox_from_masks(red: np.ndarray, yellow: np.ndarray) -> tuple[int, int, int, int]:
    """Return inclusive ``(x_min, y_min, x_max, y_max)`` for red or yellow pixels."""

    active = red | yellow
    if not np.any(active):
        raise ValueError("image contains no red/yellow active domain")
    ys, xs = np.where(active)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def segment_microfluidic_image(
    path: str | Path,
    *,
    calibration: MicrofluidicCalibration = MicrofluidicCalibration(),
    bbox_xyxy: tuple[int, int, int, int] | None = None,
    max_unknown_fraction: float = 0.0,
) -> SegmentedMicrofluidicImage:
    """Crop white canvas away and return red/yellow labels.

    The active crop is computed from ``red OR yellow`` unless a reference bbox is
    supplied. This deliberately keeps the yellow calcite body inside the
    simulation domain and excludes only the surrounding white canvas.
    """

    source_path = Path(path)
    image = np.asarray(Image.open(source_path).convert("RGB"))
    red, yellow, white = classify_red_yellow_white(image)
    bbox = active_bbox_from_masks(red, yellow) if bbox_xyxy is None else bbox_xyxy
    x0, y0, x1, y1 = bbox
    red_crop = red[y0 : y1 + 1, x0 : x1 + 1]
    yellow_crop = yellow[y0 : y1 + 1, x0 : x1 + 1]
    white_crop = white[y0 : y1 + 1, x0 : x1 + 1]
    labels = np.full(red_crop.shape, BACKGROUND_LABEL, dtype=np.uint8)
    labels[red_crop] = WATER_LABEL
    labels[yellow_crop] = CALCITE_LABEL

    unknown = ~(red_crop | yellow_crop | white_crop)
    unknown_count = int(np.count_nonzero(unknown))
    active_in_crop = red_crop | yellow_crop
    background_count = int(np.count_nonzero(~active_in_crop))
    unknown_fraction = unknown_count / labels.size
    if unknown_fraction > max_unknown_fraction:
        raise ValueError(f"unknown colored pixels in active crop: {unknown_count} ({unknown_fraction:.3%})")
    if background_count:
        raise ValueError(f"white/background pixels remain inside active crop: {background_count}")

    height_px, width_px = labels.shape
    dx_m = calibration.width_m / width_px
    dy_m = calibration.height_m / height_px
    return SegmentedMicrofluidicImage(
        source_path=source_path,
        labels=labels,
        bbox_xyxy=bbox,
        dx_m=float(dx_m),
        dy_m=float(dy_m),
        red_pixels=int(np.count_nonzero(red_crop)),
        yellow_pixels=int(np.count_nonzero(yellow_crop)),
        active_pixels=int(np.count_nonzero(active_in_crop)),
        unknown_pixels_in_crop=unknown_count,
    )


def calcite_water_interface_length(labels: np.ndarray, dx_m: float, dy_m: float) -> float:
    """Count 4-neighbor calcite-water faces and convert them to physical length."""

    water = labels == WATER_LABEL
    calcite = labels == CALCITE_LABEL
    x_faces = (water[:, :-1] & calcite[:, 1:]) | (calcite[:, :-1] & water[:, 1:])
    y_faces = (water[:-1, :] & calcite[1:, :]) | (calcite[:-1, :] & water[1:, :])
    return float(np.count_nonzero(x_faces) * dy_m + np.count_nonzero(y_faces) * dx_m)


def pore_radius_samples_m(water_mask: np.ndarray, dx_m: float, dy_m: float) -> np.ndarray:
    """Sample local pore radii on the water skeleton."""

    distance = ndimage.distance_transform_edt(water_mask, sampling=(dy_m, dx_m))
    skeleton = skeletonize(water_mask)
    radii = distance[skeleton]
    return radii[radii > 0]


def skeleton_component_lengths_m(water_mask: np.ndarray, dx_m: float, dy_m: float) -> np.ndarray:
    """Approximate connected water-skeleton lengths for membrane v1 statistics."""

    skeleton = skeletonize(water_mask)
    labeled, n_labels = ndimage.label(skeleton)
    lengths: list[float] = []
    pixel_length = 0.5 * (dx_m + dy_m)
    for label in range(1, n_labels + 1):
        count = int(np.count_nonzero(labeled == label))
        if count:
            lengths.append(count * pixel_length)
    return np.asarray(lengths, dtype=float)


def geometry_metrics(segmented: SegmentedMicrofluidicImage) -> GeometryMetrics2D:
    """Compute geometry and simple ball-stick statistics for one frame."""

    labels = segmented.labels
    pixel_area = segmented.dx_m * segmented.dy_m
    water_area = segmented.red_pixels * pixel_area
    calcite_area = segmented.yellow_pixels * pixel_area
    active_area = segmented.active_pixels * pixel_area
    radii = pore_radius_samples_m(segmented.water_mask, segmented.dx_m, segmented.dy_m)
    throat_lengths = skeleton_component_lengths_m(segmented.water_mask, segmented.dx_m, segmented.dy_m)
    skeleton_length = float(throat_lengths.sum()) if throat_lengths.size else 0.0
    return GeometryMetrics2D(
        width_px=int(labels.shape[1]),
        height_px=int(labels.shape[0]),
        dx_m=segmented.dx_m,
        dy_m=segmented.dy_m,
        water_area_m2=float(water_area),
        calcite_area_m2=float(calcite_area),
        active_area_m2=float(active_area),
        calcite_area_fraction=float(calcite_area / active_area),
        interface_length_m=calcite_water_interface_length(labels, segmented.dx_m, segmented.dy_m),
        calcite_equivalent_radius_m=float(np.sqrt(calcite_area / np.pi)) if calcite_area > 0 else 0.0,
        pore_radius_mean_m=float(np.mean(radii)) if radii.size else 0.0,
        pore_radius_median_m=float(np.median(radii)) if radii.size else 0.0,
        pore_radius_count=int(radii.size),
        skeleton_length_m=skeleton_length,
        throat_count=int(throat_lengths.size),
        throat_length_mean_m=float(np.mean(throat_lengths)) if throat_lengths.size else 0.0,
    )


def metrics_to_dict(metrics: GeometryMetrics2D) -> dict[str, float | int]:
    return {name: getattr(metrics, name) for name in metrics.__dataclass_fields__}


def default_frequencies_hz() -> np.ndarray:
    return np.array([1.0, 2.5, 10.0, 100.0, 1.0e3, 1.0e4], dtype=float)


def cec_meq_g_to_c_kg(cec_meq_g: np.ndarray | float, *, faraday_c_eq: float = FARADAY_C_PER_EQ) -> np.ndarray:
    """Convert CEC from mEq/g to C/kg.

    Numerically, ``1 mEq/g = 1 Eq/kg``. Multiplying by Faraday's constant
    converts charge equivalents to Coulombs.
    """

    return np.asarray(cec_meq_g, dtype=float) * faraday_c_eq


def interpolate_table_column(
    table: pd.DataFrame | dict[str, object],
    time_column: str,
    value_column: str,
    target_time_h: np.ndarray,
) -> np.ndarray:
    frame = pd.DataFrame(table)
    return np.interp(
        np.asarray(target_time_h, dtype=float),
        frame[time_column].to_numpy(dtype=float),
        frame[value_column].to_numpy(dtype=float),
    )


def paper_water_conductivity_at_times(
    time_h: np.ndarray | float,
    si03: pd.DataFrame | dict[str, object],
    *,
    sigma_hcl_s_m: float = 0.44,
    end_time_h: float | None = None,
) -> np.ndarray:
    """Interpolate paper ``sigma_w(t)`` and linearly extend it to HCl."""

    target = np.atleast_1d(np.asarray(time_h, dtype=float))
    frame = pd.DataFrame(si03)
    src_t = frame["Time (h)"].to_numpy(dtype=float)
    src_sigma = frame["Water conductivity (S/m)"].to_numpy(dtype=float)
    out = np.interp(np.minimum(target, src_t[-1]), src_t, src_sigma)
    later = target > src_t[-1]
    if np.any(later):
        final_time = float(np.max(target)) if end_time_h is None else float(end_time_h)
        denom = max(final_time - src_t[-1], np.finfo(float).eps)
        frac = np.clip((target[later] - src_t[-1]) / denom, 0.0, 1.0)
        out[later] = src_sigma[-1] + frac * (sigma_hcl_s_m - src_sigma[-1])
    return out


def si03_solution_conductivity_at_times(
    time_h: np.ndarray | float,
    *,
    si03: pd.DataFrame | dict[str, object],
    sigma_hcl_s_m: float = 0.44,
    end_time_h: float | None = None,
) -> np.ndarray:
    """Interpolate SI-S03 water conductivity and extend to HCl conductivity."""

    return paper_water_conductivity_at_times(
        time_h,
        si03,
        sigma_hcl_s_m=sigma_hcl_s_m,
        end_time_h=end_time_h,
    )


def paper_validation_drivers_from_tables(
    time_h: np.ndarray | float,
    *,
    si01: pd.DataFrame | dict[str, object],
    si02: pd.DataFrame | dict[str, object],
    si03: pd.DataFrame | dict[str, object],
    sigma_hcl_s_m: float = 0.44,
    end_time_h: float | None = None,
    rho_calcite_kg_m3: float = 2710.0,
    faraday_c_eq: float = FARADAY_C_PER_EQ,
) -> pd.DataFrame:
    """Return paper-derived ``phi``, ``Sw``, ``sigma_w``, CEC, and ``Qv``."""

    target = np.atleast_1d(np.asarray(time_h, dtype=float))
    phi = interpolate_table_column(si01, "Time (h)", "Porosity", target)
    sw = interpolate_table_column(si01, "Time (h)", "Water saturation", target)
    cec_meq_g = interpolate_table_column(si02, "Time (h)", "CEC (mEq/g)", target)
    sigma_w = paper_water_conductivity_at_times(
        target,
        si03,
        sigma_hcl_s_m=sigma_hcl_s_m,
        end_time_h=end_time_h,
    )
    qv_c_m3 = (
        (1.0 - phi)
        / np.maximum(phi, np.finfo(float).eps)
        * rho_calcite_kg_m3
        * cec_meq_g_to_c_kg(cec_meq_g, faraday_c_eq=faraday_c_eq)
    )
    return pd.DataFrame(
        {
            "time_h": target,
            "phi": phi,
            "sw": sw,
            "sigma_w_s_m": sigma_w,
            "cec_meq_g": cec_meq_g,
            "qv_c_m3": qv_c_m3,
        }
    )


def waxman_smits_water_increment(
    *,
    phi: np.ndarray | float,
    sw: np.ndarray | float,
    cec_meq_g: np.ndarray | float,
    rho_calcite_kg_m3: float = 2710.0,
    beta_m2_s_v: float = 19.98e-8,
    alpha: float = 6.0,
    faraday_c_eq: float = FARADAY_C_PER_EQ,
) -> np.ndarray:
    """Approximate CEC/EDL interface response as a water-phase increment.

    The returned increment is intended for the AC2D water phase:
    ``Delta sigma_w* = beta Qv / Sw + i (beta / alpha) Qv / Sw``.
    For a fully saturated, parallel-flow ``m=1`` medium this reproduces the
    Waxman--Smits source terms before the AC2D geometry solve applies the
    image-derived phase arrangement.
    """

    phi_arr = np.asarray(phi, dtype=float)
    sw_arr = np.asarray(sw, dtype=float)
    cec_arr = np.asarray(cec_meq_g, dtype=float)
    qv_c_m3 = (
        (1.0 - phi_arr)
        / np.maximum(phi_arr, np.finfo(float).eps)
        * rho_calcite_kg_m3
        * cec_meq_g_to_c_kg(cec_arr, faraday_c_eq=faraday_c_eq)
    )
    sw_safe = np.maximum(sw_arr, np.finfo(float).eps)
    return beta_m2_s_v * qv_c_m3 / sw_safe + 1j * (beta_m2_s_v / alpha) * qv_c_m3 / sw_safe


def schwarz_interface_conductivity_delta(
    frequency_hz: np.ndarray,
    *,
    interface_density_1_m: np.ndarray | float,
    sigma_s_s: float,
    characteristic_length_m: np.ndarray | float,
    diffusion_coefficient_m2_s: float,
) -> np.ndarray:
    """Upscale Schwarz/Debye surface conductance to volumetric conductivity.

    ``C_p* = i omega tau / (1 + i omega tau) * Sigma_s`` and
    ``Delta sigma* = C_p* * interface_density`` with ``tau = r^2 / (2D)``.
    """

    freq = np.asarray(frequency_hz, dtype=float)
    omega = 2.0 * np.pi * freq
    density = np.asarray(interface_density_1_m, dtype=float)
    radius = np.asarray(characteristic_length_m, dtype=float)
    if np.any(radius <= 0):
        raise ValueError("characteristic_length_m must be positive")
    if diffusion_coefficient_m2_s <= 0:
        raise ValueError("diffusion_coefficient_m2_s must be positive")
    tau = radius**2 / (2.0 * diffusion_coefficient_m2_s)
    iwt = 1j * omega * tau
    return sigma_s_s * density * iwt / (1.0 + iwt)


def mechanism_phase_conductivities(
    frequency_hz: np.ndarray,
    *,
    mechanism: str,
    metrics: GeometryMetrics2D,
    pore_radii_m: np.ndarray,
    throat_lengths_m: np.ndarray,
    all_components: tuple[str, ...] = ("maxwell", "pore", "membrane", "grain"),
    water_conductivity_s_m: float = 0.013,
    calcite_conductivity_s_m: float = 1.0e-8,
    water_relative_permittivity: float = 80.0,
    calcite_relative_permittivity: float = 8.5,
    grain_surface_conductance_s: float = 1.0e-9,
    grain_diffusion_coefficient_m2_s: float = 1.3e-9,
    interface_delta_conductivity_s_m: np.ndarray | complex | float = 0.0 + 0.0j,
) -> pd.DataFrame:
    """Create phase conductivities for one mechanism switch.

    This v1 helper keeps all mechanisms as equivalent phase-property changes so
    they can be solved by the same AC2D grid solver. Interface-specific grain
    polarization is upscaled from interface length per active area.
    """

    mechanism = mechanism.lower()
    valid = {"maxwell", "pore", "membrane", "grain", "interface", "all"}
    if mechanism not in valid:
        raise ValueError(f"mechanism must be one of {sorted(valid)}")

    freq = np.asarray(frequency_hz, dtype=float)
    omega = 2.0 * np.pi * freq
    params = PolarizationParameters(
        water_conductivity_s_m=water_conductivity_s_m,
        water_relative_permittivity=water_relative_permittivity,
        solid_relative_permittivity=calcite_relative_permittivity,
    )

    water_sigma = np.full(freq.shape, water_conductivity_s_m, dtype=np.complex128)
    calcite_sigma = np.full(freq.shape, calcite_conductivity_s_m, dtype=np.complex128)
    interface_delta = np.asarray(interface_delta_conductivity_s_m, dtype=np.complex128)
    if interface_delta.ndim == 0:
        interface_delta = np.full(freq.shape, complex(interface_delta), dtype=np.complex128)
    elif interface_delta.shape != freq.shape:
        raise ValueError("interface_delta_conductivity_s_m must be scalar or match frequency_hz shape")

    enabled = {mechanism} if mechanism != "all" else set(all_components)

    if "maxwell" in enabled:
        water_sigma += 1j * omega * water_relative_permittivity * EPSILON_0
        calcite_sigma += 1j * omega * calcite_relative_permittivity * EPSILON_0

    if "pore" in enabled and pore_radii_m.size:
        weights = np.ones_like(pore_radii_m, dtype=float)
        pore_conductance = pore_polarization_conductance(freq, pore_radii_m, weights, params)
        water_sigma += upscale_conductance_to_water_conductivity(pore_conductance, params)

    if "membrane" in enabled and throat_lengths_m.size:
        lengths = np.maximum(throat_lengths_m, np.finfo(float).tiny)
        weights = np.ones_like(lengths, dtype=float)
        area_m2 = np.pi * max(metrics.pore_radius_median_m, 0.5 * (metrics.dx_m + metrics.dy_m)) ** 2
        zdc = lengths / np.maximum(water_conductivity_s_m * area_m2, np.finfo(float).tiny)
        membrane_conductance = membrane_polarization_conductance(freq, lengths, weights, zdc, params)
        water_sigma += upscale_conductance_to_water_conductivity(membrane_conductance, params)

    if "grain" in enabled and metrics.active_area_m2 > 0 and metrics.calcite_equivalent_radius_m > 0:
        interface_density = metrics.interface_length_m / metrics.active_area_m2
        grain_relaxation_time_s = metrics.calcite_equivalent_radius_m**2 / (2.0 * grain_diffusion_coefficient_m2_s)
        iwt = 1j * omega * grain_relaxation_time_s
        grain_delta = grain_surface_conductance_s * interface_density * iwt / (1.0 + iwt)
        water_sigma += grain_delta

    if "interface" in enabled:
        water_sigma += interface_delta

    return pd.DataFrame(
        {
            "frequency_hz": freq,
            "mechanism": mechanism,
            "water_sigma_real_s_m": water_sigma.real,
            "water_sigma_imag_s_m": water_sigma.imag,
            "solid_sigma_real_s_m": calcite_sigma.real,
            "solid_sigma_imag_s_m": calcite_sigma.imag,
        }
    )


def apparent_water_sigma_from_delta(frequency_hz: np.ndarray, delta_sigma_s_m: np.ndarray) -> np.ndarray:
    """Small wrapper retained for scripts that want the Niu water convention."""

    return apparent_water_conductivity(frequency_hz, delta_sigma_s_m)
