#!/usr/bin/env python3
"""Fast full-resolution PNM parameter calibration for sample 16.

This is a diagnostic surrogate: it uses the full-resolution pnextract network
and polarization spectra to tune background conductivity, relaxation peak
position, and pore/throat polarization amplitudes without running the full AC3D
field solve at every trial.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = ROOT / "code"
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_polarization_spectra import compute_spectra, load_pnextract  # noqa: E402
from pore_scale_electrical.polarization import EPSILON_0, PolarizationParameters  # noqa: E402
from run_sample89_3d_sip_comparison import (  # noqa: E402
    add_bulk_conductivity_from_psip_impedance,
    parse_psip_csv,
)


def fit_linear_complex_surrogate(
    frequency_hz: np.ndarray,
    experiment_real_s_m: np.ndarray,
    experiment_imag_s_m: np.ndarray,
    pore_delta_s_m: np.ndarray,
    membrane_delta_s_m: np.ndarray,
    *,
    real_scale: float | None = None,
    imag_scale: float | None = None,
) -> dict[str, object]:
    """Fit a compact real/imag surrogate with separate diagnostic amplitudes.

    Real part:
        b0 + b1 Re(pore) + b2 Re(membrane) + b3 centered_log10(f)

    Imaginary part:
        c0 + c1 Im(pore) + c2 Im(membrane) + c3 omega

    The separate real/imag amplitudes are deliberate here: this is a fast
    calibration diagnostic for identifying parameter directions before an AC3D
    rerun, not a replacement for the final field solve.
    """

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    experiment_real_s_m = np.asarray(experiment_real_s_m, dtype=float)
    experiment_imag_s_m = np.asarray(experiment_imag_s_m, dtype=float)
    pore_delta_s_m = np.asarray(pore_delta_s_m, dtype=np.complex128)
    membrane_delta_s_m = np.asarray(membrane_delta_s_m, dtype=np.complex128)

    omega = 2.0 * np.pi * frequency_hz
    centered_logf = np.log10(frequency_hz) - np.log10(frequency_hz).mean()

    real_design = np.column_stack(
        [
            np.ones_like(frequency_hz),
            pore_delta_s_m.real,
            membrane_delta_s_m.real,
            centered_logf,
        ]
    )
    imag_design = np.column_stack(
        [
            np.ones_like(frequency_hz),
            pore_delta_s_m.imag,
            membrane_delta_s_m.imag,
            omega,
        ]
    )

    if real_scale is None:
        real_scale = max(float(np.ptp(experiment_real_s_m)), float(np.median(np.abs(experiment_real_s_m))), 1.0e-12)
    if imag_scale is None:
        imag_scale = max(float(np.ptp(experiment_imag_s_m)), float(np.max(np.abs(experiment_imag_s_m))), 1.0e-12)

    real_coeff, *_ = np.linalg.lstsq(real_design / real_scale, experiment_real_s_m / real_scale, rcond=None)
    imag_coeff, *_ = np.linalg.lstsq(imag_design / imag_scale, experiment_imag_s_m / imag_scale, rcond=None)

    model_real = real_design @ real_coeff
    model_imag = imag_design @ imag_coeff
    real_resid = model_real - experiment_real_s_m
    imag_resid = model_imag - experiment_imag_s_m
    score = float(np.sqrt(np.mean((real_resid / real_scale) ** 2 + (imag_resid / imag_scale) ** 2)))

    return {
        "real_coefficients": real_coeff,
        "imag_coefficients": imag_coeff,
        "model_real_s_m": model_real,
        "model_imag_s_m": model_imag,
        "real_rmse_s_m": float(np.sqrt(np.mean(real_resid**2))),
        "imag_rmse_s_m": float(np.sqrt(np.mean(imag_resid**2))),
        "normalized_rmse": score,
        "real_scale_s_m": float(real_scale),
        "imag_scale_s_m": float(imag_scale),
    }


def debye_response(frequency_hz: np.ndarray, relaxation_frequency_hz: np.ndarray) -> np.ndarray:
    """Return Debye conductivity basis i*f/f0 / (1 + i*f/f0)."""

    frequency_hz = np.asarray(frequency_hz, dtype=float)[:, None]
    relaxation_frequency_hz = np.asarray(relaxation_frequency_hz, dtype=float)[None, :]
    x = frequency_hz / relaxation_frequency_hz
    return 1j * x / (1.0 + 1j * x)


def fit_debye_dictionary_surrogate(
    frequency_hz: np.ndarray,
    experiment_real_s_m: np.ndarray,
    experiment_imag_s_m: np.ndarray,
    relaxation_frequency_hz: np.ndarray,
    *,
    real_scale: float | None = None,
    imag_scale: float | None = None,
) -> dict[str, object]:
    """Fit a shared-amplitude Debye dictionary plus small background terms."""

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    experiment_real_s_m = np.asarray(experiment_real_s_m, dtype=float)
    experiment_imag_s_m = np.asarray(experiment_imag_s_m, dtype=float)
    relaxation_frequency_hz = np.asarray(relaxation_frequency_hz, dtype=float)
    omega = 2.0 * np.pi * frequency_hz
    centered_logf = np.log10(frequency_hz) - np.log10(frequency_hz).mean()
    basis = debye_response(frequency_hz, relaxation_frequency_hz)

    if real_scale is None:
        real_scale = max(float(np.ptp(experiment_real_s_m)), float(np.median(np.abs(experiment_real_s_m))), 1.0e-12)
    if imag_scale is None:
        imag_scale = max(float(np.ptp(experiment_imag_s_m)), float(np.max(np.abs(experiment_imag_s_m))), 1.0e-12)

    n = len(frequency_hz)
    n_basis = len(relaxation_frequency_hz)
    design = np.zeros((2 * n, 4 + n_basis), dtype=float)
    target = np.concatenate([experiment_real_s_m / real_scale, experiment_imag_s_m / imag_scale])

    # Real-only background: constant and weak log-frequency trend.
    design[:n, 0] = 1.0 / real_scale
    design[:n, 1] = centered_logf / real_scale
    # Imaginary-only background: offset and high-frequency dielectric slope.
    design[n:, 2] = 1.0 / imag_scale
    design[n:, 3] = omega / imag_scale
    # Shared Debye amplitudes affect real and imaginary parts consistently.
    design[:n, 4:] = basis.real / real_scale
    design[n:, 4:] = basis.imag / imag_scale

    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    model = design @ coefficients
    model_real = model[:n] * real_scale
    model_imag = model[n:] * imag_scale
    real_resid = model_real - experiment_real_s_m
    imag_resid = model_imag - experiment_imag_s_m
    score = float(np.sqrt(np.mean((real_resid / real_scale) ** 2 + (imag_resid / imag_scale) ** 2)))

    return {
        "coefficients": coefficients,
        "background_real_s_m": float(coefficients[0]),
        "real_log10_frequency_slope_s_m": float(coefficients[1]),
        "imag_offset_s_m": float(coefficients[2]),
        "imag_dielectric_slope_f_m": float(coefficients[3]),
        "debye_amplitudes_s_m": coefficients[4:],
        "relaxation_frequency_hz": relaxation_frequency_hz,
        "model_real_s_m": model_real,
        "model_imag_s_m": model_imag,
        "real_rmse_s_m": float(np.sqrt(np.mean(real_resid**2))),
        "imag_rmse_s_m": float(np.sqrt(np.mean(imag_resid**2))),
        "normalized_rmse": score,
        "real_scale_s_m": float(real_scale),
        "imag_scale_s_m": float(imag_scale),
    }


def fit_positive_debye_dictionary_surrogate(
    frequency_hz: np.ndarray,
    experiment_real_s_m: np.ndarray,
    experiment_imag_s_m: np.ndarray,
    relaxation_frequency_hz: np.ndarray,
    *,
    max_amplitude_s_m: float = 1.0e-4,
    fit_mask: np.ndarray | None = None,
    real_scale: float | None = None,
    imag_scale: float | None = None,
) -> dict[str, object]:
    """Fit a bounded, non-negative Debye dictionary for physical diagnostics."""

    from scipy.optimize import lsq_linear

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    experiment_real_s_m = np.asarray(experiment_real_s_m, dtype=float)
    experiment_imag_s_m = np.asarray(experiment_imag_s_m, dtype=float)
    relaxation_frequency_hz = np.asarray(relaxation_frequency_hz, dtype=float)
    if fit_mask is None:
        fit_mask = np.ones_like(frequency_hz, dtype=bool)
    else:
        fit_mask = np.asarray(fit_mask, dtype=bool)

    fit_frequency = frequency_hz[fit_mask]
    fit_real = experiment_real_s_m[fit_mask]
    fit_imag = experiment_imag_s_m[fit_mask]
    omega = 2.0 * np.pi * fit_frequency
    centered_logf = np.log10(fit_frequency) - np.log10(fit_frequency).mean()
    basis = debye_response(fit_frequency, relaxation_frequency_hz)

    if real_scale is None:
        real_scale = max(float(np.ptp(fit_real)), float(np.median(np.abs(fit_real))), 1.0e-12)
    if imag_scale is None:
        imag_scale = max(float(np.ptp(fit_imag)), float(np.max(np.abs(fit_imag))), 1.0e-12)

    n = len(fit_frequency)
    n_basis = len(relaxation_frequency_hz)
    design = np.zeros((2 * n, 4 + n_basis), dtype=float)
    target = np.concatenate([fit_real / real_scale, fit_imag / imag_scale])
    design[:n, 0] = 1.0 / real_scale
    design[:n, 1] = centered_logf / real_scale
    design[n:, 2] = 1.0 / imag_scale
    design[n:, 3] = omega / imag_scale
    design[:n, 4:] = basis.real / real_scale
    design[n:, 4:] = basis.imag / imag_scale

    lower = np.concatenate([[0.0, -np.inf, -np.inf, 0.0], np.zeros(n_basis)])
    upper = np.concatenate([[np.inf, np.inf, np.inf, np.inf], np.full(n_basis, max_amplitude_s_m)])
    result = lsq_linear(design, target, bounds=(lower, upper), tol=1.0e-12, max_iter=2000)
    coefficients = result.x

    all_basis = debye_response(frequency_hz, relaxation_frequency_hz)
    all_logf = np.log10(frequency_hz) - np.log10(fit_frequency).mean()
    all_omega = 2.0 * np.pi * frequency_hz
    debye_sum = all_basis @ coefficients[4:]
    model_real = coefficients[0] + coefficients[1] * all_logf + debye_sum.real
    model_imag = coefficients[2] + coefficients[3] * all_omega + debye_sum.imag
    real_resid = model_real[fit_mask] - fit_real
    imag_resid = model_imag[fit_mask] - fit_imag
    score = float(np.sqrt(np.mean((real_resid / real_scale) ** 2 + (imag_resid / imag_scale) ** 2)))

    return {
        "coefficients": coefficients,
        "background_real_s_m": float(coefficients[0]),
        "real_log10_frequency_slope_s_m": float(coefficients[1]),
        "imag_offset_s_m": float(coefficients[2]),
        "imag_dielectric_slope_f_m": float(coefficients[3]),
        "debye_amplitudes_s_m": coefficients[4:],
        "relaxation_frequency_hz": relaxation_frequency_hz,
        "fit_mask": fit_mask,
        "max_amplitude_s_m": float(max_amplitude_s_m),
        "model_real_s_m": model_real,
        "model_imag_s_m": model_imag,
        "real_rmse_s_m": float(np.sqrt(np.mean(real_resid**2))),
        "imag_rmse_s_m": float(np.sqrt(np.mean(imag_resid**2))),
        "normalized_rmse": score,
        "real_scale_s_m": float(real_scale),
        "imag_scale_s_m": float(imag_scale),
        "optimizer_cost": float(result.cost),
        "optimizer_success": bool(result.success),
    }


def build_params(args: argparse.Namespace, diffusion_coefficient_m2_s: float) -> PolarizationParameters:
    params = PolarizationParameters()
    return replace(
        params,
        water_conductivity_s_m=args.water_conductivity_s_m,
        surface_conductance_s=args.surface_conductance_s,
        membrane_polarizability=args.membrane_polarizability,
        dynamic_pore_size_m=args.dynamic_pore_size_m,
        diffusion_coefficient_m2_s=diffusion_coefficient_m2_s,
        water_relative_permittivity=args.water_relative_permittivity,
        solid_relative_permittivity=args.solid_relative_permittivity,
    )


def fit_for_diffusion_grid(
    frequencies_hz: np.ndarray,
    experiment_real_s_m: np.ndarray,
    experiment_imag_s_m: np.ndarray,
    pores: pd.DataFrame,
    throats: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    rows: list[dict[str, float]] = []
    best: dict[str, object] | None = None
    best_base: pd.DataFrame | None = None

    if args.allow_diffusion_fit:
        diffusion_values = np.logspace(
            np.log10(args.diffusion_min_m2_s),
            np.log10(args.diffusion_max_m2_s),
            args.diffusion_steps,
        )
    else:
        diffusion_values = np.array([args.diffusion_coefficient_m2_s], dtype=float)
    for diffusion in diffusion_values:
        params = build_params(args, float(diffusion))
        base, _metadata = compute_spectra(frequencies_hz, pores, throats, params, figure5_zdc_ohm=None)
        pore_delta = (
            base["delta_sigma_pore_real_s_m"].to_numpy(dtype=float)
            + 1j * base["delta_sigma_pore_imag_s_m"].to_numpy(dtype=float)
        )
        membrane_delta = (
            base["delta_sigma_membrane_real_s_m"].to_numpy(dtype=float)
            + 1j * base["delta_sigma_membrane_imag_s_m"].to_numpy(dtype=float)
        )
        fit = fit_linear_complex_surrogate(frequencies_hz, experiment_real_s_m, experiment_imag_s_m, pore_delta, membrane_delta)

        real_coeff = np.asarray(fit["real_coefficients"], dtype=float)
        imag_coeff = np.asarray(fit["imag_coefficients"], dtype=float)
        row = {
            "diffusion_coefficient_m2_s": float(diffusion),
            "normalized_rmse": float(fit["normalized_rmse"]),
            "real_rmse_s_m": float(fit["real_rmse_s_m"]),
            "imag_rmse_s_m": float(fit["imag_rmse_s_m"]),
            "real_baseline_s_m": float(real_coeff[0]),
            "real_pore_scale": float(real_coeff[1]),
            "real_membrane_scale": float(real_coeff[2]),
            "real_log10_frequency_slope_s_m": float(real_coeff[3]),
            "imag_offset_s_m": float(imag_coeff[0]),
            "imag_pore_scale": float(imag_coeff[1]),
            "imag_membrane_scale": float(imag_coeff[2]),
            "imag_dielectric_slope_f_m": float(imag_coeff[3]),
        }
        rows.append(row)

        if best is None or row["normalized_rmse"] < float(best["scan_row"]["normalized_rmse"]):
            best = {"fit": fit, "params": params, "scan_row": row}
            best_base = base

    if best is None or best_base is None:
        raise RuntimeError("no diffusion trials were evaluated")
    return pd.DataFrame(rows), best, best_base


def make_best_fit_table(
    frequencies_hz: np.ndarray,
    experiment_real_s_m: np.ndarray,
    experiment_imag_s_m: np.ndarray,
    base: pd.DataFrame,
    fit: dict[str, object],
) -> pd.DataFrame:
    pore = base["delta_sigma_pore_real_s_m"].to_numpy(dtype=float) + 1j * base["delta_sigma_pore_imag_s_m"].to_numpy(dtype=float)
    membrane = base["delta_sigma_membrane_real_s_m"].to_numpy(dtype=float) + 1j * base[
        "delta_sigma_membrane_imag_s_m"
    ].to_numpy(dtype=float)
    logf = np.log10(frequencies_hz) - np.log10(frequencies_hz).mean()
    omega = 2.0 * np.pi * frequencies_hz
    real_coeff = np.asarray(fit["real_coefficients"], dtype=float)
    imag_coeff = np.asarray(fit["imag_coefficients"], dtype=float)
    return pd.DataFrame(
        {
            "frequency_hz": frequencies_hz,
            "experiment_real_s_m": experiment_real_s_m,
            "experiment_imag_s_m": experiment_imag_s_m,
            "model_real_s_m": np.asarray(fit["model_real_s_m"], dtype=float),
            "model_imag_s_m": np.asarray(fit["model_imag_s_m"], dtype=float),
            "real_baseline_contribution_s_m": real_coeff[0],
            "real_pore_contribution_s_m": real_coeff[1] * pore.real,
            "real_membrane_contribution_s_m": real_coeff[2] * membrane.real,
            "real_log_frequency_contribution_s_m": real_coeff[3] * logf,
            "imag_offset_contribution_s_m": imag_coeff[0],
            "imag_pore_contribution_s_m": imag_coeff[1] * pore.imag,
            "imag_membrane_contribution_s_m": imag_coeff[2] * membrane.imag,
            "imag_dielectric_contribution_s_m": imag_coeff[3] * omega,
            "base_delta_pore_real_s_m": pore.real,
            "base_delta_pore_imag_s_m": pore.imag,
            "base_delta_membrane_real_s_m": membrane.real,
            "base_delta_membrane_imag_s_m": membrane.imag,
        }
    )


def make_debye_fit_table(
    frequencies_hz: np.ndarray,
    experiment_real_s_m: np.ndarray,
    experiment_imag_s_m: np.ndarray,
    fit: dict[str, object],
) -> pd.DataFrame:
    relaxation_frequency_hz = np.asarray(fit["relaxation_frequency_hz"], dtype=float)
    amplitudes = np.asarray(fit["debye_amplitudes_s_m"], dtype=float)
    response = debye_response(frequencies_hz, relaxation_frequency_hz) @ amplitudes
    logf = np.log10(frequencies_hz) - np.log10(frequencies_hz).mean()
    omega = 2.0 * np.pi * frequencies_hz
    return pd.DataFrame(
        {
            "frequency_hz": frequencies_hz,
            "experiment_real_s_m": experiment_real_s_m,
            "experiment_imag_s_m": experiment_imag_s_m,
            "model_real_s_m": np.asarray(fit["model_real_s_m"], dtype=float),
            "model_imag_s_m": np.asarray(fit["model_imag_s_m"], dtype=float),
            "real_background_contribution_s_m": float(fit["background_real_s_m"]),
            "real_log_frequency_contribution_s_m": float(fit["real_log10_frequency_slope_s_m"]) * logf,
            "real_debye_sum_contribution_s_m": response.real,
            "imag_offset_contribution_s_m": float(fit["imag_offset_s_m"]),
            "imag_dielectric_contribution_s_m": float(fit["imag_dielectric_slope_f_m"]) * omega,
            "imag_debye_sum_contribution_s_m": response.imag,
        }
    )


def plot_best_fit(table: pd.DataFrame, output_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)
    real_ax, imag_ax = axes

    experiment_plot_mask = (table["experiment_real_s_m"] > 0.0) & (table["experiment_imag_s_m"] > 0.0)
    real_model_mask = table["model_real_s_m"] > 0.0
    imag_model_mask = table["model_imag_s_m"] > 0.0

    real_ax.semilogx(
        table.loc[experiment_plot_mask, "frequency_hz"],
        np.log10(table.loc[experiment_plot_mask, "experiment_real_s_m"]),
        "ko",
        ms=4,
        label="Experiment, imag > 0",
    )
    real_ax.semilogx(
        table.loc[real_model_mask, "frequency_hz"],
        np.log10(table.loc[real_model_mask, "model_real_s_m"]),
        color="#d62728",
        lw=2.0,
        label="Calibrated PNM surrogate",
    )
    real_ax.set_xlabel("Frequency (Hz)")
    real_ax.set_ylabel("log10 real conductivity (S/m)")
    real_ax.grid(True, which="both", alpha=0.25)
    real_ax.legend(frameon=False)

    imag_ax.semilogx(
        table.loc[experiment_plot_mask, "frequency_hz"],
        np.log10(table.loc[experiment_plot_mask, "experiment_imag_s_m"]),
        "ko",
        ms=4,
        label="Experiment, imag > 0",
    )
    imag_ax.semilogx(
        table.loc[imag_model_mask, "frequency_hz"],
        np.log10(table.loc[imag_model_mask, "model_imag_s_m"]),
        color="#d62728",
        lw=2.0,
        label="Calibrated PNM surrogate",
    )
    imag_ax.set_xlabel("Frequency (Hz)")
    imag_ax.set_ylabel("log10 imaginary conductivity (S/m)")
    imag_ax.grid(True, which="both", alpha=0.25)
    imag_ax.legend(frameon=False)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def finite_float(value: object) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"non-finite value: {value}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network-dir",
        default=str(ROOT / "results" / "pnextract" / "sample_16_Grainstone_5-16seged_fullres" / "network_parsed"),
    )
    parser.add_argument(
        "--experiment-csv",
        default=str(
            ROOT
            / "data_inventory"
            / "ct_backed_samples_raw_copy_20260605"
            / "sample_16_Grainstone"
            / "SIP"
            / "LKC-16.csv"
        ),
    )
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "sample16_3d_sip_lkc16_fullres_pnm_calibration_v1"))
    parser.add_argument("--current-resistor-ohm", type=float, default=10000.0)
    parser.add_argument("--sample-length-cm", type=float, default=3.6322)
    parser.add_argument("--sample-diameter-cm", type=float, default=2.52476)
    parser.add_argument("--water-conductivity-s-m", type=float, default=0.12)
    parser.add_argument("--surface-conductance-s", type=float, default=1.3e-9)
    parser.add_argument("--membrane-polarizability", type=float, default=0.01)
    parser.add_argument("--dynamic-pore-size-m", type=float, default=2.45913e-6)
    parser.add_argument("--water-relative-permittivity", type=float, default=80.0)
    parser.add_argument("--solid-relative-permittivity", type=float, default=7.0)
    parser.add_argument(
        "--diffusion-coefficient-m2-s",
        type=float,
        default=1.3e-9,
        help="Physically constrained ion diffusion coefficient used for publication-facing sample16 calibration.",
    )
    parser.add_argument(
        "--allow-diffusion-fit",
        action="store_true",
        help=(
            "Enable diagnostic diffusion scanning. This can reveal relaxation-frequency mismatch, "
            "but the fitted D is not used as a publication-facing recommended parameter."
        ),
    )
    parser.add_argument("--diffusion-min-m2-s", type=float, default=1.0e-10)
    parser.add_argument("--diffusion-max-m2-s", type=float, default=1.0e-4)
    parser.add_argument("--diffusion-steps", type=int, default=97)
    parser.add_argument("--debye-min-hz", type=float, default=0.03)
    parser.add_argument("--debye-max-hz", type=float, default=1.0e5)
    parser.add_argument("--debye-steps", type=int, default=61)
    parser.add_argument("--positive-debye-max-amplitude-s-m", type=float, default=1.0e-4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiment = add_bulk_conductivity_from_psip_impedance(
        parse_psip_csv(Path(args.experiment_csv)),
        current_resistor_ohm=args.current_resistor_ohm,
        sample_length_cm=args.sample_length_cm,
        sample_diameter_cm=args.sample_diameter_cm,
    )
    frequencies_hz = experiment["frequency_hz"].to_numpy(dtype=float)
    experiment_real = experiment["experiment_sigma_real_s_m"].to_numpy(dtype=float)
    experiment_imag = experiment["experiment_sigma_imag_s_m"].to_numpy(dtype=float)

    reference_params = build_params(args, args.diffusion_coefficient_m2_s)
    pores, throats = load_pnextract(Path(args.network_dir), reference_params)
    scan, best, best_base = fit_for_diffusion_grid(frequencies_hz, experiment_real, experiment_imag, pores, throats, args)
    scan_path = out_dir / "sample16_fullres_pnm_calibration_diffusion_scan.csv"
    scan.to_csv(scan_path, index=False)

    fit = best["fit"]
    params = best["params"]
    table = make_best_fit_table(frequencies_hz, experiment_real, experiment_imag, best_base, fit)
    fit_csv = out_dir / "sample16_fullres_pnm_calibrated_fit.csv"
    table.to_csv(fit_csv, index=False)
    figure_path = out_dir / "sample16_fullres_pnm_calibrated_fit.png"
    plot_best_fit(table, figure_path)

    debye_centers_hz = np.logspace(np.log10(args.debye_min_hz), np.log10(args.debye_max_hz), args.debye_steps)
    debye_fit = fit_debye_dictionary_surrogate(frequencies_hz, experiment_real, experiment_imag, debye_centers_hz)
    debye_table = make_debye_fit_table(frequencies_hz, experiment_real, experiment_imag, debye_fit)
    debye_csv = out_dir / "sample16_fullres_debye_dictionary_calibrated_fit.csv"
    debye_table.to_csv(debye_csv, index=False)
    debye_figure_path = out_dir / "sample16_fullres_debye_dictionary_calibrated_fit.png"
    plot_best_fit(debye_table, debye_figure_path)

    amplitudes = np.asarray(debye_fit["debye_amplitudes_s_m"], dtype=float)
    top_indices = np.argsort(np.abs(amplitudes))[::-1][:10]
    top_debye_terms = [
        {
            "relaxation_frequency_hz": float(debye_centers_hz[i]),
            "amplitude_s_m": float(amplitudes[i]),
        }
        for i in top_indices
    ]

    positive_fit_mask = experiment_imag > 0.0
    positive_debye_fit = fit_positive_debye_dictionary_surrogate(
        frequencies_hz,
        experiment_real,
        experiment_imag,
        debye_centers_hz,
        max_amplitude_s_m=args.positive_debye_max_amplitude_s_m,
        fit_mask=positive_fit_mask,
    )
    positive_debye_table = make_debye_fit_table(frequencies_hz, experiment_real, experiment_imag, positive_debye_fit)
    positive_debye_csv = out_dir / "sample16_fullres_positive_debye_dictionary_calibrated_fit.csv"
    positive_debye_table.to_csv(positive_debye_csv, index=False)
    positive_debye_figure_path = out_dir / "sample16_fullres_positive_debye_dictionary_calibrated_fit.png"
    plot_best_fit(positive_debye_table, positive_debye_figure_path)
    positive_amplitudes = np.asarray(positive_debye_fit["debye_amplitudes_s_m"], dtype=float)
    positive_terms_csv = out_dir / "sample16_fullres_positive_debye_terms.csv"
    pd.DataFrame(
        {
            "relaxation_frequency_hz": debye_centers_hz,
            "amplitude_s_m": positive_amplitudes,
        }
    ).to_csv(positive_terms_csv, index=False)
    positive_top_indices = np.argsort(np.abs(positive_amplitudes))[::-1][:10]
    positive_top_debye_terms = [
        {
            "relaxation_frequency_hz": float(debye_centers_hz[i]),
            "amplitude_s_m": float(positive_amplitudes[i]),
        }
        for i in positive_top_indices
        if positive_amplitudes[i] > 0.0
    ]

    real_coeff = np.asarray(fit["real_coefficients"], dtype=float)
    imag_coeff = np.asarray(fit["imag_coefficients"], dtype=float)
    real_baseline = finite_float(real_coeff[0])
    equivalent_formation_factor = args.water_conductivity_s_m / real_baseline if real_baseline > 0 else np.nan
    diagnostic_best_diffusion = finite_float(params.diffusion_coefficient_m2_s)
    publication_diffusion = finite_float(args.diffusion_coefficient_m2_s)
    diffusion_status = (
        "diagnostic_scan_only_not_publication_parameter"
        if args.allow_diffusion_fit and not np.isclose(diagnostic_best_diffusion, publication_diffusion)
        else "fixed_publication_parameter"
    )

    excluded_imag_rows = [
        {
            "frequency_hz": float(frequency),
            "experiment_real_s_m": float(real),
            "experiment_imag_s_m": float(imag),
        }
        for frequency, real, imag in zip(frequencies_hz, experiment_real, experiment_imag)
        if imag <= 0.0
    ]
    prior_ac3d_csv = (
        ROOT
        / "results"
        / "sample16_3d_sip_lkc16_fullres_ac3d_jacobi_v2"
        / "sample16_3d_sip_lkc16_fullres_ac3d_mechanism_sweeps.csv"
    )
    full_ac3d_candidate: dict[str, object] = {
        "status": "not_verified_by_full_ac3d",
        "note": (
            "These are starting values for a later full AC3D probe. They scale the previous full-resolution "
            "AC3D Maxwell background down to the filtered low-frequency experiment and scale the previous "
            "polarization amplitude toward the positive-imaginary experimental peak."
        ),
    }
    if prior_ac3d_csv.exists():
        prior = pd.read_csv(prior_ac3d_csv)
        prior_maxwell = prior[prior["mechanism"] == "maxwell"].sort_values("frequency_hz")
        prior_all = prior[prior["mechanism"] == "all"].sort_values("frequency_hz")
        positive_rows = positive_debye_table[positive_fit_mask]
        if not prior_maxwell.empty and not prior_all.empty and not positive_rows.empty:
            prior_low_real = float(prior_maxwell.iloc[0]["effective_sigma_real_s_m"])
            target_low_real = float(positive_rows.iloc[0]["experiment_real_s_m"])
            background_scale = target_low_real / prior_low_real if prior_low_real > 0 else np.nan
            prior_peak_imag = float(prior_all["effective_sigma_imag_s_m"].max())
            target_peak_imag = float(positive_rows["experiment_imag_s_m"].max())
            polarization_scale = target_peak_imag / prior_peak_imag if prior_peak_imag > 0 else np.nan
            full_ac3d_candidate.update(
                {
                    "source_prior_ac3d_csv": str(prior_ac3d_csv),
                    "background_scale_from_prior_maxwell": background_scale,
                    "polarization_scale_from_prior_all_peak": polarization_scale,
                    "parameter_overrides": {
                        "water_conductivity_s_m": args.water_conductivity_s_m * background_scale,
                        "solid_conductivity_s_m": 0.0024 * background_scale,
                        "surface_conductance_s": args.surface_conductance_s * polarization_scale,
                        "membrane_polarizability": args.membrane_polarizability * polarization_scale,
                        "diffusion_coefficient_m2_s": publication_diffusion,
                        "dynamic_pore_size_m": args.dynamic_pore_size_m,
                        "water_relative_permittivity": args.water_relative_permittivity,
                        "solid_relative_permittivity": args.solid_relative_permittivity,
                    },
                    "command_template": (
                        "python code/scripts/sip_simulation/run_sample89_3d_sip_comparison.py "
                        "--mhd results/pnextract_inputs/sample_16_Grainstone_5-16seged_fullres/"
                        "sample_16_Grainstone_5-16seged_fullres_pnextract.mhd "
                        "--network-dir results/pnextract/sample_16_Grainstone_5-16seged_fullres/network_parsed "
                        "--experiment-csv data_inventory/ct_backed_samples_raw_copy_20260605/sample_16_Grainstone/SIP/LKC-16.csv "
                        "--segmented-volume results/segmented_cores/sample_16_Grainstone_5-16seged_solid255_pore0.tiff "
                        "--sample-label Sample16_Grainstone --experiment-label LKC16_experiment "
                        "--output-prefix sample16_3d_sip_lkc16_fullres_ac3d_calibrated_probe "
                        "--sample-length-cm 3.6322 --sample-diameter-cm 2.52476 "
                        f"--water-conductivity-s-m {args.water_conductivity_s_m * background_scale:.12g} "
                        f"--solid-conductivity-s-m {0.0024 * background_scale:.12g} "
                        f"--surface-conductance-s {args.surface_conductance_s * polarization_scale:.12g} "
                        f"--membrane-polarizability {args.membrane_polarizability * polarization_scale:.12g} "
                        f"--diffusion-coefficient-m2-s {publication_diffusion:.12g}"
                    ),
                }
            )

    recommended_config = {
        "purpose": (
            "Physically constrained fast-calibrated parameter set for sample 16 after confirming the PNM "
            "network is full resolution. Ion diffusion is fixed to a literature-defensible value by default."
        ),
        "sample": "sample_16_Grainstone / 5-16seged",
        "diffusion_parameter_policy": {
            "status": diffusion_status,
            "publication_diffusion_coefficient_m2_s": publication_diffusion,
            "diagnostic_best_diffusion_coefficient_m2_s": diagnostic_best_diffusion,
            "allow_diffusion_fit": bool(args.allow_diffusion_fit),
            "note": (
                "Fitting D can move pore/membrane relaxation peaks and may improve curve matching, but a "
                "large fitted D should be reported only as a diagnostic sign of missing physics, unresolved "
                "length scales, or geometry-model mismatch. It is not a recommended sample parameter."
            ),
        },
        "full_resolution_inputs": {
            "segmented_volume": str(
                ROOT
                / "data_inventory"
                / "ct_backed_samples_raw_copy_20260605"
                / "sample_16_Grainstone"
                / "CT_slices"
                / "1-CTseg"
                / "5-16seged.tiff"
            ),
            "field_solve_mhd": str(
                ROOT
                / "results"
                / "pnextract_inputs"
                / "sample_16_Grainstone_5-16seged_fullres"
                / "sample_16_Grainstone_5-16seged_fullres_pnextract.mhd"
            ),
            "network_dir": str(Path(args.network_dir)),
        },
        "experiment_filtering": {
            "rule": "Use only rows with experiment_imag_s_m > 0 for the preferred fit and plotted experimental points.",
            "n_input_frequencies": int(len(frequencies_hz)),
            "n_used_frequencies": int(np.count_nonzero(positive_fit_mask)),
            "excluded_nonpositive_imaginary_rows": excluded_imag_rows,
        },
        "preferred_fast_verified_model": {
            "type": "bounded_nonnegative_debye_dictionary_surrogate",
            "comparison_csv": str(positive_debye_csv),
            "comparison_png": str(positive_debye_figure_path),
            "positive_debye_terms_csv": str(positive_terms_csv),
            "metrics_on_filtered_rows": {
                "normalized_rmse": positive_debye_fit["normalized_rmse"],
                "real_rmse_s_m": positive_debye_fit["real_rmse_s_m"],
                "imag_rmse_s_m": positive_debye_fit["imag_rmse_s_m"],
            },
            "background_real_s_m": positive_debye_fit["background_real_s_m"],
            "equivalent_formation_factor_if_water_sigma_fixed": (
                args.water_conductivity_s_m / float(positive_debye_fit["background_real_s_m"])
                if float(positive_debye_fit["background_real_s_m"]) > 0
                else np.nan
            ),
            "top_positive_debye_terms_by_abs_amplitude": positive_top_debye_terms,
        },
        "direct_pnm_shape_diagnostic": {
            "best_diffusion_coefficient_m2_s": diagnostic_best_diffusion,
            "publication_diffusion_coefficient_m2_s": publication_diffusion,
            "normalized_rmse": fit["normalized_rmse"],
            "real_rmse_s_m": fit["real_rmse_s_m"],
            "imag_rmse_s_m": fit["imag_rmse_s_m"],
            "note": (
                "The direct pore/membrane-shape fit is retained as a diagnostic but has signed coefficients "
                "and is not the preferred matching model. If --allow-diffusion-fit is used, its best D is "
                "not propagated to the publication-facing AC3D candidate."
            ),
        },
        "full_ac3d_candidate_starting_point": full_ac3d_candidate,
    }
    recommended_config_path = out_dir / "sample16_fullres_recommended_simulation_parameters.json"
    recommended_config_path.write_text(json.dumps(recommended_config, indent=2, ensure_ascii=False), encoding="utf-8")

    metadata = {
        "purpose": "Fast diagnostic full-resolution PNM calibration for sample 16 before expensive AC3D reruns.",
        "network_dir": str(Path(args.network_dir)),
        "experiment_csv": str(Path(args.experiment_csv)),
        "full_resolution_network": True,
        "experiment_geometry": {
            "sample_length_cm": args.sample_length_cm,
            "sample_diameter_cm": args.sample_diameter_cm,
            "current_resistor_ohm": args.current_resistor_ohm,
        },
        "base_parameters": {
            "water_conductivity_s_m": args.water_conductivity_s_m,
            "surface_conductance_s": args.surface_conductance_s,
            "membrane_polarizability": args.membrane_polarizability,
            "dynamic_pore_size_m": args.dynamic_pore_size_m,
            "diffusion_coefficient_m2_s": publication_diffusion,
            "water_relative_permittivity": args.water_relative_permittivity,
            "solid_relative_permittivity": args.solid_relative_permittivity,
        },
        "diffusion_parameter_policy": {
            "status": diffusion_status,
            "publication_diffusion_coefficient_m2_s": publication_diffusion,
            "diagnostic_best_diffusion_coefficient_m2_s": diagnostic_best_diffusion,
            "allow_diffusion_fit": bool(args.allow_diffusion_fit),
        },
        "best_fit": {
            "normalized_rmse": fit["normalized_rmse"],
            "real_rmse_s_m": fit["real_rmse_s_m"],
            "imag_rmse_s_m": fit["imag_rmse_s_m"],
            "real_coefficients": {
                "baseline_s_m": real_coeff[0],
                "pore_real_scale": real_coeff[1],
                "membrane_real_scale": real_coeff[2],
                "log10_frequency_slope_s_m": real_coeff[3],
            },
            "imag_coefficients": {
                "offset_s_m": imag_coeff[0],
                "pore_imag_scale": imag_coeff[1],
                "membrane_imag_scale": imag_coeff[2],
                "dielectric_slope_f_m": imag_coeff[3],
                "dielectric_slope_relative_eps0": imag_coeff[3] / EPSILON_0,
            },
        },
        "physical_interpretation": {
            "equivalent_formation_factor_if_water_sigma_fixed": equivalent_formation_factor,
            "effective_surface_conductance_from_pore_imag_scale_s": args.surface_conductance_s * imag_coeff[1],
            "effective_membrane_polarizability_from_membrane_imag_scale": args.membrane_polarizability * imag_coeff[2],
            "note": (
                "The real and imaginary coefficients are fitted independently as a calibration diagnostic. "
                "Use the diffusion coefficient, background conductivity scale, and order-of-magnitude "
                "polarization amplitudes as candidates for the next AC3D field solve."
            ),
        },
        "debye_dictionary_fit": {
            "relaxation_frequency_range_hz": [args.debye_min_hz, args.debye_max_hz],
            "n_relaxation_terms": args.debye_steps,
            "normalized_rmse": debye_fit["normalized_rmse"],
            "real_rmse_s_m": debye_fit["real_rmse_s_m"],
            "imag_rmse_s_m": debye_fit["imag_rmse_s_m"],
            "background_real_s_m": debye_fit["background_real_s_m"],
            "equivalent_formation_factor_if_water_sigma_fixed": (
                args.water_conductivity_s_m / float(debye_fit["background_real_s_m"])
                if float(debye_fit["background_real_s_m"]) > 0
                else np.nan
            ),
            "imag_dielectric_slope_f_m": debye_fit["imag_dielectric_slope_f_m"],
            "imag_dielectric_slope_relative_eps0": float(debye_fit["imag_dielectric_slope_f_m"]) / EPSILON_0,
            "top_debye_terms_by_abs_amplitude": top_debye_terms,
            "note": (
                "This dictionary fit is more flexible than the direct PNM-shape fit. It is useful for matching "
                "the experimental trend and locating dominant relaxation bands, but signed amplitudes should "
                "be interpreted as an empirical diagnostic rather than a unique pore-scale parameter set."
            ),
        },
        "positive_debye_dictionary_fit": {
            "fit_mask": "experiment_imag_s_m > 0",
            "n_fit_frequencies": int(np.count_nonzero(positive_fit_mask)),
            "max_amplitude_s_m": args.positive_debye_max_amplitude_s_m,
            "normalized_rmse_on_fit_mask": positive_debye_fit["normalized_rmse"],
            "real_rmse_s_m_on_fit_mask": positive_debye_fit["real_rmse_s_m"],
            "imag_rmse_s_m_on_fit_mask": positive_debye_fit["imag_rmse_s_m"],
            "background_real_s_m": positive_debye_fit["background_real_s_m"],
            "equivalent_formation_factor_if_water_sigma_fixed": (
                args.water_conductivity_s_m / float(positive_debye_fit["background_real_s_m"])
                if float(positive_debye_fit["background_real_s_m"]) > 0
                else np.nan
            ),
            "imag_dielectric_slope_f_m": positive_debye_fit["imag_dielectric_slope_f_m"],
            "imag_dielectric_slope_relative_eps0": float(positive_debye_fit["imag_dielectric_slope_f_m"]) / EPSILON_0,
            "top_positive_debye_terms_by_abs_amplitude": positive_top_debye_terms,
            "optimizer_success": positive_debye_fit["optimizer_success"],
            "note": (
                "This constrained fit is the preferred physical diagnostic. It excludes negative-imaginary "
                "outlier frequencies from fitting and enforces non-negative Debye amplitudes."
            ),
        },
        "outputs": {
            "scan_csv": str(scan_path),
            "fit_csv": str(fit_csv),
            "comparison_png": str(figure_path),
            "debye_fit_csv": str(debye_csv),
            "debye_comparison_png": str(debye_figure_path),
            "positive_debye_fit_csv": str(positive_debye_csv),
            "positive_debye_comparison_png": str(positive_debye_figure_path),
            "positive_debye_terms_csv": str(positive_terms_csv),
            "recommended_simulation_parameters_json": str(recommended_config_path),
        },
    }
    metadata_path = out_dir / "sample16_fullres_pnm_best_fit_parameters.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
