"""Debye decomposition of complex conductivity spectra and the r = sqrt(2 tau D+) map.

Model (Nordsiek & Weller, 2008, conductivity form):

    sigma*(w) = sigma_0 + sum_k m_k * i w tau_k / (1 + i w tau_k) + i w eps_inf

with non-negative ``sigma_0``, ``m_k`` and ``eps_inf`` and second-difference
Tikhonov smoothing of ``m_k`` over ``log tau``. Without the ``eps_inf`` column
this is the formulation of ``fit_debye_distribution`` in
``code/scripts/sip_simulation/plot_sample16_sip_pore_distribution.py``; the
column absorbs the high-frequency dielectric (Maxwell-Wagner) tail so that it
is not mapped to spurious small radii.

Radius mapping (Schwarz, 1962; Zhang et al., 2018): ``r = sqrt(2 tau D_+)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class DebyeDecomposition:
    relaxation_time_s: np.ndarray
    amplitudes_s_m: np.ndarray
    sigma_0_s_m: float
    epsilon_inf_f_m: float
    prediction: np.ndarray
    real_rmse_s_m: float
    imag_rmse_s_m: float
    regularization: float
    success: bool

    @property
    def total_chargeability(self) -> float:
        total = float(self.amplitudes_s_m.sum())
        return total / (self.sigma_0_s_m + total)

    def radius_m(self, diffusion_m2_s: float) -> np.ndarray:
        return radius_from_relaxation_time(self.relaxation_time_s, diffusion_m2_s)


def radius_from_relaxation_time(tau_s, diffusion_m2_s: float) -> np.ndarray:
    return np.sqrt(2.0 * np.asarray(tau_s, dtype=float) * diffusion_m2_s)


def radius_from_frequency(frequency_hz, diffusion_m2_s: float) -> np.ndarray:
    """``r = sqrt(2 tau D)`` with ``tau = 1 / (2 pi f)``, i.e. ``sqrt(D / (pi f))``."""

    return np.sqrt(diffusion_m2_s / (np.pi * np.asarray(frequency_hz, dtype=float)))


def debye_decomposition(
    frequency_hz: np.ndarray,
    sigma: np.ndarray,
    relaxation_time_s: np.ndarray,
    regularization: float,
    *,
    include_permittivity: bool = True,
) -> DebyeDecomposition:
    f = np.asarray(frequency_hz, dtype=float)
    data = np.asarray(sigma, dtype=np.complex128)
    tau = np.asarray(relaxation_time_s, dtype=float)
    if f.ndim != 1 or f.shape != data.shape or tau.ndim != 1 or tau.size < 3:
        raise ValueError("frequency/sigma must be matching 1-D arrays and tau needs >= 3 bins")
    if regularization < 0 or np.any(f <= 0) or np.any(tau <= 0):
        raise ValueError("frequencies, relaxation times and regularization must be positive")

    omega = 2.0 * np.pi * f
    iwt = 1j * omega[:, None] * tau[None, :]
    basis = iwt / (1.0 + iwt)
    real_scale = max(float(np.ptp(data.real)), 1e-30)
    imag_scale = max(float(np.max(np.abs(data.imag))), 1e-30)
    amplitude_scale = max(real_scale, imag_scale)
    dc_scale = max(float(np.median(data.real)), 1e-30)
    eps_scale = imag_scale / float(omega.max())

    columns_real = [np.full((f.size, 1), dc_scale / real_scale), basis.real * amplitude_scale / real_scale]
    columns_imag = [np.zeros((f.size, 1)), basis.imag * amplitude_scale / imag_scale]
    if include_permittivity:
        columns_real.append(np.zeros((f.size, 1)))
        columns_imag.append((omega * eps_scale / imag_scale)[:, None])
    design = np.vstack([np.hstack(columns_real), np.hstack(columns_imag)])
    target = np.r_[data.real / real_scale, data.imag / imag_scale]

    n_extra = 1 if include_permittivity else 0
    smoothing = np.diff(np.eye(tau.size), n=2, axis=0)
    if regularization:
        penalty = np.hstack(
            [np.zeros((smoothing.shape[0], 1)), np.sqrt(regularization) * smoothing, np.zeros((smoothing.shape[0], n_extra))]
        )
        design = np.vstack([design, penalty])
        target = np.r_[target, np.zeros(smoothing.shape[0])]

    fit = lsq_linear(design, target, bounds=(0.0, np.inf), tol=1e-12, max_iter=10000)
    x = fit.x
    amplitudes = x[1 : 1 + tau.size] * amplitude_scale
    sigma_0 = float(x[0] * dc_scale)
    epsilon_inf = float(x[-1] * eps_scale) if include_permittivity else 0.0
    prediction = sigma_0 + basis @ amplitudes + 1j * omega * epsilon_inf
    return DebyeDecomposition(
        relaxation_time_s=tau,
        amplitudes_s_m=amplitudes,
        sigma_0_s_m=sigma_0,
        epsilon_inf_f_m=epsilon_inf,
        prediction=prediction,
        real_rmse_s_m=float(np.sqrt(np.mean((prediction.real - data.real) ** 2))),
        imag_rmse_s_m=float(np.sqrt(np.mean((prediction.imag - data.imag) ** 2))),
        regularization=float(regularization),
        success=bool(fit.success),
    )
