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

``relaxation_decomposition`` generalizes this to several kernel families fitted
together (Debye for pore/Stern polarization, Titov for membrane polarization,
Warburg as in Revil et al., 2014), each mapped to its own length scale:
pore radius ``r = sqrt(2 D tau)`` and throat length ``L = sqrt(4 D tau_m)``.
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


def debye_kernel(omega: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Schwarz/Stern pore polarization shape ``i w tau / (1 + i w tau)`` (Niu Eq. 17)."""

    iwt = 1j * omega[:, None] * tau[None, :]
    return iwt / (1.0 + iwt)


def warburg_kernel(omega: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Cole-Cole kernel with c = 0.5, ``1 - 1 / (1 + sqrt(i w tau))`` (Revil et al., 2014)."""

    return 1.0 - 1.0 / (1.0 + np.sqrt(1j * omega[:, None] * tau[None, :]))


def titov_kernel(omega: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Normalized Titov membrane shape ``1 - (1 - exp(-2 s)) / (2 s)``, ``s = sqrt(i w tau_m)``.

    To first order in the polarizability, a throat's conductivity perturbation
    ``sigma_w (Z_dc / Z* - 1) ~ eta sigma_w`` times this kernel (Niu Eq. 19).
    """

    s = np.sqrt(1j * omega[:, None] * tau[None, :])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        frac = (1.0 - np.exp(-2.0 * s)) / (2.0 * s)
    small = np.abs(s) < 1.0e-6
    frac[small] = 1.0 - s[small] + (2.0 / 3.0) * s[small] ** 2
    return 1.0 - frac


KERNELS = {"debye": debye_kernel, "warburg": warburg_kernel, "titov": titov_kernel}


@dataclass(frozen=True)
class RelaxationDecomposition:
    """Non-negative multi-kernel decomposition; one amplitude block per kernel family."""

    relaxation_time_s: dict[str, np.ndarray]
    amplitudes_s_m: dict[str, np.ndarray]
    sigma_0_s_m: float
    epsilon_inf_f_m: float
    prediction: np.ndarray
    real_rmse_s_m: float
    imag_rmse_s_m: float
    regularization: float
    success: bool


def relaxation_decomposition(
    frequency_hz: np.ndarray,
    sigma: np.ndarray,
    blocks: dict[str, tuple[np.ndarray, str]],
    regularization: float,
    *,
    include_permittivity: bool = True,
) -> RelaxationDecomposition:
    """Fit ``sigma_0 + sum_blocks sum_k m_k K(w; tau_k) + i w eps_inf`` with m >= 0.

    ``blocks`` maps a name to ``(tau grid, kernel name)``; each block gets its own
    second-difference smoothing over ``log tau`` with the same weight.
    """

    f = np.asarray(frequency_hz, dtype=float)
    data = np.asarray(sigma, dtype=np.complex128)
    if f.ndim != 1 or f.shape != data.shape:
        raise ValueError("frequency and sigma must be matching 1-D arrays")
    if regularization < 0 or np.any(f <= 0):
        raise ValueError("frequencies and regularization must be positive")
    omega = 2.0 * np.pi * f
    bases, taus = {}, {}
    for name, (tau, kernel) in blocks.items():
        tau = np.asarray(tau, dtype=float)
        if tau.ndim != 1 or tau.size < 3 or np.any(tau <= 0):
            raise ValueError(f"block {name}: tau needs >= 3 positive bins")
        taus[name] = tau
        bases[name] = KERNELS[kernel](omega, tau)

    real_scale = max(float(np.ptp(data.real)), 1e-30)
    imag_scale = max(float(np.max(np.abs(data.imag))), 1e-30)
    amplitude_scale = max(real_scale, imag_scale)
    dc_scale = max(float(np.median(data.real)), 1e-30)
    eps_scale = imag_scale / float(omega.max())

    columns_real = [np.full((f.size, 1), dc_scale / real_scale)]
    columns_imag = [np.zeros((f.size, 1))]
    for basis in bases.values():
        columns_real.append(basis.real * amplitude_scale / real_scale)
        columns_imag.append(basis.imag * amplitude_scale / imag_scale)
    if include_permittivity:
        columns_real.append(np.zeros((f.size, 1)))
        columns_imag.append((omega * eps_scale / imag_scale)[:, None])
    design = np.vstack([np.hstack(columns_real), np.hstack(columns_imag)])
    target = np.r_[data.real / real_scale, data.imag / imag_scale]

    n_total = design.shape[1]
    if regularization:
        penalties, offset = [], 1
        for tau in taus.values():
            smoothing = np.diff(np.eye(tau.size), n=2, axis=0)
            block = np.zeros((smoothing.shape[0], n_total))
            block[:, offset : offset + tau.size] = np.sqrt(regularization) * smoothing
            penalties.append(block)
            offset += tau.size
        design = np.vstack([design, *penalties])
        target = np.r_[target, np.zeros(sum(len(b) for b in penalties))]

    fit = lsq_linear(design, target, bounds=(0.0, np.inf), tol=1e-12, max_iter=10000)
    x = fit.x
    amplitudes, offset = {}, 1
    prediction = np.full(f.shape, x[0] * dc_scale, dtype=np.complex128)
    for name, tau in taus.items():
        amplitudes[name] = x[offset : offset + tau.size] * amplitude_scale
        prediction = prediction + bases[name] @ amplitudes[name]
        offset += tau.size
    epsilon_inf = float(x[-1] * eps_scale) if include_permittivity else 0.0
    prediction = prediction + 1j * omega * epsilon_inf
    return RelaxationDecomposition(
        relaxation_time_s=taus,
        amplitudes_s_m=amplitudes,
        sigma_0_s_m=float(x[0] * dc_scale),
        epsilon_inf_f_m=epsilon_inf,
        prediction=prediction,
        real_rmse_s_m=float(np.sqrt(np.mean((prediction.real - data.real) ** 2))),
        imag_rmse_s_m=float(np.sqrt(np.mean((prediction.imag - data.imag) ** 2))),
        regularization=float(regularization),
        success=bool(fit.success),
    )


def debye_decomposition(
    frequency_hz: np.ndarray,
    sigma: np.ndarray,
    relaxation_time_s: np.ndarray,
    regularization: float,
    *,
    include_permittivity: bool = True,
) -> DebyeDecomposition:
    """Single-kernel Debye decomposition (Nordsiek & Weller, 2008)."""

    fit = relaxation_decomposition(
        frequency_hz, sigma, {"debye": (relaxation_time_s, "debye")}, regularization, include_permittivity=include_permittivity
    )
    return DebyeDecomposition(
        relaxation_time_s=fit.relaxation_time_s["debye"],
        amplitudes_s_m=fit.amplitudes_s_m["debye"],
        sigma_0_s_m=fit.sigma_0_s_m,
        epsilon_inf_f_m=fit.epsilon_inf_f_m,
        prediction=fit.prediction,
        real_rmse_s_m=fit.real_rmse_s_m,
        imag_rmse_s_m=fit.imag_rmse_s_m,
        regularization=fit.regularization,
        success=fit.success,
    )
