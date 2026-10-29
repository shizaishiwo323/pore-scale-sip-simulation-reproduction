"""Niu et al. (2020) polarization mechanisms mapped onto a 2-D phase map.

The elementary kernels are the same equations as
``pore_scale_electrical.polarization`` (tested for equality):

* pore / Stern polarization, Eq. 17:  C_p* = Sigma_S * i w tau / (1 + i w tau),
  tau = r^2 / (2 D_+); upscaled with Eq. 12: Delta sigma_w* = 2 C_p* / Lambda;
* membrane (pore-throat) polarization, Titov Eq. 19:
  Z* = Z_dc [1 - eta (1 - (1 - exp(-2 sqrt(i w tau_m))) / (2 sqrt(i w tau_m)))],
  tau_m = L^2 / (4 D);
* interfacial (Maxwell-Wagner) polarization: phase dc conductivity plus
  ``i w epsilon`` of water and solid (Eq. 14 high-frequency terms).

Mechanism isolation follows Niu Section 5.3: ``pore`` and ``membrane`` give
the solid zero conductivity and omit the permittivity background;
``interfacial`` keeps only the dc + permittivity terms; ``all`` adds every
increment to the water phase before a single field solve.

2-D translation (see README.md):

* ``pore_mode="regional"`` assigns each labelled pore region its own Eq. 12
  increment ``2 C_p*(r_k) / Lambda_k``. With ``Lambda_k`` the field-weighted
  dynamic pore size of that region, the field solve reproduces the surface
  integral ``int C |E_t|^2 dl`` of the region exactly in the small-increment
  limit.
* ``pore_mode="uniform"`` is the literal Niu 3-D upscaling: one water-phase
  increment ``2 sum_k w_k C_p*(r_k) / Lambda`` with pore-area weights ``w_k``
  and the global dynamic pore size ``Lambda``.
* Membrane polarization is applied inside each throat region as the intensive
  conductivity ``sigma_w * Z_dc / Z*``, so that the throat impedance equals the
  Titov ``Z*`` exactly. Niu's Eq. 12 upscaling of ``1/Z* - 1/Z_dc`` is not used
  in 2-D because a per-unit-depth throat conductance would make that
  increment depend on an arbitrary out-of-plane thickness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

import numpy as np


EPSILON_0 = 8.85e-12
SOLID_LABEL = 0

Mechanism = Literal["interfacial", "pore", "membrane", "all"]
PoreMode = Literal["regional", "uniform"]
MECHANISMS: tuple[str, ...] = ("interfacial", "pore", "membrane", "all")


@dataclass(frozen=True)
class SIP2DParameters:
    """Electrical and electrochemical parameters (SI units)."""

    water_conductivity_s_m: float = 0.043
    solid_conductivity_s_m: float = 0.0
    water_relative_permittivity: float = 80.0
    solid_relative_permittivity: float = 7.0
    epsilon_0_f_m: float = EPSILON_0
    surface_conductance_s: float = 1.3e-9
    stern_diffusion_m2_s: float = 1.3e-9
    membrane_diffusion_m2_s: float = 1.3e-9
    membrane_polarizability: float = 0.01

    @property
    def water_permittivity_f_m(self) -> float:
        return self.water_relative_permittivity * self.epsilon_0_f_m

    @property
    def solid_permittivity_f_m(self) -> float:
        return self.solid_relative_permittivity * self.epsilon_0_f_m


@dataclass(frozen=True)
class RegionSpec:
    """One labelled pore region of the 2-D phase map.

    ``relaxation_radius_m`` is the ``r`` in ``tau = r^2 / (2 D_+)``;
    ``dynamic_pore_size_m`` is the ``Lambda`` used by Eq. 12 in regional mode;
    ``membrane_length_m`` marks a throat that carries Titov membrane
    polarization (``None`` for pore bodies).
    """

    label: int
    name: str
    kind: Literal["pore_body", "throat"]
    relaxation_radius_m: float
    dynamic_pore_size_m: float
    membrane_length_m: float | None = None
    stern_polarization: bool = True
    metadata: Mapping[str, float] = field(default_factory=dict)


def angular_frequency(frequency_hz) -> np.ndarray:
    return 2.0 * np.pi * np.atleast_1d(np.asarray(frequency_hz, dtype=float))


def pore_relaxation_time(radius_m, diffusion_m2_s: float) -> np.ndarray:
    return np.asarray(radius_m, dtype=float) ** 2 / (2.0 * diffusion_m2_s)


def membrane_relaxation_time(length_m, diffusion_m2_s: float) -> np.ndarray:
    return np.asarray(length_m, dtype=float) ** 2 / (4.0 * diffusion_m2_s)


def stern_conductance(frequency_hz, radius_m: float, params: SIP2DParameters) -> np.ndarray:
    """Eq. 17 elementary pore conductance ``C_p*`` (S) for one radius."""

    iwt = 1j * angular_frequency(frequency_hz) * pore_relaxation_time(radius_m, params.stern_diffusion_m2_s)
    return params.surface_conductance_s * iwt / (1.0 + iwt)


def titov_bracket(frequency_hz, length_m: float, params: SIP2DParameters) -> np.ndarray:
    """``Z* / Z_dc`` of Titov's membrane impedance (Eq. 19)."""

    eta = params.membrane_polarizability
    if not 0.0 <= eta < 1.0:
        raise ValueError("membrane polarizability must satisfy 0 <= eta < 1")
    tau = membrane_relaxation_time(length_m, params.membrane_diffusion_m2_s)
    root = np.sqrt(1j * angular_frequency(frequency_hz) * tau)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        frac = (1.0 - np.exp(-2.0 * root)) / (2.0 * root)
    small = np.abs(root) < 1.0e-6
    if np.any(small):
        s = root[small]
        frac[small] = 1.0 - s + (2.0 / 3.0) * s**2 - (1.0 / 3.0) * s**3
    frac = np.where(np.isfinite(frac), frac, 1.0)
    return 1.0 - eta * (1.0 - frac)


def membrane_conductivity_increment(frequency_hz, length_m: float, params: SIP2DParameters) -> np.ndarray:
    """Throat-phase increment ``sigma_w (Z_dc / Z* - 1)`` (S/m)."""

    return params.water_conductivity_s_m * (1.0 / titov_bracket(frequency_hz, length_m, params) - 1.0)


def pore_increments(
    frequency_hz,
    regions: Mapping[int, RegionSpec],
    params: SIP2DParameters,
    *,
    pore_mode: PoreMode,
    region_areas_m2: Mapping[int, float] | None = None,
    global_dynamic_pore_size_m: float | None = None,
    psd_kinds: tuple[str, ...] | None = None,
) -> dict[int, np.ndarray]:
    """Eq. 12 water-phase increments for every region, shape ``(n_frequency,)``.

    In uniform mode the size distribution (Eq. 9) is area weighted over the
    Stern-polarized regions; ``psd_kinds=("pore_body",)`` restricts it to pore
    bodies, as Niu et al. (2020) used the pore-node size distribution only.
    """

    freq = np.atleast_1d(np.asarray(frequency_hz, dtype=float))
    if pore_mode == "regional":
        return {
            label: (
                2.0 * stern_conductance(freq, spec.relaxation_radius_m, params) / spec.dynamic_pore_size_m
                if spec.stern_polarization
                else np.zeros(freq.shape, dtype=np.complex128)
            )
            for label, spec in regions.items()
        }
    if pore_mode != "uniform":
        raise ValueError("pore_mode must be 'regional' or 'uniform'")
    if region_areas_m2 is None or global_dynamic_pore_size_m is None:
        raise ValueError("uniform mode needs region areas and the global dynamic pore size")
    polarized = [
        label
        for label, spec in regions.items()
        if spec.stern_polarization and (psd_kinds is None or spec.kind in psd_kinds)
    ]
    total_area = sum(region_areas_m2[label] for label in polarized)
    conductance = sum(
        region_areas_m2[label] / total_area * stern_conductance(freq, regions[label].relaxation_radius_m, params)
        for label in polarized
    )
    increment = 2.0 * conductance / global_dynamic_pore_size_m
    return {label: increment for label in regions}


def membrane_increments(frequency_hz, regions: Mapping[int, RegionSpec], params: SIP2DParameters) -> dict[int, np.ndarray]:
    freq = np.atleast_1d(np.asarray(frequency_hz, dtype=float))
    return {
        label: (
            membrane_conductivity_increment(freq, spec.membrane_length_m, params)
            if spec.membrane_length_m is not None
            else np.zeros(freq.shape, dtype=np.complex128)
        )
        for label, spec in regions.items()
    }


def phase_values(
    mechanism: Mechanism,
    frequency_hz,
    regions: Mapping[int, RegionSpec],
    params: SIP2DParameters,
    *,
    pore: Mapping[int, np.ndarray] | None = None,
    membrane: Mapping[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Complex conductivity of the solid and of every pore region vs frequency."""

    if mechanism not in MECHANISMS:
        raise ValueError(f"mechanism must be one of {MECHANISMS}")
    omega = angular_frequency(frequency_hz)
    zero = np.zeros(omega.shape, dtype=np.complex128)
    with_dielectric = mechanism in ("interfacial", "all")
    solid = (
        params.solid_conductivity_s_m + 1j * omega * params.solid_permittivity_f_m
        if with_dielectric
        else zero.copy()
    )
    water_base = params.water_conductivity_s_m + (1j * omega * params.water_permittivity_f_m if with_dielectric else 0.0)
    water: dict[int, np.ndarray] = {}
    for label in regions:
        value = water_base + zero
        if mechanism in ("pore", "all"):
            value = value + pore[label]
        if mechanism in ("membrane", "all"):
            value = value + membrane[label]
        water[label] = value
    return solid, water


def conductivity_map(region_map: np.ndarray, solid_value: complex, water_values: Mapping[int, complex]) -> np.ndarray:
    """Look up a complex conductivity map from a region-label image (0 = solid)."""

    labels = np.asarray(region_map)
    lut = np.zeros(int(labels.max()) + 1, dtype=np.complex128)
    lut[SOLID_LABEL] = solid_value
    for label, value in water_values.items():
        lut[label] = value
    unknown = np.setdiff1d(np.unique(labels), [SOLID_LABEL, *water_values.keys()])
    if unknown.size:
        raise ValueError(f"region map contains labels without a RegionSpec: {unknown[:10]}")
    return lut[labels]
