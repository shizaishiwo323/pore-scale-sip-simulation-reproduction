"""Niu et al. (2020) Berea AC3D/SIP reproduction parameters.

This is an executable configuration file, not a result file.  Values below
come from Niu et al. (2020), Table 1 and Sections 4-5, unless a note says the
value is a project-side provenance or diagnostic choice.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PDF = (
    PROJECT_ROOT
    / "docs"
    / "references"
    / "JGR Solid Earth - 2020 - Niu - A Framework for Pore-Scale Simulation of Effective Electrical Conductivity and Permittivity.pdf"
)
DATA_DIR = PROJECT_ROOT / "data" / "Niu 2020data"


class CTConfig(NamedTuple):
    """Digital Berea sandstone input used by the AC3D finite-difference solve."""

    raw_path: Path
    tiff_path: Path
    shape_zyx: tuple[int, int, int]
    voxel_size_um: float
    voxel_size_m: float
    pore_label: int
    solid_label: int
    note: str


class MaterialConfig(NamedTuple):
    """Intrinsic phase properties assigned to water and nonconductive solid."""

    water_conductivity_s_m: float
    solid_conductivity_s_m: float
    water_relative_permittivity: float
    solid_relative_permittivity: float
    epsilon0_f_m: float
    note: str


class PetrophysicalConfig(NamedTuple):
    """Rock-scale properties reported in Table 1 for context and validation."""

    porosity_fraction: float
    specific_surface_area_m2_g: float
    formation_factor: float
    mean_pore_size_um: float
    note: str


class PolarizationConfig(NamedTuple):
    """Electrochemical polarization parameters used for pore and membrane terms."""

    surface_conductance_s: float
    diffusion_coefficient_m2_s: float
    dynamic_pore_size_m: float
    membrane_polarizability: float
    pore_relaxation_time_formula: str
    membrane_relaxation_time_formula: str
    upscaling_formula: str
    geometry_note: str


class FrequencyConfig(NamedTuple):
    """Frequency band of the paper experiment and simulation comparison."""

    min_hz: float
    max_hz: float
    paper_default_count: int
    local_available_sweep_count: int
    note: str


class MechanismConfig(NamedTuple):
    """Section 5.3 single-mechanism material assignments."""

    definitions: dict[str, str]
    plotting_labels: dict[str, str]


class InputPolicy(NamedTuple):
    """Allowed raw inputs for this reproduction step."""

    allowed_ct_files: tuple[str, ...]
    allowed_experiment_workbooks: tuple[str, ...]
    paper_simulation_columns_used_as_simulation: bool
    note: str


class DiagnosticCorrectionConfig(NamedTuple):
    """Project-side diagnostic correction, kept separate from paper parameters."""

    membrane_length_scale: float
    membrane_zdc_scale: float
    note: str


class Niu2020BereaConfig(NamedTuple):
    """Complete parameter bundle for the Berea SIP reproduction."""

    reference_pdf: Path
    data_dir: Path
    ct: CTConfig
    petrophysical: PetrophysicalConfig
    materials: MaterialConfig
    polarization: PolarizationConfig
    frequencies: FrequencyConfig
    mechanisms: MechanismConfig
    input_policy: InputPolicy
    diagnostic_correction: DiagnosticCorrectionConfig


# CT volume:
# - The project copy stores the Berea micro-CT as a 350 x 350 x 350 volume.
# - Project convention is little-endian uint16 RAW with labels 1=pore/water and
#   2=solid.  The TIFF copy is the same segmentation in image-stack form.
# - The physical voxel edge length is 2.8 um, matching the project AGENTS.md
#   route and the Niu AC3D 350^3 REV described in Section 4.
CT = CTConfig(
    raw_path=DATA_DIR / "microCT_Berea.raw",
    tiff_path=DATA_DIR / "microCT_Berea.tiff",
    shape_zyx=(350, 350, 350),
    voxel_size_um=2.8,
    voxel_size_m=2.8e-6,
    pore_label=1,
    solid_label=2,
    note="Berea segmented REV for the 350^3 AC3D solve; labels follow the local Niu data copy.",
)


# Table 1 petrophysical properties:
# - These values describe the laboratory Berea sample and are useful for
#   validation/provenance.  They are not identical to every voxel-count
#   statistic in the segmented RAW copy; keep those as separate data audits.
PETROPHYSICAL = PetrophysicalConfig(
    porosity_fraction=0.202,
    specific_surface_area_m2_g=0.5,
    formation_factor=16.3,
    mean_pore_size_um=35.0,
    note="Table 1 Berea sample properties; do not overwrite raw label-count porosity audits with this value.",
)


# Intrinsic phase properties:
# - sigma_w = 0.043 S/m is Table 1 water conductivity.
# - The solid phase is treated as nonconductive for Berea quartz-rich sandstone;
#   Section 1 notes nonmetallic mineral conductivity is negligible, and Section
#   5.3 sets solid conductivity to zero in pore/membrane single-mechanism runs.
# - eps_w = 80 eps0 and eps_s = 7 eps0 are Table 1 intrinsic permittivities.
# - eps0 = 8.85e-12 F/m is the vacuum permittivity listed in the Table 1 note.
MATERIALS = MaterialConfig(
    water_conductivity_s_m=0.043,
    solid_conductivity_s_m=0.0,
    water_relative_permittivity=80.0,
    solid_relative_permittivity=7.0,
    epsilon0_f_m=8.85e-12,
    note="Use water dc conductivity plus high-frequency dielectric storage; solid has zero dc conductivity.",
)


# Polarization model parameters:
# - Sigma_S = 1.3e-9 S, D = 1.3e-9 m^2/s, Lambda = 2.7 um, and eta0 = 1%
#   are Table 1 values.
# - Pore polarization uses pore radii from the pore-size distribution and
#   tau_p = r^2/(2D).
# - Membrane polarization uses pore-throat lengths and dc throat resistances;
#   Table 1/Section 4 says Zdc is calculated from throat geometry and water
#   conductivity, with tau_m = L^2/(4D).
# - The volumetric water-phase increment follows Delta sigma_w* = 2 C*/Lambda.
POLARIZATION = PolarizationConfig(
    surface_conductance_s=1.3e-9,
    diffusion_coefficient_m2_s=1.3e-9,
    dynamic_pore_size_m=2.7e-6,
    membrane_polarizability=0.01,
    pore_relaxation_time_formula="tau_p = r^2 / (2D)",
    membrane_relaxation_time_formula="tau_m = L^2 / (4D)",
    upscaling_formula="Delta sigma_w* = 2 C* / Lambda",
    geometry_note=(
        "Pore radii and throat lengths/resistances are distribution-driven; "
        "Figure5.xlsx contains paper pore/throat distributions, while full "
        "local membrane runs use pnextract-derived throat geometry."
    ),
)


# The paper title and Figures 6-7 state the comparison band as 1 mHz to 1 GHz.
# Figure8.xlsx contains 40 paper simulation points, but those paper simulation
# columns are not used as this project's reproduced simulation curves.
FREQUENCIES = FrequencyConfig(
    min_hz=1.0e-3,
    max_hz=1.0e9,
    paper_default_count=40,
    local_available_sweep_count=12,
    note="Local historical full350 GPU sweeps available here are representative 12-frequency sweeps.",
)


# Section 5.3 mechanism split:
# - Interfacial/Maxwell: assign only water/solid dc conductivity and intrinsic
#   high-frequency permittivity; no pore or membrane increment.
# - Pore: water phase is sigma_w + Delta sigma_pore*; solid phase is zero.
# - Membrane: water phase is sigma_w + Delta sigma_membrane*; solid phase is zero.
# - All: combine water dc conductivity, high-frequency permittivity, pore and
#   membrane increments; solid keeps high-frequency permittivity and zero dc.
MECHANISMS = MechanismConfig(
    definitions={
        "interfacial": "water and solid receive only dc conductivity and high-frequency permittivity; no pore or membrane increment",
        "pore": "water phase is sigma_w + Delta sigma_pore*; solid phase is zero",
        "membrane": "water phase is sigma_w + Delta sigma_membrane*; solid phase is zero",
        "all": "water includes sigma_w, high-frequency permittivity, Delta sigma_pore*, and Delta sigma_membrane*; solid includes high-frequency permittivity",
    },
    plotting_labels={
        "interfacial": "Dielectric / interfacial",
        "pore": "EDL / pore",
        "membrane": "Membrane",
        "all": "EDL + Dielectric + Membrane",
    },
)


INPUT_POLICY = InputPolicy(
    allowed_ct_files=("microCT_Berea.raw", "microCT_Berea.tiff"),
    allowed_experiment_workbooks=("Figure6.xlsx", "Figure8.xlsx"),
    paper_simulation_columns_used_as_simulation=False,
    note=(
        "Use Figure6/Figure8 only for experimental scatter.  Figure8 paper "
        "simulation, pore, membrane, and interfacial columns are paper-reference "
        "blocks and must not be plotted as this project's simulation output."
    ),
)


# This is not a paper parameter.  It records the explicit project diagnostic
# correction currently documented in AGENTS.md for matching the membrane
# relaxation when pnextract throat geometry differs from the authors' internal
# pore-throat/resistance definition.
DIAGNOSTIC_CORRECTION = DiagnosticCorrectionConfig(
    membrane_length_scale=0.0446683592150963,
    membrane_zdc_scale=10.0,
    note="Project diagnostic correction only; do not describe it as a hidden Niu et al. parameter.",
)


NIU2020_BEREA_CONFIG = Niu2020BereaConfig(
    reference_pdf=REFERENCE_PDF,
    data_dir=DATA_DIR,
    ct=CT,
    petrophysical=PETROPHYSICAL,
    materials=MATERIALS,
    polarization=POLARIZATION,
    frequencies=FREQUENCIES,
    mechanisms=MECHANISMS,
    input_policy=INPUT_POLICY,
    diagnostic_correction=DIAGNOSTIC_CORRECTION,
)
