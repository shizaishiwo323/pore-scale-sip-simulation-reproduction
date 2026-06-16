from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "sip_simulation" / "niu2020_berea_ac3d_reproduction_config.py"


def load_config():
    spec = importlib.util.spec_from_file_location("niu2020_berea_ac3d_reproduction_config", CONFIG_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_niu2020_config_uses_paper_table1_values():
    config = load_config()
    cfg = config.NIU2020_BEREA_CONFIG

    assert cfg.ct.shape_zyx == (350, 350, 350)
    assert cfg.ct.voxel_size_um == 2.8
    assert cfg.petrophysical.porosity_fraction == 0.202
    assert cfg.petrophysical.formation_factor == 16.3
    assert cfg.petrophysical.mean_pore_size_um == 35.0
    assert cfg.petrophysical.specific_surface_area_m2_g == 0.5
    assert cfg.materials.water_conductivity_s_m == 0.043
    assert cfg.materials.solid_conductivity_s_m == 0.0
    assert cfg.materials.water_relative_permittivity == 80.0
    assert cfg.materials.solid_relative_permittivity == 7.0
    assert cfg.polarization.surface_conductance_s == 1.3e-9
    assert cfg.polarization.diffusion_coefficient_m2_s == 1.3e-9
    assert cfg.polarization.dynamic_pore_size_m == 2.7e-6
    assert cfg.polarization.membrane_polarizability == 0.01


def test_niu2020_config_documents_mechanism_definitions_and_input_limits():
    config = load_config()
    cfg = config.NIU2020_BEREA_CONFIG

    assert set(cfg.mechanisms.definitions) == {"interfacial", "pore", "membrane", "all"}
    assert "solid phase is zero" in cfg.mechanisms.definitions["pore"]
    assert "solid phase is zero" in cfg.mechanisms.definitions["membrane"]
    assert cfg.input_policy.allowed_experiment_workbooks == ("Figure6.xlsx", "Figure8.xlsx")
    assert cfg.input_policy.paper_simulation_columns_used_as_simulation is False
