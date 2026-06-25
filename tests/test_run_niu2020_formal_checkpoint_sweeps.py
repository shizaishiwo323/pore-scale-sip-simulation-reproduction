from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "sip_simulation" / "run_niu2020_formal_checkpoint_sweeps.py"


def load_module():
    spec = importlib.util.spec_from_file_location("run_niu2020_formal_checkpoint_sweeps", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_tasks_uses_formal_complex128_fft_pore_settings(tmp_path):
    module = load_module()

    tasks = module.build_tasks(
        result_dir=tmp_path,
        mechanisms=("interfacial", "pore"),
        directions=("x", "z"),
        frequencies=(1e3,),
    )

    assert [task.run_name for task in tasks] == [
        "chk_interfacial_x_1e3_c128",
        "chk_interfacial_z_1e3_c128",
        "chk_pore_x_1e3_c128",
        "chk_pore_z_1e3_c128",
    ]
    command = tasks[0].command
    assert "--frequency-match-mode" in command
    assert "exact" in command
    assert "--dtype" in command
    assert "complex128" in command
    assert "--preconditioner" in command
    assert "fft" in command
    assert "--fft-reference" in command
    assert "pore" in command
    assert "--rtol" in command
    assert "1e-5" in command
    assert str(tmp_path / "source_data" / "formal_inputs" / "component_spectra_checkpoints" / "polarization_spectra_interfacial.csv") in command


def test_successful_sweep_detection_requires_true_residual_pass(tmp_path):
    module = load_module()
    csv_path = tmp_path / "sweep_results.csv"

    pd.DataFrame({"info": [0], "true_residual_passed": [False]}).to_csv(csv_path, index=False)
    assert not module.is_successful_sweep(csv_path)

    pd.DataFrame({"info": [0], "true_residual_passed": [True]}).to_csv(csv_path, index=False)
    assert module.is_successful_sweep(csv_path)

    pd.DataFrame({"info": [1000], "true_residual_passed": [True]}).to_csv(csv_path, index=False)
    assert not module.is_successful_sweep(csv_path)
