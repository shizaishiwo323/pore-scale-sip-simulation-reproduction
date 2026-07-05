from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "sip_simulation"
sys.path.insert(0, str(SCRIPT_DIR))


def test_project_extracted_components_require_dynamic_pore_size_manifest():
    from make_polarization_component_spectra import resolve_component_parameters

    with pytest.raises(ValueError, match="dynamic pore size manifest"):
        resolve_component_parameters("project-extracted", None)


def test_project_extracted_components_record_manifest_lambda(tmp_path):
    from make_polarization_component_spectra import resolve_component_parameters

    manifest = tmp_path / "dynamic_pore_size.json"
    manifest.write_text('{"lambda_iso_m": 5.5e-6}', encoding="utf-8")

    params, metadata = resolve_component_parameters("project-extracted", manifest)

    assert params.dynamic_pore_size_m == pytest.approx(5.5e-6)
    assert metadata["dynamic_pore_size_manifest"] == str(manifest)
    assert metadata["paper_reference_used_as_project_value"] is False
