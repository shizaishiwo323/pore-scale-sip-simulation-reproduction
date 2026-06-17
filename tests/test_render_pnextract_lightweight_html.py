from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "pore_network" / "render_pnextract_lightweight_html.py"


def load_module():
    spec = importlib.util.spec_from_file_location("render_pnextract_lightweight_html", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    pores = pd.DataFrame(
        {
            "pore_id": [1, 2, 3],
            "pore_center_x_m": [0.0, 2.8e-6, 5.6e-6],
            "pore_center_y_m": [0.0, 0.0, 2.8e-6],
            "pore_center_z_m": [0.0, 2.8e-6, 2.8e-6],
            "pore_radius_m": [2.8e-6, 5.6e-6, 8.4e-6],
        }
    )
    throats = pd.DataFrame(
        {
            "pore1_id": [1, 2, -1],
            "pore2_id": [2, 3, 1],
            "throat_radius_m": [1.4e-6, 5.6e-6, 5.6e-6],
        }
    )
    return pores, throats


def test_lightweight_payload_keeps_full_topology_without_mesh_triangles():
    module = load_module()
    pores, throats = sample_tables()

    payload = module.build_lightweight_payload(pores, throats, voxel_size_m=2.8e-6)

    assert payload["renderer"] == "canvas-2d-lightweight"
    assert payload["counts"] == {"pores": 3, "throats_total": 3, "throats_rendered": 2}
    assert payload["pores"][1] == [1.0, 0.0, 1.0, 2.0]
    assert payload["edges"] == [[0, 1, 0.5], [1, 2, 2.0]]
    assert "segments" not in payload
    assert "triangles" not in payload


def test_write_html_embeds_payload_and_metadata(tmp_path):
    module = load_module()
    pores, throats = sample_tables()
    payload = module.build_lightweight_payload(pores, throats, voxel_size_m=2.8e-6)
    html_out = tmp_path / "network_light.html"
    metadata_out = tmp_path / "network_light_metadata.json"

    module.write_lightweight_html(
        payload,
        html_out=html_out,
        metadata_out=metadata_out,
        source_pores_csv=Path("pores.csv"),
        source_throats_csv=Path("throats.csv"),
    )

    html = html_out.read_text(encoding="utf-8")
    metadata = json.loads(metadata_out.read_text(encoding="utf-8"))
    assert "const NETWORK_PAYLOAD =" in html
    assert "canvas-2d-lightweight" in html
    assert "<script src=" not in html
    assert "Plotly.newPlot" not in html
    assert metadata["renderer"] == "canvas-2d-lightweight"
    assert metadata["pores"] == 3
    assert metadata["throats_rendered"] == 2
