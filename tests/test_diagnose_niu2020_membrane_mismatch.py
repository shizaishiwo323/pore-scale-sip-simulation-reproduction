import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "scripts" / "sip_simulation"))

from diagnose_niu2020_membrane_mismatch import grid_search_membrane_scales  # noqa: E402


def test_grid_search_membrane_scales_recovers_synthetic_scale():
    frequencies = np.logspace(0, 4, 25)
    length_scales = np.array([0.1, 0.2, 0.4])
    zdc_scales = np.array([2.0, 5.0, 10.0])

    def model(length_scale: float, zdc_scale: float) -> np.ndarray:
        peak = 100.0 / length_scale**2
        response = (frequencies / peak) / (1.0 + (frequencies / peak) ** 2)
        return response / zdc_scale

    target = model(0.2, 5.0)
    result = grid_search_membrane_scales(frequencies, target, model, length_scales, zdc_scales)

    assert result["best_length_scale"] == 0.2
    assert result["best_zdc_scale"] == 5.0
    assert result["best_score"] == 0.0
