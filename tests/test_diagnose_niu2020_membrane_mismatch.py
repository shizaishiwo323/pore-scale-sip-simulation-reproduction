import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "scripts" / "sip_simulation"))

from diagnose_niu2020_membrane_mismatch import retired_membrane_mismatch_report  # noqa: E402


def test_membrane_mismatch_diagnostic_is_retired_in_favor_of_reextracting_geometry():
    report = retired_membrane_mismatch_report()

    assert report["status"] == "retired"
    assert report["post_extraction_geometry_scaling_allowed"] is False
    assert str(report["default_pnextract_executable"]).endswith("pnextract.exe")
    assert "Re-run the original pnextract algorithm" in report["reason"]
