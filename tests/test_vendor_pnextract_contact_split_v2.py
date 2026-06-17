from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENDOR_WRITE_CNM = (
    PROJECT_ROOT
    / "code"
    / "vendor"
    / "pnextract"
    / "src"
    / "pnm"
    / "pnextract"
    / "blockNet_write_cnm.cpp"
)


def test_vendor_pnextract_uses_original_aggregated_throat_export():
    source = VENDOR_WRITE_CNM.read_text(encoding="utf-8")

    assert "ContactSplitRecord" not in source
    assert "contact patch split throats" not in source
    assert "contact_patch_split_diagnostics.csv" not in source
    assert "fprintf(fil, \"%6d\\n\", int(throatIs.size()))" in source
    assert "tr.radius()*dx" in source
    assert "t_ltrot[ti]*dx" in source
