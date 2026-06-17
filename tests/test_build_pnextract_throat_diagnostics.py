from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "pore_network" / "build_patched_pnextract_throat_diagnostics.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_patched_pnextract_throat_diagnostics", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def minimal_blocknet_write_text() -> str:
    return "\n".join(
        [
            "void  blockNetwork::writePNM() const",
            "{",
            "\tvector<double> t_radiuss(nTrots,0.);//",
            "\tvector<double> t_shapeFacts(nTrots,0.);//",
            "\tvector<double> t_lengthP1toP2s(nTrots,0.);",
            "\tvector<double> t_lp1s(nTrots,0.);//",
            "\tvector<double> t_lp2s(nTrots,0.);//",
            "\tvector<double> t_ltrot(nTrots,0.); // throat portion of t_lengthP1toP2s",
            "\tfor (int ti=0; ti<nTrots; ++ti)  {",
            "\t    throatNE& tr = *throatIs[ti];",
            "\t\tdouble lpt1 = 1;",
            "\t\tdouble rp1 = 2;",
            "\t\tdouble lpt2 = 3;",
            "\t\tdouble rp2 = 4;",
            "\t\tdouble rr = 1;",
            "\t\tdouble lengthP1toP2 = lpt1+lpt2;",
            "\t\tif (lengthP1toP2 < 3.) \tlengthP1toP2 = 3.01, ++lengthP1toP2Warnings;",
            "\t\tlp1 = lpt1*0.67;",
            "\t\tlp2 = lpt2*0.67;",
            "\t\tif (tr.e1 < 2 )\tlp1 = 1;",
            "\t\tif (tr.e2 < 2 )\tlp2 = 1;",
            "\t\tlthroat = lengthP1toP2-lp1-lp2;",
            "\t\tif (lthroat < 0.0000001) \tlthroat = 1;",
            "\t\tt_lengthP1toP2s[ti] = lengthP1toP2*1.;",
            "\t\tt_lp1s[ti] = lp1*1.;",
            "\t\tt_lp2s[ti] = lp2*1.;",
            "\t\tt_ltrot[ti] = lthroat*1;",
            "\t}",
            "\tcout<<  \" P1-to-P2 length < 3    for \" <<lengthP1toP2Warnings<<\" throats\"<<endl;",
            "\tcout<<\" shapefactor: belowAllowedG \"<<nBelowAllowedG/totalArea*100<<\"%   aboveAllowedG \"<<nAboveAllowedG/totalArea*100<<\"%\"<<endl;",
            "\tcout<<\" checkSumAt: \"<<checkSumAt<<endl;",
            "}",
            "",
        ]
    )


def test_patch_adds_csv_diagnostics_without_modifying_source(tmp_path):
    module = load_module()
    source = tmp_path / "vendor"
    copied = tmp_path / "copied"
    cpp = source / "src" / "pnm" / "pnextract" / "blockNet_write_cnm.cpp"
    cpp.parent.mkdir(parents=True)
    original = minimal_blocknet_write_text()
    cpp.write_text(original, encoding="utf-8")

    summary = module.copy_and_patch_pnextract_source(source, copied)

    assert cpp.read_text(encoding="utf-8") == original
    patched = (copied / "src" / "pnm" / "pnextract" / "blockNet_write_cnm.cpp").read_text(encoding="utf-8")
    assert "diag_raw_lthroat_before_floor" in patched
    assert "_throat_length_diagnostics.csv" in patched
    assert "raw_lthroat_voxels" in patched
    assert "center_minus_radii_voxels" in patched
    assert 'throat_volume_voxels3\\n");' in patched
    assert 'throat_volume_voxels3\\\\n");' not in patched
    assert "diagRawLt1" in patched
    assert summary["diagnostics"] == "throat_length_export"
    assert summary["source_root"] == str(source)
    assert summary["patched_root"] == str(copied)


def test_patch_can_optionally_apply_dong_blunt_formula(tmp_path):
    module = load_module()
    source = tmp_path / "vendor"
    copied = tmp_path / "copied"
    cpp = source / "src" / "pnm" / "pnextract" / "blockNet_write_cnm.cpp"
    cpp.parent.mkdir(parents=True)
    cpp.write_text(minimal_blocknet_write_text(), encoding="utf-8")

    module.copy_and_patch_pnextract_source(source, copied, beta=0.625, min_throat_length_voxels=0.25)

    patched = (copied / "src" / "pnm" / "pnextract" / "blockNet_write_cnm.cpp").read_text(encoding="utf-8")
    assert "const double poreThroatSegmentationBeta = 0.625" in patched
    assert 'cg.getOr("minThroatLengthVoxels", 0.25)' in patched
    assert "diag_raw_lthroat_before_floor[ti] = lthroat;" in patched
