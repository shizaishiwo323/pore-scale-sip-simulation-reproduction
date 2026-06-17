from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "pore_network" / "build_patched_pnextract_dong_blunt.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_patched_pnextract_dong_blunt", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_patch_replaces_fixed_fraction_with_dong_blunt_radius_ratio_formula(tmp_path):
    module = load_module()
    source = tmp_path / "source"
    copied = tmp_path / "copied"
    cpp = source / "src" / "pnm" / "pnextract" / "blockNet_write_cnm.cpp"
    cpp.parent.mkdir(parents=True)
    original_text = "\n".join(
        [
            "double lengthP1toP2 = lpt1+lpt2;",
            "lp1 = lpt1*0.67;",
            "lp2 = lpt2*0.67;",
            "if (tr.e1 < 2 )\tlp1 = 1;",
            "if (tr.e2 < 2 )\tlp2 = 1;",
            "lthroat = lengthP1toP2-lp1-lp2;",
            "if (lthroat < 0.0000001) \tlthroat = 1;",
            "",
        ]
    )
    cpp.write_text(original_text, encoding="utf-8")

    summary = module.copy_and_patch_pnextract_source(source, copied, beta=0.625, min_throat_length_voxels=0.25)

    assert cpp.read_text(encoding="utf-8") == original_text
    patched = (copied / "src" / "pnm" / "pnextract" / "blockNet_write_cnm.cpp").read_text(encoding="utf-8")
    assert "const double poreThroatSegmentationBeta = 0.625" in patched
    assert "lp1 = lpt1 * (1.0 - poreThroatSegmentationBeta * rr / rp1);" in patched
    assert "lp2 = lpt2 * (1.0 - poreThroatSegmentationBeta * rr / rp2);" in patched
    assert "std::min(std::max(lp1, 0.0), lpt1)" in patched
    assert 'const double minThroatLengthVoxels = cg.getOr("minThroatLengthVoxels", 0.25);' in patched
    assert "lthroat = std::max(lthroat, minThroatLengthVoxels);" in patched
    assert "lp1 = lpt1*0.67;" not in patched
    assert "lthroat = 1;" not in patched
    assert summary["beta"] == 0.625
    assert summary["min_throat_length_voxels"] == 0.25
    assert summary["source_root"] == str(source)
    assert summary["patched_root"] == str(copied)


def test_build_command_uses_patched_source_root(tmp_path):
    module = load_module()
    patched = tmp_path / "patched"
    build_dir = tmp_path / "build"
    exe = tmp_path / "out" / "pnextract.exe"
    cmd = module.build_compile_command(patched, build_dir, exe, cxx="g++")
    joined = " ".join(cmd)

    assert str(patched / "src" / "pnm" / "pnextract" / "blockNet.cpp") in joined
    assert str(patched / "src" / "libvoxel" / "voxelImage.cpp") in joined
    assert str(exe) in joined


def test_scan_beta_helpers_include_endpoint():
    script = PROJECT_ROOT / "code" / "scripts" / "pore_network" / "scan_dong_blunt_beta_niu2020.py"
    spec = importlib.util.spec_from_file_location("scan_dong_blunt_beta_niu2020", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.beta_values(0.50, 0.55, 0.025) == [0.5, 0.525, 0.55]
    assert module.beta_tag(0.625) == "beta_0p625"
