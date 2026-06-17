from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "pore_network" / "build_patched_pnextract_contact_split.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_patched_pnextract_contact_split", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_inject_contact_split_export_replaces_throat_writers_once():
    module = load_module()
    source = """
void  blockNetwork::writePNM() const
{
	const double dx = cg.vxlSize;
	cout<<"Writing throats";cout.flush();

	{ ///### write  _link1.dat file
		FILE* fil = fopen((cg.name() + "_link1.dat").c_str(), "w");
		fprintf(fil, "%6d\\n", int(throatIs.size()));
		for (int ti = 0; ti < int(throatIs.size()); ++ti)  {	const throatNE& tr = *throatIs[ti];

			fprintf(fil, "%6d %6d %6d %E %E %E\\n", ti+1, int(tr.e1-1), int(tr.e2-1),
						  tr.radius()*dx, t_shapeFacts[ti], t_lengthP1toP2s[ti]*dx);
		}
		fclose(fil);
	}

	{///### write  _link2.dat file
		FILE* fil = fopen((cg.name() + "_link2.dat").c_str(), "w");
		for (int ti = 0; ti < int(throatIs.size()); ++ti)  {	const throatNE& tr = *throatIs[ti];

			fprintf(fil, "%6d %6d %6d %E %E %E %E %E\\n", ti+1,  int(tr.e1-1), int(tr.e2-1),
			  t_lp1s[ti]*dx, t_lp2s[ti]*dx, t_ltrot[ti]*dx, tr.volumn*dx*dx*dx, 0.);
		}
		fclose(fil);
	}
	cout<<".\\n";cout.flush();
}
"""

    patched = module.inject_contact_split_export(source, Path("blockNet_write_cnm.cpp"))

    assert "ContactSplitRecord" in patched
    assert "contact patch split throats" in patched
    assert "int(throatIs.size()))" not in patched
    assert "contact_patch_split_diagnostics.csv" in patched
    assert "contactSplitRecords.size()" in patched


def test_contact_split_export_uses_physical_length_floor_and_conserves_conductance_terms():
    module = load_module()

    block = module.contact_split_export_block()

    assert "minThroatLengthVoxels = 0.25" in block
    assert "originalAreaOverLength" in block
    assert "targetAreaOverLength" in block
    assert "targetArea" in block
    assert "1.e-6" not in block


def test_copy_patch_refuses_existing_output_and_preserves_vendor(tmp_path: Path):
    module = load_module()
    vendor = tmp_path / "vendor"
    source_dir = vendor / "src" / "pnm" / "pnextract"
    source_dir.mkdir(parents=True)
    target = source_dir / "blockNet_write_cnm.cpp"
    original = Path(module.DEFAULT_VENDOR_ROOT / "src" / "pnm" / "pnextract" / "blockNet_write_cnm.cpp").read_text(encoding="utf-8")
    target.write_text(original, encoding="utf-8")
    out = tmp_path / "patched"

    summary = module.copy_and_patch_pnextract_source(vendor, out)

    patched_text = Path(summary["patched_file"]).read_text(encoding="utf-8")
    assert "ContactSplitRecord" in patched_text
    assert "contact_patch_split_diagnostics.csv" in patched_text
    assert target.read_text(encoding="utf-8") == original
    assert (out / "contact_split_patch_summary.json").exists()
