#!/usr/bin/env python3
"""Build a copied pnextract with contact-patch split throat export.

The vendored pnextract source is treated as read-only. This helper copies it
and patches only the copied ``blockNet_write_cnm.cpp`` so that Statoil link
files contain one diagnostic throat per disconnected contact patch.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VENDOR_ROOT = PROJECT_ROOT / "code" / "vendor" / "pnextract"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "results" / "niu2020_berea_contact_split_pnextract_build_v1"

sys.path.insert(0, str(SCRIPT_DIR))
from build_patched_pnextract_dong_blunt import build_compile_command, run_build  # noqa: E402


LINK_WRITER_START = '\t{ ///### write  _link1.dat file'
LINK_WRITER_END = '\tcout<<".\\n";cout.flush();'


def contact_split_export_block() -> str:
    return r'''
	struct ContactSplitRecord {
		int originalTi;
		int e1;
		int e2;
		double radius;
		double shapeFactor;
		double lengthP1toP2;
		double lp1;
		double lp2;
		double lthroat;
		double volume;
		int patchIndex;
		int patchCount;
		int faceCount;
	};
	vector<ContactSplitRecord> contactSplitRecords;
	contactSplitRecords.reserve(throatIs.size()*2);

	for (int ti = 0; ti < int(throatIs.size()); ++ti) {
		const throatNE& tr = *throatIs[ti];
		vector<voxel*> patchVoxels;
		patchVoxels.reserve(tr.toxels1.size()+tr.toxels2.size());
		for (auto vi: tr.toxels1) if (vi) patchVoxels.push_back(vi);
		for (auto vi: tr.toxels2) if (vi) patchVoxels.push_back(vi);

		const bool canSplitInternal = (tr.e1 >= 2 && tr.e2 >= 2 && patchVoxels.size() > 1);
		vector<vector<int> > components;
		if (canSplitInternal) {
			vector<int> visited(patchVoxels.size(), 0);
			for (int start = 0; start < int(patchVoxels.size()); ++start) {
				if (visited[start]) continue;
				vector<int> stack;
				vector<int> component;
				stack.push_back(start);
				visited[start] = 1;
				while (!stack.empty()) {
					int current = stack.back();
					stack.pop_back();
					component.push_back(current);
					voxel* vc = patchVoxels[current];
					for (int other = 0; other < int(patchVoxels.size()); ++other) {
						if (visited[other]) continue;
						voxel* vo = patchVoxels[other];
						int di = std::abs(vc->i - vo->i);
						int dj = std::abs(vc->j - vo->j);
						int dk = std::abs(vc->k - vo->k);
						if (di <= 1 && dj <= 1 && dk <= 1 && (di + dj + dk) <= 2) {
							visited[other] = 1;
							stack.push_back(other);
						}
					}
				}
				components.push_back(component);
			}
		}
		if (components.empty()) {
			components.push_back(vector<int>());
			for (int i = 0; i < int(patchVoxels.size()); ++i) components.back().push_back(i);
		}

		const double minThroatLengthVoxels = 0.25;
		const int totalFaces = std::max(1, int(patchVoxels.size()));
		const double originalShapeFactor = std::max(t_shapeFacts[ti], 1.e-12);
		const double originalLength = std::max(t_ltrot[ti], minThroatLengthVoxels);
		const double originalArea = t_radiuss[ti]*t_radiuss[ti]/4./originalShapeFactor;
		const double originalAreaOverLength = originalArea/originalLength;
		const int patchCount = int(components.size());
		for (int ci = 0; ci < patchCount; ++ci) {
			const vector<int>& comp = components[ci];
			double centroidI = 0., centroidJ = 0., centroidK = 0.;
			double maxR = 0.5;
			for (int idx: comp) {
				voxel* vi = patchVoxels[idx];
				centroidI += vi->i + 0.5;
				centroidJ += vi->j + 0.5;
				centroidK += vi->k + 0.5;
				maxR = std::max(maxR, double(vi->R));
			}
			const int faceCount = std::max(1, int(comp.size()));
			centroidI /= faceCount;
			centroidJ /= faceCount;
			centroidK /= faceCount;

			double splitLength = t_ltrot[ti];
			double splitTotalLength = t_lengthP1toP2s[ti];
			double splitLp1 = t_lp1s[ti];
			double splitLp2 = t_lp2s[ti];
			double splitRadius = t_radiuss[ti];
			double splitShapeFactor = t_shapeFacts[ti];
			if (canSplitInternal) {
				const poreNE* p1 = poreIs[tr.e1];
				const poreNE* p2 = poreIs[tr.e2];
				double d1 = std::sqrt((centroidI-p1->mb->fi)*(centroidI-p1->mb->fi)
				                    +(centroidJ-p1->mb->fj)*(centroidJ-p1->mb->fj)
				                    +(centroidK-p1->mb->fk)*(centroidK-p1->mb->fk));
				double d2 = std::sqrt((centroidI-p2->mb->fi)*(centroidI-p2->mb->fi)
				                    +(centroidJ-p2->mb->fj)*(centroidJ-p2->mb->fj)
				                    +(centroidK-p2->mb->fk)*(centroidK-p2->mb->fk));
				double rp1 = std::max(double(p1->mb->R), 1.0);
				double rp2 = std::max(double(p2->mb->R), 1.0);
				splitLp1 = std::max(d1-rp1, 0.);
				splitLp2 = std::max(d2-rp2, 0.);
				splitLength = std::max(d1+d2-rp1-rp2, minThroatLengthVoxels);
				splitTotalLength = splitLp1 + splitLp2 + splitLength;
				const double contactWeight = double(faceCount)/double(totalFaces);
				const double targetAreaOverLength = originalAreaOverLength * contactWeight;
				const double targetArea = std::max(targetAreaOverLength * splitLength, 1.e-12);
				splitShapeFactor = std::min(0.079, std::max(0.01, originalShapeFactor));
				splitRadius = std::sqrt(4.0 * splitShapeFactor * targetArea);
				splitRadius = std::max(splitRadius, 0.5);
			}

			ContactSplitRecord rec;
			rec.originalTi = ti;
			rec.e1 = tr.e1;
			rec.e2 = tr.e2;
			rec.radius = splitRadius;
			rec.shapeFactor = splitShapeFactor;
			rec.lengthP1toP2 = splitTotalLength;
			rec.lp1 = splitLp1;
			rec.lp2 = splitLp2;
			rec.lthroat = splitLength;
			rec.volume = tr.volumn * double(faceCount) / double(totalFaces);
			rec.patchIndex = ci + 1;
			rec.patchCount = patchCount;
			rec.faceCount = faceCount;
			contactSplitRecords.push_back(rec);
		}
	}

	cout<<"Writing contact patch split throats";cout.flush();
	{ ///### write  _link1.dat file
		FILE* fil = fopen((cg.name() + "_link1.dat").c_str(), "w");
		fprintf(fil, "%6d\n", int(contactSplitRecords.size()));
		for (int si = 0; si < int(contactSplitRecords.size()); ++si) {
			const ContactSplitRecord& rec = contactSplitRecords[si];
			fprintf(fil, "%6d %6d %6d %E %E %E\n", si+1, int(rec.e1-1), int(rec.e2-1),
						  rec.radius*dx, rec.shapeFactor, rec.lengthP1toP2*dx);
		}
		fclose(fil);
	}

	{///### write  _link2.dat file
		FILE* fil = fopen((cg.name() + "_link2.dat").c_str(), "w");
		for (int si = 0; si < int(contactSplitRecords.size()); ++si) {
			const ContactSplitRecord& rec = contactSplitRecords[si];
			fprintf(fil, "%6d %6d %6d %E %E %E %E %E\n", si+1, int(rec.e1-1), int(rec.e2-1),
			  rec.lp1*dx, rec.lp2*dx, rec.lthroat*dx, rec.volume*dx*dx*dx, 0.);
		}
		fclose(fil);
	}
	cout<<" ("<<contactSplitRecords.size()<<" split records from "<<throatIs.size()<<" aggregated throats)."<<"\n";cout.flush();

	{
		FILE* fil = fopen((cg.name() + "_contact_patch_split_diagnostics.csv").c_str(), "w");
		fprintf(fil, "split_throat_id,original_throat_id,pore1_id,pore2_id,patch_index,patch_count,face_count,length_voxels,radius_voxels,shape_factor,volume_voxels3\n");
		for (int si = 0; si < int(contactSplitRecords.size()); ++si) {
			const ContactSplitRecord& rec = contactSplitRecords[si];
			fprintf(fil, "%d,%d,%d,%d,%d,%d,%d,%.12g,%.12g,%.12g,%.12g\n",
				si+1, rec.originalTi+1, int(rec.e1-1), int(rec.e2-1), rec.patchIndex, rec.patchCount,
				rec.faceCount, rec.lthroat, rec.radius, rec.shapeFactor, rec.volume);
		}
		fclose(fil);
	}
'''


def inject_contact_split_export(text: str, target: Path) -> str:
    if "ContactSplitRecord" in text:
        return text
    start = text.find(LINK_WRITER_START)
    if start < 0:
        raise RuntimeError(f"could not find link writer start marker in {target}")
    end = text.find(LINK_WRITER_END, start)
    if end < 0:
        raise RuntimeError(f"could not find link writer end marker in {target}")
    end += len(LINK_WRITER_END)
    return text[:start] + contact_split_export_block() + "\n" + text[end:]


def copy_and_patch_pnextract_source(source_root: Path, patched_root: Path) -> dict[str, object]:
    if patched_root.exists():
        raise FileExistsError(f"patched_root already exists; choose a new output directory: {patched_root}")
    if not source_root.exists():
        raise FileNotFoundError(source_root)

    shutil.copytree(source_root, patched_root)
    target = patched_root / "src" / "pnm" / "pnextract" / "blockNet_write_cnm.cpp"
    text = target.read_text(encoding="utf-8")
    target.write_text(inject_contact_split_export(text, target), encoding="utf-8")
    summary = {
        "source_root": str(source_root),
        "patched_root": str(patched_root),
        "patched_file": str(target),
        "diagnostics": "contact_patch_split_export",
        "notes": [
            "The source pnextract tree is not modified by this builder.",
            "The copied writePNM exports split throat records in link1/link2 and a contact_patch_split_diagnostics.csv.",
            "Node files retain original pore centers/radii; downstream SIP spectra use the split throat rows.",
        ],
    }
    (patched_root / "contact_split_patch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-root", default=str(DEFAULT_VENDOR_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    tag = "contact_patch_split"
    patched_root = out_root / tag / "source"
    build_dir = out_root / tag / "build"
    exe_suffix = ".exe" if sys.platform.startswith("win") else ""
    exe_path = build_dir / f"pnextract_{tag}{exe_suffix}"

    summary = copy_and_patch_pnextract_source(Path(args.vendor_root), patched_root)
    build_dir.mkdir(parents=True, exist_ok=True)
    command = build_compile_command(patched_root, build_dir, exe_path, cxx=args.cxx)
    summary.update({"build_dir": str(build_dir), "exe_path": str(exe_path), "compile_command": command, "built": False})
    if not args.skip_build:
        result = run_build(command, cwd=PROJECT_ROOT)
        summary.update(
            {
                "built": True,
                "compile_stdout_tail": result.stdout[-4000:],
                "compile_stderr_tail": result.stderr[-4000:],
            }
        )
    summary_path = out_root / tag / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
