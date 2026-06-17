#!/usr/bin/env python3
"""Prepare and optionally scan Berea preprocessing variants for pnextract."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_ROOT / "scripts" / "pore_network"))
sys.path.insert(0, str(CODE_ROOT / "scripts" / "sip_simulation"))

from parse_pnextract_network import main as parse_main  # noqa: F401,E402
from run_segmented_core_pnextract_ballstick import expected_pnextract_prefix, verify_pnextract_outputs  # noqa: E402
from scan_pnextract_for_niu2020_membrane import evaluate_network, read_figure5  # noqa: E402


DEFAULT_BINARY_TIFF = (
    PROJECT_ROOT
    / "results"
    / "niu2020_berea_reproduction"
    / "segmented_core"
    / "niu2020_berea_solid255_pore0.tiff"
)
DEFAULT_FIGURE5 = PROJECT_ROOT / "data" / "Niu 2020data" / "Figure5.xlsx"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "niu2020_berea_preprocess_sensitivity_v1"
DEFAULT_PNEXTRACT_EXE = (
    PROJECT_ROOT
    / "code"
    / "vendor"
    / "pnextract"
    / "bin"
    / "pnextract.exe"
)
DEFAULT_PNEXTRACT_LINES: list[str] = []
DEFAULT_VARIANTS = [
    "identity",
    "pore_dilate_1",
    "pore_erode_1",
    "pore_open_1",
    "pore_close_1",
    "solid_dilate_1",
    "solid_erode_1",
    "remove_small_pore_components_64",
]


def connectivity_structure() -> np.ndarray:
    return ndi.generate_binary_structure(rank=3, connectivity=1)


def parse_small_component_threshold(variant: str) -> int:
    prefix = "remove_small_pore_components_"
    if not variant.startswith(prefix):
        raise ValueError(f"not a remove-small-component variant: {variant}")
    threshold = int(variant[len(prefix) :])
    if threshold <= 0:
        raise ValueError("small component threshold must be positive")
    return threshold


def to_binary_pore0_solid1(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume)
    values = set(int(v) for v in np.unique(arr))
    if values.issubset({0, 1}):
        return arr.astype(np.uint8, copy=True)
    if values.issubset({0, 255}):
        return np.where(arr == 0, 0, 1).astype(np.uint8)
    raise ValueError(f"expected binary 0/1 or 0/255 volume, found values {sorted(values)[:10]}")


def apply_preprocess_variant(volume: np.ndarray, variant: str) -> tuple[np.ndarray, dict[str, object]]:
    base = to_binary_pore0_solid1(volume)
    pore = base == 0
    structure = connectivity_structure()
    metadata: dict[str, object] = {
        "variant": variant,
        "input_pore_voxels": int(np.count_nonzero(pore)),
        "input_total_voxels": int(base.size),
    }

    if variant == "identity":
        out_pore = pore.copy()
        metadata["operation"] = "none"
    elif variant == "pore_dilate_1":
        out_pore = ndi.binary_dilation(pore, structure=structure, iterations=1)
        metadata["operation"] = "binary_dilation_on_pore_mask"
    elif variant == "pore_erode_1":
        out_pore = ndi.binary_erosion(pore, structure=structure, iterations=1, border_value=0)
        metadata["operation"] = "binary_erosion_on_pore_mask"
    elif variant == "pore_open_1":
        out_pore = ndi.binary_opening(pore, structure=structure, iterations=1)
        metadata["operation"] = "binary_opening_on_pore_mask"
    elif variant == "pore_close_1":
        out_pore = ndi.binary_closing(pore, structure=structure, iterations=1)
        metadata["operation"] = "binary_closing_on_pore_mask"
    elif variant == "solid_dilate_1":
        solid = ndi.binary_dilation(~pore, structure=structure, iterations=1)
        out_pore = ~solid
        metadata["operation"] = "binary_dilation_on_solid_mask"
    elif variant == "solid_erode_1":
        solid = ndi.binary_erosion(~pore, structure=structure, iterations=1, border_value=1)
        out_pore = ~solid
        metadata["operation"] = "binary_erosion_on_solid_mask"
    elif variant.startswith("remove_small_pore_components_"):
        threshold = parse_small_component_threshold(variant)
        labels, n_components = ndi.label(pore, structure=structure)
        counts = np.bincount(labels.ravel())
        keep_labels = np.flatnonzero(counts >= threshold)
        keep_labels = keep_labels[keep_labels != 0]
        out_pore = np.isin(labels, keep_labels)
        metadata.update(
            {
                "operation": "remove_pore_connected_components_below_threshold",
                "component_threshold_voxels": threshold,
                "n_components": int(n_components),
                "removed_components": int(n_components - len(keep_labels)),
                "removed_pore_voxels": int(np.count_nonzero(pore) - np.count_nonzero(out_pore)),
            }
        )
    else:
        raise ValueError(f"unknown preprocessing variant: {variant}")

    out = np.where(out_pore, 0, 1).astype(np.uint8)
    metadata.update(
        {
            "output_pore_voxels": int(np.count_nonzero(out == 0)),
            "output_solid_voxels": int(np.count_nonzero(out == 1)),
            "output_total_voxels": int(out.size),
            "output_porosity": float(np.count_nonzero(out == 0) / out.size),
            "pore_voxel_delta": int(np.count_nonzero(out == 0) - np.count_nonzero(pore)),
        }
    )
    return out, metadata


def write_pnextract_mhd(
    mhd_path: Path,
    raw_path: Path,
    *,
    shape_zyx: tuple[int, int, int],
    voxel_size_um: float,
    title: str,
    pnextract_lines: list[str],
) -> None:
    z, y, x = (int(v) for v in shape_zyx)
    lines = [
        "ObjectType = Image",
        "NDims = 3",
        "ElementType = MET_UCHAR",
        "ElementByteOrderMSB = False",
        f"DimSize = {x} {y} {z}",
        f"ElementSize = {voxel_size_um:g} {voxel_size_um:g} {voxel_size_um:g}",
        "Offset = 0 0 0",
        f"ElementDataFile = {raw_path.name}",
        "",
        f"title {title}",
        "write_cnm true",
        "write_vtkNetwork false",
        "write_elements false",
        "",
        *pnextract_lines,
        "",
    ]
    mhd_path.write_text("\n".join(lines), encoding="utf-8")


def write_variant_inputs(
    volume: np.ndarray,
    out_dir: Path,
    *,
    source_path: Path,
    variant: str,
    title: str,
    voxel_size_um: float,
    pnextract_lines: list[str],
    metadata: dict[str, object],
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"{title}_pore0_solid1.raw"
    mhd_path = out_dir / f"{title}.mhd"
    metadata_path = out_dir / f"{title}_preprocess_metadata.json"
    binary = to_binary_pore0_solid1(volume)
    binary.tofile(raw_path)
    write_pnextract_mhd(
        mhd_path,
        raw_path,
        shape_zyx=tuple(int(v) for v in binary.shape),
        voxel_size_um=voxel_size_um,
        title=title,
        pnextract_lines=pnextract_lines,
    )
    pore_voxels = int(np.count_nonzero(binary == 0))
    summary: dict[str, object] = {
        "variant": variant,
        "title": title,
        "source_path": str(source_path),
        "prepare_dir": str(out_dir),
        "raw_path": str(raw_path),
        "mhd_path": str(mhd_path),
        "metadata_path": str(metadata_path),
        "shape_zyx": [int(v) for v in binary.shape],
        "voxel_size_um": float(voxel_size_um),
        "pore_voxels": pore_voxels,
        "solid_voxels": int(binary.size - pore_voxels),
        "total_voxels": int(binary.size),
        "porosity": float(pore_voxels / binary.size),
        "pnextract_lines": list(pnextract_lines),
        **metadata,
    }
    metadata_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def run_command(command: list[str], *, cwd: Path, timeout_s: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
        timeout=timeout_s,
    )


def build_parse_command(python_exe: Path, prefix: Path, outdir: Path) -> list[str]:
    return [
        str(python_exe),
        str(CODE_ROOT / "scripts" / "pore_network" / "parse_pnextract_network.py"),
        "--prefix",
        str(prefix),
        "--outdir",
        str(outdir),
    ]


def prepare_variants(
    source_tiff: Path,
    out_dir: Path,
    *,
    variants: list[str],
    voxel_size_um: float,
    pnextract_lines: list[str],
) -> list[dict[str, object]]:
    source_volume = tifffile.imread(source_tiff)
    base = to_binary_pore0_solid1(source_volume)
    summaries: list[dict[str, object]] = []
    for variant in variants:
        processed, metadata = apply_preprocess_variant(base, variant)
        title = f"niu2020_berea_preprocess_{variant}"
        summaries.append(
            write_variant_inputs(
                processed,
                out_dir / "candidates" / variant / "pnextract_inputs",
                source_path=source_tiff,
                variant=variant,
                title=title,
                voxel_size_um=voxel_size_um,
                pnextract_lines=pnextract_lines,
                metadata=metadata,
            )
        )
    return summaries


def run_candidate_network(
    summary: dict[str, object],
    *,
    out_dir: Path,
    pnextract_exe: Path,
    python_exe: Path,
    paper_pore: pd.DataFrame,
    paper_throat: pd.DataFrame,
) -> dict[str, object]:
    variant = str(summary["variant"])
    prepare_dir = Path(str(summary["prepare_dir"]))
    title = str(summary["title"])
    network_dir = out_dir / "candidates" / variant / "network_parsed"
    prefix = expected_pnextract_prefix(prepare_dir, title)
    started = time.perf_counter()
    try:
        pn = run_command([str(pnextract_exe), Path(str(summary["mhd_path"])).name], cwd=prepare_dir)
        verify_pnextract_outputs(prefix)
        parse = run_command(build_parse_command(python_exe, prefix, network_dir), cwd=PROJECT_ROOT)
        pores = pd.read_csv(network_dir / "pores.csv")
        throats = pd.read_csv(network_dir / "throats.csv")
        metrics, comparison = evaluate_network(paper_pore, paper_throat, pores, throats)
        comparison_path = out_dir / "candidates" / variant / "figure5_distribution_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        metrics.update(
            {
                "variant": variant,
                "status": "ok",
                "elapsed_s": time.perf_counter() - started,
                "prepare_dir": str(prepare_dir),
                "network_dir": str(network_dir),
                "pnextract_stdout_tail": pn.stdout[-4000:],
                "pnextract_stderr": pn.stderr,
                "parse_stdout": parse.stdout,
                "parse_stderr": parse.stderr,
                "comparison_csv": str(comparison_path),
            }
        )
        return {"variant": variant, "status": "ok", "prepare_dir": str(prepare_dir), "network_dir": str(network_dir), "metrics": metrics}
    except Exception as exc:
        return {"variant": variant, "status": "failed", "prepare_dir": str(prepare_dir), "error": repr(exc)}


def write_readme(out_dir: Path, source_tiff: Path, summaries: list[dict[str, object]], *, ran_pnextract: bool) -> None:
    lines = [
        "# Niu 2020 Berea Preprocessing Sensitivity",
        "",
        "This run creates derived pore=0/solid=1 Berea volumes to test whether CT/segmentation preprocessing can explain the missing Niu Figure5 short-throat tail.",
        "",
        f"- Source segmented TIFF: `{source_tiff}`",
        "- The source TIFF/raw and paper Figure5 workbook are read only.",
        "- Derived RAW/MHD inputs are written under this result directory.",
        f"- pnextract executed: `{ran_pnextract}`",
        "",
        "## Candidate porosities",
        "",
        "| Variant | Porosity | Pore voxel delta | Operation |",
        "| --- | ---: | ---: | --- |",
    ]
    for item in summaries:
        lines.append(
            f"| `{item['variant']}` | {float(item['porosity']):.6g} | {int(item.get('pore_voxel_delta', 0))} | {item.get('operation', '')} |"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `candidate_plan.csv`: variant paths and porosity changes.",
            "- `candidates/<variant>/pnextract_inputs/`: derived RAW/MHD and metadata.",
            "- If pnextract is run, `scan_metrics.csv` and per-candidate parsed networks are also written.",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-tiff", default=str(DEFAULT_BINARY_TIFF))
    parser.add_argument("--figure5", default=str(DEFAULT_FIGURE5))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--voxel-size-um", type=float, default=2.8)
    parser.add_argument("--pnextract-lines", nargs="*", default=DEFAULT_PNEXTRACT_LINES)
    parser.add_argument("--pnextract-exe", default=str(DEFAULT_PNEXTRACT_EXE))
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--run-pnextract", action="store_true")
    args = parser.parse_args()

    source_tiff = Path(args.source_tiff)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = prepare_variants(
        source_tiff,
        out_dir,
        variants=list(args.variants),
        voxel_size_um=args.voxel_size_um,
        pnextract_lines=list(args.pnextract_lines),
    )
    plan = pd.DataFrame(summaries)
    plan.to_csv(out_dir / "candidate_plan.csv", index=False)

    results: list[dict[str, object]] = []
    if args.run_pnextract:
        paper_pore, paper_throat = read_figure5(Path(args.figure5))
        for summary in summaries:
            candidate = run_candidate_network(
                summary,
                out_dir=out_dir,
                pnextract_exe=Path(args.pnextract_exe),
                python_exe=Path(args.python_exe),
                paper_pore=paper_pore,
                paper_throat=paper_throat,
            )
            row = {
                "variant": candidate["variant"],
                "status": candidate["status"],
                "prepare_dir": candidate["prepare_dir"],
            }
            if candidate.get("network_dir"):
                row["network_dir"] = candidate["network_dir"]
            if candidate.get("metrics"):
                row.update(candidate["metrics"])  # type: ignore[arg-type]
            if candidate.get("error"):
                row["error"] = candidate["error"]
            results.append(row)
            pd.DataFrame(results).to_csv(out_dir / "scan_metrics_partial.csv", index=False)
        pd.DataFrame(results).to_csv(out_dir / "scan_metrics.csv", index=False)

    write_readme(out_dir, source_tiff, summaries, ran_pnextract=bool(args.run_pnextract))
    payload = {
        "source_tiff": str(source_tiff),
        "out_dir": str(out_dir),
        "variants": list(args.variants),
        "ran_pnextract": bool(args.run_pnextract),
        "candidate_plan": str(out_dir / "candidate_plan.csv"),
        "scan_metrics": str(out_dir / "scan_metrics.csv") if args.run_pnextract else None,
    }
    (out_dir / "preprocess_sensitivity_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
