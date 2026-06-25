#!/usr/bin/env python3
"""Prepare a segmented CT volume for pnextract and render a ball-stick HTML view."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = PROJECT_ROOT.parent
PNEXTRACT_ROOT = REPO_ROOT / "pnextract"
DEFAULT_SEGMENTED_VOLUME = (
    PROJECT_ROOT
    / "data_inventory"
    / "ct_backed_samples_raw_copy_20260605"
    / "sample_89_Grainstone"
    / "CT_slices"
    / "89seged.tiff"
)
DEFAULT_TITLE = "sample_89_Grainstone_89seged_pnextract"
DEFAULT_PREPARE_DIR = PROJECT_ROOT / "results" / "pnextract_inputs" / "sample_89_Grainstone_89seged"
DEFAULT_NETWORK_DIR = PROJECT_ROOT / "results" / "pnextract" / "sample_89_Grainstone_89seged"
DEFAULT_HTML_OUT = (
    PROJECT_ROOT
    / "figures"
    / "segmented_cores"
    / "sample_89_Grainstone_89seged_pnextract_ballstick_interactive.html"
)
DEFAULT_METADATA_OUT = (
    PROJECT_ROOT
    / "results"
    / "source_data"
    / "sample_89_Grainstone_89seged_pnextract_ballstick_interactive_metadata.json"
)
DEFAULT_DOWNSAMPLE = 1
DEFAULT_DISTRIBUTION_BINS = 32
AUTO_PORE_VALUES = "auto-minority"


def parse_int_values(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer value")
    return values


def parse_pore_values(text: str) -> str | list[int]:
    clean = text.strip().lower()
    if clean in {"auto", "auto-minority", "minority"}:
        return AUTO_PORE_VALUES
    return parse_int_values(text)


def load_segmented_volume(path: Path, *, downsample: int = 1) -> np.ndarray:
    if downsample <= 0:
        raise ValueError("downsample must be a positive integer")
    volume = tifffile.imread(path)
    if volume.ndim != 3:
        raise ValueError(f"expected a 3-D segmented TIFF stack, got shape {volume.shape}")
    if downsample > 1:
        volume = volume[::downsample, ::downsample, ::downsample]
    return np.asarray(volume)


def infer_two_class_pore_solid_values(volume: np.ndarray) -> dict:
    values, counts = np.unique(volume, return_counts=True)
    if len(values) != 2:
        raise ValueError(
            "Automatic pore/solid inference expects exactly two segmented classes; "
            f"found {len(values)} values: {[int(v) for v in values]}"
        )
    order = np.argsort(counts)
    pore_index = int(order[0])
    solid_index = int(order[1])
    if int(counts[pore_index]) == int(counts[solid_index]):
        raise ValueError("Automatic pore/solid inference cannot resolve a tie between two classes")
    return {
        "pore_values": [int(values[pore_index])],
        "solid_value": int(values[solid_index]),
        "mode": "auto_minority_pore_majority_solid",
    }


def map_segmented_volume_to_pnextract(volume: np.ndarray, *, pore_values: list[int]) -> tuple[np.ndarray, dict]:
    source_values, source_counts = np.unique(volume, return_counts=True)
    pore_mask = np.isin(volume, np.asarray(pore_values, dtype=volume.dtype))
    mapped = np.where(pore_mask, 0, 1).astype(np.uint8)
    pore_voxels = int(np.count_nonzero(mapped == 0))
    total_voxels = int(mapped.size)
    stats = {
        "shape_zyx": [int(v) for v in volume.shape],
        "dtype": str(volume.dtype),
        "pore_values": [int(v) for v in pore_values],
        "source_values": {str(int(v)): int(c) for v, c in zip(source_values, source_counts)},
        "pore_voxels": pore_voxels,
        "solid_voxels": int(total_voxels - pore_voxels),
        "total_voxels": total_voxels,
        "porosity": float(pore_voxels / total_voxels) if total_voxels else 0.0,
        "pnextract_mapping": {"0": "pore/void", "1": "solid/non-pore components"},
    }
    return mapped, stats


def write_pnextract_mhd(
    mhd_path: Path,
    raw_path: Path,
    *,
    shape_zyx: tuple[int, int, int],
    voxel_size_um: float,
    title: str,
    pnextract_lines: list[str] | None = None,
) -> None:
    z, y, x = (int(v) for v in shape_zyx)
    element_size = f"{voxel_size_um:g} {voxel_size_um:g} {voxel_size_um:g}"
    extra_lines = [line.strip() for line in (pnextract_lines or []) if line.strip()]
    text = "\n".join(
        [
            "ObjectType = Image",
            "NDims = 3",
            "ElementType = MET_UCHAR",
            "ElementByteOrderMSB = False",
            f"DimSize = {x} {y} {z}",
            f"ElementSize = {element_size}",
            "Offset = 0 0 0",
            f"ElementDataFile = {raw_path.name}",
            "",
            f"title {title}",
            "write_cnm true",
            "write_vtkNetwork true",
            "write_elements false",
            "",
            *extra_lines,
            "",
        ]
    )
    mhd_path.write_text(text, encoding="utf-8")


def prepare_pnextract_input(
    segmented_volume: Path,
    prepare_dir: Path,
    *,
    title: str,
    pore_values: str | list[int],
    voxel_size_um: float,
    downsample: int,
    pnextract_lines: list[str] | None = None,
) -> dict:
    prepare_dir.mkdir(parents=True, exist_ok=True)
    volume = load_segmented_volume(segmented_volume, downsample=downsample)
    pore_inference = None
    if pore_values == AUTO_PORE_VALUES:
        pore_inference = infer_two_class_pore_solid_values(volume)
        pore_values = pore_inference["pore_values"]
    mapped, stats = map_segmented_volume_to_pnextract(volume, pore_values=pore_values)
    effective_voxel_size_um = float(voxel_size_um) * float(downsample)
    raw_path = prepare_dir / f"{title}_pore0_solid1.raw"
    mhd_path = prepare_dir / f"{title}.mhd"
    mapped.tofile(raw_path)
    write_pnextract_mhd(
        mhd_path,
        raw_path,
        shape_zyx=tuple(mapped.shape),
        voxel_size_um=effective_voxel_size_um,
        title=title,
        pnextract_lines=pnextract_lines,
    )
    summary = {
        "segmented_volume": str(segmented_volume),
        "prepare_dir": str(prepare_dir),
        "title": title,
        "downsample": int(downsample),
        "source_voxel_size_um": float(voxel_size_um),
        "effective_voxel_size_um": effective_voxel_size_um,
        "pore_value_inference": pore_inference
        or {"pore_values": [int(v) for v in pore_values], "mode": "explicit_pore_values"},
        "raw_path": str(raw_path),
        "mhd_path": str(mhd_path),
        "pnextract_lines": [line.strip() for line in (pnextract_lines or []) if line.strip()],
        "raw_size_bytes": int(raw_path.stat().st_size),
        **stats,
    }
    summary_path = prepare_dir / f"{title}_prepare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def extract_pnextract_bin(archive: Path, out_dir: Path) -> None:
    existing = list(out_dir.rglob("pnextract.exe")) if out_dir.exists() else []
    if existing:
        return
    try:
        import py7zr
    except ImportError as exc:
        raise RuntimeError(
            "py7zr is required to extract pnextract/bin.7z; install it in the active Python environment."
        ) from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive, mode="r") as archive_file:
        archive_file.extractall(path=out_dir)


def find_pnextract_exe(*roots: Path) -> Path | None:
    for root in roots:
        if not root or not root.exists():
            continue
        direct = root / "pnextract.exe"
        if direct.exists():
            return direct
        matches = sorted(root.rglob("pnextract.exe"))
        if matches:
            return matches[0]
    found = shutil.which("pnextract.exe") or shutil.which("pnextract")
    return Path(found) if found else None


def build_pnextract_command(pnextract_exe: Path, mhd_path: Path) -> list[str]:
    return [str(pnextract_exe), str(mhd_path.name)]


def build_parse_command(*, python_exe: Path, prefix: Path, outdir: Path) -> list[str]:
    return [
        str(python_exe),
        str(PROJECT_ROOT / "code" / "scripts" / "pore_network" / "parse_pnextract_network.py"),
        "--prefix",
        str(prefix),
        "--outdir",
        str(outdir),
    ]


def build_render_command(
    *,
    python_exe: Path,
    pores_csv: Path,
    throats_csv: Path,
    html_out: Path,
    metadata_out: Path,
    segmented_volume: Path,
    voxel_size_m: float,
    pore_value: int,
    solid_value: int,
    paraview_out_dir: Path | None = None,
    paraview_prefix: str | None = None,
) -> list[str]:
    command = [
        str(python_exe),
        str(PROJECT_ROOT / "code" / "scripts" / "pore_network" / "render_berea_pore_network_html.py"),
        "--pores",
        str(pores_csv),
        "--throats",
        str(throats_csv),
        "--out",
        str(html_out),
        "--metadata-out",
        str(metadata_out),
        "--voxel-size-m",
        f"{voxel_size_m:.12g}",
        "--segmented-volume",
        str(segmented_volume),
        "--pore-value",
        str(pore_value),
        "--solid-value",
        str(solid_value),
    ]
    if paraview_out_dir is not None:
        command.extend(["--paraview-out-dir", str(paraview_out_dir)])
        command.extend(["--paraview-prefix", str(paraview_prefix or html_out.stem)])
    return command


def run_command(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
    )


def _positive_finite_um(values_m: pd.Series | np.ndarray) -> np.ndarray:
    values_um = np.asarray(values_m, dtype=float) * 1.0e6
    return values_um[np.isfinite(values_um) & (values_um > 0)]


def frequency_histogram_um(values_m: pd.Series | np.ndarray, *, bins: int) -> dict:
    values_um = _positive_finite_um(values_m)
    if values_um.size == 0:
        raise ValueError("Cannot build a pore/throat distribution from empty or non-positive values")
    if bins < 1:
        raise ValueError("bins must be a positive integer")

    min_um = float(values_um.min())
    max_um = float(values_um.max())
    if np.isclose(min_um, max_um):
        low = min_um / np.sqrt(10.0)
        high = max_um * np.sqrt(10.0)
    else:
        low = min_um
        high = max_um
    edges = np.logspace(np.log10(low), np.log10(high), int(bins) + 1)
    counts, edges = np.histogram(values_um, bins=edges)
    total = int(counts.sum())
    fractions = counts.astype(float) / float(total) if total else counts.astype(float)
    centers = np.sqrt(edges[:-1] * edges[1:])
    return {
        "size_unit": "um",
        "fraction_kind": "frequency_fraction",
        "sample_count": int(values_um.size),
        "bin_edges_um": [float(v) for v in edges],
        "bin_centers_um": [float(v) for v in centers],
        "hist_count": [int(v) for v in counts],
        "hist_fraction": [float(v) for v in fractions],
        "min_um": min_um,
        "max_um": max_um,
        "mean_um": float(values_um.mean()),
        "median_um": float(np.median(values_um)),
    }


def write_pore_throat_distribution_figure(
    *,
    pores_csv: Path,
    throats_csv: Path,
    figure_out: Path,
    metadata_out: Path,
    bins: int = DEFAULT_DISTRIBUTION_BINS,
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pores = pd.read_csv(pores_csv)
    throats = pd.read_csv(throats_csv)
    pore_hist = frequency_histogram_um(pores["pore_radius_m"], bins=bins)
    throat_hist = frequency_histogram_um(throats["throat_length_m"], bins=bins)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    panels = [
        (axes[0], pore_hist, "Pore node size distribution", "Pore node radius ($\\mu$m)"),
        (axes[1], throat_hist, "Pore throat length distribution", "Pore throat length ($\\mu$m)"),
    ]
    for ax, hist, title, xlabel in panels:
        edges = np.asarray(hist["bin_edges_um"], dtype=float)
        fractions_percent = np.asarray(hist["hist_fraction"], dtype=float) * 100.0
        ax.bar(
            edges[:-1],
            fractions_percent,
            width=np.diff(edges),
            align="edge",
            color="#8a8a8a",
            edgecolor="#202020",
            linewidth=0.65,
        )
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency fraction (%)")
        ax.set_title(title)
        ax.grid(axis="y", color="#d7d7d7", linewidth=0.6, alpha=0.75)
        ax.tick_params(which="both", direction="in", top=True, right=True)

    figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_out, dpi=240)
    plt.close(fig)

    summary = {
        "figure_png": str(figure_out),
        "metadata_json": str(metadata_out),
        "input_pores_csv": str(pores_csv),
        "input_throats_csv": str(throats_csv),
        "units": "um",
        "bins": int(bins),
        "pore_node_size": pore_hist,
        "pore_throat_length": throat_hist,
        "notes": [
            "Pore node size uses pnextract node2 pore_radius_m converted to micrometers.",
            "Pore throat length uses pnextract link2 throat_length_m converted to micrometers.",
            "Histogram heights are unweighted frequency fractions, not volume-weighted fractions.",
        ],
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def annotate_render_metadata(metadata_path: Path, annotation: dict) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["pnextract_extraction"] = annotation
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def expected_pnextract_prefix(prepare_dir: Path, title: str) -> Path:
    return prepare_dir / title


def verify_pnextract_outputs(prefix: Path) -> None:
    required = [prefix.with_name(prefix.name + suffix) for suffix in ("_node1.dat", "_node2.dat", "_link1.dat", "_link2.dat")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"pnextract did not produce expected network files: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_SEGMENTED_VOLUME))
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--pore-values", type=parse_pore_values, default=AUTO_PORE_VALUES)
    parser.add_argument("--solid-value", type=int, default=255)
    parser.add_argument("--voxel-size-um", type=float, default=1.7)
    parser.add_argument("--downsample", type=int, default=DEFAULT_DOWNSAMPLE)
    parser.add_argument("--prepare-dir", default=str(DEFAULT_PREPARE_DIR))
    parser.add_argument("--network-dir", default=str(DEFAULT_NETWORK_DIR))
    parser.add_argument("--html-out", default=str(DEFAULT_HTML_OUT))
    parser.add_argument("--metadata-out", default=str(DEFAULT_METADATA_OUT))
    parser.add_argument("--distribution-out")
    parser.add_argument("--distribution-metadata-out")
    parser.add_argument("--distribution-bins", type=int, default=DEFAULT_DISTRIBUTION_BINS)
    parser.add_argument("--pnextract-exe")
    parser.add_argument("--pnextract-archive", default=str(PNEXTRACT_ROOT / "bin.7z"))
    parser.add_argument("--pnextract-bin-dir", default=str(PROJECT_ROOT / "results" / "tools" / "pnextract_bin"))
    parser.add_argument("--skip-pnextract", action="store_true", help="Only prepare pnextract RAW/MHD input.")
    parser.add_argument("--skip-render", action="store_true", help="Run pnextract/parse but do not render HTML.")
    parser.add_argument("--skip-distribution", action="store_true", help="Do not write pore/throat distribution PNG.")
    parser.add_argument(
        "--pnextract-line",
        action="append",
        default=[],
        help="Append one raw pnextract parameter line to the generated .mhd input; repeat for multiple lines.",
    )
    args = parser.parse_args()

    segmented_volume = Path(args.input)
    prepare_dir = Path(args.prepare_dir)
    network_dir = Path(args.network_dir)
    python_exe = Path(sys.executable)

    prepare_summary = prepare_pnextract_input(
        segmented_volume,
        prepare_dir,
        title=args.title,
        pore_values=args.pore_values,
        voxel_size_um=args.voxel_size_um,
        downsample=args.downsample,
        pnextract_lines=args.pnextract_line,
    )
    resolved_pore_value = int(prepare_summary["pore_values"][0])
    resolved_solid_value = int(
        prepare_summary.get("pore_value_inference", {}).get("solid_value", int(args.solid_value))
    )

    run_summary: dict = {"prepare": prepare_summary}
    if args.skip_pnextract:
        print(json.dumps(run_summary, indent=2, ensure_ascii=False))
        return

    pnextract_exe = Path(args.pnextract_exe) if args.pnextract_exe else None
    if pnextract_exe is None:
        extract_pnextract_bin(Path(args.pnextract_archive), Path(args.pnextract_bin_dir))
        pnextract_exe = find_pnextract_exe(Path(args.pnextract_bin_dir), PNEXTRACT_ROOT)
    if pnextract_exe is None or not pnextract_exe.exists():
        raise FileNotFoundError("Could not find pnextract.exe after checking configured paths.")

    mhd_path = Path(prepare_summary["mhd_path"])
    pnextract_result = run_command(build_pnextract_command(pnextract_exe, mhd_path), cwd=prepare_dir)
    prefix = expected_pnextract_prefix(prepare_dir, args.title)
    verify_pnextract_outputs(prefix)

    parsed_dir = network_dir / "network_parsed"
    parse_result = run_command(build_parse_command(python_exe=python_exe, prefix=prefix, outdir=parsed_dir), cwd=PROJECT_ROOT)

    distribution_summary = None
    if not args.skip_distribution:
        distribution_out = (
            Path(args.distribution_out)
            if args.distribution_out
            else PROJECT_ROOT / "figures" / "segmented_cores" / f"{args.title}_pore_throat_distribution.png"
        )
        distribution_metadata_out = (
            Path(args.distribution_metadata_out)
            if args.distribution_metadata_out
            else PROJECT_ROOT / "results" / "source_data" / f"{args.title}_pore_throat_distribution_metadata.json"
        )
        distribution_summary = write_pore_throat_distribution_figure(
            pores_csv=parsed_dir / "pores.csv",
            throats_csv=parsed_dir / "throats.csv",
            figure_out=distribution_out,
            metadata_out=distribution_metadata_out,
            bins=int(args.distribution_bins),
        )

    render_result = None
    if not args.skip_render:
        metadata_out = Path(args.metadata_out)
        html_out = Path(args.html_out)
        paraview_out_dir = html_out.parent / "paraview"
        render_result = run_command(
            build_render_command(
                python_exe=python_exe,
                pores_csv=parsed_dir / "pores.csv",
                throats_csv=parsed_dir / "throats.csv",
                html_out=html_out,
                metadata_out=metadata_out,
                segmented_volume=segmented_volume,
                voxel_size_m=float(args.voxel_size_um) * float(args.downsample) * 1e-6,
                pore_value=resolved_pore_value,
                solid_value=resolved_solid_value,
                paraview_out_dir=paraview_out_dir,
                paraview_prefix=html_out.stem,
            ),
            cwd=PROJECT_ROOT,
        )
        annotate_render_metadata(
            metadata_out,
            {
                "segmented_volume": str(segmented_volume),
                "pnextract_input_mhd": str(mhd_path),
                "pnextract_prefix": str(prefix),
                "downsample": int(args.downsample),
                "source_voxel_size_um": float(args.voxel_size_um),
                "effective_voxel_size_um": float(args.voxel_size_um) * float(args.downsample),
                "pore_value": resolved_pore_value,
                "solid_value": resolved_solid_value,
                "voxel_size_note": (
                    "source_voxel_size_um is the isotropic voxel spacing supplied to this run; "
                    "rerun with --voxel-size-um for calibrated physical pore/throat units."
                ),
                "porosity_note": "The HTML overlay porosity is computed from the original full-resolution segmented volume.",
            },
        )

    run_summary.update(
        {
            "pnextract_exe": str(pnextract_exe),
            "pnextract_prefix": str(prefix),
            "parsed_network_dir": str(parsed_dir),
            "html_out": str(Path(args.html_out)) if not args.skip_render else None,
            "metadata_out": str(Path(args.metadata_out)) if not args.skip_render else None,
            "distribution_out": None if distribution_summary is None else distribution_summary["figure_png"],
            "distribution_metadata_out": None if distribution_summary is None else distribution_summary["metadata_json"],
            "commands": {
                "pnextract": build_pnextract_command(pnextract_exe, mhd_path),
                "parse": build_parse_command(python_exe=python_exe, prefix=prefix, outdir=parsed_dir),
                "render": None
                if args.skip_render
                else build_render_command(
                    python_exe=python_exe,
                    pores_csv=parsed_dir / "pores.csv",
                    throats_csv=parsed_dir / "throats.csv",
                    html_out=Path(args.html_out),
                    metadata_out=Path(args.metadata_out),
                    segmented_volume=segmented_volume,
                    voxel_size_m=float(args.voxel_size_um) * float(args.downsample) * 1e-6,
                    pore_value=resolved_pore_value,
                    solid_value=resolved_solid_value,
                    paraview_out_dir=Path(args.html_out).parent / "paraview",
                    paraview_prefix=Path(args.html_out).stem,
                ),
            },
            "stdout": {
                "pnextract": pnextract_result.stdout[-4000:],
                "parse": parse_result.stdout[-4000:],
                "render": None if render_result is None else render_result.stdout[-4000:],
            },
            "stderr": {
                "pnextract": pnextract_result.stderr[-4000:],
                "parse": parse_result.stderr[-4000:],
                "render": None if render_result is None else render_result.stderr[-4000:],
            },
        }
    )
    run_summary_path = network_dir / f"{args.title}_ballstick_run_summary.json"
    run_summary_path.parent.mkdir(parents=True, exist_ok=True)
    run_summary_path.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    run_summary["run_summary_path"] = str(run_summary_path)
    print(json.dumps(run_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
