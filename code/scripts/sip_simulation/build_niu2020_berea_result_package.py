#!/usr/bin/env python3
"""Build a self-contained Niu 2020 Berea reproduction result directory."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTER_LEGACY_ROOT = PROJECT_ROOT.parent
DEFAULT_RESULT_DIR = PROJECT_ROOT / "results" / "niu2020_berea_reproduction_20260617_original_pnextract_defaults"
DEFAULT_SWEEP_DIR = DEFAULT_RESULT_DIR / "simulation_sweeps"
DEFAULT_FORMAL_SWEEP_PATHS = {
    "all": DEFAULT_SWEEP_DIR / "niu2020_berea_full350_all_original_pnextract_fft_x" / "sweep_results.csv",
    "pore": DEFAULT_SWEEP_DIR / "niu2020_berea_full350_pore_fft_x" / "sweep_results.csv",
    "membrane": DEFAULT_SWEEP_DIR / "niu2020_berea_full350_membrane_original_pnextract_fft_x" / "sweep_results.csv",
    "interfacial": DEFAULT_SWEEP_DIR / "niu2020_berea_full350_interfacial_precision_merged" / "sweep_results.csv",
}
DEFAULT_FIGURE5 = PROJECT_ROOT / "data" / "Niu 2020data" / "Figure5.xlsx"
DEFAULT_TIFF = PROJECT_ROOT / "data" / "Niu 2020data" / "microCT_Berea.tiff"
DEFAULT_RAW = PROJECT_ROOT / "data" / "Niu 2020data" / "microCT_Berea.raw"
CONFIG_SOURCE = PROJECT_ROOT / "configs" / "sip_simulation" / "niu2020_berea_ac3d_reproduction_config.py"
DEFAULT_LEGACY_NETWORK_DIR = OUTER_LEGACY_ROOT / "outputs" / "figure5_pnextract_comparison" / "network_parsed"
DEFAULT_PNEXTRACT_EXE = PROJECT_ROOT / "code" / "vendor" / "pnextract" / "bin" / "pnextract.exe"
DEFAULT_PNEXTRACT_PARAMETER_FILE = (
    PROJECT_ROOT
    / "code"
    / "vendor"
    / "pnextract"
    / "config"
    / "niu2020_contact_split_conserve_pnextract_lines.txt"
)
SAMPLE_ID = "niu2020_berea"
VOXEL_SIZE_UM = 2.8
NETWORK_PORE_COORD_COLUMNS = {"pore_center_x_m", "pore_center_y_m", "pore_center_z_m"}
NETWORK_REQUIRED_PORE_COLUMNS = {"pore_id", "pore_radius_m", *NETWORK_PORE_COORD_COLUMNS}
NETWORK_REQUIRED_THROAT_COLUMNS = {"pore1_id", "pore2_id", "throat_radius_m"}
DIAGNOSTIC_MEMBRANE_COMPONENT_COLUMNS = {
    "membrane_geometry_mode",
    "passive_area_source",
    "passive_length_source",
    "passive_weight_source",
    "passive_zdc_source",
    "passive_branch_alpha",
}
DIAGNOSTIC_MEMBRANE_OPTIONAL_PROVENANCE_COLUMNS = (
    "passive_branch_alpha_source",
    "passive_transport_number_difference",
    "passive_anion_transport_number_reference",
    "active_anion_transport_number_inferred",
    "passive_transport_number_difference_cap",
    "passive_transport_number_capped_fraction",
    "passive_transport_number_edl_limited_fraction",
    "edl_debye_length_m",
    "edl_thickness_multiplier",
    "edl_selectivity",
    "maximum_transport_number_difference",
    "passive_branch_alpha_median",
    "passive_branch_alpha_mean",
    "passive_branch_alpha_min",
    "passive_branch_alpha_max",
    "edl_selection_rule",
    "edl_selection_rmse_tolerance",
    "edl_selected_peak_normalized_rmse",
)
MEMBRANE_REFERENCE_DIR = PROJECT_ROOT / "docs" / "references" / "niu2020_membrane_polarization_refs_20260617"


def diagnostic_membrane_literature_basis() -> list[dict[str, str]]:
    """Structured literature support for the active/passive diagnostic membrane branch."""

    return [
        {
            "citation_key": "Marshall_Madden_1959",
            "doi": "10.1190/1.1438659",
            "model_role": "active/passive impedance foundation",
            "supports": "Membrane polarization can be represented by ion-selective active zones and less selective passive zones with a DC impedance and frequency-dependent relaxation.",
            "local_pdf": str(MEMBRANE_REFERENCE_DIR / "1959_marshall_madden_1959_10.1190_1.1438659.pdf"),
            "local_text": str(MEMBRANE_REFERENCE_DIR / "extracted_text" / "1959_marshall_madden_1959_10.1190_1.1438659.txt"),
        },
        {
            "citation_key": "Titov_Komarov_Tarasov_Levitski_2002",
            "doi": "10.1016/S0926-9851(02)00168-4",
            "model_role": "transport-number and geometry-factor chargeability",
            "supports": "The membrane-polarization chargeability depends on active/passive transport-number contrast and the lengths and sections of passive and active zones.",
            "local_pdf": str(MEMBRANE_REFERENCE_DIR / "2002_titov_komarov_tarasov_levitski_10.1016_S0926-9851(02)00168-4.pdf"),
            "local_text": str(
                MEMBRANE_REFERENCE_DIR
                / "extracted_text"
                / "2002_titov_komarov_tarasov_levitski_10.1016_S0926-9851(02)00168-4.txt"
            ),
        },
        {
            "citation_key": "Buecker_Hoerdt_2013b",
            "doi": "10.1190/GEO2012-0548.1",
            "model_role": "SNP/LNP two-time-scale interpretation",
            "supports": "The Marshall-Madden impedance contains active and passive zone time constants; limiting SNP/LNP regimes can be controlled by narrow active-zone or wide passive-zone lengths.",
            "local_pdf": str(MEMBRANE_REFERENCE_DIR / "2013_bucker_hordt_2013b_10.1190_geo2012-0548.1.pdf"),
            "local_text": str(MEMBRANE_REFERENCE_DIR / "extracted_text" / "2013_bucker_hordt_2013b_10.1190_geo2012-0548.1.txt"),
        },
        {
            "citation_key": "Buecker_FloresOrozco_Undorf_Kemna_2019",
            "doi": "10.1029/2019JB017679",
            "model_role": "Stern/diffuse-layer caution for pore constrictions",
            "supports": "Pore-constriction membrane polarization can be affected by Stern- and diffuse-layer coupling, so the diagnostic branch should not be presented as a complete published Niu parameter.",
            "local_pdf": str(MEMBRANE_REFERENCE_DIR / "2019_bucker_flores-orozco_undorf_kemna_10.1029_2019JB017679.pdf"),
            "local_text": str(
                MEMBRANE_REFERENCE_DIR
                / "extracted_text"
                / "2019_bucker_flores-orozco_undorf_kemna_10.1029_2019JB017679.txt"
            ),
        },
    ]


def load_pnextract_parameter_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def write_chinese_parameter_config(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG_SOURCE, out)


def read_figure5_geometry(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_excel(path, sheet_name=0, header=None)
    data = table.iloc[1:].copy()
    pore = data.iloc[:, [0, 1]].copy()
    pore.columns = ["size_m", "relative_volume"]
    throat = data.iloc[:, [2, 3]].copy()
    throat.columns = ["size_m", "relative_volume"]
    for frame in (pore, throat):
        frame["size_m"] = pd.to_numeric(frame["size_m"], errors="coerce")
        frame["relative_volume"] = pd.to_numeric(frame["relative_volume"], errors="coerce")
        frame.dropna(subset=["size_m", "relative_volume"], inplace=True)
        total = frame["relative_volume"].sum()
        if total > 0:
            frame["relative_volume"] = frame["relative_volume"] / total
    return pore.reset_index(drop=True), throat.reset_index(drop=True)


def remap_niu_tiff_to_notebook_binary(
    input_tiff: Path,
    out_dir: Path,
    *,
    sample_id: str = SAMPLE_ID,
    input_volume: np.ndarray | None = None,
    voxel_size_um: float = VOXEL_SIZE_UM,
) -> dict:
    """Follow notebooks/seged_DRP_and_PNM.ipynb: remap segmentation to pore=0, solid=255."""
    out_dir.mkdir(parents=True, exist_ok=True)
    volume = np.asarray(input_volume if input_volume is not None else tifffile.imread(input_tiff))
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D TIFF stack, got shape {volume.shape}")

    values, counts = np.unique(volume, return_counts=True)
    if len(values) < 2:
        raise ValueError(f"Expected at least two segmented values, got {[int(v) for v in values]}")
    solid_source_values = [int(values[int(np.argmax(counts))])]
    pore_source_values = [int(v) for v in values if int(v) not in solid_source_values]
    binary = np.where(np.isin(volume, solid_source_values), 255, 0).astype(np.uint8)

    binary_tiff = out_dir / f"{sample_id}_solid255_pore0.tiff"
    binary_raw = out_dir / f"{sample_id}_solid255_pore0.raw"
    metadata_json = out_dir / f"{sample_id}_solid255_pore0_remap_metadata.json"
    tifffile.imwrite(binary_tiff, binary, photometric="minisblack")
    binary.tofile(binary_raw)

    pore_voxels = int(np.count_nonzero(binary == 0))
    solid_voxels = int(np.count_nonzero(binary == 255))
    total_voxels = int(binary.size)
    metadata = {
        "input_tiff": str(input_tiff),
        "binary_tiff": str(binary_tiff),
        "binary_raw": str(binary_raw),
        "metadata_json": str(metadata_json),
        "shape_zyx": [int(v) for v in binary.shape],
        "dtype": str(binary.dtype),
        "source_value_counts": {str(int(v)): int(c) for v, c in zip(values, counts)},
        "solid_source_values": solid_source_values,
        "pore_source_values": pore_source_values,
        "solid_output_value": 255,
        "pore_output_value": 0,
        "solid_voxels": solid_voxels,
        "pore_voxels": pore_voxels,
        "total_voxels": total_voxels,
        "porosity": float(pore_voxels / total_voxels) if total_voxels else 0.0,
        "voxel_size_um_assumed_xyz": [float(voxel_size_um), float(voxel_size_um), float(voxel_size_um)],
        "notebook_reference": "notebooks/seged_DRP_and_PNM.ipynb remap cell: majority class -> solid 255; other class(es) -> pore 0.",
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def _bar_widths_log(x: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.asarray([x[0] * 0.2], dtype=float)
    mids = np.sqrt(x[:-1] * x[1:])
    first = x[0] ** 2 / mids[0]
    last = x[-1] ** 2 / mids[-1]
    edges = np.concatenate([[first], mids, [last]])
    return np.diff(edges)


def write_figure4_style_distribution(figure5: Path, figure_out: Path, source_data_csv: Path) -> dict:
    pore, throat = read_figure5_geometry(figure5)
    combined = pd.concat(
        [
            pore.assign(kind="pore_node_size"),
            throat.assign(kind="pore_throat_length"),
        ],
        ignore_index=True,
    )
    source_data_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(source_data_csv, index=False)

    plt.rcParams.update({"font.family": "serif", "font.size": 12, "axes.linewidth": 1.0})
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2), constrained_layout=True)
    panels = [
        (axes[0], pore, "(a)", "Pore node size (m)"),
        (axes[1], throat, "(b)", "Pore throat length (m)"),
    ]
    for ax, frame, panel, xlabel in panels:
        x = frame["size_m"].to_numpy(dtype=float)
        y = frame["relative_volume"].to_numpy(dtype=float)
        ax.bar(x, y, width=_bar_widths_log(x) * 0.48, color="#6f6f6f", edgecolor="#202020", linewidth=0.8, align="center")
        ax.set_xscale("log")
        ax.set_ylim(0.0, 0.15)
        ax.set_ylabel("Volume fraction")
        ax.set_xlabel(xlabel)
        ax.text(-0.18, 1.03, panel, transform=ax.transAxes, fontsize=18, ha="left", va="bottom")
        ax.tick_params(which="both", direction="in", top=False, right=False)
        ax.minorticks_on()
    figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_out, dpi=360, bbox_inches="tight")
    plt.close(fig)
    return {
        "figure_png": str(figure_out),
        "source_data_csv": str(source_data_csv),
        "figure5_xlsx": str(figure5),
        "note": "Figure 4 style distribution redrawn from Niu 2020 Figure5.xlsx geometry data.",
    }


def build_visualization_commands(*, result_dir: Path, python_exe: Path, binary: dict) -> list[list[str]]:
    digital_dir = result_dir / "digital_rock"
    return [
        [
            str(python_exe),
            str(PROJECT_ROOT / "code" / "scripts" / "digital_rock_visualization" / "render_segmented_core_fiji3d_html.py"),
            "--input",
            str(binary["binary_tiff"]),
            "--out",
            str(digital_dir / "berea_fiji3d_volume_interactive.html"),
            "--metadata-out",
            str(digital_dir / "berea_fiji3d_volume_interactive_metadata.json"),
            "--solid-value",
            "255",
            "--components",
            "0",
            "255",
            "--initial-visible",
            "pore",
            "--downsample",
            "1",
            "--voxel-size-um",
            "2.8",
            "--interpolation",
            "linear",
        ],
    ]


def build_sip_plot_command(*, result_dir: Path, python_exe: Path) -> list[str]:
    return [
        str(python_exe),
        str(PROJECT_ROOT / "code" / "scripts" / "sip_simulation" / "plot_niu2020_conductivity_mechanism_comparison.py"),
        "--figure-base",
        str(result_dir / "figures" / "niu2020_conductivity_mechanism_comparison"),
        "--source-data-csv",
        str(result_dir / "source_data" / "niu2020_conductivity_mechanism_comparison_source_data.csv"),
        "--provenance-md",
        str(result_dir / "provenance" / "niu2020_conductivity_mechanism_comparison_provenance.md"),
    ]


def write_missing_sweep_reproduction_plan(result_dir: Path, missing_sweeps: dict[str, Path]) -> dict:
    """Record why formal SIP curves were not regenerated and how to run them."""

    provenance_dir = result_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)
    plan_json = provenance_dir / "formal_fullgrid_sweep_reproduction_plan.json"
    readme_md = provenance_dir / "formal_fullgrid_sweep_reproduction_plan.md"
    checkpoint_frequencies = [1e-3, 1e-2, 1e-1, 1.0, 1e3, 1e6, 1e9]
    directions = ["x", "y", "z"]
    mechanisms = ["interfacial", "pore", "membrane", "all"]
    command_templates: list[dict[str, object]] = []
    for mechanism in mechanisms:
        for direction in directions:
            command_templates.append(
                {
                    "mechanism": mechanism,
                    "direction": direction,
                    "command": [
                        "python",
                        "code/scripts/sip_simulation/run_ac3d_matrix_free_gpu_sweep.py",
                        "--raw",
                        "data/Niu 2020data/microCT_Berea.raw",
                        "--shape",
                        "350",
                        "350",
                        "350",
                        "--pore-label",
                        "1",
                        "--solid-label",
                        "2",
                        "--voxel-size-m",
                        "2.8e-6",
                        "--spectra",
                        f"results/<run_name>/source_data/{mechanism}_phase_conductivity_spectrum.csv",
                        "--frequency-match-mode",
                        "exact",
                        "--direction",
                        direction,
                        "--dtype",
                        "complex128",
                        "--preconditioner",
                        "fft",
                        "--fft-reference",
                        "pore",
                        "--gauge-mode",
                        "auto",
                        "--rtol",
                        "1e-5",
                        "--residual-every",
                        "50",
                        "--out-dir",
                        f"results/<run_name>/simulation_sweeps/{mechanism}_{direction}_complex128",
                    ],
                }
            )
    checkpoint_commands = [
        {
            "frequency_hz": frequency,
            "command": [
                "python",
                "code/scripts/sip_simulation/run_ac3d_matrix_free_gpu_single.py",
                "--frequency",
                f"{frequency:g}",
                "--frequency-match-mode",
                "exact",
                "--dtype",
                "complex128",
                "--preconditioner",
                "fft",
                "--fft-reference",
                "pore",
                "--gauge-mode",
                "auto",
                "--rtol",
                "1e-5",
            ],
        }
        for frequency in checkpoint_frequencies
    ]
    plan = {
        "status": "formal_sweep_results_missing",
        "missing_sweeps": {key: str(path) for key, path in missing_sweeps.items()},
        "required_mechanisms": mechanisms,
        "required_directions": directions,
        "checkpoint_frequencies_hz": checkpoint_frequencies,
        "frequency_match_mode": "exact",
        "gauge_mode": "auto",
        "formal_acceptance_rtol": 1.0e-5,
        "trusted_fullgrid_dtype": "complex128",
        "fft_reference": "pore",
        "residual_policy": "Use true_residual_norm and true_residual_passed; recursive_residual_norm is diagnostic.",
        "all_mechanism_policy": "all must be an independent AC3D field solve from all phase conductivities, not algebraic effective-conductivity splicing.",
        "directional_summary_tool": "code/scripts/sip_simulation/summarize_ac3d_directional_sweeps.py",
        "precision_checkpoint_tool": "code/scripts/sip_simulation/verify_ac3d_precision_checkpoints.py",
        "command_templates": command_templates,
        "complex128_checkpoint_commands": checkpoint_commands,
    }
    plan_json.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Formal full-grid sweep reproduction plan",
        "",
        "当前结果包未声称完成正式 Niu 2020 full-grid 机制曲线复现：以下 `sweep_results.csv` 缺失。",
        "",
    ]
    for key, path in missing_sweeps.items():
        lines.append(f"- `{key}`: `{path}` sweep_results.csv 缺失")
    lines.extend(
        [
            "",
            "正式运行要求：",
            "- `--frequency-match-mode exact`，禁止 silent nearest-frequency 匹配。",
            "- `--gauge-mode auto`，pore-only/membrane-only 的 `solid = 0` 由 active-domain gauge 处理。",
            "- `--dtype complex128 --fft-reference pore --rtol 1e-5` 作为可信 full-grid sweep 口径；complex64 只在通过 true residual 和 complex128 对照后才可作为正式曲线。",
            "- 每个频点写出 `recursive_residual_norm`、`true_residual_norm`、`true_residual_passed`。",
            "- `all` 机制必须独立 AC3D 场求解，不允许有效电导代数拼接。",
            "- x/y/z 三方向运行后用 `summarize_ac3d_directional_sweeps.py` 输出 `sigma_xx/sigma_yy/sigma_zz`、directional mean 和 anisotropy ratio。",
            "- 关键频点用 `complex128 checkpoint` 复核，并由 `verify_ac3d_precision_checkpoints.py` 输出 complex64 vs complex128 相对误差。",
            "",
            f"机器可读计划：`{plan_json}`",
        ]
    )
    readme_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"status": plan["status"], "plan_json": str(plan_json), "readme_md": str(readme_md)}


def network_has_renderable_coordinates(pores_csv: Path, throats_csv: Path) -> bool:
    """Return True when the pnextract tables include physical pore centers."""
    try:
        pore_columns = set(pd.read_csv(pores_csv, nrows=0).columns)
        throat_columns = set(pd.read_csv(throats_csv, nrows=0).columns)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return False
    return NETWORK_REQUIRED_PORE_COLUMNS.issubset(pore_columns) and NETWORK_REQUIRED_THROAT_COLUMNS.issubset(throat_columns)


def build_pnextract_ballstick_command(*, result_dir: Path, python_exe: Path, binary: dict) -> list[str]:
    command = [
        str(python_exe),
        str(PROJECT_ROOT / "code" / "scripts" / "pore_network" / "run_segmented_core_pnextract_ballstick.py"),
        "--input",
        str(binary["binary_tiff"]),
        "--title",
        f"{SAMPLE_ID}_pnextract",
        "--pore-values",
        "0",
        "--solid-value",
        "255",
        "--voxel-size-um",
        f"{VOXEL_SIZE_UM:g}",
        "--downsample",
        "1",
        "--prepare-dir",
        str(result_dir / "pore_network" / "pnextract_inputs"),
        "--network-dir",
        str(result_dir / "pore_network" / "pnextract"),
        "--html-out",
        str(result_dir / "pore_network" / "berea_pnextract_ballstick_interactive.html"),
        "--metadata-out",
        str(result_dir / "pore_network" / "berea_pnextract_ballstick_interactive_metadata.json"),
        "--distribution-out",
        str(result_dir / "pore_network" / "berea_pnextract_pore_throat_frequency_distribution.png"),
        "--distribution-metadata-out",
        str(result_dir / "pore_network" / "berea_pnextract_pore_throat_frequency_distribution_metadata.json"),
        "--pnextract-exe",
        str(DEFAULT_PNEXTRACT_EXE),
    ]
    for line in load_pnextract_parameter_lines(DEFAULT_PNEXTRACT_PARAMETER_FILE):
        command.extend(["--pnextract-line", line])
    return command


def build_network_html_command(*, result_dir: Path, python_exe: Path, pores_csv: Path, throats_csv: Path) -> list[str]:
    return [
        str(python_exe),
        str(PROJECT_ROOT / "code" / "scripts" / "pore_network" / "render_berea_pore_network_html.py"),
        "--pores",
        str(pores_csv),
        "--throats",
        str(throats_csv),
        "--out",
        str(result_dir / "pore_network" / "berea_pore_network_ballstick_interactive.html"),
        "--metadata-out",
        str(result_dir / "pore_network" / "berea_pore_network_ballstick_interactive_metadata.json"),
        "--voxel-size-m",
        "2.8e-6",
    ]


def copy_legacy_network(network_dir: Path, result_dir: Path) -> tuple[Path, Path]:
    out_dir = result_dir / "pore_network" / "network_parsed"
    out_dir.mkdir(parents=True, exist_ok=True)
    pores_out = out_dir / "pores.csv"
    throats_out = out_dir / "throats.csv"
    shutil.copy2(network_dir / "pores.csv", pores_out)
    shutil.copy2(network_dir / "throats.csv", throats_out)
    summary = network_dir / "network_summary.json"
    if summary.exists():
        shutil.copy2(summary, out_dir / "network_summary.json")
    return pores_out, throats_out


def run_command(command: list[str], *, cwd: Path) -> dict:
    result = subprocess.run(command, cwd=str(cwd), text=True, encoding="utf-8", errors="replace", capture_output=True)
    return {
        "command": command,
        "returncode": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def write_diagnostic_membrane_model_summary(
    *,
    component_csv: Path,
    fullgrid_summary_json: Path,
    out: Path,
) -> dict:
    component = pd.read_csv(component_csv)
    missing = DIAGNOSTIC_MEMBRANE_COMPONENT_COLUMNS.difference(component.columns)
    if missing:
        raise ValueError(f"diagnostic membrane component CSV is missing columns: {sorted(missing)}")
    if component.empty:
        raise ValueError(f"diagnostic membrane component CSV is empty: {component_csv}")

    first = component.iloc[0]
    fullgrid_summary = json.loads(fullgrid_summary_json.read_text(encoding="utf-8"))
    metrics = fullgrid_summary.get("metrics", {}).get("volume_area_dual_length", {})
    if not metrics:
        metrics = {
            key: value
            for key, value in fullgrid_summary.items()
            if key
            in {
                "common_frequency_count",
                "peak_normalized_rmse",
                "median_ratio",
                "min_ratio",
                "max_ratio",
                "paper_peak_frequency_hz",
                "candidate_peak_frequency_hz",
                "candidate_peak_imag_s_m",
                "paper_peak_imag_s_m",
                "ratio_at_paper_peak",
                "ratio_at_candidate_peak",
                "all_solver_info_zero",
                "max_relative_residual_norm",
            }
        }
    summary = {
        "status": "diagnostic_not_default_niu2020_parameter",
        "component_spectrum_csv": str(component_csv),
        "fullgrid_summary_json": str(fullgrid_summary_json),
        "membrane_geometry_mode": str(first["membrane_geometry_mode"]),
        "passive_area_source": str(first["passive_area_source"]),
        "passive_length_source": str(first["passive_length_source"]),
        "passive_weight_source": str(first["passive_weight_source"]),
        "passive_zdc_source": str(first["passive_zdc_source"]),
        "passive_branch_alpha": float(first["passive_branch_alpha"]),
        "fullgrid_metrics": {
            key: float(value)
            if isinstance(value, (float, np.floating))
            else int(value)
            if isinstance(value, np.integer)
            else bool(value)
            if isinstance(value, (bool, np.bool_))
            else value
            for key, value in metrics.items()
        },
        "literature_basis": [
            "Marshall-Madden active/passive membrane zones",
            "Titov et al. active/passive transport-number and geometry factor",
            "Buecker-Hoerdt SNP/LNP two-time-scale interpretation",
        ],
        "literature_basis_structured": diagnostic_membrane_literature_basis(),
    }
    for column in DIAGNOSTIC_MEMBRANE_OPTIONAL_PROVENANCE_COLUMNS:
        if column in component.columns and not pd.isna(first[column]):
            value = first[column]
            if isinstance(value, (float, np.floating)):
                summary[column] = float(value)
            elif isinstance(value, (int, np.integer)):
                summary[column] = int(value)
            elif isinstance(value, (bool, np.bool_)):
                summary[column] = bool(value)
            else:
                summary[column] = str(value)
    edl_limited = "edl_limited" in str(summary["membrane_geometry_mode"])
    summary["interpretation"] = (
        "This membrane spectrum is a diagnostic active/passive two-length candidate. "
        + (
            "The best-EDL version parameterizes the passive branch with Titov-style transport-number "
            "contrast and an EDL-limited selectivity cap. "
            if edl_limited
            else ""
        )
        + "It is not a hidden length_scale, not a hidden zdc_scale, and not a default Niu 2020 published parameter."
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return summary


def write_best_edl_membrane_parameter_config(
    out: Path,
    *,
    diagnostic_summary: dict,
    verification: dict | None = None,
) -> dict:
    """Write a Chinese, auditable config for the best-EDL diagnostic membrane model."""

    verification = verification or {}
    literature = diagnostic_summary.get("literature_basis_structured", [])
    doi_lines = "\n".join(
        f"# - {item.get('citation_key', 'reference')}: {item.get('doi')}" for item in literature if item.get("doi")
    )
    config = {
        "status": diagnostic_summary.get("status"),
        "membrane_geometry_mode": diagnostic_summary.get("membrane_geometry_mode"),
        "passive_branch_alpha_source": diagnostic_summary.get("passive_branch_alpha_source"),
        "passive_transport_number_edl_limited_fraction": diagnostic_summary.get(
            "passive_transport_number_edl_limited_fraction"
        ),
        "edl_debye_length_m": diagnostic_summary.get("edl_debye_length_m"),
        "edl_thickness_multiplier": diagnostic_summary.get("edl_thickness_multiplier"),
        "edl_selectivity": diagnostic_summary.get("edl_selectivity"),
        "maximum_transport_number_difference": diagnostic_summary.get("maximum_transport_number_difference"),
        "edl_selection_rule": diagnostic_summary.get("edl_selection_rule"),
        "edl_selection_rmse_tolerance": diagnostic_summary.get("edl_selection_rmse_tolerance"),
        "edl_selected_peak_normalized_rmse": diagnostic_summary.get("edl_selected_peak_normalized_rmse"),
        "component_spectrum_csv": diagnostic_summary.get("component_spectrum_csv"),
        "fullgrid_summary_json": diagnostic_summary.get("fullgrid_summary_json"),
        "verification_report_json": verification.get("report_json"),
        "verification_passed": verification.get("passed"),
        "verification_criteria": verification.get("criteria", {}),
        "notes": {
            "zh": (
                "该配置记录 best-EDL 膜极化诊断参数。它不是 length_scale，不是 zdc_scale，"
                "也不是 Niu 2020 正文公开给出的默认参数；它把文献支持的 active/passive "
                "膜极化框架、Titov-style transport-number geometry factor 与 EDL-limited "
                "selectivity cap 显式写入可复跑结果。"
            )
        },
    }
    text = f'''# Niu 2020 Berea best-EDL 膜极化诊断参数
# 生成位置：{out}
#
# 目的：
# - 记录当前最接近 Niu 2020 Figure 8 membrane component 的膜极化参数口径；
# - 让每个关键参数都有中文解释、文献来源和 verifier 证据；
# - 明确该模型不是 length_scale，不是 zdc_scale，也不是隐藏缩放。
#
# 物理来源简述：
# - Marshall-Madden: active/passive ion-selective membrane impedance 框架；
# - Titov et al.: active/passive transport-number contrast 与几何因子控制膜极化强度；
# - Buecker-Hoerdt: SNP/LNP 双时间尺度说明 active/passive 长度都可能进入弛豫；
# - Buecker et al. 2019: Stern/diffuse layer 作用提醒该分支仍应标为 diagnostic。
#
# 参考 DOI：
{doi_lines}

NIU2020_BEST_EDL_MEMBRANE_CONFIG = {json.dumps(config, indent=4, ensure_ascii=False, allow_nan=False)}
'''
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    result = {"config_py": str(out), **config}
    return result


def write_manifest(result_dir: Path, records: dict) -> None:
    manifest = result_dir / "manifest.json"
    manifest.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Niu 2020 Berea Reproduction Result Package",
        "",
        "本目录集中保存本次复现的参数配置、SIP 对比图、Fiji/VTK 风格三维数字岩心可视化、孔隙网络可视化、Figure 4 风格孔径/孔喉分布图和 provenance。",
        "",
        "- `configs/`: 中文注释参数配置。",
        "- `figures/`: 论文图风格 PNG/SVG/PDF。",
        "- `segmented_core/`: 按 notebook 链路重映射的 `pore=0, solid=255` 二值 TIFF/RAW。",
        "- `digital_rock/`: 基于二值体的 Fiji 3D Viewer 风格交互 HTML；后续数字岩心可视化只保留这种风格。",
        "- `pore_network/`: 由 `run_segmented_core_pnextract_ballstick.py` 生成的 pnextract 输入、网络 CSV、交互 HTML、孔径/孔喉分布图。",
        "- `dynamic_pore_size/`: project-extracted 动态孔径 manifest、方向积分和 provenance（若本次已生成或登记）。",
        "- `source_data/`: 图件源数据。",
        "- `provenance/`: 脚本运行记录。",
        "",
        "Policy: extracted pore/throat geometry and geometry-derived Zdc are not post-scaled. The pore network is generated with the original pnextract algorithm and its built-in default medial-surface parameters unless a run config explicitly records otherwise.",
    ]
    dynamic_pore_size = records.get("dynamic_pore_size")
    if dynamic_pore_size:
        lines.extend(
            [
                "",
                "## Dynamic pore size",
                "",
                f"- 状态：`{dynamic_pore_size.get('status', 'recorded')}`。",
                f"- 来源：`{dynamic_pore_size.get('source')}`。",
                f"- manifest：`{dynamic_pore_size.get('manifest')}`。",
            ]
        )
        if dynamic_pore_size.get("lambda_iso_m") is not None:
            lines.append(f"- `lambda_iso_m = {dynamic_pore_size.get('lambda_iso_m')}` m。")
        if dynamic_pore_size.get("lambda_x_m") is not None:
            lines.append(
                "- 方向值："
                f"`lambda_x_m={dynamic_pore_size.get('lambda_x_m')}`, "
                f"`lambda_y_m={dynamic_pore_size.get('lambda_y_m')}`, "
                f"`lambda_z_m={dynamic_pore_size.get('lambda_z_m')}` m。"
            )
    formal_plan = records.get("formal_sweep_reproduction_plan")
    if formal_plan:
        lines.extend(
            [
                "",
                "## Formal full-grid sweep status",
                "",
                f"- 状态：`{formal_plan.get('status')}`。",
                f"- 详细说明：`{formal_plan.get('readme_md')}`。",
                f"- 机器可读命令计划：`{formal_plan.get('plan_json')}`。",
                "- 因此本结果包只记录参数、几何派生产物和正式 full-grid sweep 的待运行计划；不把论文工作簿 simulation 列冒充为本项目模拟曲线。",
            ]
        )
    diagnostic = records.get("diagnostic_membrane_model")
    if diagnostic:
        lines.extend(
            [
                "",
                "## Diagnostic membrane model",
                "",
                (
                    f"- 模式：`{diagnostic.get('membrane_geometry_mode')}`；"
                    f"passive area source：`{diagnostic.get('passive_area_source')}`；"
                    f"alpha：`{diagnostic.get('passive_branch_alpha')}`。"
                ),
                "- 该模型用于 active/passive 双长度膜极化诊断，不是 Niu 2020 已公开给出的默认参数，也不是隐藏的 `length_scale` 或 `zdc_scale`。",
            ]
        )
        if "edl_limited" in str(diagnostic.get("membrane_geometry_mode", "")):
            lines.append(
                "- EDL-limited 口径：使用 Titov-style transport-number contrast，并用 Debye length 与有效 EDL 厚度限制每条喉道的迁移数差异。"
            )
            if diagnostic.get("edl_debye_length_m") is not None:
                lines.append(
                    f"- Debye length：`{diagnostic.get('edl_debye_length_m')}` m；"
                    f"EDL thickness multiplier：`{diagnostic.get('edl_thickness_multiplier')}`；"
                    f"EDL-limited throat fraction：`{diagnostic.get('passive_transport_number_edl_limited_fraction')}`。"
                )
    verification = records.get("best_edl_membrane_verification")
    if verification:
        criteria = verification.get("criteria", {})
        lines.extend(
            [
                "",
                "## Best-EDL membrane verification",
                "",
                f"- 自动审计报告：`{verification.get('report_json')}`。",
                (
                    "- 判据：膜分量峰频一致，"
                    f"`peak_normalized_rmse <= {criteria.get('max_peak_normalized_rmse')}`，"
                    f"`{criteria.get('min_ratio_at_paper_peak')} <= ratio_at_paper_peak <= {criteria.get('max_ratio_at_paper_peak')}`，"
                    "full-grid sweep 收敛，且未使用 Figure8 paper Simulation 列作为本项目模拟结果。"
                ),
            ]
        )
    best_edl_config = records.get("best_edl_membrane_parameter_config")
    if best_edl_config:
        lines.extend(
            [
                "",
                "## Best-EDL membrane parameter config",
                "",
                f"- 中文参数配置：`{best_edl_config.get('config_py')}`。",
                f"- 模型口径：`{best_edl_config.get('membrane_geometry_mode')}`。",
            ]
        )
    model_card = records.get("best_edl_membrane_model_card")
    if model_card:
        lines.extend(
            [
                "",
                "## Best-EDL membrane model card",
                "",
                f"- 复核用模型卡：`{model_card}`。",
                "- 该文件浓缩记录文献依据、参数含义、验证结果、方向平均诊断和已知限制。",
            ]
        )
    package_verification = records.get("best_edl_membrane_package_verification")
    if package_verification:
        lines.extend(
            [
                "",
                "## Best-EDL package verification",
                "",
                f"- 包级审计报告：`{package_verification.get('report_json')}`。",
                (
                    "- 审计范围："
                    f"{package_verification.get('scope', 'Package-level audit for the diagnostic best-EDL membrane branch.')}"
                ),
                "- 该审计检查结果包中的曲线 summary、component verifier、中文参数、模型卡、机制图 provenance、manifest/README 是否互相一致。",
            ]
        )
    md = result_dir / "README.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def register_dynamic_pore_size_manifest(result_dir: Path, manifest_path: Path | None) -> dict:
    dynamic_dir = result_dir / "dynamic_pore_size"
    dynamic_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path is None:
        readme = dynamic_dir / "dynamic_pore_size_not_generated.md"
        readme.write_text(
            "\n".join(
                [
                    "# Dynamic pore size not generated",
                    "",
                    "本次结果包构建未运行 `compute_dynamic_pore_size.py`，因此不声明正式 project-extracted Lambda。",
                    "后续应先生成 `dynamic_pore_size/dynamic_pore_size.json`，再用 project-extracted 模式生成 pore/membrane/all spectra。",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "status": "not_generated_by_builder",
            "source": "missing_project_extracted_microct_laplace_field",
            "manifest": str(readme.relative_to(result_dir)),
            "reason": "compute_dynamic_pore_size.py was not run or not provided via --dynamic-pore-size-manifest",
        }

    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_manifest = dynamic_dir / "dynamic_pore_size.json"
    if manifest_path != target_manifest.resolve():
        shutil.copy2(manifest_path, target_manifest)
    for sibling_name in ("directional_integrals.csv", "config.yml", "input_manifest.json"):
        sibling = manifest_path.parent / sibling_name
        if sibling.exists():
            target = dynamic_dir / sibling_name
            if sibling.resolve() != target.resolve():
                shutil.copy2(sibling, target)
    provenance = manifest_path.parent / "provenance" / "dynamic_pore_size_provenance.md"
    if provenance.exists():
        target_provenance_dir = dynamic_dir / "provenance"
        target_provenance_dir.mkdir(exist_ok=True)
        target = target_provenance_dir / "dynamic_pore_size_provenance.md"
        if provenance.resolve() != target.resolve():
            shutil.copy2(provenance, target)
    return {
        "status": "recorded",
        "lambda_iso_m": manifest.get("lambda_iso_m"),
        "lambda_x_m": manifest.get("lambda_x_m"),
        "lambda_y_m": manifest.get("lambda_y_m"),
        "lambda_z_m": manifest.get("lambda_z_m"),
        "source": "project_extracted_microct_laplace_field",
        "manifest": str(target_manifest.relative_to(result_dir)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--legacy-network-dir", default=str(DEFAULT_LEGACY_NETWORK_DIR))
    parser.add_argument("--skip-visualizations", action="store_true")
    parser.add_argument("--skip-network-html", action="store_true")
    parser.add_argument("--dynamic-pore-size-manifest")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    python_exe = Path(args.python_exe)
    if not DEFAULT_PNEXTRACT_EXE.exists():
        raise FileNotFoundError(f"default pnextract executable is missing: {DEFAULT_PNEXTRACT_EXE}")
    if not DEFAULT_PNEXTRACT_PARAMETER_FILE.exists():
        raise FileNotFoundError(f"default pnextract parameter file is missing: {DEFAULT_PNEXTRACT_PARAMETER_FILE}")

    config_out = result_dir / "configs" / "niu2020_berea_parameters_zh.py"
    write_chinese_parameter_config(config_out)
    binary_summary = remap_niu_tiff_to_notebook_binary(
        DEFAULT_TIFF,
        result_dir / "segmented_core",
        sample_id=SAMPLE_ID,
        voxel_size_um=VOXEL_SIZE_UM,
    )

    figure4_summary = write_figure4_style_distribution(
        DEFAULT_FIGURE5,
        result_dir / "pore_network" / "niu2020_figure4_pore_node_throat_distribution.png",
        result_dir / "source_data" / "niu2020_figure4_pore_node_throat_distribution_source_data.csv",
    )
    dynamic_pore_size_summary = register_dynamic_pore_size_manifest(
        result_dir,
        Path(args.dynamic_pore_size_manifest) if args.dynamic_pore_size_manifest else None,
    )

    command_records = []
    missing_sweeps = {key: path for key, path in DEFAULT_FORMAL_SWEEP_PATHS.items() if not path.exists()}
    formal_sweep_plan = None
    if missing_sweeps:
        formal_sweep_plan = write_missing_sweep_reproduction_plan(result_dir, missing_sweeps)
    else:
        command_records.append(run_command(build_sip_plot_command(result_dir=result_dir, python_exe=python_exe), cwd=PROJECT_ROOT))
    if not args.skip_visualizations:
        for command in build_visualization_commands(result_dir=result_dir, python_exe=python_exe, binary=binary_summary):
            command_records.append(run_command(command, cwd=PROJECT_ROOT))
    if not args.skip_network_html:
        command_records.append(run_command(build_pnextract_ballstick_command(result_dir=result_dir, python_exe=python_exe, binary=binary_summary), cwd=PROJECT_ROOT))
        network_html_status = "generated by run_segmented_core_pnextract_ballstick.py"
    elif args.skip_network_html:
        network_html_status = "skipped by --skip-network-html"

    records = {
        "result_dir": str(result_dir),
        "chinese_parameter_config": str(config_out),
        "binary_remap": binary_summary,
        "figure4_distribution": figure4_summary,
        "dynamic_pore_size": dynamic_pore_size_summary,
        "notebook_reference": "notebooks/seged_DRP_and_PNM.ipynb: remap TIFF -> binary 0/255, render Fiji/VTK HTML, then run pnextract wrapper.",
        "network_html_status": network_html_status,
        "formal_sweep_reproduction_plan": formal_sweep_plan,
        "commands": command_records,
    }
    (result_dir / "provenance").mkdir(parents=True, exist_ok=True)
    (result_dir / "provenance" / "build_niu2020_berea_result_package_commands.json").write_text(
        json.dumps(command_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_manifest(result_dir, records)
    print(json.dumps(records, indent=2, ensure_ascii=False))
    failed = [record for record in command_records if record["returncode"] != 0]
    if failed:
        raise SystemExit(f"{len(failed)} command(s) failed; see {result_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
