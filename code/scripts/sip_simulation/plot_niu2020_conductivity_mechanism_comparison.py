#!/usr/bin/env python3
"""Plot Niu 2020 Berea sigma' and sigma'' mechanism comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_DIR = PROJECT_ROOT / "results" / "niu2020_berea_reproduction_20260617_original_pnextract_defaults"
DEFAULT_SWEEP_DIR = DEFAULT_RESULT_DIR / "simulation_sweeps"
MECHANISMS = ["interfacial", "pore", "membrane", "all"]
FREQUENCY_X_LIMITS = (1.0e-4, 1.0e9)
SIGMA_IMAG_Y_LIMITS = (1.0e-7, 1.0e1)
LABELS = {
    "experiment": "Experiment",
    "interfacial": "Dielectric",
    "pore": "EDL",
    "membrane": "Membrane",
    "all": "EDL+Dielectric+Membrane",
}
COLORS = {
    "interfacial": "#f28e2b",
    "pore": "#4e91e4",
    "membrane": "#65c987",
    "all": "#f15b62",
}
LINESTYLES = {
    "interfacial": (0, (5, 5)),
    "pore": (0, (5, 5)),
    "membrane": (0, (5, 5)),
    "all": "-",
}
DIAGNOSTIC_MEMBRANE_PROVENANCE_FIELDS = [
    "edl_debye_length_m",
    "edl_thickness_multiplier",
    "edl_selectivity",
    "maximum_transport_number_difference",
    "passive_transport_number_edl_limited_fraction",
    "edl_selection_rule",
    "edl_selection_rmse_tolerance",
    "edl_selected_peak_normalized_rmse",
]
SUMMARY_FIELD_ALIASES = {
    "membrane_geometry_mode": ["diagnostic_membrane_geometry_mode"],
}
DEFAULT_SWEEPS = {
    "all": DEFAULT_SWEEP_DIR / "niu2020_berea_full350_all_original_pnextract_fft_x" / "sweep_results.csv",
    "pore": DEFAULT_SWEEP_DIR / "niu2020_berea_full350_pore_fft_x" / "sweep_results.csv",
    "membrane": DEFAULT_SWEEP_DIR / "niu2020_berea_full350_membrane_original_pnextract_fft_x" / "sweep_results.csv",
    "interfacial": DEFAULT_SWEEP_DIR / "niu2020_berea_full350_interfacial_precision_merged" / "sweep_results.csv",
}


def summary_value(summary: dict, field: str):
    candidates = [field, f"diagnostic_membrane_{field}", *SUMMARY_FIELD_ALIASES.get(field, [])]
    for candidate in candidates:
        if candidate in summary:
            return summary[candidate]
    return None


def required_summary_value(summary: dict, field: str, default=None):
    value = summary_value(summary, field)
    if value is None:
        if default is not None:
            return default
        raise KeyError(field)
    return value


def read_numeric_block(path: Path, cols: list[int], names: list[str]) -> pd.DataFrame:
    table = pd.read_excel(path, sheet_name=0, header=None)
    block = table.iloc[2:, cols].copy()
    block.columns = names
    for name in names:
        block[name] = pd.to_numeric(block[name], errors="coerce")
    return block.dropna(subset=[names[0]]).reset_index(drop=True)


def read_real_conductivity_experiment(data_dir: Path) -> pd.DataFrame:
    return read_numeric_block(data_dir / "Figure7.xlsx", [2, 3], ["frequency_hz", "real_conductivity_s_m"])


def read_imaginary_conductivity_experiment(data_dir: Path) -> pd.DataFrame:
    return read_numeric_block(data_dir / "Figure8.xlsx", [0, 1], ["frequency_hz", "imaginary_conductivity_s_m"])


def load_component_results(paths: dict[str, Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for mechanism in MECHANISMS:
        path = paths[mechanism]
        frame = pd.read_csv(path)
        frame["mechanism"] = mechanism
        frame["source_csv"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def load_diagnostic_membrane_candidate(fullgrid_csv: Path, summary_json: Path) -> pd.DataFrame:
    frame = pd.read_csv(fullgrid_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    frame["source_csv"] = str(fullgrid_csv)
    frame["diagnostic_status"] = summary.get("status", "diagnostic_not_default_niu2020_parameter")
    frame["membrane_geometry_mode"] = required_summary_value(summary, "membrane_geometry_mode")
    frame["passive_area_source"] = required_summary_value(summary, "passive_area_source")
    frame["passive_branch_alpha"] = float(required_summary_value(summary, "passive_branch_alpha"))
    for field in DIAGNOSTIC_MEMBRANE_PROVENANCE_FIELDS:
        value = summary_value(summary, field)
        if value is not None:
            frame[field] = value
    return frame


def load_diagnostic_all_candidate(fullgrid_csv: Path, summary_json: Path) -> pd.DataFrame:
    frame = load_diagnostic_membrane_candidate(fullgrid_csv, summary_json)
    return frame


def positive(values: pd.Series | np.ndarray, floor: float = 1.0e-30) -> np.ndarray:
    return np.maximum(np.abs(np.asarray(values, dtype=float)), floor)


def collect_source_data(
    real_experiment: pd.DataFrame,
    imag_experiment: pd.DataFrame,
    components: pd.DataFrame,
    diagnostic_membrane: pd.DataFrame | None = None,
    diagnostic_all_fullgrid: pd.DataFrame | None = None,
) -> pd.DataFrame:
    real = pd.DataFrame(
        {
            "dataset": "experiment_real",
            "frequency_hz": real_experiment["frequency_hz"],
            "real_conductivity_s_m": real_experiment["real_conductivity_s_m"],
            "source": "data/Niu 2020data/Figure7.xlsx columns 2-3",
            "paper_simulation_columns_used": False,
        }
    )
    imag = pd.DataFrame(
        {
            "dataset": "experiment_imag",
            "frequency_hz": imag_experiment["frequency_hz"],
            "imaginary_conductivity_s_m": imag_experiment["imaginary_conductivity_s_m"],
            "source": "data/Niu 2020data/Figure8.xlsx columns 0-1",
            "paper_simulation_columns_used": False,
        }
    )
    sim = pd.DataFrame(
        {
            "dataset": "simulation_" + components["mechanism"].astype(str),
            "mechanism": components["mechanism"],
            "frequency_hz": components["frequency_hz"],
            "real_conductivity_s_m": components["effective_sigma_real_s_m"],
            "imaginary_conductivity_s_m": components["effective_sigma_imag_s_m"],
            "imaginary_conductivity_magnitude_s_m": positive(components["effective_sigma_imag_s_m"]),
            "source_csv": components["source_csv"],
            "paper_simulation_columns_used": False,
        }
    )
    frames = [real, imag, sim]
    has_true_diagnostic_all = diagnostic_all_fullgrid is not None and not diagnostic_all_fullgrid.empty
    if diagnostic_membrane is not None and not diagnostic_membrane.empty:
        diagnostic_columns = {
            "dataset": "diagnostic_membrane_volume_area",
            "mechanism": "membrane",
            "frequency_hz": diagnostic_membrane["frequency_hz"],
            "real_conductivity_s_m": diagnostic_membrane["effective_sigma_real_s_m"],
            "imaginary_conductivity_s_m": diagnostic_membrane["effective_sigma_imag_s_m"],
            "imaginary_conductivity_magnitude_s_m": positive(diagnostic_membrane["effective_sigma_imag_s_m"]),
            "source_csv": diagnostic_membrane["source_csv"],
            "paper_simulation_columns_used": False,
            "diagnostic_status": diagnostic_membrane.get(
                "diagnostic_status",
                pd.Series("diagnostic_not_default_niu2020_parameter", index=diagnostic_membrane.index),
            ),
            "membrane_geometry_mode": diagnostic_membrane.get("membrane_geometry_mode"),
            "passive_area_source": diagnostic_membrane.get("passive_area_source"),
            "passive_branch_alpha": diagnostic_membrane.get("passive_branch_alpha"),
        }
        for field in DIAGNOSTIC_MEMBRANE_PROVENANCE_FIELDS:
            if field in diagnostic_membrane.columns:
                diagnostic_columns[field] = diagnostic_membrane[field]
        diagnostic = pd.DataFrame(diagnostic_columns)
        frames.append(diagnostic)
        all_frame = components.loc[
            components["mechanism"] == "all",
            ["frequency_hz", "effective_sigma_real_s_m", "effective_sigma_imag_s_m", "source_csv"],
        ].copy()
        membrane_frame = components.loc[
            components["mechanism"] == "membrane",
            ["frequency_hz", "effective_sigma_real_s_m", "effective_sigma_imag_s_m", "source_csv"],
        ].copy()
        if not has_true_diagnostic_all and not all_frame.empty and not membrane_frame.empty:
            all_frame.rename(
                columns={
                    "effective_sigma_real_s_m": "all_real_s_m",
                    "effective_sigma_imag_s_m": "all_imag_s_m",
                    "source_csv": "all_source_csv",
                },
                inplace=True,
            )
            membrane_frame.rename(
                columns={
                    "effective_sigma_real_s_m": "membrane_real_s_m",
                    "effective_sigma_imag_s_m": "membrane_imag_s_m",
                    "source_csv": "membrane_source_csv",
                },
                inplace=True,
            )
            diagnostic_membrane_columns = [
                "frequency_hz",
                "effective_sigma_real_s_m",
                "effective_sigma_imag_s_m",
                "source_csv",
                "diagnostic_status",
                "membrane_geometry_mode",
                "passive_area_source",
                "passive_branch_alpha",
            ]
            diagnostic_membrane_columns.extend(
                field for field in DIAGNOSTIC_MEMBRANE_PROVENANCE_FIELDS if field in diagnostic_membrane.columns
            )
            diagnostic_all = (
                all_frame.merge(membrane_frame, on="frequency_hz", how="inner")
                .merge(
                    diagnostic_membrane[diagnostic_membrane_columns].rename(
                        columns={
                            "effective_sigma_real_s_m": "diagnostic_membrane_real_s_m",
                            "effective_sigma_imag_s_m": "diagnostic_membrane_imag_s_m",
                            "source_csv": "diagnostic_membrane_source_csv",
                        }
                    ),
                    on="frequency_hz",
                    how="inner",
                )
            )
            if not diagnostic_all.empty:
                diagnostic_all_columns = {
                    "dataset": "diagnostic_all_volume_area",
                    "mechanism": "all",
                    "frequency_hz": diagnostic_all["frequency_hz"],
                    "real_conductivity_s_m": diagnostic_all["all_real_s_m"]
                    - diagnostic_all["membrane_real_s_m"]
                    + diagnostic_all["diagnostic_membrane_real_s_m"],
                    "imaginary_conductivity_s_m": diagnostic_all["all_imag_s_m"]
                    - diagnostic_all["membrane_imag_s_m"]
                    + diagnostic_all["diagnostic_membrane_imag_s_m"],
                    "source_csv": diagnostic_all["diagnostic_membrane_source_csv"],
                    "paper_simulation_columns_used": False,
                    "diagnostic_status": diagnostic_all["diagnostic_status"],
                    "membrane_geometry_mode": diagnostic_all["membrane_geometry_mode"],
                    "passive_area_source": diagnostic_all["passive_area_source"],
                    "passive_branch_alpha": diagnostic_all["passive_branch_alpha"],
                }
                for field in DIAGNOSTIC_MEMBRANE_PROVENANCE_FIELDS:
                    if field in diagnostic_all.columns:
                        diagnostic_all_columns[field] = diagnostic_all[field]
                diagnostic_all_out = pd.DataFrame(diagnostic_all_columns)
                diagnostic_all_out["imaginary_conductivity_magnitude_s_m"] = positive(
                    diagnostic_all_out["imaginary_conductivity_s_m"]
                )
                frames.append(diagnostic_all_out)
    if has_true_diagnostic_all:
        diagnostic_all_columns = {
            "dataset": "diagnostic_all_volume_area",
            "mechanism": "all",
            "frequency_hz": diagnostic_all_fullgrid["frequency_hz"],
            "real_conductivity_s_m": diagnostic_all_fullgrid["effective_sigma_real_s_m"],
            "imaginary_conductivity_s_m": diagnostic_all_fullgrid["effective_sigma_imag_s_m"],
            "imaginary_conductivity_magnitude_s_m": positive(diagnostic_all_fullgrid["effective_sigma_imag_s_m"]),
            "source_csv": diagnostic_all_fullgrid["source_csv"],
            "paper_simulation_columns_used": False,
            "diagnostic_status": diagnostic_all_fullgrid.get(
                "diagnostic_status",
                pd.Series("diagnostic_not_default_niu2020_parameter", index=diagnostic_all_fullgrid.index),
            ),
            "membrane_geometry_mode": diagnostic_all_fullgrid.get("membrane_geometry_mode"),
            "passive_area_source": diagnostic_all_fullgrid.get("passive_area_source"),
            "passive_branch_alpha": diagnostic_all_fullgrid.get("passive_branch_alpha"),
        }
        for field in DIAGNOSTIC_MEMBRANE_PROVENANCE_FIELDS:
            if field in diagnostic_all_fullgrid.columns:
                diagnostic_all_columns[field] = diagnostic_all_fullgrid[field]
        frames.append(pd.DataFrame(diagnostic_all_columns))
    return pd.concat(frames, ignore_index=True, sort=False)


def summarize_pair_against_experiment(source: pd.DataFrame, dataset: str) -> dict:
    paper = source.loc[source["dataset"] == "experiment_imag", ["frequency_hz", "imaginary_conductivity_s_m"]].copy()
    candidate = source.loc[source["dataset"] == dataset, ["frequency_hz", "imaginary_conductivity_s_m"]].copy()
    paper.rename(columns={"imaginary_conductivity_s_m": "paper_imag_s_m"}, inplace=True)
    candidate.rename(columns={"imaginary_conductivity_s_m": "candidate_imag_s_m"}, inplace=True)
    merged = paper.merge(candidate, on="frequency_hz", how="inner").sort_values("frequency_hz")
    if merged.empty:
        return {"common_frequency_count": 0}
    paper_values = merged["paper_imag_s_m"].to_numpy(dtype=float)
    candidate_values = merged["candidate_imag_s_m"].to_numpy(dtype=float)
    frequency = merged["frequency_hz"].to_numpy(dtype=float)
    peak_scale = float(np.nanmax(np.abs(paper_values)))
    ratios = candidate_values / paper_values
    paper_peak_idx = int(np.nanargmax(paper_values))
    candidate_peak_idx = int(np.nanargmax(candidate_values))
    return {
        "common_frequency_count": int(len(merged)),
        "peak_normalized_rmse": float(np.sqrt(np.nanmean(((candidate_values - paper_values) / peak_scale) ** 2))),
        "median_ratio": float(np.nanmedian(ratios)),
        "min_ratio": float(np.nanmin(ratios)),
        "max_ratio": float(np.nanmax(ratios)),
        "paper_peak_frequency_hz": float(frequency[paper_peak_idx]),
        "candidate_peak_frequency_hz": float(frequency[candidate_peak_idx]),
        "ratio_at_paper_peak": float(ratios[paper_peak_idx]),
    }


def summarize_source_data(source: pd.DataFrame) -> dict:
    paper_used = source.get("paper_simulation_columns_used", pd.Series(dtype=object)).dropna()
    summary = {
        "rows": int(len(source)),
        "datasets": sorted(str(value) for value in source["dataset"].dropna().unique()),
        "paper_simulation_columns_used_any": bool(paper_used.astype(bool).any()) if not paper_used.empty else False,
    }
    if "simulation_all" in set(source["dataset"].dropna()):
        summary["simulation_all_vs_experiment_imag"] = summarize_pair_against_experiment(source, "simulation_all")
    if "diagnostic_membrane_volume_area" in set(source["dataset"].dropna()):
        summary["diagnostic_membrane_volume_area_vs_experiment_imag"] = summarize_pair_against_experiment(
            source, "diagnostic_membrane_volume_area"
        )
    if "diagnostic_all_volume_area" in set(source["dataset"].dropna()):
        summary["diagnostic_all_volume_area_vs_experiment_imag"] = summarize_pair_against_experiment(
            source, "diagnostic_all_volume_area"
        )
    return summary


def plot_mechanism(ax: plt.Axes, components: pd.DataFrame, mechanism: str, y_col: str) -> None:
    frame = components.loc[components["mechanism"] == mechanism].sort_values("frequency_hz")
    width = 2.7 if mechanism == "all" else 1.6
    ax.loglog(
        frame["frequency_hz"],
        positive(frame[y_col]),
        color=COLORS[mechanism],
        linestyle=LINESTYLES[mechanism],
        linewidth=width,
        solid_capstyle="round",
        label=LABELS[mechanism],
    )


def make_figure(
    real_experiment: pd.DataFrame,
    imag_experiment: pd.DataFrame,
    components: pd.DataFrame,
    output_base: Path,
    title: str = "Berea sandstone",
    diagnostic_membrane: pd.DataFrame | None = None,
    diagnostic_all: pd.DataFrame | None = None,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 13,
            "axes.linewidth": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 9.6), constrained_layout=True)
    axes[0].set_title(title, fontsize=18, pad=14)

    ax = axes[0]
    ax.loglog(
        real_experiment["frequency_hz"],
        positive(real_experiment["real_conductivity_s_m"]),
        marker="v",
        linestyle="None",
        markersize=4.8,
        color="black",
        label=LABELS["experiment"],
    )
    plot_mechanism(ax, components, "all", "effective_sigma_real_s_m")
    if diagnostic_all is not None and not diagnostic_all.empty:
        frame = diagnostic_all.sort_values("frequency_hz")
        ax.loglog(
            frame["frequency_hz"],
            positive(frame["real_conductivity_s_m"]),
            color="#7f3c8d",
            linestyle=":",
            linewidth=2.4,
            solid_capstyle="round",
            label="All diagnostic",
        )
    ax.set_ylabel("$\\sigma'$ (S/m)")
    ax.set_xlim(*FREQUENCY_X_LIMITS)
    ax.set_ylim(1.0e-4, 1.0e-1)
    ax.legend(loc="upper left", frameon=False, fontsize=12, handlelength=2.8)

    ax = axes[1]
    ax.loglog(
        imag_experiment["frequency_hz"],
        positive(imag_experiment["imaginary_conductivity_s_m"]),
        marker="o",
        linestyle="None",
        markersize=3.7,
        color="black",
        label=LABELS["experiment"],
    )
    for mechanism in ["interfacial", "pore", "membrane", "all"]:
        plot_mechanism(ax, components, mechanism, "effective_sigma_imag_s_m")
    if diagnostic_all is not None and not diagnostic_all.empty:
        frame = diagnostic_all.sort_values("frequency_hz")
        ax.loglog(
            frame["frequency_hz"],
            positive(frame["imaginary_conductivity_s_m"]),
            color="#7f3c8d",
            linestyle=":",
            linewidth=2.4,
            solid_capstyle="round",
            label="All diagnostic",
        )
    if diagnostic_membrane is not None and not diagnostic_membrane.empty:
        frame = diagnostic_membrane.sort_values("frequency_hz")
        ax.loglog(
            frame["frequency_hz"],
            positive(frame["effective_sigma_imag_s_m"]),
            color="#2ca02c",
            linestyle=":",
            linewidth=2.1,
            solid_capstyle="round",
            label="Membrane diagnostic",
        )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("$\\sigma''$ (S/m)")
    ax.set_xlim(*FREQUENCY_X_LIMITS)
    ax.set_ylim(*SIGMA_IMAG_Y_LIMITS)
    ax.legend(loc="upper left", frameon=False, fontsize=12, handlelength=2.8)

    for axis in axes:
        axis.grid(False)
        axis.tick_params(which="major", length=6, width=1.0)
        axis.tick_params(which="minor", length=3, width=0.8)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {".png": {"dpi": 420}, ".svg": {}, ".pdf": {}}.items():
        fig.savefig(output_base.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_provenance(
    path: Path,
    data_dir: Path,
    component_paths: dict[str, Path],
    source_data_csv: Path,
    diagnostic_membrane: dict | None = None,
    diagnostic_membrane_verification: dict | None = None,
    diagnostic_all: dict | None = None,
) -> None:
    spectra_dir = PROJECT_ROOT / "results" / "spectra" / "niu2020_berea_fullres_original_pnextract_defaults"
    base_spectrum = spectra_dir / "polarization_spectra_from_pnextract.csv"
    component_spectra_dir = spectra_dir / "components_paper_mode"
    lines = [
        "# Niu 2020 Conductivity Mechanism Comparison Provenance",
        "",
        "This figure compares Niu 2020 Berea experimental conductivity data against local AC3D sweep outputs.",
        "",
        "## Allowed Inputs",
        "",
        f"- CT segmentation: `{data_dir / 'microCT_Berea.raw'}` and `{data_dir / 'microCT_Berea.tiff'}`.",
        f"- Real-conductivity experiment: `{data_dir / 'Figure7.xlsx'}` columns 2-3.",
        f"- Imaginary-conductivity experiment: `{data_dir / 'Figure8.xlsx'}` columns 0-1.",
        "- Figure8 paper simulation/component columns are not used as this project's simulation curves.",
        "",
        "## Polarization Input Spectra",
        "",
        f"- Base pnextract spectrum: `{base_spectrum}`.",
        f"- Paper-mode mechanism spectra directory: `{component_spectra_dir}`.",
        "- No post-extraction pore/throat geometry or geometry-derived Zdc scaling is used.",
        "- Pore/throat geometry must come from the original pnextract executable with built-in default medial-surface parameters unless explicit run metadata says otherwise.",
        "",
        "## Plotted Data",
        "",
        f"- Source data CSV: `{source_data_csv}`.",
        "- The plotted sigma'' simulation values use absolute magnitude for log-axis display; signed values are preserved in source data.",
        "",
        "## Simulation CSVs",
        "",
        "| mechanism | source CSV |",
        "|---|---|",
    ]
    for mechanism in MECHANISMS:
        lines.append(f"| {mechanism} | `{component_paths[mechanism]}` |")
    if diagnostic_membrane:
        lines.extend(
            [
                "",
                "## Diagnostic Membrane Candidate",
                "",
                f"- Status: `{diagnostic_membrane.get('status')}`.",
                f"- Geometry mode: `{diagnostic_membrane.get('membrane_geometry_mode')}`.",
                f"- Full-grid CSV: `{diagnostic_membrane.get('fullgrid_csv')}`.",
                f"- Summary JSON: `{diagnostic_membrane.get('summary_json')}`.",
                "- This optional curve is plotted as a diagnostic candidate only; it is not a hidden `length_scale` or `zdc_scale` and is not the default Niu 2020 published membrane parameterization.",
            ]
        )
        edl_fields = [
            field for field in DIAGNOSTIC_MEMBRANE_PROVENANCE_FIELDS if diagnostic_membrane.get(field) is not None
        ]
        if edl_fields:
            lines.extend(
                [
                    "",
                    "### EDL-limited transport-number parameters",
                    "",
                    "| field | value |",
                    "|---|---|",
                ]
            )
            for field in edl_fields:
                lines.append(f"| {field} | `{diagnostic_membrane.get(field)}` |")
    if diagnostic_membrane_verification:
        checks = diagnostic_membrane_verification.get("checks", {})
        peak_rmse = checks.get("peak_normalized_rmse_within_threshold", {}).get("value")
        peak_ratio = checks.get("ratio_at_paper_peak_within_threshold", {}).get("value")
        max_residual = checks.get("sweep_converged", {}).get("max_relative_residual_norm")
        lines.extend(
            [
                "",
                "## Diagnostic Membrane Verification",
                "",
                f"- Report JSON: `{diagnostic_membrane_verification.get('report_json')}`.",
                f"- Passed: `{diagnostic_membrane_verification.get('passed')}`.",
                f"- Failed checks: `{diagnostic_membrane_verification.get('failed_checks')}`.",
                f"- Peak-normalized RMSE: `{peak_rmse}`.",
                f"- Ratio at paper membrane peak: `{peak_ratio}`.",
                f"- Max relative residual norm: `{max_residual}`.",
            ]
        )
    if diagnostic_all:
        lines.extend(
            [
                "",
                "## Diagnostic All Candidate",
                "",
                f"- Status: `{diagnostic_all.get('status')}`.",
                f"- Geometry mode: `{diagnostic_all.get('membrane_geometry_mode')}`.",
                f"- Full-grid CSV: `{diagnostic_all.get('fullgrid_csv')}`.",
                f"- Summary JSON: `{diagnostic_all.get('summary_json')}`.",
                "- This optional curve is a true full-grid all-mechanism diagnostic candidate; it is not an algebraic replacement of the default all curve.",
            ]
        )
        edl_fields = [field for field in DIAGNOSTIC_MEMBRANE_PROVENANCE_FIELDS if diagnostic_all.get(field) is not None]
        if edl_fields:
            lines.extend(
                [
                    "",
                    "### Diagnostic all EDL-limited transport-number parameters",
                    "",
                    "| field | value |",
                    "|---|---|",
                ]
            )
            for field in edl_fields:
                lines.append(f"| {field} | `{diagnostic_all.get(field)}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_component_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "all": Path(args.all_csv),
        "pore": Path(args.pore_csv),
        "membrane": Path(args.membrane_csv),
        "interfacial": Path(args.interfacial_csv),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "Niu 2020data"))
    parser.add_argument("--all-csv", default=str(DEFAULT_SWEEPS["all"]))
    parser.add_argument("--pore-csv", default=str(DEFAULT_SWEEPS["pore"]))
    parser.add_argument("--membrane-csv", default=str(DEFAULT_SWEEPS["membrane"]))
    parser.add_argument("--interfacial-csv", default=str(DEFAULT_SWEEPS["interfacial"]))
    parser.add_argument(
        "--figure-base",
        default=str(DEFAULT_RESULT_DIR / "figures" / "niu2020_conductivity_mechanism_comparison"),
    )
    parser.add_argument(
        "--source-data-csv",
        default=str(DEFAULT_RESULT_DIR / "source_data" / "niu2020_conductivity_mechanism_comparison_source_data.csv"),
    )
    parser.add_argument("--summary-json", default=None)
    parser.add_argument(
        "--provenance-md",
        default=str(DEFAULT_RESULT_DIR / "provenance" / "niu2020_conductivity_mechanism_comparison_provenance.md"),
    )
    parser.add_argument(
        "--diagnostic-membrane-fullgrid-csv",
        default=None,
        help="Optional full-grid sweep_results.csv for a diagnostic membrane candidate.",
    )
    parser.add_argument(
        "--diagnostic-membrane-summary-json",
        default=None,
        help="Optional provenance summary JSON for the diagnostic membrane candidate.",
    )
    parser.add_argument(
        "--diagnostic-membrane-verification-json",
        default=None,
        help="Optional verifier report JSON for the diagnostic membrane candidate.",
    )
    parser.add_argument(
        "--diagnostic-all-fullgrid-csv",
        default=None,
        help="Optional full-grid sweep_results.csv for a true diagnostic all-mechanism candidate.",
    )
    parser.add_argument(
        "--diagnostic-all-summary-json",
        default=None,
        help="Optional provenance summary JSON for the true diagnostic all-mechanism candidate.",
    )
    parser.add_argument("--title", default="Niu 2020 Berea")
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    component_paths = parse_component_paths(args)
    real_experiment = read_real_conductivity_experiment(data_dir)
    imag_experiment = read_imaginary_conductivity_experiment(data_dir)
    components = load_component_results(component_paths)
    diagnostic_membrane = None
    diagnostic_all_fullgrid = None
    diagnostic_provenance = None
    diagnostic_membrane_verification = None
    diagnostic_all_provenance = None
    if args.diagnostic_membrane_fullgrid_csv or args.diagnostic_membrane_summary_json:
        if not (args.diagnostic_membrane_fullgrid_csv and args.diagnostic_membrane_summary_json):
            raise SystemExit(
                "--diagnostic-membrane-fullgrid-csv and --diagnostic-membrane-summary-json must be provided together"
            )
        diagnostic_membrane = load_diagnostic_membrane_candidate(
            Path(args.diagnostic_membrane_fullgrid_csv),
            Path(args.diagnostic_membrane_summary_json),
        )
        diagnostic_provenance = {
            "fullgrid_csv": Path(args.diagnostic_membrane_fullgrid_csv),
            "summary_json": Path(args.diagnostic_membrane_summary_json),
            "status": str(diagnostic_membrane["diagnostic_status"].iloc[0]),
            "membrane_geometry_mode": str(diagnostic_membrane["membrane_geometry_mode"].iloc[0]),
        }
        for field in DIAGNOSTIC_MEMBRANE_PROVENANCE_FIELDS:
            if field in diagnostic_membrane.columns:
                diagnostic_provenance[field] = diagnostic_membrane[field].iloc[0]
    if args.diagnostic_membrane_verification_json:
        verification_path = Path(args.diagnostic_membrane_verification_json)
        diagnostic_membrane_verification = json.loads(verification_path.read_text(encoding="utf-8"))
        diagnostic_membrane_verification["report_json"] = verification_path
    if args.diagnostic_all_fullgrid_csv or args.diagnostic_all_summary_json:
        if not (args.diagnostic_all_fullgrid_csv and args.diagnostic_all_summary_json):
            raise SystemExit("--diagnostic-all-fullgrid-csv and --diagnostic-all-summary-json must be provided together")
        diagnostic_all_fullgrid = load_diagnostic_all_candidate(
            Path(args.diagnostic_all_fullgrid_csv),
            Path(args.diagnostic_all_summary_json),
        )
        diagnostic_all_provenance = {
            "fullgrid_csv": Path(args.diagnostic_all_fullgrid_csv),
            "summary_json": Path(args.diagnostic_all_summary_json),
            "status": str(diagnostic_all_fullgrid["diagnostic_status"].iloc[0]),
            "membrane_geometry_mode": str(diagnostic_all_fullgrid["membrane_geometry_mode"].iloc[0]),
        }
        for field in DIAGNOSTIC_MEMBRANE_PROVENANCE_FIELDS:
            if field in diagnostic_all_fullgrid.columns:
                diagnostic_all_provenance[field] = diagnostic_all_fullgrid[field].iloc[0]
    source = collect_source_data(
        real_experiment,
        imag_experiment,
        components,
        diagnostic_membrane=diagnostic_membrane,
        diagnostic_all_fullgrid=diagnostic_all_fullgrid,
    )

    source_path = Path(args.source_data_csv)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(source_path, index=False)
    diagnostic_all = source.loc[source["dataset"] == "diagnostic_all_volume_area"].copy()
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summarize_source_data(source), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
    make_figure(
        real_experiment,
        imag_experiment,
        components,
        Path(args.figure_base),
        title=args.title,
        diagnostic_membrane=diagnostic_membrane,
        diagnostic_all=diagnostic_all,
    )
    write_provenance(
        Path(args.provenance_md),
        data_dir,
        component_paths,
        source_path,
        diagnostic_membrane=diagnostic_provenance,
        diagnostic_membrane_verification=diagnostic_membrane_verification,
        diagnostic_all=diagnostic_all_provenance,
    )

    print(f"wrote {Path(args.figure_base).with_suffix('.png')}")
    print(f"wrote {Path(args.figure_base).with_suffix('.svg')}")
    print(f"wrote {Path(args.figure_base).with_suffix('.pdf')}")
    print(f"wrote {source_path}")
    print(f"wrote {args.provenance_md}")


if __name__ == "__main__":
    main()
