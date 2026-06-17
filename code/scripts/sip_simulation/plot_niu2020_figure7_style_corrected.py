#!/usr/bin/env python3
"""Plot corrected Niu 2020 AC3D reproduction in the paper Figure 7 style."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EPSILON0_F_M = 8.8541878128e-12
CORRECTION_TAG = "original_pnextract_defaults_no_geometry_scaling"
COMPONENT_CORRECTIONS = {
    "all": "includes_original_pnextract_membrane",
    "membrane": "original_pnextract_geometry",
    "pore": "unmodified_full350_ac3d",
    "interfacial": "precision_merged_low_frequency",
}
MECHANISMS = ["pore", "membrane", "interfacial", "all"]
LABELS = {
    "experiment": "Experimental data",
    "pore": "Pore polarization",
    "membrane": "Membrane polarization",
    "interfacial": "Interfacial polarization",
    "all": "Simulation (all polarizations)",
}
COLORS = {
    "pore": "green",
    "membrane": "red",
    "interfacial": "blue",
    "all": "#777777",
}
LINESTYLES = {
    "pore": "--",
    "membrane": (0, (2.0, 1.2)),
    "interfacial": (0, (3.0, 3.0)),
    "all": "-",
}


def read_numeric_block(path: Path, cols: list[int], names: list[str]) -> pd.DataFrame:
    table = pd.read_excel(path, sheet_name=0, header=None)
    block = table.iloc[2:, cols].copy()
    block.columns = names
    for name in names:
        block[name] = pd.to_numeric(block[name], errors="coerce")
    return block.dropna(subset=[names[0]]).reset_index(drop=True)


def read_figure8_experiment(data_dir: Path) -> pd.DataFrame:
    return read_numeric_block(
        data_dir / "Figure8.xlsx",
        [0, 1, 2],
        ["frequency_hz", "imaginary_conductivity_s_m", "real_relative_permittivity"],
    )


def add_real_relative_permittivity(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    omega = 2.0 * np.pi * out["frequency_hz"].to_numpy(dtype=float)
    out["real_relative_permittivity"] = out["effective_sigma_imag_s_m"].to_numpy(dtype=float) / (omega * EPSILON0_F_M)
    return out


def load_component_results(paths: dict[str, Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for mechanism in MECHANISMS:
        frame = add_real_relative_permittivity(pd.read_csv(paths[mechanism]))
        frame["mechanism"] = mechanism
        frame["source_csv"] = str(paths[mechanism])
        frame["correction"] = CORRECTION_TAG
        frame["component_correction"] = COMPONENT_CORRECTIONS[mechanism]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def collect_source_data(experiment: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    component_correction = components.get("component_correction")
    if component_correction is None:
        component_correction = components["mechanism"].map(COMPONENT_CORRECTIONS)
    exp = pd.DataFrame(
        {
            "dataset": "experiment",
            "frequency_hz": experiment["frequency_hz"],
            "imaginary_conductivity_s_m": experiment["imaginary_conductivity_s_m"],
            "real_relative_permittivity": experiment["real_relative_permittivity"],
        }
    )
    sim = pd.DataFrame(
        {
            "dataset": "our_corrected_" + components["mechanism"].astype(str),
            "correction": CORRECTION_TAG,
            "frequency_hz": components["frequency_hz"],
            "real_conductivity_s_m": components.get("effective_sigma_real_s_m"),
            "imaginary_conductivity_s_m": components["effective_sigma_imag_s_m"],
            "imaginary_conductivity_magnitude_s_m": components["effective_sigma_imag_s_m"].abs(),
            "real_relative_permittivity": components["real_relative_permittivity"],
            "real_relative_permittivity_magnitude": components["real_relative_permittivity"].abs(),
            "iterations": components.get("iterations"),
            "relative_residual_norm": components.get("relative_residual_norm"),
            "info": components.get("info"),
            "source_csv": components.get("source_csv"),
            "component_correction": component_correction,
        }
    )
    return pd.concat([exp, sim], ignore_index=True, sort=False)


def positive(values: pd.Series | np.ndarray, floor: float = 1.0e-30) -> np.ndarray:
    return np.maximum(np.abs(np.asarray(values, dtype=float)), floor)


def plot_component(ax: plt.Axes, frame: pd.DataFrame, mechanism: str, y_col: str) -> None:
    ordered = frame.loc[frame["mechanism"] == mechanism].sort_values("frequency_hz")
    linewidth = 2.8 if mechanism == "all" else 1.35
    ax.loglog(
        ordered["frequency_hz"],
        positive(ordered[y_col]),
        color=COLORS[mechanism],
        linestyle=LINESTYLES[mechanism],
        linewidth=linewidth,
        label=LABELS[mechanism],
    )


def make_figure(experiment: pd.DataFrame, components: pd.DataFrame, output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 10,
            "axes.linewidth": 0.85,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": False,
            "ytick.right": False,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 8.1), constrained_layout=True)
    panels = [
        ("real_relative_permittivity", "Effective permittivity $\\epsilon'_{eff}/\\epsilon_0$", "(a)"),
        ("effective_sigma_imag_s_m", "Effective imaginary conductivity $\\sigma''_{eff}$, S m$^{-1}$", "(b)"),
    ]

    for ax, (sim_col, ylabel, panel_label) in zip(axes, panels):
        exp_col = "real_relative_permittivity" if sim_col == "real_relative_permittivity" else "imaginary_conductivity_s_m"
        ax.loglog(
            experiment["frequency_hz"],
            positive(experiment[exp_col]),
            marker="^",
            linestyle="None",
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.75,
            label=LABELS["experiment"],
            zorder=5,
        )
        for mechanism in MECHANISMS:
            plot_component(ax, components, mechanism, sim_col)
        ax.set_xlim(1.0e-3, 1.0e10)
        ax.set_xlabel("Frequency $f$, Hz")
        ax.set_ylabel(ylabel)
        ax.text(-0.095, 1.02, panel_label, transform=ax.transAxes, ha="left", va="bottom", fontsize=20)
        ax.grid(False)
        ax.tick_params(which="major", length=6, width=0.85)
        ax.tick_params(which="minor", length=3, width=0.65)

    axes[0].set_ylim(1.0, 1.0e10)
    axes[1].set_ylim(1.0e-7, 1.0e1)
    axes[0].legend(loc="upper right", frameon=False, fontsize=10, handlelength=2.6)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {".png": {"dpi": 420}, ".svg": {}, ".pdf": {}}.items():
        fig.savefig(output_base.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def write_summary(components: pd.DataFrame, output_md: Path) -> None:
    lines = [
        "# Corrected Niu 2020 Figure 7-Style SIP Reproduction",
        "",
        "This figure uses Niu 2020 experimental columns as scatter data and this project's corrected AC3D sweeps as mechanism curves.",
        "The membrane component uses original pnextract geometry without post-extraction length or Zdc scaling; paper simulation curves are not used as plotted input.",
        "The interfacial component uses the precision-merged low-frequency AC3D curve because the original complex64 low-frequency sweep is dominated by residual-floor error after division by omega epsilon0.",
        "",
        "| mechanism | points | converged info=0 | max residual | source |",
        "|---|---:|---:|---:|---|",
    ]
    for mechanism in MECHANISMS:
        frame = components.loc[components["mechanism"] == mechanism]
        converged = int((frame["info"] == 0).sum()) if "info" in frame else 0
        max_resid = float(frame["relative_residual_norm"].max()) if "relative_residual_norm" in frame else float("nan")
        source = str(frame["source_csv"].iloc[0]) if len(frame) else ""
        lines.append(f"| {mechanism} | {len(frame)} | {converged} | {max_resid:.3e} | `{source}` |")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_component_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "all": Path(args.all_csv),
        "pore": Path(args.pore_csv),
        "membrane": Path(args.membrane_csv),
        "interfacial": Path(args.interfacial_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    default_result_dir = PROJECT_ROOT / "results" / "niu2020_berea_reproduction_20260617_original_pnextract_defaults"
    default_sweep_dir = default_result_dir / "simulation_sweeps"
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "Niu 2020data"))
    parser.add_argument(
        "--all-csv",
        default=str(default_sweep_dir / "niu2020_berea_full350_all_original_pnextract_fft_x" / "sweep_results.csv"),
    )
    parser.add_argument(
        "--pore-csv",
        default=str(default_sweep_dir / "niu2020_berea_full350_pore_fft_x" / "sweep_results.csv"),
    )
    parser.add_argument(
        "--membrane-csv",
        default=str(default_sweep_dir / "niu2020_berea_full350_membrane_original_pnextract_fft_x" / "sweep_results.csv"),
    )
    parser.add_argument(
        "--interfacial-csv",
        default=str(default_sweep_dir / "niu2020_berea_full350_interfacial_precision_merged" / "sweep_results.csv"),
    )
    parser.add_argument(
        "--figure-base",
        default=str(PROJECT_ROOT / "figures" / "niu2020" / "niu2020_figure7_style_corrected_reproduction"),
    )
    parser.add_argument(
        "--source-data-csv",
        default=str(PROJECT_ROOT / "results" / "source_data" / "niu2020_figure7_style_corrected_reproduction_source_data.csv"),
    )
    parser.add_argument(
        "--summary-md",
        default=str(PROJECT_ROOT / "results" / "niu2020" / "niu2020_figure7_style_corrected_reproduction_summary.md"),
    )
    args = parser.parse_args()

    experiment = read_figure8_experiment(Path(args.data_dir))
    components = load_component_results(parse_component_paths(args))
    source_data = collect_source_data(experiment, components)
    source_path = Path(args.source_data_csv)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_data.to_csv(source_path, index=False)
    make_figure(experiment, components, Path(args.figure_base))
    write_summary(components, Path(args.summary_md))
    print(f"wrote {Path(args.figure_base).with_suffix('.png')}")
    print(f"wrote {Path(args.figure_base).with_suffix('.svg')}")
    print(f"wrote {Path(args.figure_base).with_suffix('.pdf')}")
    print(f"wrote {source_path}")
    print(f"wrote {args.summary_md}")


if __name__ == "__main__":
    main()
