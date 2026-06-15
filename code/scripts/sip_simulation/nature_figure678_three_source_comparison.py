#!/usr/bin/env python3
"""Nature-style three-source comparison for paper Figures 6-8."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_figure6_figure8_comparison import read_paper_tables  # noqa: E402


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["xtick.major.width"] = 0.6
plt.rcParams["ytick.major.width"] = 0.6
plt.rcParams["xtick.minor.width"] = 0.45
plt.rcParams["ytick.minor.width"] = 0.45
plt.rcParams["legend.frameon"] = False


EPSILON0_F_M = 8.8541878128e-12

COLORS = {
    "ours_main": "#B64342",
    "ours_light": "#E9A6A1",
    "experiment_main": "#0F4D92",
    "experiment_light": "#3775BA",
    "simulation_main": "#2F8F83",
    "simulation_light": "#77D7D1",
    "simulation_soft": "#B9E3DF",
    "neutral": "#707070",
}


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )


def style_log_axes(ax: plt.Axes, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, which="major", color="#D0D0D0", linewidth=0.45, alpha=0.7)
    ax.grid(True, which="minor", color="#EAEAEA", linewidth=0.25, alpha=0.55)
    ax.tick_params(axis="both", which="major", length=3.0, pad=1.5)
    ax.tick_params(axis="both", which="minor", length=1.8)


def plot_line(
    ax: plt.Axes,
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    color: str,
    label: str,
    linestyle: str = "-",
    marker: str | None = None,
    linewidth: float = 1.15,
    markersize: float = 3.2,
    alpha: float = 1.0,
) -> None:
    ax.plot(
        x,
        y,
        color=color,
        label=label,
        linestyle=linestyle,
        marker=marker,
        linewidth=linewidth,
        markersize=markersize,
        markerfacecolor="white" if marker else color,
        markeredgewidth=0.7,
        alpha=alpha,
    )


def collect_source_data(tables: dict[str, pd.DataFrame], ours: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    ours_long = pd.DataFrame(
        {
            "figure_context": "ours_full_grid_gpu_fft",
            "data_source": "our_simulation",
            "frequency_hz": ours["frequency_hz"],
            "real_conductivity_s_m": ours["effective_sigma_real_s_m"],
            "imaginary_conductivity_s_m": ours["effective_sigma_imag_s_m"],
            "real_relative_permittivity": ours["real_relative_permittivity"],
            "iterations": ours["iterations"],
            "relative_residual_norm": ours["relative_residual_norm"],
        }
    )
    rows.append(ours_long)

    for key, table in tables.items():
        source = "paper_experiment" if "experiment" in key else "paper_simulation"
        frame = table.copy()
        frame["figure_context"] = key
        frame["data_source"] = source
        rows.append(frame)

    return pd.concat(rows, ignore_index=True, sort=False)


def make_composite(tables: dict[str, pd.DataFrame], ours: pd.DataFrame, output_base: Path) -> None:
    fig = plt.figure(figsize=(7.2, 6.8), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0])
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[2, 0]),
        fig.add_subplot(grid[2, 1]),
    ]

    cond6 = tables["figure6_conductivity_experiment"]
    perm6 = tables["figure6_permittivity_experiment"]
    fig7_sim = tables["figure7_simulation_real"]
    fig7_exp = tables["figure7_experiment_real"]
    fig8_exp = tables["figure8_experiment"]
    fig8_sim = tables["figure8_simulation_all"]
    fig8_pore = tables["figure8_pore"]
    fig8_membrane = tables["figure8_membrane"]
    fig8_interfacial = tables["figure8_interfacial"]

    ax = axes[0]
    plot_line(ax, cond6["frequency_hz"], cond6["real_conductivity_s_m"], color=COLORS["experiment_main"], label="Paper experiment, real", marker="o")
    plot_line(ax, cond6["frequency_hz"], cond6["imaginary_conductivity_s_m"], color=COLORS["experiment_light"], label="Paper experiment, imaginary", marker="s", linestyle="--")
    plot_line(ax, ours["frequency_hz"], ours["effective_sigma_real_s_m"], color=COLORS["ours_main"], label="Our simulation, real", marker="^")
    plot_line(ax, ours["frequency_hz"], ours["effective_sigma_imag_s_m"], color=COLORS["ours_light"], label="Our simulation, imaginary", marker="v", linestyle="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Figure 6: complex conductivity", fontsize=7.5, pad=3)
    style_log_axes(ax, ylabel="Conductivity (S m$^{-1}$)")
    add_panel_label(ax, "a")

    ax = axes[1]
    plot_line(ax, perm6["frequency_hz"], perm6["real_relative_permittivity"], color=COLORS["experiment_main"], label="Paper experiment, real", marker="o")
    plot_line(ax, ours["frequency_hz"], ours["real_relative_permittivity"], color=COLORS["ours_main"], label="Our simulation, real", marker="^")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Figure 6: real permittivity", fontsize=7.5, pad=3)
    style_log_axes(ax, ylabel="Relative permittivity")
    add_panel_label(ax, "b")

    ax = axes[2]
    plot_line(ax, fig7_exp["frequency_hz"], fig7_exp["real_conductivity_s_m"], color=COLORS["experiment_main"], label="Paper experiment", marker="o", linestyle="None")
    plot_line(ax, fig7_sim["frequency_hz"], fig7_sim["real_conductivity_s_m"], color=COLORS["simulation_main"], label="Paper simulation", marker=None)
    plot_line(ax, ours["frequency_hz"], ours["effective_sigma_real_s_m"], color=COLORS["ours_main"], label="Our simulation", marker="^")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Figure 7: real conductivity", fontsize=7.5, pad=3)
    style_log_axes(ax, ylabel="Real conductivity (S m$^{-1}$)")
    add_panel_label(ax, "c")

    ax = axes[3]
    plot_line(ax, fig8_exp["frequency_hz"], fig8_exp["imaginary_conductivity_s_m"], color=COLORS["experiment_main"], label="Paper experiment", marker="o", linestyle="None")
    plot_line(ax, fig8_sim["frequency_hz"], fig8_sim["imaginary_conductivity_s_m"], color=COLORS["simulation_main"], label="Paper simulation, all", marker=None)
    plot_line(ax, fig8_pore["frequency_hz"], fig8_pore["imaginary_conductivity_s_m"], color=COLORS["simulation_light"], label="Paper simulation, pore", linestyle="--", alpha=0.8)
    plot_line(ax, fig8_membrane["frequency_hz"], fig8_membrane["imaginary_conductivity_s_m"], color=COLORS["simulation_light"], label="Paper simulation, membrane", linestyle="-.", alpha=0.8)
    plot_line(ax, fig8_interfacial["frequency_hz"], fig8_interfacial["imaginary_conductivity_s_m"], color=COLORS["simulation_soft"], label="Paper simulation, interfacial", linestyle=":", alpha=0.9)
    plot_line(ax, ours["frequency_hz"], ours["effective_sigma_imag_s_m"], color=COLORS["ours_main"], label="Our simulation", marker="^")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Figure 8: imaginary conductivity", fontsize=7.5, pad=3)
    style_log_axes(ax, ylabel="Imaginary conductivity (S m$^{-1}$)")
    add_panel_label(ax, "d")

    ax = axes[4]
    plot_line(ax, fig8_exp["frequency_hz"], fig8_exp["real_relative_permittivity"], color=COLORS["experiment_main"], label="Paper experiment", marker="o", linestyle="None")
    plot_line(ax, fig8_sim["frequency_hz"], fig8_sim["real_relative_permittivity"], color=COLORS["simulation_main"], label="Paper simulation, all")
    plot_line(ax, fig8_pore["frequency_hz"], fig8_pore["real_relative_permittivity"], color=COLORS["simulation_light"], label="Paper simulation, pore", linestyle="--", alpha=0.8)
    plot_line(ax, fig8_membrane["frequency_hz"], fig8_membrane["real_relative_permittivity"], color=COLORS["simulation_light"], label="Paper simulation, membrane", linestyle="-.", alpha=0.8)
    plot_line(ax, fig8_interfacial["frequency_hz"], fig8_interfacial["real_relative_permittivity"], color=COLORS["simulation_soft"], label="Paper simulation, interfacial", linestyle=":", alpha=0.9)
    plot_line(ax, ours["frequency_hz"], ours["real_relative_permittivity"], color=COLORS["ours_main"], label="Our simulation", marker="^")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Figure 8: real permittivity", fontsize=7.5, pad=3)
    style_log_axes(ax, xlabel="Frequency (Hz)", ylabel="Relative permittivity")
    add_panel_label(ax, "e")

    ax = axes[5]
    ax.axis("off")
    legend_specs = [
        ("Our simulation", COLORS["ours_main"], "^", "-"),
        ("Paper experiment", COLORS["experiment_main"], "o", "-"),
        ("Paper simulation", COLORS["simulation_main"], None, "-"),
        ("Paper simulation sub-mechanisms", COLORS["simulation_light"], None, "--"),
    ]
    y0 = 0.86
    for i, (label, color, marker, linestyle) in enumerate(legend_specs):
        y = y0 - 0.12 * i
        ax.plot([0.08, 0.24], [y, y], color=color, linestyle=linestyle, linewidth=1.4, transform=ax.transAxes)
        if marker:
            ax.plot([0.16], [y], color=color, marker=marker, markerfacecolor="white", markersize=4, transform=ax.transAxes)
        ax.text(0.29, y, label, transform=ax.transAxes, va="center", fontsize=7)
    ax.text(
        0.08,
        0.28,
        "Our curve: full 350$^3$ Berea, GPU FFT-preconditioned Krylov, rtol $=10^{-5}$.",
        transform=ax.transAxes,
        fontsize=6.7,
        va="top",
        color=COLORS["neutral"],
        wrap=True,
    )
    add_panel_label(ax, "f")

    for ax in axes[:5]:
        ax.legend(fontsize=5.6, loc="best", handlelength=1.8, borderaxespad=0.25)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_results_discussion(ours: pd.DataFrame, metrics: pd.DataFrame, output_md: Path) -> None:
    converged = int((ours["info"] == 0).sum())
    n_points = len(ours)
    max_resid = float(ours["relative_residual_norm"].max())
    hardest = ours.sort_values("iterations", ascending=False).head(3)[["frequency_hz", "iterations", "relative_residual_norm"]]
    one_hz = ours.loc[np.isclose(ours["frequency_hz"], 1.0)].iloc[0]
    hardest_lines = ["| frequency_hz | iterations | relative_residual_norm |", "|---:|---:|---:|"]
    for _, row in hardest.iterrows():
        hardest_lines.append(
            f"| {float(row['frequency_hz']):.6g} | {int(row['iterations'])} | {float(row['relative_residual_norm']):.3e} |"
        )
    discussion = [
        "# Figure 6-8 Three-Source Comparison: Results and Discussion",
        "",
        "## Result Summary",
        "",
        f"- The full-grid GPU-FFT sweep contains {n_points} representative frequencies; {converged}/{n_points} reached `info = 0`.",
        f"- The maximum final relative residual is `{max_resid:.3e}`, so the plotted curve is no longer dominated by linear-solver non-convergence.",
        f"- At `1 Hz`, the reproduced effective conductivity is `{one_hz['effective_sigma_real_s_m']:.6g} + {one_hz['effective_sigma_imag_s_m']:.6g} i S/m`.",
        "",
        "## Hardest Frequencies",
        "",
        "\n".join(hardest_lines),
        "",
        "## Interpretation",
        "",
        "- The reproduced real conductivity follows the same broad frequency-increasing trend as the paper data, but its magnitude remains offset from both the paper simulation and experiment over several bands.",
        "- The imaginary conductivity and derived real permittivity show the largest deviations, especially where polarization mechanisms dominate. This suggests the remaining mismatch is more likely controlled by material-parameter assumptions, polarization-mechanism partitioning, tensor-direction averaging, or phase assignment than by Krylov convergence.",
        "- The paper Figure 8 simulation decomposes pore, membrane, and interfacial terms; the current reproduced curve corresponds to the available all-polarization material model in a single imposed-field direction, so it should be interpreted as a first full-grid reproduction curve rather than a final one-to-one mechanism-resolved reproduction.",
        "",
        "## Next Checks",
        "",
        "- Add `y` and `z` imposed-field directions and compare the diagonal/tensor average with the paper curves.",
        "- Run key frequencies in `complex128` or verify final residuals with a higher-precision checkpoint.",
        "- Reproduce the paper's separate pore, membrane, and interfacial polarization input sets before interpreting Figure 8 mechanism-level discrepancies.",
    ]
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(discussion) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "论文数据"))
    parser.add_argument("--ours-csv", default=str(ROOT / "outputs" / "figure6_figure8_stage7_gpu_fft_sweep.csv"))
    parser.add_argument("--metrics-csv", default=str(ROOT / "outputs" / "figure6_figure8_stage7_gpu_fft_metrics.csv"))
    parser.add_argument("--source-data-csv", default=str(ROOT / "outputs" / "figure678_three_source_comparison_source_data.csv"))
    parser.add_argument("--figure-base", default=str(ROOT / "figures" / "figure678_three_source_comparison_nature"))
    parser.add_argument("--discussion-md", default=str(ROOT / "notes" / "figure678_results_discussion.md"))
    args = parser.parse_args()

    tables = read_paper_tables(Path(args.data_dir))
    ours = pd.read_csv(args.ours_csv)
    metrics = pd.read_csv(args.metrics_csv)
    source_data = collect_source_data(tables, ours)
    source_data.to_csv(args.source_data_csv, index=False)
    make_composite(tables, ours, Path(args.figure_base))
    make_results_discussion(ours, metrics, Path(args.discussion_md))

    print(f"wrote {Path(args.figure_base).with_suffix('.svg')}")
    print(f"wrote {Path(args.figure_base).with_suffix('.pdf')}")
    print(f"wrote {Path(args.figure_base).with_suffix('.png')}")
    print(f"wrote {Path(args.figure_base).with_suffix('.tiff')}")
    print(f"wrote {args.source_data_csv}")
    print(f"wrote {args.discussion_md}")


if __name__ == "__main__":
    main()
