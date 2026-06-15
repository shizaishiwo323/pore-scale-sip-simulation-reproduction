#!/usr/bin/env python3
"""Nature-style Figure 8 comparison using our mechanism-resolved sweeps only."""

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
    "experiment": "#0F4D92",
    "experiment_light": "#3775BA",
    "all": "#272727",
    "interfacial": "#7A2E2E",
    "pore": "#B64342",
    "membrane": "#E0907D",
    "neutral": "#707070",
    "grid_major": "#D0D0D0",
    "grid_minor": "#EAEAEA",
}

COMPONENTS = [
    ("interfacial", "Interfacial polarization", "o", ":"),
    ("pore", "Pore polarization", "^", "--"),
    ("membrane", "Membrane polarization", "s", "-"),
]


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
    ax.grid(True, which="major", color=COLORS["grid_major"], linewidth=0.45, alpha=0.7)
    ax.grid(True, which="minor", color=COLORS["grid_minor"], linewidth=0.25, alpha=0.55)
    ax.tick_params(axis="both", which="major", length=3.0, pad=1.5)
    ax.tick_params(axis="both", which="minor", length=1.8)


def load_component(path: Path, mechanism: str) -> pd.DataFrame:
    frame = pd.read_csv(path).sort_values("frequency_hz").reset_index(drop=True)
    omega = 2.0 * np.pi * frame["frequency_hz"].astype(float)
    frame["real_relative_permittivity"] = frame["effective_sigma_imag_s_m"] / (omega * EPSILON0_F_M)
    frame["mechanism"] = mechanism
    frame["data_source"] = f"our_{mechanism}_component"
    frame["imaginary_conductivity_magnitude_s_m"] = frame["effective_sigma_imag_s_m"].abs()
    frame["real_relative_permittivity_magnitude"] = frame["real_relative_permittivity"].abs()
    return frame


def collect_source_data(
    experiment: pd.DataFrame,
    components: dict[str, pd.DataFrame],
    all_curve: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    exp = experiment.copy()
    exp["figure_context"] = "paper_figure8_experiment"
    exp["data_source"] = "paper_experiment"
    exp["mechanism"] = "experiment"
    exp["imaginary_conductivity_magnitude_s_m"] = exp["imaginary_conductivity_s_m"].abs()
    exp["real_relative_permittivity_magnitude"] = exp["real_relative_permittivity"].abs()
    rows.append(
        exp[
            [
                "figure_context",
                "data_source",
                "mechanism",
                "frequency_hz",
                "imaginary_conductivity_s_m",
                "imaginary_conductivity_magnitude_s_m",
                "real_relative_permittivity",
                "real_relative_permittivity_magnitude",
            ]
        ]
    )

    for mechanism, frame in components.items():
        source = pd.DataFrame(
            {
                "figure_context": "our_component_gpu_fft",
                "data_source": frame["data_source"],
                "mechanism": mechanism,
                "frequency_hz": frame["frequency_hz"],
                "imaginary_conductivity_s_m": frame["effective_sigma_imag_s_m"],
                "imaginary_conductivity_magnitude_s_m": frame["imaginary_conductivity_magnitude_s_m"],
                "real_relative_permittivity": frame["real_relative_permittivity"],
                "real_relative_permittivity_magnitude": frame["real_relative_permittivity_magnitude"],
                "iterations": frame["iterations"],
                "relative_residual_norm": frame["relative_residual_norm"],
                "info": frame["info"],
                "rtol": frame["rtol"],
                "preconditioner": frame["preconditioner"],
                "dtype": frame["dtype"],
            }
        )
        rows.append(source)
    if all_curve is not None:
        source = pd.DataFrame(
            {
                "figure_context": "our_all_polarization_gpu_fft",
                "data_source": all_curve["data_source"],
                "mechanism": "all_polarization",
                "frequency_hz": all_curve["frequency_hz"],
                "imaginary_conductivity_s_m": all_curve["effective_sigma_imag_s_m"],
                "imaginary_conductivity_magnitude_s_m": all_curve["imaginary_conductivity_magnitude_s_m"],
                "real_relative_permittivity": all_curve["real_relative_permittivity"],
                "real_relative_permittivity_magnitude": all_curve["real_relative_permittivity_magnitude"],
                "iterations": all_curve["iterations"],
                "relative_residual_norm": all_curve["relative_residual_norm"],
                "info": all_curve["info"],
                "rtol": all_curve["rtol"],
                "preconditioner": all_curve["preconditioner"],
                "dtype": all_curve["dtype"],
            }
        )
        rows.append(source)
    return pd.concat(rows, ignore_index=True, sort=False)


def plot_component(
    ax: plt.Axes,
    frame: pd.DataFrame,
    y_col: str,
    *,
    color: str,
    label: str,
    marker: str,
    linestyle: str,
) -> None:
    ax.plot(
        frame["frequency_hz"],
        frame[y_col],
        color=color,
        label=label,
        marker=marker,
        linestyle=linestyle,
        linewidth=1.25,
        markersize=3.4,
        markerfacecolor="white",
        markeredgewidth=0.75,
        alpha=0.98,
    )


def make_figure(
    experiment: pd.DataFrame,
    components: dict[str, pd.DataFrame],
    output_base: Path,
    all_curve: pd.DataFrame | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), constrained_layout=True)

    ax = axes[0]
    ax.plot(
        experiment["frequency_hz"],
        experiment["imaginary_conductivity_s_m"],
        color=COLORS["experiment"],
        label="Paper experiment",
        marker="D",
        linestyle="None",
        markersize=3.8,
        markerfacecolor="white",
        markeredgewidth=0.75,
    )
    for key, label, marker, linestyle in COMPONENTS:
        plot_component(
            ax,
            components[key],
            "imaginary_conductivity_magnitude_s_m",
            color=COLORS[key],
            label=f"Our {label.lower()}",
            marker=marker,
            linestyle=linestyle,
        )
    if all_curve is not None:
        plot_component(
            ax,
            all_curve,
            "imaginary_conductivity_magnitude_s_m",
            color=COLORS["all"],
            label="Our all-polarization",
            marker="X",
            linestyle="-",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Figure 8: imaginary conductivity", fontsize=7.5, pad=3)
    style_log_axes(ax, xlabel="Frequency (Hz)", ylabel="|Imaginary conductivity| (S m$^{-1}$)")
    add_panel_label(ax, "a")

    ax = axes[1]
    ax.plot(
        experiment["frequency_hz"],
        experiment["real_relative_permittivity"].abs(),
        color=COLORS["experiment"],
        label="Paper experiment",
        marker="D",
        linestyle="None",
        markersize=3.8,
        markerfacecolor="white",
        markeredgewidth=0.75,
    )
    for key, label, marker, linestyle in COMPONENTS:
        plot_component(
            ax,
            components[key],
            "real_relative_permittivity_magnitude",
            color=COLORS[key],
            label=f"Our {label.lower()}",
            marker=marker,
            linestyle=linestyle,
        )
    if all_curve is not None:
        plot_component(
            ax,
            all_curve,
            "real_relative_permittivity_magnitude",
            color=COLORS["all"],
            label="Our all-polarization",
            marker="X",
            linestyle="-",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Figure 8: real permittivity", fontsize=7.5, pad=3)
    style_log_axes(ax, xlabel="Frequency (Hz)", ylabel="|Real relative permittivity|")
    add_panel_label(ax, "b")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=5 if all_curve is not None else 4,
        fontsize=6.2,
        handlelength=1.8,
        columnspacing=1.25,
    )
    for ax in axes:
        ax.margins(x=0.03)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_discussion(
    components: dict[str, pd.DataFrame],
    discussion_md: Path,
    all_curve: pd.DataFrame | None = None,
) -> None:
    rows = []
    for key, label, _, _ in COMPONENTS:
        frame = components[key]
        peak_imag = frame.loc[frame["imaginary_conductivity_magnitude_s_m"].idxmax()]
        hardest = frame.loc[frame["iterations"].idxmax()]
        rows.append(
            {
                "mechanism": label,
                "peak_frequency_hz": float(peak_imag["frequency_hz"]),
                "peak_abs_imag_s_m": float(peak_imag["imaginary_conductivity_magnitude_s_m"]),
                "hardest_frequency_hz": float(hardest["frequency_hz"]),
                "max_iterations": int(hardest["iterations"]),
                "max_residual": float(frame["relative_residual_norm"].max()),
            }
        )
    if all_curve is not None:
        peak_imag = all_curve.loc[all_curve["imaginary_conductivity_magnitude_s_m"].idxmax()]
        hardest = all_curve.loc[all_curve["iterations"].idxmax()]
        rows.append(
            {
                "mechanism": "All polarization",
                "peak_frequency_hz": float(peak_imag["frequency_hz"]),
                "peak_abs_imag_s_m": float(peak_imag["imaginary_conductivity_magnitude_s_m"]),
                "hardest_frequency_hz": float(hardest["frequency_hz"]),
                "max_iterations": int(hardest["iterations"]),
                "max_residual": float(all_curve["relative_residual_norm"].max()),
            }
        )
    summary = pd.DataFrame(rows)
    table_lines = [
        "| mechanism | peak_frequency_hz | peak_abs_imag_s_m | hardest_frequency_hz | max_iterations | max_residual |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        table_lines.append(
            f"| {row['mechanism']} | {row['peak_frequency_hz']:.6g} | {row['peak_abs_imag_s_m']:.3e} | "
            f"{row['hardest_frequency_hz']:.6g} | {int(row['max_iterations'])} | {row['max_residual']:.3e} |"
        )

    negative = []
    for key, label, _, _ in COMPONENTS:
        n_negative = int((components[key]["effective_sigma_imag_s_m"] < 0).sum())
        if n_negative:
            negative.append(f"{label}: {n_negative} low-frequency point(s)")

    discussion = [
        "# Figure 8 Mechanism-Resolved Comparison",
        "",
        "## Figure Contract",
        "",
        "- Claim: the reproduced 350^3 GPU-FFT simulations can be separated into operational interfacial, pore, and membrane polarization contributions and compared directly with the paper's experimental Figure 8 data.",
        "- Evidence: panel a compares imaginary conductivity; panel b compares the derived real relative permittivity.",
        "- Scope: the paper simulation curves are intentionally excluded from this figure.",
        "",
        "## Results",
        "",
        "\n".join(table_lines),
        "",
        "## Interpretation",
        "",
        "- The membrane-polarization run dominates the low- to mid-frequency response in the reproduced component split, which is consistent with the full all-polarization curve being largely controlled by this term around the main dispersion band.",
        "- The pore-polarization run is smaller over most frequencies, but it still contributes a resolvable low-frequency imaginary-conductivity shoulder.",
        "- The interfacial-only run is weak at low frequency and becomes important mainly at high frequency where dielectric contrast terms dominate.",
        *(
            [
                "- The all-polarization curve is plotted as the direct full material response; use it as the primary comparison to experiment, while the three mechanism curves show which enabled term controls each band."
            ]
            if all_curve is not None
            else []
        ),
        "",
        "## Review Risks",
        "",
        "- This is an operational decomposition based on selectively enabling terms in the current material spectrum, not a guaranteed one-to-one reconstruction of the paper authors' internal Figure 8 mechanism files.",
        "- The plotted y-values are magnitudes because the interfacial-only low-frequency imaginary response can be slightly negative under this sign convention and numerical setup.",
        "- All three component sweeps used the x-direction, complex64, full 350^3 Berea grid, GPU BiCGSTAB, FFT-Poisson preconditioning, and `rtol = 1e-5`; tensor-direction averaging remains a separate validation step.",
    ]
    if negative:
        discussion.extend(["", f"- Signed negative imaginary values occurred for: {', '.join(negative)}."])
    discussion_md.parent.mkdir(parents=True, exist_ok=True)
    discussion_md.write_text("\n".join(discussion) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(ROOT / "论文数据"))
    parser.add_argument(
        "--interfacial-csv",
        default=str(ROOT / "outputs" / "ac3d_gpu_full350_complex64_fft_component_interfacial_seq_rtol1e-5" / "sweep_results.csv"),
    )
    parser.add_argument(
        "--pore-csv",
        default=str(ROOT / "outputs" / "ac3d_gpu_full350_complex64_fft_component_pore_seq_rtol1e-5" / "sweep_results.csv"),
    )
    parser.add_argument(
        "--membrane-csv",
        default=str(ROOT / "outputs" / "ac3d_gpu_full350_complex64_fft_component_membrane_seq_rtol1e-5" / "sweep_results.csv"),
    )
    parser.add_argument(
        "--all-csv",
        default=None,
        help="Optional all-polarization sweep CSV to add as the combined response.",
    )
    parser.add_argument(
        "--source-data-csv",
        default=str(ROOT / "outputs" / "figure8_our_components_vs_experiment_source_data.csv"),
    )
    parser.add_argument(
        "--figure-base",
        default=str(ROOT / "figures" / "figure8_our_components_vs_experiment_nature"),
    )
    parser.add_argument(
        "--discussion-md",
        default=str(ROOT / "notes" / "figure8_our_components_vs_experiment_discussion.md"),
    )
    args = parser.parse_args()

    tables = read_paper_tables(Path(args.data_dir))
    experiment = tables["figure8_experiment"].sort_values("frequency_hz").reset_index(drop=True)
    components = {
        "interfacial": load_component(Path(args.interfacial_csv), "interfacial"),
        "pore": load_component(Path(args.pore_csv), "pore"),
        "membrane": load_component(Path(args.membrane_csv), "membrane"),
    }
    all_curve = load_component(Path(args.all_csv), "all_polarization") if args.all_csv else None

    source_data = collect_source_data(experiment, components, all_curve)
    source_path = Path(args.source_data_csv)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_data.to_csv(source_path, index=False)

    make_figure(experiment, components, Path(args.figure_base), all_curve)
    make_discussion(components, Path(args.discussion_md), all_curve)

    print(f"wrote {Path(args.figure_base).with_suffix('.svg')}")
    print(f"wrote {Path(args.figure_base).with_suffix('.pdf')}")
    print(f"wrote {Path(args.figure_base).with_suffix('.png')}")
    print(f"wrote {Path(args.figure_base).with_suffix('.tiff')}")
    print(f"wrote {source_path}")
    print(f"wrote {args.discussion_md}")


if __name__ == "__main__":
    main()
