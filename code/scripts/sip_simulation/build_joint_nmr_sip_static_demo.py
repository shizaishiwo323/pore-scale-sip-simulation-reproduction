"""Build the static Berea demonstration for the joint NMR--SIP paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import tifffile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiff", type=Path, required=True)
    parser.add_argument("--nmr-spectrum", type=Path, required=True)
    parser.add_argument("--sip-sweep", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--slice-index", type=int, default=175)
    return parser.parse_args()


def box(ax: plt.Axes, x: float, y: float, text: str, color: str) -> None:
    ax.add_patch(FancyBboxPatch((x, y), 0.24, 0.2, boxstyle="round,pad=0.02", fc=color, ec="none"))
    ax.text(x + 0.12, y + 0.1, text, ha="center", va="center", fontsize=5.5, weight="bold")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.out_dir / "figures"
    source_dir = args.out_dir / "source_data"
    provenance_dir = args.out_dir / "provenance"
    for directory in (figure_dir, source_dir, provenance_dir):
        directory.mkdir(exist_ok=True)

    volume = tifffile.imread(args.tiff)
    if not 0 <= args.slice_index < volume.shape[0]:
        raise ValueError(f"slice index {args.slice_index} outside 0..{volume.shape[0] - 1}")
    image = volume[args.slice_index]
    nmr = pd.read_csv(args.nmr_spectrum)
    sip = pd.read_csv(args.sip_sweep)
    required_nmr = {"t2_ms", "normalized_amplitude"}
    required_sip = {"frequency_hz", "effective_sigma_real_s_m", "effective_sigma_imag_s_m", "true_residual_passed"}
    if missing := required_nmr.difference(nmr.columns):
        raise ValueError(f"NMR spectrum missing {sorted(missing)}")
    if missing := required_sip.difference(sip.columns):
        raise ValueError(f"SIP sweep missing {sorted(missing)}")
    if not sip["true_residual_passed"].astype(str).str.lower().eq("true").all():
        raise ValueError("SIP sweep contains a frequency that failed the residual gate")

    nmr.loc[:, ["t2_ms", "normalized_amplitude"]].to_csv(source_dir / "berea_nmr_t2_source_data.csv", index=False)
    sip.loc[:, ["frequency_hz", "effective_sigma_real_s_m", "effective_sigma_imag_s_m", "true_residual_norm", "iterations"]].to_csv(
        source_dir / "berea_sip_source_data.csv", index=False
    )

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 5.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(5.0, 3.4), constrained_layout=True)

    ax = axes[0, 0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(ax, 0.02, 0.55, "Berea\ngeometry", "#dce6ef")
    box(ax, 0.38, 0.55, "NMR + SIP\nsolvers", "#d8eadf")
    box(ax, 0.74, 0.55, "$T_2$ + $\\sigma^*$\nresponses", "#e6dff0")
    for start, end in [((0.26, 0.65), (0.38, 0.65)), ((0.62, 0.65), (0.74, 0.65))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, lw=1, color="#46515c"))
    ax.text(0.5, 0.16, "Shared geometry; independent physics", ha="center", color="#46515c")
    ax.set_title("a  Framework", loc="left", weight="bold", fontsize=6.5)

    ax = axes[0, 1]
    ax.imshow(image, cmap="gray", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
    ax.set_title(f"b  Berea slice {args.slice_index}", loc="left", weight="bold", fontsize=6.5)

    ax = axes[1, 0]
    ax.plot(nmr["t2_ms"], nmr["normalized_amplitude"], color="#2f7d68", lw=1.8)
    peak = nmr.loc[nmr["normalized_amplitude"].idxmax()]
    ax.scatter([peak["t2_ms"]], [peak["normalized_amplitude"]], color="#b24a3b", s=16, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 1e5)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlabel(r"$T_2$ (ms)")
    ax.set_ylabel("Normalized amplitude")
    ax.grid(alpha=0.18, which="both")
    ax.set_title("c  NMR relaxation", loc="left", weight="bold", fontsize=6.5)

    ax = axes[1, 1]
    ax.plot(sip["frequency_hz"], sip["effective_sigma_real_s_m"], "o-", color="#315f78", lw=1.5, ms=3, label=r"$\sigma'$ ")
    ax.plot(sip["frequency_hz"], sip["effective_sigma_imag_s_m"].abs(), "s-", color="#c27a38", lw=1.5, ms=3, label=r"$|\sigma''|$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Effective conductivity (S m$^{-1}$)")
    ax.grid(alpha=0.18, which="both")
    ax.legend(loc="upper left")
    ax.set_title("d  SIP response", loc="left", weight="bold", fontsize=6.5)

    output = figure_dir / "joint_nmr_sip_static_demo.png"
    fig.savefig(output, dpi=100, facecolor="white")
    plt.close(fig)

    equation_output = figure_dir / "governing_equations.png"
    eq_fig = plt.figure(figsize=(5.0, 0.6))
    eq_fig.text(0.5, 0.69, r"NMR: $\partial_t M=D\nabla^2M-M/T_{2B}$,   $D\,\mathbf{n}\!\cdot\!\nabla M+\rho_sM=0$", ha="center", fontsize=8)
    eq_fig.text(0.5, 0.19, r"SIP: $\nabla\!\cdot[\sigma^*(\mathbf{x},\omega)\nabla u]=0$,   $\sigma^*_{\mathrm{eff}}=\langle J\rangle/\langle E\rangle$", ha="center", fontsize=8)
    eq_fig.savefig(equation_output, dpi=100, facecolor="white")
    plt.close(eq_fig)

    metadata = {
        "status": "static theoretical demonstration; not experimental validation",
        "geometry": {"input": str(args.tiff.resolve()), "shape": list(volume.shape), "slice_index": args.slice_index},
        "nmr": {"source": str(args.nmr_spectrum.resolve()), "peak_t2_ms": float(peak["t2_ms"]), "model": "porosity-grouped 2D representative-slice approximation"},
        "sip": {"source": str(args.sip_sweep.resolve()), "frequencies_hz": sip["frequency_hz"].tolist(), "all_residual_gates_passed": True, "model": "64^3 AC3D diagnostic crop"},
        "figure": str(output.resolve()),
        "equation_figure": str(equation_output.resolve()),
    }
    (provenance_dir / "static_demo_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    assert output.stat().st_size > 10_000
    assert equation_output.stat().st_size > 1_000
    print(output.resolve())


if __name__ == "__main__":
    main()
