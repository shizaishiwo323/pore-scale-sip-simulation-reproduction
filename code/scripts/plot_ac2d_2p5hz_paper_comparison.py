#!/usr/bin/env python3
"""Compare AC2D 2.5 Hz time series with the validation paper data."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAPER = PROJECT_ROOT / "docs" / "validation" / "2024gl111271-sup-0003-data set si-s02.csv"
DEFAULT_OURS = PROJECT_ROOT / "results" / "ac2d_microfluidic" / "interface_images_si03_mechanistic_v1" / "ac2d_sweep_results.csv"
DEFAULT_OUT = PROJECT_ROOT / "figures" / "ac2d_microfluidic" / "ac2d_2p5hz_si03_mechanistic_vs_paper.png"
DEFAULT_SOURCE = PROJECT_ROOT / "results" / "ac2d_microfluidic" / "interface_images_si03_mechanistic_v1" / "ac2d_2p5hz_paper_comparison_source_data.csv"
DEFAULT_LABEL = "AC2D mechanistic: SI03 sigma_w(t) + MW + Schwarz"


def load_paper_series(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep=";")
    return table.rename(
        columns={
            "Time (h)": "time_h",
            "Real conductivity (S/m)": "paper_sigma_real_s_m",
            "Imaginary conductivity (S/m)": "paper_sigma_imag_s_m",
            "CEC (mEq/g)": "paper_cec_meq_g",
        }
    )


def load_our_series(path: Path, mechanism: str, frequency_hz: float) -> pd.DataFrame:
    table = pd.read_csv(path)
    selected = table[(table["mechanism"] == mechanism) & np.isclose(table["frequency_hz"], frequency_hz)].copy()
    if selected.empty:
        raise ValueError(f"no rows for mechanism={mechanism!r}, frequency={frequency_hz}")
    return selected.rename(
        columns={
            "sigma_real_s_m": "our_sigma_real_s_m",
            "sigma_imag_s_m": "our_sigma_imag_s_m",
        }
    )[["time_h", "frame", "our_sigma_real_s_m", "our_sigma_imag_s_m"]]


def interpolation_mae(paper: pd.DataFrame, ours: pd.DataFrame, column_paper: str, column_ours: str) -> float:
    mask = (paper["time_h"] >= ours["time_h"].min()) & (paper["time_h"] <= ours["time_h"].max())
    if not mask.any():
        return float("nan")
    interpolated = np.interp(paper.loc[mask, "time_h"], ours["time_h"], ours[column_ours])
    return float(np.mean(np.abs(paper.loc[mask, column_paper].to_numpy(dtype=float) - interpolated)))


def add_zoom_inset(ax: plt.Axes, ours: pd.DataFrame, y_column: str, title: str) -> None:
    inset = inset_axes(ax, width="34%", height="38%", loc="upper right", borderpad=1.0)
    inset.plot(ours["time_h"], ours[y_column], color="#d62728", linewidth=1.4)
    inset.scatter(ours["time_h"], ours[y_column], color="#d62728", s=8)
    inset.set_title(title, fontsize=7)
    inset.tick_params(labelsize=6)
    inset.grid(True, alpha=0.25, linewidth=0.5)


def plot_comparison(paper: pd.DataFrame, ours: pd.DataFrame, out: Path, mechanism: str, model_label: str) -> None:
    real_mae = interpolation_mae(paper, ours, "paper_sigma_real_s_m", "our_sigma_real_s_m")
    imag_mae = interpolation_mae(paper, ours, "paper_sigma_imag_s_m", "our_sigma_imag_s_m")

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
    fig.suptitle("2.5 Hz conductivity time-series comparison", fontsize=14, weight="bold")

    axes[0].scatter(
        paper["time_h"],
        paper["paper_sigma_real_s_m"],
        color="black",
        marker="s",
        s=22,
        label="Paper SI-S02 measurement",
        zorder=3,
    )
    axes[0].plot(
        ours["time_h"],
        ours["our_sigma_real_s_m"],
        color="#d62728",
        linewidth=1.8,
        marker="o",
        markersize=3.2,
        label=model_label,
    )
    axes[0].set_ylabel("Real, sigma' (S/m)")
    axes[0].set_ylim(0.0, max(0.65, paper["paper_sigma_real_s_m"].max() * 1.08))
    axes[0].text(0.98, 0.92, f"MAE = {real_mae:.3g} S/m", transform=axes[0].transAxes, ha="right", va="top", fontsize=8)
    add_zoom_inset(axes[0], ours, "our_sigma_real_s_m", "AC2D zoom")

    axes[1].scatter(
        paper["time_h"],
        paper["paper_sigma_imag_s_m"],
        color="black",
        marker="s",
        s=22,
        label="Paper SI-S02 measurement",
        zorder=3,
    )
    axes[1].plot(
        ours["time_h"],
        ours["our_sigma_imag_s_m"],
        color="#d62728",
        linewidth=1.8,
        marker="o",
        markersize=3.2,
        label=model_label,
    )
    y_min = min(-0.01, paper["paper_sigma_imag_s_m"].min() * 1.2)
    y_max = max(0.075, paper["paper_sigma_imag_s_m"].max() * 1.08)
    axes[1].set_ylim(y_min, y_max)
    axes[1].set_ylabel("Imaginary, sigma'' (S/m)")
    axes[1].set_xlabel("Time, t (h)")
    axes[1].text(0.98, 0.92, f"MAE = {imag_mae:.3g} S/m", transform=axes[1].transAxes, ha="right", va="top", fontsize=8)
    add_zoom_inset(axes[1], ours, "our_sigma_imag_s_m", "AC2D zoom")

    for ax in axes:
        ax.set_xlim(0.0, max(6.0, paper["time_h"].max(), ours["time_h"].max()))
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fontsize=8, loc="upper left")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=260, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_source_data(paper: pd.DataFrame, ours: pd.DataFrame, source_path: Path, mechanism: str) -> None:
    paper_out = paper[["time_h", "paper_sigma_real_s_m", "paper_sigma_imag_s_m", "paper_cec_meq_g"]].copy()
    paper_out["series"] = "paper_si_s02"
    ours_out = ours[["time_h", "our_sigma_real_s_m", "our_sigma_imag_s_m", "frame"]].copy()
    ours_out["series"] = f"ac2d_{mechanism}"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([paper_out, ours_out], ignore_index=True, sort=False).to_csv(source_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", default=str(DEFAULT_PAPER))
    parser.add_argument("--ours", default=str(DEFAULT_OURS))
    parser.add_argument("--mechanism", default="all")
    parser.add_argument("--frequency-hz", type=float, default=2.5)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--source-data", default=str(DEFAULT_SOURCE))
    args = parser.parse_args()

    paper = load_paper_series(Path(args.paper))
    ours = load_our_series(Path(args.ours), args.mechanism, args.frequency_hz)
    plot_comparison(paper, ours, Path(args.out), args.mechanism, args.label)
    write_source_data(paper, ours, Path(args.source_data), args.mechanism)
    print(f"wrote {args.out}")
    print(f"wrote {Path(args.out).with_suffix('.pdf')}")
    print(f"wrote {args.source_data}")


if __name__ == "__main__":
    main()
