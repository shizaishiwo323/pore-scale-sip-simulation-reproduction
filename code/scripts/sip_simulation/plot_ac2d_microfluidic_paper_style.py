#!/usr/bin/env python3
"""Create a paper-style AC2D SIP spectra plus microfluidic structure figure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from pore_scale_electrical.microfluidic_2d import MicrofluidicCalibration, segment_microfluidic_image  # noqa: E402


DEFAULT_RESULTS = PROJECT_ROOT / "results" / "ac2d_microfluidic" / "interface_images_si03_mechanistic_v1" / "ac2d_sweep_results.csv"
DEFAULT_GEOMETRY = PROJECT_ROOT / "results" / "ac2d_microfluidic" / "interface_images_si03_mechanistic_v1" / "geometry_metrics.csv"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "data" / "dissolution_results-Da_40.4424_Pe_4.1640_L_0.1200_square" / "interface_images"
DEFAULT_OUT = PROJECT_ROOT / "figures" / "ac2d_microfluidic" / "ac2d_microfluidic_paper_style_si03_mechanistic.png"
DEFAULT_OUT_RESULTS = PROJECT_ROOT / "results" / "ac2d_microfluidic" / "interface_images_si03_mechanistic_v1" / "ac2d_microfluidic_paper_style_si03_mechanistic.png"

ROW_TARGETS_H = [
    (0.0, 0.22),
    (0.69, 1.12),
    (2.50, 3.53),
    (4.20, 4.53),
]
MECHANISM = "all"
COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#111111", "#8c564b"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def frame_path(image_dir: Path, frame: int) -> Path:
    return image_dir / f"timestep_{frame:04d}.png"


def nearest_frame(geometry: pd.DataFrame, target_h: float) -> int:
    idx = np.abs(geometry["time_h"].to_numpy(dtype=float) - target_h).argmin()
    return int(geometry.iloc[int(idx)]["frame"])


def selected_frame_rows(geometry: pd.DataFrame) -> list[tuple[int, int]]:
    return [(nearest_frame(geometry, a), nearest_frame(geometry, b)) for a, b in ROW_TARGETS_H]


def image_crop_from_active_domain(path: Path) -> np.ndarray:
    segmented = segment_microfluidic_image(path, calibration=MicrofluidicCalibration())
    x0, y0, x1, y1 = segmented.bbox_xyxy
    image = np.asarray(Image.open(path).convert("RGB"))
    return image[y0 : y1 + 1, x0 : x1 + 1]


def plot_spectrum_panel(ax: plt.Axes, results: pd.DataFrame, frames: tuple[int, int], component: str) -> None:
    for index, frame in enumerate(frames):
        group = results[(results["frame"] == frame) & (results["mechanism"] == MECHANISM)].sort_values("frequency_hz")
        if group.empty:
            continue
        time_h = float(group["time_h"].iloc[0])
        if component == "real":
            y = group["sigma_real_s_m"].to_numpy(dtype=float)
            ylabel = "In-phase, sigma' (S/m)"
        else:
            y = np.abs(group["sigma_imag_s_m"].to_numpy(dtype=float))
            ylabel = "|Quadrature, sigma''| (S/m)"
        ax.plot(
            group["frequency_hz"],
            y,
            color=COLORS[index],
            marker=MARKERS[index],
            linewidth=1.5,
            markersize=3.8,
            label=f"t = {time_h:.2f} h",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Frequency, f (Hz)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(frameon=True, fontsize=6, loc="best")


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.5},
    )


def plot_structure_panel(ax: plt.Axes, image_dir: Path, frame: int, time_h: float) -> None:
    path = frame_path(image_dir, frame)
    segmented = segment_microfluidic_image(path, calibration=MicrofluidicCalibration())
    image = image_crop_from_active_domain(path)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.98,
        0.06,
        f"t = {time_h:.2f} h",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.5},
    )
    calcite_yx = np.argwhere(segmented.calcite_mask)
    if calcite_yx.size:
        cy, cx = calcite_yx.mean(axis=0)
        label_x = cx / max(segmented.labels.shape[1] - 1, 1)
        label_y = cy / max(segmented.labels.shape[0] - 1, 1)
        label_text = "CaCO3"
    else:
        label_x = 0.50
        label_y = 0.50
        label_text = "dissolved"
    ax.text(
        label_x,
        label_y,
        label_text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="white",
        fontsize=8,
        weight="bold",
        alpha=0.9,
    )
    ax.annotate(
        "Flow",
        xy=(0.08, 0.16),
        xytext=(0.08, 0.36),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.5},
        color="white",
        fontsize=8,
        ha="center",
        va="center",
        rotation=90,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--geometry", default=str(DEFAULT_GEOMETRY))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--out-results", default=str(DEFAULT_OUT_RESULTS))
    args = parser.parse_args()

    results = pd.read_csv(args.results)
    geometry = pd.read_csv(args.geometry)
    image_dir = Path(args.image_dir)
    frame_pairs = selected_frame_rows(geometry)

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(13.0, 14.5),
        gridspec_kw={"width_ratios": [1.05, 1.05, 1.0, 1.0], "wspace": 0.26, "hspace": 0.26},
    )
    fig.suptitle("AC2D mechanistic SIP: SI03 sigma_w(t) + MW + Schwarz", fontsize=16, weight="bold", y=0.992)
    fig.text(0.255, 0.968, "Simulated SIP spectra", ha="center", va="center", fontsize=11, weight="bold")
    fig.text(0.745, 0.968, "Interface-image structure panels", ha="center", va="center", fontsize=11, weight="bold")

    panel = ord("a")
    for row, frames in enumerate(frame_pairs):
        plot_spectrum_panel(axes[row, 0], results, frames, "real")
        add_panel_label(axes[row, 0], chr(panel))
        panel += 1
        plot_spectrum_panel(axes[row, 1], results, frames, "imag")
        add_panel_label(axes[row, 1], chr(panel))
        panel += 1
        for col_offset, frame in enumerate(frames):
            time_h = float(geometry.loc[geometry["frame"] == frame, "time_h"].iloc[0])
            ax = axes[row, 2 + col_offset]
            plot_structure_panel(ax, image_dir, frame, time_h)
            add_panel_label(ax, chr(panel))
            panel += 1

    for path in [Path(args.out), Path(args.out_results)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")
    print(f"wrote {args.out_results}")
    print("selected_frames", frame_pairs)


if __name__ == "__main__":
    main()
