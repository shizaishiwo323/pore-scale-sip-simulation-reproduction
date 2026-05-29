#!/usr/bin/env python3
"""Run the first 2-D microfluidic SIP mechanism sweep."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from pore_scale_electrical.ac2d_solver import phase_conductivity_grid_2d, solve_ac2d_dirichlet  # noqa: E402
from pore_scale_electrical.microfluidic_2d import (  # noqa: E402
    CALCITE_LABEL,
    WATER_LABEL,
    MicrofluidicCalibration,
    default_frequencies_hz,
    geometry_metrics,
    mechanism_phase_conductivities,
    metrics_to_dict,
    pore_radius_samples_m,
    segment_microfluidic_image,
    skeleton_component_lengths_m,
)


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "dissolution_results-Da_40.4424_Pe_4.1640_L_0.1200_square" / "interface_images"
DEFAULT_TIME_LOG = PROJECT_ROOT / "data" / "dissolution_results-Da_40.4424_Pe_4.1640_L_0.1200_square" / "global_evolution_log.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "ac2d_microfluidic" / "interface_images_v1"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "figures" / "ac2d_microfluidic"
DEFAULT_FRAMES = [1, 10, 20, 28]
MECHANISMS = ["maxwell", "pore", "membrane", "grain", "all"]
PLOT_ORDER = ["maxwell", "grain", "pore", "membrane", "all"]
PLOT_STYLE = {
    "maxwell": {"color": "#2ca02c", "linestyle": "-", "marker": "o", "zorder": 2},
    "grain": {"color": "#ff7f0e", "linestyle": "-", "marker": "s", "zorder": 3},
    "pore": {"color": "#9467bd", "linestyle": "-", "marker": "^", "zorder": 4},
    "membrane": {"color": "#d62728", "linestyle": "-", "marker": "o", "zorder": 5},
    "all": {"color": "#111111", "linestyle": "--", "marker": "D", "zorder": 8},
}


def frame_path(input_dir: Path, frame: int) -> Path:
    return input_dir / f"timestep_{frame:04d}.png"


def downsample_labels(labels: np.ndarray, factor: int) -> np.ndarray:
    """Block-majority downsample for two labels."""

    if factor <= 1:
        return labels.copy()
    height = (labels.shape[0] // factor) * factor
    width = (labels.shape[1] // factor) * factor
    if height == 0 or width == 0:
        raise ValueError("downsample factor is larger than the label image")
    trimmed = labels[:height, :width]
    blocks = trimmed.reshape(height // factor, factor, width // factor, factor)
    calcite_fraction = np.mean(blocks == CALCITE_LABEL, axis=(1, 3))
    return np.where(calcite_fraction >= 0.5, CALCITE_LABEL, WATER_LABEL).astype(np.uint8)


def parse_frequencies(values: list[str] | None) -> np.ndarray:
    if not values:
        return default_frequencies_hz()
    return np.asarray([float(v) for v in values], dtype=float)


def load_frame_times(path: Path, max_frame: int) -> dict[int, dict[str, float]]:
    """Map 1-based frame ids to the global evolution time log."""

    table = pd.read_csv(path)
    if len(table) < max_frame:
        raise ValueError(f"time log has {len(table)} rows but frame {max_frame} was requested")
    out: dict[int, dict[str, float]] = {}
    for frame, row in enumerate(table.itertuples(index=False), start=1):
        time_s = float(row.time_s)
        out[frame] = {"time_s": time_s, "time_h": time_s / 3600.0}
    return out


def write_plot_outputs(results: pd.DataFrame, out_dir: Path, figure_dir: Path) -> None:
    """Write the two v1 summary figures requested in the plan."""

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    first_frame = int(results["frame"].min())
    subset = results[results["frame"] == first_frame]
    for mechanism in PLOT_ORDER:
        group = subset[subset["mechanism"] == mechanism]
        if group.empty:
            continue
        group = group.sort_values("frequency_hz")
        style = PLOT_STYLE[mechanism]
        marker_face = "white" if mechanism == "all" else style["color"]
        ax.loglog(
            group["frequency_hz"],
            np.abs(group["sigma_imag_s_m"]),
            label=mechanism,
            linewidth=2.0 if mechanism == "all" else 1.6,
            markersize=4.8,
            markerfacecolor=marker_face,
            markeredgewidth=1.2,
            **style,
        )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|sigma''| (S/m)")
    ax.set_title(f"AC2D mechanism split, frame {first_frame:04d}")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig_path = figure_dir / "ac2d_microfluidic_mechanism_spectrum.png"
    fig.savefig(fig_path, dpi=220)
    fig.savefig(out_dir / fig_path.name, dpi=220)
    plt.close(fig)

    target_frequency = 2.5
    at_25 = results[np.isclose(results["frequency_hz"], target_frequency)]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for mechanism in PLOT_ORDER:
        group = at_25[at_25["mechanism"] == mechanism]
        if group.empty:
            continue
        group = group.sort_values("frame")
        style = PLOT_STYLE[mechanism]
        marker_face = "white" if mechanism == "all" else style["color"]
        ax.semilogy(
            group["frame"],
            np.abs(group["sigma_imag_s_m"]),
            label=mechanism,
            linewidth=2.0 if mechanism == "all" else 1.6,
            markersize=4.8,
            markerfacecolor=marker_face,
            markeredgewidth=1.2,
            **style,
        )
    ax.set_xlabel("Frame")
    ax.set_ylabel("|sigma''| at 2.5 Hz (S/m)")
    ax.set_title("AC2D 2.5 Hz response during dissolution")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig_path = figure_dir / "ac2d_microfluidic_2p5hz_time_response.png"
    fig.savefig(fig_path, dpi=220)
    fig.savefig(out_dir / fig_path.name, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--time-log", default=str(DEFAULT_TIME_LOG))
    parser.add_argument("--frames", nargs="+", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--frequencies", nargs="*")
    parser.add_argument("--width-m", type=float, default=4.0e-3)
    parser.add_argument("--height-m", type=float, default=1.5e-3)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--mechanisms", nargs="+", default=MECHANISMS, choices=MECHANISMS)
    parser.add_argument("--water-conductivity", type=float, default=0.013)
    parser.add_argument("--calcite-conductivity", type=float, default=1.0e-8)
    parser.add_argument("--grain-surface-conductance", type=float, default=1.3e-9)
    parser.add_argument("--grain-diffusion-coefficient", type=float, default=1.3e-9)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    figure_dir = Path(args.figure_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    calibration = MicrofluidicCalibration(width_m=args.width_m, height_m=args.height_m)
    frequencies = parse_frequencies(args.frequencies)
    frame_times = load_frame_times(Path(args.time_log), max(args.frames))

    reference = segment_microfluidic_image(frame_path(input_dir, args.frames[0]), calibration=calibration)
    reference_bbox = reference.bbox_xyxy
    all_components = tuple(mechanism for mechanism in args.mechanisms if mechanism != "all")
    geometry_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    spectra_rows: list[pd.DataFrame] = []

    for frame in args.frames:
        segmented = segment_microfluidic_image(frame_path(input_dir, frame), calibration=calibration, bbox_xyxy=reference_bbox)
        metrics = geometry_metrics(segmented)
        full_radii = pore_radius_samples_m(segmented.water_mask, segmented.dx_m, segmented.dy_m)
        full_lengths = skeleton_component_lengths_m(segmented.water_mask, segmented.dx_m, segmented.dy_m)
        metrics_row: dict[str, object] = {
            "frame": frame,
            "time_s": frame_times[frame]["time_s"],
            "time_h": frame_times[frame]["time_h"],
            "source_path": str(segmented.source_path),
            "bbox_x0": segmented.bbox_xyxy[0],
            "bbox_y0": segmented.bbox_xyxy[1],
            "bbox_x1": segmented.bbox_xyxy[2],
            "bbox_y1": segmented.bbox_xyxy[3],
            "red_pixels": segmented.red_pixels,
            "yellow_pixels": segmented.yellow_pixels,
            "active_pixels": segmented.active_pixels,
            "unknown_pixels_in_crop": segmented.unknown_pixels_in_crop,
            **metrics_to_dict(metrics),
        }
        geometry_rows.append(metrics_row)

        solve_labels = downsample_labels(segmented.labels, args.downsample)
        solve_dx = segmented.dx_m * args.downsample
        solve_dy = segmented.dy_m * args.downsample
        for mechanism in args.mechanisms:
            spectra = mechanism_phase_conductivities(
                frequencies,
                mechanism=mechanism,
                metrics=metrics,
                pore_radii_m=full_radii,
                throat_lengths_m=full_lengths,
                all_components=all_components,
                water_conductivity_s_m=args.water_conductivity,
                calcite_conductivity_s_m=args.calcite_conductivity,
                grain_surface_conductance_s=args.grain_surface_conductance,
                grain_diffusion_coefficient_m2_s=args.grain_diffusion_coefficient,
            )
            spectra.insert(0, "frame", frame)
            spectra_rows.append(spectra)
            for row in spectra.itertuples(index=False):
                water_sigma = complex(row.water_sigma_real_s_m, row.water_sigma_imag_s_m)
                solid_sigma = complex(row.solid_sigma_real_s_m, row.solid_sigma_imag_s_m)
                sigma_grid = phase_conductivity_grid_2d(solve_labels, WATER_LABEL, CALCITE_LABEL, water_sigma, solid_sigma)
                result = solve_ac2d_dirichlet(sigma_grid, dx_m=solve_dx, dy_m=solve_dy)
                result_rows.append(
                    {
                        "frame": frame,
                        "time_s": frame_times[frame]["time_s"],
                        "time_h": frame_times[frame]["time_h"],
                        "mechanism": mechanism,
                        "frequency_hz": float(row.frequency_hz),
                        "sigma_real_s_m": result.effective_conductivity_s_m.real,
                        "sigma_imag_s_m": result.effective_conductivity_s_m.imag,
                        "residual_norm": result.residual_norm,
                        "solve_height_px": int(solve_labels.shape[0]),
                        "solve_width_px": int(solve_labels.shape[1]),
                        "downsample": int(args.downsample),
                        "water_sigma_real_s_m": water_sigma.real,
                        "water_sigma_imag_s_m": water_sigma.imag,
                        "solid_sigma_real_s_m": solid_sigma.real,
                        "solid_sigma_imag_s_m": solid_sigma.imag,
                    }
                )

    geometry_df = pd.DataFrame(geometry_rows)
    results_df = pd.DataFrame(result_rows)
    spectra_df = pd.concat(spectra_rows, ignore_index=True)
    geometry_df.to_csv(out_dir / "geometry_metrics.csv", index=False)
    spectra_df.to_csv(out_dir / "mechanism_phase_spectra.csv", index=False)
    results_df.to_csv(out_dir / "ac2d_sweep_results.csv", index=False)

    metadata = {
        "calibration": {"width_m": args.width_m, "height_m": args.height_m},
        "reference_bbox_xyxy": reference_bbox,
        "frames": args.frames,
        "frequencies_hz": [float(v) for v in frequencies],
        "mechanisms": args.mechanisms,
        "all_components": list(all_components),
        "downsample": args.downsample,
        "grain_surface_conductance_s": args.grain_surface_conductance,
        "grain_diffusion_coefficient_m2_s": args.grain_diffusion_coefficient,
        "labels": {"water": int(WATER_LABEL), "calcite": int(CALCITE_LABEL)},
        "gas_phase": "not modeled in v1",
        "white_canvas": "excluded by red-or-yellow active bbox",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_plot_outputs(results_df, out_dir, figure_dir)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
