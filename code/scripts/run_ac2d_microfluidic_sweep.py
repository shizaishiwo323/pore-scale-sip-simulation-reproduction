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
    paper_validation_drivers_from_tables,
    pore_radius_samples_m,
    schwarz_interface_conductivity_delta,
    segment_microfluidic_image,
    si03_solution_conductivity_at_times,
    skeleton_component_lengths_m,
    waxman_smits_water_increment,
)


DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "dissolution_results-Da_40.4424_Pe_4.1640_L_0.1200_square" / "interface_images"
DEFAULT_TIME_LOG = PROJECT_ROOT / "data" / "dissolution_results-Da_40.4424_Pe_4.1640_L_0.1200_square" / "global_evolution_log.csv"
DEFAULT_SI01 = PROJECT_ROOT / "docs" / "validation" / "2024gl111271-sup-0002-data set si-s01.csv"
DEFAULT_SI02 = PROJECT_ROOT / "docs" / "validation" / "2024gl111271-sup-0003-data set si-s02.csv"
DEFAULT_SI03 = PROJECT_ROOT / "docs" / "validation" / "2024gl111271-sup-0004-data set si-s03.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "ac2d_microfluidic" / "interface_images_si03_mechanistic_v1"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "figures" / "ac2d_microfluidic"
DEFAULT_FRAMES = list(range(1, 29))
AVAILABLE_MECHANISMS = ["maxwell", "pore", "membrane", "grain", "interface", "all"]
DEFAULT_MECHANISMS = ["maxwell", "interface", "all"]
PLOT_ORDER = ["maxwell", "grain", "interface", "pore", "membrane", "all"]
PLOT_STYLE = {
    "maxwell": {"color": "#2ca02c", "linestyle": "-", "marker": "o", "zorder": 2},
    "grain": {"color": "#ff7f0e", "linestyle": "-", "marker": "s", "zorder": 3},
    "interface": {"color": "#17becf", "linestyle": "-", "marker": "P", "zorder": 6},
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


def read_semicolon_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def build_paper_drivers(
    *,
    frames: list[int],
    frame_times: dict[int, dict[str, float]],
    si01_path: Path,
    si02_path: Path,
    si03_path: Path,
    sigma_hcl_s_m: float,
    rho_calcite_kg_m3: float,
) -> dict[int, dict[str, float]]:
    times = np.asarray([frame_times[frame]["time_h"] for frame in frames], dtype=float)
    drivers = paper_validation_drivers_from_tables(
        times,
        si01=read_semicolon_csv(si01_path),
        si02=read_semicolon_csv(si02_path),
        si03=read_semicolon_csv(si03_path),
        sigma_hcl_s_m=sigma_hcl_s_m,
        end_time_h=float(np.max(times)),
        rho_calcite_kg_m3=rho_calcite_kg_m3,
    )
    return {
        frame: {
            "phi": float(row.phi),
            "sw": float(row.sw),
            "sigma_w_s_m": float(row.sigma_w_s_m),
            "cec_meq_g": float(row.cec_meq_g),
            "qv_c_m3": float(row.qv_c_m3),
        }
        for frame, row in zip(frames, drivers.itertuples(index=False), strict=True)
    }


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
    parser.add_argument("--mechanisms", nargs="+", default=DEFAULT_MECHANISMS, choices=AVAILABLE_MECHANISMS)
    parser.add_argument("--driver-mode", choices=["si03", "paper-cec", "constant"], default="si03")
    parser.add_argument("--water-conductivity", type=float, default=0.013)
    parser.add_argument("--calcite-conductivity", type=float, default=1.0e-8)
    parser.add_argument("--grain-surface-conductance", type=float, default=1.3e-9)
    parser.add_argument("--grain-diffusion-coefficient", type=float, default=1.3e-9)
    parser.add_argument("--schwarz-sigma-s", type=float, default=7.0e-5)
    parser.add_argument("--schwarz-characteristic-length-m", type=float, default=1.3e-5)
    parser.add_argument("--schwarz-diffusion-coefficient", type=float, default=1.3e-9)
    parser.add_argument("--paper-si01", default=str(DEFAULT_SI01))
    parser.add_argument("--paper-si02", default=str(DEFAULT_SI02))
    parser.add_argument("--paper-si03", default=str(DEFAULT_SI03))
    parser.add_argument("--sigma-hcl", type=float, default=0.44)
    parser.add_argument("--rho-calcite", type=float, default=2710.0)
    parser.add_argument("--beta", type=float, default=19.98e-8)
    parser.add_argument("--alpha", type=float, default=6.0)
    parser.add_argument("--saturation-exponent", type=float, default=2.0)
    parser.add_argument("--ignore-paper-saturation", action="store_true")
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
    final_frame_time_h = max(frame_times[frame]["time_h"] for frame in args.frames)
    si03_table = read_semicolon_csv(Path(args.paper_si03)) if args.driver_mode == "si03" else None
    paper_drivers: dict[int, dict[str, float]] = {}
    if args.driver_mode == "paper-cec":
        paper_drivers = build_paper_drivers(
            frames=args.frames,
            frame_times=frame_times,
            si01_path=Path(args.paper_si01),
            si02_path=Path(args.paper_si02),
            si03_path=Path(args.paper_si03),
            sigma_hcl_s_m=args.sigma_hcl,
            rho_calcite_kg_m3=args.rho_calcite,
        )

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
        if args.driver_mode == "paper-cec":
            metrics_row.update({f"paper_{key}": value for key, value in paper_drivers[frame].items()})
        geometry_rows.append(metrics_row)

        solve_labels = downsample_labels(segmented.labels, args.downsample)
        solve_dx = segmented.dx_m * args.downsample
        solve_dy = segmented.dy_m * args.downsample
        if args.driver_mode == "si03":
            if si03_table is None:
                raise RuntimeError("si03_table was not loaded")
            solution_sigma_w = float(
                si03_solution_conductivity_at_times(
                    np.array([frame_times[frame]["time_h"]]),
                    si03=si03_table,
                    sigma_hcl_s_m=args.sigma_hcl,
                    end_time_h=final_frame_time_h,
                )[0]
            )
            interface_density = metrics.interface_length_m / max(metrics.active_area_m2, np.finfo(float).eps)
            schwarz_tau_s = args.schwarz_characteristic_length_m**2 / (2.0 * args.schwarz_diffusion_coefficient)
            water_conductivity = solution_sigma_w
            interface_delta = schwarz_interface_conductivity_delta(
                frequencies,
                interface_density_1_m=interface_density,
                sigma_s_s=args.schwarz_sigma_s,
                characteristic_length_m=args.schwarz_characteristic_length_m,
                diffusion_coefficient_m2_s=args.schwarz_diffusion_coefficient,
            )
            driver = {
                "solution_sigma_w_s_m": solution_sigma_w,
                "interface_density_1_m": float(interface_density),
                "schwarz_sigma_s_s": float(args.schwarz_sigma_s),
                "schwarz_characteristic_length_m": float(args.schwarz_characteristic_length_m),
                "schwarz_diffusion_coefficient_m2_s": float(args.schwarz_diffusion_coefficient),
                "schwarz_tau_s": float(schwarz_tau_s),
            }
            saturation_factor = 1.0
            interface_saturation_factor = 1.0
        elif args.driver_mode == "paper-cec":
            driver = paper_drivers[frame]
            saturation_factor = 1.0 if args.ignore_paper_saturation else driver["sw"] ** args.saturation_exponent
            interface_saturation_factor = (
                1.0 if args.ignore_paper_saturation else driver["sw"] ** (args.saturation_exponent - 1.0)
            )
            water_conductivity = driver["sigma_w_s_m"] * saturation_factor
            interface_delta = complex(waxman_smits_water_increment(
                phi=driver["phi"],
                sw=1.0,
                cec_meq_g=driver["cec_meq_g"],
                rho_calcite_kg_m3=args.rho_calcite,
                beta_m2_s_v=args.beta,
                alpha=args.alpha,
            )) * interface_saturation_factor
        else:
            driver = {}
            saturation_factor = 1.0
            interface_saturation_factor = 1.0
            water_conductivity = args.water_conductivity
            interface_delta = np.zeros_like(frequencies, dtype=np.complex128)
        interface_delta_values = np.asarray(interface_delta, dtype=np.complex128)
        if interface_delta_values.ndim == 0:
            interface_delta_values = np.full(frequencies.shape, complex(interface_delta_values), dtype=np.complex128)
        for mechanism in args.mechanisms:
            spectra = mechanism_phase_conductivities(
                frequencies,
                mechanism=mechanism,
                metrics=metrics,
                pore_radii_m=full_radii,
                throat_lengths_m=full_lengths,
                all_components=all_components,
                water_conductivity_s_m=water_conductivity,
                calcite_conductivity_s_m=args.calcite_conductivity,
                grain_surface_conductance_s=args.grain_surface_conductance,
                grain_diffusion_coefficient_m2_s=args.grain_diffusion_coefficient,
                interface_delta_conductivity_s_m=interface_delta_values,
            )
            spectra.insert(0, "frame", frame)
            spectra["driver_mode"] = args.driver_mode
            spectra["saturation_factor"] = float(saturation_factor)
            spectra["interface_saturation_factor"] = float(interface_saturation_factor)
            spectra["interface_delta_real_s_m"] = np.real(interface_delta_values)
            spectra["interface_delta_imag_s_m"] = np.imag(interface_delta_values)
            for key, value in driver.items():
                spectra[key] = value
            if args.driver_mode == "paper-cec":
                for key, value in driver.items():
                    spectra[f"paper_{key}"] = value
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
                        "driver_mode": args.driver_mode,
                        "saturation_factor": float(saturation_factor),
                        "interface_saturation_factor": float(interface_saturation_factor),
                        "interface_delta_real_s_m": float(row.interface_delta_real_s_m),
                        "interface_delta_imag_s_m": float(row.interface_delta_imag_s_m),
                        "solution_sigma_w_s_m": driver.get("solution_sigma_w_s_m"),
                        "interface_density_1_m": driver.get("interface_density_1_m"),
                        "schwarz_sigma_s_s": driver.get("schwarz_sigma_s_s"),
                        "schwarz_characteristic_length_m": driver.get("schwarz_characteristic_length_m"),
                        "schwarz_diffusion_coefficient_m2_s": driver.get("schwarz_diffusion_coefficient_m2_s"),
                        "schwarz_tau_s": driver.get("schwarz_tau_s"),
                        "paper_phi": driver.get("phi"),
                        "paper_sw": driver.get("sw"),
                        "paper_sigma_w_s_m": driver.get("sigma_w_s_m"),
                        "paper_cec_meq_g": driver.get("cec_meq_g"),
                        "paper_qv_c_m3": driver.get("qv_c_m3"),
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
        "driver_mode": args.driver_mode,
        "si03_driver": {
            "path": str(Path(args.paper_si03)),
            "sigma_hcl_s_m": args.sigma_hcl,
            "extension": "linear from last SI-S03 value to sigma_hcl at final simulated frame",
        }
        if args.driver_mode == "si03"
        else None,
        "schwarz_parameters": {
            "sigma_s_s": args.schwarz_sigma_s,
            "characteristic_length_m": args.schwarz_characteristic_length_m,
            "diffusion_coefficient_m2_s": args.schwarz_diffusion_coefficient,
            "tau_s": args.schwarz_characteristic_length_m**2 / (2.0 * args.schwarz_diffusion_coefficient),
        }
        if args.driver_mode == "si03"
        else None,
        "paper_inputs": {
            "si01": str(Path(args.paper_si01)),
            "si02": str(Path(args.paper_si02)),
            "si03": str(Path(args.paper_si03)),
            "sigma_hcl_s_m": args.sigma_hcl,
            "rho_calcite_kg_m3": args.rho_calcite,
            "beta_m2_s_v": args.beta,
            "alpha": args.alpha,
            "saturation_exponent": args.saturation_exponent,
            "apply_paper_saturation": not args.ignore_paper_saturation,
        }
        if args.driver_mode == "paper-cec"
        else None,
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
