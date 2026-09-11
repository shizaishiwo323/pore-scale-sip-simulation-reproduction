#!/usr/bin/env python3
"""Build a diagnostic Sample 16 network with a ~500 Hz membrane relaxation peak.

Only the explicit membrane relaxation-length column is changed. Extracted
throat geometry and the geometry-derived membrane DC resistance are retained.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code" / "src"))

from pore_scale_electrical.polarization import (  # noqa: E402
    PolarizationParameters,
    membrane_polarization_conductance,
    throat_cross_section_area_from_radius_shape_factor,
    throat_zdc_from_length_area,
)


def truncated_normal(
    rng: np.random.Generator,
    count: int,
    *,
    mean_m: float,
    std_m: float,
    low_m: float,
    high_m: float,
) -> np.ndarray:
    accepted: list[float] = []
    while len(accepted) < count:
        draw = rng.normal(mean_m, std_m, max(64, count * 2))
        accepted.extend(draw[(draw >= low_m) & (draw <= high_m)].tolist())
    return np.asarray(accepted[:count], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-network-dir", type=Path, required=True)
    parser.add_argument("--output-network-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--target-weight-fraction", type=float, default=0.95)
    parser.add_argument("--mean-um", type=float, default=1.216121)
    parser.add_argument("--std-um", type=float, default=0.18)
    parser.add_argument("--low-um", type=float, default=0.75)
    parser.add_argument("--high-um", type=float, default=1.80)
    parser.add_argument("--seed", type=int, default=951500)
    parser.add_argument("--water-conductivity-s-m", type=float, default=0.12)
    parser.add_argument("--diffusion-coefficient-m2-s", type=float, default=1.3e-9)
    parser.add_argument("--membrane-polarizability", type=float, default=7.6e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_throats = args.source_network_dir / "throats.csv"
    source_pores = args.source_network_dir / "pores.csv"
    source_summary = args.source_network_dir / "network_summary.json"
    if not source_throats.exists() or not source_pores.exists():
        raise FileNotFoundError("source network must contain throats.csv and pores.csv")

    args.output_network_dir.mkdir(parents=True, exist_ok=True)
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    throats = pd.read_csv(source_throats)
    internal_mask = (throats["pore1_id"].astype(int) >= 1) & (throats["pore2_id"].astype(int) >= 1)
    internal = throats.loc[internal_mask].copy()

    original_length = internal["throat_length_m"].to_numpy(dtype=float)
    area = throat_cross_section_area_from_radius_shape_factor(
        internal["throat_radius_m"].to_numpy(dtype=float),
        internal["throat_shape_factor"].to_numpy(dtype=float),
    )
    original_zdc = throat_zdc_from_length_area(original_length, area, args.water_conductivity_s_m)
    effective_weight = internal["throat_volume_m3"].to_numpy(dtype=float) / original_zdc
    order = np.argsort(effective_weight)[::-1]
    cumulative = np.cumsum(effective_weight[order]) / effective_weight.sum()
    count = int(np.searchsorted(cumulative, args.target_weight_fraction, side="left") + 1)
    selected_positions = order[:count]
    selected_index = internal.index.to_numpy()[selected_positions]

    mean_m = args.mean_um * 1e-6
    rng = np.random.default_rng(args.seed)
    target_lengths = truncated_normal(
        rng,
        count,
        mean_m=mean_m,
        std_m=args.std_um * 1e-6,
        low_m=args.low_um * 1e-6,
        high_m=args.high_um * 1e-6,
    )
    target_lengths[0] = mean_m

    throats["membrane_relaxation_length_m"] = throats["throat_length_m"].astype(float)
    throats.loc[selected_index, "membrane_relaxation_length_m"] = target_lengths
    throats.to_csv(args.output_network_dir / "throats.csv", index=False)
    shutil.copy2(source_pores, args.output_network_dir / "pores.csv")
    if source_summary.exists():
        shutil.copy2(source_summary, args.output_network_dir / "network_summary.json")

    params = PolarizationParameters(
        water_conductivity_s_m=args.water_conductivity_s_m,
        diffusion_coefficient_m2_s=args.diffusion_coefficient_m2_s,
        membrane_polarizability=args.membrane_polarizability,
    )
    fine_frequency_hz = np.logspace(1.0, 4.0, 3001)
    relaxation_lengths = internal["throat_length_m"].to_numpy(dtype=float)
    relaxation_lengths[selected_positions] = target_lengths
    response = membrane_polarization_conductance(
        fine_frequency_hz,
        relaxation_lengths,
        internal["throat_volume_m3"].to_numpy(dtype=float),
        original_zdc,
        params,
    )
    peak_index = int(np.argmax(response.imag))
    fine = pd.DataFrame(
        {
            "frequency_hz": fine_frequency_hz,
            "membrane_conductance_real_s": response.real,
            "membrane_conductance_imag_s": response.imag,
        }
    )
    fine.to_csv(args.audit_dir / "membrane_fc500_fine_spectrum.csv", index=False)

    selected = internal.iloc[selected_positions].copy()
    selected["effective_weight_fraction"] = effective_weight[selected_positions] / effective_weight.sum()
    selected["cumulative_effective_weight_fraction"] = np.cumsum(
        effective_weight[selected_positions]
    ) / effective_weight.sum()
    selected["original_throat_length_um"] = selected["throat_length_m"] * 1e6
    selected["membrane_relaxation_length_um"] = target_lengths * 1e6
    selected.to_csv(args.audit_dir / "selected_membrane_relaxation_throats.csv", index=False)

    metadata = {
        "property": "diagnostic membrane-only effective relaxation-length sensitivity",
        "source_network_dir": str(args.source_network_dir),
        "output_network_dir": str(args.output_network_dir),
        "internal_throat_count": int(len(internal)),
        "selected_throat_count": int(count),
        "selected_count_fraction": float(count / len(internal)),
        "target_effective_weight_fraction": args.target_weight_fraction,
        "realized_effective_weight_fraction": float(effective_weight[selected_positions].sum() / effective_weight.sum()),
        "distribution": {
            "type": "truncated Gaussian; dominant throat pinned at the theoretical center",
            "mean_um": args.mean_um,
            "std_um": args.std_um,
            "low_um": args.low_um,
            "high_um": args.high_um,
            "seed": args.seed,
            "realized_min_um": float(target_lengths.min() * 1e6),
            "realized_mean_um": float(target_lengths.mean() * 1e6),
            "realized_max_um": float(target_lengths.max() * 1e6),
        },
        "unchanged_columns": [
            "throat_length_m",
            "throat_radius_m",
            "throat_shape_factor",
            "throat_volume_m3",
        ],
        "membrane_zdc_geometry": "retained from original throat_length_m, radius, and shape factor",
        "fine_spectrum_peak_hz": float(fine_frequency_hz[peak_index]),
        "fine_spectrum_peak_imag_conductance_s": float(response.imag[peak_index]),
        "parameters": {
            "water_conductivity_s_m": args.water_conductivity_s_m,
            "diffusion_coefficient_m2_s": args.diffusion_coefficient_m2_s,
            "membrane_polarizability": args.membrane_polarizability,
        },
    }
    (args.audit_dir / "membrane_fc500_network_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
