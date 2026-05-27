#!/usr/bin/env python3
"""Create apparent-water spectra for separate polarization mechanisms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pore_scale_electrical.polarization import PolarizationParameters  # noqa: E402


def make_component_spectrum(
    base: pd.DataFrame,
    component: str,
    params: PolarizationParameters,
    mode: str,
) -> pd.DataFrame:
    omega = base["omega_rad_s"].to_numpy(dtype=float)
    water_dielectric_imag = omega * params.water_permittivity_f_m
    solid_dielectric_imag = omega * params.solid_permittivity_f_m
    out = pd.DataFrame({"frequency_hz": base["frequency_hz"], "omega_rad_s": base["omega_rad_s"]})

    if component == "interfacial":
        delta_real = np.zeros(len(base), dtype=float)
        delta_imag = np.zeros(len(base), dtype=float)
    elif component == "pore":
        delta_real = base["delta_sigma_pore_real_s_m"].to_numpy(dtype=float)
        delta_imag = base["delta_sigma_pore_imag_s_m"].to_numpy(dtype=float)
    elif component == "membrane":
        delta_real = base["delta_sigma_membrane_real_s_m"].to_numpy(dtype=float)
        delta_imag = base["delta_sigma_membrane_imag_s_m"].to_numpy(dtype=float)
    elif component == "all":
        delta_real = base["delta_sigma_total_real_s_m"].to_numpy(dtype=float)
        delta_imag = base["delta_sigma_total_imag_s_m"].to_numpy(dtype=float)
    else:
        raise ValueError("component must be interfacial, pore, membrane, or all")

    if mode == "legacy":
        water_real = params.water_conductivity_s_m + delta_real
        water_imag = water_dielectric_imag + delta_imag
        solid_real = np.zeros(len(base), dtype=float)
        solid_imag = solid_dielectric_imag
    elif mode == "paper":
        if component in {"pore", "membrane"}:
            # Niu et al. section 5.3: for pore-only or membrane-only runs,
            # sigma_w* = sigma_w + Delta sigma_w*, and the solid phase is zero.
            water_real = params.water_conductivity_s_m + delta_real
            water_imag = delta_imag
            solid_real = np.zeros(len(base), dtype=float)
            solid_imag = np.zeros(len(base), dtype=float)
        else:
            water_real = params.water_conductivity_s_m + delta_real
            water_imag = water_dielectric_imag + delta_imag
            solid_real = np.zeros(len(base), dtype=float)
            solid_imag = solid_dielectric_imag
    else:
        raise ValueError("mode must be legacy or paper")

    out["component"] = component
    out["component_mode"] = mode
    out["delta_sigma_component_real_s_m"] = delta_real
    out["delta_sigma_component_imag_s_m"] = delta_imag
    out["apparent_water_sigma_real_s_m"] = water_real
    out["apparent_water_sigma_imag_s_m"] = water_imag
    out["solid_sigma_real_s_m"] = solid_real
    out["solid_sigma_imag_s_m"] = solid_imag
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "outputs" / "polarization_spectra_from_pnextract.csv"))
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "polarization_component_spectra"))
    parser.add_argument("--components", nargs="+", default=["interfacial", "pore", "membrane", "all"])
    parser.add_argument(
        "--mode",
        choices=["paper", "legacy"],
        default="paper",
        help="paper follows Niu et al. section 5.3 for individual mechanisms; legacy preserves the earlier dielectric-background split.",
    )
    args = parser.parse_args()

    base = pd.read_csv(args.input)
    params = PolarizationParameters()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": args.input,
        "mode": args.mode,
        "water_conductivity_s_m": params.water_conductivity_s_m,
        "water_permittivity_f_m": params.water_permittivity_f_m,
        "solid_permittivity_f_m": params.solid_permittivity_f_m,
        "definition": {
            "interfacial": "paper mode: water dc conductivity plus water/solid high-frequency permittivity only; no pore or membrane increment",
            "pore": "paper mode: water dc conductivity plus delta_sigma_pore only; solid phase is zero",
            "membrane": "paper mode: water dc conductivity plus delta_sigma_membrane only; solid phase is zero",
            "all": "paper mode: water dc conductivity and permittivity plus pore and membrane increments; solid phase has high-frequency permittivity",
        },
    }
    for component in args.components:
        spectrum = make_component_spectrum(base, component, params, args.mode)
        path = out_dir / f"polarization_spectra_{component}.csv"
        spectrum.to_csv(path, index=False)
        metadata[component] = str(path)
        print(f"wrote {path}")
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
