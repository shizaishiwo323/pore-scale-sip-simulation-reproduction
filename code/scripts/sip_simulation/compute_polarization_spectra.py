#!/usr/bin/env python3
"""Compute pore and membrane polarization spectra for Berea inputs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pore_scale_electrical.polarization import (  # noqa: E402
    PolarizationParameters,
    apparent_water_conductivity,
    membrane_polarization_conductance,
    pore_polarization_conductance,
    throat_zdc_from_geometry,
    upscale_conductance_to_water_conductivity,
)


NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def reject_post_extraction_geometry_scale(name: str, value: float) -> None:
    """Forbid post-extraction geometry scaling in production spectra."""
    if not np.isclose(float(value), 1.0, rtol=0.0, atol=1.0e-15):
        raise ValueError(
            f"{name}={value:g} is not allowed. Re-extract the pore network with a calibrated pnextract "
            "algorithm instead of scaling extracted geometry or geometry-derived resistance."
        )


def col_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref).group(0)
    idx = 0
    for ch in letters:
        idx = idx * 26 + ord(ch) - ord("A") + 1
    return idx - 1


def read_xlsx_sheet1(path: Path) -> list[list[str]]:
    with ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", NS):
                shared.append(html.unescape("".join(t.text or "" for t in si.iter(f"{{{NS['a']}}}t"))))
        root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row in root.findall(".//a:sheetData/a:row", NS):
            values: dict[int, str] = {}
            max_col = -1
            for cell in row.findall("a:c", NS):
                idx = col_index(cell.attrib["r"])
                max_col = max(max_col, idx)
                value_node = cell.find("a:v", NS)
                value = "" if value_node is None else value_node.text or ""
                if cell.attrib.get("t") == "s" and value:
                    value = shared[int(value)]
                values[idx] = value
            rows.append([values.get(i, "") for i in range(max_col + 1)])
    return rows


def load_figure5(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = read_xlsx_sheet1(path)
    pore_rows = []
    throat_rows = []
    for row in rows[1:]:
        if len(row) >= 2 and row[0] and row[1]:
            pore_rows.append({"radius_m": float(row[0]), "weight": float(row[1])})
        if len(row) >= 4 and row[2] and row[3]:
            throat_rows.append({"length_m": float(row[2]), "weight": float(row[3])})
    return pd.DataFrame(pore_rows), pd.DataFrame(throat_rows)


def load_pnextract(network_dir: Path, params: PolarizationParameters) -> tuple[pd.DataFrame, pd.DataFrame]:
    pores = pd.read_csv(network_dir / "pores.csv")
    throats = pd.read_csv(network_dir / "throats.csv")
    pore_df = pd.DataFrame({"radius_m": pores["pore_radius_m"], "weight": pores["pore_volume_m3"]})
    zdc = throat_zdc_from_geometry(
        length_m=throats["throat_length_m"].to_numpy(dtype=float),
        radius_m=throats["throat_radius_m"].to_numpy(dtype=float),
        shape_factor=throats["throat_shape_factor"].to_numpy(dtype=float),
        water_conductivity_s_m=params.water_conductivity_s_m,
    )
    throat_df = pd.DataFrame(
        {
            "length_m": throats["throat_length_m"],
            "weight": throats["throat_volume_m3"],
            "zdc_ohm": zdc,
        }
    )
    return pore_df, throat_df


def default_frequencies() -> np.ndarray:
    return np.logspace(-3, 9, 97)


def compute_spectra(
    frequency_hz: np.ndarray,
    pores: pd.DataFrame,
    throats: pd.DataFrame,
    params: PolarizationParameters,
    figure5_zdc_ohm: float | None,
    pore_radius_scale: float = 1.0,
    membrane_length_scale: float = 1.0,
    membrane_zdc_scale: float = 1.0,
) -> tuple[pd.DataFrame, dict]:
    if pore_radius_scale <= 0:
        raise ValueError("pore_radius_scale must be positive")
    if membrane_length_scale <= 0:
        raise ValueError("membrane_length_scale must be positive")
    if membrane_zdc_scale <= 0:
        raise ValueError("membrane_zdc_scale must be positive")
    reject_post_extraction_geometry_scale("pore_radius_scale", pore_radius_scale)
    reject_post_extraction_geometry_scale("membrane_length_scale", membrane_length_scale)
    reject_post_extraction_geometry_scale("membrane_zdc_scale", membrane_zdc_scale)

    effective_radius = pores["radius_m"].to_numpy(dtype=float) * pore_radius_scale
    cp = pore_polarization_conductance(
        frequency_hz,
        effective_radius,
        pores["weight"].to_numpy(dtype=float),
        params,
    )

    membrane_mode = "geometry_zdc"
    if "zdc_ohm" in throats.columns:
        zdc = throats["zdc_ohm"].to_numpy(dtype=float)
    elif figure5_zdc_ohm is not None:
        zdc = np.full(len(throats), figure5_zdc_ohm, dtype=float)
        membrane_mode = f"assumed_constant_zdc_{figure5_zdc_ohm:g}_ohm"
    else:
        zdc = None
        membrane_mode = "not_computed_missing_zdc"

    if zdc is None:
        cm = np.full_like(cp, np.nan + 1j * np.nan)
        c_total = cp
        effective_length = np.full(len(throats), np.nan, dtype=float)
        effective_zdc = np.full(len(throats), np.nan, dtype=float)
    else:
        effective_length = throats["length_m"].to_numpy(dtype=float) * membrane_length_scale
        effective_zdc = zdc * membrane_zdc_scale
        cm = membrane_polarization_conductance(
            frequency_hz,
            effective_length,
            throats["weight"].to_numpy(dtype=float),
            effective_zdc,
            params,
        )
        c_total = cp + cm

    delta_pore = upscale_conductance_to_water_conductivity(cp, params)
    delta_membrane = upscale_conductance_to_water_conductivity(cm, params)
    delta_total = upscale_conductance_to_water_conductivity(c_total, params)
    sigma_w_app = apparent_water_conductivity(frequency_hz, delta_total, params)

    out = pd.DataFrame(
        {
            "frequency_hz": frequency_hz,
            "omega_rad_s": 2.0 * np.pi * frequency_hz,
            "pore_conductance_real_s": cp.real,
            "pore_conductance_imag_s": cp.imag,
            "membrane_conductance_real_s": cm.real,
            "membrane_conductance_imag_s": cm.imag,
            "total_conductance_real_s": c_total.real,
            "total_conductance_imag_s": c_total.imag,
            "delta_sigma_pore_real_s_m": delta_pore.real,
            "delta_sigma_pore_imag_s_m": delta_pore.imag,
            "delta_sigma_membrane_real_s_m": delta_membrane.real,
            "delta_sigma_membrane_imag_s_m": delta_membrane.imag,
            "delta_sigma_total_real_s_m": delta_total.real,
            "delta_sigma_total_imag_s_m": delta_total.imag,
            "apparent_water_sigma_real_s_m": sigma_w_app.real,
            "apparent_water_sigma_imag_s_m": sigma_w_app.imag,
            "membrane_effective_length_m_mean": np.full(len(frequency_hz), np.nanmean(effective_length)),
            "membrane_effective_zdc_ohm_mean": np.full(len(frequency_hz), np.nanmean(effective_zdc)),
        }
    )
    finite_length = effective_length[np.isfinite(effective_length)]
    finite_zdc = effective_zdc[np.isfinite(effective_zdc)]
    finite_radius = effective_radius[np.isfinite(effective_radius)]
    metadata = {
        "n_pores_or_bins": int(len(pores)),
        "n_throats_or_bins": int(len(throats)),
        "pore_radius_scale": float(pore_radius_scale),
        "pore_effective_radius_m": {
            "min": float(finite_radius.min()) if finite_radius.size else None,
            "mean": float(finite_radius.mean()) if finite_radius.size else None,
            "max": float(finite_radius.max()) if finite_radius.size else None,
        },
        "membrane_mode": membrane_mode,
        "membrane_length_scale": float(membrane_length_scale),
        "membrane_zdc_scale": float(membrane_zdc_scale),
        "membrane_effective_length_m": {
            "min": float(finite_length.min()) if finite_length.size else None,
            "mean": float(finite_length.mean()) if finite_length.size else None,
            "max": float(finite_length.max()) if finite_length.size else None,
        },
        "membrane_effective_zdc_ohm": {
            "min": float(finite_zdc.min()) if finite_zdc.size else None,
            "mean": float(finite_zdc.mean()) if finite_zdc.size else None,
            "max": float(finite_zdc.max()) if finite_zdc.size else None,
        },
        "parameters": params.__dict__,
    }
    return out, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["figure5", "pnextract"], required=True)
    parser.add_argument("--figure5", default=str(ROOT / "论文数据" / "Figure5.xlsx"))
    parser.add_argument("--network-dir", default=str(ROOT / "outputs" / "figure5_pnextract_comparison" / "network_parsed"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--metadata-out")
    parser.add_argument("--figure5-zdc-ohm", type=float, default=None)
    parser.add_argument(
        "--pore-radius-scale",
        type=float,
        default=1.0,
        help="Must remain 1.0. Post-extraction pore geometry scaling is forbidden.",
    )
    parser.add_argument(
        "--membrane-length-scale",
        type=float,
        default=1.0,
        help="Must remain 1.0. Re-extract geometry instead of scaling throat length.",
    )
    parser.add_argument(
        "--membrane-zdc-scale",
        type=float,
        default=1.0,
        help="Must remain 1.0. Re-extract geometry instead of scaling geometry-derived Zdc.",
    )
    args = parser.parse_args()

    params = PolarizationParameters()
    if args.source == "figure5":
        pores, throats = load_figure5(Path(args.figure5))
    else:
        pores, throats = load_pnextract(Path(args.network_dir), params)

    spectra, metadata = compute_spectra(
        default_frequencies(),
        pores,
        throats,
        params,
        args.figure5_zdc_ohm,
        pore_radius_scale=args.pore_radius_scale,
        membrane_length_scale=args.membrane_length_scale,
        membrane_zdc_scale=args.membrane_zdc_scale,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    spectra.to_csv(out, index=False)

    metadata_out = Path(args.metadata_out) if args.metadata_out else out.with_suffix(".metadata.json")
    metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print("wrote", out)


if __name__ == "__main__":
    main()

