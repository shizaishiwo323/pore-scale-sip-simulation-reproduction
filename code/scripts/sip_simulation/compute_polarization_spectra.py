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
from dataclasses import replace
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
    throat_cross_section_area_from_radius_shape_factor,
    throat_zdc_from_length_area,
    throat_zdc_from_geometry,
    upscale_conductance_to_water_conductivity,
)


NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
DEFAULT_FIGURE5_PATH = ROOT.parent / "data" / "Niu 2020data" / "Figure5.xlsx"
ALLOWED_MEMBRANE_WEIGHT_MODES = (
    "volume",
    "volume_density_renormalized_diagnostic",
    "volume_density_renormalized",
    "volume_linear_density",
    "volume_over_length",
    "volume_over_length2",
    "count",
)
ALLOWED_MEMBRANE_ZDC_LENGTH_MODES = ("throat", "center_to_center", "conduit")
PAPER_REFERENCE_DYNAMIC_PORE_SIZE_M = 2.7e-6


def resolve_polarization_parameters(
    parameter_mode: str,
    dynamic_pore_size_manifest: Path | str | None,
) -> tuple[PolarizationParameters, dict]:
    if parameter_mode == "niu2020-paper":
        params = replace(PolarizationParameters(), dynamic_pore_size_m=PAPER_REFERENCE_DYNAMIC_PORE_SIZE_M)
        return (
            params,
            {
                "dynamic_pore_size_m": params.dynamic_pore_size_m,
                "dynamic_pore_size_source": "niu2020_table1",
                "dynamic_pore_size_manifest": None,
                "paper_reference_dynamic_pore_size_m": PAPER_REFERENCE_DYNAMIC_PORE_SIZE_M,
                "paper_reference_used_as_project_value": True,
            },
        )
    if parameter_mode != "project-extracted":
        raise ValueError("parameter_mode must be niu2020-paper or project-extracted")
    if dynamic_pore_size_manifest is None:
        raise ValueError("project-extracted spectra require dynamic pore size manifest")
    manifest_path = Path(dynamic_pore_size_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lambda_iso_m = float(manifest["lambda_iso_m"])
    if lambda_iso_m <= 0:
        raise ValueError("dynamic pore size manifest lambda_iso_m must be positive")
    params = replace(PolarizationParameters(), dynamic_pore_size_m=lambda_iso_m)
    return (
        params,
        {
            "dynamic_pore_size_m": lambda_iso_m,
            "dynamic_pore_size_source": "project_extracted_microct_laplace_field",
            "dynamic_pore_size_manifest": str(manifest_path),
            "paper_reference_dynamic_pore_size_m": float(manifest.get("paper_reference_lambda_m", PAPER_REFERENCE_DYNAMIC_PORE_SIZE_M)),
            "paper_reference_used_as_project_value": False,
        },
    )


def canonical_membrane_weight_mode(mode: str) -> str:
    if mode in {"volume_density_renormalized_diagnostic", "volume_density_renormalized", "volume_linear_density"}:
        return "volume_density_renormalized_diagnostic"
    return mode


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


def load_pnextract(
    network_dir: Path,
    params: PolarizationParameters,
    *,
    include_boundary_throats: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pores = pd.read_csv(network_dir / "pores.csv")
    throats = pd.read_csv(network_dir / "throats.csv")
    raw_throat_count = int(len(throats))
    excluded_boundary_throat_count = 0
    if not include_boundary_throats and {"pore1_id", "pore2_id"}.issubset(throats.columns):
        internal = (throats["pore1_id"].astype(int) >= 1) & (throats["pore2_id"].astype(int) >= 1)
        excluded_boundary_throat_count = int((~internal).sum())
        throats = throats.loc[internal].copy()
    pore_df = pd.DataFrame({"radius_m": pores["pore_radius_m"], "weight": pores["pore_volume_m3"]})

    def zdc_for_length(length_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        if "membrane_active_area_m2" in throats.columns:
            area_m2 = throats["membrane_active_area_m2"].to_numpy(dtype=float)
            return (
                throat_zdc_from_length_area(length_m, area_m2, params.water_conductivity_s_m),
                area_m2,
                "explicit_membrane_active_area_m2",
            )
        if "membrane_active_radius_m" in throats.columns:
            area_m2 = throat_cross_section_area_from_radius_shape_factor(
                throats["membrane_active_radius_m"].to_numpy(dtype=float),
                throats["throat_shape_factor"].to_numpy(dtype=float),
            )
            return (
                throat_zdc_from_length_area(length_m, area_m2, params.water_conductivity_s_m),
                area_m2,
                "explicit_membrane_active_radius_m_with_throat_shape_factor",
            )
        area_m2 = throat_cross_section_area_from_radius_shape_factor(
            throats["throat_radius_m"].to_numpy(dtype=float),
            throats["throat_shape_factor"].to_numpy(dtype=float),
        )
        return (
            throat_zdc_from_length_area(length_m, area_m2, params.water_conductivity_s_m),
            area_m2,
            "throat_radius_m_with_throat_shape_factor",
        )

    explicit_input_geometry = any(
        column in throats.columns
        for column in (
            "membrane_relaxation_length_m",
            "membrane_zdc_length_m",
            "membrane_active_area_m2",
            "membrane_active_radius_m",
        )
    )
    zdc_length_source = "membrane_zdc_length_m" if "membrane_zdc_length_m" in throats.columns else "throat_length_m"
    zdc_length_m = throats[zdc_length_source].to_numpy(dtype=float)
    zdc, active_area_m2, zdc_geometry_source = zdc_for_length(zdc_length_m)
    throat_df = pd.DataFrame(
        {
            "length_m": throats["throat_length_m"],
            "weight": throats["throat_volume_m3"],
            "zdc_ohm": zdc,
            "membrane_zdc_length_m": zdc_length_m,
            "membrane_active_area_m2": active_area_m2,
        }
    )
    if "membrane_relaxation_length_m" in throats.columns:
        throat_df["membrane_relaxation_length_m"] = throats["membrane_relaxation_length_m"].to_numpy(dtype=float)
    if "membrane_active_radius_m" in throats.columns:
        throat_df["membrane_active_radius_m"] = throats["membrane_active_radius_m"].to_numpy(dtype=float)
    throat_df.attrs["pnextract_raw_throat_count"] = raw_throat_count
    throat_df.attrs["pnextract_excluded_boundary_throat_count"] = excluded_boundary_throat_count
    throat_df.attrs["pnextract_include_boundary_throats"] = bool(include_boundary_throats)
    throat_df.attrs["membrane_relaxation_length_source"] = (
        "membrane_relaxation_length_m" if "membrane_relaxation_length_m" in throats.columns else "throat_length_m"
    )
    throat_df.attrs["membrane_zdc_length_source"] = zdc_length_source
    throat_df.attrs["membrane_zdc_geometry_source"] = zdc_geometry_source
    throat_df.attrs["membrane_geometry_mode"] = (
        "explicit_membrane_geometry_columns" if explicit_input_geometry else "single_throat_geometry"
    )
    if "pore_center_to_center_length_m" in throats.columns:
        throat_df["zdc_center_to_center_ohm"] = zdc_for_length(
            throats["pore_center_to_center_length_m"].to_numpy(dtype=float)
        )[0]
    if {"pore1_length_m", "pore2_length_m", "throat_length_m"}.issubset(throats.columns):
        conduit_length_m = (
            throats["pore1_length_m"].to_numpy(dtype=float)
            + throats["throat_length_m"].to_numpy(dtype=float)
            + throats["pore2_length_m"].to_numpy(dtype=float)
        )
        throat_df["zdc_conduit_ohm"] = zdc_for_length(conduit_length_m)[0]
    return pore_df, throat_df


def default_frequencies() -> np.ndarray:
    return np.logspace(-3, 9, 97)


def membrane_weights(throats: pd.DataFrame, mode: str) -> np.ndarray:
    if mode not in ALLOWED_MEMBRANE_WEIGHT_MODES:
        raise ValueError(f"membrane_weight_mode must be one of: {', '.join(ALLOWED_MEMBRANE_WEIGHT_MODES)}")
    mode = canonical_membrane_weight_mode(mode)
    base = throats["weight"].to_numpy(dtype=float)
    length_column = "membrane_relaxation_length_m" if "membrane_relaxation_length_m" in throats.columns else "length_m"
    length = throats[length_column].to_numpy(dtype=float)
    if mode == "volume":
        values = base
    elif mode in {
        "volume_density_renormalized_diagnostic",
        "volume_density_renormalized",
        "volume_linear_density",
        "volume_over_length",
    }:
        values = base / np.maximum(length, np.finfo(float).tiny)
    elif mode == "volume_over_length2":
        values = base / np.maximum(length**2, np.finfo(float).tiny)
    elif mode == "count":
        values = np.ones(len(throats), dtype=float)
    else:
        raise AssertionError(f"unhandled membrane_weight_mode: {mode}")
    return values


def nanmean_or_nan(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


def compute_spectra(
    frequency_hz: np.ndarray,
    pores: pd.DataFrame,
    throats: pd.DataFrame,
    params: PolarizationParameters,
    figure5_zdc_ohm: float | None,
    pore_radius_scale: float = 1.0,
    membrane_length_scale: float = 1.0,
    membrane_zdc_scale: float = 1.0,
    membrane_weight_mode: str = "volume",
    membrane_zdc_length_mode: str = "throat",
    dynamic_pore_size_metadata: dict | None = None,
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
    if membrane_weight_mode not in ALLOWED_MEMBRANE_WEIGHT_MODES:
        raise ValueError(f"membrane_weight_mode must be one of: {', '.join(ALLOWED_MEMBRANE_WEIGHT_MODES)}")
    if membrane_zdc_length_mode not in ALLOWED_MEMBRANE_ZDC_LENGTH_MODES:
        raise ValueError(f"membrane_zdc_length_mode must be one of: {', '.join(ALLOWED_MEMBRANE_ZDC_LENGTH_MODES)}")
    requested_membrane_weight_mode = membrane_weight_mode
    membrane_weight_mode = canonical_membrane_weight_mode(membrane_weight_mode)

    effective_radius = pores["radius_m"].to_numpy(dtype=float) * pore_radius_scale
    cp = pore_polarization_conductance(
        frequency_hz,
        effective_radius,
        pores["weight"].to_numpy(dtype=float),
        params,
    )

    membrane_mode = "geometry_zdc"
    zdc_column_by_mode = {
        "throat": "zdc_ohm",
        "center_to_center": "zdc_center_to_center_ohm",
        "conduit": "zdc_conduit_ohm",
    }
    zdc_column = zdc_column_by_mode[membrane_zdc_length_mode]
    if zdc_column in throats.columns:
        zdc = throats[zdc_column].to_numpy(dtype=float)
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
        effective_zdc_length = np.full(len(throats), np.nan, dtype=float)
        effective_active_area = np.full(len(throats), np.nan, dtype=float)
    else:
        relaxation_length_column = (
            "membrane_relaxation_length_m" if "membrane_relaxation_length_m" in throats.columns else "length_m"
        )
        effective_length = throats[relaxation_length_column].to_numpy(dtype=float) * membrane_length_scale
        effective_zdc = zdc * membrane_zdc_scale
        effective_zdc_length = (
            throats["membrane_zdc_length_m"].to_numpy(dtype=float)
            if "membrane_zdc_length_m" in throats.columns
            else throats["length_m"].to_numpy(dtype=float)
        )
        effective_active_area = (
            throats["membrane_active_area_m2"].to_numpy(dtype=float)
            if "membrane_active_area_m2" in throats.columns
            else np.full(len(throats), np.nan, dtype=float)
        )
        cm = membrane_polarization_conductance(
            frequency_hz,
            effective_length,
            membrane_weights(throats, membrane_weight_mode),
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
            "membrane_effective_length_m_mean": np.full(len(frequency_hz), nanmean_or_nan(effective_length)),
            "membrane_relaxation_length_m_mean": np.full(len(frequency_hz), nanmean_or_nan(effective_length)),
            "membrane_zdc_length_m_mean": np.full(len(frequency_hz), nanmean_or_nan(effective_zdc_length)),
            "membrane_active_area_m2_mean": np.full(len(frequency_hz), nanmean_or_nan(effective_active_area)),
            "membrane_effective_zdc_ohm_mean": np.full(len(frequency_hz), nanmean_or_nan(effective_zdc)),
        }
    )
    finite_length = effective_length[np.isfinite(effective_length)]
    finite_zdc_length = effective_zdc_length[np.isfinite(effective_zdc_length)]
    finite_active_area = effective_active_area[np.isfinite(effective_active_area)]
    finite_zdc = effective_zdc[np.isfinite(effective_zdc)]
    finite_radius = effective_radius[np.isfinite(effective_radius)]
    explicit_membrane_geometry = any(
        column in throats.columns
        for column in ("membrane_relaxation_length_m", "membrane_zdc_length_m", "membrane_active_area_m2", "membrane_active_radius_m")
    )
    membrane_geometry_mode = throats.attrs.get(
        "membrane_geometry_mode",
        "explicit_membrane_geometry_columns" if explicit_membrane_geometry else "single_throat_geometry",
    )
    metadata = {
        "spectrum_quantity": "water_phase_polarization_increment",
        "effective_conductivity_requires_full_grid_solve": True,
        "comparison_note": (
            "delta_sigma_* columns are local apparent water-phase polarization increments. "
            "Compare Niu effective conductivity only after AC3D/full-grid upscaling."
        ),
        "n_pores_or_bins": int(len(pores)),
        "n_throats_or_bins": int(len(throats)),
        "pore_radius_scale": float(pore_radius_scale),
        "pore_effective_radius_m": {
            "min": float(finite_radius.min()) if finite_radius.size else None,
            "mean": float(finite_radius.mean()) if finite_radius.size else None,
            "max": float(finite_radius.max()) if finite_radius.size else None,
        },
        "membrane_mode": membrane_mode,
        "membrane_geometry_mode": membrane_geometry_mode,
        "membrane_relaxation_length_source": throats.attrs.get(
            "membrane_relaxation_length_source",
            "membrane_relaxation_length_m" if "membrane_relaxation_length_m" in throats.columns else "length_m",
        ),
        "membrane_zdc_length_source": throats.attrs.get(
            "membrane_zdc_length_source",
            "membrane_zdc_length_m" if "membrane_zdc_length_m" in throats.columns else "length_m",
        ),
        "membrane_zdc_geometry_source": throats.attrs.get(
            "membrane_zdc_geometry_source",
            "precomputed_zdc_ohm",
        ),
        "membrane_weight_mode": membrane_weight_mode,
        "membrane_weight_mode_requested": requested_membrane_weight_mode,
        "membrane_weight_mode_note": (
            "volume is the historical/default implementation. Other modes are diagnostic or candidate "
            "Eq. 10 f(L) convolution interpretations and do not modify extracted geometry or Zdc. "
            "volume_density_renormalized_diagnostic/volume_linear_density renormalize volume/L values "
            "directly; they are diagnostic modes and are not the same as a closed linear-density integral "
            "f(L) dL."
        ),
        "membrane_zdc_length_mode": membrane_zdc_length_mode,
        "membrane_length_scale": float(membrane_length_scale),
        "membrane_zdc_scale": float(membrane_zdc_scale),
        "membrane_effective_length_m": {
            "min": float(finite_length.min()) if finite_length.size else None,
            "mean": float(finite_length.mean()) if finite_length.size else None,
            "max": float(finite_length.max()) if finite_length.size else None,
        },
        "membrane_relaxation_length_m": {
            "min": float(finite_length.min()) if finite_length.size else None,
            "mean": float(finite_length.mean()) if finite_length.size else None,
            "max": float(finite_length.max()) if finite_length.size else None,
        },
        "membrane_zdc_length_m": {
            "min": float(finite_zdc_length.min()) if finite_zdc_length.size else None,
            "mean": float(finite_zdc_length.mean()) if finite_zdc_length.size else None,
            "max": float(finite_zdc_length.max()) if finite_zdc_length.size else None,
        },
        "membrane_active_area_m2": {
            "min": float(finite_active_area.min()) if finite_active_area.size else None,
            "mean": float(finite_active_area.mean()) if finite_active_area.size else None,
            "max": float(finite_active_area.max()) if finite_active_area.size else None,
        },
        "membrane_effective_zdc_ohm": {
            "min": float(finite_zdc.min()) if finite_zdc.size else None,
            "mean": float(finite_zdc.mean()) if finite_zdc.size else None,
            "max": float(finite_zdc.max()) if finite_zdc.size else None,
        },
        "pnextract_raw_throat_count": throats.attrs.get("pnextract_raw_throat_count"),
        "pnextract_excluded_boundary_throat_count": throats.attrs.get("pnextract_excluded_boundary_throat_count"),
        "pnextract_include_boundary_throats": throats.attrs.get("pnextract_include_boundary_throats"),
        "parameters": params.__dict__,
    }
    if dynamic_pore_size_metadata is not None:
        metadata.update(dynamic_pore_size_metadata)
    return out, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["figure5", "pnextract"], required=True)
    parser.add_argument("--figure5", default=str(DEFAULT_FIGURE5_PATH))
    parser.add_argument("--network-dir", default=str(ROOT / "outputs" / "figure5_pnextract_comparison" / "network_parsed"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--metadata-out")
    parser.add_argument(
        "--parameter-mode",
        choices=["niu2020-paper", "project-extracted"],
        default="niu2020-paper",
    )
    parser.add_argument("--dynamic-pore-size-manifest")
    parser.add_argument(
        "--include-boundary-throats",
        action="store_true",
        help="Diagnostic only: include pnextract artificial boundary throats in membrane/pore spectra.",
    )
    parser.add_argument("--frequencies", nargs="+", type=float)
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
    parser.add_argument(
        "--membrane-weight-mode",
        choices=list(ALLOWED_MEMBRANE_WEIGHT_MODES),
        default="volume",
        help=(
            "Eq. 10 f(L) convolution weight mode. Default preserves the historical direct volume-weighted "
            "implementation; other modes are diagnostic/candidate interpretations, not geometry scaling."
        ),
    )
    parser.add_argument(
        "--membrane-zdc-length-mode",
        choices=list(ALLOWED_MEMBRANE_ZDC_LENGTH_MODES),
        default="throat",
        help=(
            "Geometry-derived dc resistance path for membrane Zdc. This selects among extracted geometry "
            "columns and is not a multiplicative zdc_scale."
        ),
    )
    args = parser.parse_args()

    params, dynamic_pore_size_metadata = resolve_polarization_parameters(
        args.parameter_mode,
        Path(args.dynamic_pore_size_manifest) if args.dynamic_pore_size_manifest else None,
    )
    if args.source == "figure5":
        pores, throats = load_figure5(Path(args.figure5))
    else:
        pores, throats = load_pnextract(
            Path(args.network_dir),
            params,
            include_boundary_throats=args.include_boundary_throats,
        )

    spectra, metadata = compute_spectra(
        np.asarray(args.frequencies, dtype=float) if args.frequencies else default_frequencies(),
        pores,
        throats,
        params,
        args.figure5_zdc_ohm,
        pore_radius_scale=args.pore_radius_scale,
        membrane_length_scale=args.membrane_length_scale,
        membrane_zdc_scale=args.membrane_zdc_scale,
        membrane_weight_mode=args.membrane_weight_mode,
        membrane_zdc_length_mode=args.membrane_zdc_length_mode,
        dynamic_pore_size_metadata=dynamic_pore_size_metadata,
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

