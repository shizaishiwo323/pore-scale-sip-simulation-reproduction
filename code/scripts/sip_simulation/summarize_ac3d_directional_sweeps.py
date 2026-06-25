#!/usr/bin/env python3
"""Summarize x/y/z AC3D sweep CSVs into directional means and anisotropy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "frequency_hz",
    "effective_sigma_real_s_m",
    "effective_sigma_imag_s_m",
}


def _load_sweep(path: Path, direction: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if "direction" in frame.columns and not frame["direction"].astype(str).eq(direction).all():
        raise ValueError(f"{path} direction column does not match {direction}")
    out = frame.sort_values("frequency_hz").reset_index(drop=True).copy()
    out["direction"] = direction
    return out


def _relative_range(values: np.ndarray) -> np.ndarray:
    numerator = np.nanmax(values, axis=1)
    denominator = np.nanmin(values, axis=1)
    return numerator / np.maximum(np.abs(denominator), np.finfo(float).eps)


def summarize_directional_sweeps(paths: dict[str, Path]) -> dict[str, object]:
    directions = ["x", "y", "z"]
    missing_directions = [direction for direction in directions if direction not in paths]
    if missing_directions:
        raise ValueError(f"missing directional sweep(s): {missing_directions}")

    sweeps = {direction: _load_sweep(Path(paths[direction]), direction) for direction in directions}
    frequency_grid = sweeps["x"]["frequency_hz"].to_numpy(dtype=float)
    for direction in directions[1:]:
        other = sweeps[direction]["frequency_hz"].to_numpy(dtype=float)
        if not np.array_equal(frequency_grid, other):
            raise ValueError("x/y/z sweeps must use the same exact frequency grid")

    real = np.column_stack([sweeps[direction]["effective_sigma_real_s_m"].to_numpy(dtype=float) for direction in directions])
    imag = np.column_stack([sweeps[direction]["effective_sigma_imag_s_m"].to_numpy(dtype=float) for direction in directions])
    rows = {
        "frequency_hz": frequency_grid,
        "sigma_xx_real_s_m": real[:, 0],
        "sigma_yy_real_s_m": real[:, 1],
        "sigma_zz_real_s_m": real[:, 2],
        "sigma_xx_imag_s_m": imag[:, 0],
        "sigma_yy_imag_s_m": imag[:, 1],
        "sigma_zz_imag_s_m": imag[:, 2],
        "directional_mean_real_s_m": np.nanmean(real, axis=1),
        "directional_mean_imag_s_m": np.nanmean(imag, axis=1),
        "anisotropy_ratio_real": _relative_range(real),
        "anisotropy_ratio_imag": _relative_range(np.abs(imag)),
    }
    residual_columns = [sweeps[direction].get("true_residual_norm") for direction in directions]
    if all(column is not None for column in residual_columns):
        residuals = np.column_stack([column.to_numpy(dtype=float) for column in residual_columns if column is not None])
        rows["max_true_residual_norm"] = np.nanmax(residuals, axis=1)
    passed_columns = [sweeps[direction].get("true_residual_passed") for direction in directions]
    if all(column is not None for column in passed_columns):
        passed = np.column_stack([column.astype(bool).to_numpy() for column in passed_columns if column is not None])
        rows["all_true_residual_passed"] = np.all(passed, axis=1)
    frame = pd.DataFrame(rows)
    return {"directions": directions, "frame": frame}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", required=True, type=Path)
    parser.add_argument("--y", required=True, type=Path)
    parser.add_argument("--z", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()

    summary = summarize_directional_sweeps({"x": args.x, "y": args.y, "z": args.z})
    frame: pd.DataFrame = summary["frame"]  # type: ignore[assignment]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    report = {
        "directions": summary["directions"],
        "out_csv": str(args.out_csv),
        "frequency_count": int(len(frame)),
        "max_anisotropy_ratio_real": float(frame["anisotropy_ratio_real"].max()),
        "max_anisotropy_ratio_imag": float(frame["anisotropy_ratio_imag"].max()),
    }
    if "max_true_residual_norm" in frame:
        report["max_true_residual_norm"] = float(frame["max_true_residual_norm"].max())
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
