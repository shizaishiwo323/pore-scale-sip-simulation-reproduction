#!/usr/bin/env python3
"""Compare complex64 and complex128 AC3D checkpoint sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CHECKPOINT_FREQUENCIES = [1e-3, 1e-2, 1e-1, 1.0, 1e3, 1e6, 1e9]


def _load_required(path: Path, frequencies: list[float]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"frequency_hz", "effective_sigma_real_s_m", "effective_sigma_imag_s_m"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    available = frame["frequency_hz"].to_numpy(dtype=float)
    missing_freqs = [float(freq) for freq in frequencies if not np.any(np.isclose(available, freq, rtol=1e-12, atol=0.0))]
    if missing_freqs:
        raise ValueError(f"{path} is missing checkpoint frequencies: {missing_freqs}")
    rows = []
    for freq in frequencies:
        idx = np.flatnonzero(np.isclose(available, freq, rtol=1e-12, atol=0.0))[0]
        rows.append(frame.iloc[int(idx)])
    return pd.DataFrame(rows).reset_index(drop=True)


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.abs(candidate - reference) / np.maximum(np.abs(reference), np.finfo(float).eps)


def compare_precision_checkpoints(
    complex64_csv: Path,
    complex128_csv: Path,
    *,
    checkpoint_frequencies: list[float] | None = None,
    max_relative_error: float = 1e-2,
) -> dict[str, object]:
    frequencies = [float(freq) for freq in (checkpoint_frequencies or DEFAULT_CHECKPOINT_FREQUENCIES)]
    c64 = _load_required(Path(complex64_csv), frequencies)
    c128 = _load_required(Path(complex128_csv), frequencies)
    real_error = _relative_error(
        c64["effective_sigma_real_s_m"].to_numpy(dtype=float),
        c128["effective_sigma_real_s_m"].to_numpy(dtype=float),
    )
    imag_error = _relative_error(
        c64["effective_sigma_imag_s_m"].to_numpy(dtype=float),
        c128["effective_sigma_imag_s_m"].to_numpy(dtype=float),
    )
    max_real = round(float(np.nanmax(real_error)), 12)
    max_imag = round(float(np.nanmax(imag_error)), 12)
    rows = []
    for idx, freq in enumerate(frequencies):
        rows.append(
            {
                "frequency_hz": freq,
                "complex64_real_s_m": float(c64.loc[idx, "effective_sigma_real_s_m"]),
                "complex128_real_s_m": float(c128.loc[idx, "effective_sigma_real_s_m"]),
                "complex64_imag_s_m": float(c64.loc[idx, "effective_sigma_imag_s_m"]),
                "complex128_imag_s_m": float(c128.loc[idx, "effective_sigma_imag_s_m"]),
                "relative_error_real": float(real_error[idx]),
                "relative_error_imag": float(imag_error[idx]),
            }
        )
    return {
        "checkpoint_frequencies_hz": frequencies,
        "max_allowed_relative_error": float(max_relative_error),
        "max_relative_error_real": max_real,
        "max_relative_error_imag": max_imag,
        "passed": bool(max(max_real, max_imag) <= max_relative_error),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--complex64-csv", required=True, type=Path)
    parser.add_argument("--complex128-csv", required=True, type=Path)
    parser.add_argument("--checkpoint-frequencies", nargs="+", type=float, default=DEFAULT_CHECKPOINT_FREQUENCIES)
    parser.add_argument("--max-relative-error", type=float, default=1e-2)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    args = parser.parse_args()

    report = compare_precision_checkpoints(
        args.complex64_csv,
        args.complex128_csv,
        checkpoint_frequencies=args.checkpoint_frequencies,
        max_relative_error=args.max_relative_error,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = pd.DataFrame(report["rows"])
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
