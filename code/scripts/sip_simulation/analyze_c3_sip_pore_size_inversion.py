"""What pore-size information can be inverted from the C3 SIP spectra, mechanism by mechanism.

Reads a result package written by ``run_ideal_two_pore_channel_sip2d.py`` with
``configs/sip_simulation/c3_two_pore_niu2020_sip2d_config.py`` and writes
``source_data/inversion_*.csv`` plus ``source_data/inversion_summary.json``.

Inversion routes (data restricted to the laboratory SIP band, default 1e-3-1e4 Hz):

1. Debye decomposition (Nordsiek & Weller, 2008) + ``r = sqrt(2 D tau)`` (Schwarz, 1962;
   Revil et al., 2014; Zhang et al., 2018), with Niu's D = 1.3e-9 m2/s and, as a
   sensitivity, the clay Stern-layer value 3.8e-12 m2/s.
2. Peak frequency of sigma'': ``r = sqrt(D / (pi f_peak))`` (Scott & Barker, 2003 style).
3. Amplitude: for a uniform water-phase increment with an insulating solid,
   ``sigma_inf - sigma_0 = 2 Sigma_S / (F Lambda)`` -> apparent dynamic pore size Lambda.
4. Membrane-only spectrum read under two hypotheses: Debye/pore (radius) and
   Titov/membrane (``L = sqrt(4 D tau_m)``), to show the mechanism ambiguity.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "src"))

from pore_scale_electrical_2d import relaxation_decomposition  # noqa: E402

D_NIU = 1.3e-9  # Niu et al. (2020) Table 1; Revil et al. (2014): 1.32e-9 for clean sands/sandstones
D_CLAY = 3.8e-12  # Revil et al. (2014) / Zhang et al. (2018): Stern layer on clay minerals
RUNS = ("pore_niu", "membrane", "all_niu", "pore_regional", "all_regional", "interfacial")


def load(results: Path):
    manifest = json.loads((results / "provenance" / "run_manifest.json").read_text(encoding="utf-8"))
    sweeps = {name: pd.read_csv(results / "simulation_sweeps" / name / "sweep_results.csv") for name in RUNS}
    lam = pd.read_csv(results / "source_data" / "dynamic_pore_size_table.csv")
    return manifest, sweeps, lam


def band(frame: pd.DataFrame, fmin: float, fmax: float):
    keep = frame["frequency_hz"].between(fmin, fmax)
    sub = frame.loc[keep]
    return sub["frequency_hz"].to_numpy(), sub["sigma_real_s_m"].to_numpy() + 1j * sub["sigma_imag_s_m"].to_numpy()


def peaks(x: np.ndarray, a: np.ndarray, threshold: float = 0.01) -> list[dict]:
    """Local maxima of an amplitude distribution with the amplitude fraction between valleys."""

    order = np.argsort(x)
    x, a = x[order], a[order]
    if a.max() <= 0:
        return []
    idx = [i for i in range(1, len(a) - 1) if a[i] > a[i - 1] and a[i] >= a[i + 1] and a[i] > threshold * a.max()]
    edges = [0] + [idx[k] + int(np.argmin(a[idx[k] : idx[k + 1]])) for k in range(len(idx) - 1)] + [len(a)]
    out = []
    for k, i in enumerate(idx):
        seg = slice(edges[k], edges[k + 1])
        out.append(
            {
                "peak_um": float(x[i]),
                "centroid_um": float(10 ** np.average(np.log10(x[seg]), weights=a[seg] + 1e-300)),
                "amplitude_fraction": float(a[seg].sum() / a.sum()),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results" / "c3_two_pore_niu2020_sip2d_20261014")
    parser.add_argument("--band-min-hz", type=float, default=1e-3)
    parser.add_argument("--band-max-hz", type=float, default=1e4)
    parser.add_argument("--regularization", type=float, default=1e-3)
    args = parser.parse_args()
    out = args.results / "source_data"
    manifest, sweeps, lam = load(args.results)
    if manifest["status"] != "complete":
        print(f"warning: result package status is {manifest['status']!r}", flush=True)
    params = manifest["parameters"]
    geo = manifest["geometry"]
    truth = {
        "big_radius_um": geo["big_radius_m"] * 1e6,
        "small_radius_um": geo["small_radius_m"] * 1e6,
        "throat_half_width_um": 0.5 * geo["throat_width_m"] * 1e6,
        "throat_length_um": geo["throat_length_m"] * 1e6,
        "dynamic_pore_size_um": manifest["global_dynamic_pore_size_m"] * 1e6,
        "formation_factor": manifest["dc_formation_factor"],
    }

    # Relaxation-time grid: one decade beyond the band on each side, 10 bins per decade.
    f0 = np.logspace(math.log10(args.band_min_hz) - 1, math.log10(args.band_max_hz) + 1, 10 * int(round(math.log10(args.band_max_hz / args.band_min_hz) + 2)) + 1)
    tau = 1.0 / (2.0 * np.pi * f0)
    rows, rtd_tables, summary = [], [], {"truth": truth, "band_hz": [args.band_min_hz, args.band_max_hz], "regularization": args.regularization, "routes": {}}

    for name in ("pore_niu", "membrane", "all_niu", "pore_regional", "all_regional"):
        f, sigma = band(sweeps[name], args.band_min_hz, args.band_max_hz)
        fit = relaxation_decomposition(f, sigma, {"debye": (tau, "debye")}, args.regularization)
        amp = fit.amplitudes_s_m["debye"]
        r_niu = np.sqrt(2 * tau * D_NIU) * 1e6
        r_clay = np.sqrt(2 * tau * D_CLAY) * 1e6
        rtd_tables.append(pd.DataFrame({"run": name, "kernel": "debye", "relaxation_time_s": tau, "amplitude_s_m": amp,
                                        "radius_um_D1.3e-9": r_niu, "radius_um_D3.8e-12": r_clay}))
        # Peak frequency of sigma'' in the band (dielectric slope excluded by using the DD prediction minus i w eps_inf).
        imag_pol = (fit.prediction.imag - 2 * np.pi * f * fit.epsilon_inf_f_m)
        f_peak = float(f[np.argmax(imag_pol)])
        delta_sigma = float(amp.sum())
        formation_factor = params["water_conductivity_s_m"] / fit.sigma_0_s_m
        lambda_apparent_um = 2 * params["surface_conductance_s"] / (formation_factor * delta_sigma) * 1e6
        route = {
            "debye_peaks_D_niu": peaks(r_niu, amp),
            "debye_peaks_D_clay": peaks(r_clay, amp),
            "sigma0_s_m": fit.sigma_0_s_m,
            "delta_sigma_s_m": delta_sigma,
            "chargeability": delta_sigma / (fit.sigma_0_s_m + delta_sigma),
            "formation_factor_from_sigma0": formation_factor,
            "lambda_from_amplitude_um": lambda_apparent_um,
            "f_peak_hz": f_peak,
            "r_peak_um_D_niu": math.sqrt(D_NIU / (math.pi * f_peak)) * 1e6,
            "imag_rmse_over_max": fit.imag_rmse_s_m / float(np.max(np.abs(sigma.imag))),
        }
        summary["routes"][name] = route
        for p in route["debye_peaks_D_niu"]:
            rows.append({"run": name, "route": "debye_D1.3e-9", **p})
        for p in route["debye_peaks_D_clay"]:
            rows.append({"run": name, "route": "debye_D3.8e-12", **p})
        rows.append({"run": name, "route": "peak_frequency_D1.3e-9", "peak_um": route["r_peak_um_D_niu"], "centroid_um": np.nan, "amplitude_fraction": np.nan})
        rows.append({"run": name, "route": "amplitude_Lambda", "peak_um": lambda_apparent_um, "centroid_um": np.nan, "amplitude_fraction": np.nan})

    # Membrane-only spectrum under the two mechanism hypotheses.
    f, sigma = band(sweeps["membrane"], args.band_min_hz, args.band_max_hz)
    as_pore = relaxation_decomposition(f, sigma, {"x": (tau, "debye")}, args.regularization)
    as_membrane = relaxation_decomposition(f, sigma, {"x": (tau, "titov")}, args.regularization)
    radius = np.sqrt(2 * tau * D_NIU) * 1e6
    length = np.sqrt(4 * tau * D_NIU) * 1e6
    rtd_tables.append(pd.DataFrame({"run": "membrane", "kernel": "titov", "relaxation_time_s": tau,
                                    "amplitude_s_m": as_membrane.amplitudes_s_m["x"], "radius_um_D1.3e-9": radius,
                                    "radius_um_D3.8e-12": np.sqrt(2 * tau * D_CLAY) * 1e6, "length_um_D1.3e-9": length}))
    summary["membrane_hypotheses"] = {
        "as_pore_debye": {"peaks_radius_um": peaks(radius, as_pore.amplitudes_s_m["x"]), "imag_rmse_over_max": as_pore.imag_rmse_s_m / float(np.max(sigma.imag))},
        "as_membrane_titov": {"peaks_length_um": peaks(length, as_membrane.amplitudes_s_m["x"]), "imag_rmse_over_max": as_membrane.imag_rmse_s_m / float(np.max(sigma.imag))},
        "note": "Both kernels fit the same spectrum; the mechanism (and thus radius vs throat length) is not identifiable from spectral shape alone.",
    }
    for p in summary["membrane_hypotheses"]["as_membrane_titov"]["peaks_length_um"]:
        rows.append({"run": "membrane", "route": "titov_length_D1.3e-9", **p})

    # Interfacial polarization: characteristic frequency from the full band (no length scale in the model).
    inter = sweeps["interfacial"]
    eps = inter["effective_relative_permittivity"].to_numpy()
    fi = inter["frequency_hz"].to_numpy()
    mid = 0.5 * (eps[0] + eps[-1])
    summary["interfacial"] = {
        "eps_low_relative": float(eps[0]),
        "eps_high_relative": float(eps[-1]),
        "transition_frequency_hz": float(fi[np.argmin(np.abs(eps - mid))]),
        "radius_if_misread_um_D_niu": float(math.sqrt(D_NIU / (math.pi * fi[np.argmin(np.abs(eps - mid))])) * 1e6),
    }

    pd.DataFrame(rows).to_csv(out / "inversion_peaks.csv", index=False)
    pd.concat(rtd_tables, ignore_index=True).to_csv(out / "inversion_relaxation_distributions.csv", index=False)
    (out / "inversion_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "routes"}, indent=1, ensure_ascii=False)[:3000])
    for name, route in summary["routes"].items():
        print(name, "F=%.2f Lambda_amp=%.3f um f_peak=%.3g Hz r_peak=%.3g um" % (route["formation_factor_from_sigma0"], route["lambda_from_amplitude_um"], route["f_peak_hz"], route["r_peak_um_D_niu"]),
              [(round(p["peak_um"], 3), round(p["amplitude_fraction"], 3)) for p in route["debye_peaks_D_niu"]])


if __name__ == "__main__":
    main()
