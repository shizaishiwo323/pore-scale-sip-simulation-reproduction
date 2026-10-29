"""Figures for the two-scale regional SIP simulation of LKC sample 16 (PNG only; legend is the only annotation).

Reads ``results/<run>/`` written by ``run_sample16_regional_sip3d.py`` and writes:

    figures/sample16_regional_sip_vs_experiment.png     sigma' and sigma'' of the full 3-D solves vs experiment
    figures/sample16_parameter_screening_sigma_imag.png  total sigma'' of every literature parameter set (linear response)
    pore_network/sample16_multiscale_length_distributions.png  CT pores/throats, NMR micropores, matrix bridges

Simulated curves join the 10 solved frequencies with straight segments (no interpolation between solves);
the screening figure uses the dense linear-response spectra and is labelled as such.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "code" / "scripts" / "pore_network"))
from parse_pnextract_network import read_link1, read_link2, read_node2  # noqa: E402
from plot_berea_regional_sip3d_vs_experiment import export, positive, style_axis  # noqa: E402  (shared style + QA export)

DEFAULT_RUN = PROJECT_ROOT / "results" / "sample16_regional_sip_pnextract_20261014"
X_LIM = (5e-2, 3e4)
REAL_LIM = (1e-3, 1e-2)
IMAG_LIM = (1e-8, 1e-3)
RESOLUTION_TOLERANCE = 0.05
SIM = {
    "central_interfacial": dict(color="#F28E2B", ls=(0, (4, 2)), lw=1.0, label="Dielectric"),
    "central_edl_macro_regional": dict(color="#4C8FD6", ls=(0, (4, 2)), lw=1.0, label="EDL, CT pores (regional)"),
    "central_edl_micro_matrix": dict(color="#8E6BBF", ls=(0, (4, 2)), lw=1.0, label="EDL, matrix micropores (NMR)"),
    "central_membrane_regional": dict(color="#46B37F", ls=(0, (4, 2)), lw=1.0, label="Membrane, CT throats (regional)"),
    "central_membrane_matrix": dict(color="#1B6E4F", ls=(0, (1.2, 1.2)), lw=1.2, label="Membrane, matrix bridges"),
    "central_all": dict(color="#E5534B", ls="-", lw=1.8, label="All, central literature set"),
    "best_all": dict(color="#7A1F1A", ls=(0, (5, 1.5, 1.2, 1.5)), lw=1.4, label="All, best literature set"),
}
MARKER = dict(marker="o", ms=2.4, mew=0.0)
EXP_RELIABLE = dict(color="#111111", ls="none", ms=2.6, mew=0.0, zorder=4)
EXP_COUPLING = dict(color="#111111", ls="none", ms=2.8, mfc="white", mew=0.6, zorder=4)
THESIS = dict(color="#111111", ls="none", marker="D", ms=4.2, mfc="#F2C14E", mew=0.7, zorder=5)


def resolved(frame: pd.DataFrame) -> np.ndarray:
    ok = frame["imag_relative_difference_mean_current_vs_energy"].to_numpy() <= RESOLUTION_TOLERANCE
    return np.where(ok, frame["sigma_imag_s_m"].to_numpy(), np.nan)


def in_window(values, low, high):
    values = np.asarray(values, dtype=float)
    return np.where((values >= low) & (values <= high), values, np.nan)


def experiment_markers(ax, experiment: pd.DataFrame, column: str, marker: str) -> None:
    reliable = experiment.reliable_band
    ax.plot(experiment.frequency_hz[reliable], positive(experiment[column][reliable]), marker=marker,
            label="Experiment LKC-16 (≤1 kHz)", **EXP_RELIABLE)
    ax.plot(experiment.frequency_hz[~reliable], positive(experiment[column][~reliable]), marker=marker,
            label="Experiment, >1 kHz (coupling risk)", **EXP_COUPLING)


def figure_main(run: Path, experiment: pd.DataFrame, thesis: dict, sweep_root: Path, png: Path, show_thesis: bool = True) -> tuple[dict, pd.DataFrame]:
    sweeps = {name: pd.read_csv(sweep_root / name / "sweep_results.csv").sort_values("frequency_hz")
              for name in SIM if (sweep_root / name / "sweep_results.csv").exists()}
    fig, (real_ax, imag_ax) = plt.subplots(2, 1, figsize=(6.4, 5.4), gridspec_kw={"hspace": 0.24})
    fig.subplots_adjust(left=0.11, right=0.58)
    experiment_markers(real_ax, experiment, "experiment_sigma_real_s_m", "o")
    experiment_markers(imag_ax, experiment, "experiment_sigma_imag_s_m", "o")
    if show_thesis:
        real_ax.plot([thesis["frequency_hz"]], [thesis["sigma_real_s_m"]], label="Thesis Table 4.2, σw = 0.12 S/m", **THESIS)
        imag_ax.plot([thesis["frequency_hz"]], [thesis["sigma_imag_s_m"]], label="Thesis Table 4.2, σw = 0.12 S/m", **THESIS)
    rows = []
    for name, style in SIM.items():
        if name not in sweeps:
            continue
        frame = sweeps[name]
        if name in ("central_all", "best_all"):
            real_ax.plot(frame.frequency_hz, in_window(frame.sigma_real_s_m, *REAL_LIM), **style, **MARKER, zorder=3)
        shown = in_window(positive(resolved(frame)), *IMAG_LIM)
        imag_ax.plot(frame.frequency_hz, shown, **style, **MARKER, zorder=2 if name.endswith("all") else 3)
        for row, value in zip(frame.itertuples(index=False), shown):
            rows.append({"run": name, "series": style["label"], "frequency_hz": row.frequency_hz, "sigma_real_s_m": row.sigma_real_s_m,
                         "sigma_imag_s_m": row.sigma_imag_s_m, "sigma_imag_plotted": bool(np.isfinite(value))})
    style_axis(real_ax, REAL_LIM, "σ′ (S/m)")
    style_axis(imag_ax, IMAG_LIM, "σ″ (S/m)")
    for ax in (real_ax, imag_ax):
        ax.set_xlim(*X_LIM)
    imag_ax.set_xlabel("Frequency (Hz)")
    # One shared legend outside the plot areas (experiment markers are identical in both panels).
    handles, labels = imag_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.6, 0.5), handlelength=2.6, fontsize=6.5, frameon=False)
    for row in experiment.itertuples(index=False):
        rows.append({"run": "experiment_LKC-16.csv", "series": "Experiment", "frequency_hz": row.frequency_hz,
                     "sigma_real_s_m": row.experiment_sigma_real_s_m, "sigma_imag_s_m": row.experiment_sigma_imag_s_m,
                     "sigma_imag_plotted": bool(row.experiment_sigma_imag_s_m > 0)})
    rows.append({"run": "thesis_table_4_2", "series": "Thesis", "frequency_hz": thesis["frequency_hz"], "sigma_real_s_m": thesis["sigma_real_s_m"],
                 "sigma_imag_s_m": thesis["sigma_imag_s_m"], "sigma_imag_plotted": True})
    report = export(fig, png)
    return report, pd.DataFrame(rows)


def figure_screening(run: Path, experiment: pd.DataFrame, thesis: dict, prefix: str, png: Path, show_thesis: bool = True) -> dict:
    spectra = pd.read_csv(run / "source_data" / f"{prefix}linear_response_spectra.csv")
    screening = pd.read_csv(run / "source_data" / f"{prefix}linear_response_scenario_screening.csv")
    central = "S1.8nS_D5.10e-10_rho2ct_boundary_Sd0.30nS"
    total = spectra[spectra.mechanism == "all"]
    wide = total.pivot(index="frequency_hz", columns="scenario", values="sigma_imag_s_m")
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.fill_between(wide.index, wide.min(axis=1), wide.max(axis=1), color="#D9D9D9", lw=0, label=f"All, {wide.shape[1]} literature sets")
    no_mm = spectra[(spectra.mechanism == "all_no_matrix_membrane") & (spectra.scenario == central)]
    ax.plot(no_mm.frequency_hz, no_mm.sigma_imag_s_m, color="#4C8FD6", ls=(0, (4, 2)), lw=1.0, label="All without matrix membrane (central)")
    c = total[total.scenario == central]
    ax.plot(c.frequency_hz, c.sigma_imag_s_m, color="#E5534B", lw=1.6, label="All, central literature set")
    b = total[total.scenario == screening.scenario.iloc[0]]
    ax.plot(b.frequency_hz, b.sigma_imag_s_m, color="#7A1F1A", ls=(0, (5, 1.5, 1.2, 1.5)), lw=1.3, label="All, best literature set")
    experiment_markers(ax, experiment, "experiment_sigma_imag_s_m", "o")
    if show_thesis:
        ax.plot([thesis["frequency_hz"]], [thesis["sigma_imag_s_m"]], label="Thesis Table 4.2, σw = 0.12 S/m", **THESIS)
    style_axis(ax, (1e-7, 1e-3), "σ″ (S/m), linear response")
    ax.set_xlim(*X_LIM)
    ax.set_xlabel("Frequency (Hz)")
    ax.legend(loc="upper left", handlelength=2.4, fontsize=5.4, borderaxespad=0.4)
    return export(fig, png)


def volume_hist(values_m, weights, bins):
    hist, _ = np.histogram(values_m, bins=bins, weights=weights)
    return hist / np.sum(weights)


def figure_lengths(run: Path) -> tuple[dict, pd.DataFrame]:
    prefix = run / "pore_network" / "pnextract_input" / "sample16_rev336"
    pores = read_node2(Path(f"{prefix}_node2.dat"))
    throats = read_link1(Path(f"{prefix}_link1.dat")).merge(read_link2(Path(f"{prefix}_link2.dat")), on="throat_id")
    micro = pd.read_csv(run / "source_data" / "nmr_micropore_model_central.csv")
    bridges = pd.read_csv(run / "source_data" / "matrix_bridge_classes.csv")
    bins = np.logspace(-8, -3.4, 47)
    series = [
        ("CT pore-body radius (pnextract)", pores.pore_radius_m, pores.pore_volume_m3, "#4C8FD6"),
        ("CT throat radius (pnextract)", throats.throat_radius_m, throats.throat_volume_m3, "#46B37F"),
        ("Micropore radius (NMR, ρ2 from CT boundary)", micro.radius_m, micro.weight, "#8E6BBF"),
    ]
    fig, (size_ax, length_ax) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"wspace": 0.28})
    rows = []
    for label, values, weights, color in series:
        frac = volume_hist(values, weights, bins)
        size_ax.stairs(frac, bins, color=color, lw=1.2, label=label)
        rows += [{"panel": "radius", "series": label, "bin_left_m": lo, "bin_right_m": hi, "volume_fraction": f}
                 for lo, hi, f in zip(bins[:-1], bins[1:], frac)]
    for label, values, weights, color in [
        ("CT throat length (pnextract)", throats.throat_length_m, throats.throat_volume_m3, "#46B37F"),
        ("Matrix bridge length (CT local thickness)", bridges.bridge_length_m, bridges.voxel_count, "#1B6E4F"),
    ]:
        frac = volume_hist(values, weights, bins)
        length_ax.stairs(frac, bins, color=color, lw=1.2, label=label)
        rows += [{"panel": "length", "series": label, "bin_left_m": lo, "bin_right_m": hi, "volume_fraction": f}
                 for lo, hi, f in zip(bins[:-1], bins[1:], frac)]
    for ax, xlabel in ((size_ax, "Radius (m)"), (length_ax, "Length (m)")):
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Volume fraction")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, None)
        ax.legend(loc="upper left", fontsize=5.6, handlelength=1.6)
    report = export(fig, run / "pore_network" / "sample16_multiscale_length_distributions.png")
    return report, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--variant", default=None, help="second-round variant name (see the config 'variants')")
    args = parser.parse_args()
    run = Path(args.run)
    run = run if run.is_absolute() else PROJECT_ROOT / run
    config = json.loads((run / "configs" / "resolved_config.json").read_text(encoding="utf-8"))
    thesis = config["experiment"]["thesis_point"]
    experiment = pd.read_csv(run / "source_data" / "lkc16_experiment_sigma.csv")
    (run / "figures").mkdir(parents=True, exist_ok=True)
    reports = {}
    if args.variant:
        tag, prefix, root = f"_{args.variant}", f"{args.variant}_", run / "simulation_sweeps" / args.variant
        summary = json.loads((run / "provenance" / f"{args.variant}_summary.json").read_text(encoding="utf-8"))
        show_thesis = summary["definition"]["brine"] == "thesis"
    else:
        tag, prefix, root, show_thesis = "", "", run / "simulation_sweeps", True
    reports["main"], source = figure_main(run, experiment, thesis, root, run / "figures" / f"sample16_regional_sip_vs_experiment{tag}.png", show_thesis)
    source.to_csv(run / "source_data" / f"sample16_regional_sip_vs_experiment{tag}_source_data.csv", index=False)
    reports["screening"] = figure_screening(run, experiment, thesis, prefix, run / "figures" / f"sample16_parameter_screening_sigma_imag{tag}.png", show_thesis)
    if not args.variant:
        reports["lengths"], lengths = figure_lengths(run)
        lengths.to_csv(run / "source_data" / "sample16_multiscale_length_distributions_source_data.csv", index=False)
    (run / "figures" / f"figure_qa_summary{tag}.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps({k: (v["text_audit_exit"], v["collision_exit"]) for k, v in reports.items()}))


if __name__ == "__main__":
    main()
