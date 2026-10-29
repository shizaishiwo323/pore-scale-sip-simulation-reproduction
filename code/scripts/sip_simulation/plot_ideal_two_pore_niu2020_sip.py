"""Plot sigma', sigma'' and phase of the Niu (2020)-faithful ideal two-pore channel run.

Reads ``simulation_sweeps/<run>/sweep_results.csv`` written by
``run_ideal_two_pore_channel_sip2d.py --config configs/sip_simulation/ideal_two_pore_channel_sip2d_niu2020_config.py``
and writes ``figures/sip2d_niu2020_spectra.png``. Vertical dotted lines mark the
pore-body Stern peaks f = D+/(pi R^2) with the run's own D+.

    python code/scripts/sip_simulation/plot_ideal_two_pore_niu2020_sip.py --results results/ideal_two_pore_channel_sip2d_niu2020_20261026
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUNS = {
    "interfacial": ("#7a7a7a", "interfacial (Maxwell-Wagner)"),
    "pore_niu": ("#3b6fb6", "pore / Stern (Niu Eq. 12)"),
    "membrane": ("#d28e38", "membrane (Titov throat)"),
    "all_niu": ("#111111", "all mechanisms"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads((args.results / "configs" / "resolved_config.json").read_text(encoding="utf-8"))
    stern_d = config["polarization"]["stern_diffusion_m2_s"]
    geometry = config["geometry"]
    peaks = {f"R = {geometry[k] * 1e6:g} um": stern_d / (np.pi * geometry[k] ** 2) for k in ("big_radius_m", "small_radius_m")}

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    for name, (color, label) in RUNS.items():
        frame = pd.read_csv(args.results / "simulation_sweeps" / name / "sweep_results.csv")
        f = frame["frequency_hz"]
        axes[0].loglog(f, frame["sigma_real_s_m"], color=color, lw=1.2, label=label)
        axes[1].loglog(f, frame["sigma_imag_s_m"].where(frame["sigma_imag_s_m"] > 0), color=color, lw=1.2)
        axes[2].semilogx(f, frame["phase_mrad"], color=color, lw=1.2)
    for ax in axes[1:]:
        for fpk in peaks.values():
            ax.axvline(fpk, color="#3b6fb6", ls=":", lw=0.8)
    for text, fpk in peaks.items():
        axes[1].text(fpk, 0.97, text, transform=axes[1].get_xaxis_transform(), rotation=90, ha="right", va="top", fontsize=6, color="#3b6fb6")
    for ax, ylabel, tag in zip(axes, ("sigma' (S/m)", "sigma'' (S/m)", "Phase (mrad)"), "abc"):
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(ylabel)
        ax.set_title(tag, loc="left", fontweight="bold")
        ax.grid(alpha=0.3, lw=0.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    out = args.results / "figures" / "sip2d_niu2020_spectra.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
