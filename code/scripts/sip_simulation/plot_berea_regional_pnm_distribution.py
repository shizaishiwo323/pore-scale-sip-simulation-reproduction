"""Pore-body size and throat-length distributions of the pnextract network used by the regional SIP run.

Left: pnextract pore-body inscribed radius (node2), pore-volume weighted.
Right: pnextract throat length (link2 ``throat_length``), throat-volume weighted.
Both are histogrammed on the size bins of Niu et al. (2020) Figure 4 (``Figure5.xlsx``) so that the
paper distribution can be overlaid as a reference. The paper data are NOT simulation inputs; every
size used by the simulation comes from this pnextract run.

Output (PNG only, legend is the only annotation text):
    results/<run>/pore_network/berea_pnextract_pore_throat_distribution.png
    results/<run>/source_data/berea_pnextract_pore_throat_distribution_source_data.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "code" / "scripts" / "pore_network"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_pnextract_network import read_link2, read_node2  # noqa: E402
from plot_berea_regional_sip3d_vs_experiment import DEFAULT_RUN, export  # noqa: E402  (shared rcParams + QA export)

FIGURE5 = PROJECT_ROOT / "data" / "Niu 2020data" / "Figure5.xlsx"
PNEXTRACT_COLOR = "#4C8FD6"
NIU_COLOR = "#111111"


def read_figure5(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_excel(path, sheet_name=0, header=None).iloc[1:]
    out = []
    for cols in ([0, 1], [2, 3]):
        frame = data.iloc[:, cols].apply(pd.to_numeric, errors="coerce").dropna()
        frame.columns = ["size_m", "relative_volume"]
        frame = frame[frame.size_m > 0].sort_values("size_m")
        frame["relative_volume"] /= frame["relative_volume"].sum()
        out.append(frame.reset_index(drop=True))
    return out[0], out[1]


def log_edges(centers: np.ndarray) -> np.ndarray:
    mids = np.sqrt(centers[:-1] * centers[1:])
    return np.concatenate([[centers[0] ** 2 / mids[0]], mids, [centers[-1] ** 2 / mids[-1]]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    args = parser.parse_args()
    run = Path(args.run)
    run = run if run.is_absolute() else PROJECT_ROOT / run
    prefix = run / "pore_network" / "pnextract_input" / "microCT_Berea"
    pores = read_node2(Path(f"{prefix}_node2.dat"))
    throats = read_link2(Path(f"{prefix}_link2.dat"))
    niu_pore, niu_throat = read_figure5(FIGURE5)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"wspace": 0.28})
    rows = []
    panels = [
        (axes[0], niu_pore, pores.pore_radius_m.to_numpy(), pores.pore_volume_m3.to_numpy(),
         "Pore-body size (m)", "pnextract inscribed radius (this run)", "pore_body_radius"),
        (axes[1], niu_throat, throats.throat_length_m.to_numpy(), throats.throat_volume_m3.to_numpy(),
         "Throat length (m)", "pnextract throat length (this run)", "throat_length"),
    ]
    for ax, niu, values, weights, xlabel, label, kind in panels:
        edges = log_edges(niu.size_m.to_numpy())
        hist, _ = np.histogram(values, bins=edges, weights=weights)
        fraction = hist / weights.sum()
        outside = float(weights[(values < edges[0]) | (values >= edges[-1])].sum() / weights.sum())
        ax.stairs(fraction, edges, fill=True, color=PNEXTRACT_COLOR, alpha=0.55, linewidth=0, label=label)
        ax.plot(niu.size_m, niu.relative_volume, color=NIU_COLOR, marker="o", ms=2.2, lw=0.8,
                label="Niu et al. (2020) Fig. 4 (reference only)")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Volume fraction")
        ax.set_ylim(0, max(fraction.max(), niu.relative_volume.max()) * 1.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", handlelength=1.6)
        for lo, hi, f_pn, f_niu in zip(edges[:-1], edges[1:], fraction, niu.relative_volume):
            rows.append({"kind": kind, "bin_left_m": lo, "bin_right_m": hi, "pnextract_volume_fraction": f_pn,
                         "niu2020_figure5_volume_fraction": f_niu, "pnextract_weight_outside_bins": outside})
    out = run / "pore_network" / "berea_pnextract_pore_throat_distribution.png"
    report = export(fig, out)
    pd.DataFrame(rows).to_csv(run / "source_data" / "berea_pnextract_pore_throat_distribution_source_data.csv", index=False)
    print(report)


if __name__ == "__main__":
    main()
