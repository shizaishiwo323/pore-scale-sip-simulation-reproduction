"""对比样品16 Avizo与默认全分辨率pnextract的孔/孔喉几何分布。"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
AVIZO = ROOT / "results/sample16_avizo_ct_geometry_gpu7_20260916/pore_network"
PNEXTRACT = ROOT / "results/sample16_gpu_7freq_20260912_2340/pore_network/network_parsed"
OUT = ROOT / "results/sample16_avizo_vs_pnextract_geometry_20260916"
COLORS = {"Avizo": "#E15759", "pnextract": "#4C78A8"}


def summarize(values_um, weights, method, quantity):
    return {
        "method": method, "quantity": quantity, "count": len(values_um),
        "min_um": float(values_um.min()), "median_um": float(np.median(values_um)),
        "mean_um": float(values_um.mean()), "max_um": float(values_um.max()),
        "volume_weighted_mean_um": float(np.average(values_um, weights=weights)),
        "total_weight_m3": float(weights.sum()),
    }


def histogram(values_um, weights, edges, method, quantity):
    count, _ = np.histogram(values_um, bins=edges)
    volume, _ = np.histogram(values_um, bins=edges, weights=weights)
    return pd.DataFrame({
        "method": method, "quantity": quantity,
        "left_um": edges[:-1], "right_um": edges[1:], "center_um": np.sqrt(edges[:-1] * edges[1:]),
        "count": count, "frequency_fraction": count / count.sum(), "volume_fraction": volume / volume.sum(),
    })


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    avizo_pores, avizo_throats = pd.read_csv(AVIZO / "pores.csv"), pd.read_csv(AVIZO / "throats.csv")
    pn_pores, pn_throats = pd.read_csv(PNEXTRACT / "pores.csv"), pd.read_csv(PNEXTRACT / "throats.csv")
    pn_raw_throat_count = len(pn_throats)
    pn_throats = pn_throats[(pn_throats.pore1_id >= 1) & (pn_throats.pore2_id >= 1)].copy()
    datasets = {
        "pore_radius": {
            "Avizo": (avizo_pores.pore_radius_m.to_numpy() * 1e6, avizo_pores.pore_volume_m3.to_numpy()),
            "pnextract": (pn_pores.pore_radius_m.to_numpy() * 1e6, pn_pores.pore_volume_m3.to_numpy()),
        },
        "throat_length": {
            "Avizo": (avizo_throats.throat_length_m.to_numpy() * 1e6, avizo_throats.throat_volume_m3.to_numpy()),
            "pnextract": (pn_throats.throat_length_m.to_numpy() * 1e6, pn_throats.throat_volume_m3.to_numpy()),
        },
    }
    records, summaries = [], []
    for quantity, methods in datasets.items():
        all_values = np.concatenate([values for values, _ in methods.values()])
        edges = np.geomspace(all_values.min(), all_values.max(), 41)
        for method, (values, weights) in methods.items():
            assert np.isfinite(values).all() and np.isfinite(weights).all()
            assert (values > 0).all() and (weights > 0).all()
            records.append(histogram(values, weights, edges, method, quantity))
            summaries.append(summarize(values, weights, method, quantity))
    source = pd.concat(records, ignore_index=True)
    summary = pd.DataFrame(summaries)
    source.to_csv(OUT / "avizo_vs_pnextract_histogram_source_data.csv", index=False)
    summary.to_csv(OUT / "avizo_vs_pnextract_summary.csv", index=False)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7, "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.8, "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex="col")
    panels = [
        (0, 0, "pore_radius", "frequency_fraction", "Pore radius: frequency fraction"),
        (0, 1, "throat_length", "frequency_fraction", "Throat length: frequency fraction"),
        (1, 0, "pore_radius", "volume_fraction", "Pore radius: volume fraction"),
        (1, 1, "throat_length", "volume_fraction", "Throat length: volume fraction"),
    ]
    for label, (row, col, quantity, fraction, title) in zip("abcd", panels):
        ax = axes[row, col]
        for method in ("Avizo", "pnextract"):
            frame = source[(source.quantity == quantity) & (source.method == method)]
            ax.stairs(frame[fraction], frame[["left_um", "right_um"]].to_numpy().ravel()[::2].tolist() + [frame.right_um.iloc[-1]],
                      fill=True, alpha=0.28, color=COLORS[method], linewidth=1.1, label=method)
            ax.stairs(frame[fraction], np.r_[frame.left_um.to_numpy(), frame.right_um.iloc[-1]],
                      color=COLORS[method], linewidth=1.25)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=8)
        ax.set_ylabel("Fraction per log bin")
        ax.grid(axis="y", color="#E5E5E5", lw=0.5)
        ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=9)
    axes[1, 0].set_xlabel(r"Pore radius ($\mu$m)")
    axes[1, 1].set_xlabel(r"Throat length ($\mu$m)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.52, 1.005), ncol=2)
    fig.subplots_adjust(left=.10, right=.985, bottom=.11, top=.90, hspace=.30, wspace=.25)
    figure = OUT / "sample16_avizo_vs_pnextract_pore_throat_distributions.png"
    fig.savefig(figure, dpi=600, bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "figure_contract": {
            "core_conclusion": "Avizo and pnextract produce materially different Sample16 pore-radius and throat-length distributions under identical normalized-bin comparisons.",
            "evidence_chain": {"a": "pore count", "b": "throat count", "c": "pore volume", "d": "throat volume"},
            "archetype": "quantitative grid", "backend": "Python/matplotlib", "export": "183 mm PNG at 600 dpi",
        },
        "comparison": "Same 40 logarithmic bins per geometric quantity; each method/fraction is normalized independently to sum to one.",
        "pore_scope": "all positive finite extracted pores",
        "throat_scope": "internal throats only; pnextract boundary throats excluded for comparability with Avizo",
        "pnextract_throats": {"raw": pn_raw_throat_count, "internal_used": len(pn_throats), "boundary_excluded": pn_raw_throat_count - len(pn_throats)},
        "avizo_throats_used": len(avizo_throats), "figure_png": str(figure), "png_dpi": 600,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf8")
    (OUT / "README.md").write_text(
        "# 样品16 Avizo 与 pnextract 几何分布对比\n\n"
        "孔使用两种方法的全部有效孔；孔喉仅使用内部孔喉。pnextract原始11096条喉中排除417条边界喉，使用10679条；Avizo使用15113条。\n\n"
        "每个几何量使用完全相同的40个对数分箱。数量分数和体积分数分别在每种方法内归一化为1，因此比较的是分布形状，不是两种方法提取总体积的绝对差异。\n",
        encoding="utf8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
