"""Overlay published volume fractions and extracted network on common bins."""
import json
import numpy as np
import pandas as pd
from plot_niu2020_workbook_histograms import ROOT, plt, AutoMinorLocator, FuncFormatter, centers_to_log_edges


def main():
    source = ROOT / 'data/Niu 2020data/Figure5.xlsx'
    network = ROOT / 'results/niu2020_berea_gpu7_20260912/pore_network/network_parsed'
    out = network.parent / 'geometry_overlay'
    out.mkdir(exist_ok=True)
    raw = pd.read_excel(source)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 14,
                         'axes.linewidth': 0.8, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    checks = []
    for i, ax in enumerate(axes):
        x, paper = raw.iloc[:, 2*i:2*i+2].dropna().to_numpy(float).T
        frame = pd.read_csv(network / ['pores.csv', 'throats.csv'][i])
        field, weight = [('pore_radius_m', 'pore_volume_m3'), ('throat_length_m', 'throat_volume_m3')][i]
        values, weights = frame[[field, weight]].to_numpy(float).T
        assert np.isfinite(values).all() and (values > 0).all()
        assert np.isfinite(weights).all() and (weights >= 0).all() and weights.sum() > 0
        assert np.isfinite(paper).all() and (paper >= 0).all() and np.isclose(paper.sum(), 1)
        edges = centers_to_log_edges(x)
        # 保持论文分箱不变；必要时向外延伸同宽对数分箱，避免截断网络数据。
        ratio = edges[1]/edges[0]
        while values.min() < edges[0]:
            edges = np.r_[edges[0]/ratio, edges]
            paper = np.r_[0., paper]
        while values.max() > edges[-1]:
            edges = np.r_[edges, edges[-1]*ratio]
            paper = np.r_[paper, 0.]
        centers = np.sqrt(edges[:-1]*edges[1:])
        volumes, _ = np.histogram(values, bins=edges, weights=weights)
        counts, _ = np.histogram(values, bins=edges)
        ours = volumes/weights.sum()
        assert counts.sum() == len(values) and np.isclose(ours.sum(), 1)
        for y, color, label in [(paper, '#2878B5', 'Niu 2020'), (ours, '#E58632', 'Extracted network')]:
            ax.bar(centers, y, width=np.diff(edges)*0.78, color=color, edgecolor=color,
                   alpha=0.50, linewidth=0.8, label=label)
        ax.set_xscale('log')
        ax.set_xlim(edges[0]/1.1, edges[-1]*1.1)
        top = max(0.15, np.ceil(max(paper.max(), ours.max())*1.1/0.05)*0.05)
        ax.set_ylim(0, top)
        ax.set_yticks(np.arange(0, top+0.025, 0.05))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(which='major', direction='in', length=7)
        ax.tick_params(which='minor', direction='in', length=3.5)
        ax.set_xlabel(['Pore node size (m)', 'Pore throat length (m)'][i], fontsize=17, labelpad=9)
        ax.set_ylabel('Volume fraction', fontsize=17)
        ax.text(-0.18, 1.03, f'({chr(97+i)})', transform=ax.transAxes, fontsize=21)
        ax.legend(loc='upper left', frameon=False, fontsize=12)
        pd.DataFrame({'bin_left_m': edges[:-1], 'bin_right_m': edges[1:], 'bin_center_m': centers,
                      'niu_volume_fraction': paper, 'network_volume_fraction': ours,
                      'network_count': counts}).to_csv(out / f'{field}_comparison.csv', index=False)
        checks.append({'field': field, 'elements': len(values), 'network_fraction_sum': float(ours.sum()),
                       'paper_fraction_sum': float(paper.sum())})
    fig.subplots_adjust(left=0.095, right=0.98, bottom=0.19, top=0.90, wspace=0.34)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out / f'niu2020_geometry_overlay.{ext}', dpi=300)
    plt.close(fig)
    metadata = {'paper_source': str(source), 'network_source': str(network),
                'bins': 'Paper centers converted to geometric-midpoint edges; network rebinned by volume. Extra bins, if needed, contain zero paper fraction.',
                'definition': 'Paper pore node size retained as supplied; network uses pore_radius_m, no diameter conversion. Definition equivalence is not independently verified here.',
                'boundary_throats': 'Included', 'alpha': 0.5, 'checks': checks,
                'command': 'python code/scripts/pore_network/plot_niu2020_geometry_overlay.py'}
    (out / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
