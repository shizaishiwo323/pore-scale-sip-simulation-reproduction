"""Plot the published binned geometry data without treating bins as raw samples."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
from compare_niu2020_pnextract_geometry import centers_to_log_edges

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network-dir', type=Path)
    args = parser.parse_args()
    source = ROOT / 'data/Niu 2020data/Figure5.xlsx'
    out = ROOT / 'results/niu2020_geometry_histograms_20260912'
    if args.network_dir:
        source = args.network_dir.resolve()
        out = source.parent / 'geometry_histograms'
    out.mkdir(parents=True, exist_ok=True)
    raw = None if args.network_dir else pd.read_excel(source)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 14,
                         'axes.linewidth': 0.8, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    metadata = {'input': str(source.relative_to(ROOT)), 'sheet': 'Sheet1',
                'statistic': 'Original binned volume fraction; not number frequency.',
                'size_definition': 'Pore node size retained as labeled; no radius/diameter conversion.',
                'bin_edges': 'Geometric midpoints inferred from supplied centers for bar placement only.',
                'normalization': 'None; original fractions retained.',
                'panels': []}
    if args.network_dir:
        metadata.update(sheet=None, statistic='Volume-weighted histogram of individual network elements.',
                        size_definition='Pore radius from pore_radius_m; throat length from throat_length_m.',
                        bin_edges='49 equal log-width bins spanning each positive data range.',
                        normalization='Each bin volume divided by total volume of its element type; boundary throats included.')
    for i, (ax, name, label) in enumerate(zip(axes, ['pore_node_size', 'pore_throat_length'],
                                             ['Pore node size (m)', 'Pore throat length (m)'])):
        if args.network_dir:
            frame = pd.read_csv(source / ['pores.csv', 'throats.csv'][i])
            field = ['pore_radius_m', 'throat_length_m'][i]
            volume_field = ['pore_volume_m3', 'throat_volume_m3'][i]
            values, weights = frame[[field, volume_field]].to_numpy(float).T
            assert np.isfinite(values).all() and np.isfinite(weights).all()
            assert (values > 0).all() and (weights >= 0).all() and weights.sum() > 0
            edges = np.geomspace(values.min(), values.max(), 50)
            counts, _ = np.histogram(values, bins=edges)
            volumes, _ = np.histogram(values, bins=edges, weights=weights)
            assert counts.sum() == len(values) and np.isclose(volumes.sum()/weights.sum(), 1)
            x, y = np.sqrt(edges[:-1]*edges[1:]), volumes/weights.sum()
            data = np.column_stack([x, y])
            if i == 0:
                label = 'Pore node radius (m)'
            frame.to_csv(out / f'{name}_elements.csv', index=False)
        else:
            data = raw.iloc[:, 2*i:2*i+2].dropna().to_numpy(float)
            x, y = data.T
        assert np.isfinite(data).all() and (x > 0).all() and (y >= 0).all()
        assert (np.diff(x) > 0).all() and np.isclose(y.sum(), 1)
        if not args.network_dir:
            edges = centers_to_log_edges(x)
        # 细柱宽度为对数分箱宽度的 42%，保留参考图的间隔；柱高不变。
        ax.bar(x, y, width=np.diff(edges)*0.42, color='#777777', edgecolor='#222222', linewidth=0.7)
        ax.set_xscale('log')
        ax.set_xlim(10**np.floor(np.log10(edges[0])), 10**np.ceil(np.log10(edges[-1])))
        top = max(0.15, np.ceil(y.max()*1.1/0.05)*0.05)
        ax.set_ylim(0, top)
        ax.set_yticks(np.arange(0, top+0.025, 0.05))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(which='major', direction='in', length=7, width=0.8)
        ax.tick_params(which='minor', direction='in', length=3.5, width=0.7)
        ax.set_xlabel(label, fontsize=17, labelpad=9)
        ax.set_ylabel('Volume fraction', fontsize=17)
        ax.text(-0.18, 1.03, f'({chr(97+i)})', transform=ax.transAxes, fontsize=21)
        table = pd.DataFrame({'bin_center_m': x, 'bin_center_um': x*1e6,
                              'bin_left_m': edges[:-1], 'bin_right_m': edges[1:],
                              'volume_fraction': y})
        if args.network_dir:
            table['count'] = counts
            table['number_fraction'] = counts/len(values)
        table.to_csv(out / f'{name}_source_data.csv', index=False)
        metadata['panels'].append({'quantity': name, 'bins': len(x), 'fraction_sum': float(y.sum()),
                                   'peak_bin_m': float(x[y.argmax()]), 'peak_fraction': float(y.max())})
    fig.subplots_adjust(left=0.095, right=0.98, bottom=0.19, top=0.90, wspace=0.34)
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out / f'niu2020_geometry_histograms.{ext}', dpi=300)
    plt.close(fig)
    (out / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metadata['panels'], indent=2))
    print(out)


if __name__ == '__main__':
    main()
