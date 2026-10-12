"""Berea geometry comparisons with explicit statistical and geometric definitions."""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
XML=Path(r'C:\Users\13404\Desktop\niu 2020 avizo data.xml')
NET=ROOT/'results/niu2020_berea_mechanistic_membrane_gpu7_20260914/pore_network/network_parsed'
PAPER=ROOT/'data/Niu 2020data/Figure5.xlsx'
OUT=ROOT/'results/niu2020_avizo_pnextract_geometry_20260916'
COLORS=['#4477AA','#EE9944','#228877']


def read_xml(path):
    text=path.read_bytes().decode('gbk')
    root=ET.fromstring(text[text.index('<Workbook'):])
    ns={'s':'urn:schemas-microsoft-com:office:spreadsheet'}
    tables={}
    for sheet in root.findall('s:Worksheet',ns):
        rows=[]
        for row in sheet.findall('s:Table/s:Row',ns):
            values=[]
            for cell in row.findall('s:Cell',ns):
                index=int(cell.get('{'+ns['s']+'}Index',len(values)+1))
                values.extend([None]*(index-1-len(values)))
                value=cell.find('s:Data',ns)
                values.append(None if value is None else value.text)
            rows.append(values)
        tables[sheet.get('{'+ns['s']+'}Name')]=pd.DataFrame(rows[1:],columns=rows[0]).apply(pd.to_numeric,errors='raise')
    return tables['Pores'],tables['Throats']


def bins(values,n=36):
    data=np.concatenate([np.asarray(x) for x in values])
    assert np.all(np.isfinite(data)&(data>0))
    return np.geomspace(data.min()*.999,data.max()*1.001,n+1)


def fraction(values,edges,weights=None):
    count=np.histogram(values,edges,weights=weights)[0]
    total=len(values) if weights is None else np.sum(weights)
    assert np.isclose(count.sum(),total)
    return count/total*100


def bars(ax,edges,y,color,label):
    ax.bar(edges[:-1],y,width=np.diff(edges),align='edge',color=color,
           edgecolor=color,alpha=.40,linewidth=.65,label=label)
    ax.stairs(y,edges,color=color,lw=1.1)


def style(ax,xlabel,ylabel):
    ax.set(xscale='log',xlabel=xlabel,ylabel=ylabel,ylim=(0,None))
    ax.grid(axis='y',color='#dddddd',lw=.5)
    ax.set_axisbelow(True)
    ax.tick_params(which='both',direction='in',top=True,right=True)


def main():
    for folder in ['figures','source_data','provenance','configs']:
        (OUT/folder).mkdir(parents=True,exist_ok=True)
    ap,at=read_xml(XML)
    pp=pd.read_csv(NET/'pores.csv'); pt_all=pd.read_csv(NET/'throats.csv')
    pt=pt_all.loc[(pt_all.pore1_id>=1)&(pt_all.pore2_id>=1)].copy()
    paper=pd.read_excel(PAPER)
    points=ap.set_index('Pore ID')
    xyz=['X Coord','Y Coord','Z Coord']
    distances=np.linalg.norm(points.loc[at['Pore ID #1'],xyz].to_numpy()-points.loc[at['Pore ID #2'],xyz].to_numpy(),axis=1)
    np.testing.assert_allclose(distances,at.ChannelLength,rtol=2e-7)
    np.testing.assert_allclose(ap.EqRadius,(3*ap.Volume/(4*np.pi))**(1/3),rtol=2e-6)
    volume_error=abs(ap.Volume.sum()/(9924264*2.8**3)-1)
    assert volume_error<1e-5
    for name,frame in [('avizo_pores',ap),('avizo_throats',at)]:
        frame.to_csv(OUT/'source_data'/f'{name}.csv',index=False)
    plt.rcParams.update({'font.family':'Arial','font.size':9,'legend.frameon':False,'axes.linewidth':.8})
    # Main overview preserves each source's actual statistics. Paper has no count data.
    fig,axes=plt.subplots(3,2,figsize=(10,8.3))
    source=[]
    for col in range(2):
        x,y=paper.iloc[:,2*col:2*col+2].dropna().to_numpy(float).T
        assert np.isclose(y.sum(),1,rtol=1e-5)
        # Paper r is the pore radius used in Niu eq18; display 2r to match diameter units.
        x=x*1e6*(2 if col==0 else 1)
        edges=np.r_[x[0]**2/np.sqrt(x[0]*x[1]),np.sqrt(x[:-1]*x[1:]),x[-1]**2/np.sqrt(x[-2]*x[-1])]
        bars(axes[0,col],edges,y*100,COLORS[0],'Niu 2020 (published)')
        xlabel=['Pore diameter, 2r (μm)','Pore throat length (μm)'][col]
        style(axes[0,col],xlabel,'Volume fraction (%)')
        for lo,hi,v in zip(edges[:-1],edges[1:],y*100):
            source.append(dict(source='Niu2020',quantity=xlabel,statistic='volume_percent',left_um=lo,right_um=hi,percent=v))
    original=[[ap.EqRadius.to_numpy()*2,at.ChannelLength.to_numpy()],
              [pp.pore_radius_m.to_numpy()*2e6,pt.throat_length_m.to_numpy()*1e6]]
    labels=[['Equivalent pore diameter (μm)','Channel length = center distance (μm)'],
            ['Inscribed pore diameter (μm)','Net throat length (μm)']]
    for col in range(2):
        edges=bins([original[0][col],original[1][col]])
        for row in (1,2):
            values=original[row-1][col]; y=fraction(values,edges)
            name=['Avizo','pnextract'][row-1]
            bars(axes[row,col],edges,y,COLORS[row],f'{name} (n={len(values):,})')
            style(axes[row,col],labels[row-1][col],'Count frequency (%)')
            axes[row,col].set_xlim(edges[0],edges[-1])
            for lo,hi,v in zip(edges[:-1],edges[1:],y):
                source.append(dict(source=name,quantity=labels[row-1][col],statistic='count_percent',left_um=lo,right_um=hi,percent=v))
        top=max(axes[1,col].get_ylim()[1],axes[2,col].get_ylim()[1])
        for row in (1,2): axes[row,col].set_ylim(0,top)
    for i,ax in enumerate(axes.flat):
        ax.legend(fontsize=8,loc='upper right')
        ax.text(-.12,1.04,chr(97+i),transform=ax.transAxes,fontweight='bold',fontsize=12)
    fig.suptitle('Berea geometry: Niu 2020, Avizo and pnextract',fontsize=13,y=.987)
    fig.text(.5,.018,'Published data are volume-weighted; extracted data are count-weighted. Geometric definitions differ (see axis labels).',ha='center',fontsize=8)
    fig.subplots_adjust(left=.09,right=.98,bottom=.10,top=.93,wspace=.27,hspace=.60)
    fig.savefig(OUT/'figures/three_source_original_distributions.png',dpi=600)
    plt.close(fig)
    pd.DataFrame(source).to_csv(OUT/'source_data/three_source_histograms.csv',index=False)
    # Matched definitions: equivalent-volume diameter and center-to-center distance.
    matched=[[ap.EqRadius.to_numpy()*2,at.ChannelLength.to_numpy()],
             [2*(3*pp.pore_volume_m3.to_numpy()/(4*np.pi))**(1/3)*1e6,
              pt.pore_center_to_center_length_m.to_numpy()*1e6]]
    fig,axes=plt.subplots(1,2,figsize=(10,4.2))
    matched_source=[]; summaries=[]
    for col,ax in enumerate(axes):
        edges=bins([matched[0][col],matched[1][col]])
        for j,name in enumerate(['Avizo','pnextract']):
            values=matched[j][col]; y=fraction(values,edges)
            bars(ax,edges,y,COLORS[j+1],f'{name} (n={len(values):,})')
            for lo,hi,v in zip(edges[:-1],edges[1:],y):
                matched_source.append(dict(source=name,quantity=['equivalent_diameter','center_distance'][col],left_um=lo,right_um=hi,count_percent=v))
            summaries.append(dict(source=name,quantity=['equivalent_diameter','center_distance'][col],n=len(values),median_um=np.median(values),mean_um=np.mean(values)))
        style(ax,['Equivalent-volume pore diameter (μm)','Pore center-to-center distance (μm)'][col],'Count frequency (%)')
        ax.set_xlim(edges[0],edges[-1]);ax.legend(fontsize=8)
        ax.text(-.12,1.04,chr(97+col),transform=ax.transAxes,fontweight='bold',fontsize=12)
    fig.suptitle('Avizo vs pnextract: matched definitions and common logarithmic bins',fontsize=12)
    fig.text(.5,.015,'Internal connections only; Niu 2020 count frequencies are unavailable in the published workbook.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.08,right=.98,bottom=.19,top=.86,wspace=.28)
    fig.savefig(OUT/'figures/avizo_pnextract_count_overlay.png',dpi=600)
    plt.close(fig)
    pd.DataFrame(matched_source).to_csv(OUT/'source_data/matched_count_histograms.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'source_data/summary.csv',index=False)
    inputs={}
    for path in [XML,PAPER,NET/'pores.csv',NET/'throats.csv']:
        with path.open('rb') as f: inputs[str(path)]=hashlib.file_digest(f,'sha256').hexdigest()
    metadata={'inputs_sha256':inputs,'avizo_pores':len(ap),'avizo_throats':len(at),
        'pnextract_pores':len(pp),'pnextract_internal_throats':len(pt),'excluded_boundary_throats':len(pt_all)-len(pt),
        'unit_inference':'XML has no explicit units. Coordinates span ~977.2, min pore volume=2.8^3 and total pore volume matches CT within 1e-5: interpret length as um, volume as um^3.',
        'avizo_volume_relative_error_vs_CT':volume_error,
        'avizo_channel_length_equals_center_distance_rtol':2e-7,
        'paper_definition':'Figure5 workbook has volume fractions. Niu Section4.2 eq18 identifies r as pore radius and uses Figure4a distribution; diameter displayed as 2r.',
        'count_conversion':'No conversion of paper volume fractions to counts: individual volumes/counts unavailable.',
        'avizo_throat_volume':'Not exported. No fabricated area*length volume weighting.',
        'bins':'36 common logarithmic bins for extracted data; original 49 bins for published volume fractions.',
        'contract':'Quantitative grid; compare shape with explicit weighting/geometry; Python matplotlib; PNG600dpi. No significance tests for deterministic geometry distributions.'}
    (OUT/'provenance/metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf8')
    (OUT/'configs'/Path(__file__).name).write_text(Path(__file__).read_text(encoding='utf8'),encoding='utf8')
    (OUT/'README.md').write_text('# Berea三来源几何分布\n\n'
        '三来源总览：蓝色Niu论文、橙色Avizo、绿色pnextract。论文为体积分数，其余为数量频率；不能直接比较柱高。'
        '论文孔径按Section4.2式18的半径r转换为2r；Avizo使用等体积直径，pnextract原始使用内切直径。\n\n'
        '公平口径叠加图：两种提取均采用等体积孔径和孔中心距离；36个统一对数分箱，百分比各自归一为100%。'
        'Avizo ChannelLength与坐标计算的中心距离相符；不可当成pnextract净喉长。'
        'pnextract仅保留3666条内部连接，排除190条外部边界连接。\n\n'
        'XML未明示单位，依据坐标范围、最小体积=2.8³以及体积总和与原CT孔隙体积的一致性按μm解释。'
        'Avizo无喉体积、论文无原始数量，因此没有伪造三来源统一数量频率或喉体积分数。'
        '865/2119个孔和1773/3666条内部连接也反映两种提取划分的差异，分布不是逐元素配对验证。\n',encoding='utf8')
    print(pd.DataFrame(summaries).to_string(index=False))


def volume_overlay():
    from compare_niu2020_pnextract_geometry import centers_to_log_edges
    ap,at=read_xml(XML)
    pp=pd.read_csv(NET/'pores.csv'); pt=pd.read_csv(NET/'throats.csv')
    pt=pt.loc[(pt.pore1_id>=1)&(pt.pore2_id>=1)]
    paper=pd.read_excel(PAPER)
    plt.rcParams.update({'font.family':'Arial','font.size':10,'legend.frameon':False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    checks=[]
    for col,ax in enumerate(axes):
        x,y=paper.iloc[:,2*col:2*col+2].dropna().to_numpy(float).T
        x=x*1e6*(2 if col==0 else 1)
        edges=centers_to_log_edges(x)
        if col==0:
            values=[ap.EqRadius.to_numpy()*2,pp.pore_radius_m.to_numpy()*2e6]
            weights=[ap.Volume.to_numpy(),pp.pore_volume_m3.to_numpy()]
        else:
            values=[at.ChannelLength.to_numpy(),pt.throat_length_m.to_numpy()*1e6]
            # Estimated channel volume only; Avizo XML contains no throat volume.
            weights=[(at.Area*at.ChannelLength).to_numpy(),pt.throat_volume_m3.to_numpy()]
        ratio=edges[1]/edges[0]
        while min(v.min() for v in values)<edges[0]:
            edges=np.r_[edges[0]/ratio,edges];y=np.r_[0.,y]
        while max(v.max() for v in values)>edges[-1]:
            edges=np.r_[edges,edges[-1]*ratio];y=np.r_[y,0.]
        series=[y*100]
        for v,w in zip(values,weights):
            assert np.all(np.isfinite(v)&(v>0)) and np.all(np.isfinite(w)&(w>=0)) and w.sum()>0
            series.append(fraction(v,edges,w))
        names=['Niu 2020','Avizo' if col==0 else 'Avizo (estimated channel volume)','pnextract']
        for data,color,name in zip(series,COLORS,names):
            assert np.isclose(data.sum(),100,rtol=1e-5)
            bars(ax,edges,data,color,name)
        style(ax,['Pore diameter (μm)','Throat / channel length (μm)'][col],'Volume fraction (%)')
        ax.set_xlim(edges[0],edges[-1])
        ax.set_title(['Pore size distribution','Throat / channel length distribution'][col],pad=12)
        ax.legend(loc='upper left',fontsize=8)
        ax.text(-.12,1.06,chr(97+col),transform=ax.transAxes,fontweight='bold',fontsize=13)
        pd.DataFrame({'left_um':edges[:-1],'right_um':edges[1:],'niu_volume_percent':series[0],
                      'avizo_volume_percent':series[1],'pnextract_volume_percent':series[2]}).to_csv(
                      OUT/'source_data'/f'volume_overlay_{col}.csv',index=False)
        checks.append({'panel':col,'sums_percent':[float(a.sum()) for a in series]})
    fig.subplots_adjust(left=.075,right=.985,bottom=.25,top=.87,wspace=.27)
    fig.text(.5,.10,'Pore diameter: Niu 2r; Avizo equivalent-volume diameter; pnextract inscribed diameter.',ha='center',fontsize=8)
    fig.text(.5,.055,'Avizo channel volume ≈ area × center distance; Niu / pnextract use throat length. Each distribution sums to 100%.',ha='center',fontsize=8)
    target=OUT/'figures/three_source_volume_fraction_overlay.png'
    fig.savefig(target,dpi=600)
    plt.close(fig)
    (OUT/'provenance/volume_overlay.json').write_text(json.dumps({'checks':checks,
        'avizo_throat_weight':'Estimated area * ChannelLength, not measured throat volume; may include pore-body portions.',
        'bins':'Original paper geometric-midpoint bin boundaries, extended outward without changing paper mass.',
        'scope':'Three source volume fraction overlay; geometric definitions remain different.',
        'pnextract_boundary_connections':'Excluded: 190; internal throats: 3666',
        'dpi':600},indent=2),encoding='utf8')
    (OUT/'configs'/Path(__file__).name).write_text(Path(__file__).read_text(encoding='utf8'),encoding='utf8')
    print(target)


if __name__=='__main__':
    import sys
    if '--volume-overlay' in sys.argv:
        volume_overlay()
    else:
        main()
