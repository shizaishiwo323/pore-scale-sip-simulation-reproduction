"""样品16孔直径与净孔喉长度的体积加权分布，保留分箱源数据。"""
from pathlib import Path
import json
import shutil
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter

ROOT=Path(__file__).resolve().parents[3]
NETWORK=ROOT/'results/sample16_gpu_7freq_20260912_2340/pore_network/network_parsed'
OUT=NETWORK.parent/'volume_weighted_distributions'

def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':16,'mathtext.fontset':'stix',
                         'axes.linewidth':.8,'xtick.direction':'in','ytick.direction':'in'})
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    fig.subplots_adjust(left=.09,right=.98,bottom=.20,top=.92,wspace=.30)
    metadata={'sample':16,'bins':50,'binning':'equal logarithmic width',
              'normalization':'each population normalized by its own total retained volume',
              'pore_size_definition':'diameter = 2 * pore_radius_m; no fitted geometric scaling',
              'throat_definition':'throat_length_m (net throat length), including boundary-connected throats',
              'dpi':600,'populations':{}}
    for ax,(name,file,column,volume_column,factor,xlabel) in zip(axes,[
        ('pore','pores.csv','pore_radius_m','pore_volume_m3',2.,'Pore node diameter (m)'),
        ('throat','throats.csv','throat_length_m','throat_volume_m3',1.,'Pore throat length (m)')]):
        source=NETWORK/file
        data=pd.read_csv(source)
        sizes=data[column].to_numpy()*factor
        weights=data[volume_column].to_numpy()
        keep=np.isfinite(sizes)&(sizes>0)&np.isfinite(weights)&(weights>0)
        assert keep.any()
        sizes,weights=sizes[keep],weights[keep]
        edges=np.geomspace(sizes.min(),sizes.max(),51)
        edges[-1]=np.nextafter(edges[-1],np.inf)
        totals,_=np.histogram(sizes,bins=edges,weights=weights)
        counts,_=np.histogram(sizes,bins=edges)
        fractions=totals/weights.sum()
        assert np.isclose(fractions.sum(),1.,atol=1e-12) and counts.sum()==keep.sum()
        centers=np.sqrt(edges[:-1]*edges[1:])
        # 对数坐标上的等宽窄柱；空隙不影响各箱体积分数。
        half_width=.28*np.log(edges[1:]/edges[:-1])
        left=centers*np.exp(-half_width)
        right=centers*np.exp(half_width)
        ax.bar(left,fractions,width=right-left,align='edge',color='#777777',edgecolor='black',linewidth=.65)
        upper=max(.15,np.ceil((fractions.max()*1.12)/.05)*.05)
        ax.set(xscale='log',xlim=(10**np.floor(np.log10(sizes.min())),10**np.ceil(np.log10(sizes.max()))),
               ylim=(0,upper),xlabel=xlabel,ylabel='Volume fraction')
        ax.yaxis.set_major_locator(MultipleLocator(.05 if upper<=.3 else .1))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:'' if abs(x)<1e-12 else f'{x:g}'))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(which='major',length=7,width=.8)
        ax.tick_params(which='minor',length=3.5,width=.7)
        ax.text(-.18,1.035,'(a)' if name=='pore' else '(b)',transform=ax.transAxes,fontsize=23)
        pd.DataFrame({'bin_left_m':edges[:-1],'bin_right_m':edges[1:],'bin_center_m':centers,
                      'count':counts,'summed_volume_m3':totals,'volume_fraction':fractions}).to_csv(OUT/f'{name}_histogram.csv',index=False)
        data.assign(size_for_histogram_m=data[column]*factor,retained=keep).to_csv(OUT/f'{name}_source_data.csv',index=False)
        metadata['populations'][name]={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'total_rows':len(data),'retained_rows':int(keep.sum()),'excluded_rows':int((~keep).sum()),
            'size_min_m':float(sizes.min()),'size_max_m':float(sizes.max()),'total_volume_m3':float(weights.sum()),
            'fraction_sum':float(fractions.sum()),'y_max':float(upper)}
    fig.savefig(OUT/'sample16_pore_diameter_throat_length_volume_fraction.png',dpi=600)
    plt.close(fig)
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf8')
    (OUT/'README.md').write_text('样品16孔径／孔喉长度体积加权分布\n\n孔径定义为直径，即2×pore_radius_m；孔喉长度为throat_length_m，不使用孔中心间距。横轴单位为米，采用50个对数等宽分箱。每箱纵值为该箱体积之和除以本类保留对象的总体积，两幅图分别归一化为1，并非数量频率分布或概率密度。包含与边界相连的孔喉。仅保留尺寸与体积均有限且为正的记录，排除数量见metadata.json。\n\n图件为600 dpi PNG；histogram.csv保留分箱与体积分数，source_data.csv保留逐对象统计依据。\n',encoding='utf8')
    shutil.copy2(__file__,OUT/Path(__file__).name)
    print(json.dumps(metadata['populations'],ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
