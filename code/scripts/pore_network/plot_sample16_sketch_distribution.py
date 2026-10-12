"""手绘约束的分布级目标预览，不伪装为重新提取的孔喉网络。"""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'results/sample16_gpu_7freq_20260912_2340/pore_network/network_parsed/throats.csv'
OUT=ROOT/'results/sample16_sketch_throat_distribution_spike_diagnostic_20260913'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(SOURCE)
    length=data.throat_length_m.to_numpy()
    volume=data.throat_volume_m3.to_numpy()
    assert np.all(np.isfinite(length)&(length>0)&np.isfinite(volume)&(volume>0))
    edges=np.geomspace(length.min(),np.nextafter(length.max(),np.inf),51)
    centers=np.sqrt(edges[:-1]*edges[1:])
    original=np.histogram(length,bins=edges,weights=volume)[0]/volume.sum()
    # 手绘的大致包络：短端小丘，5-7微米谷，20-30微米宽峰，60微米后快速衰减。
    # 用户确认短端需要尖峰：最近2.3微米的单箱体积分数设为0.05。
    nodes=np.array([.63,.9,1.05,1.4,1.9,2.6,4,6,8,10,13,16,20,26,31,38,47,55,62,75,100,135,160,200])*1e-6
    heights=np.array([0,0,.003,.004,.007,.004,.003,0,.004,.014,.035,.063,.069,.069,.065,.05,.04,.03,.009,.004,.002,.001,0,0])
    raw=np.maximum(PchipInterpolator(np.log10(nodes),heights,extrapolate=False)(np.log10(centers)),0)
    target=raw/raw.sum()
    spike=int(np.argmin(abs(np.log(centers/2.3e-6))))
    target[spike]=0
    target*=.95/target.sum()
    target[spike]=.05
    assert 2e-6<centers[spike]<3e-6 and target[spike]>4*max(target[spike-1],target[spike+1])
    assert np.isclose(original.sum(),1) and np.isclose(target.sum(),1)
    assert 16e-6<centers[target.argmax()]<35e-6
    pd.DataFrame({'bin_left_m':edges[:-1],'bin_right_m':edges[1:],'bin_center_m':centers,'original_volume_fraction':original,'sketch_target_volume_fraction':target}).to_csv(OUT/'distribution_source_data.csv',index=False)
    pd.DataFrame({'length_m':nodes,'sketch_relative_height':heights}).to_csv(OUT/'sketch_control_points.csv',index=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':16,'xtick.direction':'in','ytick.direction':'in'})
    fig,ax=plt.subplots(figsize=(8,5.8))
    fig.subplots_adjust(left=.14,right=.97,bottom=.20,top=.91)
    ax.bar(edges[:-1],original,width=np.diff(edges),align='edge',color='#377eb8',edgecolor='#24577e',alpha=.40,lw=.6,label='Original extracted distribution')
    ax.bar(edges[:-1],target,width=np.diff(edges),align='edge',color='#e55335',edgecolor='#bd4028',alpha=.40,lw=.6,label='Sketch-based target')
    dense=np.geomspace(centers[0],centers[-1],500)
    ax.plot(dense,PchipInterpolator(np.log10(centers),target)(np.log10(dense)),color='#d9472a',lw=2)
    ax.set(xscale='log',xlim=(1e-7,1e-3),ylim=(0,.2),xlabel='Pore throat length (m)',ylabel='Volume fraction')
    ax.set_yticks([0,.05,.1,.15,.2])
    ax.legend(frameon=False,fontsize=12,loc='upper left')
    ax.set_title('Sample 16 | throat-length distribution',fontsize=16)
    fig.text(.14,.055,'Shared 50 logarithmic bins; each distribution sums to 1.\nTarget distribution preview; no network reconstruction or SIP rerun.',fontsize=10)
    fig.savefig(OUT/'original_vs_sketch_target.png',dpi=600)
    plt.close(fig)
    metadata={'source':str(SOURCE),'type':'用户手绘约束的分布级诊断预览','bins':50,'target_peak_length_m':float(centers[target.argmax()]),'target_peak_fraction':float(target.max()),'original_peak_length_m':float(centers[original.argmax()]),'original_peak_fraction':float(original.max()),'normalization':'各自体积分数之和为1；保持原图的50个对数等宽分箱。','spike_center_m':float(centers[spike]),'spike_volume_fraction':float(target[spike]),'interpretation':'按用户确认，在最接近2.3微米的单箱加入0.05体积分数尖峰；其他箱按比例重归一化到0.95。','limitations':'仅调整汇总体积分布，未生成逐孔喉几何或重算体积、电阻和SIP；目标曲线不代表真实CT提取。'}
    (OUT/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')
    (OUT/'README.md').write_text('样品16手绘孔喉分布预览\n\n原始体积分数直接由原始throats.csv计算，采用原图50个log等宽箱。目标分布用手绘控制点进行保形插值，再归一化为1。红色包络为目标各箱体积分数的平滑连接，不是连续概率密度。按用户确认，在最接近2.3微米的单箱加入体积分数0.05的窄尖峰，其余箱等比例归一化到0.95；主峰形状保留。控制点CSV记录基础包络，尖峰规则见脚本与metadata。\n\n本次只建立分布级目标，尚未将目标分配到逐孔喉记录，因此不能直接用于SIP求解。原始提取网络、上一轮短孔喉模型和膜极化参数均未改动。\n',encoding='utf8')
    shutil.copy2(__file__,OUT/Path(__file__).name)
    print(json.dumps(metadata,ensure_ascii=False))

if __name__=='__main__':
    main()
