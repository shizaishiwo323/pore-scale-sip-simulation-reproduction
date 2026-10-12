"""用户指定的孔喉缩短敏感性分析；不修改默认模型和原始网络。"""
from pathlib import Path
import sys, json, shutil, subprocess
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'code/src'))
sys.path.insert(0,str(ROOT/'code/scripts/sip_simulation'))
from compute_polarization_spectra import load_pnextract
from pore_scale_electrical.polarization import PolarizationParameters, membrane_polarization_conductance, upscale_conductance_to_water_conductivity
BASE=ROOT/'results/sample16_gpu_7freq_20260912_2340'
Y=ROOT/'results/sample16_gpu_yz_7freq_20260913/y'
RUN=ROOT/'results/sample16_membrane_short_throats_diagnostic_20260913'
PARAMS=PolarizationParameters(water_conductivity_s_m=.12,dynamic_pore_size_m=2.45913e-6)

def response(f, throats):
    return upscale_conductance_to_water_conductivity(membrane_polarization_conductance(
        np.atleast_1d(f),throats.length_m.to_numpy(),throats.weight.to_numpy(),throats.zdc_ohm.to_numpy(),PARAMS),PARAMS)

def find_peak(throats):
    grid=np.linspace(-10,9,381)
    values=np.array([response(10**x,throats)[0].imag for x in grid])
    i=int(np.argmax(values))
    assert 0<i<len(grid)-1
    opt=minimize_scalar(lambda x:-response(10**x,throats)[0].imag,bounds=(grid[i-1],grid[i+1]),method='bounded')
    return float(10**opt.x)

def prepare():
    for d in ['configs','effective_network','source_data','figures','provenance','simulation_sweeps']:
        (RUN/d).mkdir(parents=True,exist_ok=True)
    network=BASE/'pore_network/network_parsed'
    _,old=load_pnextract(network,PARAMS)
    before=find_peak(old)
    target=float(np.sqrt(1e3*1e4))
    scale=float(np.sqrt(before/target))
    assert 0<scale<1
    raw=pd.read_csv(network/'throats.csv')
    adjusted=raw.copy()
    adjusted['throat_length_m']*=scale
    adjusted['throat_volume_m3']*=scale
    adjusted.to_csv(RUN/'effective_network/throats.csv',index=False)
    shutil.copy2(network/'pores.csv',RUN/'effective_network/pores.csv')
    _,new=load_pnextract(RUN/'effective_network',PARAMS)
    after=find_peak(new)
    assert 1e3<after<1e4 and abs(after/target-1)<1e-4
    assert np.allclose(new.zdc_ohm,old.zdc_ohm*scale)
    assert np.allclose(new.weight/new.weight.sum(),old.weight/old.weight.sum())
    # 同步重算Zdc，不能仅横向平移曲线而隐藏幅值变化。
    check=np.geomspace(.1,2e4,7)
    assert np.allclose(response(check,new),response(check*scale**2,old)/scale,rtol=1e-6,atol=1e-10)
    params={'性质':'用户指定的人工有效孔喉长度敏感性分析；不是原始CT几何或实验拟合结果。',
        '方向':'solver y = CT存储Y；ds2场网格保持不变。','target_peak_hz':target,
        'original_peak_hz':before,'adjusted_peak_hz':after,'length_multiplier':scale,
        '调整规则':'所有孔喉净长度同乘length_multiplier；半径不变；体积同乘该因子，归一化体积权重不变；Zdc由新长度和原面积重新计算。',
        '模型边界':'仅调整膜模型的有效孔喉长度，CT场网格与孔中心坐标未重构，effective_network不能作为真实几何重建网络。',
        '频率':'GPU仍只计算原来7个实验频点；密集曲线由同一膜公式与GPU几何系数计算，用于峰位定位。',
        'source_voxel_size_um':1.92,'old_length_range_um':[float(raw.throat_length_m.min()*1e6),float(raw.throat_length_m.max()*1e6)],
        'new_length_range_um':[float(adjusted.throat_length_m.min()*1e6),float(adjusted.throat_length_m.max()*1e6)],
        'simulation_internal_throats':len(old),'distribution_all_throats':len(raw)}
    (RUN/'configs/parameters_zh.json').write_text(json.dumps(params,indent=2,ensure_ascii=False),encoding='utf8')
    command=json.loads((Y/'configs/membrane_command.json').read_text(encoding='utf-8-sig'))
    for flag,value in [('--network-dir',str(RUN/'effective_network')),('--out-dir',str(RUN)),('--sample-label','Sample 16: shortened effective throats, diagnostic')]:
        command[command.index(flag)+1]=value
    (RUN/'configs/membrane_command.json').write_text(json.dumps(command,indent=2),encoding='utf8')
    print(json.dumps(params,indent=2,ensure_ascii=False),flush=True)
    with (RUN/'provenance/gpu.log').open('w',encoding='utf8') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    for result in (RUN/'simulation_sweeps/membrane').glob('frequency_*/result.json'):
        (result.parent/'config.yml').write_text(json.dumps({'command':command},indent=2),encoding='utf8')

def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    params=json.loads((RUN/'configs/parameters_zh.json').read_text(encoding='utf8'))
    old_gpu=pd.read_csv(Y/'simulation_sweeps/membrane/sweep_results.csv')
    new_gpu=pd.read_csv(RUN/'simulation_sweeps/membrane/sweep_results.csv')
    assert len(new_gpu)==7 and new_gpu.converged_strict.all() and (new_gpu.relative_residual_norm<=1e-6).all()
    assert np.allclose(old_gpu.frequency_hz,new_gpu.frequency_hz)
    f=np.geomspace(.1,20000,401)
    colors=['#377eb8','#ed8a28']
    plt.rcParams.update({'font.family':'Times New Roman','font.size':13,'mathtext.fontset':'stix','xtick.direction':'in','ytick.direction':'in'})
    fig,axes=plt.subplots(2,1,figsize=(8,9))
    fig.subplots_adjust(left=.16,right=.97,bottom=.13,top=.94,hspace=.28)
    fig.suptitle('Sample 16 | y direction | effective-length sensitivity',fontsize=13)
    source=[]
    factor_checks={}
    for label,network,gpu,color in [('Original',BASE/'pore_network/network_parsed',old_gpu,colors[0]),('Shortened',RUN/'effective_network',new_gpu,colors[1])]:
        _,throats=load_pnextract(network,PARAMS)
        water=gpu.water_sigma_real_s_m.to_numpy()+1j*gpu.water_sigma_imag_s_m.to_numpy()
        effective=gpu.effective_sigma_real_s_m.to_numpy()+1j*gpu.effective_sigma_imag_s_m.to_numpy()
        factor=float(np.median((effective/water).real))
        dense=factor*(PARAMS.water_conductivity_s_m+response(f,throats))
        discrete=factor*(PARAMS.water_conductivity_s_m+response(gpu.frequency_hz.to_numpy(),throats))
        error=float(np.max(np.abs(discrete-effective)/np.maximum(np.abs(effective),1e-30)))
        assert error<.01
        factor_checks[label]={'gpu_geometry_factor':factor,'maximum_relative_difference_vs_gpu':error}
        source.append(pd.DataFrame({'case':label,'frequency_hz':f,'sigma_real_s_m':dense.real,'sigma_imag_s_m':dense.imag}))
        for ax,part in zip(axes,['real','imag']):
            ax.plot(f,getattr(dense,part),color=color,lw=2,label=label+' membrane model')
            ax.plot(gpu.frequency_hz,gpu[f'effective_sigma_{part}_s_m'],'o',ms=5,mfc='white',mec=color,label=label+' GPU (7 points)')
    exp=pd.read_csv(Y/'source_data/experiment_all.csv')
    for i,(ax,part) in enumerate(zip(axes,['real','imag'])):
        ey=exp[f'experiment_sigma_{part}_s_m']; keep=ey>0
        ax.plot(exp.frequency_hz[keep],ey[keep],'^',mfc='white',mec='black',ms=4,label='Experiment')
        ax.axvspan(1e3,1e4,color='#777777',alpha=.10,zorder=0)
        ax.set(xscale='log',yscale='log',xlim=(.08,25000),xlabel='Frequency (Hz)',ylabel=f"Conductivity {part} (S/m)")
        ax.text(-.12,1.02,f'({chr(97+i)})',transform=ax.transAxes,fontsize=19)
    axes[0].legend(fontsize=9,loc='best',frameon=False,ncol=2)
    axes[1].axvline(params['adjusted_peak_hz'],color=colors[1],ls=':',lw=1)
    axes[1].annotate(f"Peak: {params['adjusted_peak_hz']:.0f} Hz",xy=(params['adjusted_peak_hz'],.84),xycoords=('data','axes fraction'),ha='center',va='top',fontsize=11)
    fig.text(.16,.035,f"Effective throat lengths multiplied by {params['length_multiplier']:.5g}; Zdc recomputed.\nDiagnostic geometry adjustment; unchanged CT field grid. Lines: model; circles: GPU.",fontsize=10)
    fig.savefig(RUN/'figures/membrane_before_after.png',dpi=600); plt.close(fig)
    pd.concat(source).to_csv(RUN/'source_data/membrane_dense_curves.csv',index=False)
    pd.concat([old_gpu.assign(case='Original'),new_gpu.assign(case='Shortened')]).to_csv(RUN/'source_data/membrane_gpu_comparison.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5))
    fig.subplots_adjust(left=.12,right=.97,bottom=.23,top=.92)
    raw=pd.read_csv(BASE/'pore_network/network_parsed/throats.csv')
    new=pd.read_csv(RUN/'effective_network/throats.csv')
    edges=np.geomspace(min(new.throat_length_m.min(),raw.throat_length_m.min())*.999,max(new.throat_length_m.max(),raw.throat_length_m.max())*1.001,71)
    bins=[]
    for label,data,color in [('Original',raw,colors[0]),('Shortened',new,colors[1])]:
        fractions,_=np.histogram(data.throat_length_m,bins=edges,weights=data.throat_volume_m3/data.throat_volume_m3.sum())
        assert np.isclose(fractions.sum(),1)
        ax.bar(edges[:-1],fractions,width=np.diff(edges),align='edge',color=color,edgecolor=color,alpha=.45,label=label,lw=.6)
        bins.append(pd.DataFrame({'case':label,'bin_left_m':edges[:-1],'bin_right_m':edges[1:],'volume_fraction':fractions}))
    ax.set(xscale='log',xlabel='Pore throat length (m)',ylabel='Volume fraction',title='Sample 16 | original vs shortened effective throats')
    ax.legend(frameon=False)
    fig.text(.12,.05,'Shared logarithmic bins; separate volume normalization; all extracted throats included.\nLength adjustment is a model sensitivity case, not a new CT reconstruction.',fontsize=10)
    fig.savefig(RUN/'figures/throat_length_before_after.png',dpi=600); plt.close(fig)
    pd.concat(bins).to_csv(RUN/'source_data/throat_length_histogram.csv',index=False)
    raw[['throat_id','throat_length_m','throat_volume_m3']].rename(columns={'throat_length_m':'original_length_m','throat_volume_m3':'original_volume_m3'}).assign(shortened_length_m=new.throat_length_m,shortened_volume_m3=new.throat_volume_m3).to_csv(RUN/'source_data/throat_length_before_after.csv',index=False)
    (RUN/'provenance/verification.json').write_text(json.dumps({'gpu_points':7,'max_true_residual':float(new_gpu.relative_residual_norm.max()),'peak_hz':params['adjusted_peak_hz'],'curve_checks':factor_checks},indent=2),encoding='utf8')
    text=f"# 样品16：有效孔喉缩短诊断\n\n原始膜极化虚部峰值为{params['original_peak_hz']:.6g} Hz；调整后为{params['adjusted_peak_hz']:.6g} Hz。长度统一乘{params['length_multiplier']:.8g}，Zdc由新几何重新计算，峰高随之改变。目标只是峰位落入1e3–1e4 Hz，不是同时拟合实验幅值。\n\n调整后长度范围为{params['new_length_range_um']} µm；原始CT体素为1.92 µm。低于体素的长度是有效模型参数，不能视为CT分辨出的实际孔喉。CT场网格、孔径和连通性未改变。本包人工网络不得用作真实几何重建或默认模型。\n\n峰位由密集膜模型求极值定位；GPU依然只计算原7个实验频点。由于水相均匀且固相严格绝缘，同一y向场的几何系数与频率无关，密集曲线使用GPU几何系数乘以局部复电导；对7点的一致性检查见provenance/verification.json。实验数据仅作对照，未参与幅值拟合。\n\n膜模型沿用默认内部孔喉筛选；长度分布图包含全部孔喉，与此前分布图口径一致。体积随长度同乘缩放因子，归一化权重不变。原始网络、原结果均未修改。\n"
    (RUN/'README.md').write_text(text,encoding='utf8')
    shutil.copy2(__file__,RUN/'configs'/Path(__file__).name)

if __name__=='__main__':
    if '--plot-only' not in sys.argv: prepare()
    plot()
