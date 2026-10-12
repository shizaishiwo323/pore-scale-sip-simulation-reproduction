"""使用手绘分箱总体，eta固定1%，仅拟合公共有效截面积。"""
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import sample16_short_throat_diagnostic as prior

OUT=prior.ROOT/'results/sample16_sketch_area_fit_diagnostic_20260913'

def main():
    for name in ['configs','source_data','figures','provenance','effective_distribution']:
        (OUT/name).mkdir(parents=True,exist_ok=True)
    target_dir=prior.ROOT/'results/sample16_sketch_throat_distribution_spike_diagnostic_20260913'
    bins=pd.read_csv(target_dir/'distribution_source_data.csv')
    _,original_t=prior.load_pnextract(prior.BASE/'pore_network/network_parsed',prior.PARAMS)
    area_reference=float(np.average(original_t.membrane_active_area_m2,weights=original_t.weight))
    t=pd.DataFrame({'length_m':bins.bin_center_m,'weight':bins.sketch_target_volume_fraction})
    assert np.isclose(t.weight.sum(),1) and np.isclose(t.weight.max(),.07413943188809811)
    shutil.copy2(target_dir/'original_vs_sketch_target.png',OUT/'figures/throat_distribution.png')
    shutil.copy2(target_dir/'distribution_source_data.csv',OUT/'source_data/target_distribution.csv')
    old=pd.read_csv(prior.RUN/'simulation_sweeps/membrane/sweep_results.csv')
    water=old.water_sigma_real_s_m.to_numpy()+1j*old.water_sigma_imag_s_m.to_numpy()
    eff=old.effective_sigma_real_s_m.to_numpy()+1j*old.effective_sigma_imag_s_m.to_numpy()
    factor=float(np.median((eff/water).real))
    exp=pd.read_csv(prior.Y/'source_data/experiment_all.csv')
    f=exp.frequency_hz.to_numpy()
    obs=exp.experiment_sigma_imag_s_m.to_numpy()
    mask=obs>0
    assert mask.sum()>=3
    def model(freq,area):
        p=prior.PARAMS
        zdc=t.length_m.to_numpy()/(p.water_conductivity_s_m*float(area))
        c=prior.membrane_polarization_conductance(np.atleast_1d(freq),t.length_m.to_numpy(),t.weight.to_numpy(),zdc,p)
        return factor*(p.water_conductivity_s_m+prior.upscale_conductance_to_water_conductivity(c,p))
    # 固定eta时响应虚部与截面积严格成正比，对数最小二乘有解析解。
    area_fit=area_reference*float(10**np.mean(np.log10(obs[mask]/model(f[mask],area_reference).imag)))
    assert np.isfinite(area_fit) and area_fit>0
    scan=np.linspace(-5,7,481)
    j=int(np.argmax(model(10**scan,area_fit).imag))
    assert 0<j<len(scan)-1
    peak=minimize_scalar(lambda x:-model(10**x,area_fit)[0].imag,bounds=(scan[j-1],scan[j+1]),method='bounded')
    peak_hz=float(10**peak.x)
    metrics={}
    for name,value in [('before',area_reference),('fitted',area_fit)]:
        pred=model(f,value)
        metrics[name]={}
        for region,keep in [('target_band',(f>=1e3)&(f<=1e4)&(obs>0)),('all_positive',obs>0)]:
            metrics[name][region]={'n':int(keep.sum()),'imag_log10_rmse':float(np.sqrt(np.mean(np.log10(pred.imag[keep]/obs[keep])**2))), 'imag_median_relative_error':float(np.median(np.abs(pred.imag[keep]/obs[keep]-1)))}
    config={'性质':'手绘分布的统计膜模型；不是重建CT网络。', 'membrane_polarizability':.01,
        'area_reference_m2':area_reference,'fitted_area_m2':area_fit,'equivalent_circular_radius_um':float(np.sqrt(area_fit/np.pi)*1e6),'peak_hz':peak_hz,
        '拟合定义':'全部48个正实验虚部的log10等权最小二乘；eta固定1%，不拟合实部、不加背景。',
        '几何表示':'50个分箱中心作为代表长度；红色体积分数作为膜模型权重，含短端5%尖峰；所有分箱共用拟合有效截面积。不是逐孔喉网络。',
        '参考面积':'原始内部孔喉截面积的体积加权平均值；仅用于同一红色分布下幅值调整前后的对比，不冒充原始网络完整响应。',
        'geometry_factor':factor,'metrics':metrics,
        '边界':'有效面积是条件拟合参数，没有独立物性验证；实验包括全部机制，膜单机制不能等同于总实验。场保持y向ds2。面积只能改幅值，不能改变峰位。'}
    (OUT/'configs/parameters_zh.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(config,ensure_ascii=False,indent=2),flush=True)
    cmd=json.loads((prior.RUN/'configs/membrane_command.json').read_text())
    cmd[cmd.index('--out-dir')+1]=str(OUT)
    cmd[cmd.index('--sample-label')+1]='Sample 16: conditional membrane-strength fit'
    ensemble=pd.DataFrame({'throat_length_m':t.length_m,'throat_volume_m3':t.weight*float(pd.read_csv(prior.BASE/'pore_network/network_parsed/throats.csv').throat_volume_m3.sum()),'membrane_active_area_m2':area_fit})
    ensemble.to_csv(OUT/'effective_distribution/throats.csv',index=False)
    shutil.copy2(prior.BASE/'pore_network/network_parsed/pores.csv',OUT/'effective_distribution/pores.csv')
    cmd[cmd.index('--network-dir')+1]=str(OUT/'effective_distribution')
    cmd+=['--membrane-polarizability','0.01']
    (OUT/'configs/membrane_command.json').write_text(json.dumps(cmd,indent=2),encoding='utf8')
    with (OUT/'provenance/gpu.log').open('w',encoding='utf8') as log:
        subprocess.run(cmd,cwd=prior.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    gpu=pd.read_csv(OUT/'simulation_sweeps/membrane/sweep_results.csv')
    assert len(gpu)==7 and gpu.converged_strict.all() and (gpu.relative_residual_norm<=1e-6).all()
    actual=gpu.effective_sigma_real_s_m.to_numpy()+1j*gpu.effective_sigma_imag_s_m.to_numpy()
    check=float(np.max(np.abs(model(gpu.frequency_hz.to_numpy(),area_fit)-actual)/np.abs(actual)))
    assert check<1e-5
    (OUT/'provenance/verification.json').write_text(json.dumps({'gpu_points':7,'max_relative_residual':float(gpu.relative_residual_norm.max()),'model_gpu_relative_difference':check,'peak_hz':peak_hz},indent=2),encoding='utf8')
    for result in (OUT/'simulation_sweeps/membrane').glob('frequency_*/result.json'):
        (result.parent/'config.yml').write_text(json.dumps({'command':cmd},indent=2),encoding='utf8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Times New Roman','font.size':13,'xtick.direction':'in','ytick.direction':'in'})
    fig,axes=plt.subplots(2,1,figsize=(8,9))
    fig.subplots_adjust(left=.16,right=.97,bottom=.12,top=.94,hspace=.29)
    fig.suptitle('Sample 16 | y direction | membrane polarizability = 1%',fontsize=14)
    dense=np.geomspace(.1,20000,401)
    rows=[]
    for label,value,color,points in [('Reference area',area_reference,'#ed8a28',None),('Fitted area',area_fit,'#377eb8',gpu)]:
        curve=model(dense,value)
        rows.append(pd.DataFrame({'case':label,'frequency_hz':dense,'sigma_real_s_m':curve.real,'sigma_imag_s_m':curve.imag}))
        for ax,part in zip(axes,['real','imag']):
            ax.plot(dense,getattr(curve,part),color=color,lw=2,label=f'{label}: {value*1e12:.3g} um2')
            if points is not None: ax.plot(points.frequency_hz,points[f'effective_sigma_{part}_s_m'],'o',ms=5,mfc='white',mec=color)
    for i,(ax,part) in enumerate(zip(axes,['real','imag'])):
        values=exp[f'experiment_sigma_{part}_s_m']; keep=values>0
        ax.plot(f[keep],values[keep],'^',mfc='white',mec='black',ms=4,label='Experiment')
        ax.axvspan(1e3,1e4,color='grey',alpha=.10)
        ax.set(xscale='log',yscale='log',xlim=(.08,25000),xlabel='Frequency (Hz)',ylabel=f'Conductivity {part} (S/m)')
        ax.text(-.12,1.02,f'({chr(97+i)})',transform=ax.transAxes,fontsize=19)
    axes[0].legend(frameon=False,fontsize=11)
    fig.text(.16,.035,f'Sketch-based length distribution; area fitted to all positive imaginary data.\nCircles: 7 GPU points; lines: model. Calibration, not independent validation.',fontsize=10)
    fig.savefig(OUT/'figures/membrane_area_comparison.png',dpi=600)
    plt.close(fig)
    pd.concat(rows).to_csv(OUT/'source_data/dense_curves.csv',index=False)
    comparison=exp.copy()
    comparison['used_for_fit']=mask
    for label,value in [('before',area_reference),('fitted',area_fit)]:
        v=model(f,value)
        comparison[label+'_real_s_m']=v.real
        comparison[label+'_imag_s_m']=v.imag
    comparison.to_csv(OUT/'source_data/experimental_frequency_comparison.csv',index=False)
    (OUT/'README.md').write_text(f"# 手绘分布膜极化面积标定\n\neta固定1%。红色目标分布含2.39微米处5%尖峰，以50箱中心长度和体积分数直接进入膜模型。有效公共截面积拟合为{area_fit:.8g}平方米；等效圆半径{np.sqrt(area_fit/np.pi)*1e6:.6g}微米。拟合全部正实验虚部的对数误差，非独立验证。实际峰位{peak_hz:.6g} Hz；面积不能改变峰位。\n\n统计分箱CSV仅是求解器兼容输入，无孔喉连接关系，不能作为三维提取网络。体积列用目标权重乘原始总孔喉体积表示统计体积份额，不是单根圆柱体积；不得据此重建孔隙率。CT场保持y方向ds2。未修改原始网络。原图、目标直方图和分箱源数据随包保存；本次不重新生成未变化的数字岩心HTML。\n\n连续曲线为同一模型乘GPU几何系数，7个圆点为新GPU计算。实部未拟合，完整误差见配置；单一面积不能消除频谱形状差异。\n",encoding='utf8')
    shutil.copy2(__file__,OUT/'configs'/Path(__file__).name)

if __name__=='__main__':
    main()
