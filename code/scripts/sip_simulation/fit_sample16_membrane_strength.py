"""固定人工短孔喉几何，仅标定膜极化率；结果不是独立验证。"""
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import sample16_short_throat_diagnostic as prior

OUT=prior.ROOT/'results/sample16_membrane_strength_fit_diagnostic_20260913'

def main():
    for name in ['configs','source_data','figures','provenance']:
        (OUT/name).mkdir(parents=True,exist_ok=True)
    _,t=prior.load_pnextract(prior.RUN/'effective_network',prior.PARAMS)
    old=pd.read_csv(prior.RUN/'simulation_sweeps/membrane/sweep_results.csv')
    water=old.water_sigma_real_s_m.to_numpy()+1j*old.water_sigma_imag_s_m.to_numpy()
    eff=old.effective_sigma_real_s_m.to_numpy()+1j*old.effective_sigma_imag_s_m.to_numpy()
    factor=float(np.median((eff/water).real))
    exp=pd.read_csv(prior.Y/'source_data/experiment_all.csv')
    f=exp.frequency_hz.to_numpy()
    obs=exp.experiment_sigma_imag_s_m.to_numpy()
    mask=(f>=1e3)&(f<=1e4)&(obs>0)
    assert mask.sum()>=3
    def model(freq,eta):
        p=replace(prior.PARAMS,membrane_polarizability=float(eta))
        c=prior.membrane_polarization_conductance(np.atleast_1d(freq),t.length_m.to_numpy(),t.weight.to_numpy(),t.zdc_ohm.to_numpy(),p)
        return factor*(p.water_conductivity_s_m+prior.upscale_conductance_to_water_conductivity(c,p))
    def loss(logeta):
        return float(np.mean(np.log10(model(f[mask],10**logeta).imag/obs[mask])**2))
    fit=minimize_scalar(loss,bounds=(-7,-1),method='bounded',options={'xatol':1e-8})
    eta=float(10**fit.x)
    assert fit.success and 1e-7<eta<.1 and loss(fit.x)<loss(-2)
    peak=minimize_scalar(lambda x:-model(10**x,eta)[0].imag,bounds=(2,5),method='bounded')
    peak_hz=float(10**peak.x)
    assert 1e3<peak_hz<1e4
    metrics={}
    for name,value in [('before',.01),('fitted',eta)]:
        pred=model(f,value)
        metrics[name]={}
        for region,keep in [('target_band',mask),('all_positive',obs>0)]:
            metrics[name][region]={'n':int(keep.sum()),'imag_log10_rmse':float(np.sqrt(np.mean(np.log10(pred.imag[keep]/obs[keep])**2))), 'imag_median_relative_error':float(np.median(np.abs(pred.imag[keep]/obs[keep]-1)))}
    area=t.membrane_active_area_m2.to_numpy()
    config={'性质':'短孔喉条件下，用实验虚部标定一个有效膜极化率；不是独立验证或样品实测极化率。',
        'membrane_polarizability_before':.01,'membrane_polarizability_fitted':eta,'peak_hz':peak_hz,
        '拟合定义':'仅使用1e3–1e4 Hz内正实验虚部，最小化log10预测/观测的均方误差；不拟合实部，不添加背景，不改面积或长度。',
        '参数界限':'1e-7 < eta < 0.1，仅为搜索约束，不代表文献证实范围。',
        '面积说明':'逐孔喉由radius^2/(4*shape_factor)计算，并非手工指定平均面积；没有发现本次读取中的单位转换错误。有效面积与极化率的幅值作用近似退化，不能同时唯一识别。',
        'area_m2_min_median_max':[float(np.min(area)),float(np.median(area)),float(np.max(area))],
        'area_m2_volume_weighted_mean':float(np.average(area,weights=t.weight)),
        'geometry_factor':factor,'network':str(prior.RUN/'effective_network'),
        'source_reference':'https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020JB020515',
        '参考值说明':'Niu 2020 Berea eta=1%，不能直接确认为样品16参数；本次拟合值没有独立物性验证。',
        '边界':'保留人工短孔喉及ds2 y向场；低于CT分辨率；实验含多机制，因此这是膜机制对总实验虚部的条件拟合。',
        'metrics':metrics}
    (OUT/'configs/parameters_zh.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(config,ensure_ascii=False,indent=2),flush=True)
    cmd=json.loads((prior.RUN/'configs/membrane_command.json').read_text())
    cmd[cmd.index('--out-dir')+1]=str(OUT)
    cmd[cmd.index('--sample-label')+1]='Sample 16: conditional membrane-strength fit'
    cmd+=['--membrane-polarizability',str(eta)]
    (OUT/'configs/membrane_command.json').write_text(json.dumps(cmd,indent=2),encoding='utf8')
    with (OUT/'provenance/gpu.log').open('w',encoding='utf8') as log:
        subprocess.run(cmd,cwd=prior.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    gpu=pd.read_csv(OUT/'simulation_sweeps/membrane/sweep_results.csv')
    assert len(gpu)==7 and gpu.converged_strict.all() and (gpu.relative_residual_norm<=1e-6).all()
    actual=gpu.effective_sigma_real_s_m.to_numpy()+1j*gpu.effective_sigma_imag_s_m.to_numpy()
    check=float(np.max(np.abs(model(gpu.frequency_hz.to_numpy(),eta)-actual)/np.abs(actual)))
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
    fig.suptitle('Sample 16 | y direction | membrane-strength calibration',fontsize=14)
    dense=np.geomspace(.1,20000,401)
    rows=[]
    for label,value,color,points in [('Before',.01,'#ed8a28',old),('Calibrated',eta,'#377eb8',gpu)]:
        curve=model(dense,value)
        rows.append(pd.DataFrame({'case':label,'frequency_hz':dense,'sigma_real_s_m':curve.real,'sigma_imag_s_m':curve.imag}))
        for ax,part in zip(axes,['real','imag']):
            ax.plot(dense,getattr(curve,part),color=color,lw=2,label=f'{label}: eta = {value:.3g}')
            ax.plot(points.frequency_hz,points[f'effective_sigma_{part}_s_m'],'o',ms=5,mfc='white',mec=color)
    for i,(ax,part) in enumerate(zip(axes,['real','imag'])):
        values=exp[f'experiment_sigma_{part}_s_m']; keep=values>0
        ax.plot(f[keep],values[keep],'^',mfc='white',mec='black',ms=4,label='Experiment')
        ax.axvspan(1e3,1e4,color='grey',alpha=.10)
        ax.set(xscale='log',yscale='log',xlim=(.08,25000),xlabel='Frequency (Hz)',ylabel=f'Conductivity {part} (S/m)')
        ax.text(-.12,1.02,f'({chr(97+i)})',transform=ax.transAxes,fontsize=19)
    axes[0].legend(frameon=False,fontsize=11)
    fig.text(.16,.035,f'Fixed shortened geometry; eta fitted to imaginary data at 1-10 kHz.\nCircles: 7 GPU points; lines: model. Calibration, not independent validation.',fontsize=10)
    fig.savefig(OUT/'figures/membrane_strength_comparison.png',dpi=600)
    plt.close(fig)
    pd.concat(rows).to_csv(OUT/'source_data/dense_curves.csv',index=False)
    comparison=exp.copy()
    comparison['used_for_fit']=mask
    for label,value in [('before',.01),('fitted',eta)]:
        v=model(f,value)
        comparison[label+'_real_s_m']=v.real
        comparison[label+'_imag_s_m']=v.imag
    comparison.to_csv(OUT/'source_data/experimental_frequency_comparison.csv',index=False)
    (OUT/'README.md').write_text(f"# 样品16膜极化强度条件标定\n\n固定上一轮人工短孔喉、逐孔喉面积、权重、y方向和ds2场。只将eta由0.01调整为{eta:.8g}，峰位{peak_hz:.6g} Hz。拟合目标是1–10 kHz内正实验虚部的对数误差。源数据记录全部频率及拟合筛选；实部未拟合。\n\n原参数沿用Niu 2020 Berea的1%，不能断定在样品16中成立。长度缩短而面积不变使Zdc显著降低，是上一轮幅值放大的直接原因。截面积逐孔喉计算，未使用自由平均面积。此次固定面积只标定eta，不能据此唯一反演膜面积或证明参数物理真实性。\n\n实验包含全部机制，膜单机制不能代表总响应。该标定会保留频谱形状与实部的偏差，详见configs/parameters_zh.json；不可称为全频段或全部机制拟合完成。原始数据和上一轮结果未修改。未重复导出未变化的网络分布/三维HTML，使用配置中引用的网络；本次只生成参数拟合和7频点膜机制结果。\n\n连续线由已核验的GPU几何系数与同一膜模型生成，7个空心圆为本次实际GPU结果。峰位为模型极值，不是额外GPU频点。\n",encoding='utf8')
    shutil.copy2(__file__,OUT/'configs'/Path(__file__).name)

if __name__=='__main__':
    main()
