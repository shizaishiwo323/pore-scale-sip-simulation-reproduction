"""样品16：固定0.3微米有效长度，重新计算7频点独立膜机制。"""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from pathlib import Path
import json
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import run_sample16_effective_membrane_length_gpu7 as previous

ROOT, SOURCE = previous.ROOT, previous.RUN
RUN = ROOT/'results/sample16_membrane_only_L030_gpu7_20260913'
solver, dump, sha = previous.solver, previous.dump, previous.sha
CALIBRATION = json.loads((SOURCE/'configs/calibration_parameters_zh.json').read_text(encoding='utf8'))
PARAMS = previous.prior.PolarizationParameters(**CALIBRATION['parameters'])
FREQUENCIES = previous.FREQ


def run():
    for folder in ['configs/source_snapshot','pore_network','spectra','source_data','provenance','figures','simulation_sweeps']:
        (RUN/folder).mkdir(parents=True,exist_ok=True)
    inputs = {}
    for source,target in [(SOURCE/'configs'/f,RUN/'configs'/f) for f in ['field_ds2.mhd','field_ds2.raw']] + [
            (SOURCE/'pore_network/L030'/f,RUN/'pore_network'/f) for f in ['pores.csv','throats.csv']]:
        shutil.copy2(source,target)
        assert sha(source)==sha(target)
        inputs[str(target.relative_to(RUN))]={'source':str(source),'sha256':sha(source)}
    _,t = previous.prior.load_pnextract(RUN/'pore_network',PARAMS)
    assert np.all(t.membrane_relaxation_length_m.to_numpy()==.3e-6)
    _,original = previous.prior.load_pnextract(SOURCE/'pore_network/original',PARAMS)
    assert np.array_equal(t.zdc_ohm,original.zdc_ohm) and np.array_equal(t.weight,original.weight)
    _,paths,_ = solver.make_component_spectra_files(output_prefix='membrane_only',frequencies_hz=FREQUENCIES,
        network_dir=RUN/'pore_network',out_dir=RUN/'spectra',mode='paper',params=PARAMS,
        solid_conductivity_s_m=0.,solid_background_components=set())
    options = json.loads((SOURCE/'configs/calibrated_L030_membrane_solver.json').read_text(encoding='utf8'))
    options.update(spectra_path=paths['membrane'],out_dir=RUN/'simulation_sweeps/membrane',resume=False)
    serialized={k:str(v) if isinstance(v,Path) else v for k,v in options.items()}
    dump(RUN/'configs/solver.json',serialized)
    dump(RUN/'configs/parameters_zh.json',{'性质':'固定有效长度的独立膜极化诊断，本次重新GPU求解，未重新拟合参数。',
        'parameters':asdict(PARAMS),'effective_relaxation_length_m':.3e-6,
        '参数来源':'沿用上一轮条件标定eta=0.002442810092526288；有效长度0.3微米为假设尺度，几何长度及Zdc不变。',
        '相电导率':'水相仅为sigma_w+Delta_sigma_membrane；固相严格为0。无孔极化、高频介电项或固相电导背景。',
        '边界与网格':'周期边界、y方向、400×333×333 ds2场、体素3.84微米，pore=0/solid=1。不是完整分辨率场。',
        '频点':FREQUENCIES.tolist(),'求解':serialized,
        '绘图':'两个面板分别显示独立膜机制实部和虚部；7个空心圆为新GPU结果，连续线为同一局部膜模型乘经核验的GPU几何系数。峰位来自模型极值。',
        '范围':'按本次明确的单机制要求，不重跑综合机制。几何未改变，三维HTML及几何分布沿用原包，不把有效长度画成真实几何。',
        'geometry_visualization_source':str(previous.BASE),'input_files':inputs,
        'command':[sys.executable,str(Path(__file__).resolve())]})
    for path in [Path(__file__),Path(solver.__file__),ROOT/'code/src/pore_scale_electrical/polarization.py',
                 ROOT/'code/src/pore_scale_electrical/ac3d_pore_gpu.py',ROOT/'code/scripts/sip_simulation/compute_polarization_spectra.py']:
        shutil.copy2(path,RUN/'configs/source_snapshot'/path.name)
    info=solver.parse_mhd(RUN/'configs/field_ds2.mhd')
    labels=solver.load_label_volume(info)
    assert labels.shape==(400,333,333) and set(np.unique(labels))=={0,1}
    print('START fresh membrane-only GPU sweep',flush=True)
    with (RUN/'provenance/gpu.log').open('w',encoding='utf8') as log,redirect_stdout(log),redirect_stderr(log):
        solver.run_component_sweep(labels=labels,**options)
    for path in options['out_dir'].glob('frequency_*/result.json'):
        dump(path.parent/'config.yml',serialized)
    print('DONE fresh membrane-only GPU sweep',flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gpu=pd.read_csv(RUN/'simulation_sweeps/membrane/sweep_results.csv')
    assert len(gpu)==7 and np.array_equal(gpu.frequency_hz,FREQUENCIES)
    assert gpu.converged_strict.all() and (gpu.relative_residual_norm<=1e-6).all()
    assert (gpu.solid_sigma_real_s_m==0).all() and (gpu.solid_sigma_imag_s_m==0).all()
    _,t=previous.prior.load_pnextract(RUN/'pore_network',PARAMS)
    def increment(f):
        return previous.prior.upscale_conductance_to_water_conductivity(previous.prior.membrane_polarization_conductance(
            np.atleast_1d(f),t.membrane_relaxation_length_m.to_numpy(),t.weight.to_numpy(),t.zdc_ohm.to_numpy(),PARAMS),PARAMS)
    water=gpu.water_sigma_real_s_m.to_numpy()+1j*gpu.water_sigma_imag_s_m.to_numpy()
    effective=gpu.effective_sigma_real_s_m.to_numpy()+1j*gpu.effective_sigma_imag_s_m.to_numpy()
    assert np.allclose(water,PARAMS.water_conductivity_s_m+increment(FREQUENCIES),rtol=1e-12,atol=1e-15)
    factor=float(np.median((effective/water).real))
    check=factor*(PARAMS.water_conductivity_s_m+increment(FREQUENCIES))
    error=float(np.max(np.abs(check-effective)/np.abs(effective)))
    imag_error=float(np.max(np.abs(check.imag/effective.imag-1)))
    assert error<1e-5 and imag_error<1e-5
    opt=minimize_scalar(lambda x:-increment(10**x)[0].imag,bounds=(3,5),method='bounded')
    assert opt.success
    peak=10**opt.x
    peak_imag=float(factor*increment(peak)[0].imag)
    f=np.unique(np.r_[np.geomspace(.1,20000,601),peak])
    curve=factor*(PARAMS.water_conductivity_s_m+increment(f))
    pd.DataFrame({'frequency_hz':f,'effective_sigma_real_s_m':curve.real,'effective_sigma_imag_s_m':curve.imag}).to_csv(RUN/'source_data/dense_membrane_model.csv',index=False)
    gpu.to_csv(RUN/'source_data/membrane_gpu7.csv',index=False)
    old=pd.read_csv(SOURCE/'simulation_sweeps/calibrated_L030_membrane/sweep_results.csv')
    old_complex=old.effective_sigma_real_s_m.to_numpy()+1j*old.effective_sigma_imag_s_m.to_numpy()
    repeat_error=float(np.max(np.abs(effective-old_complex)/np.abs(old_complex)))
    assert repeat_error<1e-5
    verification={'new_gpu_points':7,'all_strictly_converged':True,'max_residual':float(gpu.relative_residual_norm.max()),
        'solid_exactly_zero':True,'pure_membrane_water_verified':True,'geometry_factor':factor,
        'dense_model_vs_gpu_relative_complex_error':error,'dense_model_vs_gpu_relative_imag_error':imag_error,
        'repeat_vs_previous_independent_membrane_relative_error':repeat_error,
        'model_peak_hz':peak,'model_peak_effective_sigma_imag_s_m':peak_imag,
        'peak_definition':'continuous membrane model, not an extra GPU frequency',
        'comparison_scope':'standalone membrane mechanism with insulating solid; not all-minus-no-membrane'}
    dump(RUN/'provenance/verification.json',verification)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':13,'mathtext.fontset':'stix',
        'axes.linewidth':.8,'xtick.direction':'in','ytick.direction':'in'})
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    fig.subplots_adjust(left=.135,right=.98,bottom=.22,top=.79,wspace=.32)
    for j,part in enumerate(['real','imag']):
        ax=axes[j]
        ax.plot(f,getattr(curve,part),color='#439878',lw=2,label='Membrane model')
        ax.plot(FREQUENCIES,getattr(effective,part),'o',color='#439878',mfc='white',ms=5,label='7 GPU frequencies')
        ax.set(xscale='log',yscale='log',xlim=(.08,25000),xlabel='Frequency (Hz)',ylabel=[r"$\sigma'$ (S/m)",r"$\sigma''$ (S/m)"][j])
        ax.text(-.12,1.03,['(a)','(b)'][j],transform=ax.transAxes,fontsize=16)
    axes[1].annotate(f'Peak ≈ {peak/1000:.2f} kHz\n{peak_imag:.2e} S/m',xy=(peak,peak_imag),
        xytext=(.12,.84),textcoords='axes fraction',fontsize=11,
        arrowprops={'arrowstyle':'->','color':'#777777','lw':.8})
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,.91),ncol=2,frameon=False,fontsize=11)
    fig.suptitle(f'Sample 16 | membrane only | L = 0.3 μm | η = {PARAMS.membrane_polarizability:.4g}',fontsize=14,y=.98)
    fig.text(.53,.085,'Insulating solid; water = DC conductivity + membrane increment.',ha='center',fontsize=11)
    fig.text(.53,.035,'Effective-length diagnostic; ds2 field, y direction. Curves use the verified GPU geometry factor.',ha='center',fontsize=10)
    for suffix,dpi in [('',600),('_preview',150)]:
        fig.savefig(RUN/'figures'/f'membrane_only_L030{suffix}.png',dpi=dpi)
    plt.close(fig)
    (RUN/'README.md').write_text(f'# 样品16：0.3微米独立膜极化\n\n固定有效弛豫长度0.3 μm、eta={PARAMS.membrane_polarizability:.10g}、D=1.3e-9 m²/s、动态孔径2.45913 μm、水电导率0.12 S/m。本次未重新拟合eta。\n\n水相仅含直流电导和膜极化增量，固相严格为0；无孔极化和高频介电背景。直接求解独立膜机制，并非综合结果相减。真实几何、Zdc与体积权重保持不变。\n\n重新运行7个GPU频点，全部残差小于1e-6。沿用400×333×333 ds2场、3.84 μm体素、y方向及周期边界。0.3 μm仍为亚体素假设尺度，不是提取测得的喉长。\n\n纯膜虚部模型峰位{peak:.4f} Hz，峰值{peak_imag:.8e} S/m。7个空心圆是实际GPU结果，连续线使用同一膜公式和经7点核验的几何系数；峰位来自连续模型，不是第8个GPU点。实部包含水相直流背景，不能解释为单独的极化增量。\n\n本次按单机制要求输出独立图，不将含全部机制的实验总响应作为纯膜拟合目标。三维HTML与几何分布没有变化，沿用参数文件所列原结果包。配置及源码快照在configs，数值解与逐频残差在simulation_sweeps，源数据在source_data。\n',encoding='utf8')
    print(json.dumps(verification,indent=2),flush=True)


if __name__=='__main__':
    if '--plot-only' not in sys.argv:
        run()
    plot()
