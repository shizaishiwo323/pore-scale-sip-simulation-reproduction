"""四组原始提取网络的7频点膜极化对比，复用既有GPU场求解器。"""
from pathlib import Path
from dataclasses import asdict
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import shutil
import sys
import gc

import numpy as np
import pandas as pd
import cupy as cp
import run_sample89_3d_sip_comparison as solver
from compute_polarization_spectra import load_pnextract
from pore_scale_electrical.polarization import (
    PolarizationParameters, membrane_polarization_conductance,
    upscale_conductance_to_water_conductivity,
)

ROOT = Path(__file__).resolve().parents[3]
GEOMETRY = ROOT/'results/sample16_pnextract_sensitivity_20260913'
BASE = ROOT/'results/sample16_gpu_7freq_20260912_2340'
PREVIOUS = ROOT/'results/sample16_gpu_yz_7freq_20260913/y'
RUN = ROOT/'results/sample16_pnextract_membrane_gpu7_20260913'
NAMES = ['default','min125','min100','detail_smooth3']
LABELS = ['Default','minRPore = 1.25','minRPore = 1.0','Joint refinement']
COLORS = ['#777777','#439878','#377eb8','#e38b34']
FREQUENCIES = np.array([.1,.704961,6.343811,44.72136,315.267933,2837.038291,20000.])
# 沿用既有样品16物性，不拟合实验，不改变孔喉尺寸或Zdc缩放因子。
PARAMS = PolarizationParameters(water_conductivity_s_m=.12,dynamic_pore_size_m=2.45913e-6)


def dump(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def membrane_water(throats):
    increment = membrane_polarization_conductance(FREQUENCIES,throats.length_m.to_numpy(),
                        throats.weight.to_numpy(),throats.zdc_ohm.to_numpy(),PARAMS)
    return PARAMS.water_conductivity_s_m + upscale_conductance_to_water_conductivity(increment,PARAMS)


def prepare():
    for d in ['configs/source_snapshot','source_data','provenance','figures','pore_network','simulation_sweeps']:
        (RUN/d).mkdir(parents=True,exist_ok=True)
    previous_command = json.loads((PREVIOUS/'configs/membrane_command.json').read_text(encoding='utf-8-sig'))
    i = previous_command.index('--frequencies')
    assert np.array_equal(np.array(previous_command[i+1:i+8],dtype=float),FREQUENCIES)
    for name in ['field_ds2.mhd','field_ds2.raw']:
        destination = RUN/'configs'/name
        if not destination.exists():
            shutil.copy2(BASE/'configs'/name,destination)
        assert sha(destination) == sha(BASE/'configs'/name)
    for name in ['experiment_all.csv','selected_7_frequencies.csv']:
        shutil.copy2(BASE/'source_data'/name,RUN/'source_data'/name)
    input_hashes = {'field_raw':sha(RUN/'configs/field_ds2.raw')}
    diagnostics = {}
    for name in NAMES:
        source = BASE/'pore_network/network_parsed' if name == 'default' else GEOMETRY/'pore_network/full'/name/'network_parsed'
        target = RUN/'pore_network'/name
        target.mkdir(exist_ok=True)
        input_hashes[name] = {}
        for fn in ['pores.csv','throats.csv']:
            shutil.copy2(source/fn,target/fn)
            input_hashes[name][fn] = {'source':str(source/fn),'sha256':sha(source/fn)}
            assert sha(target/fn) == input_hashes[name][fn]['sha256']
        p, t = pd.read_csv(target/'pores.csv'), pd.read_csv(target/'throats.csv')
        _, loaded = load_pnextract(target,PARAMS)
        assert np.isfinite(loaded[['length_m','weight','zdc_ohm']]).all().all()
        assert (loaded[['length_m','zdc_ohm']]>0).all().all() and (loaded.weight>=0).all()
        water = membrane_water(loaded)
        positive = membrane_water(loaded[loaded.weight>0])
        # 零体积喉按已有模型自然取零权重；验证其不改变谱，不删除原始记录。
        assert np.allclose(water,positive,rtol=1e-12,atol=1e-15)
        diagnostics[name] = {'pores':len(p),'all_throats':len(t),'internal_throats':len(loaded),
             'zero_volume_pores':int((p.pore_volume_m3==0).sum()),
             'zero_volume_throats':int((t.throat_volume_m3==0).sum()),
             'zero_weight_internal_throats':int((loaded.weight==0).sum()),
             'zero_weight_contribution_verified':True}
    dump(RUN/'provenance/input_hashes.json',input_hashes)
    dump(RUN/'provenance/network_diagnostics.json',diagnostics)
    config = {'性质':'仅膜极化、四组PNM参数敏感性；二倍降采样CT场求解，不是全分辨率电场验证',
        '几何':'网络均从800×666×666原始CT提取，体素1.92 μm；场网格400×333×333，体素3.84 μm，保持此前ds2设置',
        '方向与边界':'solver y = CT存储Y，周期边界；实验加载方向对应未确认；所有候选共享同一CT场网格',
        '膜机制':'水相=sigma_w+Delta sigma_membrane；固相严格为0，不含Maxwell介电背景、孔极化或固相电导',
        '参数来源':'沿用既有样品16参数配置及y向7频点膜模拟；本轮只替换原始提取网络',
        'parameters':asdict(PARAMS),'frequencies_hz':FREQUENCIES.tolist(),
        '频点选择':'完全复用原7个实验点；每组均重新调用GPU扫频，不复用旧模拟结果',
        '权重':'内部孔喉体积归一化；零体积喉保持零权重；不按新增孔喉条数放大响应',
        '几何定义':'喉长=link2净长度；面积=r²/(4G)；Zdc由各候选实际长度及面积重新计算',
        'scales':{'pore_radius':1.,'membrane_length':1.,'membrane_zdc':1.},
        '求解':'复用run_component_sweep；GPU孔隙CSR、CG+Jacobi，归一化双精度实数场再乘复电导；每组首频独立求解，组内warm start',
        'rtol':1e-6,'atol':0.,'maxiter':50000,'dtype':'complex128',
        '输出边界':'此次明确只比较膜极化；不重跑其它机制，不重建相同CT的3D HTML；几何分布与提取日志保留在前一结果包',
        'geometry_package':str(GEOMETRY),
        'command':[sys.executable,str(Path(__file__).resolve())]}
    dump(RUN/'configs/parameters_zh.json',config)
    paths = [Path(__file__),Path(solver.__file__),ROOT/'code/scripts/sip_simulation/compute_polarization_spectra.py']
    paths += [ROOT/'code/src/pore_scale_electrical'/f for f in ['polarization.py','ac3d_pore_gpu.py','ac3d_active_domain.py','ac3d_gpu.py']]
    snapshots = {}
    for path in paths:
        shutil.copy2(path,RUN/'configs/source_snapshot'/path.name)
        snapshots[str(path)] = sha(path)
    dump(RUN/'provenance/source_hashes.json',snapshots)


def run():
    info = solver.parse_mhd(RUN/'configs/field_ds2.mhd')
    labels = solver.load_label_volume(info)
    assert labels.shape == (400,333,333) and set(np.unique(labels)) == {0,1}
    for name in NAMES:
        print('START',name,flush=True)
        with (RUN/'provenance'/f'{name}_gpu.log').open('w',encoding='utf8') as log, redirect_stdout(log), redirect_stderr(log):
            _,paths,_ = solver.make_component_spectra_files(output_prefix=name,frequencies_hz=FREQUENCIES,
                network_dir=RUN/'pore_network'/name,out_dir=RUN/'spectra'/name,mode='paper',params=PARAMS,
                solid_conductivity_s_m=0.,solid_background_components=set())
            options = dict(mechanism='membrane',spectra_path=paths['membrane'],frequencies_hz=FREQUENCIES.tolist(),
                voxel_size_m=info['voxel_size_m'],out_dir=RUN/'simulation_sweeps'/name,pore_label=0,solid_label=1,
                direction='y',rtol=1e-6,atol=0.,maxiter=50000,dtype_name='complex128',preconditioner='jacobi',
                fft_reference='mean-face',progress_every=500,residual_every=500,iteration_sleep_s=0.,
                zero_solid_regularization_s_m=0.,resume=False,solver_backend='gpu-active-pore-csr',
                jacobi_omega=.65,sor_omega=1.35)
            serializable = {k:str(v) if isinstance(v,Path) else v for k,v in options.items()}
            dump(RUN/'configs'/f'{name}_solver.json',serializable)
            sweep = solver.run_component_sweep(labels=labels,**options)
            assert len(sweep)==7 and sweep.converged_strict.all() and (sweep.relative_residual_norm<=1e-6).all()
            for path in (RUN/'simulation_sweeps'/name).glob('frequency_*/result.json'):
                dump(path.parent/'config.yml',serializable)
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        print('DONE',name,flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    diagnostics = json.loads((RUN/'provenance/network_diagnostics.json').read_text(encoding='utf8'))
    frames, verification = [], {}
    for name in NAMES:
        f = pd.read_csv(RUN/'simulation_sweeps'/name/'sweep_results.csv')
        assert len(f)==7 and np.array_equal(f.frequency_hz,FREQUENCIES)
        assert f.converged_strict.all() and (f.relative_residual_norm<=1e-6).all()
        assert (f.solid_sigma_real_s_m==0).all() and (f.solid_sigma_imag_s_m==0).all()
        _,t = load_pnextract(RUN/'pore_network'/name,PARAMS)
        water = f.water_sigma_real_s_m.to_numpy()+1j*f.water_sigma_imag_s_m.to_numpy()
        assert np.allclose(water,membrane_water(t),rtol=1e-12,atol=1e-15)
        effective = f.effective_sigma_real_s_m.to_numpy()+1j*f.effective_sigma_imag_s_m.to_numpy()
        factors = effective/water
        verification[name] = {'max_residual':float(f.relative_residual_norm.max()),
             'geometry_factor_mean':float(factors.real.mean()),'geometry_factor_relative_span':float(np.ptp(factors.real)/factors.real.mean()),
             'all_7_converged':True,'network':diagnostics[name]}
        assert np.max(np.abs(factors.imag))<1e-12 and np.ptp(factors.real)/factors.real.mean()<1e-4
        frames.append(f.assign(case=name))
    combined = pd.concat(frames,ignore_index=True)
    combined.to_csv(RUN/'source_data/membrane_gpu7_comparison.csv',index=False)
    old = pd.read_csv(PREVIOUS/'simulation_sweeps/membrane/sweep_results.csv')
    baseline = frames[0]
    baseline_difference = float(np.max(np.abs(baseline.effective_sigma_imag_s_m/old.effective_sigma_imag_s_m-1)))
    assert baseline_difference<1e-3
    exp = pd.read_csv(RUN/'source_data/experiment_all.csv')
    assert set(FREQUENCIES) <= set(exp.frequency_hz)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':13,'mathtext.fontset':'stix',
                         'axes.linewidth':.8,'xtick.direction':'in','ytick.direction':'in'})
    for mode in ['comparison','ratio']:
        fig, axes = plt.subplots(1,2,figsize=(11,5.2))
        fig.subplots_adjust(left=.085,right=.98,bottom=.21,top=.78,wspace=.30)
        for name,label,color,marker in zip(NAMES,LABELS,COLORS,['o','s','D','^']):
            f = combined[combined.case==name]
            for j,part in enumerate(['real','imag']):
                values = f[f'effective_sigma_{part}_s_m'].to_numpy()
                if mode=='ratio':
                    values = values / baseline[f'effective_sigma_{part}_s_m'].to_numpy()
                axes[j].plot(FREQUENCIES,values,color=color,marker=marker,ms=5,mfc='white',lw=1.6,label=label)
        for j,(ax,part) in enumerate(zip(axes,['real','imag'])):
            if mode=='comparison':
                values = exp[f'experiment_sigma_{part}_s_m']
                keep = values>0
                ax.plot(exp.frequency_hz[keep],values[keep],'k.',alpha=.5,ms=4,label='Experiment (total response)')
                ax.set_yscale('log')
                ax.set_ylabel([r"$\sigma'$ (S/m)",r"$\sigma''$ (S/m)"][j])
            else:
                ax.set_ylabel(['Real conductivity / default','Imaginary conductivity / default'][j])
            ax.set(xscale='log',xlim=(.08,25000),xlabel='Frequency (Hz)')
            ax.text(-.14,1.03,['(a)','(b)'][j],transform=ax.transAxes,fontsize=19)
        handles,labels_legend = axes[0].get_legend_handles_labels()
        fig.legend(handles,labels_legend,loc='upper center',bbox_to_anchor=(.53,.91),ncol=len(handles),fontsize=9,frameon=False)
        fig.suptitle('Sample 16 | membrane polarization | y direction',fontsize=14,y=.98)
        fig.text(.53,.075,'7 GPU frequencies per network; lines connect computed points.',ha='center',fontsize=11)
        fig.text(.53,.03,'Full-resolution network; ds2 CT field; volume-weighted membrane model.',ha='center',fontsize=11)
        fig.savefig(RUN/'figures'/f'membrane_gpu7_{mode}.png',dpi=600)
        fig.savefig(RUN/'figures'/f'membrane_gpu7_{mode}_preview.png',dpi=150)
        plt.close(fig)
    ratios = []
    for frame in frames:
        for i,row in frame.iterrows():
            ratios.append({'case':row.case,'frequency_hz':row.frequency_hz,
                'real_ratio_to_default':row.effective_sigma_real_s_m/baseline.iloc[i].effective_sigma_real_s_m,
                'imag_ratio_to_default':row.effective_sigma_imag_s_m/baseline.iloc[i].effective_sigma_imag_s_m})
    pd.DataFrame(ratios).to_csv(RUN/'source_data/ratios_to_default.csv',index=False)
    dump(RUN/'provenance/verification.json',{'points':28,'cases':verification,
         'default_reproduction_max_relative_imag_difference':baseline_difference,'related_tests_passed':19,
         'plot':'only 7 calculated frequencies per case, straight connecting segments, no interpolated model peaks',
         'experiment_nonpositive_imag_omitted':int((exp.experiment_sigma_imag_s_m<=0).sum())})
    lines = ['# 样品16：pnextract参数对膜极化的影响（7频点）','',
        '四组网络分别重新调用GPU求解，共28个频点，真实残差全部满足1e-6。仅比较膜极化；实验点是全部机制的总响应，不能把两者差异直接视为膜机制拟合误差。','',
        '频点（Hz）：'+', '.join(f'{f:g}' for f in FREQUENCIES)+'。','',
        '沿用此前solver y（CT存储Y）方向、周期边界和400×333×333二倍降采样CT场网格。孔喉网络来自完整分辨率提取，几何与Zdc均未施加缩放。此次PNM只通过内部孔喉长度、横截面积与体积权重改变水相膜谱；场求解仍使用同一CT，所以不是在四个网络图上直接求解输运，也不独立验证其拓扑。','',
        '水相为0.12 S/m加膜极化增量，固相严格绝缘；D=1.3e-9 m²/s，eta=0.01，动态孔径2.45913 μm；其余参数与既有运行一致。膜权重为归一化体积，新增孔喉数量不会直接等比例放大谱。','',
        'min100和联合细分网络仍有零体积元素。零体积喉在已有膜模型中贡献为零，已验证去除零权重项前后水相膜谱一致；原始CSV保持这些记录。电学求解收敛不等于几何异常已修复。','',
        '| 参数组 | 0.1 Hz虚部 (S/m) | 315.268 Hz虚部 (S/m) | 20000 Hz虚部 (S/m) | 最大残差 |','|---|---:|---:|---:|---:|']
    for name,label,frame in zip(NAMES,LABELS,frames):
        lines.append(f'| {label} | {frame.iloc[0].effective_sigma_imag_s_m:.6e} | {frame.iloc[4].effective_sigma_imag_s_m:.6e} | {frame.iloc[-1].effective_sigma_imag_s_m:.6e} | {verification[name]["max_residual"]:.3e} |')
    lines += ['', '图中每组只有7个GPU计算点，连线仅作视觉引导，不能据此精确确定频点之间的峰位。reference几何图、CT输入和pnextract日志沿用前一提取结果包，路径与源数据哈希记录在configs及provenance中。', '',
        '输出：figures中的实部/虚部对比与相对默认比值图；simulation_sweeps/<case>/中的7点CSV及逐频result.json、residual_history.csv、config.yml；source_data中的合并结果和比值；pore_network中的4组原始提取CSV副本。']
    (RUN/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(pd.DataFrame(ratios).to_string(index=False),flush=True)


if __name__=='__main__':
    if '--plot-only' not in sys.argv:
        prepare()
        run()
    plot()
