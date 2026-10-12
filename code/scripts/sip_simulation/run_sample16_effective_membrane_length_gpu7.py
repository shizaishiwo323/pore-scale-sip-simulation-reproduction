"""固定真实几何和Zdc，诊断独立膜弛豫长度；每组7频点GPU膜/综合机制。"""
from dataclasses import asdict, replace
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import gc
import csv
import json
import shutil
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import run_sample16_pnextract_membrane_gpu7 as prior

solver, cp = prior.solver, prior.cp
ROOT, BASE, PREVIOUS = prior.ROOT, prior.BASE, prior.PREVIOUS
PARAMS, FREQ = prior.PARAMS, prior.FREQUENCIES
RUN = ROOT/'results/sample16_effective_membrane_length_gpu7_20260913'
CASES = {'L030': .3e-6, 'L050': .5e-6, 'L100': 1e-6}
dump, sha = prior.dump, prior.sha


def response(freq, throats):
    col = 'membrane_relaxation_length_m' if 'membrane_relaxation_length_m' in throats else 'length_m'
    value = prior.membrane_polarization_conductance(np.atleast_1d(freq), throats[col].to_numpy(),
        throats.weight.to_numpy(), throats.zdc_ohm.to_numpy(), PARAMS)
    return prior.upscale_conductance_to_water_conductivity(value, PARAMS)


def prepare():
    for folder in ['configs/source_snapshot', 'source_data', 'provenance', 'figures', 'pore_network', 'simulation_sweeps']:
        (RUN/folder).mkdir(parents=True, exist_ok=True)
    for name in ['field_ds2.mhd', 'field_ds2.raw']:
        target = RUN/'configs'/name
        if not target.exists():
            shutil.copy2(BASE/'configs'/name, target)
        assert sha(target) == sha(BASE/'configs'/name)
    for name in ['experiment_all.csv', 'selected_7_frequencies.csv']:
        shutil.copy2(BASE/'source_data'/name, RUN/'source_data'/name)
    source = BASE/'pore_network/network_parsed'
    raw = pd.read_csv(source/'throats.csv')
    _, original = prior.load_pnextract(source, PARAMS)
    audits, peaks, dense = {}, [], []
    for name, length in {'original': None, **CASES}.items():
        target = RUN/'pore_network'/name
        target.mkdir(exist_ok=True)
        shutil.copy2(source/'pores.csv', target/'pores.csv')
        if length is None:
            shutil.copy2(source/'throats.csv', target/'throats.csv')
        else:
            # 保留原CSV数值文本，避免浮点读写改变几何列的末位。
            with (source/'throats.csv').open(newline='',encoding='utf8') as src, (target/'throats.csv').open('w',newline='',encoding='utf8') as dst:
                reader, writer = csv.reader(src), csv.writer(dst)
                header = next(reader)
                length_index = header.index('throat_length_m')
                writer.writerow(header+['membrane_relaxation_length_m','membrane_zdc_length_m'])
                for row in reader:
                    writer.writerow(row+[repr(length),row[length_index]])
        check = pd.read_csv(target/'throats.csv')
        pd.testing.assert_frame_equal(check[raw.columns], raw, check_exact=True)
        _, t = prior.load_pnextract(target, PARAMS)
        assert np.allclose(t.zdc_ohm, original.zdc_ohm, rtol=1e-13, atol=0)
        assert np.allclose(t.weight, original.weight, rtol=1e-13, atol=0)
        assert np.allclose(t.length_m, original.length_m, rtol=1e-13, atol=0)
        assert (t.zdc_ohm > 0).all() and (t.weight > 0).all()
        audits[name] = {'original_geometry_columns_unchanged': True, 'zdc_unchanged': True,
            'volume_weights_unchanged': True, 'internal_throats': len(t), 'source_network': str(source),
            'throats_sha256': sha(target/'throats.csv'), 'pores_sha256': sha(target/'pores.csv')}
        f = np.geomspace(1e-4, 1e6, 601)
        v = response(f, t)
        idx = int(np.argmax(v.imag))
        assert 0 < idx < len(f)-1
        optimum = minimize_scalar(lambda x: -response(10**x, t)[0].imag,
            bounds=(np.log10(f[idx-1]), np.log10(f[idx+1])), method='bounded')
        peak = 10**optimum.x
        assert optimum.success
        if length is not None:
            # 独立弛豫长度只改时间尺度；同eta单长度核满足 f_peak * L^2 = 常数。
            assert abs(peak*length**2/(4*PARAMS.diffusion_coefficient_m2_s)*2*np.pi/.9062102306683663-1) < 1e-4
        peaks.append({'case': name, 'relaxation_length_um': length*1e6 if length else np.nan,
                      'local_membrane_peak_hz': peak, 'local_peak_delta_sigma_imag_s_m': response(peak,t)[0].imag})
        dense.append(pd.DataFrame({'case': name, 'frequency_hz': f,
            'water_membrane_increment_real_s_m': v.real, 'water_membrane_increment_imag_s_m': v.imag}))
        solver.make_component_spectra_files(output_prefix=name, frequencies_hz=FREQ, network_dir=target,
            out_dir=RUN/'spectra'/name, mode='paper', params=PARAMS, solid_conductivity_s_m=.0024,
            solid_background_components={'maxwell', 'all'})
    pd.DataFrame(peaks).to_csv(RUN/'source_data/local_membrane_peaks.csv', index=False)
    pd.concat(dense).to_csv(RUN/'source_data/dense_local_membrane.csv', index=False)
    dump(RUN/'provenance/geometry_invariance.json', audits)
    # 未变化的对照复用已经收敛的真实GPU结果；逐项校验其相电导率与当前配置一致。
    reused = {}
    for mechanism in ['membrane', 'all', 'pore', 'maxwell']:
        source_sweep = PREVIOUS/'simulation_sweeps'/mechanism
        old = pd.read_csv(source_sweep/'sweep_results.csv')
        spectra = pd.read_csv(RUN/'spectra/original'/f'original_component_spectra_{mechanism}.csv')
        assert len(old)==7 and np.array_equal(old.frequency_hz, FREQ)
        assert old.converged_strict.all() and (old.relative_residual_norm <= 1e-6).all()
        phases = np.asarray([solver.phase_conductivities(row,row.frequency_hz) for _,row in spectra.iterrows()])
        for phase,index in [('water',0),('solid',1)]:
            for part in ['real','imag']:
                col = f'{phase}_sigma_{part}_s_m'
                assert np.allclose(old[col], getattr(phases[:,index],part), rtol=1e-11, atol=1e-16), (mechanism,col)
        target = RUN/'simulation_sweeps'/f'original_{mechanism}'
        shutil.copytree(source_sweep, target, dirs_exist_ok=True)
        reused[mechanism] = {'source': str(source_sweep), 'sha256': sha(source_sweep/'sweep_results.csv'),
                            'status': 'reused verified GPU control; embedded paths retain original provenance'}
    dump(RUN/'provenance/reused_controls.json', reused)
    dump(RUN/'configs/parameters_zh.json', {
        '性质': '独立有效膜弛豫长度诊断；不是CT提取出亚微米孔喉，也不是物理参数的独立验证。',
        '唯一变化': '以默认pnextract网络为共同对照，只新增独立膜弛豫长度；原始长度、面积、体积及Zdc保持不变。',
        '长度定义': 'tau=L_relax^2/(4D)，Zdc=L_real/(sigma_w*A)；两种长度明确分开。',
        '长度来源': '用户授权测试的0.3/0.5/1.0微米是假设值，低于1.92微米CT体素，不能声称CT直接观测。',
        'parameters': asdict(PARAMS), 'cases_m': CASES, 'frequencies_hz': FREQ.tolist(),
        '频点': '沿用7个LKC-16实验频率；密集曲线只用于解析局部膜谱，不增加GPU频点。',
        '场': '400×333×333二倍降采样CT、3.84微米体素、pore=0/solid=1、y方向、周期边界。非全分辨率场。',
        '机制': '膜与孔单机制固相严格为0；综合及Maxwell沿用样品16固相等效电导0.0024 S/m及介电背景。综合机制重新解场，不加总独立机制的有效电导率。',
        '固定项': 'D=1.3e-9 m2/s、eta=0.01及其它既有参数固定。所有几何缩放系数为1；无自由幅值乘子。',
        '实验约束': '同时评估综合机制实部及正虚部的log误差；保留20kHz负虚部原值并单列符号不符，不将其取绝对值拟合。',
        '求解': '膜机制GPU CSR CG/Jacobi；综合机制GPU BiCGSTAB/FFT，complex128，rtol=1e-6，atol=0。',
        '复用': reused, 'field_source': str(BASE/'configs/field_ds2.mhd'),
        'field_raw_sha256': sha(RUN/'configs/field_ds2.raw'),
        '三维与分布': '真实几何完全相同，引用已有数字岩心/孔隙网络HTML与分布；本包不重复渲染，避免将有效长度画成CT几何。',
        'geometry_package': str(prior.GEOMETRY), 'visualization_package': str(BASE),
        'command': [sys.executable, str(Path(__file__).resolve())]})
    paths = [Path(__file__),Path(prior.__file__),Path(solver.__file__),
             ROOT/'code/scripts/sip_simulation/compute_polarization_spectra.py']
    paths += [ROOT/'code/src/pore_scale_electrical'/fn for fn in ['polarization.py','ac3d_gpu.py','ac3d_pore_gpu.py','ac3d_active_domain.py']]
    for path in paths:
        shutil.copy2(path, RUN/'configs/source_snapshot'/path.name)
    dump(RUN/'provenance/source_hashes.json', {str(p): sha(p) for p in paths})
    print(pd.DataFrame(peaks).to_string(index=False), flush=True)


def run(names=None, mechanisms=('membrane','all')):
    info = solver.parse_mhd(RUN/'configs/field_ds2.mhd')
    labels = solver.load_label_volume(info)
    assert labels.shape==(400,333,333) and set(np.unique(labels))=={0,1}
    for name in CASES if names is None else names:
        for mechanism in mechanisms:
            key = f'{name}_{mechanism}'
            print('START',key,flush=True)
            is_membrane = mechanism=='membrane'
            options = dict(mechanism=mechanism,
                spectra_path=RUN/'spectra'/name/f'{name}_component_spectra_{mechanism}.csv',
                frequencies_hz=FREQ.tolist(), voxel_size_m=info['voxel_size_m'],
                out_dir=RUN/'simulation_sweeps'/key, pore_label=0, solid_label=1, direction='y',
                rtol=1e-6, atol=0., maxiter=50000 if is_membrane else 5000, dtype_name='complex128',
                preconditioner='jacobi' if is_membrane else 'fft', fft_reference='mean-face',
                progress_every=500 if is_membrane else 100, residual_every=500 if is_membrane else 100,
                iteration_sleep_s=0., zero_solid_regularization_s_m=0., resume=True,
                solver_backend='gpu-active-pore-csr' if is_membrane else 'gpu-ac3d',
                jacobi_omega=.65, sor_omega=1.35)
            serial = {k:str(v) if isinstance(v,Path) else v for k,v in options.items()}
            existing = options['out_dir']/'sweep_results.csv'
            if existing.exists():
                spectra = pd.read_csv(options['spectra_path'])
                for _,row in pd.read_csv(existing).iterrows():
                    expected = solver.phase_conductivities(solver.nearest_spectrum_row(spectra,row.frequency_hz),row.frequency_hz)
                    actual = [complex(row.water_sigma_real_s_m,row.water_sigma_imag_s_m),complex(row.solid_sigma_real_s_m,row.solid_sigma_imag_s_m)]
                    assert row.direction=='y' and row['dtype']=='complex128' and row.frequency_hz in FREQ
                    assert np.allclose(actual,expected,rtol=1e-11,atol=1e-16), 'Existing results use different parameters; choose a new RUN directory.'
            dump(RUN/'configs'/f'{key}_solver.json', serial)
            with (RUN/'provenance'/f'{key}_gpu.log').open('a',encoding='utf8') as log, redirect_stdout(log), redirect_stderr(log):
                sweep = solver.run_component_sweep(labels=labels, **options)
                assert len(sweep)==7 and sweep.converged_strict.all() and (sweep.relative_residual_norm<=1e-6).all()
                for path in (RUN/'simulation_sweeps'/key).glob('frequency_*/result.json'):
                    dump(path.parent/'config.yml',serial)
            gc.collect()
            cp.fft.config.get_plan_cache().clear()
            cp.get_default_memory_pool().free_all_blocks()
            print('DONE',key,flush=True)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    exp = pd.read_csv(RUN/'source_data/experiment_all.csv')
    selected = pd.read_csv(RUN/'source_data/selected_7_frequencies.csv')
    dense = pd.read_csv(RUN/'source_data/dense_local_membrane.csv')
    peaks = pd.read_csv(RUN/'source_data/local_membrane_peaks.csv')
    sweeps = {p.parent.name: pd.read_csv(p) for p in (RUN/'simulation_sweeps').glob('*/sweep_results.csv')}
    for key,frame in sweeps.items():
        assert len(frame)==7 and np.array_equal(frame.frequency_hz,FREQ)
        assert frame.converged_strict.all() and (frame.relative_residual_norm<=1e-6).all()
    pd.concat([f.assign(case=k) for k,f in sweeps.items()]).to_csv(RUN/'source_data/gpu7_all_results.csv',index=False)
    metrics, checks, curves = [], {}, {}
    for name in ['original', *CASES]:
        frame = sweeps[f'{name}_membrane']
        water = frame.water_sigma_real_s_m.to_numpy()+1j*frame.water_sigma_imag_s_m.to_numpy()
        eff = frame.effective_sigma_real_s_m.to_numpy()+1j*frame.effective_sigma_imag_s_m.to_numpy()
        factor = float(np.median((eff/water).real))
        _,t = prior.load_pnextract(RUN/'pore_network'/name, PARAMS)
        prediction = factor*(PARAMS.water_conductivity_s_m+response(FREQ,t))
        error = float(np.max(np.abs(prediction-eff)/np.abs(eff)))
        imag_error = float(np.max(np.abs(prediction.imag/eff.imag-1)))
        assert error<1e-5 and imag_error<1e-5
        checks[name] = {'geometry_factor':factor,'dense_vs_gpu_complex_relative_error':error,'dense_vs_gpu_imag_relative_error':imag_error}
        d = dense[dense.case==name].copy()
        d['effective_sigma_real_s_m'] = factor*(PARAMS.water_conductivity_s_m+d.water_membrane_increment_real_s_m)
        d['effective_sigma_imag_s_m'] = factor*d.water_membrane_increment_imag_s_m
        curves[name] = d
        all_frame = sweeps[f'{name}_all']
        entry = {'case':name}
        for part in ['real','imag']:
            obs = selected[f'experiment_sigma_{part}_s_m'].to_numpy()
            pred = all_frame[f'effective_sigma_{part}_s_m'].to_numpy()
            keep = obs>0
            entry[f'{part}_n_positive'] = int(keep.sum())
            entry[f'{part}_log10_rmse'] = float(np.sqrt(np.mean(np.log10(pred[keep]/obs[keep])**2)))
            entry[f'{part}_median_relative_error'] = float(np.median(np.abs(pred[keep]/obs[keep]-1)))
            entry[f'{part}_sign_mismatches'] = int((np.sign(pred)!=np.sign(obs)).sum())
        entry['joint_log10_rmse'] = float(np.sqrt((entry['real_log10_rmse']**2+entry['imag_log10_rmse']**2)/2))
        metrics.append(entry)
    pd.concat(curves.values()).to_csv(RUN/'source_data/dense_membrane_geometry_factor_model.csv',index=False)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(RUN/'source_data/combined_experiment_metrics.csv',index=False)
    dump(RUN/'provenance/verification.json',{'new_gpu_points':sum(len(f) for k,f in sweeps.items() if not k.startswith('original')),'reused_gpu_control_points':28,
        'max_new_relative_residual':max(float(f.relative_residual_norm.max()) for k,f in sweeps.items() if not k.startswith('original')),
        'geometry_factor_checks':checks,'experiment_nonpositive_imag_omitted_from_log_plot':int((exp.experiment_sigma_imag_s_m<=0).sum()),
        'curves':'Only membrane dense curves use checked geometry-factor model; all other lines connect 7 actual GPU points.'})
    plt.rcParams.update({'font.family':'Times New Roman','font.size':12,'mathtext.fontset':'stix',
        'axes.linewidth':.8,'xtick.direction':'in','ytick.direction':'in'})
    fig,axes = plt.subplots(1,2,figsize=(11,5))
    fig.subplots_adjust(left=.085,right=.98,bottom=.22,top=.80,wspace=.28)
    for name,color in zip(['original',*CASES],['#777777','#D89032','#B15A9A','#3B85B5']):
        label = 'Original throat length' if name=='original' else f'L = {CASES[name]*1e6:g} μm'
        d,f = curves[name],sweeps[f'{name}_membrane']
        keep = d.frequency_hz.between(.1,20000)
        for j,part in enumerate(['real','imag']):
            axes[j].plot(d.frequency_hz[keep],d[f'effective_sigma_{part}_s_m'][keep],color=color,lw=1.8,label=label)
            axes[j].plot(FREQ,f[f'effective_sigma_{part}_s_m'],'o',ms=4,mfc='white',mec=color)
    for j,ax in enumerate(axes):
        ax.set(xscale='log',yscale='log',xlim=(.08,25000),xlabel='Frequency (Hz)',ylabel=[r"$\sigma'$ (S/m)",r"$\sigma''$ (S/m)"][j])
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,.91),ncol=4,frameon=False)
    fig.suptitle('Sample 16 | membrane relaxation length | fixed geometry and Zdc',fontsize=14)
    fig.text(.53,.09,'Circles: 7 GPU frequencies. Curves: membrane model with verified GPU geometry factor.',ha='center',fontsize=10)
    fig.text(.53,.035,'Effective-length diagnostic; y direction; ds2 field; D = 1.3 × 10⁻⁹ m²/s; η = 0.01.',ha='center',fontsize=10)
    save(fig,'membrane_length_comparison')
    fig,axes = plt.subplots(2,3,figsize=(14,8.5),sharex=True,sharey='row')
    fig.subplots_adjust(left=.065,right=.98,bottom=.16,top=.84,wspace=.14,hspace=.20)
    for col,(name,length) in enumerate(CASES.items()):
        for row,part in enumerate(['real','imag']):
            ax=axes[row,col]
            values=exp[f'experiment_sigma_{part}_s_m'];keep=values>0
            ax.plot(exp.frequency_hz[keep],values[keep],'k.',ms=4,label='LKC-16 experiment')
            for mech,label,color,style in [('maxwell','Interfacial + dielectric','#E49C4E','--'),
                     ('pore','Pore (EDL model)','#578ECD','--'),('membrane','Membrane','#59B78D','--'),('all','All mechanisms','#ED6868','-')]:
                f=sweeps[f'{name}_{mech}' if mech in ['membrane','all'] else f'original_{mech}']
                ax.plot(FREQ,f[f'effective_sigma_{part}_s_m'],linestyle='None' if mech=='membrane' else style,color=color,lw=1.8,marker='o',ms=3,mfc='white',label=label)
                if mech=='membrane':
                    d=curves[name]; within=d.frequency_hz.between(.1,20000)
                    ax.plot(d.frequency_hz[within],d[f'effective_sigma_{part}_s_m'][within],ls='--',color=color,lw=1.2)
            ax.set(xscale='log',yscale='log',xlim=(.08,25000))
            if row==1: ax.set_xlabel('Frequency (Hz)')
            if col==0: ax.set_ylabel([r"$\sigma'$ (S/m)",r"$\sigma''$ (S/m)"][row])
        peak=float(peaks.loc[peaks.case==name,'local_membrane_peak_hz'].iloc[0])
        axes[0,col].set_title(f'L = {length*1e6:g} μm | membrane peak ≈ {peak:.0f} Hz',fontsize=12)
        axes[0,col].set_ylim(1e-4,1e-2)
        axes[1,col].set_ylim(1e-10,1e-3)
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.52,.935),ncol=5,frameon=False,fontsize=10)
    fig.suptitle('Sample 16 | independent effective membrane lengths | 7 GPU frequencies',fontsize=15,y=.98)
    fig.text(.52,.075,'All mechanisms are solved together; standalone mechanism curves are not additive.',ha='center',fontsize=11)
    fig.text(.52,.035,'Diagnostic ds2 field, y direction; η = 0.01 fixed. Nonpositive experimental imaginary points omitted on log axes.',ha='center',fontsize=10)
    save(fig,'mechanism_comparison')
    print(metric_frame.to_string(index=False),flush=True)
    (RUN/'README.md').write_text('# 样品16：独立有效膜弛豫长度诊断\n\n'
        '在原始pnextract默认网络上新增0.3、0.5、1.0 μm有效膜弛豫长度，真实孔喉长度、横截面积、体积权重及Zdc全部保持不变；所有几何缩放因子为1。有效长度是亚体素物理假设，不能称为提取到的真实小孔喉。\n\n'
        '每组重新运行7频点GPU膜机制和综合机制，共42点。默认长度及不变的孔/Maxwell机制复用已收敛GPU对照，源文件哈希和相电导一致性已核验。使用原来的400×333×333 ds2场、y方向、周期边界；不是完整CT电场验证。\n\n'
        '三个长度的局部膜谱峰位与GPU膜谱峰位一致，密集膜曲线使用经7点核验的GPU几何系数乘局部水相膜谱；峰位不是新增GPU计算点。其它机制连线仅连接7个GPU点。综合机制独立解场，不将单机制结果相加。\n\n'
        'D=1.3e-9 m²/s、eta=0.01固定；水电导率0.12 S/m、动态孔径2.45913 μm。膜/孔单机制固相为0；综合与Maxwell保留既有固相有效电导0.0024 S/m及介电背景。这项背景会使综合结果与单机制幅度不可直接加和。\n\n'
        'source_data/combined_experiment_metrics.csv同时评价7频点综合实部及正虚部，两者log误差等权。20 kHz实验虚部为负，保留在源数据并单列符号不符；log图仅过滤非正点。3组均是固定eta的敏感性结果，误差最小不等于完成物理拟合。\n\n'
        '几何没有变化，本次不重新生成三维HTML或几何分布；沿用configs/parameters_zh.json列出的原结果包。新增有效长度只用于膜模型，不能覆盖真实几何分布。\n\n'
        '完整可复跑参数、源码快照及求解日志分别在configs与provenance；逐频数值结果、残差历史和配置在simulation_sweeps；600 dpi图及预览在figures。\n',encoding='utf8')


def save(fig,name):
    import matplotlib.pyplot as plt
    fig.savefig(RUN/'figures'/f'{name}.png',dpi=600)
    fig.savefig(RUN/'figures'/f'{name}_preview.png',dpi=150)
    plt.close(fig)


def calibrate():
    """用零膜/1%膜两组场结果估计强度，再以独立GPU扫频核验候选。"""
    shutil.copy2(__file__,RUN/'configs/source_snapshot/calibration_driver.py')
    dump(RUN/'provenance/calibration_source.json',{'source':str(Path(__file__).resolve()),'sha256':sha(Path(__file__)),
        'command':[sys.executable,str(Path(__file__).resolve()),'--calibrate-only']})
    for mechanism in ['membrane','pore','maxwell','all']:
        command = json.loads((PREVIOUS/'configs'/f'{mechanism}_command.json').read_text(encoding='utf-8-sig'))
        dump(RUN/'configs'/f'reused_{mechanism}_command.json',command)
        for path in (RUN/'simulation_sweeps'/f'original_{mechanism}').glob('frequency_*/result.json'):
            dump(path.parent/'config.yml',{'reused_control':True,'original_command':command})
    exp = pd.read_csv(RUN/'source_data/selected_7_frequencies.csv')
    obs = exp.experiment_sigma_real_s_m.to_numpy()+1j*exp.experiment_sigma_imag_s_m.to_numpy()
    positive = obs.imag>0
    def complex_values(frame, prefix):
        return frame[f'{prefix}_sigma_real_s_m'].to_numpy()+1j*frame[f'{prefix}_sigma_imag_s_m'].to_numpy()
    def loss(pred):
        if (pred.real<=0).any() or (pred.imag[positive]<=0).any():
            return np.inf
        return float((np.mean(np.log10(pred.real/obs.real)**2)+np.mean(np.log10(pred.imag[positive]/obs.imag[positive])**2))/2)
    zero = replace(PARAMS,membrane_polarizability=0.)
    solver.make_component_spectra_files(output_prefix='eta_zero',frequencies_hz=FREQ,
        network_dir=RUN/'pore_network/original',out_dir=RUN/'spectra/eta_zero',mode='paper',params=zero,
        solid_conductivity_s_m=.0024,solid_background_components={'maxwell','all'})
    run(['eta_zero'], mechanisms=('all',))
    zero_frame = pd.read_csv(RUN/'simulation_sweeps/eta_zero_all/sweep_results.csv')
    eff0 = complex_values(zero_frame,'effective')
    candidates = []
    predictions = {}
    for name in CASES:
        fixed = pd.read_csv(RUN/'simulation_sweeps'/f'{name}_all/sweep_results.csv')
        eff1 = complex_values(fixed,'effective')
        _,t = prior.load_pnextract(RUN/'pore_network'/name,PARAMS)
        def water_increment(eta):
            params = replace(PARAMS,membrane_polarizability=float(eta))
            return prior.upscale_conductance_to_water_conductivity(prior.membrane_polarization_conductance(
                FREQ,t.membrane_relaxation_length_m.to_numpy(),t.weight.to_numpy(),t.zdc_ohm.to_numpy(),params),params)
        water1 = water_increment(.01)
        # ponytail: 一阶复响应只用于挑选eta；最终结果必须实际GPU复算，不能把近似当拟合完成。
        def estimate(eta):
            return eff0+(eff1-eff0)*water_increment(eta)/water1
        fit = minimize_scalar(lambda x:loss(estimate(10**x)),bounds=(-5,-2),method='bounded')
        assert fit.success
        eta = float(10**fit.x)
        predictions[name] = estimate(eta)
        candidates.append({'case':name,'eta_estimated':eta,'estimated_joint_log10_rmse':float(np.sqrt(fit.fun)),
                           'fixed_eta_joint_log10_rmse':float(np.sqrt(loss(eff1)))})
    ranking = pd.DataFrame(candidates).sort_values('estimated_joint_log10_rmse')
    ranking.to_csv(RUN/'source_data/eta_candidate_estimates.csv',index=False)
    winner = ranking.iloc[0]
    name, eta = str(winner['case']), float(winner.eta_estimated)
    params = replace(PARAMS,membrane_polarizability=eta)
    calibrated_name = 'calibrated_'+name
    solver.make_component_spectra_files(output_prefix=calibrated_name,frequencies_hz=FREQ,
        network_dir=RUN/'pore_network'/name,out_dir=RUN/'spectra'/calibrated_name,mode='paper',params=params,
        solid_conductivity_s_m=.0024,solid_background_components={'maxwell','all'})
    config = {'性质':'同一7点实验上的条件强度标定，不是独立验证或严格全局最优。',
        'length_case':name,'relaxation_length_m':CASES[name],'parameters':asdict(params),
        '方法':'零膜与eta=0.01综合GPU场响应构成一阶复响应估计；在1e-5至0.01内对eta做实部/正虚部log均方误差等权最小化，选择三长度中的最低估计误差，随后实际GPU复算。',
        '参数界限':'eta界限为本轮减弱强度诊断搜索范围，不是实测置信区间。',
        '验证边界':'候选估计只用于选参数；展示采用实际GPU结果。负虚部不进入log损失，符号不符单列。',
        '固定项':'保持原几何、Zdc、D、孔模型、介电背景、固相电导、CT场和方向不变。'}
    dump(RUN/'configs/calibration_parameters_zh.json',config)
    print('CALIBRATION CANDIDATE',name,'eta',eta,flush=True)
    run([calibrated_name])
    actual_frame = pd.read_csv(RUN/'simulation_sweeps'/f'{calibrated_name}_all/sweep_results.csv')
    actual = complex_values(actual_frame,'effective')
    fixed_frame = pd.read_csv(RUN/'simulation_sweeps'/f'{name}_all/sweep_results.csv')
    fixed = complex_values(fixed_frame,'effective')
    predicted = predictions[name]
    audit = {'selected_case':name,'eta':eta,'joint_log10_rmse_fixed':float(np.sqrt(loss(fixed))),
        'joint_log10_rmse_calibrated_actual_gpu':float(np.sqrt(loss(actual))),
        'joint_log10_rmse_no_membrane':float(np.sqrt(loss(eff0))),
        'estimate_vs_gpu_max_relative_real_error':float(np.max(np.abs(predicted.real/actual.real-1))),
        'estimate_vs_gpu_max_relative_imag_error':float(np.max(np.abs(predicted.imag/actual.imag-1))),
        'imag_sign_mismatches':int((np.sign(actual.imag)!=np.sign(obs.imag)).sum()),
        'calibration_improves_joint_error':bool(loss(actual)<loss(fixed)),
        'limitation':'One-step conditional calibration; estimated rankings are not certified exact optima.'}
    assert audit['calibration_improves_joint_error']
    comparison = exp.copy()
    for key,values in [('eta_zero',eff0),('fixed_eta',fixed),('eta_estimate',predicted),('calibrated_gpu',actual)]:
        comparison[key+'_real_s_m']=values.real
        comparison[key+'_imag_s_m']=values.imag
    comparison.to_csv(RUN/'source_data/calibration_gpu_comparison.csv',index=False)
    dump(RUN/'provenance/calibration_verification.json',audit)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Times New Roman','font.size':12,'mathtext.fontset':'stix',
        'axes.linewidth':.8,'xtick.direction':'in','ytick.direction':'in'})
    full_exp = pd.read_csv(RUN/'source_data/experiment_all.csv')
    fig,axes = plt.subplots(1,2,figsize=(11,5))
    fig.subplots_adjust(left=.085,right=.98,bottom=.22,top=.79,wspace=.28)
    for j,part in enumerate(['real','imag']):
        ax=axes[j]
        values=full_exp[f'experiment_sigma_{part}_s_m'];keep=values>0
        ax.plot(full_exp.frequency_hz[keep],values[keep],'k.',ms=4,label='LKC-16 experiment')
        for label,curve,color,style in [('η = 0 (control)',eff0,'#999999','--'),('η = 0.01',fixed,'#D79542','--'),(f'η = {eta:.4g}',actual,'#DB6268','-')]:
            ax.plot(FREQ,getattr(curve,part),style,color=color,marker='o',ms=4,mfc='white',lw=1.8,label=label)
        ax.set(xscale='log',yscale='log',xlim=(.08,25000),xlabel='Frequency (Hz)',ylabel=[r"$\sigma'$ (S/m)",r"$\sigma''$ (S/m)"][j])
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,.91),ncol=4,frameon=False,fontsize=11)
    fig.suptitle(f'Sample 16 | all mechanisms | L = {CASES[name]*1e6:g} μm | conditional strength calibration',fontsize=13)
    fig.text(.53,.09,'Circles: actual GPU results at 7 frequencies; lines connect computed points.',ha='center',fontsize=11)
    fig.text(.53,.035,'Real and positive imaginary data used jointly; same-data calibration, not independent validation.',ha='center',fontsize=10)
    save(fig,'combined_strength_calibration')
    # 同背景逐步加入机制的差值可严格相加，但依赖加入顺序，不等同于独立机制谱。
    background = complex_values(pd.read_csv(RUN/'simulation_sweeps/original_maxwell/sweep_results.csv'),'effective')
    pore_increment, membrane_increment = eff0-background, actual-eff0
    assert np.allclose(background+pore_increment+membrane_increment,actual,rtol=1e-13,atol=1e-16)
    contributions = pd.DataFrame({'frequency_hz':FREQ})
    for key,values in [('interfacial_dielectric',background),('pore_conditional',pore_increment),('membrane_conditional',membrane_increment),('all',actual)]:
        contributions[key+'_real_s_m']=values.real
        contributions[key+'_imag_s_m']=values.imag
    contributions.to_csv(RUN/'source_data/calibrated_conditional_components.csv',index=False)
    fig,axes = plt.subplots(2,1,figsize=(7.5,8.3),sharex=True)
    fig.subplots_adjust(left=.14,right=.97,bottom=.16,top=.93,hspace=.19)
    for j,part in enumerate(['real','imag']):
        ax=axes[j]
        values=full_exp[f'experiment_sigma_{part}_s_m'];keep=values>0
        ax.plot(full_exp.frequency_hz[keep],values[keep],'kv' if j==0 else 'k.',ms=4,label='LKC-16 experiment')
        if j==1:
            for label,values,color in [('Interfacial + dielectric',background,'#E49C4E'),('Pore increment',pore_increment,'#578ECD'),('Membrane increment',membrane_increment,'#59B78D')]:
                keep=values.imag>0
                ax.plot(FREQ[keep],values.imag[keep],'--o',color=color,mfc='white',ms=3,lw=1.6,label=label)
        ax.plot(FREQ,getattr(actual,part),'-o',color='#ED6868',mfc='white',ms=4,lw=2,label='All mechanisms')
        ax.set(xscale='log',yscale='log',xlim=(.08,25000),ylabel=[r"$\sigma'$ (S/m)",r"$\sigma''$ (S/m)"][j])
        ax.legend(frameon=False,fontsize=10,loc='upper left')
    axes[0].set_ylim(1e-4,1e-1)
    axes[1].set_ylim(1e-7,1e-3)
    axes[1].set_xlabel('Frequency (Hz)')
    fig.suptitle(f'Sample 16 | L = {CASES[name]*1e6:g} μm | η = {eta:.4g}',fontsize=15,y=.98)
    fig.text(.14,.065,'Conditional increments: add pore to background, then membrane.\nAll curves use 7 GPU points; lines are guides. Effective-length calibration.',fontsize=10)
    save(fig,'calibrated_reference_style')
    dump(RUN/'provenance/conditional_components.json',{'definition':'background=maxwell; pore increment=all(eta=0)-background; membrane increment=calibrated all-all(eta=0)',
        'order':'background -> pore -> membrane; order-dependent, not standalone mechanism spectra',
        'exact_sum_verified':True,'pore_nonpositive_imag':int((pore_increment.imag<=0).sum()),
        'membrane_nonpositive_imag':int((membrane_increment.imag<=0).sum())})
    all_new = [pd.read_csv(p) for p in (RUN/'simulation_sweeps').glob('*/sweep_results.csv') if not p.parent.name.startswith('original')]
    assert sum(map(len,all_new))==63 and all(f.converged_strict.all() for f in all_new)
    pd.concat([pd.read_csv(p).assign(case=p.parent.name) for p in (RUN/'simulation_sweeps').glob('*/sweep_results.csv')]).to_csv(RUN/'source_data/gpu7_all_results.csv',index=False)
    final_check = json.loads((RUN/'provenance/verification.json').read_text(encoding='utf8'))
    final_check.update(new_gpu_points=63,max_new_relative_residual=max(float(f.relative_residual_norm.max()) for f in all_new),
        calibration=audit,related_cpu_tests_passed=9)
    dump(RUN/'provenance/verification.json',final_check)
    with (RUN/'README.md').open('a',encoding='utf8') as out:
        out.write(f'\n## 强度条件标定\n\n补充零膜对照7点及候选膜/综合14点，新增GPU总计63点。用零膜和eta=0.01两端综合响应估计较低eta，得到L={CASES[name]*1e6:g} μm、eta={eta:.8g}，然后实际GPU复算。联合log10 RMSE由{audit["joint_log10_rmse_fixed"]:.4f}降至{audit["joint_log10_rmse_calibrated_actual_gpu"]:.4f}；近似估计的最大虚部相对偏差{audit["estimate_vs_gpu_max_relative_imag_error"]:.2%}。这是一轮条件标定，未经独立数据验证，不能宣称严格最优。20 kHz实验负虚部仍不能由当前正虚部模型重现。\n\ncalibrated_reference_style图采用同一固相背景的条件差值：孔增量=关闭膜的综合结果−Maxwell，膜增量=标定综合结果−关闭膜的综合结果；三项相加严格等于综合结果，但依赖加入顺序。这与mechanism_comparison图中独立膜/孔机制（固相为0）的含义不同。\n')
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':
    if '--calibrate-only' in sys.argv:
        calibrate()
    elif '--plot-only' in sys.argv:
        plot()
    else:
        prepare()
        run()
        plot()
        calibrate()
