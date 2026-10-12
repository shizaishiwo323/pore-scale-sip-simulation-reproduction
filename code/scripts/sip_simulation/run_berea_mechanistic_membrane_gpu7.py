"""Berea: EDL-constrained eta_j + original Niu/Titov kernel (hybrid sensitivity)."""
from pathlib import Path
from dataclasses import asdict
import sys
import json
import shutil
import hashlib
import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.special import i0e

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/src'))
from compute_polarization_spectra import load_pnextract
from make_polarization_component_spectra import make_component_spectrum
from pore_scale_electrical.polarization import (PolarizationParameters,
    membrane_polarization_conductance, upscale_conductance_to_water_conductivity)

BASE = ROOT / 'results/niu2020_berea_gpu7_20260912'
RUN = ROOT / 'results/niu2020_berea_mechanistic_membrane_gpu7_20260914'
PARAMS = PolarizationParameters()
FREQ = np.array([.002, .2, 20, 1000, 45000, 10111102.998, 859729899.463])
T, ZETA, FQ = 298.15, -.075, .6
DP, DN = 1.334e-9, 2.032e-9
KB, E, F, R = 1.380649e-23, 1.602176634e-19, 96485.33212, 8.314462618
C0 = PARAMS.water_conductivity_s_m * R * T / (F**2 * (DP + DN))
DEBYE = np.sqrt(80 * 8.8541878128e-12 * KB * T / (2*C0*E*F))


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf8')


def concentrations(radius, zeta=ZETA, order=96):
    """BH2013a eq18-19; integrate EDL excess in boundary-layer coordinates."""
    radius = np.atleast_1d(np.asarray(radius, float))
    if not np.all(np.isfinite(radius) & (radius > 0)):
        raise ValueError('Radii must be finite and positive')
    nodes, weights = leggauss(order)
    kr = radius[:, None] / DEBYE
    extent = np.minimum(kr, 40.)
    distance = (nodes+1)/2 * extent
    local = kr-distance
    ratio = i0e(local)/i0e(kr)*np.exp(-distance)
    exponent = E*zeta*ratio/(KB*T)
    area_weight = 2*local/kr**2 * weights/2 * extent
    return (1+np.sum(np.expm1(-exponent)*area_weight, axis=1),
            1+np.sum(np.expm1(exponent)*area_weight, axis=1))


def pair_eta(r1, l1, r2, l2, zeta=ZETA):
    """BH2013a common-area averaging; BH2013b eq17-19 DC chargeability."""
    r1, l1, r2, l2 = np.broadcast_arrays(r1, l1, r2, l2)
    if not np.all(np.isfinite(l1) & np.isfinite(l2) & (l1 > 0) & (l2 > 0)):
        raise ValueError('Pair lengths must be finite and positive')
    p1, n1 = concentrations(r1, zeta)
    p2, n2 = concentrations(r2, zeta)
    # Stern surplus correction precedes normalization to the common larger area.
    common = np.maximum(r1, r2)**2
    a1, a2 = r1**2/common, r2**2/common
    dp1, dn1 = DP*(p1-FQ)/(1-FQ)*a1, DN*n1*a1
    dp2, dn2 = DP*(p2-FQ)/(1-FQ)*a2, DN*n2*a2
    tp1, tp2 = dp1/(dp1+dn1), dp2/(dp2+dn2)
    tau1, tau2 = l1**2/(8*dp1*(1-tp1)), l2**2/(8*dp2*(1-tp2))
    passive = l1/(dp1+dn1)+l2/(dp2+dn2)
    polar = 8*(tp1-tp2)**2/(l1/tau1+l2/tau2)
    return polar/(passive+polar)


def check():
    r = np.geomspace(1e-9, 1e-3, 50)
    for a,b in zip(concentrations(r), concentrations(r, order=192)):
        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12)
    assert pair_eta(10e-6,10e-6,1e-6,5e-6,zeta=0)[0] < 1e-25
    assert pair_eta(1e-6,10e-6,1e-6,5e-6)[0] < 1e-25
    eta = pair_eta(r, np.ones(50)*1e-5, r/2, np.ones(50)*2e-6)
    assert np.all(np.isfinite(eta) & (eta >= 0) & (eta < 1))
    np.testing.assert_allclose(eta, pair_eta(r/2,np.ones(50)*2e-6,r,np.ones(50)*1e-5))
    return {'zero_zeta':True, 'equal_radii':True, 'pair_symmetry':True,
            'quadrature_96_vs_192_rtol':1e-10, 'eta_bounds':True}


def prepare():
    for folder in ('configs','source_data','provenance','figures','pore_network','digital_rock','spectra'):
        (RUN/folder).mkdir(parents=True, exist_ok=True)
    paths = ['pore_network/network_parsed/pores.csv','pore_network/network_parsed/throats.csv',
             'pore_network/berea.html','pore_network/berea.json','pore_network/distribution.png',
             'pore_network/distribution.json','pore_network/geometry_comparison.csv',
             'digital_rock/berea.html','digital_rock/berea.json',
             'source_data/experiment_imag_permittivity.csv']
    reused = {}
    for relative in paths:
        src, dst = BASE/relative, RUN/relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src,dst)
        with src.open('rb') as stream:
            reused[relative] = {'source':str(src),'sha256':hashlib.file_digest(stream,'sha256').hexdigest()}
    pores = pd.read_csv(RUN/'pore_network/network_parsed/pores.csv').set_index('pore_id')
    raw = pd.read_csv(RUN/'pore_network/network_parsed/throats.csv')
    raw = raw.loc[(raw.pore1_id>=1)&(raw.pore2_id>=1)].copy()
    rt, lt = raw.throat_radius_m.to_numpy(), raw.throat_length_m.to_numpy()
    weights, etas = [], []
    for endpoint in (1,2):
        rp = raw[f'pore{endpoint}_id'].map(pores.pore_radius_m).to_numpy()
        lp = raw[f'pore{endpoint}_length_m'].to_numpy()
        good = lp>0
        eta = np.zeros(len(raw))
        eta[good] = pair_eta(rp[good],lp[good],rt[good],lt[good])
        raw[f'eta_endpoint{endpoint}'] = eta
        raw[f'pore{endpoint}_radius_m'] = rp
        weights.append(np.where(good,lp,0))
        etas.append(eta)
    weight = np.sum(weights,axis=0)
    # ponytail: local pair length-weighting is a network closure, not a 3-section PNP solve.
    eta = np.divide(np.sum(np.array(weights)*etas,axis=0),weight,
                    out=np.zeros(len(raw)),where=weight>0)
    raw['membrane_polarizability'] = eta
    raw.to_csv(RUN/'source_data/throat_mechanistic_polarizability.csv',index=False)
    _, throat = load_pnextract(RUN/'pore_network/network_parsed', PARAMS)
    np.testing.assert_array_equal(throat.length_m,raw.throat_length_m)
    delta = upscale_conductance_to_water_conductivity(membrane_polarization_conductance(
        FREQ,throat.length_m.to_numpy(),throat.weight.to_numpy(),throat.zdc_ohm.to_numpy(),
        PARAMS,polarizability=eta),PARAMS)
    spectrum = pd.DataFrame({'frequency_hz':FREQ,'omega_rad_s':2*np.pi*FREQ,
        'delta_sigma_membrane_real_s_m':delta.real,'delta_sigma_membrane_imag_s_m':delta.imag})
    component = make_component_spectrum(spectrum,'membrane',PARAMS,'paper')
    component.to_csv(RUN/'spectra/membrane.csv',index=False)
    stats = {'count':len(eta),'min':float(eta.min()),'median':float(np.median(eta)),
             'max':float(eta.max()),'volume_weighted_mean':float(np.average(eta,weights=raw.throat_volume_m3)),
             'both_endpoint_lengths_zero':int((weight==0).sum())}
    dump(RUN/'configs/parameters_zh.json',{
        '性质':'Berea机制改进敏感性复算：BH逐喉eta + Niu/Titov原阻抗核；非完整BH阻抗模型，也非实验拟合。',
        '几何':'复用原始pnextract默认网络；圆柱EDL使用提取半径，共同面积归一化；Niu Zdc继续用原半径/shape factor面积及原喉长，不缩放。',
        '网络闭合':'每条喉与两端孔段分别组成一对，eta按原孔段长度加权；零长度端不参与。此平均为建模假设。',
        '电化学':'zeta=-75 mV、fQ=0.6为BH2013a图7示例；25C、NaCl扩散系数为假设。浓度由0.043 S/m的Nernst-Einstein关系推算，非实测盐度。线性圆柱电势+Boltzmann浓度沿用BH近似，非非线性PB验证。',
        '物理边界':'水相=0.043+膜增量，固相严格0；不加入其他机制。保留Niu tau=L_geom^2/(4D)，D=1.3e-9；并非BH几何相关弛豫时间。',
        '数值':'350^3原分辨率，2.8um，x方向周期边界；GPU active-pore CSR CG归一化/Jacobi，rtol=1e-6，无warm start。',
        'polarization_parameters':asdict(PARAMS),'temperature_k':T,'zeta_v':ZETA,'stern_fraction':FQ,
        'cation_diffusion_m2_s':DP,'anion_diffusion_m2_s':DN,'equivalent_nacl_mol_m3':C0,
        'debye_length_m':DEBYE,'frequencies_hz':FREQ.tolist(),'eta_summary':stats,
        'sources':['https://doi.org/10.1093/gji/ggt136','https://doi.org/10.1190/GEO2012-0548.1']})
    dump(RUN/'provenance/reused_geometry.json',reused)
    dump(RUN/'provenance/model_checks.json',check())
    print(json.dumps(stats,indent=2),flush=True)


def run():
    import tifffile
    import run_sample89_3d_sip_comparison as solver
    labels = tifffile.imread(BASE/'digital_rock/niu2020_berea_solid255_pore0.tiff')
    assert labels.shape==(350,350,350) and set(np.unique(labels))=={0,255}
    options = dict(mechanism='membrane',spectra_path=RUN/'spectra/membrane.csv',
        frequencies_hz=FREQ.tolist(),voxel_size_m=2.8e-6,out_dir=RUN/'simulation_sweeps/membrane',
        pore_label=0,solid_label=255,direction='x',rtol=1e-6,atol=0.,maxiter=50000,
        dtype_name='complex128',preconditioner='jacobi',fft_reference='mean-face',progress_every=500,
        residual_every=100,iteration_sleep_s=0.,zero_solid_regularization_s_m=0.,resume=False,
        solver_backend='gpu-active-pore-csr',jacobi_omega=.65,sor_omega=1.35,disable_warm_start=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in options.items()}
    dump(RUN/'configs/solver.json',config)
    sweep=solver.run_component_sweep(labels=labels,**options)
    assert len(sweep)==7 and sweep.converged_strict.all() and (sweep.relative_residual_norm<=1e-6).all()
    for result in (RUN/'simulation_sweeps/membrane').glob('frequency_*/result.json'):
        dump(result.parent/'config.yml',config)


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    new = pd.read_csv(RUN/'simulation_sweeps/membrane/sweep_results.csv').sort_values('frequency_hz')
    old_path = BASE/'simulation_sweeps/membrane/sweep_results.csv'
    old = pd.read_csv(old_path).sort_values('frequency_hz')
    exp = pd.read_csv(RUN/'source_data/experiment_imag_permittivity.csv')
    assert len(new)==len(old)==7 and new.converged_strict.all() and old.true_residual_passed.all()
    np.testing.assert_allclose(new.frequency_hz,FREQ,rtol=0,atol=0)
    np.testing.assert_allclose(old.frequency_hz,FREQ,rtol=0,atol=0)
    _, throat=load_pnextract(RUN/'pore_network/network_parsed',PARAMS)
    fixed_water=PARAMS.water_conductivity_s_m+upscale_conductance_to_water_conductivity(
        membrane_polarization_conductance(FREQ,throat.length_m.to_numpy(),throat.weight.to_numpy(),
                                         throat.zdc_ohm.to_numpy(),PARAMS),PARAMS)
    np.testing.assert_allclose(fixed_water,old.water_sigma_real_s_m+1j*old.water_sigma_imag_s_m,rtol=1e-12)
    assert (new.n_cells==350**3).all() and (new.solid_sigma_real_s_m==0).all() and (new.solid_sigma_imag_s_m==0).all()
    # Confirm solver change leaves the geometric response unchanged at the old tolerance.
    factors = []
    for frame in (old,new):
        sigma = frame.effective_sigma_real_s_m.to_numpy()+1j*frame.effective_sigma_imag_s_m.to_numpy()
        water = frame.water_sigma_real_s_m.to_numpy()+1j*frame.water_sigma_imag_s_m.to_numpy()
        factors.append(sigma/water)
        frame['epsilon_relative'] = frame.effective_sigma_imag_s_m/(2*np.pi*frame.frequency_hz*PARAMS.epsilon_0_f_m)
    np.testing.assert_allclose(factors[0],factors[1],rtol=2e-5,atol=0)
    combined=pd.concat([old.assign(case='fixed_eta_0.01_reused'),new.assign(case='mechanistic_eta_j_new_gpu')])
    combined.to_csv(RUN/'source_data/membrane_before_after.csv',index=False)
    plt.rcParams.update({'font.family':'Arial','font.size':9,'legend.frameon':False,
                         'xtick.direction':'in','ytick.direction':'in'})
    for reference in (True,False):
        fig,axes=plt.subplots(2,1,figsize=(7.2,7.5),layout='constrained')
        for i,(ax,key,obs,ylabel) in enumerate(zip(axes,
                ['epsilon_relative','effective_sigma_imag_s_m'],
                ['epsilon_relative','sigma_imag_s_m'],
                [r'Relative permittivity $\epsilon^\prime_{\mathrm{eff}}/\epsilon_0$',
                 r'Imaginary conductivity $\sigma^{\prime\prime}_{\mathrm{eff}}$ (S m$^{-1}$)'])):
            ax.loglog(exp.frequency_hz,exp[obs],'^',mfc='none',mec='#333333',ms=3.7,mew=.65,
                      label='Experiment (total response)')
            ax.loglog(FREQ,old[key],'s--',color='#C65A37',ms=4,lw=1.5,label=r'Membrane: fixed $\eta=1\%$')
            ax.loglog(FREQ,new[key],'o-',color='#007F86',ms=4,lw=1.7,label=r'Membrane: mechanistic $\eta_j$')
            ax.set(xlabel='Frequency (Hz)',ylabel=ylabel)
            ax.text(-.1,1.02,'(a)' if i==0 else '(b)',transform=ax.transAxes,fontsize=13)
            if reference:
                ax.set(xlim=(1e-3 if i==0 else 1e-4,1e10),ylim=(1,1e10) if i==0 else (1e-7,7),
                       xticks=[1e-3,1,1e3,1e6,1e9],yticks=[1,1e3,1e6,1e9] if i==0 else [1e-6,1e-3,1])
            else:
                ax.set(xlim=(1e-3,1e10))
                values=np.r_[new[key],old[key],exp[obs]]
                ax.set_ylim(10**np.floor(np.log10(values[values>0].min())),
                            10**np.ceil(np.log10(values.max())))
                ax.grid(which='major',color='#e6e6e6',lw=.5)
        axes[0].legend(loc='upper right',fontsize=8)
        if reference:
            axes[1].text(.03,.08,'Mechanistic membrane: all 7 points below this axis range',
                         color='#007F86',transform=axes[1].transAxes,fontsize=8)
        fig.get_layout_engine().set(rect=(0,.045,1,.94))
        fig.text(.5,.013,'350 x 350 x 350; x direction; 7 GPU points; lines connect sampled frequencies.',
                 ha='center',fontsize=7)
        name='membrane_mechanistic_reference_axes' if reference else 'membrane_mechanistic_full_range'
        fig.savefig(RUN/'figures'/f'{name}.png',dpi=600)
        plt.close(fig)
    eta_frame=pd.read_csv(RUN/'source_data/throat_mechanistic_polarizability.csv')
    params=json.loads((RUN/'configs/parameters_zh.json').read_text(encoding='utf8'))
    ratio=(new.effective_sigma_imag_s_m.to_numpy()/old.effective_sigma_imag_s_m.to_numpy())
    verification={'new_gpu_points':7,'all_strictly_converged':True,
        'max_true_relative_residual':float(new.relative_residual_norm.max()),
        'new_to_old_imag_ratio':ratio.tolist(),'eta_summary':params['eta_summary'],
        'new_geometric_factor':float(factors[1][0].real),
        'max_relative_geometric_difference':float(np.max(np.abs(factors[0]/factors[1]-1))),
        'reference_axes':{'top':{'x':[1e-3,1e10],'y':[1,1e10]},'bottom':{'x':[1e-4,1e10],'y':[1e-7,7]}},
        'png_dpi':600,'experiment_boundary':'Total response, not membrane-only experimental data.',
        'mechanism_scope':'Only membrane recalculated; all/pore/interfacial not recomputed or relabeled.'}
    verification['throats_wider_than_at_least_one_pore']=int(((eta_frame.throat_radius_m>eta_frame.pore1_radius_m)|
                                                           (eta_frame.throat_radius_m>eta_frame.pore2_radius_m)).sum())
    verification['wide_throat_handling']='Both sections normalized to max radius; symmetric pair eta formula; no geometry modified or discarded.'
    dump(RUN/'provenance/verification.json',verification)
    paths=[Path(__file__),old_path,ROOT/'code/src/pore_scale_electrical/ac3d_pore_gpu.py',
        ROOT/'code/src/pore_scale_electrical/ac3d_active_domain.py',ROOT/'code/src/pore_scale_electrical/polarization.py',
        ROOT/'code/scripts/sip_simulation/run_sample89_3d_sip_comparison.py',
        ROOT/'code/scripts/sip_simulation/compute_polarization_spectra.py',
        ROOT/'code/scripts/sip_simulation/make_polarization_component_spectra.py']
    hashes={}
    for path in paths:
        dst=RUN/'configs/source_snapshot'/path.name
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dst)
        with path.open('rb') as stream:
            hashes[str(path)] = hashlib.file_digest(stream,'sha256').hexdigest()
    dump(RUN/'provenance/source_hashes.json',hashes)
    summary=params['eta_summary']
    (RUN/'README.md').write_text(
        '# Berea 膜极化：mechanistic improvement\n\n'
        f"3666条内部孔喉，eta体积加权均值={summary['volume_weighted_mean']:.8g}，"
        f"中位数={summary['median']:.8g}，范围={summary['min']:.8g}至{summary['max']:.8g}。"
        '这些是逐喉极化率，体积均值不是等价的体有效极化率。\n\n'
        '原固定eta=0.01膜机制结果作为基线复用；新膜机制在350³、2.8um、x方向周期场上运行7个GPU点。'
        f"全部达到rtol=1e-6，最大真实相对残差={verification['max_true_relative_residual']:.3e}。"
        f"新旧几何因子最大相对差={verification['max_relative_geometric_difference']:.3e}，"
        '该差异约0.0018%，对响应降低数千倍的结论没有实质影响；残差阈值不等同于解误差界。逐频result.json、残差和config.yml在simulation_sweeps/membrane。\n\n'
        f"虚部新/旧比值（按升序频率）={ratio.tolist()}。\n\n"
        '图membrane_mechanistic_reference_axes.png沿用原参考图轴范围；'
        'membrane_mechanistic_full_range.png扩展纵轴以显示全部7点。实验是综合响应，不能用独立膜机制与实验的差值判断全部模型拟合。'
        '没有把新膜机制直接加到旧all曲线上。\n\n'
        '模型：BH2013a式15、18、19、21给出EDL截面浓度与Stern修正，按共同较大截面积归一化，'
        'BH2013b式17–19给出eta=P/(A+P)。每条喉与两端孔段各构成一对，再按孔段长度平均eta（网络闭合假设）。'
        '原始几何、shape-factor面积、Zdc、体积权重以及Niu tau=L²/(4D)全部保留。'
        '因此这是BH约束eta与Niu/Titov频谱核的混合敏感性模型，不是完整BH阻抗或非线性PNP求解。\n\n'
        f'电化学假设：25C，zeta=-75mV，fQ=0.6；NaCl等效浓度{C0:.5f} mol/m³，Debye长度{DEBYE*1e9:.4f} nm。'
        'zeta与fQ取自BH2013a图7示例，非Berea实测；盐组成、pH、表面电位的不确定性仍存在。'
        '圆柱电势采用BH线性近似，强表面电势下不能视为非线性PB精确解。\n\n'
        '复用几何HTML和分布图，来源与哈希见provenance/reused_geometry.json。'
        '模型零zeta、等半径、交换对称性、积分收敛检查见provenance/model_checks.json。\n\n'
        '文献：https://doi.org/10.1093/gji/ggt136 ; https://doi.org/10.1190/GEO2012-0548.1\n\n'
        '复跑：设置PYTHONNOUSERSITE=1，以conda ml运行code/scripts/sip_simulation/run_berea_mechanistic_membrane_gpu7.py。'
        '仅重绘追加--plot-only。脚本复跑会重新计算；如更改参数，须另存运行目录。\n',encoding='utf8')
    print(json.dumps(verification,indent=2),flush=True)


if __name__=='__main__':
    if '--check' in sys.argv:
        print(check())
    elif '--plot-only' in sys.argv:
        plot()
    else:
        prepare()
        run()
        plot()
