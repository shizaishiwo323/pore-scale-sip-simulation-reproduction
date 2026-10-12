"""Invert measured complex SIP into conditional effective geometry distributions.

CSV columns: frequency_hz, sigma_real_s_m, sigma_imag_s_m.
Only --niu-workbook reads Figure6's measured complex-conductivity columns.
"""
import argparse
from dataclasses import asdict, fields, replace
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'code/src'))
from pore_scale_electrical.sip_geometry_inversion import GeometryKernel, InversionSettings, validate_data
from pore_scale_electrical.polarization import PolarizationParameters


def quantiles(grid, mass):
    loggrid = np.log(grid)
    edges = np.r_[1.5*loggrid[0]-.5*loggrid[1],(loggrid[:-1]+loggrid[1:])/2,1.5*loggrid[-1]-.5*loggrid[-2]]
    return np.exp(np.interp([.1,.5,.9],np.r_[0,np.cumsum(mass)/mass.sum()],edges))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input',type=Path)
    source.add_argument('--niu-workbook',type=Path)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/niu2020_sip_geometry_inversion.json')
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    kwargs = {f.name:cfg[f.name] for f in fields(InversionSettings) if f.name in cfg and f.name != 'parameters'}
    for key in ('radius_range_m','length_range_m','area_range_m2','epsilon_range','dc_range_s_m'):
        if key in kwargs: kwargs[key] = tuple(kwargs[key])
    settings = InversionSettings(parameters=PolarizationParameters(**cfg.get('parameters',{})),**kwargs)
    src = args.input or args.niu_workbook
    if args.input:
        data = pd.read_csv(src)[['frequency_hz','sigma_real_s_m','sigma_imag_s_m']]
    else:
        raw = pd.read_excel(src,header=None)
        data = raw.iloc[2:,:3].copy()
        data.columns = ['frequency_hz','sigma_real_s_m','sigma_imag_s_m']
        data = data.dropna(how='all')
    data = data.apply(pd.to_numeric,errors='raise').sort_values('frequency_hz').reset_index(drop=True)
    validate_data(*data.to_numpy().T)
    keep = data.frequency_hz.between(*cfg['frequency_range_hz'])
    excluded = data.loc[~keep].copy()
    data = data.loc[keep].reset_index(drop=True)
    f,real,imag = validate_data(*data.to_numpy().T)
    out = args.out.resolve()
    for folder in ('configs','source_data','figures','provenance'):
        (out/folder).mkdir(parents=True,exist_ok=True)
    data.to_csv(out/'source_data/observations.csv',index=False)
    excluded.to_csv(out/'source_data/excluded_frequency_rows.csv',index=False)
    (out/'configs/parameters_zh.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    k = GeometryKernel(f,settings)
    regularization = []
    for alpha in cfg['regularization_grid']:
        scores = []
        for fold in range(3):
            test = np.arange(len(f))%3 == fold
            fitted = k.fit(real,imag,alpha,indices=np.flatnonzero(~test))
            m,y = k.residual_system(real,imag,np.flatnonzero(test))
            scores.append(float(np.sqrt(np.mean((m@fitted['x']-y)**2))))
        regularization.append({'alpha':alpha,'cv_mean_weighted_rmse':np.mean(scores),
                               **{f'fold_{i}_weighted_rmse':v for i,v in enumerate(scores)}})
    cv = pd.DataFrame(regularization)
    best_cv = cv.cv_mean_weighted_rmse.min()
    alpha = float(cv.loc[cv.cv_mean_weighted_rmse <= 1.05*best_cv,'alpha'].max())
    cv['selected'] = cv.alpha == alpha
    cv.to_csv(out/'source_data/regularization_cv.csv',index=False)
    best = k.fit(real,imag,alpha)
    print('Selected alpha:',alpha,'area:',best['area_m2'],'relative RMSE real/imag:',best['real_relative_rmse'],best['imag_relative_rmse'],flush=True)
    sensitivity = []
    for factor in cfg.get('length_upper_factors',[.1,.3,1.,3.,10.]):
        altered = replace(settings,length_range_m=(settings.length_range_m[0],settings.length_range_m[1]*factor))
        sk = GeometryKernel(f,altered)
        sr = sk.fit(real,imag,alpha)
        sensitivity.append({'length_upper_factor':factor,'length_upper_m':altered.length_range_m[1],
                            'area_m2':sr['area_m2'],'weighted_rmse':sr['weighted_rmse'],
                            'throat_length_p50_m':quantiles(sk.length,sr['throat_mass'])[1],
                            'throat_top_two_bins_mass':float(sr['throat_mass'][-2:].sum())})
    pd.DataFrame(sensitivity).to_csv(out/'source_data/length_bound_sensitivity.csv',index=False)
    profile, solutions = [], []
    areas = np.unique(np.r_[np.geomspace(*settings.area_range_m2,cfg['profile_points']),best['area_m2']])
    for area in areas:
        try:
            fit = k.fit(real,imag,alpha,fixed_area_m2=area,initial=best['x'])
        except RuntimeError as error:
            profile.append({'area_m2':area,'success':False,'message':str(error)})
            continue
        row = {'area_m2':area,'success':True,'weighted_rmse':fit['weighted_rmse'],
               'pore_radius_p50_m':quantiles(k.radius,fit['pore_mass'])[1],
               'throat_length_p50_m':quantiles(k.length,fit['throat_mass'])[1]}
        profile.append(row); solutions.append(fit)
    table = pd.DataFrame(profile)
    threshold = table.weighted_rmse.min()+cfg['profile_tolerance']
    table['acceptable'] = table.success & (table.weighted_rmse <= threshold)
    table.to_csv(out/'source_data/area_profile.csv',index=False)
    accepted = [s for s in solutions if s['weighted_rmse'] <= threshold]
    if not accepted: raise RuntimeError('No acceptable area profile solutions.')
    # These are sampled alternative solutions, not statistical confidence limits.
    for name,grid,key in [('pore_radius',k.radius,'pore_mass'),('throat_length',k.length,'throat_mass')]:
        masses = np.array([s[key] for s in accepted])
        frame = pd.DataFrame({f'{name}_m':grid,f'{name}_um':grid*1e6,
                              'fraction_per_log_bin':best[key],
                              'density_per_log10_m':best[key]/np.diff(np.log10(grid)).mean(),
                              'acceptable_solution_min':masses.min(axis=0),'acceptable_solution_max':masses.max(axis=0)})
        if name == 'pore_radius': frame['pore_diameter_um'] = 2*grid*1e6
        frame.to_csv(out/f'source_data/{name}_distribution.csv',index=False)
    pd.DataFrame([{'area_m2':s['area_m2'], **{f'pore_bin_{i}':v for i,v in enumerate(s['pore_mass'])},
                   **{f'throat_bin_{i}':v for i,v in enumerate(s['throat_mass'])}} for s in accepted]).to_csv(out/'source_data/acceptable_geometry_solutions.csv',index=False)
    prediction = data.copy()
    prediction['fitted_real_s_m'] = best['prediction'].real
    prediction['fitted_imag_s_m'] = best['prediction'].imag
    for name,sl in [('pore',slice(0,k.n)),('membrane',slice(k.n,2*k.n)),('dielectric_background',slice(-1,None))]:
        response = k.design[:,sl]@best['x'][sl]
        prediction[f'{name}_real_s_m'] = response.real
        prediction[f'{name}_imag_s_m'] = response.imag
    prediction.to_csv(out/'source_data/fitted_spectrum.csv',index=False)
    sv = np.linalg.svd(k.residual_system(real,imag)[0],compute_uv=False)
    rank = int(np.sum(sv>sv[0]*1e-8))
    status = []
    for name,key in [('pore','pore_mass'),('throat','throat_mass')]:
        if best[key][:2].sum()+best[key][-2:].sum() > .05:
            status.append(f'{name}: more than 5% mass in first/last two bins; grid-boundary sensitivity.')
    amin,amax = min(s['area_m2'] for s in accepted),max(s['area_m2'] for s in accepted)
    if np.isclose(amin,settings.area_range_m2[0],rtol=.001,atol=0) or np.isclose(amax,settings.area_range_m2[1],rtol=.001,atol=0):
        status.append('Area profile reaches configured bounds; range is not closed by these data.')
    if best['epsilon_effective'] > .99*settings.epsilon_range[1] or best['epsilon_effective'] < 1.01*settings.epsilon_range[0]:
        status.append('Effective dielectric background reaches a bound.')
    if best['weighted_rmse'] > 1: status.append('Fit exceeds assumed residual scales; model discrepancy remains.')
    summary = {key:best[key] for key in ('area_m2','sigma_dc_s_m','epsilon_effective','weighted_rmse','real_relative_rmse','imag_relative_rmse')}
    summary.update(status='conditional_effective_geometry_diagnostic',frequency_points=len(f),selected_alpha=alpha,
                   area_equivalent_radius_um=float(np.sqrt(best['area_m2']/np.pi)*1e6),
                   acceptable_area_range_m2=[amin,amax],
                   pore_radius_p10_p50_p90_um=(quantiles(k.radius,best['pore_mass'])*1e6).tolist(),
                   throat_length_p10_p50_p90_um=(quantiles(k.length,best['throat_mass'])*1e6).tolist(),
                   data_matrix_rank_at_1e8=rank,unknown_count=2*k.n+2,observed_scalar_count=2*len(f),
                   warnings=status,settings=asdict(settings),
                   uncertainty_label='sampled acceptable profile envelope, not a confidence interval',
                   area_model='one common electrical cross-section, independent of throat length',
                   surrogate='sigma_dc + 2/(Lambda*F)*(Cp+Cm) + i*omega*epsilon0*epsilon_eff',
                   shape_assumption='Figure5-like volume-fraction interpretation of the normalized electrical kernel weights is assumed, not measured.',
                   source=str(src.resolve()))
    summary['length_bound_sensitivity'] = sensitivity
    summary['length_bound_area_range_m2'] = [min(s['area_m2'] for s in sensitivity),max(s['area_m2'] for s in sensitivity)]
    summary['absolute_geometry_resolved'] = False
    summary['characteristic_band_radius_range_um'] = (np.sqrt(settings.parameters.diffusion_coefficient_m2_s/(np.pi*np.array([f.max(),f.min()])))*1e6).tolist()
    summary['characteristic_band_length_range_um'] = (np.sqrt(2*settings.parameters.diffusion_coefficient_m2_s/(np.pi*np.array([f.max(),f.min()])))*1e6).tolist()
    # Independent Figure5 comparison is deliberately after fitting/selection.
    comparison = None
    if cfg.get('figure5_validation'):
        from compute_polarization_spectra import load_figure5
        p,t = load_figure5(ROOT/cfg['figure5_validation'])
        comparison = (p,t)
        p.to_csv(out/'source_data/figure5_validation_pores.csv',index=False)
        t.to_csv(out/'source_data/figure5_validation_throats.csv',index=False)
        summary['figure5_validation_only'] = {'pore_radius_p50_um':float(quantiles(p.radius_m.to_numpy(),p.weight.to_numpy())[1]*1e6),
                                             'throat_length_p50_um':float(quantiles(t.length_m.to_numpy(),t.weight.to_numpy())[1]*1e6)}
    (out/'provenance/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame([
        {'parameter':'effective_throat_area','value':best['area_m2'],'unit':'m2'},
        {'parameter':'equivalent_circular_throat_radius','value':summary['area_equivalent_radius_um'],'unit':'um'},
        {'parameter':'conditional_area_profile_min','value':amin,'unit':'m2'},
        {'parameter':'conditional_area_profile_max','value':amax,'unit':'m2'},
        {'parameter':'fitted_dc_conductivity','value':best['sigma_dc_s_m'],'unit':'S/m'},
        {'parameter':'fitted_relative_permittivity','value':best['epsilon_effective'],'unit':'1'},
        *[{'parameter':f'{name}_p{q}','value':v,'unit':'um'} for name,key in [('pore_radius','pore_radius_p10_p50_p90_um'),('throat_length','throat_length_p10_p50_p90_um')] for q,v in zip([10,50,90],summary[key])]
    ]).to_csv(out/'source_data/parameter_summary.csv',index=False)
    hashes = [src,args.config,Path(__file__),ROOT/'code/src/pore_scale_electrical/sip_geometry_inversion.py',ROOT/'code/src/pore_scale_electrical/polarization.py']
    if cfg.get('figure5_validation'): hashes.append(ROOT/cfg['figure5_validation'])
    (out/'provenance/input_code_hashes.json').write_text(json.dumps({str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in hashes},indent=2),encoding='utf-8')
    (out/'provenance/command.json').write_text(json.dumps([sys.executable,*sys.argv],ensure_ascii=False,indent=2),encoding='utf-8')
    plot_results(out,k,best,prediction,accepted,table,comparison)
    plot_reference_histograms(out,k,best,comparison)
    write_report(out,summary)
    print(json.dumps({key:summary[key] for key in ('area_m2','acceptable_area_range_m2','pore_radius_p10_p50_p90_um','throat_length_p10_p50_p90_um','warnings')},indent=2),flush=True)


def plot_results(out,k,best,prediction,accepted,profile,comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'Arial','font.size':10,'legend.frameon':False,'xtick.direction':'in','ytick.direction':'in'})
    fig,ax = plt.subplots(2,2,figsize=(10,7),layout='constrained')
    f = k.f
    for j,part in enumerate(('real','imag')):
        axis = ax[0,j]
        axis.semilogx(f,prediction[f'sigma_{part}_s_m'],'o',mfc='none',color='black',ms=5,label='Measured')
        axis.semilogx(f,prediction[f'fitted_{part}_s_m'],color='#005f73',lw=2,label='Joint inversion')
        if part == 'imag': axis.set_yscale('log')
        axis.set(xlabel='Frequency (Hz)',ylabel=f'{part.capitalize()} conductivity (S/m)')
        axis.legend()
    ax[1,0].semilogx(f,100*(prediction.fitted_real_s_m/prediction.sigma_real_s_m-1),'o-',label='Real')
    ax[1,0].semilogx(f,100*(prediction.fitted_imag_s_m/prediction.sigma_imag_s_m-1),'s-',label='Imaginary')
    ax[1,0].axhline(0,color='gray',lw=.7)
    ax[1,0].set(xlabel='Frequency (Hz)',ylabel='Relative residual (%)'); ax[1,0].legend()
    for name,color,label in [('pore','green','Pore'),('membrane','red','Membrane'),('dielectric_background','blue','Dielectric background')]:
        ax[1,1].loglog(f,np.maximum(prediction[f'{name}_imag_s_m'],1e-20),'o--',ms=3,color=color,label=label)
    ax[1,1].loglog(f,best['prediction'].imag,color='gray',lw=2,label='All (surrogate)')
    ax[1,1].set(xlabel='Frequency (Hz)',ylabel='Imaginary conductivity (S/m)'); ax[1,1].legend(fontsize=8)
    for i,a in enumerate(ax.flat): a.set_title(f'({chr(97+i)})',loc='left')
    fig.savefig(out/'figures/sip_joint_fit.png',dpi=600); plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for i,(grid,key,label) in enumerate([(k.radius,'pore_mass','Pore radius (um)'),(k.length,'throat_mass','Throat length (um)')]):
        axis = axes[i]
        alternative = np.array([s[key] for s in accepted])
        axis.fill_between(grid*1e6,alternative.min(axis=0),alternative.max(axis=0),color='#94d2bd',alpha=.5,label='Acceptable-area envelope')
        axis.semilogx(grid*1e6,best[key],'-o',ms=3,color='#005f73',label='Conditional inversion')
        if comparison is not None:
            paper = comparison[i]; sizes = paper.iloc[:,0].to_numpy(); weights = paper.weight.to_numpy()
            edges = np.r_[grid[0]**1.5/grid[1]**.5,np.sqrt(grid[:-1]*grid[1:]),grid[-1]**1.5/grid[-2]**.5]
            mass,_ = np.histogram(sizes,bins=edges,weights=weights/weights.sum())
            axis.semilogx(grid*1e6,mass,'--',color='black',label='Figure5 (validation only)')
        axis.set(xlabel=label,ylabel='Fraction per log bin',ylim=(0,None))
        axis.legend(fontsize=7)
    good = profile.loc[profile.success]
    axes[2].semilogx(good.area_m2,good.weighted_rmse,'o-',ms=3,color='#005f73')
    axes[2].axvline(best['area_m2'],color='red',ls='--',label='Selected area')
    accepted_area = profile.loc[profile.acceptable,'area_m2']
    axes[2].axvspan(accepted_area.min(),accepted_area.max(),alpha=.25,color='#94d2bd',label='Sampled acceptable range')
    axes[2].set(xlabel='Common throat area (m$^2$)',ylabel='Weighted spectral RMSE'); axes[2].legend(fontsize=7)
    for i,a in enumerate(axes): a.set_title(f'({chr(97+i)})',loc='left')
    fig.savefig(out/'figures/inferred_geometry_and_area.png',dpi=600); plt.close(fig)


def plot_reference_histograms(out,k,best,comparison):
    """Match the earlier task's histogram style without calling bins raw counts."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, FuncFormatter
    sys.path.insert(0,str(ROOT/'code/scripts/pore_network'))
    from compare_niu2020_pnextract_geometry import centers_to_log_edges
    plt.rcParams.update({'font.family':'Times New Roman','font.size':14,'axes.linewidth':.8})
    panels = []
    for i,(grid,key) in enumerate([(k.radius,'pore_mass'),(k.length,'throat_mass')]):
        edges = centers_to_log_edges(grid)
        paper_edges = edges if comparison is None else centers_to_log_edges(comparison[i].iloc[:,0].to_numpy())
        common = np.geomspace(min(edges[0],paper_edges[0]),max(edges[-1],paper_edges[-1]),50)
        # Display rebinning assumes uniform mass within each original log bin.
        mass = np.diff(np.interp(np.log(common),np.log(edges),np.r_[0,np.cumsum(best[key])]))
        paper = None
        if comparison is not None:
            weights = comparison[i].weight.to_numpy(); weights = weights/weights.sum()
            paper = np.diff(np.interp(np.log(common),np.log(paper_edges),np.r_[0,np.cumsum(weights)]))
            assert np.isclose(paper.sum(),1)
        assert np.isclose(mass.sum(),1) and np.all(mass >= -1e-12)
        panels.append((common,mass,paper))
        frame = pd.DataFrame({'bin_left_m':common[:-1],'bin_right_m':common[1:],
                              'bin_center_m':np.sqrt(common[:-1]*common[1:]),'inverted_model_fraction':mass})
        if paper is not None: frame['figure5_volume_fraction'] = paper
        frame.to_csv(out/f'source_data/histogram_panel_{i+1}.csv',index=False)
    for overlay in ([False,True] if comparison is not None else [False]):
        fig,axes = plt.subplots(1,2,figsize=(11,5.2))
        for i,(axis,(edges,mass,paper)) in enumerate(zip(axes,panels)):
            centers = np.sqrt(edges[:-1]*edges[1:])
            if overlay:
                axis.bar(centers,paper,width=np.diff(edges)*.78,color='#2878B5',edgecolor='#2878B5',alpha=.5,lw=.8,label='Niu 2020 (validation)')
            axis.bar(centers,mass,width=np.diff(edges)*(.78 if overlay else .42),
                     color='#E58632' if overlay else '#777777',edgecolor='#E58632' if overlay else '#222222',
                     alpha=.5 if overlay else 1.,lw=.8,label='SIP inversion (conditional)')
            top = max(.15,np.ceil(max(mass.max(),paper.max() if overlay else 0)*1.1/.05)*.05)
            axis.set(xscale='log',xlim=(edges[0]/1.1,edges[-1]*1.1),ylim=(0,top),
                     xlabel=['Pore radius (m)','Pore throat length (m)'][i],ylabel='Relative fraction')
            axis.set_yticks(np.arange(0,top+.025,.05))
            axis.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
            axis.yaxis.set_minor_locator(AutoMinorLocator(5))
            axis.tick_params(which='major',direction='in',length=7)
            axis.tick_params(which='minor',direction='in',length=3.5)
            axis.text(-.18,1.03,f'({chr(97+i)})',transform=axis.transAxes,fontsize=21)
            if overlay: axis.legend(loc='upper left',frameon=False,fontsize=11)
        fig.subplots_adjust(left=.095,right=.98,bottom=.19,top=.90,wspace=.34)
        fig.savefig(out/f'figures/geometry_histograms{"_overlay" if overlay else ""}.png',dpi=600)
        plt.close(fig)
    (out/'provenance/histogram_metadata.json').write_text(json.dumps({
        'statistic':'Normalized model kernel mass; Figure5 relative volume for independent comparison, not number frequency.',
        'paper_pore_size_interpretation':'Used as radius for comparison, consistent with existing Niu model; workbook labels size, not an independent diameter definition.',
        'rebinning':'49 common log bins spanning both supports; uniform log density within source bins, no mass discarded.',
        'styles':'gray thin bars; blue/orange half-transparent overlay; reference task 01a0948a-ee76-7662-9075-bd324c21bdf4',
        'backend':'Python matplotlib','dpi':600},indent=2),encoding='utf-8')


def write_report(out,s):
    area = s['area_m2']; low,high = s['acceptable_area_range_m2']
    report = f'''# SIP 实验反演等效孔隙几何

本结果用 {s['frequency_points']} 个实测复电导率频点联合反演。实部相对RMSE为 {100*s['real_relative_rmse']:.2f}%，虚部为 {100*s['imag_relative_rmse']:.2f}%。这里的分布是固定电化学参数和模型假设下的等效分布，不是唯一的真实孔网重建。

## 本次结果

- 等效孔喉面积：{area:.6g} m²；圆形等效半径：{s['area_equivalent_radius_um']:.3f} μm。
- 可接受面积扫描范围：{low:.6g} 至 {high:.6g} m²。仅表示采样的替代解范围，不是置信区间。
- 改变孔喉长度上界后的面积范围：{s['length_bound_area_range_m2']} m²，说明上述固定边界范围不能当作总体不确定性。
- 孔半径P10/P50/P90：{s['pore_radius_p10_p50_p90_um']} μm；孔直径为上述值的两倍。
- 孔喉长度P10/P50/P90：{s['throat_length_p10_p50_p90_um']} μm。
- 数据矩阵数值秩：{s['data_matrix_rank_at_1e8']}；标量观测数：{s['observed_scalar_count']}；未知量数：{s['unknown_count']}。平滑和非负约束选择一个稳定解，不会创造新的观测信息。

## 数据和模型边界

输入是实验复电导率。Niu运行仅读取Figure6前三列，Figure5仅在拟合及面积扫描完成后用于独立对照。未使用Figure7/8的论文模拟曲线，也没有沿用此前2.8e-11 m²标定面积作为输入。

采用 sigma*=sigma_dc+2/(Lambda F)*(Cp*+Cm*)+i omega epsilon0 epsilon_eff 的准静态有效体近似。Cp、Cm复用项目Niu模型；固定D、eta、表面电导、Lambda、F；共同拟合实部和虚部，直流与低频介电背景单独拟合。有效介电背景不是全频Maxwell场求解。未运行350³ AC3D，因此拟合误差仅适用于本近似模型。

孔分布非负并归一化。孔喉系数=A×归一化权重，面积由系数和得到；所有长度档共用一个面积，不能据此输出独立的孔喉半径分布。导出的fraction是模型核的质量权重，解释为相对体积需要额外假设，不能称为实测孔隙率。

用三折按频率交错留出选择平滑强度。面积扫描在固定面积时重新优化其余变量，可接受范围由最小加权RMSE加0.2定义。2%实部、5%虚部只是残差权重，非仪器误差；解包络不是统计置信区间。未探索的扩散系数、核函数、频带、边界先验变化可带来更大不确定性。

存在明确的非唯一性：D乘4、孔半径和孔喉长度乘2、孔喉面积乘2，可保持当前模型的复频谱完全不变（已用自动化检查验证）。所以SIP单独无法给出不依赖电化学先验的绝对几何。

## 检查与输出

警示：{'; '.join(s['warnings']) or '当前最优解未触发边界或残差阈值警示；这不代表几何唯一。'}

source_data 中有孔半径/直径分布、孔喉长度分布、拟合谱、面积扫描、可接受替代解及参数汇总CSV。figures 中有联合拟合图和几何/面积图。configs保存本次中文说明配置，provenance保存输入/代码哈希与完整运行命令。

geometry_histograms.png为灰色细柱双联图；geometry_histograms_overlay.png为蓝色Figure5、橙色条件反演的半透明叠加图。两者采用共同对数分箱，绘图重分箱假设原始每个log分箱内部均匀，质量总和保持为1。纵轴为相对分数，不是按个数统计的频率。对照中将Figure5的pore node size按现有Niu模型解释为半径，该工作簿本身未明确写radius/diameter。

本反演不能唯一确定三维拓扑，因此未生成数字岩心、孔网HTML或冒称full-grid sweep；这些需要独立CT或网络连通性约束。若需物理几何可信度更高，应加入独立孔喉面积/长度关系、CT、压汞或NMR约束后再做联合反演。

尺寸分位数按各原始log分箱内部质量均匀的约定计算，与直方图重分箱一致。

方法背景：[Spectral induced polarization porosimetry](https://academic.oup.com/gji/article/198/2/1016/2874197)讨论从弛豫谱映射孔隙尺度；本实现具体使用本项目Niu核函数，而非直接套用该文的几何系数。
'''
    (out/'README_zh.md').write_text(report,encoding='utf-8')


if __name__ == '__main__':
    main()
