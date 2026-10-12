"""用本轮真实7频点结果绘制上下两联机制叠加图。"""
from pathlib import Path
import json
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT/'results/sample16_gpu_7freq_20260912_2340'
EPS0 = 8.8541878128e-12

def main():
    global RUN
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,default=RUN)
    args = parser.parse_args()
    RUN = args.run_dir.resolve()
    # 定量两联图：比较实测与四机制；只连接7个计算点，不平滑、不外推。
    exp = pd.read_csv(RUN/'source_data/experiment_all.csv')
    chosen = pd.read_csv(RUN/'source_data/selected_7_frequencies.csv').frequency_hz.to_numpy()
    frames = []
    for name in ['pore','membrane','maxwell','all']:
        source = RUN/f'simulation_sweeps/{name}/sweep_results.csv'
        a = pd.read_csv(source).sort_values('frequency_hz')
        assert len(a) == 7 and np.allclose(a.frequency_hz, chosen)
        assert a.converged_strict.all() and (a.relative_residual_norm <= 1e-6).all()
        a['relative_permittivity'] = a.effective_sigma_imag_s_m/(2*np.pi*a.frequency_hz*EPS0)
        a['source_sweep'] = str(source.relative_to(RUN))
        frames.append(a)
    data = pd.concat(frames, ignore_index=True)
    directions = data.direction.unique()
    assert len(directions) == 1
    direction = directions[0]
    near_zero = [name for name in ['pore','membrane'] if data.loc[data.mechanism==name,'effective_sigma_real_s_m'].abs().max()<1e-12 and data.loc[data.mechanism==name,'effective_sigma_imag_s_m'].abs().max()<1e-12]
    exp['relative_permittivity'] = exp.experiment_sigma_imag_s_m/(2*np.pi*exp.frequency_hz*EPS0)
    data.to_csv(RUN/'source_data/niu_style_simulation.csv', index=False)
    exp.to_csv(RUN/'source_data/niu_style_experiment.csv', index=False)
    plt.rcParams.update({'font.family':'Arial','font.size':12,'mathtext.fontset':'dejavusans',
                         'axes.linewidth':0.9,'axes.spines.top':True,'axes.spines.right':True,
                         'legend.frameon':False,'xtick.direction':'in','ytick.direction':'in'})
    styles = {'pore':('green','--',1.6,'Pore polarization'),
              'membrane':('red',(0,(2,1.3)),1.6,'Membrane polarization'),
              'maxwell':('blue',(0,(3,3)),1.6,'Interfacial polarization'),
              'all':('#808080','-',3.4,'Simulation (all polarizations)')}
    metadata = {'classification':f'Sample 16, {direction} direction, ds2 diagnostic', 'frequency_hz':chosen.tolist(),
                'permittivity_conversion':'sigma_imag / (2*pi*f*epsilon0)',
                'line_policy':'straight segments between seven solved points in log-log coordinates; no smoothing or extrapolation',
                'near_zero_policy':'pore/membrane retain raw numerical values; curves below displayed limits are not raised to the axis floor',
                'nonpositive_experiment_imag_omitted':int((exp.experiment_sigma_imag_s_m<=0).sum()), 'exports':{}}
    for variant, upper in [('permittivity_imaginary','relative_permittivity'),('real_imaginary','real')]:
        fig, axes = plt.subplots(2,1,figsize=(8.0,9.8))
        fig.subplots_adjust(left=.16,right=.97,bottom=.14,top=.94,hspace=.32)
        fig.suptitle(f'Sample 16 | {direction} direction | ds2 diagnostic | 7 simulated frequencies',fontsize=11,y=.985)
        limits = []
        for index,(ax,part) in enumerate(zip(axes,[upper,'imag'])):
            simcol = 'relative_permittivity' if part == upper == 'relative_permittivity' else f'effective_sigma_{part}_s_m'
            expcol = 'relative_permittivity' if part == 'relative_permittivity' else f'experiment_sigma_{part}_s_m'
            positive = exp[expcol] > 0
            ax.plot(exp.frequency_hz[positive],exp.loc[positive,expcol],linestyle='none',marker='^',
                    markersize=5.2,markerfacecolor='white',markeredgecolor='black',markeredgewidth=.8,label='Experimental data',zorder=4)
            for name,(color,ls,lw,label) in styles.items():
                a = data[data.mechanism==name]
                ax.plot(a.frequency_hz,a[simcol].where(a[simcol]>0),color=color,linestyle=ls,linewidth=lw,label=label,zorder=2)
            # 有限零响应不决定log坐标跨度；明确标注它们位于显示范围以下。
            visible = np.concatenate([exp.loc[positive,expcol],data.loc[~data.mechanism.isin(near_zero),simcol]])
            visible = visible[np.isfinite(visible)&(visible>0)]
            low = 10**np.floor(np.log10(visible.min()))
            high = 10**np.ceil(np.log10(visible.max()))
            if index == 0: high *= 100 # 图内图例保留独立空间。
            ax.set(xscale='log',yscale='log',xlim=(chosen[0]/1.25,chosen[-1]*1.25),ylim=(low,high),xlabel=r'Frequency $f$, Hz')
            ylabel = (r'Effective permittivity $\varepsilon^{\prime}_{\mathrm{eff}}/\varepsilon_0$' if part=='relative_permittivity'
                      else r'Effective '+('real' if part=='real' else 'imaginary')+r' conductivity $\sigma^{'+('\prime' if part=='real' else '\prime\prime')+r'}_{\mathrm{eff}}$, S m$^{-1}$')
            ax.set_ylabel(ylabel)
            ax.yaxis.set_major_locator(LogLocator(base=10,numticks=5))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(which='major',length=6,width=.8)
            ax.tick_params(which='minor',length=3,width=.6)
            ax.text(-.13,1.04,f'({chr(97+index)})',transform=ax.transAxes,fontsize=21)
            if index == 0: ax.legend(loc='upper right',fontsize=10.5,handlelength=3.5,labelspacing=.3)
            limits.append([float(low),float(high)])
        note = ('Pore and membrane responses are approximately zero (below the log-axis limits).\n' if len(near_zero)==2 else '')
        fig.text(.16,.035,note+'Nonpositive experimental values are omitted only from the log plots.',fontsize=9,linespacing=1.5)
        name = f'sample16_niu_style_{variant}.png'
        fig.savefig(RUN/'figures'/name,dpi=600)
        plt.close(fig)
        metadata['exports'][name] = {'size_inches':[8,9.8],'dpi':600,'ylim':limits}
    (RUN/'provenance/niu_style_figures.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    shutil.copy2(__file__, RUN/'configs'/Path(__file__).name)

if __name__ == '__main__':
    main()
