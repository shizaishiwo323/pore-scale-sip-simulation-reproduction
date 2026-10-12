"""绘制四机制各自与完整实验谱的对比；仅使用本轮实际GPU输出。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'results/sample16_gpu_7freq_20260912_2340'

def main():
    exp=pd.read_csv(RUN/'source_data/experiment_all.csv')
    chosen=pd.read_csv(RUN/'source_data/selected_7_frequencies.csv').frequency_hz.to_numpy()
    plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    names={'maxwell':'Interfacial','pore':'Pore','membrane':'Membrane','all':'All'}
    colors={'maxwell':'#32749a','pore':'#9a6ca5','membrane':'#638c53','all':'#b65245'}
    fig,axes=plt.subplots(4,2,figsize=(11,13),layout='constrained')
    frames=[]
    for row,(mechanism,name) in enumerate(names.items()):
        path=RUN/f'simulation_sweeps/{mechanism}/sweep_results.csv'
        a=pd.read_csv(path).sort_values('frequency_hz')
        assert len(a)==7 and np.allclose(a.frequency_hz,chosen,rtol=0,atol=1e-10)
        assert a.converged_strict.all() and (a.relative_residual_norm<=1e-6).all()
        a['source_sweep']=str(path.relative_to(RUN)); frames.append(a)
        for col,part in enumerate(['real','imag']):
            ax=axes[row,col]
            y=a[f'effective_sigma_{part}_s_m']; ey=exp[f'experiment_sigma_{part}_s_m']
            if mechanism in ['pore','membrane']:
                # 零附近及负号是数值信息，使用symlog保留，不取绝对值伪造正响应。
                ax.set_yscale('symlog',linthresh=1e-8)
                keep=np.ones(len(exp),dtype=bool)
            else:
                ax.set_yscale('log'); keep=ey>0
            ax.set_xscale('log')
            ax.plot(exp.frequency_hz[keep],ey[keep],'o',color='#333333',ms=3,label='Experiment (51 frequencies)')
            ax.plot(a.frequency_hz,y,'o-',color=colors[mechanism],ms=5,lw=1.6,label=f'{name}: 7 GPU solves')
            ax.set_title(name+' — '+('real' if part=='real' else 'imaginary'))
            ax.set_ylabel(r"$\sigma'$ (S/m)" if part=='real' else r"$\sigma''$ (S/m)")
            ax.set_xlabel('Frequency (Hz)'); ax.grid(alpha=.18); ax.legend(fontsize=8)
            if mechanism in ['pore','membrane']:
                ax.text(.03,.06,'x pores do not percolate; near-zero values\nare numerical residuals (symlog axis)',transform=ax.transAxes,fontsize=8)
    fig.suptitle('Sample 16 | x direction | ds2 full-domain GPU diagnostic',fontsize=14)
    fig.savefig(RUN/'figures/four_mechanisms_7frequencies.png',dpi=600)
    plt.close(fig)
    data=pd.concat(frames,ignore_index=True)
    data.to_csv(RUN/'source_data/four_mechanisms_7frequencies.csv',index=False)
    for mechanism,name in names.items():
        f,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        a=data[data.mechanism==mechanism]
        for axis,part in zip(ax,['real','imag']):
            axis.set_xscale('log')
            ey=exp[f'experiment_sigma_{part}_s_m']
            keep=ey>0 if mechanism in ['maxwell','all'] else np.ones(len(exp),dtype=bool)
            if mechanism in ['pore','membrane']: axis.set_yscale('symlog',linthresh=1e-8)
            else: axis.set_yscale('log')
            axis.plot(exp.frequency_hz[keep],ey[keep],'o',color='black',ms=3,label='Experiment')
            axis.plot(a.frequency_hz,a[f'effective_sigma_{part}_s_m'],'o-',color=colors[mechanism],label='7 GPU frequencies')
            axis.set(xlabel='Frequency (Hz)',ylabel=f'Conductivity {part} (S/m)'); axis.legend(fontsize=8); axis.grid(alpha=.2)
        f.suptitle(f'{name} | Sample 16 | x direction | ds2 diagnostic')
        f.savefig(RUN/f'figures/{mechanism}_7frequencies.png',dpi=600); plt.close(f)
    summary={'gpu_solves':len(data),'frequency_hz':chosen.tolist(),'max_true_residual':float(data.relative_residual_norm.max()),
             'classification':'ds2 full-domain diagnostic, not original-resolution sweep','nonpositive_experiment_imag':int((exp.experiment_sigma_imag_s_m<=0).sum())}
    (RUN/'provenance/verification.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(summary)

if __name__=='__main__': main()
