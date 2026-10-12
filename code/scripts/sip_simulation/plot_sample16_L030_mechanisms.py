"""叠加已验证的独立机制、联合场解和LKC-16实验；不重新拟合或模拟。"""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'results/sample16_membrane_only_L030_gpu7_20260913'
CAL=ROOT/'results/sample16_effective_membrane_length_gpu7_20260913'
INPUTS={
    'maxwell':CAL/'simulation_sweeps/original_maxwell/sweep_results.csv',
    'pore':CAL/'simulation_sweeps/original_pore/sweep_results.csv',
    'membrane':RUN/'simulation_sweeps/membrane/sweep_results.csv',
    'all':CAL/'simulation_sweeps/calibrated_L030_all/sweep_results.csv',
}


def complex_column(frame,phase):
    return frame[f'{phase}_sigma_real_s_m'].to_numpy()+1j*frame[f'{phase}_sigma_imag_s_m'].to_numpy()


def main():
    frames={key:pd.read_csv(path) for key,path in INPUTS.items()}
    frequencies=frames['membrane'].frequency_hz.to_numpy()
    for frame in frames.values():
        assert len(frame)==7 and np.array_equal(frame.frequency_hz,frequencies)
        assert frame.converged_strict.all() and (frame.relative_residual_norm<=1e-6).all()
        assert (frame.direction=='y').all() and (frame.voxel_size_m==3.84e-6).all()
        assert (frame['shape']=='[400, 333, 333]').all() and (frame['dtype']=='complex128').all()
    for key in ['pore','membrane']:
        assert (complex_column(frames[key],'solid')==0).all()
    assert np.allclose(complex_column(frames['all'],'solid'),complex_column(frames['maxwell'],'solid'),rtol=1e-12,atol=1e-16)
    # 各机制水相输入相容；有效电导率的总响应直接读取联合求解结果。
    sum_water=sum(complex_column(frames[k],'water') for k in ['maxwell','pore','membrane'])-2*.12
    assert np.allclose(sum_water,complex_column(frames['all'],'water'),rtol=1e-12,atol=1e-15)
    experiment_source=CAL/'source_data/experiment_all.csv'
    experiment=pd.read_csv(experiment_source)
    assert set(frequencies)<=set(experiment.frequency_hz)
    dense_source=RUN/'source_data/dense_membrane_model.csv'
    dense=pd.read_csv(dense_source)
    factor=complex_column(frames['membrane'],'effective')/complex_column(frames['membrane'],'water')
    assert np.max(np.abs(factor.imag))<1e-12 and np.ptp(factor.real)/np.mean(factor.real)<1e-5
    merged=pd.concat([f.assign(component=k) for k,f in frames.items()],ignore_index=True)
    merged.to_csv(RUN/'source_data/mechanism_overlay_gpu7.csv',index=False)
    experiment.to_csv(RUN/'source_data/mechanism_overlay_experiment.csv',index=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':12,'mathtext.fontset':'stix',
        'axes.linewidth':.8,'xtick.direction':'in','ytick.direction':'in'})
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    fig.subplots_adjust(left=.085,right=.98,bottom=.23,top=.76,wspace=.27)
    styles=[('maxwell','Interfacial + dielectric','#E49C4E','s'),
            ('pore','Pore polarization','#578ECD','D'),
            ('membrane','Membrane polarization','#439878','o'),
            ('all','All mechanisms (joint solve)','#ED6868','^')]
    for j,part in enumerate(['real','imag']):
        ax=axes[j]
        values=experiment[f'experiment_sigma_{part}_s_m'];keep=values>0
        ax.plot(experiment.frequency_hz[keep],values[keep],'k.',ms=4,label='LKC-16 experiment',zorder=4)
        for key,label,color,marker in styles:
            frame=frames[key]
            linestyle='None' if key=='membrane' else ('-' if key=='all' else '--')
            ax.plot(frequencies,frame[f'effective_sigma_{part}_s_m'],linestyle=linestyle,color=color,
                marker=marker,mfc='white',ms=4,lw=1.8,label=label,zorder=3)
            if key=='membrane':
                ax.plot(dense.frequency_hz,dense[f'effective_sigma_{part}_s_m'],ls='--',color=color,lw=1.8,zorder=2)
        ax.set(xscale='log',yscale='log',xlim=(.08,25000),xlabel='Frequency (Hz)',
               ylabel=[r"$\sigma'$ (S/m)",r"$\sigma''$ (S/m)"][j])
        ax.text(-.13,1.03,['(a)','(b)'][j],transform=ax.transAxes,fontsize=16)
    axes[0].set_ylim(1e-4,1e-2)
    axes[1].set_ylim(5e-11,5e-4)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,.91),ncol=3,frameon=False,fontsize=11)
    fig.suptitle('Sample 16 | mechanism comparison | L = 0.3 μm | η = 0.002443',fontsize=15,y=.98)
    fig.text(.53,.12,'Standalone mechanisms and joint total: effective conductivities are not additive.',ha='center',fontsize=11)
    fig.text(.53,.065,'7 GPU points per mechanism; membrane curve uses a verified model. ds2 field, y direction.',ha='center',fontsize=10)
    fig.text(.53,.025,'Three nonpositive experimental imaginary values are omitted from the logarithmic panel.',ha='center',fontsize=10)
    for suffix,dpi in [('',600),('_preview',150)]:
        fig.savefig(RUN/'figures'/f'mechanism_overlay_L030{suffix}.png',dpi=dpi)
    plt.close(fig)
    metadata={'性质':'已有GPU结果的叠加绘图；没有新增模拟、修改参数或拟合实验。',
        '参数':'L_relax=0.3微米，eta=0.002442810092526288，D=1.3e-9 m²/s，sigma_w=0.12 S/m。',
        '机制':'膜和孔单机制固相严格为0；界面/介电与综合机制保持原有0.0024 S/m固相背景及介电项。',
        '总曲线':'直接使用calibrated_L030_all联合GPU场解；不将独立机制有效电导率相加。',
        '曲线含义':'所有曲线为相应机制设置下的有效复电导率；实部含直流背景，绿线为独立膜机制而非综合差值。',
        'figure_contract':{'conclusion':'独立机制的频谱与综合响应、实验总响应在同一坐标系下比较。',
            'panels':['real conductivity','imaginary conductivity'],'archetype':'quantitative grid','backend':'Python/matplotlib','export':'PNG 600 dpi'},
        'reused_gpu_points':28,'new_gpu_points':0,'all_points_strictly_converged':True,
        'phase_input_consistency_verified':True,'max_residual':float(merged.relative_residual_norm.max()),
        'experiment_nonpositive_imag_omitted':int((experiment.experiment_sigma_imag_s_m<=0).sum()),
        '展示':'黑点为全部原始实验频率；模拟每机制7点。膜连续线来自先前核验的解析膜模型，其余连线只连接GPU点。',
        'sources':{key:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for key,path in {
            **INPUTS,'experiment':experiment_source,'dense_membrane':dense_source}.items()}}
    assert metadata['experiment_nonpositive_imag_omitted']==3
    (RUN/'provenance/mechanism_overlay.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')
    shutil.copy2(__file__,RUN/'configs/source_snapshot'/Path(__file__).name)
    note='\n## 独立机制与综合响应叠加图\n\nfigures/mechanism_overlay_L030.png叠加当前纯膜、孔极化、界面/介电机制、联合求解的综合结果和LKC-16实验。复用28个已严格收敛的GPU点，未新增求解。膜/孔单机制固相为0，综合与界面机制沿用原固相背景，因此有效电导率不能直接相加。源数据、输入哈希、非正实验虚部过滤记录见source_data/mechanism_overlay_*与provenance/mechanism_overlay.json。\n'
    readme=RUN/'README.md'
    if '## 独立机制与综合响应叠加图' not in readme.read_text(encoding='utf8'):
        with readme.open('a',encoding='utf8') as out: out.write(note)
    print(json.dumps({'gpu_points':28,'new_simulation_points':0,'max_residual':metadata['max_residual'],'plot':str(RUN/'figures/mechanism_overlay_L030.png')},indent=2))


if __name__=='__main__':
    main()
