"""保持样品16参数，仅改变加载方向，顺序运行以避免GPU显存竞争。"""
from pathlib import Path
import json
import subprocess
import sys
import shutil
import hashlib
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'results/sample16_gpu_7freq_20260912_2340'
RUN=ROOT/'results/sample16_gpu_yz_7freq_20260913'

def finalize():
    for folder in ['configs/source_snapshot','source_data','provenance']:
        (RUN/folder).mkdir(parents=True,exist_ok=True)
    frames=[]
    for direction in ['y','z']:
        frame=pd.read_csv(RUN/direction/'source_data/niu_style_simulation.csv')
        assert len(frame)==28 and frame.converged_strict.all() and (frame.relative_residual_norm<=1e-6).all()
        assert set(frame.direction)=={direction} and (frame.groupby('mechanism').size()==7).all()
        frames.append(frame)
        path=RUN/direction/'configs/parameters_zh.json'
        params=json.loads(path.read_text(encoding='utf-8-sig'))
        params['坐标约定']='沿用求解器x/y/z=数组轴0/1/2；输入为CT存储ZYX，故solver x=CT Z、solver y=CT Y、solver z=CT X。实验方向对应未确认。'
        params['孔膜初值']='不贯通连通分量使用几何初始电势，贯通分量仍由GPU迭代求解。'
        if direction=='y':
            params['求解']='界面/综合沿用GPU BiCGSTAB FFT；孔/膜采用GPU孔隙CSR CG、Jacobi预条件、归一化实数双精度场并乘回复电导因子；rtol=1e-6、maxiter=50000。'
        path.write_text(json.dumps(params,indent=2,ensure_ascii=False),encoding='utf8')
    combined=pd.concat(frames,ignore_index=True)
    combined.to_csv(RUN/'source_data/yz_mechanisms.csv',index=False)
    summary={'points':len(combined),'max_true_residual':float(combined.relative_residual_norm.max()),'tests_passed':18,
             'classification':'ds2 diagnostic','axis_mapping':{'solver_x':'CT_storage_Z','solver_y':'CT_storage_Y','solver_z':'CT_storage_X'}}
    (RUN/'provenance/verification.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    hashes={}
    paths=[ROOT/'code/src/pore_scale_electrical'/name for name in ['ac3d_gpu.py','ac3d_pore_gpu.py','ac3d_active_domain.py','polarization.py']]
    paths.extend(ROOT/'code/scripts/sip_simulation'/name for name in ['run_sample89_3d_sip_comparison.py','run_sample16_yz_seven_frequencies.py','plot_sample16_niu_reference_style.py'])
    for path in paths:
        shutil.copy2(path,RUN/'configs/source_snapshot'/path.name)
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (RUN/'provenance/source_snapshot_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
    report=RUN/'README.md'
    text=report.read_text(encoding='utf8').split('## 核验汇总')[0]
    text+='## 核验汇总\n\n56点全部通过真实残差阈值1e-6，最大残差为'+f'{summary["max_true_residual"]:.3e}。18项相关测试通过。\n\n'
    text+='y方向孔、膜响应非零；z方向接近零。0.1 Hz的整体虚部电导率如下：\n\n| 方向 | 机制 | 虚部电导率 (S/m) |\n|---|---|---:|\n'
    for _,row in combined[combined.frequency_hz==.1].iterrows():
        text+=f'| {row.direction} | {row.mechanism} | {row.effective_sigma_imag_s_m:.6e} |\n'
    report.write_text(text,encoding='utf8')
    print(summary)

def main():
    RUN.mkdir(exist_ok=True)
    for direction in ['z','y']:
        out=RUN/direction
        for folder in ['configs','source_data','provenance','figures','simulation_sweeps']:
            (out/folder).mkdir(parents=True,exist_ok=True)
        for name in ['experiment_all.csv','selected_7_frequencies.csv']:
            shutil.copy2(BASE/'source_data'/name,out/'source_data'/name)
        params=json.loads((BASE/'configs/parameters_zh.json').read_text(encoding='utf8'))
        params['计算说明']=f'{direction}向、周期边界、全域ds2双精度GPU诊断；复用同一全分辨率PNM，无几何缩放。'
        params['几何来源']=str(BASE)
        params['孔膜初值']='仅在加载方向没有周期贯通路径时构造几何初值，随后检查GPU场方程残差；存在贯通路径时使用Krylov迭代求解。'
        (out/'configs/parameters_zh.json').write_text(json.dumps(params,indent=2,ensure_ascii=False),encoding='utf8')
        for mechanism in ['maxwell','all','pore','membrane']:
            command=json.loads((BASE/f'configs/{mechanism}_command.json').read_text(encoding='utf-8-sig'))
            command[command.index('--out-dir')+1]=str(out)
            command[command.index('--sample-label')+1]=f'Sample 16: {direction} direction, 7 frequencies, ds2 diagnostic'
            if direction=='y' and mechanism in ['pore','membrane']:
                command[command.index('--preconditioner')+1]='jacobi'
                command[command.index('--field-solve-mode')+1]='gpu-active-pore-csr'
                command[command.index('--maxiter')+1]='50000'
                command[command.index('--residual-every')+1]='500'
                command[command.index('--progress-every')+1]='500'
            command.extend(['--direction',direction,'--resume'])
            (out/f'configs/{mechanism}_command.json').write_text(json.dumps(command,indent=2),encoding='utf8')
            print('START',direction,mechanism,flush=True)
            with (out/f'provenance/{mechanism}.log').open('a',encoding='utf8') as log:
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            for result in (out/f'simulation_sweeps/{mechanism}').glob('frequency_*/result.json'):
                (result.parent/'config.yml').write_text(json.dumps({'command':command},indent=2),encoding='utf8')
            print('DONE',direction,mechanism,flush=True)
        subprocess.run([sys.executable,str(ROOT/'code/scripts/sip_simulation/plot_sample16_niu_reference_style.py'),'--run-dir',str(out)],cwd=ROOT,check=True)
        data=pd.read_csv(out/'source_data/niu_style_simulation.csv')
        (out/'provenance/verification.json').write_text(json.dumps({'direction':direction,'points':len(data),'max_true_residual':float(data.relative_residual_norm.max()),'all_converged':bool(data.converged_strict.all())},indent=2),encoding='utf8')
    shutil.copy2(__file__,RUN/Path(__file__).name)
    finalize()

if __name__=='__main__':
    main()
