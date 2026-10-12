"""样品16：从原始CT重新提取网络，执行7频点GPU机制分析。"""
from pathlib import Path
import sys, json, subprocess, hashlib, shutil
import numpy as np
import pandas as pd
import tifffile

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'code/src'))
sys.path.insert(0,str(ROOT/'code/scripts/pore_network'))
from run_sample89_3d_sip_comparison import parse_psip_csv, add_bulk_conductivity_from_psip_impedance
from run_segmented_core_pnextract_ballstick import write_pnextract_mhd, write_pore_throat_distribution_figure

RUN=ROOT/'results/sample16_gpu_7freq_20260912_2340'
def command(args,log,cwd=ROOT):
    with log.open('w',encoding='utf8') as handle:
        subprocess.run(list(map(str,args)),cwd=cwd,stdout=handle,stderr=subprocess.STDOUT,check=True)

def main():
    for d in ['configs','digital_rock','pore_network/input','simulation_sweeps','source_data','figures','provenance']:
        (RUN/d).mkdir(parents=True,exist_ok=True)
    src=ROOT/'data/ct_backed_samples_raw_copy_20260605/sample_16_Grainstone/CT_slices/5-16seged.tiff'
    exp=ROOT/'data/ct_backed_samples_raw_copy_20260605/sample_16_Grainstone/SIP/LKC-16.csv'
    e=add_bulk_conductivity_from_psip_impedance(parse_psip_csv(exp),current_resistor_ohm=10000,sample_length_cm=3.6322,sample_diameter_cm=2.52476)
    ids=np.round(np.linspace(0,len(e)-1,7)).astype(int)
    frequencies=e.frequency_hz.iloc[ids].tolist()
    assert len(set(frequencies))==7 and frequencies[0]==e.frequency_hz.min() and frequencies[-1]==e.frequency_hz.max()
    e.to_csv(RUN/'source_data/experiment_all.csv',index=False)
    e.iloc[ids].to_csv(RUN/'source_data/selected_7_frequencies.csv',index=False)
    print('FREQUENCIES',frequencies,flush=True)
    a=tifffile.memmap(src)
    assert set(np.unique(a))=={1,2}
    b=np.where(a==2,0,255).astype('u1')
    tifffile.imwrite(RUN/'digital_rock/solid255_pore0.tiff',b,metadata={'axes':'ZYX'})
    b.tofile(RUN/'digital_rock/solid255_pore0.raw')
    pnm=RUN/'pore_network/input'
    (b==255).astype('u1').tofile(pnm/'sample16.raw')
    write_pnextract_mhd(pnm/'sample16.mhd',pnm/'sample16.raw',shape_zyx=b.shape,voxel_size_um=1.92,title='sample16')
    ds=(b[::2,::2,::2]==255).astype('u1')
    ds.tofile(RUN/'configs/field_ds2.raw')
    write_pnextract_mhd(RUN/'configs/field_ds2.mhd',RUN/'configs/field_ds2.raw',shape_zyx=ds.shape,voxel_size_um=3.84,title='field_ds2')
    meta={'输入':str(src),'sha256':hashlib.file_digest(src.open('rb'),'sha256').hexdigest(),'原始形状':list(b.shape),'孔隙率':float(np.mean(b==0)),
          '标签说明':'原始2为孔隙，1为固相；重映射为0/255。','频点':frequencies,'频点选择':'在51个实验点的排序索引上近似均匀选择7点，包含两个端点。',
          '计算说明':'x向、周期边界、全域ds2双精度GPU诊断；PNM使用原始全分辨率，无几何缩放。',
          '物性':'水0.12 S/m、固0.0024 S/m，epsr=80/7，表面电导和扩散系数均1.3e-9，eta0=0.01，动态孔径2.45913e-6 m；来自样品属性表模拟参数页，未拟合。',
          '求解':'complex128；界面和综合FFT；孔和膜采用Jacobi预条件，固相严格为0；真实相对残差1e-6，不接受未收敛结果。',
          '实验换算':'CSV电阻10000 ohm；长度3.6322 cm、直径2.52476 cm；sigma=L/(A*R*ratio*exp(i*phase))。'}
    (RUN/'configs/parameters_zh.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf8')
    del a,b,ds
    command([ROOT/'pnextract/bin/pnextract.exe','sample16.mhd'],pnm/'pnextract.log',pnm)
    command([sys.executable,ROOT/'code/scripts/pore_network/parse_pnextract_network.py','--prefix',pnm/'sample16','--outdir',RUN/'pore_network/network_parsed'],RUN/'provenance/parse.log')
    write_pore_throat_distribution_figure(pores_csv=RUN/'pore_network/network_parsed/pores.csv',throats_csv=RUN/'pore_network/network_parsed/throats.csv',figure_out=RUN/'pore_network/distribution.png',metadata_out=RUN/'pore_network/distribution.json',bins=32)
    print('NETWORK READY',flush=True)
    for mechanism in ['maxwell','all','pore','membrane']:
        args=[sys.executable,ROOT/'code/scripts/sip_simulation/run_sample89_3d_sip_comparison.py','--mhd',RUN/'configs/field_ds2.mhd','--network-dir',RUN/'pore_network/network_parsed',
              '--experiment-csv',exp,'--segmented-volume',RUN/'digital_rock/solid255_pore0.tiff','--out-dir',RUN,'--output-prefix',f'sample16_{mechanism}',
              '--sample-label','Sample 16: 7 frequencies, ds2 diagnostic','--experiment-label','LKC-16','--sample-length-cm','3.6322','--sample-diameter-cm','2.52476',
              '--dynamic-pore-size-m','2.45913e-6','--components',mechanism,'--frequencies',*map(str,frequencies),'--field-solve-mode','gpu-ac3d','--dtype','complex128',
              '--rtol','1e-6','--maxiter','5000','--residual-every','100','--progress-every','100','--preconditioner','jacobi' if mechanism in ['pore','membrane'] else 'fft']
        (RUN/f'configs/{mechanism}_command.json').write_text(json.dumps(list(map(str,args)),indent=2),encoding='utf8')
        print('START',mechanism,flush=True)
        try: command(args,RUN/f'provenance/{mechanism}.log')
        except subprocess.CalledProcessError: print('FAILED',mechanism,flush=True)
        else: print('DONE',mechanism,flush=True)
    shutil.copy2(__file__,RUN/'configs/run_package_snapshot.py')

if __name__=='__main__': main()
