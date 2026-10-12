"""样品16原始 pnextract 参数敏感性：全分辨率裁块筛选，再验证完整体。

不改变 CT 分割、体素尺度、原始网络或第三方源码。所有候选均为诊断设置。
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from parse_pnextract_network import parse_network

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/sample16_pnextract_sensitivity_20260913'
BASE = ROOT / 'results/sample16_gpu_20260912/pore_network/full_resolution'
EXE = ROOT / 'pnextract/bin/pnextract.exe'
SHAPE = (800, 666, 666)
VOXEL = 1.92e-6
# 参数顺序严格对应本机 medialSurf.cpp；长度类参数单位为体素。
DEFAULT = [0.05, 0.98, 0.7, 2.75, 0.6, 1.1, 3, 0.15, 1.75]
CASES = {
    'default': (None, None, '内置默认值，作为同裁块基线'),
    'fixed175': (1.75, DEFAULT, '显式固定完整体实际默认参数，作为单参数试验对照；裁块自动阈值会变化'),
    'min125': (1.25, None, '仅指定 minRPore；注意同时自动改变 RMedSurfNoise 和 RCors'),
    'min100': (1.0, None, '进一步降低小孔半径筛选阈值；检验小孔保留能力'),
    'min100_fixed': (1.0, DEFAULT, '固定其余九个参数，隔离 minRPore 本身的影响'),
    'smooth1': (1.75, DEFAULT[:6] + [1] + DEFAULT[7:], '平滑次数从3减为1，保留局部极大值'),
    'noise1': (1.75, DEFAULT[:3] + [1.0] + DEFAULT[4:], '仅降低最大球噪声容限，减少候选球吞并'),
    'spacing': (1.75, DEFAULT[:7] + [0.075, 0.875], '减小最大球中心排斥距离'),
    'merge_low': (1.75, DEFAULT[:4] + [0.3, 1.02] + DEFAULT[6:], '收紧跨孔合并条件；配位数可能下降'),
    'mid090': (1.75, DEFAULT[:2] + [0.9] + DEFAULT[3:], '提高中点半径比例，限制跨越收缩部的父球连接'),
    'detail': (1.0, [0.05, 0.98, 0.7, 1.0, 0.3, 1.02, 1, 0.075, 1.0],
               '联合细分候选；只作为敏感性试验，需检查过分割和短喉下限堆积'),
    'detail_mid090': (1.0, [0.05, 0.98, 0.9, 1.0, 0.3, 1.02, 1, 0.075, 1.0],
                      '在联合细分候选上提高中点半径比例，测试对短孔喉的额外影响'),
    'detail_smooth3': (1.0, [0.05, 0.98, 0.7, 1.0, 0.3, 1.02, 3, 0.075, 1.0],
                       '在联合细分候选上恢复3次平滑，检查平滑和合并参数耦合'),
    'detail_min075': (0.75, [0.05, 0.98, 0.7, 1.0, 0.3, 1.02, 1, 0.075, 1.0],
                      '较激进的0.75体素孔半径筛选，仅作为分辨率敏感性边界'),
}


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def metrics(directory, name, scope):
    p = pd.read_csv(directory / 'pores.csv')
    all_t = pd.read_csv(directory / 'throats.csv')
    t = all_t[(all_t.pore1_id > 0) & (all_t.pore2_id > 0)]
    assert len(t) and p.pore_id.is_unique and all_t.throat_id.is_unique
    assert set(t.pore1_id) | set(t.pore2_id) <= set(p.pore_id)
    for frame, columns in [(p, ['pore_radius_m', 'pore_volume_m3']),
                           (all_t, ['throat_length_m', 'throat_radius_m', 'throat_volume_m3'])]:
        assert np.isfinite(frame[columns]).all().all() and (frame[columns] >= 0).all().all()
        assert (frame[[c for c in columns if 'volume' not in c]] > 0).all().all()
    ids = pd.Series(np.arange(len(p)), index=p.pore_id)
    a, b = ids.loc[t.pore1_id].to_numpy(), ids.loc[t.pore2_id].to_numpy()
    graph = coo_matrix((np.ones(len(a)*2), (np.r_[a,b], np.r_[b,a])), shape=(len(p),len(p)))
    nc, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    inlet = set(all_t.loc[(all_t.pore1_id == -1) | (all_t.pore2_id == -1), ['pore1_id','pore2_id']].to_numpy().ravel()) & set(p.pore_id)
    outlet = set(all_t.loc[(all_t.pore1_id == 0) | (all_t.pore2_id == 0), ['pore1_id','pore2_id']].to_numpy().ravel()) & set(p.pore_id)
    inlet_labels = set(labels[ids.loc[list(inlet)].to_numpy()])
    outlet_labels = set(labels[ids.loc[list(outlet)].to_numpy()])
    lengths = t.throat_length_m.to_numpy()*1e6
    result = dict(case=name, scope=scope, n_pores=len(p), n_throats_all=len(all_t),
                  zero_volume_pores=int((p.pore_volume_m3 == 0).sum()),
                  zero_volume_throats=int((all_t.throat_volume_m3 == 0).sum()),
                  n_internal=len(t), n_boundary=len(all_t)-len(t), mean_coordination=2*len(t)/len(p),
                  components=int(nc), largest_component_pore_fraction=float(sizes.max()/len(p)),
                  x_spanning=bool(inlet_labels & outlet_labels),
                  pore_diameter_median_um=float(np.median(p.pore_radius_m)*2e6),
                  throat_length_min_um=float(lengths.min()), throat_length_median_um=float(np.median(lengths)),
                  throat_length_vw_mean_um=float(np.average(lengths,weights=t.throat_volume_m3)),
                  internal_total_volume_m3=float(t.throat_volume_m3.sum()),
                  network_total_volume_m3=float(p.pore_volume_m3.sum()+all_t.throat_volume_m3.sum()))
    for threshold in [1.92, 2, 5, 10, 20]:
        mask = lengths < threshold
        result[f'n_lt_{threshold:g}um'] = int(mask.sum())
        result[f'fraction_lt_{threshold:g}um'] = float(mask.mean())
        result[f'volume_fraction_lt_{threshold:g}um'] = float(t.throat_volume_m3.to_numpy()[mask].sum()/t.throat_volume_m3.sum())
    return result


def extract(name, scope, raw, shape):
    folder = OUT / 'pore_network' / scope / name
    folder.mkdir(parents=True, exist_ok=True)
    rp, medial, why = CASES[name]
    # 用相对输入路径兼容旧 Windows 二进制；共享只读 RAW，不复制完整体。
    relative = os.path.relpath(raw, folder).replace('\\', '/')
    header = ['ObjectType = Image','NDims = 3','ElementType = MET_UCHAR',
              'DimSize = ' + ' '.join(map(str,shape[::-1])), 'ElementSize = 1.92 1.92 1.92',
              'Offset = 0 0 0', f'ElementDataFile = {relative}', '', 'title sample16',
              'write_cnm true', 'write_vtkNetwork false', 'write_elements false']
    if rp is not None:
        header.append(f'minRPore {rp};')
    if medial is not None:
        header.append('medialSurfaceSettings ' + ' '.join(map(str,medial)) + ';')
    mhd = folder / 'sample16.mhd'
    mhd.write_text('\n'.join(header)+'\n', encoding='utf8')
    config = {'case':name,'scope':scope,'minRPore':rp,'medialSurfaceSettings':medial,
              '说明':why,'input':str(raw),'shape_zyx':shape,'voxel_size_m':VOXEL,'downsample':1,
              'command':[str(EXE),mhd.name],'cwd':str(folder)}
    (OUT/'configs'/f'{scope}_{name}.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    parsed = folder / 'network_parsed'
    if not (folder/'completed.json').exists():
        start = time.perf_counter()
        with (folder/'pnextract.log').open('w',encoding='utf8') as log:
            subprocess.run([str(EXE),mhd.name],cwd=folder,stdout=log,stderr=subprocess.STDOUT,check=True)
        parse_network(folder/'sample16', parsed)
        log_text = (folder/'pnextract.log').read_text(encoding='utf8',errors='replace')
        actual = re.findall(r'medialSurfaceSettings: ([^\n]+)', log_text)[-1].strip()
        done = {'elapsed_seconds':time.perf_counter()-start,'actual_medial_settings':actual}
        (folder/'completed.json').write_text(json.dumps(done,indent=2),encoding='utf8')
    result = metrics(parsed,name,scope)
    actual = json.loads((folder/'completed.json').read_text())['actual_medial_settings']
    if medial is not None:
        assert np.allclose(np.fromstring(actual,sep=' '),medial), '生效参数与请求不同'
    result['actual_medial_settings'] = actual
    print(json.dumps(result),flush=True)
    return result


def finish():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator

    full = pd.DataFrame([metrics(BASE.parent/'network_parsed','default','full')] +
                        [metrics(p/'network_parsed',p.name,'full')
                         for p in sorted((OUT/'pore_network/full').iterdir()) if (p/'completed.json').exists()])
    full.to_csv(OUT/'source_data/full_metrics.csv',index=False)
    crop = pd.read_csv(OUT/'source_data/crop_metrics.csv')
    names = full.case.tolist()
    networks = {}
    source_hashes = {}
    for name in names:
        folder = BASE.parent/'network_parsed' if name == 'default' else OUT/'pore_network/full'/name/'network_parsed'
        p, t = pd.read_csv(folder/'pores.csv'), pd.read_csv(folder/'throats.csv')
        t = t[(t.pore1_id > 0) & (t.pore2_id > 0)]
        networks[name] = [(p.pore_radius_m.to_numpy()*2,p.pore_volume_m3.to_numpy()),
                          (t.throat_length_m.to_numpy(),t.throat_volume_m3.to_numpy())]
        source_hashes[name] = {f: {'path':str(folder/f),'sha256':digest(folder/f)} for f in ['pores.csv','throats.csv']}
    # 本轮展示短喉最多的诊断候选；零体积元素单独报告，不外推为物理最优。
    eligible = full[(full.zero_volume_pores == 0) & (full.zero_volume_throats == 0)]
    clean_case = eligible.sort_values('n_lt_10um',ascending=False).iloc[0]['case']
    chosen = full.sort_values('n_lt_10um',ascending=False).iloc[0]['case']
    edges = [np.geomspace(10**np.floor(np.log10(min(networks[n][j][0].min() for n in names))),
                         10**np.ceil(np.log10(max(networks[n][j][0].max() for n in names))),51) for j in range(2)]
    hist = {}
    rows = []
    for name in names:
        hist[name] = []
        for j, (x, w) in enumerate(networks[name]):
            counts = np.histogram(x,bins=edges[j])[0]
            volumes = np.histogram(x,bins=edges[j],weights=w)[0]
            assert counts.sum() == len(x) and np.isclose(volumes.sum(),w.sum(),rtol=1e-12,atol=0)
            vf, nf = volumes/w.sum(), counts/len(x)
            assert np.isclose(vf.sum(),1) and np.isclose(nf.sum(),1)
            hist[name].append((vf,nf))
            rows.extend(dict(case=name,population=['pore','internal_throat'][j],bin_left_m=a,
                             bin_right_m=b,count=int(c),volume_fraction=float(v),number_fraction=float(n))
                        for a,b,c,v,n in zip(edges[j][:-1],edges[j][1:],counts,vf,nf))
    pd.DataFrame(rows).to_csv(OUT/'source_data/shared_bin_distributions.csv',index=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':13,'mathtext.fontset':'stix',
                         'axes.linewidth':0.8,'xtick.direction':'in','ytick.direction':'in'})
    color_map = {'default':'#777777','min100':'#377eb8','min125':'#4b987b','detail_smooth3':'#e38b34'}
    labels = {'default':'Default','min100':'minRPore = 1.0','min125':'minRPore = 1.25','detail_smooth3':'Detail + smoothing 3'}
    for mode in ['reference','volume_comparison','number_comparison']:
        fig, axes = plt.subplots(1,2,figsize=(10.4,4.9))
        fig.subplots_adjust(left=.09,right=.98,bottom=.19,top=.83,wspace=.30)
        for j, ax in enumerate(axes):
            index = 1 if mode == 'number_comparison' else 0
            for k,name in enumerate(names):
                y = hist[name][j][index]
                if mode == 'reference':
                    if name != clean_case:
                        continue
                    centers = np.sqrt(edges[j][:-1]*edges[j][1:])
                    half = .27*np.log(edges[j][1:]/edges[j][:-1])
                    left, right = centers*np.exp(-half), centers*np.exp(half)
                    ax.bar(left,y,width=right-left,align='edge',color='#777777',edgecolor='black',linewidth=.6)
                else:
                    ax.stairs(y,edges[j],color=color_map[name],fill=True,alpha=.20)
                    ax.stairs(y,edges[j],color=color_map[name],linewidth=1.3,label=labels.get(name,name))
            ax.set(xscale='log',xlim=(edges[j][0],edges[j][-1]),ylim=(0,None),
                   xlabel=['Pore node diameter (m)','Pore throat length (m)'][j],
                   ylabel='Number fraction' if index else 'Volume fraction')
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(4))
            ax.tick_params(which='major',length=6)
            ax.tick_params(which='minor',length=3)
            ax.text(-.16,1.04,['(a)','(b)'][j],transform=ax.transAxes,fontsize=19)
        if mode != 'reference':
            handles, legend_labels = axes[0].get_legend_handles_labels()
            fig.legend(handles,legend_labels,loc='upper center',ncol=2,frameon=False,fontsize=11)
        else:
            fig.suptitle('Sample 16 | ' + labels.get(clean_case,clean_case) + ' (sensitivity)',fontsize=14,y=.96)
        fig.text(.53,.025,'Full resolution: 1.92 '+r'$\mu$m voxels; internal throats only',ha='center',fontsize=11)
        fig.savefig(OUT/'figures'/f'sample16_{mode}.png',dpi=600)
        fig.savefig(OUT/'figures'/f'sample16_{mode}_preview.png',dpi=150)
        plt.close(fig)
    baseline = full[full.case == 'default'].iloc[0]
    best = full[full.case == chosen].iloc[0]
    def table(df):
        columns = ['case','n_pores','n_internal','n_lt_5um','n_lt_10um','throat_length_median_um','zero_volume_pores','zero_volume_throats']
        heads = ['参数组','孔节点数','内部孔喉数','<5 μm','<10 μm','长度中位数/μm','零体积孔','零体积喉']
        body = ['| '+' | '.join(heads)+' |','|'+'---|'*len(heads)]
        for _,r in df.iterrows():
            body.append('| '+' | '.join(f'{r[c]:.2f}' if c == 'throat_length_median_um' else str(r[c]) for c in columns)+' |')
        return '\n'.join(body)
    rp, medial, _ = CASES[chosen]
    if medial is None:
        medial = DEFAULT if rp is None else DEFAULT[:3] + [rp+1] + DEFAULT[4:8] + [rp]
    report = f'''# 样品16 pnextract 参数敏感性分析

## 结论

在同一份完整分辨率 CT 上，本轮 `{chosen}` 参数组产生最多的短喉。内部孔喉数从 {int(baseline.n_internal):,} 增至 {int(best.n_internal):,}；小于10 μm的内部孔喉从 {int(baseline['n_lt_10um']):,} 增至 {int(best['n_lt_10um']):,}（增加 {(best['n_lt_10um']/baseline['n_lt_10um']-1)*100:.1f}%）；中位长度从 {baseline.throat_length_median_um:.2f} 降至 {best.throat_length_median_um:.2f} μm。

该组有 {int(best.zero_volume_pores)} 个零体积孔、{int(best.zero_volume_throats)} 条零体积喉。全体候选中，没有零体积元素且短喉最多的组为 `{clean_case}`。这一区分必须保留：短喉最多并不等于可直接用于正式输运计算。零体积元素原样保留在CSV中，数量分布计数、体积图权重为零；没有静默删除或用小正数替换。

实际下一步优先从 `minRPore 1.25;` 开始，仅追加这一行，其余参数由程序产生。该方案在完整体未出现零体积元素，且小于10 μm内部孔喉数量接近默认的两倍；仍需做输运性质验证。5 μm和10 μm仅为便于比较短喉数量的诊断统计阈值，不是拟合目标。

保守方案的体积分数曲线与默认结果接近，因为新增的小孔喉体积较小；数量分布的变化更明显。大孔仍占总体积主导，不能将短喉条数增加解释成短喉已经主导膜极化响应。

这是本轮有限参数候选中的几何细分结果，尚未通过渗透率或 SIP 响应验证，不能称为物理上最准确或全局最优网络。未改变原始分割、体素尺寸、第三方程序或旧网络，也没有把提取后的长度乘缩放因子。

## 完整体实测对比

{table(full)}

完整体为800×666×666体素，体素边长1.92 μm。全部孔喉数包含边界喉，评分与右侧尺寸图只使用两端孔ID均大于0的内部喉。所有孔节点都参与左侧分布；不排除孤立节点。

细分组小于10 μm孔喉的数量占比为 {best['fraction_lt_10um']*100:.2f}%，体积占比为 {best['volume_fraction_lt_10um']*100:.2f}%；默认组分别为 {baseline['fraction_lt_10um']*100:.2f}% 和 {baseline['volume_fraction_lt_10um']*100:.2f}%。新增短喉对数量分布的影响和体积分数分布的影响不同。

## 参数为什么有效

本机 `medialSurf.cpp` 先构建距离场与最大球，再筛选、组织父球层级并生成孔网络。参数顺序为 clipROutx、clipROutyz、midRFrac、RMedSurfNoise、lenNf、vmvRadRelNf、nRSmoothing、RCorsf、RCors。

| 参数 | 完整体默认 | 本轮细分候选 | 作用与实测解释 |
|---|---:|---:|---|
| minRPore | 1.75 | 1.0 | 候选最大球半径筛选门槛，单位体素；降低可保留更小节点，但不是直接指定孔喉长度 |
| RMedSurfNoise | 2.75 | 1.0 | 参与最大球包含判据与父球竞争；降低后保留更多几何细节 |
| RCorsf / RCors | 0.15 / 1.75 | 0.075 / 1.0 | 球中心排斥距离为 RCorsf×R+RCors；单独降低效果有限 |
| lenNf / vmvRadRelNf | 0.6 / 1.1 | 0.3 / 1.02 | 收紧跨孔合并条件；单独修改几乎没有改善，不应按参数名猜测喉长缩放 |
| nRSmoothing | 3 | 3 | 保留默认平滑；联合方案恢复3次比1次略好，参数影响并非单调 |
| midRFrac | 0.7 | 0.7 | 保留默认；提高到0.9可强烈细分，但激进联合方案出现零体积元素 |
| clipROutx / clipROutyz | 0.05 / 0.98 | 不变 | 控制边界外最大球截断，不将边界处理当作短喉优化手段 |

`minRPore` 未指定时，本版会在距离场完成后设为 min(1.25,平均距离场×0.25)+0.5。完整体为1.75，中心裁块为1.47486；所以裁块自动默认不等于完整体默认。`fixed175` 是裁块单参数对比的共同对照。只写 `minRPore 1.0` 时，还会连带把 RMedSurfNoise 设为2.0、RCors设为1.0；`min100_fixed` 用于区分阈值本身与这些联动。

## 中心裁块筛选

中心256³裁块保持1.92 μm体素，不降采样；孔隙率17.0033%，不同于完整体19.1059%。单裁块仅用于快速筛选，三组候选已在完整体重提取。裁块未暴露的零体积问题在完整体出现，说明不能仅靠裁块确认参数可用性。各次原始 node/link、MHD、日志与解析CSV均保留。

{table(crop)}

`detail_mid090` 和 `detail_min075` 虽产生更多短喉，但出现零体积孔或喉，未作为当前推荐结果，也未静默清除这些异常。它们表明追求条数会逐渐进入过分割区域。参数扫描没有穷举。

## 为什么不能简单提取到0.1 μm

本机 `blockNet_write_cnm.cpp` 的内部喉导出规则是：先将两侧孔中心到喉中心距离相加得到L，再给两侧孔分别分配其对应距离的67%；正常情况下净喉长为0.33L。当L<3体素时，总长被置为3.01体素。因此内部净喉长的理论下界约为0.99体素，即1.9008 μm。边界喉采用不同规则，不能用它们的短长度代表内部结构。

参考图只是排版与统计方式参考，不作为样品16的目标真值。样品16的1.92 μm体素也不能直接证明0.1 μm真实几何。若研究确实需要亚微米短喉，应优先获取更高分辨率成像；或另建有独立实验约束的亚网格模型。修改源码中0.67的长度分配规则会改变几何定义，必须作为独立模型验证，不能混作参数优化。

## 图与检验边界

`figures/sample16_reference.png` 是无零体积元素的保守候选 `{clean_case}` 的灰色窄柱双联图；`sample16_volume_comparison.png` 是默认、单阈值及联合细分的半透明体积分数叠加；`sample16_number_comparison.png` 用相同分箱显示数量分数。左轴明确使用孔节点直径2R，不将含糊的pore node size直接视作半径；右轴使用link2净孔喉长度，不使用link1的路径总长。每类分别按自身体积或数量归一化，50个统一对数分箱，无核密度平滑、截尾或横轴缩放，PNG为600 dpi。

细分组平均配位数为 {best.mean_coordination:.3f}（默认 {baseline.mean_coordination:.3f}），网络连通分量为 {int(best.components)}（默认 {int(baseline.components)}），最大连通分量孔节点比例为 {best.largest_component_pore_fraction*100:.2f}%（默认 {baseline.largest_component_pore_fraction*100:.2f}%）。X向是否贯通：默认 {baseline.x_spanning}，细分组 {best.x_spanning}。连通分量增加可能来自新增小孤立孔，不能仅凭条数判定更真实；CT的连通性问题不会由调参自动解决。

本轮只做孔网络提取参数诊断与分布图，不运行SIP扫频，不重建未改变的数字岩心HTML；后续物理验证应在保留原始网格与实验频点的条件下，检查网络输运性质及SIP响应。既有人工缩短有效喉长脚本与本轮重新提取路线不同，不作为证据。

## 可复跑设置

优先验证的保守设置：

```text
minRPore 1.25;
```

短喉最多的联合细分诊断设置：

```text
minRPore {rp};
medialSurfaceSettings {' '.join(map(str,medial))};
```

将这两行加入对应二值RAW的MHD，保持其余输入不变。`configs/full_*.json` 保留输入路径、命令与中文说明；`provenance/input_audit.json` 保留输入与二进制SHA256。正式默认参数未被替换。

官方手册将 medialSurfaceSettings 定义为可用于敏感性分析的技术参数，并给出九个参数的顺序：[pnextract guide](https://raw.githubusercontent.com/ImperialCollegeLondon/pnflow/master/doc/pnextract_guide.pdf)。本报告对本机行为的判断以项目源码、实际生效日志和实测输出为依据。
'''
    (OUT/'provenance/样品16孔喉提取参数分析.md').write_text(report,encoding='utf8')
    metadata = {'selected_case':chosen,'clean_case':clean_case,'reference_figure_case':clean_case,
                'selection_is_global_optimum':False,'sources':source_hashes,
                'normalization':'each population separately; shared 50 log bins; zero weights retained',
                'size_definitions':['pore diameter = 2R','internal net throat length from link2'],
                'statistics':'descriptive single CT parameter sensitivity, not independent sample replicates',
                'png_dpi':600,'histogram_sums_checked':True,'network_schema_and_references_checked':True}
    metadata['algorithm_sources'] = {f:digest(ROOT/'pnextract/src/pnm/pnextract'/f)
                                     for f in ['medialSurf.cpp','medialRadius.cpp','blockNet_write_cnm.cpp']}
    # 导出的内部净长度与总路径长度、两侧孔分配长度一致，容差按文本输出精度设置。
    for name in names:
        folder = BASE.parent/'network_parsed' if name == 'default' else OUT/'pore_network/full'/name/'network_parsed'
        t = pd.read_csv(folder/'throats.csv')
        t = t[(t.pore1_id>0)&(t.pore2_id>0)]
        assert np.allclose(t.pore1_length_m+t.pore2_length_m+t.throat_length_m,
                           t.pore_center_to_center_length_m,rtol=2e-6,atol=1e-12)
        regular = t[t.pore_center_to_center_length_m > 3.011*VOXEL]
        assert np.allclose(regular.throat_length_m,.33*regular.pore_center_to_center_length_m,rtol=2e-6,atol=1e-12)
        assert t.throat_length_m.min() >= .99*VOXEL*(1-2e-6)
    assert digest(BASE/'sample16.raw') == json.loads((OUT/'provenance/input_audit.json').read_text(encoding='utf8'))['source_sha256']
    metadata['length_partition_and_input_hash_checked'] = True
    (OUT/'provenance/figure_metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf8')
    shutil.copy2(__file__,OUT/'configs'/Path(__file__).name)
    print(json.dumps({'selected':chosen,'summary':best.to_dict()},indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['crop','full','finish'])
    parser.add_argument('--cases', nargs='+')
    args = parser.parse_args()
    for sub in ['configs','provenance','source_data','figures','pore_network','inputs']:
        (OUT/sub).mkdir(parents=True,exist_ok=True)
    if args.stage == 'finish':
        finish()
        return
    if args.stage == 'crop':
        raw = OUT/'inputs/center256_pore0_solid1.raw'
        volume = np.memmap(BASE/'sample16.raw',dtype=np.uint8,mode='r',shape=SHAPE)
        # 全分辨率中心裁块，保留1.92微米体素；不是降采样。
        slices = tuple(slice((n-256)//2,(n-256)//2+256) for n in SHAPE)
        crop = np.asarray(volume[slices])
        assert set(np.unique(crop)) == {0,1}
        if not raw.exists():
            crop.tofile(raw)
        meta = {'性质':'参数敏感性诊断，不是已验证最优算法', 'raw_source':str(BASE/'sample16.raw'),
                'source_sha256':digest(BASE/'sample16.raw'),'executable_sha256':digest(EXE),
                'shape_zyx':SHAPE,'crop_slices_zyx':[[s.start,s.stop] for s in slices],
                'crop_porosity':float(np.mean(crop==0)),'voxel_size_m':VOXEL,'downsample':1,
                'population':'长度评分只用内部孔喉；边界孔喉单独记录',
                'selection':'优先检查 <10 um 内部孔喉绝对数，并同时检查中位数、体积和连通性',
                'source_definitions':'孔节点直径=2*radius；长度=link2净喉长；不缩放几何',
                'figure_contract':'定量双联图；同箱同轴默认与候选叠加；体积分数为主，数量分布为补充；600 dpi PNG'}
        (OUT/'provenance/input_audit.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
        rows=[]
        for name in args.cases or CASES:
            rows.append(extract(name,'crop256',raw,(256,256,256)))
            pd.DataFrame(rows).to_csv(OUT/'source_data/crop_metrics.csv',index=False)
    else:
        rows=[metrics(BASE.parent/'network_parsed','default','full')]
        for name in args.cases or ['min100','detail']:
            rows.append(extract(name,'full',BASE/'sample16.raw',SHAPE))
            pd.DataFrame(rows).to_csv(OUT/'source_data/full_metrics.csv',index=False)
    shutil.copy2(__file__,OUT/'configs'/Path(__file__).name)


if __name__ == '__main__':
    main()
