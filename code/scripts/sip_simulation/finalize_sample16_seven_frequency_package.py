"""核验并整理样品16本轮结果，不修改输入或数值结果。"""
from pathlib import Path
import json
import hashlib
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'results/sample16_gpu_7freq_20260912_2340'

def main():
    verification = json.loads((RUN/'provenance/verification.json').read_text())
    assert verification['gpu_solves'] == 28
    for mechanism in ['maxwell', 'pore', 'membrane', 'all']:
        command = json.loads((RUN/f'configs/{mechanism}_command.json').read_text())
        # JSON is also valid YAML; preserve the exact invocation beside each solve.
        for result in (RUN/f'simulation_sweeps/{mechanism}').glob('frequency_*/result.json'):
            (result.parent/'config.yml').write_text(json.dumps({'command':command,'parameters':'../../../configs/parameters_zh.json'},indent=2),encoding='utf8')
    for name in ['run_sample16_seven_frequency_package.py', 'plot_sample16_seven_frequency_package.py', Path(__file__).name]:
        shutil.copy2(ROOT/'code/scripts/sip_simulation'/name, RUN/'configs'/name)
    sources = [ROOT/'code/src/pore_scale_electrical'/name for name in ['ac3d_gpu.py','ac3d_active_domain.py','polarization.py']]
    sources.append(ROOT/'code/scripts/sip_simulation/run_sample89_3d_sip_comparison.py')
    snapshot = RUN/'configs/source_snapshot'
    snapshot.mkdir(exist_ok=True)
    hashes = {}
    for path in sources:
        if path.exists():
            shutil.copy2(path, snapshot/path.name)
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (RUN/'provenance/source_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf8')
    for relative in ['digital_rock/sample16_fiji3d.html','pore_network/sample16_network.html']:
        path = RUN/relative
        html = path.read_text(encoding='utf8')
        if 'id="vtk-view-container"' not in html:
            html = html.replace('<body>', '<body><div id="vtk-view-container" class="content" style="position:fixed;inset:0"></div>', 1)
        path.write_text(html,encoding='utf8')
    data = pd.read_csv(RUN/'source_data/four_mechanisms_7frequencies.csv')
    parameters_path = RUN/'configs/parameters_zh.json'
    parameters = json.loads(parameters_path.read_text(encoding='utf8'))
    parameters['孔膜初值'] = '无x向绕周期贯通时，解包孔隙坐标，构造每连通分量定标的线性初始电势。GPU仍组装并检验完整场方程残差，再从面电流求有效电导；不直接填零。初值已满足阈值时迭代数为0。'
    parameters_path.write_text(json.dumps(parameters,indent=2,ensure_ascii=False),encoding='utf8')
    rows = []
    for name, group in data.groupby('mechanism',sort=False):
        rows.append(f"| {name} | {len(group)} | {group.relative_residual_norm.max():.3e} |")
    report = '''# 样品16：7频点GPU机制分析

本结果为全域二倍降采样（400×333×333）的x方向双精度GPU诊断，不是原始800×666×666网格的正式复现，也不是实验参数拟合。

从原始CT重新生成二值体、重新运行全分辨率pnextract，并分别求解界面、孔、膜、综合四种机制。频点来自LKC-16实验记录：0.1、0.704961、6.343811、44.72136、315.267933、2837.038291、20000 Hz。每种机制仅7个频点。

## 收敛核验

| 机制 | GPU频点数 | 最大真实相对残差 |
|---|---:|---:|
'''+ '\n'.join(rows)+'''

## 结果解释与边界

- 原始体与ds2体在周期六邻域孔隙图中均只有y方向贯通，x方向不贯通。孔／膜机制的固相电导严格为零，因此x方向宏观响应应为零；求解得到的微小非零量须按数值残余解释。对应图用symlog保留正负号，不取绝对值。
- 孔／膜采用周期连通分量的几何电势初值，经GPU完整场方程检验并由面电流计算响应。初值达到容差时迭代数为0，这不代表跳过GPU残差检验。未收敛冷启动尝试保存在simulation_sweeps/diagnostic_pore_cold_start，未绘入比较图。相关测试17项通过。
- 界面／综合机制包含非零固相电导。其曲线与实验的差异反映当前模型及参数、方向、降采样的综合影响，不能据此认定已完成实验拟合。
- 相对残差合格不等于极小虚部具有相同相对精度，尤其低频界面虚部需要进一步容差敏感性验证。
- 对数图过滤实验虚部的3个非正值，CSV保留全部原始换算数据。模拟仅7点，实验显示完整51点。
- 三维HTML与分布图使用本轮重新提取数据；数字岩心显示降采样4，孔隙率按原始体计数为19.105887%。体素按1.92 µm各向同性处理。
- 孔隙网络HTML已在浏览器确认可见。数字岩心HTML导出成功但浏览器只显示边框，VTK.js报错getNumberOfComponents；该项暂不可用，不计为已验证三维交付。二值体与metadata保留，后续修复入口为code/scripts/digital_rock_visualization/render_segmented_core_fiji3d_html.py的体数据导出链路。该问题不影响数值场求解与本轮对比图。
- 主要比较图使用未经幅值缩放的实验换算数据。历史通用入口额外导出的scaled实验CSV不作为本图数据来源。

## 文件结构

```text
configs/             中文参数、准确命令、脚本快照、ds2场网格
digital_rock/        原始重映射二值体、Fiji风格HTML、metadata
pore_network/       全分辨率提取输入/日志、网络CSV、分布图、三维HTML
simulation_sweeps/  maxwell、pore、membrane、all
  <机制>/sweep_results.csv
  <机制>/frequency_*/result.json、residual_history.csv、config.yml
figures/            四机制总图与各机制实部/虚部对比PNG
source_data/        实验51点、选定7点、四机制28点源数据
provenance/         运行日志、收敛核验、连通性核验
```

复跑使用conda ml并设置PYTHONNOUSERSITE=1。运行脚本会写入固定运行目录；为保留本次结果，重新运行前须更改RUN为新目录。
'''
    (RUN/'README.md').write_text(report,encoding='utf8')
    print(verification)

if __name__ == '__main__':
    main()
