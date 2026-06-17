"""Niu 2020 Berea AC3D/SIP 复现参数。

这是可执行 Python 配置文件，也是本项目复现 Niu et al. (2020) Berea
砂岩 SIP 模拟时的参数清单。每个数值都尽量保留论文锚点，便于后期
核对：

- Figure 3 / Section 4: Berea 砂岩二值 micro-CT REV，350^3 体素，
  体素边长 2.8 um。
- Section 4.1 / Table 1: 样品孔隙度、比表面积、formation factor、
  平均孔径和水相电导率。
- Section 4.2 / Table 1: AC3D 相属性、EDL 孔极化、膜极化和动态孔径
  参数。
- Equation 12: 将孔/膜极化复电导 C* 转换为水相体积复电导率增量。
- Equation 17 / Equation 18: Schwarz 型孔极化 C_p* 与 tau_p。
- Equation 19 / Equation 20 / Equation 21: Titov 型膜极化 Z_m*、tau_m
  与 C_m*。
- Section 5.3: interfacial / pore / membrane / all 单机制模拟定义。

除非字段明确标注为 project diagnostic/provenance，参数均按论文描述取值。
不要把 Figure8.xlsx 的 Simulation 列当成本项目模拟结果；正式曲线必须
追溯到本项目 sweep_results.csv。
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PDF = (
    PROJECT_ROOT
    / "docs"
    / "references"
    / "JGR Solid Earth - 2020 - Niu - A Framework for Pore-Scale Simulation of Effective Electrical Conductivity and Permittivity.pdf"
)
DATA_DIR = PROJECT_ROOT / "data" / "Niu 2020data"


class CTConfig(NamedTuple):
    """数字岩心输入：AC3D 有限差分场求解使用的 Berea 砂岩分割体。"""

    raw_path: Path
    tiff_path: Path
    shape_zyx: tuple[int, int, int]
    voxel_size_um: float
    voxel_size_m: float
    pore_label: int
    solid_label: int
    note: str


class MaterialConfig(NamedTuple):
    """相属性：水相和非金属固相的本征电导率/介电常数。"""

    water_conductivity_s_m: float
    solid_conductivity_s_m: float
    water_relative_permittivity: float
    solid_relative_permittivity: float
    epsilon0_f_m: float
    note: str


class PetrophysicalConfig(NamedTuple):
    """Table 1 岩石物性：用于论文参数核对和 provenance，不覆盖体素审计。"""

    porosity_fraction: float
    specific_surface_area_m2_g: float
    formation_factor: float
    mean_pore_size_um: float
    note: str


class PolarizationConfig(NamedTuple):
    """电化学极化参数：孔极化、膜极化和动态孔径上尺度。"""

    surface_conductance_s: float
    diffusion_coefficient_m2_s: float
    dynamic_pore_size_m: float
    membrane_polarizability: float
    pore_relaxation_time_formula: str
    membrane_relaxation_time_formula: str
    upscaling_formula: str
    geometry_note: str


class FrequencyConfig(NamedTuple):
    """频率范围：论文实验/模拟对比覆盖 1 mHz 到 1 GHz。"""

    min_hz: float
    max_hz: float
    paper_default_count: int
    local_available_sweep_count: int
    note: str


class MechanismConfig(NamedTuple):
    """Section 5.3 单机制材料赋值定义。"""

    definitions: dict[str, str]
    plotting_labels: dict[str, str]


class InputPolicy(NamedTuple):
    """输入数据红线：限定本复现步骤允许使用的数据来源。"""

    allowed_ct_files: tuple[str, ...]
    allowed_geometry_workbooks: tuple[str, ...]
    allowed_experiment_workbooks: tuple[str, ...]
    paper_simulation_columns_used_as_simulation: bool
    post_extraction_geometry_scaling_allowed: bool
    default_pnextract_executable: Path
    note: str


class Niu2020BereaConfig(NamedTuple):
    """Niu 2020 Berea SIP 复现的完整参数包。"""

    reference_pdf: Path
    data_dir: Path
    ct: CTConfig
    petrophysical: PetrophysicalConfig
    materials: MaterialConfig
    polarization: PolarizationConfig
    frequencies: FrequencyConfig
    mechanisms: MechanismConfig
    input_policy: InputPolicy


# CT 体素尺寸信息：
# - 论文 Figure 3 标注 Berea 砂岩二值图像为 350^3 voxels，
#   每个体素尺寸为 2.8 x 2.8 x 2.8 um。
# - 本项目 microCT_Berea.raw 按 little-endian uint16 读取，形状为
#   (z, y, x) = (350, 350, 350)。这一步只读取原始 CT，不覆盖或改名。
# - 本地 raw 标签约定为 1 = pore/water，2 = solid；TIFF 是同一分割体
#   的图像栈副本，后续可视化会另外生成 pore=0/solid=255 的派生二值体。
CT = CTConfig(
    raw_path=DATA_DIR / "microCT_Berea.raw",
    tiff_path=DATA_DIR / "microCT_Berea.tiff",
    shape_zyx=(350, 350, 350),
    voxel_size_um=2.8,
    voxel_size_m=2.8e-6,
    pore_label=1,
    solid_label=2,
    note="Figure 3 / Section 4: Berea segmented REV for the 350^3 AC3D solve; labels follow the local Niu data copy.",
)


# Table 1 岩石物性参数：
# - 孔隙度 phi = 20.2%，比表面积 Sm = 0.5 m2/g，formation factor F = 16.3，
#   mean pore size rm = 35 um。
# - 这些是论文报告的样品尺度物性，用于校验和说明；它们不强行覆盖
#   microCT_Berea.raw 的体素计数孔隙度，因为分割 REV 与实验样品统计口径
#   可能不同。
PETROPHYSICAL = PetrophysicalConfig(
    porosity_fraction=0.202,
    specific_surface_area_m2_g=0.5,
    formation_factor=16.3,
    mean_pore_size_um=35.0,
    note="Table 1 Berea sample properties; do not overwrite raw label-count porosity audits with this value.",
)


# 固体、液体的电导率和介电常数：
# - Section 4.1 说明实验用 NaCl 溶液饱和，水相电导率 sigma_w = 0.043 S/m；
#   Table 1 也列出该值。
# - Berea 砂岩主要为石英，非金属矿物固相按 nonconductive 处理，因此
#   固相 dc 电导率设为 0 S/m。
# - Section 4.2 / Table 1 给出水相本征介电常数 epsilon_w = 80 epsilon0，
#   固相本征介电常数 epsilon_s = 7 epsilon0。
# - Table 1 note 给出真空介电常数 epsilon0 = 8.85e-12 F/m。
MATERIALS = MaterialConfig(
    water_conductivity_s_m=0.043,
    solid_conductivity_s_m=0.0,
    water_relative_permittivity=80.0,
    solid_relative_permittivity=7.0,
    epsilon0_f_m=8.85e-12,
    note="Section 4.2 / Table 1: water dc conductivity plus high-frequency dielectric storage; solid has zero dc conductivity.",
)


# 孔极化、膜极化和动态孔径：
# - Section 4.2 / Table 1 给出 Sigma_S = 1.3e-9 S、D = 1.3e-9 m2/s、
#   Lambda = 2.7 um、膜极化 polarizability eta0 = 1%。
# - Equation 17: C_p* = Sigma_S * i omega tau_p / (1 + i omega tau_p)，
#   即 Schwarz 型孔极化复电导；Equation 18: tau_p = r^2 / (2D)。
# - 孔极化使用 Figure 4a / Figure5.xlsx 的 pore node size distribution；
#   r 是孔半径/孔节点尺寸相关特征长度。
# - Equation 19: Titov 型膜极化复阻抗 Z_m*；Equation 20:
#   tau_m = L^2 / (4D)；Equation 21: C_m* = 1/Z_m* - 1/Zdc。
# - 膜极化使用论文 Figure 4b 分布，或本项目 pnextract 孔喉长度/阻抗统计；
#   Zdc 由孔喉几何、孔喉长度和水电导率计算。
# - Equation 12: 动态孔径 Lambda 用于上尺度，Delta sigma_w* = 2 C* / Lambda。
POLARIZATION = PolarizationConfig(
    surface_conductance_s=1.3e-9,
    diffusion_coefficient_m2_s=1.3e-9,
    dynamic_pore_size_m=2.7e-6,
    membrane_polarizability=0.01,
    pore_relaxation_time_formula="tau_p = r^2 / (2D)",
    membrane_relaxation_time_formula="tau_m = L^2 / (4D)",
    upscaling_formula="Delta sigma_w* = 2 C* / Lambda",
    geometry_note=(
        "Pore radii and throat lengths/resistances are distribution-driven; "
        "the paper Figure 4 distributions are stored locally as Figure5.xlsx, "
        "while full local membrane runs use pnextract-derived throat geometry."
    ),
)


# 频率范围：
# - 论文标题、摘要、Section 4 和 Figures 6-7 都描述对比频带为
#   10^-3 到 10^9 Hz。
# - Figure8.xlsx 中含有论文作者的 simulation/component 数值块；本地工作簿
#   的这些块有 40 个频点。该数量用于 provenance，不代表论文正文另有
#   “40 个频点”的文字说明。这些数值块只可用于对照或误差审计，不作为
#   本项目复现模拟曲线。
FREQUENCIES = FrequencyConfig(
    min_hz=1.0e-3,
    max_hz=1.0e9,
    paper_default_count=40,
    local_available_sweep_count=12,
    note="Local historical full350 GPU sweeps available here are representative 12-frequency sweeps.",
)


# Section 5.3 单机制拆分：
# - interfacial / Maxwell-Wagner：只给水相和固相赋 dc conductivity 与
#   high-frequency permittivity；不加入孔极化或膜极化。
# - pore polarization：水相为 sigma_w + Delta sigma_pore*；solid phase is zero；
#   不加入 Maxwell 背景和膜极化。
# - membrane polarization：水相为 sigma_w + Delta sigma_membrane*；solid phase is zero；
#   不加入 Maxwell 背景和孔极化。
# - all：水相包含 sigma_w、高频介电项、Delta sigma_pore* 和
#   Delta sigma_membrane*；固相包含高频介电项，dc 电导率仍为 0。
MECHANISMS = MechanismConfig(
    definitions={
        "interfacial": "water and solid receive only dc conductivity and high-frequency permittivity; no pore or membrane increment",
        "pore": "water phase is sigma_w + Delta sigma_pore*; solid phase is zero",
        "membrane": "water phase is sigma_w + Delta sigma_membrane*; solid phase is zero",
        "all": "water includes sigma_w, high-frequency permittivity, Delta sigma_pore*, and Delta sigma_membrane*; solid includes high-frequency permittivity",
    },
    plotting_labels={
        "interfacial": "Dielectric / interfacial",
        "pore": "EDL / pore",
        "membrane": "Membrane",
        "all": "EDL + Dielectric + Membrane",
    },
)


INPUT_POLICY = InputPolicy(
    allowed_ct_files=("microCT_Berea.raw", "microCT_Berea.tiff"),
    allowed_geometry_workbooks=("Figure5.xlsx",),
    allowed_experiment_workbooks=("Figure7.xlsx", "Figure8.xlsx"),
    paper_simulation_columns_used_as_simulation=False,
    post_extraction_geometry_scaling_allowed=False,
    default_pnextract_executable=PROJECT_ROOT / "code" / "vendor" / "pnextract" / "bin" / "pnextract.exe",
    note=(
        "Use Figure7/Figure8 only for experimental scatter. Figure8 paper "
        "simulation, pore, membrane, and interfacial columns are paper-reference "
        "blocks. 不要把 Figure8.xlsx 的 Simulation 列当成本项目模拟结果。"
        "不允许对 pnextract 提取后的孔径、孔喉长度或由几何计算的 Zdc 再施加缩放因子；"
        "若几何不匹配，必须用原版 pnextract 算法和明确记录的参数重新提取；默认不追加 "
        "minRPore 或 medialSurfaceSettings 校准行，使用 pnextract 内置默认参数。"
    ),
)


NIU2020_BEREA_CONFIG = Niu2020BereaConfig(
    reference_pdf=REFERENCE_PDF,
    data_dir=DATA_DIR,
    ct=CT,
    petrophysical=PETROPHYSICAL,
    materials=MATERIALS,
    polarization=POLARIZATION,
    frequencies=FREQUENCIES,
    mechanisms=MECHANISMS,
    input_policy=INPUT_POLICY,
)
