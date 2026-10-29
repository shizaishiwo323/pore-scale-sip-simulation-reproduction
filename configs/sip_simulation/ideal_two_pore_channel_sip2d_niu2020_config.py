"""理想二维“大孔-狭长喉道-小孔”周期链 SIP：严格按 Niu et al. (2020) 算法的对照运行。

与 ``ideal_two_pore_channel_sip2d_config.py`` 的几何、网格、电学参数和频带完全相同，
只把孔（Stern）极化改为 Niu (2020) 原文做法：

1. 孔极化扩散系数 D+ = 1.3e-9 m2/s（Niu 2020 Berea 原文值），不用用户指定的 3.8e-12；
2. 孔极化只作用于孔体（pore node）尺寸分布，喉道不计 Stern 极化（Niu 网络中喉道只有膜极化）；
3. 孔极化上推用 Niu Eq.12 原式：按孔体面积加权的 C*(r) 与全局动态孔径 Lambda，
   所有水相同一增量（pore_mode = uniform，不使用 regional）；
4. Debye 分解半径换算同样用 D+ = 1.3e-9 m2/s，使正演与反算自洽。

仍不可避免的二维改写（与主配置相同，见 pore_scale_electrical_2d/polarization.py 文档）：
- 膜极化以喉道内强度量 sigma_w Z_dc / Z* 施加，而非 Niu Eq.12 对 1/Z* - 1/Z_dc 的上推
  （二维每单位纵深的喉道电导会使该增量依赖任意的出平面厚度）；
- 模型维度为二维（单位纵深挤出），不是三维。
"""

from __future__ import annotations

CONFIG = {
    "geometry": {
        # 与主配置相同（设计依据见主配置注释）。
        "big_radius_m": 20.0e-6,
        "small_radius_m": 5.0e-6,
        "throat_width_m": 0.1e-6,
        "throat_length_m": 15.0e-6,
        "solid_margin_m": 2.0e-6,
    },
    "grid": {
        # 与主配置相同。
        "coarse_step_m": 0.125e-6,
        "throat_cells": 8,
        "growth": 1.2,
    },
    "electrical": {
        # Niu et al. (2020) Berea Table 1 / Section 5.3。
        "water_conductivity_s_m": 0.043,
        "solid_conductivity_s_m": 0.0,
        "water_relative_permittivity": 80.0,
        "solid_relative_permittivity": 7.0,
        "epsilon_0_f_m": 8.85e-12,
    },
    "polarization": {
        # Niu et al. (2020) Berea：Sigma_S = 1.3e-9 S。
        "surface_conductance_s": 1.3e-9,
        # Niu et al. (2020) 原文孔极化扩散系数。
        "stern_diffusion_m2_s": 1.3e-9,
        # Niu et al. (2020)：膜极化体相扩散系数与极化率。
        "membrane_diffusion_m2_s": 1.3e-9,
        "membrane_polarizability": 0.01,
        # Niu 网络：孔极化只用孔体半径，喉道不计 Stern 极化。
        "throat_stern_polarization": False,
        # Niu Eq.9：孔径分布只取孔体（pore node）。
        "uniform_psd_kinds": ["pore_body"],
    },
    "sweep": {
        "frequency_min_hz": 1.0e-4,
        "frequency_max_hz": 1.0e5,
        "points_per_decade": 10,
        "direction": "x",
        "workers": 6,
        # 只跑 Niu 原式（uniform）；interfacial / membrane 与 pore_mode 无关。
        "runs": [
            {"name": "interfacial", "mechanism": "interfacial", "pore_mode": "uniform"},
            {"name": "pore_niu", "mechanism": "pore", "pore_mode": "uniform"},
            {"name": "membrane", "mechanism": "membrane", "pore_mode": "uniform"},
            {"name": "all_niu", "mechanism": "all", "pore_mode": "uniform"},
        ],
    },
    "debye": {
        "relaxation_frequency_min_hz": 1.0e-5,
        "relaxation_frequency_max_hz": 1.0e6,
        "bins": 111,
        "regularization": 1.0e-3,
        "include_permittivity": True,
        # 与正演 D+ 一致（Niu 2020）。
        "radius_diffusion_m2_s": 1.3e-9,
    },
}
