"""bmsoc —— 电池 SOC 在线估计（EKF / UKF / AEKF）参考实现。

模块划分
--------
- ``io_panasonic`` : Panasonic 18650PF（Kollmeyer）动态工况与 HPPC 脉冲读取
- ``io_calce``     : CALCE A123 OCV 测试与 DST/US06/FUDS 工况读取
- ``profiles``     : 统一工况容器、重采样、真值 SOC 构造
- ``ocv``          : OCV(SOC) 曲线（单调化 + 温度插值）
- ``ecm``          : 二阶 RC 等效电路（含精确零阶保持离散化）
- ``filters``      : EKF / UKF / AEKF 三版估计器
- ``metrics``      : SOC 误差指标与收敛时间
"""
from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["io_panasonic", "io_calce", "profiles", "ocv", "ecm", "filters", "metrics"]
