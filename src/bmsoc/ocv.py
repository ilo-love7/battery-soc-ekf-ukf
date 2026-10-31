"""OCV(SOC) 曲线：单调化、温度插值、导数。

两条来源
--------
1. **HPPC 静置法**（Panasonic）：五脉冲测试每放 10% SOC 静置，取静置末电压为 OCV，
   SOC 由测试自带的累计电量给出。这是估计器实际使用的曲线。
2. **小电流分段法**（CALCE OCV 包）：低倍率充放 + 长静置，用于跨化学体系对照。

单调性是硬约束：OCV 必须随 SOC 非降，否则 EKF 的观测雅可比会变号、滤波器发散。
实测点若因滞回或温度漂移出现倒挂，用保序回归（PAVA）压平，并记录被改动的幅度。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import isotonic_regression


def _pava(y: np.ndarray) -> np.ndarray:
    """保序回归（非降），返回拟合值。"""
    y = np.asarray(y, dtype=float)
    if y.size <= 1:
        return y.copy()
    return np.asarray(isotonic_regression(y, increasing=True).x, dtype=float)


def _linear_extrap(s: np.ndarray, base: np.ndarray, lo: float, hi: float,
                   d_lo: float, d_hi: float) -> np.ndarray:
    """区间内用插值值，区间外**按端点斜率线性外推**（而不是夹紧）。

    为什么不能用 ``np.clip`` 夹紧：无迹卡尔曼的 sigma 点会越过 SOC=0/1，若 OCV 在
    边界外被夹成常数，``E[OCV(soc)]`` 就会系统性低于 ``OCV(mu)``，滤波器于是持续
    把 SOC 往边界上推、卡在限幅里不动（本仓库在 -20 °C 上实测到的 UKF 发散就是
    这个机制）。线性外推既保持了 C0 连续与单调，也让 sigma 点的均值无偏。
    """
    out = np.array(base, dtype=float, copy=True)
    below = s < lo
    above = s > hi
    if np.any(below):
        out[below] = base[below] + (s[below] - lo) * d_lo
    if np.any(above):
        out[above] = base[above] + (s[above] - hi) * d_hi
    return out


@dataclass
class OCVCurve:
    """单温度 OCV(SOC)，PCHIP 插值 + 保序约束。"""

    soc: np.ndarray
    ocv: np.ndarray
    temp_c: float = float("nan")
    source: str = ""
    raw_soc: Optional[np.ndarray] = None        # 平滑前的原始静置点
    raw_ocv: Optional[np.ndarray] = None
    smooth_residual_v: float = 0.0

    def __post_init__(self) -> None:
        soc = np.asarray(self.soc, dtype=float)
        ocv = np.asarray(self.ocv, dtype=float)
        order = np.argsort(soc)
        soc, ocv = soc[order], ocv[order]
        # 去重（同一 SOC 多点时取均值）
        uniq, inv = np.unique(np.round(soc, 9), return_inverse=True)
        if uniq.size < soc.size:
            agg = np.zeros(uniq.size)
            cnt = np.zeros(uniq.size)
            np.add.at(agg, inv, ocv)
            np.add.at(cnt, inv, 1.0)
            ocv = agg / np.maximum(cnt, 1.0)
            soc = uniq
        ocv_mono = _pava(ocv)
        self.soc = soc
        self.ocv = ocv_mono
        self.violation_v = float(np.max(np.abs(ocv - ocv_mono))) if ocv.size else 0.0
        self._f = PchipInterpolator(self.soc, self.ocv, extrapolate=True)
        span_lo = self.soc[1] - self.soc[0]
        span_hi = self.soc[-1] - self.soc[-2]
        self._d_lo = float((self.ocv[1] - self.ocv[0]) / span_lo) if span_lo > 0 else 0.0
        self._d_hi = float((self.ocv[-1] - self.ocv[-2]) / span_hi) if span_hi > 0 else 0.0

    def __call__(self, soc) -> np.ndarray:
        s = np.asarray(soc, dtype=float)
        base = self._f(np.clip(s, self.soc[0], self.soc[-1]))
        return _linear_extrap(s, base, float(self.soc[0]), float(self.soc[-1]),
                              self._d_lo, self._d_hi)

    def docv(self, soc, h: float = 1e-4) -> np.ndarray:
        """dOCV/dSOC（中心差分；用 1e-4 而非解析导数，避免 PCHIP 端点病态）。"""
        s = np.asarray(soc, dtype=float)
        lo = np.clip(s - h, self.soc[0], self.soc[-1])
        hi = np.clip(s + h, self.soc[0], self.soc[-1])
        d = hi - lo
        d = np.where(np.abs(d) < 1e-12, 1e-12, d)
        out = (self._f(hi) - self._f(lo)) / d
        return np.where(s < self.soc[0], self._d_lo,
                        np.where(s > self.soc[-1], self._d_hi, out))

    def inverse(self, voltage, n_grid: int = 2001) -> np.ndarray:
        """OCV → SOC（查表反解，用于静置点定标）。"""
        grid = np.linspace(self.soc[0], self.soc[-1], n_grid)
        v = self._f(grid)
        return np.interp(voltage, v, grid)

    def to_dict(self) -> Dict[str, object]:
        return {"temp_c": self.temp_c, "soc": self.soc.tolist(), "ocv": self.ocv.tolist(),
                "violation_v": self.violation_v, "source": self.source,
                # 落盘的是**最终曲线**（可能已经平滑过）。读取方必须据此决定要不要再平滑，
                # 否则会二次平滑把曲线压坏——本项目踩过一次。
                "smoothed": self.raw_soc is not None,
                "smooth_residual_v": self.smooth_residual_v}


@dataclass
class OCVTable:
    """多温度 OCV：温度方向线性插值，SOC 方向 PCHIP。"""

    curves: Dict[float, OCVCurve]

    def temps(self) -> np.ndarray:
        return np.array(sorted(self.curves), dtype=float)

    def at(self, temp_c: float) -> OCVCurve:
        temps = self.temps()
        if temp_c in self.curves:
            return self.curves[temp_c]
        if temp_c <= temps[0]:
            return self.curves[temps[0]]
        if temp_c >= temps[-1]:
            return self.curves[temps[-1]]
        lo = temps[temps <= temp_c].max()
        hi = temps[temps >= temp_c].min()
        w = (temp_c - lo) / (hi - lo)
        soc = self.curves[lo].soc
        ocv = (1 - w) * self.curves[lo](soc) + w * self.curves[hi](soc)
        return OCVCurve(soc, ocv, temp_c=temp_c, source=f"interp({lo:.0f},{hi:.0f})")


def from_rest_points(soc: Sequence[float], ocv: Sequence[float], temp_c: float = float("nan"),
                     source: str = "rest", smooth_mv: float = 1.5,
                     n_grid: int = 201) -> OCVCurve:
    """静置点 → OCV 曲线（自动排序、去重、保序、可选平滑）。

    为什么必须平滑
    --------------
    HPPC 每个 SOC 档有 5 个不同倍率的脉冲，它们的「脉冲前静置末电压」被记成 5 个
    几乎同 SOC 的点（档内 SOC 跨度约 0.005），电压差只有 0.6–8 mV——本质是**重复测量**，
    不是真实的斜率。−20 °C 下更极端：35 个点里有 25 对相邻点的电压完全相同（记录分辨率
    0.1 mV，20 min 静置在低温下几乎不松弛），PAVA 之后整段变成阶梯。

    直接让 PCHIP 穿过阶梯，`dOCV/dSOC` 会在 0.02 与 2.9 V/unit 之间乱跳（25 °C 实测），
    而滤波器正是用这个斜率算卡尔曼增益的。用平滑样条（残差目标 = 记录精度量级）
    在 SOC 方向重建曲线，斜率恢复到物理上合理的范围（0.43–2.87）。

    ``smooth_mv=0`` 关闭平滑；点数少于 12 时自动跳过（没有阶梯可压）。
    """
    s = np.asarray(list(soc), dtype=float)
    v = np.asarray(list(ocv), dtype=float)
    if s.size < 3:
        raise ValueError(f"静置点过少（{s.size}），无法建 OCV 曲线")
    curve = OCVCurve(s, v, temp_c=temp_c, source=source)
    if smooth_mv <= 0 or s.size < 12:
        return curve
    from scipy.interpolate import UnivariateSpline

    sig = float(smooth_mv) / 1000.0
    # OCVCurve.__post_init__ 已经按 SOC 排序去重，这里直接用（不要再按原始顺序取）
    ss, vv = curve.soc, curve.ocv
    spl = UnivariateSpline(ss, vv, k=3, s=(sig ** 2) * ss.size)
    grid = np.linspace(float(ss[0]), float(ss[-1]), int(n_grid))
    vals = _pava(np.asarray(spl(grid), dtype=float))
    # PAVA 会把样条的轻微非单调区压回**阶梯**（斜率为 0），而阶梯正是要消灭的东西。
    # 对单调序列做窗口平均仍然单调，但阶梯会变成斜坡——这一步是平滑能不能用的关键。
    k = 5
    if vals.size > 2 * k:
        h = k // 2
        # 两端用**线性外推**填，避免用端点值填（那会把首尾压出 S 形，
        # 让 dOCV/dSOC 在 SOC≈0/1 处偏掉——直线型 OCV 必须仍然是直线）
        left = vals[0] - np.arange(h, 0, -1) * (vals[1] - vals[0])
        right = vals[-1] + np.arange(1, h + 1) * (vals[-1] - vals[-2])
        ext = np.concatenate([left, vals, right])
        vals = np.convolve(ext, np.ones(k) / k, mode="valid")
    fitted = OCVCurve(grid, vals, temp_c=temp_c,
                      source=f"{source}+smooth({smooth_mv:g}mV)")
    fitted.raw_soc, fitted.raw_ocv = ss, vv
    fitted.smooth_residual_v = float(np.sqrt(np.mean((spl(ss) - vv) ** 2)))
    return fitted


def oof_rmse(curve: OCVCurve, soc: Iterable[float], ocv: Iterable[float]) -> float:
    """曲线对留出点的拟合误差（用于报告 OCV 表质量）。"""
    s = np.asarray(list(soc), dtype=float)
    v = np.asarray(list(ocv), dtype=float)
    return float(np.sqrt(np.mean((curve(s) - v) ** 2)))


def voltage_gap(curve: OCVCurve, soc: float = 0.5) -> float:
    """中段斜率倒数（V per 1e-3 SOC）——LFP 平台区会很大。"""
    d = float(curve.docv(np.array([soc]))[0])
    return float(1e-3 / max(d, 1e-9))


class OCV2D:
    """OCV(SOC, T)：温度方向线性插值，SOC 方向线性插值（共用 SOC 网格）。

    为什么必须要二维
    ----------------
    这批测试里有大量「边降温边跑」的工况：标称 10 °C 的文件里电芯实测温度从
    23.7 °C 一路降到 12 °C，标称 -20 °C 的从 16 °C 降到 -20 °C。若整段当成一个
    温度，OCV 会错到几十毫伏，滤波器会把它算成 SOC 偏差。
    BMS 本来就有电芯温度，所以按**实测温度**查表既是物理正确的，也是可部署的。
    """

    def __init__(self, table: OCVTable, n_grid: int = 401):
        self.table = table
        self.soc_grid = np.linspace(0.0, 1.0, n_grid)
        self.temps = table.temps()
        self.matrix = np.vstack([table.curves[t](self.soc_grid) for t in self.temps])
        self._cache: Dict[float, np.ndarray] = {}

    def row(self, temp_c: float) -> np.ndarray:
        key = round(float(temp_c) * 2.0) / 2.0      # 0.5 K 量化，缓存
        row = self._cache.get(key)
        if row is None:
            if key <= self.temps[0]:
                row = self.matrix[0]
            elif key >= self.temps[-1]:
                row = self.matrix[-1]
            else:
                j = int(np.searchsorted(self.temps, key))
                w = (key - self.temps[j - 1]) / (self.temps[j] - self.temps[j - 1])
                row = (1 - w) * self.matrix[j - 1] + w * self.matrix[j]
            if len(self._cache) < 512:
                self._cache[key] = row
        return row

    def __call__(self, soc, temp_c: float) -> np.ndarray:
        s = np.asarray(soc, dtype=float)
        row = self.row(temp_c)
        base = np.interp(np.clip(s, 0.0, 1.0), self.soc_grid, row)
        d_lo, d_hi = self._edge_slope(row)
        return _linear_extrap(s, base, 0.0, 1.0, d_lo, d_hi)

    def _edge_slope(self, row: np.ndarray):
        h = float(self.soc_grid[1] - self.soc_grid[0])
        return float((row[1] - row[0]) / h), float((row[-1] - row[-2]) / h)

    def docv(self, soc, temp_c: float, h: float = 1e-4) -> np.ndarray:
        row = self.row(temp_c)
        s = np.asarray(soc, dtype=float)
        lo = np.clip(s - h, 0.0, 1.0)
        hi = np.clip(s + h, 0.0, 1.0)
        d = np.maximum(hi - lo, 1e-12)
        out = (np.interp(hi, self.soc_grid, row) - np.interp(lo, self.soc_grid, row)) / d
        d_lo, d_hi = self._edge_slope(row)
        return np.where(s < 0.0, d_lo, np.where(s > 1.0, d_hi, out))

    def batch(self, soc: np.ndarray, temp_c: np.ndarray) -> np.ndarray:
        """按样本温度批量求 OCV（温度量化到 0.5 K 后分组，避免逐点开销）。"""
        s = np.asarray(soc, dtype=float)
        t = np.asarray(temp_c, dtype=float)
        keys = np.round(t * 2.0) / 2.0
        out = np.empty_like(s)
        for k in np.unique(keys):
            m = keys == k
            out[m] = self(s[m], float(k))
        return out


class ShapeCorrected:
    """OCV 形状修正包装：``OCV'(soc,T) = OCV(soc,T) + Δ(soc)``。

    Δ 是分段线性的 SOC 相关加性修正（结点上以 mV 给出）。残差摸底（脚本 19/20）
    表明 LFP 的 LOO 开环残差主体是 OCV 表的**形状误差**（老化后的电芯在动态工况
    下与 C/20 静置曲线差 100–400 mV 的 SOC 相关形状），常值 ``dv`` 与极化支路
    都吃不掉它；把训练折残差按 SOC 分箱得到的 Δ 回灌到留出折，残差 37.4 →
    17.6 mV。该类就是那个 Δ 的滤波器侧载体。

    ``docv`` 同步修正斜率（Δ 的导数叠加到原斜率上），否则 EKF 的 ``H`` 会错。
    """

    def __init__(self, base, soc_knots, docv_mv):
        self.base = base
        self.knots = np.asarray(soc_knots, dtype=float).reshape(-1)
        self.delta = np.asarray(docv_mv, dtype=float).reshape(-1) * 1e-3  # → V
        if self.knots.size != self.delta.size or self.knots.size < 2:
            raise ValueError("soc_knots 与 docv_mv 必须同长且 ≥2")

    def delta_at(self, soc) -> np.ndarray:
        return np.interp(np.asarray(soc, dtype=float), self.knots, self.delta)

    def __call__(self, soc, temp_c=None):
        s = np.asarray(soc, dtype=float)
        if temp_c is None:
            return self.base(s) + self.delta_at(s)
        try:
            return self.base(s, temp_c) + self.delta_at(s)
        except TypeError:
            return self.base(s) + self.delta_at(s)

    def docv(self, soc, temp_c=None, h: float = 1e-4) -> np.ndarray:
        s = np.asarray(soc, dtype=float)
        if temp_c is None:
            d = self.base.docv(s, h=h)
        else:
            try:
                d = self.base.docv(s, temp_c, h=h)
            except TypeError:
                d = self.base.docv(s, h=h)
        # Δ 的导数：线性插值的局部斜率（中心差分足够）
        lo = np.clip(s - h, 0.0, 1.0)
        hi = np.clip(s + h, 0.0, 1.0)
        dd = (self.delta_at(hi) - self.delta_at(lo)) / np.maximum(hi - lo, 1e-12)
        return d + dd

    def batch(self, soc: np.ndarray, temp_c: np.ndarray) -> np.ndarray:
        s = np.asarray(soc, dtype=float)
        return self.base.batch(s, np.asarray(temp_c, dtype=float)) + self.delta_at(s)
