"""统一工况容器：重采样、累计电量、静置段检测、真值 SOC。

设计约定
--------
- **电流符号**：正 = 充电，负 = 放电（与两个原始数据集一致）。
- **重采样**：电流按区间均值（保证 ∫I dt 守恒，从而库仑计数不失真），
  电压取区间末值（模拟 BMS 采样），温度取均值。
- **真值 SOC**：SOC_ref(t) = 1 + ∫I dt / Q_ref，Q_ref 为该次测试的放电容量。
  只依赖测试自身的电量，模型不参与，避免循环论证。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


def cum_charge_ah(t_s: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    """梯形积分累计电量（Ah），放电为负。"""
    t = np.asarray(t_s, dtype=float)
    i = np.asarray(current_a, dtype=float)
    if t.size == 0:
        return np.zeros(0)
    dt = np.diff(t)
    return np.concatenate([[0.0], np.cumsum(0.5 * (i[1:] + i[:-1]) * dt)]) / 3600.0


def rest_mask(current_a: np.ndarray, thr: float = 0.05) -> np.ndarray:
    """静置掩码（|I| ≤ thr）。"""
    return np.abs(np.asarray(current_a, dtype=float)) <= thr


def rest_segments(t_s: np.ndarray, current_a: np.ndarray, thr: float = 0.05,
                  min_dur_s: float = 120.0):
    """切出静置段，返回 [(i0, i1, dur_s)]（闭区间索引）。"""
    mask = rest_mask(current_a, thr)
    out = []
    n = mask.size
    i = 0
    t = np.asarray(t_s, dtype=float)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        dur = float(t[j] - t[i])
        if dur >= min_dur_s:
            out.append((i, j, dur))
        i = j + 1
    return out


def _hold_fill(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """用**前一个有效值**填充无效位置（零阶保持）；首端无效则用第一个有效值。

    这是修复过的一个真实 bug：早先版本写 ``arr[bad] = arr[good[0]] if good[0] > 0
    else arr[good].mean()``，当第一个采样点就有效时（``good[0] == 0``，几乎总是如此），
    全部空区间被填成**整段电压的均值**。本数据集的低温文件开头有 2 h 温箱静置、
    采样间隔 60 s，重采样到 1 s 后 59/60 的区间是空的，于是 55% 的样本电压被灌成
    3.47 V 的均值（真实值 4.17 V），滤波器被直接拽到底。
    """
    if arr.size == 0 or not valid.any():
        return np.zeros_like(arr)
    src = np.where(valid, np.arange(arr.size), 0)
    np.maximum.accumulate(src, out=src)
    first = int(np.flatnonzero(valid)[0])
    src[:first] = first
    return arr[src]


def sample_gaps(t_s: np.ndarray, dt: float = 1.0) -> Dict[str, float]:
    """量化原始采样相对 ``dt`` 网格的稀疏程度（供数据勘察与报告引用）。"""
    t = np.asarray(t_s, dtype=float)
    if t.size < 2:
        return {"gap_frac": 0.0, "gap_max_s": 0.0, "gap_frac_active": 0.0}
    gap = np.diff(t)
    span = float(t[-1] - t[0])
    empty = np.maximum(gap - dt, 0.0)
    return {
        "gap_frac": float(empty.sum() / span) if span > 0 else 0.0,
        "gap_max_s": float(gap.max()),
        "gap_frac_active": float(empty.sum() / span) if span > 0 else 0.0,
    }


def resample(t_s: np.ndarray, current_a: np.ndarray, voltage_v: np.ndarray,
             temp_c: np.ndarray, dt: float = 1.0) -> Dict[str, np.ndarray]:
    """把任意采样重采样到固定步长 ``dt`` 秒。

    电流取区间均值（保证 ∫I dt 守恒），电压/温度取区间末值（模拟 BMS 采样）。
    空区间（原始采样比 ``dt`` 稀疏）一律用**前值保持**填充。
    """
    t = np.asarray(t_s, dtype=float)
    i = np.asarray(current_a, dtype=float)
    v = np.asarray(voltage_v, dtype=float)
    T = np.asarray(temp_c, dtype=float)
    if t.size < 2:
        raise ValueError("样本过少")
    edges = np.arange(t[0], t[-1] + dt, dt)
    if edges.size < 2:
        edges = np.array([t[0], t[-1]])
    nbin = edges.size - 1
    # 每个原始样本落在哪个区间（右端闭合，保证末点被包含）
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, nbin - 1)
    cnt = np.bincount(idx, minlength=nbin).astype(float)
    has = cnt > 0
    cnt_safe = np.where(has, cnt, 1.0)
    i_mean = _hold_fill(np.bincount(idx, weights=i, minlength=nbin) / cnt_safe, has)
    T_mean = _hold_fill(np.bincount(idx, weights=T, minlength=nbin) / cnt_safe, has)
    # 电压取区间末值
    order = np.argsort(idx, kind="stable")
    last_of_bin = np.zeros(nbin, dtype=int) - 1
    last_of_bin[idx[order]] = order
    v_last = np.full(nbin, np.nan)
    v_last[has] = v[last_of_bin[has]]
    v_last = _hold_fill(v_last, has)
    return {"t_s": edges[:-1], "current_a": i_mean, "voltage_v": v_last, "temp_c": T_mean}


@dataclass
class DrivingProfile:
    """一段连续工况（含真值 SOC）。"""

    t_s: np.ndarray
    current_a: np.ndarray
    voltage_v: np.ndarray
    temp_c: np.ndarray
    soc_ref: np.ndarray
    capacity_ah: float
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t_s = np.asarray(self.t_s, dtype=float)
        self.current_a = np.asarray(self.current_a, dtype=float)
        self.voltage_v = np.asarray(self.voltage_v, dtype=float)
        self.temp_c = np.asarray(self.temp_c, dtype=float)
        self.soc_ref = np.asarray(self.soc_ref, dtype=float)
        n = self.t_s.size
        for name in ("current_a", "voltage_v", "temp_c", "soc_ref"):
            if getattr(self, name).size != n:
                raise ValueError(f"{name} 长度与 t_s 不一致")

    def __len__(self) -> int:
        return int(self.t_s.size)

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.t_s)))

    @property
    def key(self) -> str:
        return f"{self.meta.get('dataset','?')}|{self.meta.get('temp_c', float('nan')):g}C|{self.meta.get('profile','?')}"

    def slice(self, i0: int = 0, i1: Optional[int] = None) -> "DrivingProfile":
        i1 = len(self) if i1 is None else i1
        return DrivingProfile(
            self.t_s[i0:i1] - self.t_s[i0], self.current_a[i0:i1], self.voltage_v[i0:i1],
            self.temp_c[i0:i1], self.soc_ref[i0:i1], self.capacity_ah, dict(self.meta),
        )

    def trim_leading_rest(self, keep_s: float = 300.0, thr: float = 0.05) -> "DrivingProfile":
        """裁掉开头过长的静置段，只保留 ``keep_s`` 秒。

        为什么需要：低温文件开头有 2 h 温箱静置（电流 0、电压恒定），修好重采样后
        它占样本数的 55%。留着会把 RMSE 稀释成「静置有多准」，而本项目评的是
        **动态工况**精度。保留 300 s 是为了让滤波器有真实的上电收敛窗口。
        """
        i = np.abs(self.current_a)
        active = np.flatnonzero(i > thr)
        if active.size == 0:
            return self
        t_start = float(self.t_s[active[0]])
        t_keep = t_start - float(keep_s)
        if t_keep <= float(self.t_s[0]) + 1e-9:
            return self
        i0 = int(np.searchsorted(self.t_s, t_keep))
        return self.slice(i0)


def build_profile(t_s, current_a, voltage_v, temp_c, capacity_ah: float,
                  soc0: float = 1.0, dt: float = 1.0,
                  meta: Optional[Dict[str, object]] = None,
                  resample_first: bool = True) -> DrivingProfile:
    """从原始序列构造 DrivingProfile。

    真值 SOC 由重采样后的电流重新积分得到（与滤波器看到的数据严格一致），
    起点取 ``soc0``；若给出 ``capacity_ah`` 则终点约为 ``soc0 - Q/Q_ref``。
    """
    if resample_first:
        r = resample(t_s, current_a, voltage_v, temp_c, dt=dt)
    else:
        r = {"t_s": np.asarray(t_s, float), "current_a": np.asarray(current_a, float),
             "voltage_v": np.asarray(voltage_v, float), "temp_c": np.asarray(temp_c, float)}
    q = cum_charge_ah(r["t_s"], r["current_a"])
    soc = np.clip(soc0 + q / float(capacity_ah), 0.0, 1.0)
    return DrivingProfile(
        t_s=r["t_s"], current_a=r["current_a"], voltage_v=r["voltage_v"], temp_c=r["temp_c"],
        soc_ref=soc, capacity_ah=float(capacity_ah), meta=dict(meta or {}),
    )


def oof_split(profile: DrivingProfile, n_folds: int = 5):
    """把一段工况切成 ``n_folds`` 块（用于时间外推的可视化，不做训练）。"""
    n = len(profile)
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    return [profile.slice(edges[k], edges[k + 1]) for k in range(n_folds)]


def to_frame(profile: DrivingProfile) -> pd.DataFrame:
    df = pd.DataFrame({
        "t_s": profile.t_s, "current_a": profile.current_a, "voltage_v": profile.voltage_v,
        "temp_c": profile.temp_c, "soc_ref": profile.soc_ref,
    })
    for k, v in profile.meta.items():
        df[k] = v
    df["capacity_ah"] = profile.capacity_ah
    return df
