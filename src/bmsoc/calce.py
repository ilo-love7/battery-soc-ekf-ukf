"""CALCE A123（LFP）数据集级逻辑：分段、OCV 曲线、工况构造。

和 :mod:`bmsoc.io_calce` 的分工
-------------------------------
``io_calce`` 只负责「把 zip 里的 Arbin 表读成 DataFrame」；
本模块负责把一张表变成**基准能用的东西**：三个动态放电段、一条 OCV(SOC) 曲线。

数据实际形态（勘察结论，与文档描述有出入，记录在此）
--------------------------------------------------
- ``A123_OCV<T>``：不是「10% 步进 + 静置」，而是
  **C/20（0.05 A）全程慢放 21 h → 静置 1 h → C/20 慢充 21 h → 静置 1 h**。
  所以 OCV 曲线取慢放支路（与动态工况同为放电，自洽）；LFP 的充放滞后
  在 C/20 下仍有 10–20 mV，用单一支路是有偏的，这一点写进报告。
- ``A123_DST-US06-FUDS<T>``：每个包 = 满充 → 动态放电到 2.0 V → 满充 → …，
  共三段，按时间顺序是 DST / US06 / FUDS。每段内部是**同一工况重复若干次**
  （所以时长 2 h 而不是 6 min），直到截止电压。
- 采样：动态段 1 s，静置段稀疏（几十秒一个点）。稀疏填充依赖
  :func:`bmsoc.profiles.resample` 的零阶保持。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import io_calce as io
from .profiles import DrivingProfile, build_profile, cum_charge_ah

# 三段动态放电的标签（按包名 DST-US06-FUDS 的顺序）
SEG_LABELS = ("dst", "us06", "fuds")

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "results", "cache")


def _cache_path(zip_path: str) -> str:
    return os.path.join(_CACHE_DIR, "calce_" + os.path.basename(zip_path).replace(".zip", "") + ".parquet")


def load_cached(zip_path: str) -> pd.DataFrame:
    """读一个包，带 parquet 缓存（xlsx 解析一次约 20–40 s，24 个包太贵）。"""
    dst = _cache_path(zip_path)
    meta_stamp = os.path.getmtime(zip_path)
    if os.path.exists(dst) and os.path.getmtime(dst) >= meta_stamp:
        return pd.read_parquet(dst)
    df = io.load_package(zip_path)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    df.to_parquet(dst, index=False)
    return df


def _step_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("step").agg(n=("t_s", "size"), t0=("t_s", "min"), t1=("t_s", "max"),
                               i_mean=("current_a", "mean"), i_absmax=("current_a",
                                                                            lambda s: float(np.abs(s).max())),
                               v_min=("voltage_v", "min"), v_max=("voltage_v", "max"))
    g["dur_s"] = g["t1"] - g["t0"]
    return g


def segments(df: pd.DataFrame, min_dur_s: float = 1200.0,
             min_abs_i: float = 1.0) -> Dict[str, pd.DataFrame]:
    """切出三段动态放电，返回 ``{"dst": df, "us06": df, "fuds": df}``。

    判据：持续 ≥ 20 min 且峰值电流 ≥ 1 A 的 step。包内这样的 step 恰好三个，
    按时间顺序对应包名里的 DST / US06 / FUDS。
    """
    st = _step_stats(df)
    # 只取放电段：低温下 CC-CV 充电也会超过 20 min（0 °C 的充电 step 有 6.9 h），
    # 只看时长与峰值电流会把充电段一起选进来。
    pick = st[(st["dur_s"] >= min_dur_s) & (st["i_absmax"] >= min_abs_i)
              & (st["i_mean"] < 0)].sort_values("t0")
    if len(pick) != 3:
        raise ValueError(f"期望 3 段动态放电，实际 {len(pick)} 段（{list(pick.index)}）")
    return {label: df[df["step"] == step].reset_index(drop=True)
            for label, step in zip(SEG_LABELS, pick.index)}


def ocv_curve_points(df: pd.DataFrame, n_grid: int = 121,
                     i_tol: float = 0.08) -> Tuple[np.ndarray, np.ndarray, float]:
    """从 C/20 慢放支路取 OCV(SOC)。

    返回 ``(soc, ocv, q_slow_ah)``，``q_slow_ah`` 是该支路的总放电量
    （= 这条曲线上 SOC 轴的容量基准）。SOC 按 ``1 - q_dis/q_total`` 算，
    与库仑计数的符号约定（充电为正）一致。
    """
    st = _step_stats(df)
    cand = st[(st["i_mean"] < -i_tol * 0.3) & (st["i_mean"] > -0.2) & (st["dur_s"] > 3 * 3600)]
    if cand.empty:
        raise ValueError("未找到 C/20 慢放支路")
    step = cand.sort_values("dur_s").index[-1]
    seg = df[df["step"] == step].sort_values("t_s")
    q = cum_charge_ah(seg["t_s"].to_numpy(float), seg["current_a"].to_numpy(float))
    q_tot = float(-q[-1])                      # 放电为负 → 取正
    soc = 1.0 + q / q_tot
    v = seg["voltage_v"].to_numpy(float)
    keep = np.concatenate(([True], np.diff(soc) < 0))   # SOC 必须严格递减
    soc, v = soc[keep], v[keep]
    order = np.argsort(soc)
    soc, v = soc[order], v[order]
    # 均匀重采样到 n_grid 点（后面还要 PCHIP + 平滑，稀疏点会导致局部斜率病态）
    grid = np.linspace(soc[0], soc[-1], n_grid)
    return grid, np.interp(grid, soc, v), q_tot


def _temp(seg: pd.DataFrame) -> np.ndarray:
    """箱温列名是 ``temp_c_set``（io_calce 落盘的原名），保底兼容 ``temp_c``。"""
    col = "temp_c_set" if "temp_c_set" in seg.columns else "temp_c"
    return seg[col].to_numpy(float)


def segment_capacity_ah(seg: pd.DataFrame, dt: float) -> float:
    """一段动态放电的净放电量（Ah）。

    注意它**不是** SOC 的归一化基准**（见 :func:`build_calce_profiles` 的说明）：
    在 −10 °C，高倍率放电提前撞 2.0 V 截止，净放电量只有 C/20 容量的 70%，
    用它归一化会把「还剩 30% 电量」硬说成 0%。
    """
    from .profiles import resample

    r = resample(seg["t_s"].to_numpy(float), seg["current_a"].to_numpy(float),
                 seg["voltage_v"].to_numpy(float), _temp(seg), dt=dt)
    q = cum_charge_ah(r["t_s"], r["current_a"])
    return float(-q[-1])


def build_calce_profiles(dt: float = 1.0, keep_rest_s: float = 300.0,
                         temps: Optional[List[float]] = None) -> List[DrivingProfile]:
    """把 8 个温度 × 3 段工况构造成 ``DrivingProfile`` 列表。

    **容量基准取该温度的 C/20 慢放容量**（与 OCV 曲线的 SOC 轴同源）。

    这一点必须强调：如果改用「该段自己的净放电量」归一化，每段都会从 1.0 走到 0.0，
    但 −10 °C 时高倍率放电只放出 C/20 容量的 70% 就撞到 2.0 V 截止 —— 剩下那 30%
    仍然在电芯里。用段容量归一化等于把 SOC 轴压扁 30%，OCV 查表会系统性错位，
    滤波器被迫用 SOC 去吸收一个容量口径错误。这是 Panasonic 项目
    「容量基准必须与 OCV 的 SOC 轴同源」那条教训的 LFP 版本。
    """
    q_ref = {t: float(q) for t, (_, _, q) in ocv_table().items()}
    out: List[DrivingProfile] = []
    for path, meta in io.list_packages(kind="dst"):
        temp = float(meta["temp_c"])
        if temps is not None and temp not in temps:
            continue
        df = load_cached(path)
        Q = q_ref[temp]
        for label, seg in segments(df).items():
            q = segment_capacity_ah(seg, dt=dt)
            t0 = float(seg["t_s"].iloc[0])
            p = build_profile(seg["t_s"].to_numpy(float) - t0,
                              seg["current_a"].to_numpy(float),
                              seg["voltage_v"].to_numpy(float), _temp(seg),
                              Q, soc0=1.0, dt=dt,
                              meta={"profile": label, "temp_c": temp,
                                    "source": os.path.basename(path),
                                    "q_seg_ah": q, "q_ref_ah": Q,
                                    "soc_end": 1.0 - q / Q})
            p = p.trim_leading_rest(keep_s=keep_rest_s)
            p.meta["soc0"] = 1.0
            p.meta["n_total"] = len(seg)
            out.append(p)
    out.sort(key=lambda p: (float(p.meta["temp_c"]), p.meta["profile"]))
    return out


def ocv_table(dt: float = 1.0) -> Dict[float, Tuple[np.ndarray, np.ndarray, float]]:
    """每个温度一条 OCV 曲线：``{temp: (soc, ocv, q_slow_ah)}``。"""
    out = {}
    for path, meta in io.list_packages(kind="ocv"):
        temp = float(meta["temp_c"])
        df = load_cached(path)
        out[temp] = ocv_curve_points(df)
    return out
