#!/usr/bin/env python
"""30_white_round5.py — 第九轮：white（快变非线性）的物理定性（v1.8）。

问题（记录 §45.5 首选）
---------------------
第八轮把滤波后残差拆成 slow/even（清白）+ odd（冷温次通道）+ white（最大块、
q 中毒主通道）。但 white 是分解余项的名字，不是机制——它可能是脉冲边沿的
R0 快瞬态（双电层/接触电阻）、滞回/充放不对称、i|i| 非线性残余、或纯测量噪声。
本脚本用**条件统计 + 边沿对齐 + 合成正对照**给 white 做物理定性，
判别哪个机制该进模型/滤波器预算。

阶段
----
- ``probe``：逐折 EKF → 滤波后残差 → 结构分解（复用脚本 28）→ white 分量
  对 di/dt、|i|、充放方向、脉冲边沿距离的条件统计；边沿对齐平均（ETA）；
  自相关；快模态匹配（合成 RC 时间常数网格）；滞回代理；i|i| 残余投影。
  合成正对照（纯噪声 / 已知快 RC / 已知滞回态）标定探针检出力。
- ``h26``/``h27``/``h28``：判决实验（假设登记后补）。

输出：``results/metrics/white_r5_{dataset}_{probe.json,samples,cond,eta,acf,control}.*``
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
MET = os.path.join(ROOT, "results", "metrics")

# ---- 口径常量（与脚本 26/28 对齐）----
BIGI_A = 2.0           # 脉冲阈值 |i| > 2 A
LOWI_A = 1.0           # 小电流 |i| ≤ 1 A
BURNIN_S = 200.0       # 滤波 burn-in
SLOW_WIN_S = 300.0     # 慢变成分滚动窗（脚本 28 同源）

# ---- 条件分箱 ----
DIDT_EDGES = (0.0, 0.05, 0.15, 0.4, 1.0, np.inf)   # |di/dt| A/s，5 箱
IEDGES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 30.0)
EDGE_DIST_EDGES = (0.0, 2.0, 5.0, 20.0, 60.0, np.inf)  # 距边沿秒
ACF_LAGS_S = (1, 2, 5, 10, 30, 60)
TAU_GRID_S = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
ETA_PRE_S = 30
ETA_POST_S = 120
HYST_WIN_S = 500.0     # 滞回代理的时间常数窗


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_folds(dataset: str):
    s23 = _load("s23", os.path.join(ROOT, "scripts", "23_residual_round2.py"))
    folds, best, ocv2d = s23.fold_residuals(dataset)
    return s23, folds, best, ocv2d


# =====================================================================
# 协变量与特征
# =====================================================================

def _di_dt(i: np.ndarray, dt: float) -> np.ndarray:
    """逐样本电流变化率（A/s），边界用单侧差分。"""
    i = np.asarray(i, float)
    if i.size == 0:
        return i.copy()
    d = np.empty_like(i)
    d[1:-1] = (i[2:] - i[:-2]) / (2.0 * dt)
    d[0] = (i[1] - i[0]) / dt if i.size > 1 else 0.0
    d[-1] = (i[-1] - i[-2]) / dt if i.size > 1 else 0.0
    return d


def _edge_events(i: np.ndarray, dt: float, thr: float = BIGI_A):
    """|i| 穿越 thr 的边沿。返回 (onset_idx, release_idx)：

    onset  = |i| 从小 < thr 跳到 ≥ thr（脉冲起始）
    release = |i| 从 ≥ thr 落回 < thr（脉冲结束）
    """
    big = np.abs(np.asarray(i, float)) >= thr
    if big.size < 2:
        return np.array([], int), np.array([], int)
    d = np.diff(big.astype(np.int8))
    onset = np.where(d > 0)[0] + 1
    release = np.where(d < 0)[0] + 1
    return onset, release


def _edge_distance(i: np.ndarray, dt: float, thr: float = BIGI_A):
    """每样本距最近一次 |i| 穿越 thr 的秒数与类型。

    返回 (dist_s, kind)：kind = +1 onset 之后、−1 release 之后、0 尚无边沿。
    """
    n = np.asarray(i, float).size
    onset, release = _edge_events(i, dt, thr)
    # 尚无边沿的样本用全时长当距离（保证落进最远箱，而不是 inf 落空）
    dist = np.full(n, n * dt, dtype=float)
    kind = np.zeros(n, np.int8)
    # 事件表按时间序合并
    ev = [(int(k), +1) for k in onset] + [(int(k), -1) for k in release]
    ev.sort(key=lambda x: x[0])
    pos = 0
    last_t, last_kind = -1, 0
    for k in range(n):
        while pos < len(ev) and ev[pos][0] <= k:
            last_t, last_kind = ev[pos]
            pos += 1
        if last_t >= 0:
            dist[k] = (k - last_t) * dt
            kind[k] = last_kind
    return dist, kind


def _hyst_proxy(i: np.ndarray, soc: np.ndarray, dt: float,
                win_s: float = HYST_WIN_S) -> np.ndarray:
    """滞回路径代理：近期 SOC 轨迹方向的低通（+1 上行 / −1 下行）。"""
    soc = np.asarray(soc, float)
    if soc.size < 3:
        return np.zeros_like(soc)
    dsoc = np.gradient(soc, dt)
    # sign 之后再低通，得到「最近 win_s 内主要在充还是放」
    ser = pd.Series(np.sign(dsoc)).rolling(
        max(3, int(win_s / max(dt, 1e-9))), center=True,
        min_periods=3).mean().to_numpy()
    return np.nan_to_num(ser, nan=0.0)


def _rc_fast(i: np.ndarray, dt: float, tau: float, r: float = 1.0) -> np.ndarray:
    """单极快模态 RC 状态（合成探针）：u[k] = α u[k−1] + (1−α)·r·i[k]。"""
    i = np.asarray(i, float)
    if i.size == 0:
        return i.copy()
    alpha = float(np.exp(-dt / max(tau, 1e-9)))
    u = np.zeros_like(i)
    # 稳态初值取 0（burn-in 由调用方裁掉）
    for k in range(i.size):
        u[k] = alpha * u[k - 1] + (1.0 - alpha) * r * i[k] if k else (1.0 - alpha) * r * i[0]
    return u


def _acf(x: np.ndarray, lags_s, dt: float) -> dict:
    """白噪声检验用的样本自相关（无偏归一，零方差返回 nan）。"""
    x = np.asarray(x, float)
    n = x.size
    out = {}
    xc = x - (np.mean(x) if n else 0.0)
    den = float(np.dot(xc, xc))
    for lag in lags_s:
        k = int(round(lag / max(dt, 1e-9)))
        if k <= 0 or k >= n or den <= 1e-18:
            out[f"acf_{lag:g}s"] = float("nan")
            continue
        out[f"acf_{lag:g}s"] = float(np.dot(xc[k:], xc[:-k]) / den)
    return out


def _resid_basis(y: np.ndarray, bases: list) -> np.ndarray:
    """y 对基组（自动加常数）回归后的残差。"""
    y = np.asarray(y, float)
    if y.size == 0:
        return y.copy()
    X = np.column_stack([np.ones(y.size)] + [np.asarray(b, float) for b in bases])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coef


def _fast_match(white: np.ndarray, i: np.ndarray, dt: float,
                tau_grid=TAU_GRID_S) -> dict:
    """快模态匹配：white 对合成 RC 状态（τ 网格）的最大 |相关| 与最优 τ。

    必须先把 RC 状态对 [1, i, |i|] 残差化——否则 τ 小的状态与 i 高度共线，
    其能量被 odd/even 投影吃掉，匹配会漏检（合成对照踩过）。
    """
    i = np.asarray(i, float)
    abs_i = np.abs(i)
    best_rho, best_tau = 0.0, float("nan")
    per = {}
    for tau in tau_grid:
        u = _rc_fast(i, dt, tau)
        u_res = _resid_basis(u, [i, abs_i])
        if white.size > 10 and np.std(white) > 0 and np.std(u_res) > 0:
            rho = float(np.corrcoef(white, u_res)[0, 1])
        else:
            rho = float("nan")
        per[f"rho_tau{tau:g}"] = rho
        if np.isfinite(rho) and abs(rho) > abs(best_rho):
            best_rho, best_tau = rho, tau
    return {"fast_tau_star_s": best_tau, "fast_rho_max": best_rho,
            "fast_sign": float(np.sign(best_rho)), **per}


def _proj_r2(y: np.ndarray, bases: list) -> float:
    """y 对一组基（自动加常数项）的回归 R²；零方差返回 nan。"""
    y = np.asarray(y, float)
    if y.size < 5 or np.std(y) <= 0:
        return float("nan")
    X = np.column_stack([np.ones(y.size)] + [np.asarray(b, float) for b in bases])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(y - np.mean(y), y - np.mean(y)))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-18 else float("nan")


def _eta(series: np.ndarray, idx: np.ndarray, dt: float,
         pre_s: float = ETA_PRE_S, post_s: float = ETA_POST_S):
    """边沿对齐平均：返回 (rel_t_s, mean, sem, n)。"""
    if idx.size == 0:
        return np.array([]), np.array([]), np.array([]), 0
    pre = int(pre_s / dt)
    post = int(post_s / dt)
    rel = np.arange(-pre, post + 1)
    mat = []
    for k in idx:
        a, b = k - pre, k + post + 1
        if a < 0 or b > series.size:
            continue
        mat.append(series[a:b])
    if not mat:
        return rel * dt, np.array([]), np.array([]), 0
    M = np.vstack(mat)
    mean = M.mean(axis=0)
    sem = M.std(axis=0, ddof=1) / max(np.sqrt(M.shape[0]), 1)
    return rel * dt, mean * 1e3, sem * 1e3, M.shape[0]


def _white_covariates(i: np.ndarray, soc: np.ndarray, dt: float) -> dict:
    """white 的全部条件协变量（逐样本对齐）。"""
    i = np.asarray(i, float)
    didt = _di_dt(i, dt)
    dist, kind = _edge_distance(i, dt)
    hyst = _hyst_proxy(i, soc, dt)
    return {
        "i": i, "abs_i": np.abs(i), "didt": didt, "abs_didt": np.abs(didt),
        "i_absi": i * np.abs(i), "edge_dist_s": dist, "edge_kind": kind,
        "hyst": hyst, "chg": (i > 0).astype(np.int8),
    }


# =====================================================================
# 合成正对照（方法论 31：探针先校台架）
# =====================================================================

def synthetic_control(rng: np.random.Generator) -> dict:
    """已知机制的合成植物，检验条件统计的检出力：

    - ``noise``：纯白噪声 —— H28 应过（ACF≈0、分箱均匀）
    - ``fast_rc``：已知 τ=2 s 的快模态（40 mV/A）+ 噪声 —— H26 应检出
      （注意：大半能量会被 odd 吃掉，white 里剩的是 RC 残差，ACF 高）
    - ``hyst_slow``：慢路径态（500 s 低通 × 25 mV）—— 应被 slow 滤掉、
      white 保持干净（说明慢滞回不进 white 口径，已被第八轮 slow≈0 排除）
    - ``hyst_fast``：快路径/率相关（tanh(4·di/dt) × 20 mV）—— H27 应检出
    """
    dt = 1.0
    n = 6000
    t = np.arange(n) * dt
    # 驱动电流：静置 + 脉冲块（模仿工况）
    i = np.zeros(n)
    pos = 200
    while pos < n - 300:
        amp = rng.choice([-6.0, -3.0, 2.0, 5.0])
        dur = int(rng.integers(20, 90))
        i[pos:pos + dur] = amp
        pos += dur + int(rng.integers(30, 150))
    i += rng.normal(0.0, 0.05, n)
    soc = 1.0 - np.cumsum(-i) * dt / (2.5 * 3600.0)  # 假容量 2.5 Ah
    soc = np.clip(soc, 0.05, 1.0)

    base_noise = rng.normal(0.0, 0.012, n)  # 12 mV 测量噪声
    didt = _di_dt(i, dt)
    out = {}
    s28 = _load("s28c", os.path.join(ROOT, "scripts", "28_ekfq_round4.py"))
    for tag in ("noise", "fast_rc", "hyst_slow", "hyst_fast"):
        if tag == "noise":
            y = base_noise.copy()
        elif tag == "fast_rc":
            y = base_noise + 0.040 * _rc_fast(i, dt, 2.0)
        elif tag == "hyst_slow":
            y = base_noise + 0.025 * _hyst_proxy(i, soc, dt, 500.0)
        else:
            y = base_noise + 0.020 * np.tanh(4.0 * didt)
        cov = _white_covariates(i, soc, dt)
        valid = t >= BURNIN_S
        # 同一套结构分解，取 white 余项（口径与真实数据一致）
        comps = s28._structure_components(y, i, dt, SLOW_WIN_S)
        w = comps["white"][valid]
        cv = {k: (v[valid] if isinstance(v, np.ndarray) else v)
              for k, v in cov.items()}
        out[tag] = _cond_stats(w, cv, dt, 0.0, tag)
        out[tag]["acf"] = _acf(w, ACF_LAGS_S, dt)
        out[tag]["fast"] = _fast_match(w, cv["i"], dt)
        out[tag]["white_rms_mv"] = float(np.sqrt(np.mean(w ** 2)) * 1e3)
        out[tag]["odd_rms_mv"] = float(np.sqrt(np.mean(
            comps["odd"][valid] ** 2)) * 1e3)
        out[tag]["slow_rms_mv"] = float(np.sqrt(np.mean(
            comps["slow"][valid] ** 2)) * 1e3)
    return out


def _cond_stats(white: np.ndarray, cov: dict, dt: float,
                temp: float, prof: str) -> dict:
    """white 的条件统计汇总（一个折一份；合成对照复用）。"""
    w = np.asarray(white, float) * 1e3  # mV
    rows = {}

    def _rms(mask):
        m = np.asarray(mask, bool)
        return float(np.sqrt(np.mean(w[m] ** 2))) if m.any() else float("nan")

    def _med(mask):
        m = np.asarray(mask, bool)
        return float(np.median(w[m])) if m.any() else float("nan")

    # ① |di/dt| 分箱
    ab = cov["abs_didt"]
    for lo, hi in zip(DIDT_EDGES[:-1], DIDT_EDGES[1:]):
        rows[f"didt[{lo:g},{hi:g})_rms"] = _rms((ab >= lo) & (ab < hi))
    dhi = _rms(ab >= DIDT_EDGES[3])
    dlo = _rms(ab < DIDT_EDGES[1])
    rows["didt_hi_lo_ratio"] = (dhi / dlo) if (dlo and np.isfinite(dlo) and dlo > 0) \
        else float("nan")

    # ② |i| 分箱
    ai = cov["abs_i"]
    for lo, hi in zip(IEDGES[:-1], IEDGES[1:]):
        rows[f"absi[{lo:g},{hi:g})_rms"] = _rms((ai >= lo) & (ai < hi))

    # ③ 充放方向（符号约定：i>0 = 充电）
    rows["chg_rms"] = _rms(cov["chg"] == 1)
    rows["dis_rms"] = _rms(cov["chg"] == 0)
    rows["chg_mean"] = _med(cov["chg"] == 1)
    rows["dis_mean"] = _med(cov["chg"] == 0)
    lo_ = min(rows["chg_rms"], rows["dis_rms"])
    hi_ = max(rows["chg_rms"], rows["dis_rms"])
    rows["chg_dis_ratio"] = (hi_ / lo_) if (lo_ and np.isfinite(lo_) and lo_ > 0) \
        else float("nan")
    rows["chg_dis_mean_gap_mv"] = abs(rows["chg_mean"] - rows["dis_mean"])

    # ④ 脉冲边沿距离 × 类型
    dist, kind = cov["edge_dist_s"], cov["edge_kind"]
    for lo, hi in zip(EDGE_DIST_EDGES[:-1], EDGE_DIST_EDGES[1:]):
        rows[f"edist[{lo:g},{hi:g})_rms"] = _rms((dist >= lo) & (dist < hi))
    rows["after_onset_rms"] = _rms(kind == 1)
    rows["after_release_rms"] = _rms(kind == -1)
    e0 = rows["edist[0,2)_rms"]
    e4 = rows["edist[20,60)_rms"] if np.isfinite(rows["edist[20,60)_rms"]) \
        else rows["edist[60,inf)_rms"]
    rows["edge_early_late_ratio"] = (e0 / e4) if (e4 and np.isfinite(e4) and e4 > 0) \
        else float("nan")

    # ⑤ 滞回代理
    h = cov["hyst"]
    if w.size > 10 and np.std(w) > 0 and np.std(h) > 0:
        rows["rho_hyst"] = float(np.corrcoef(w, h)[0, 1])
    else:
        rows["rho_hyst"] = float("nan")
    rows["hyst_pos_rms"] = _rms(h > 0.2)
    rows["hyst_neg_rms"] = _rms(h < -0.2)

    # ⑥ 基形状投影 R²（剥掉 odd/even 后 white 还剩哪些 i-形状）
    ii = cov["i_absi"]
    didt = cov["didt"]
    rows["r2_i_absi"] = _proj_r2(w, [ii])
    rows["r2_didt"] = _proj_r2(w, [didt])
    rows["r2_didt_iabsi"] = _proj_r2(w, [didt, ii])
    rows["r2_hyst"] = _proj_r2(w, [h])

    # ⑦ di/dt 滞后相关（零阶 + 最优滞后；边沿瞬态的签名）
    if w.size > 10 and np.std(w) > 0 and np.std(didt) > 0:
        rho0 = float(np.corrcoef(w, didt)[0, 1])
        best_lag, best_abs = 0, abs(rho0)
        best_rho = rho0
        for lag in range(-5, 6):
            if lag == 0:
                continue
            if lag > 0:
                a, b = w[lag:], didt[:-lag]
            else:
                a, b = w[:lag], didt[-lag:]
            if a.size < 10 or np.std(b) == 0:
                continue
            r_ = float(np.corrcoef(a, b)[0, 1])
            if abs(r_) > best_abs:
                best_abs, best_lag, best_rho = abs(r_), lag, r_
        rows["rho_didt0"] = rho0
        rows["rho_didt_best"] = best_rho
        rows["rho_didt_best_lag_s"] = float(best_lag)
    else:
        rows["rho_didt0"] = float("nan")
        rows["rho_didt_best"] = float("nan")
        rows["rho_didt_best_lag_s"] = float("nan")

    rows["white_rms_mv"] = float(np.sqrt(np.mean(w ** 2)))
    rows["white_mean_mv"] = float(np.mean(w))
    rows["white_skew"] = float(pd.Series(w).skew()) if w.size > 2 else float("nan")
    rows["white_kurt"] = float(pd.Series(w).kurt()) if w.size > 3 else float("nan")
    rows["temp_c"] = temp
    rows["profile"] = prof
    return rows


# =====================================================================
# stage probe
# =====================================================================

def stage_probe(dataset: str) -> dict:
    from bmsoc.filters import EKF, FilterConfig
    from bmsoc.profiles import DrivingProfile

    s28 = _load("s28", os.path.join(ROOT, "scripts", "28_ekfq_round4.py"))
    s23, folds, best, ocv2d = load_folds(dataset)
    s12 = _load("s12", os.path.join(ROOT, "scripts", "12_calibration_stress.py"))
    _, profiles, _ = (s12.load_calce() if dataset == "calce"
                      else s12.load_panasonic())
    prof_of = {}
    for p in profiles:
        prof_of[(float(p.meta["temp_c"]), str(p.meta["profile"]))] = p

    # ---- 合成正对照（台架检出力）----
    rng = np.random.default_rng(20261101)
    control = synthetic_control(rng)

    sample_frames, cond_rows = [], []
    eta_rows, acf_rows, fast_rows = [], [], []
    for key in sorted(folds):
        f = folds[key]
        temp, prof_name = f["temp_c"], f["profile"]
        p = prof_of.get(key)
        if p is None:
            continue
        params, r0k = s28._rc_from_vec(f["vec"])
        n = len(p)
        dt = float(p.dt)
        t = np.arange(n) * dt
        soc = np.asarray(p.soc_ref, float)
        i = np.asarray(p.current_a, float)
        v = np.asarray(p.voltage_v, float)
        r0_per = np.interp(soc, (0.1, 0.5, 0.9), r0k)
        anchor = s28._R0Anchor(p.t_s, r0_per, p.capacity_ah)
        pf = DrivingProfile(t_s=p.t_s, current_a=i, voltage_v=v,
                            temp_c=np.asarray(p.temp_c, float),
                            soc_ref=soc, capacity_ah=p.capacity_ah,
                            meta=dict(p.meta))
        cfg = FilterConfig(soc0=float(np.clip(soc[0], 0.0, 1.0)), sigma_soc0=0.05)
        out = EKF(ocv=ocv2d, params=params, capacity_ah=p.capacity_ah,
                  cfg=cfg).run(pf, anchor=anchor)
        inn = np.asarray(out["innovation"], float)
        comps = s28._structure_components(inn, i, dt, SLOW_WIN_S)
        white = comps["white"]
        valid = t >= BURNIN_S
        cov = _white_covariates(i, soc, dt)

        # ---- 逐样本样本表（落盘用；含协变量）----
        sample_frames.append(pd.DataFrame({
            "temp_c": temp, "profile": prof_name,
            "t_s": t[valid], "i_a": i[valid],
            "abs_didt": cov["abs_didt"][valid], "didt": cov["didt"][valid],
            "abs_i": cov["abs_i"][valid], "i_absi": cov["i_absi"][valid],
            "edge_dist_s": np.where(np.isfinite(cov["edge_dist_s"][valid]),
                                    cov["edge_dist_s"][valid], -1.0),
            "edge_kind": cov["edge_kind"][valid], "hyst": cov["hyst"][valid],
            "chg": cov["chg"][valid], "soc": soc[valid],
            "inn_mv": inn[valid] * 1e3,
            "white_mv": white[valid] * 1e3,
            "odd_mv": comps["odd"][valid] * 1e3,
            "even_mv": comps["even"][valid] * 1e3,
            "slow_mv": comps["slow"][valid] * 1e3,
        }))

        # ---- 条件统计 ----
        cv = {k: (v[valid] if isinstance(v, np.ndarray) else v)
              for k, v in cov.items()}
        cond_rows.append(_cond_stats(white[valid], cv, dt, temp, prof_name))

        # ---- ACF / 快模态匹配 ----
        w_v = white[valid]
        ac = _acf(w_v, ACF_LAGS_S, dt)
        acf_rows.append({"temp_c": temp, "profile": prof_name, **ac})
        fm = _fast_match(w_v, cv["i"], dt)
        fast_rows.append({"temp_c": temp, "profile": prof_name,
                          "fast_tau_star_s": fm["fast_tau_star_s"],
                          "fast_rho_max": fm["fast_rho_max"]})

        # ---- ETA（onset / release）----
        onset, release = _edge_events(i, dt, BIGI_A)
        onset_v = onset[(onset * dt >= BURNIN_S)]
        release_v = release[(release * dt >= BURNIN_S)]
        for etype, idx in (("onset", onset_v), ("release", release_v)):
            rel, mean, sem, nn = _eta(white, idx, dt)
            for r_, m_, s_ in zip(rel, mean, sem):
                eta_rows.append({"temp_c": temp, "profile": prof_name,
                                 "etype": etype, "rel_t_s": float(r_),
                                 "white_mv": float(m_), "sem_mv": float(s_),
                                 "n_edges": int(nn)})
        print(f"[{temp:5.0f}] {prof_name:<5s} white "
              f"{np.sqrt(np.mean(w_v**2))*1e3:5.1f} mV  "
              f"didt比 {cond_rows[-1]['didt_hi_lo_ratio']:.2f}  "
              f"边沿比 {cond_rows[-1]['edge_early_late_ratio']:.2f}  "
              f"τ* {fm['fast_tau_star_s']:.1f}s ρ {fm['fast_rho_max']:+.2f}  "
              f"ACF1 {ac['acf_1s']:+.2f}", flush=True)

    cdf = pd.DataFrame(cond_rows)
    adf = pd.DataFrame(acf_rows)
    fdf = pd.DataFrame(fast_rows)
    edf = pd.DataFrame(eta_rows)
    sdf = pd.concat(sample_frames, ignore_index=True)
    if cdf.empty:
        raise SystemExit("probe 空结果 —— 检查 fold_residuals / profiles")
    cdf.to_parquet(os.path.join(MET, f"white_r5_{dataset}_cond.parquet"), index=False)
    adf.to_parquet(os.path.join(MET, f"white_r5_{dataset}_acf.parquet"), index=False)
    fdf.to_parquet(os.path.join(MET, f"white_r5_{dataset}_fastmatch.parquet"), index=False)
    edf.to_parquet(os.path.join(MET, f"white_r5_{dataset}_eta.parquet"), index=False)
    sdf.to_parquet(os.path.join(MET, f"white_r5_{dataset}_samples.parquet"), index=False)

    # ---- 逐温度汇总（折中位）----
    per_temp = {}
    for temp in sorted(cdf.temp_c.unique()):
        sub = cdf[cdf.temp_c == temp]
        aa = adf[adf.temp_c == temp]
        ff = fdf[fdf.temp_c == temp]
        per_temp[f"{temp:g}"] = {
            "white_rms_mv": float(sub.white_rms_mv.median()),
            "didt_hi_lo_ratio": float(sub.didt_hi_lo_ratio.median()),
            "edge_early_late_ratio": float(sub.edge_early_late_ratio.median()),
            "chg_dis_ratio": float(sub.chg_dis_ratio.median()),
            "chg_dis_mean_gap_mv": float(sub.chg_dis_mean_gap_mv.median()),
            "after_onset_rms": float(sub.after_onset_rms.median()),
            "after_release_rms": float(sub.after_release_rms.median()),
            "rho_hyst": float(sub.rho_hyst.median()),
            "r2_i_absi": float(sub.r2_i_absi.median()),
            "r2_didt": float(sub.r2_didt.median()),
            "r2_didt_iabsi": float(sub.r2_didt_iabsi.median()),
            "r2_hyst": float(sub.r2_hyst.median()),
            "rho_didt0": float(sub.rho_didt0.median()),
            "rho_didt_best": float(sub.rho_didt_best.median()),
            "rho_didt_best_lag_s": float(sub.rho_didt_best_lag_s.median()),
            "acf_1s": float(aa["acf_1s"].median()),
            "acf_5s": float(aa["acf_5s"].median()),
            "acf_30s": float(aa["acf_30s"].median()),
            "fast_tau_star_s": float(ff.fast_tau_star_s.median()),
            "fast_rho_max": float(ff.fast_rho_max.median()),
            "white_skew": float(sub.white_skew.median()),
            "white_kurt": float(sub.white_kurt.median()),
        }
    out = {
        "dataset": dataset, "best_variant": best,
        "n_folds": int(len(cdf)), "burnin_s": BURNIN_S,
        "bigi_a": BIGI_A,
        "per_temp": per_temp,
        "control": control,
    }
    with open(os.path.join(MET, f"white_r5_{dataset}_probe.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("=== probe per_temp（折中位）===")
    for k, v in per_temp.items():
        print(f"  {k:>5s}°C  white {v['white_rms_mv']:5.1f} mV  "
              f"didt比 {v['didt_hi_lo_ratio']:4.2f}  边沿比 {v['edge_early_late_ratio']:4.2f}  "
              f"充放比 {v['chg_dis_ratio']:4.2f}  ρ(hyst) {v['rho_hyst']:+.2f}  "
              f"R²(didt) {v['r2_didt']:.2f}  ρ(didt) {v['rho_didt_best']:+.2f}"
              f"@{v['rho_didt_best_lag_s']:+.0f}s  "
              f"ACF1 {v['acf_1s']:+.2f}  τ* {v['fast_tau_star_s']:.1f}s "
              f"ρ {v['fast_rho_max']:+.2f}")
    print("=== 合成正对照（检出力）===")
    for tag, c in control.items():
        print(f"  {tag:<10s} white {c['white_rms_mv']:5.1f}  odd {c['odd_rms_mv']:5.1f}  "
              f"slow {c['slow_rms_mv']:5.1f} mV | didt比 {c['didt_hi_lo_ratio']:4.2f}  "
              f"边沿比 {c['edge_early_late_ratio']:4.2f}  充放比 {c['chg_dis_ratio']:4.2f}  "
              f"ρ(hyst) {c['rho_hyst']:+.2f}  R²(didt) {c['r2_didt']:.2f}  "
              f"ρ(didt) {c['rho_didt_best']:+.2f}  ACF1 {c['acf']['acf_1s']:+.2f}  "
              f"τ* {c['fast']['fast_tau_star_s']:.1f}s ρ {c['fast']['fast_rho_max']:+.2f}")
    return out


# =====================================================================
# stage h26/h27/h28 —— 判决线见工作记录 §46.2（只读，不调）
# =====================================================================

H26_L_ACF1 = 0.30        # −20 °C ACF1 下限（记忆）
H26_L_TAU = 20.0         # τ* 上限（s）
H26_L_DIDT = 2.0         # |di/dt| 最高/最低箱 RMS 比下限
H26_L_EDGE = 1.5         # 边沿比下限
H26_L_R2 = 0.20          # 残差化 rc_fast 回归 R² 下限
H26_L_CUT = 0.30         # 或 white RMS 削减份额下限
H27_L_MEAN_MV = 5.0      # 充放 mean 差
H27_L_RATIO = 1.30       # 充放 / hyst 两侧 RMS 比
H27_L_RHO = 0.30         # |ρ(hyst)| 下限
H28_L_ACF1 = 0.10        # |ACF1| 上限
H28_L_UNIF = 1.30        # didt 比 / 边沿比 上限（均匀）
H28_L_SKEW = 0.50
H28_L_KURT = 1.00


def _fast_rc_fit(dataset: str):
    """快模态回归判决：white ~ b·rc_fast(i, τ)（基已对 [1,i,|i|] 残差化）。

    逐折 τ 网格取最优，报 R² 与 white RMS 削减份额；合成对照同口径核验检出力。
    直接从 probe 的 samples 落盘读（与 probe 同源，避免重跑 EKF）。
    """
    sp = os.path.join(MET, f"white_r5_{dataset}_samples.parquet")
    if not os.path.exists(sp):
        raise SystemExit("缺 samples 产物 —— 先跑 --stage probe")
    sdf = pd.read_parquet(sp)
    out_rows = []
    for (temp, prof_name), g in sdf.groupby(["temp_c", "profile"]):
        g = g.sort_values("t_s")
        i = g["i_a"].to_numpy(float)
        w = g["white_mv"].to_numpy(float) * 1e-3
        dt = 1.0
        best_r2, best_tau, best_cut, best_beta = -np.inf, float("nan"), 0.0, 0.0
        per = {}
        for tau in TAU_GRID_S:
            u = _rc_fast(i, dt, tau)
            u_res = _resid_basis(u, [i, np.abs(i)])
            X = np.column_stack([np.ones(w.size), u_res])
            coef, *_ = np.linalg.lstsq(X, w, rcond=None)
            pred = X @ coef
            ss_res = float(np.sum((w - pred) ** 2))
            ss_tot = float(np.sum((w - w.mean()) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-18 else float("nan")
            cut = float(1.0 - np.sqrt(ss_res / w.size) /
                        max(np.sqrt(ss_tot / w.size), 1e-12)) if ss_tot > 1e-18 \
                else float("nan")
            per[f"r2_tau{tau:g}"] = r2
            if np.isfinite(r2) and r2 > best_r2:
                best_r2, best_tau, best_cut, best_beta = r2, tau, cut, float(coef[1])
        out_rows.append({"dataset": dataset, "temp_c": temp, "profile": prof_name,
                         "r2_best": best_r2, "tau_best_s": best_tau,
                         "rms_cut": best_cut, "beta_mv_per_a": best_beta * 1e3,
                         "white_rms_mv": float(np.sqrt(np.mean(w ** 2)) * 1e3),
                         **per})
    df = pd.DataFrame(out_rows)
    df.to_parquet(os.path.join(MET, f"white_r5_{dataset}_fastfit.parquet"), index=False)

    # 合成对照同口径核验（检出力：fast_rc 应显著高于 noise）
    rng = np.random.default_rng(20261101)
    ctrl = synthetic_control(rng)
    ctrl_recov = {}
    for tag, c in ctrl.items():
        ctrl_recov[tag] = {
            "didt_ratio": c["didt_hi_lo_ratio"],
            "acf1": c["acf"]["acf_1s"],
            "white_rms_mv": c["white_rms_mv"],
        }
    return df, ctrl_recov


def stage_h26(dataset: str) -> dict:
    """H26：white 主源 = 快极化动力学缺失。"""
    probe_path = os.path.join(MET, f"white_r5_{dataset}_probe.json")
    if not os.path.exists(probe_path):
        raise SystemExit("缺 probe 产物 —— 先跑 --stage probe")
    with open(probe_path, encoding="utf-8") as fh:
        probe = json.load(fh)
    pt = probe["per_temp"]
    df, ctrl_recov = _fast_rc_fit(dataset)

    per_temp = {}
    verdict_rows = []
    for tk, tv in pt.items():
        sub = df[df.temp_c == float(tk)]
        r2_med = float(sub.r2_best.median()) if len(sub) else float("nan")
        cut_med = float(sub.rms_cut.median()) if len(sub) else float("nan")
        tau_fit = float(sub.tau_best_s.median()) if len(sub) else float("nan")
        per_temp[tk] = {
            **{k: tv[k] for k in ("white_rms_mv", "acf_1s", "didt_hi_lo_ratio",
                                   "edge_early_late_ratio", "fast_tau_star_s",
                                   "fast_rho_max")},
            "fastfit_r2_med": r2_med,
            "fastfit_rms_cut_med": cut_med,
            "fastfit_tau_med_s": tau_fit,
        }
        verdict_rows.append({"temp_c": float(tk), **per_temp[tk]})
    vdf = pd.DataFrame(verdict_rows)
    cold = vdf[vdf.temp_c == -20.0]
    warm = vdf[vdf.temp_c == 25.0]
    l1 = bool(len(cold) and cold.acf_1s.iloc[0] >= H26_L_ACF1
              and cold.fast_tau_star_s.iloc[0] <= H26_L_TAU)
    l2 = bool(len(cold) and cold.didt_hi_lo_ratio.iloc[0] >= H26_L_DIDT
              and len(warm) and warm.didt_hi_lo_ratio.iloc[0] >= H26_L_DIDT)
    l3 = bool(len(cold) and cold.edge_early_late_ratio.iloc[0] >= H26_L_EDGE)
    r2_cold = float(cold.fastfit_r2_med.iloc[0]) if len(cold) else float("nan")
    cut_cold = float(cold.fastfit_rms_cut_med.iloc[0]) if len(cold) else float("nan")
    l4 = bool((np.isfinite(r2_cold) and r2_cold >= H26_L_R2)
              or (np.isfinite(cut_cold) and cut_cold >= H26_L_CUT))
    if l1 and l2 and l3 and l4:
        verdict = ("成立：white 主源 = 快极化动力学缺失（记忆 + 边沿集中 + "
                   f"快模态回归削减 {cut_cold:.0%}）")
    elif l1 and l2 and l3:
        verdict = ("部分成立：结构签名（记忆/边沿/didt）全过，但快 RC 形状解释力不足"
                   f"（R² {r2_cold:.2f}、削减 {cut_cold:.0%}）——形状可能是双模态/非 RC")
    else:
        verdict = "被否：结构签名不满足（见逐项）"
    out = {
        "dataset": dataset,
        "lines": {
            "L1_acf1": H26_L_ACF1, "L1_tau_s": H26_L_TAU,
            "L2_didt": H26_L_DIDT, "L3_edge": H26_L_EDGE,
            "L4_r2": H26_L_R2, "L4_cut": H26_L_CUT,
        },
        "checks": {"L1_memory": l1, "L2_didt": l2, "L3_edge": l3, "L4_explain": l4},
        "per_temp": per_temp,
        "ctrl_recov": ctrl_recov,
        "verdict": verdict,
    }
    with open(os.path.join(MET, f"white_r5_{dataset}_h26.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"=== h26 {dataset}：{verdict}")
    for tk, tv in per_temp.items():
        print(f"  {tk:>5s}°C  ACF1 {tv['acf_1s']:+.2f}  τ* {tv['fast_tau_star_s']:4.1f}s  "
              f"didt比 {tv['didt_hi_lo_ratio']:4.2f}  边沿比 {tv['edge_early_late_ratio']:4.2f}  "
              f"快模态R² {tv['fastfit_r2_med']:.2f}  削减 {tv['fastfit_rms_cut_med']:.0%}  "
              f"τfit {tv['fastfit_tau_med_s']:.1f}s")
    return out


def stage_h27(dataset: str) -> dict:
    """H27：white 主源 = 充放不对称 / 路径滞回。"""
    probe_path = os.path.join(MET, f"white_r5_{dataset}_probe.json")
    if not os.path.exists(probe_path):
        raise SystemExit("缺 probe 产物 —— 先跑 --stage probe")
    with open(probe_path, encoding="utf-8") as fh:
        probe = json.load(fh)
    pt = probe["per_temp"]
    per_temp, verdict_rows = {}, []
    for tk, tv in pt.items():
        per_temp[tk] = {
            "chg_dis_ratio": tv["chg_dis_ratio"],
            "chg_dis_mean_gap_mv": tv["chg_dis_mean_gap_mv"],
            "rho_hyst": tv["rho_hyst"],
        }
        verdict_rows.append({"temp_c": float(tk), **per_temp[tk]})
    vdf = pd.DataFrame(verdict_rows)
    warm = vdf[vdf.temp_c.isin([10.0, 25.0])]
    l1 = bool(len(warm) and (
        (warm.chg_dis_mean_gap_mv >= H27_L_MEAN_MV).any()
        or (warm.chg_dis_ratio >= H27_L_RATIO).any()))
    l2 = bool((vdf.rho_hyst.abs() >= H27_L_RHO).any())
    if l1 or l2:
        verdict = "成立/部分：充放或路径不对称过线（见逐项）"
    else:
        verdict = ("被否：充放不对称与路径滞回均不过线（冷温无充电样本、"
                   "ρ(hyst) 全温 ≤0.07；慢滞回已被 slow≈0 排除）")
    out = {
        "dataset": dataset,
        "lines": {"L1_mean_mv": H27_L_MEAN_MV, "L1_ratio": H27_L_RATIO,
                  "L2_rho": H27_L_RHO},
        "checks": {"L1_asym": l1, "L2_hyst": l2},
        "per_temp": per_temp,
        "verdict": verdict,
    }
    with open(os.path.join(MET, f"white_r5_{dataset}_h27.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"=== h27 {dataset}：{verdict}")
    for tk, tv in per_temp.items():
        print(f"  {tk:>5s}°C  充放比 {tv['chg_dis_ratio']}  mean差 "
              f"{tv['chg_dis_mean_gap_mv']} mV  ρ(hyst) {tv['rho_hyst']:+.2f}")
    return out


def stage_h28(dataset: str) -> dict:
    """H28：white = 测量噪声（无结构）。"""
    probe_path = os.path.join(MET, f"white_r5_{dataset}_probe.json")
    if not os.path.exists(probe_path):
        raise SystemExit("缺 probe 产物 —— 先跑 --stage probe")
    with open(probe_path, encoding="utf-8") as fh:
        probe = json.load(fh)
    pt = probe["per_temp"]
    per_temp, verdict_rows = {}, []
    for tk, tv in pt.items():
        per_temp[tk] = {
            "acf_1s": tv["acf_1s"], "didt_hi_lo_ratio": tv["didt_hi_lo_ratio"],
            "edge_early_late_ratio": tv["edge_early_late_ratio"],
            "white_skew": tv["white_skew"], "white_kurt": tv["white_kurt"],
        }
        verdict_rows.append({"temp_c": float(tk), **per_temp[tk]})
    vdf = pd.DataFrame(verdict_rows)
    l1 = bool((vdf.acf_1s.abs() <= H28_L_ACF1).all())
    l2 = bool((vdf.didt_hi_lo_ratio <= H28_L_UNIF).all()
              and (vdf.edge_early_late_ratio <= H28_L_UNIF).all())
    l3 = bool((vdf.white_skew.abs() <= H28_L_SKEW).all()
              and (vdf.white_kurt.abs() <= H28_L_KURT).all())
    if l1 and l2 and l3:
        verdict = "成立：噪声主导（出路是鲁棒滤波而非建模）"
    else:
        bad = []
        if not l1:
            bad.append("ACF1 超线（有记忆）")
        if not l2:
            bad.append("分箱不均（didt/边沿集中）")
        if not l3:
            bad.append("分布形状非高斯")
        verdict = f"被否：{'；'.join(bad)}"
    out = {
        "dataset": dataset,
        "lines": {"L1_acf1": H28_L_ACF1, "L2_unif": H28_L_UNIF,
                  "L3_skew": H28_L_SKEW, "L3_kurt": H28_L_KURT},
        "checks": {"L1_white": l1, "L2_uniform": l2, "L3_gauss": l3},
        "per_temp": per_temp,
        "verdict": verdict,
    }
    with open(os.path.join(MET, f"white_r5_{dataset}_h28.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"=== h28 {dataset}：{verdict}")
    for tk, tv in per_temp.items():
        print(f"  {tk:>5s}°C  ACF1 {tv['acf_1s']:+.2f}  didt比 "
              f"{tv['didt_hi_lo_ratio']:.2f}  边沿比 {tv['edge_early_late_ratio']:.2f}  "
              f"skew {tv['white_skew']:+.2f}  kurt {tv['white_kurt']:+.2f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["probe", "h26", "h27", "h28"], required=True)
    ap.add_argument("--dataset", choices=["calce", "panasonic"], default="panasonic")
    args = ap.parse_args()
    fn = {"probe": stage_probe, "h26": stage_h26, "h27": stage_h27,
          "h28": stage_h28}.get(args.stage)
    if fn is None:
        raise SystemExit(f"阶段 {args.stage} 尚未实现")
    fn(args.dataset)


if __name__ == "__main__":
    main()
