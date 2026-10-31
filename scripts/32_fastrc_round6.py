#!/usr/bin/env python
"""32_fastrc_round6.py — 第十轮：快支路重拟合压 white 与 EKFQ 联动（v1.9）。

问题（记录 §47.5 首选）
---------------------
第九轮把滤波后 white 定性为「快极化动力学缺失」（τ≈5–10 s；快模态回归只解释
38% 方差、RMS 削减 21%），出口写成「快极化支路 + 采样对齐」。第十轮把出口
变成实测：模型族吸收快支路后 white 削减多少、EKFQ 的 q 收敛率是否回升
（直接回灌第八轮阈值 white ≤~10 mV / q 收敛率）。

摸底事实（脚本 20 产物）：best 变体是 2rc（无第三支路），τ2 顶 6000 s 上界、
τ1 落 16–66 s —— 快端（τ≤10 s）无人认领。故「加 τ≈5 s 快 RC」的模型级实现 =
3rc + tau3 钉快端重拟合（3rc_fast，LOO 协议）；对照组 = 3rc + tau3 慢端
（3rc_slow，脚本 20 产物直接复用）—— §47.3「同一枚硬币的两面」的直接检验。

阶段
----
- ``probe``：逐折 LOO 双路线——(A) 模型级 3rc_fast 重拟合（tau3_bounds=(0.5,30)），
  对照 2rc 基线与 3rc_slow；(B) 结构级加支路两段式拟合（含 LOO-τ）；两者都测
  滤波后 innovation 结构（white/slow/odd/even）前后对照；v/i 对齐审计
  （i 平移 RMSE 扫描，合成对照校准检出力）。
- ``h29``/``h30``/``h31``：判决实验（登记线见工作记录 §48.2，实现只读常量）。

输出：``results/metrics/fastrc_r6_{dataset}_{probe.json,refit.parquet,
innovation.parquet,fastfit.parquet,lag.parquet,h29.json,h30.json,h31.json}``
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

# ---- 口径常量（与脚本 28/30 对齐）----
BURNIN_S = 200.0
TAU_GRID_S = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0)  # 快端网格（≥0.5 防退化基，方法论 37）
TAU3_FAST = (0.5, 30.0)          # 模型级快端钉死区间
LAGS_S = tuple(range(-5, 6))     # v/i 对齐探针的滞后窗（s，1 Hz 采样）
MAX_NFEV = 300
CAP_SCALES = (0.95, 1.05)
REPLAY_FOLDS = [(-20.0, "hwfet"), (-20.0, "la92"), (25.0, "nn"), (25.0, "udds")]
REPLAY_SCALES = (0.5, 1.0, 1.5)
INJECT_SET = ["none", "full", "fast_only", "minus_fast", "refit_full"]


_MOD_CACHE: dict = {}


def _load(name: str, path: str):
    if name in _MOD_CACHE:
        return _MOD_CACHE[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _MOD_CACHE[name] = mod
    return mod


def load_folds(dataset: str):
    s23 = _load("s23", os.path.join(ROOT, "scripts", "23_residual_round2.py"))
    folds, best, ocv2d = s23.fold_residuals(dataset)
    return s23, folds, best, ocv2d


# =====================================================================
# 快支路两段式拟合（结构级）与滞后互相关
# =====================================================================

def _rc_branch(i: np.ndarray, dt: float, tau: float, r: float = 1.0) -> np.ndarray:
    """单极 RC 状态：u[k] = α u[k−1] + (1−α)·r·i[k]（与脚本 30 _rc_fast 同口径）。"""
    i = np.asarray(i, float)
    if i.size == 0:
        return i.copy()
    alpha = float(np.exp(-dt / max(tau, 1e-9)))
    u = np.zeros_like(i)
    for k in range(i.size):
        u[k] = alpha * u[k - 1] + (1.0 - alpha) * r * i[k] if k else (1.0 - alpha) * r * i[0]
    return u


def _resid_basis(y: np.ndarray, bases: list) -> np.ndarray:
    """y 对基组（自动加常数）回归后的残差。"""
    y = np.asarray(y, float)
    if y.size == 0:
        return y.copy()
    X = np.column_stack([np.ones(y.size)] + [np.asarray(b, float) for b in bases])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ coef


def _fit_fast_two_stage(y: np.ndarray, i: np.ndarray, dt: float,
                        tau: float | None = None,
                        tau_grid=TAU_GRID_S) -> dict:
    """white ~ β·rc_res(τ)：τ 网格取最优（或指定 τ），基对 [1,i,|i|] 残差化。

    返回 ``tau_best/beta/r2/cut/component``；``component`` 是 white 空间里的
    快支路分量（β·u_res，零均值），可直接从 white 里剥掉。
    ``cut = 1 − RMS(white−component)/RMS(white−ȳ)``（与第九轮 rms_cut 同口径）。
    """
    y = np.asarray(y, float)
    i = np.asarray(i, float)
    if y.size < 10 or np.std(y) <= 1e-15:
        return {"tau_best": float("nan"), "beta": 0.0, "r2": float("nan"),
                "cut": 0.0, "component": np.zeros_like(y)}
    grid = (tau,) if tau is not None else tuple(tau_grid)
    best = {"tau_best": float("nan"), "beta": 0.0, "r2": -np.inf,
            "cut": 0.0, "component": np.zeros_like(y)}
    for t in grid:
        u = _rc_branch(i, dt, t)
        u_res = _resid_basis(u, [i, np.abs(i)])
        if np.std(u_res) <= 1e-12:
            continue
        X = np.column_stack([np.ones(y.size), u_res])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-18 else float("nan")
        cut = float(1.0 - np.sqrt(ss_res / y.size) /
                    max(np.sqrt(ss_tot / y.size), 1e-12)) if ss_tot > 1e-18 else 0.0
        if np.isfinite(r2) and r2 > best["r2"]:
            best = {"tau_best": float(t), "beta": float(coef[1]), "r2": r2,
                    "cut": cut, "component": coef[1] * u_res}
    return best


def _lag_profile(y: np.ndarray, x: np.ndarray, lags_s=LAGS_S) -> dict:
    """ρ(k) = corr(y[t], x[t−k])；k>0 表示 x 侧延迟（数据侧通道滞后方向）。

    零方差/太短返回 nan；``peak_lag`` 取 |ρ| 最大的滞后（不含 nan）。
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    out = {}
    for k in lags_s:
        if k >= 0:
            a, b = y[k:], x[: x.size - k] if k else x
        else:
            a, b = y[: y.size + k], x[-k:]
        m = min(a.size, b.size)
        a, b = a[:m], b[:m]
        if m < 20 or np.std(a) <= 1e-15 or np.std(b) <= 1e-15:
            out[k] = float("nan")
        else:
            out[k] = float(np.corrcoef(a, b)[0, 1])
    finite = {k: v for k, v in out.items() if np.isfinite(v)}
    peak = max(finite, key=lambda k: abs(finite[k])) if finite else None
    return {"rho": out, "peak_lag": peak,
            "peak_rho": finite.get(peak, float("nan")) if peak is not None else float("nan"),
            "rho0": out.get(0, float("nan"))}


def _lag_delay_like(lag: dict, rho_thr: float, ratio_thr: float,
                    min_lag: int = 1) -> bool:
    """延迟签名：存在 |k*|≥min_lag 且 |ρ(k*)|≥rho_thr 且 |ρ(k*)|≥ratio·|ρ(0)|。"""
    rho = lag["rho"]
    r0 = abs(rho.get(0, float("nan")))
    for k, v in rho.items():
        if abs(k) < min_lag or not np.isfinite(v):
            continue
        if abs(v) >= rho_thr and (not np.isfinite(r0) or abs(v) >= ratio_thr * max(r0, 1e-12)):
            return True
    return False


def _align_sweep(v_resid: np.ndarray, i: np.ndarray, dt: float,
                 lags_s=LAGS_S, tau: float = 5.0,
                 signal_rel: float = 0.05) -> dict:
    """数据侧 v/i 对齐审计：把 i 平移 k 样本后拟合 ``v_resid ~ r0·i_k + β·rc_k(τ)``。

    ``v_resid = v − ocv(soc)``（去 OCV 后的端电压残差）。逐 k 报 RMSE，
    ``delta_hat`` = RMSE 最小的 k（抛物线插值细化）；v 通道相对 i 延迟 δ 时
    最优 k=δ。``signal``：RMSE(k) 曲线相对起伏 ≥ signal_rel 才算有信号，
    否则 delta_hat 置 nan（噪声/平坦 ⇒ 无信号，不许出延迟估计）。
    """
    y = np.asarray(v_resid, float)
    i = np.asarray(i, float)
    rmse = {}
    for k in lags_s:
        if k >= 0:
            a, b = y[k:], i[: i.size - k] if k else i
        else:
            a, b = y[: y.size + k], i[-k:]
        m = min(a.size, b.size)
        a, b = a[:m], b[:m]
        if m < 50:
            rmse[k] = float("nan")
            continue
        u = _rc_branch(b, dt, tau)
        X = np.column_stack([np.ones(m), b, u])
        coef, *_ = np.linalg.lstsq(X, a, rcond=None)
        r = a - X @ coef
        rmse[k] = float(np.sqrt(np.mean(r ** 2)))
    finite = {k: v for k, v in rmse.items() if np.isfinite(v)}
    if len(finite) < 3:
        return {"rmse": rmse, "delta_hat": float("nan"), "signal": False,
                "peak_rmse": float("nan"), "rel_depth": float("nan")}
    ks = sorted(finite)
    best_k = min(finite, key=lambda k: finite[k])
    r_best = finite[best_k]
    r_max = max(finite.values())
    rel_depth = float((r_max - r_best) / max(r_best, 1e-12))
    signal = bool(rel_depth >= signal_rel)
    delta = float(best_k)
    # 抛物线插值细化（三点）
    if best_k not in (ks[0], ks[-1]):
        y0, y1, y2 = finite[best_k - 1], finite[best_k], finite[best_k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-15:
            delta = float(best_k + 0.5 * (y0 - y2) / den)
    return {"rmse": rmse, "delta_hat": delta if signal else float("nan"),
            "signal": signal, "peak_rmse": r_best, "rel_depth": rel_depth}


# =====================================================================
# stage probe
# =====================================================================

def _variant_kwargs():
    return dict(n_rc=3, fit_r0_cur=False)


def _make_inits(vec_base: np.ndarray):
    """3rc_fast 初值两组：基线补快支路 / 独立快端形状（tau3 均在快区间内）。"""
    r0k, r1, tau1, r2, tau2, _r3, _t3, _cur, dv = (
        _unpack(vec_base))
    a = np.array([*r0k, r1, tau1, r2, tau2, 0.02, 5.0, dv], float)
    b = np.array([*r0k * 1.1, 0.010, 8.0, 0.040, 400.0, 0.030, 15.0, 0.0], float)
    return [a, b]


def _unpack(vec):
    from bmsoc.ecm import _unpack_vec
    return _unpack_vec(vec)


def _run_ekf_innovation(vec, p, ocv2d, temp):
    """真值温度口径跑 EKF，返回 (innovation V, 结构 dict, v_clean)。"""
    from bmsoc.ecm import open_loop_from_truth
    from bmsoc.filters import EKF, FilterConfig
    from bmsoc.profiles import DrivingProfile

    s28 = _load("s28", os.path.join(ROOT, "scripts", "28_ekfq_round4.py"))
    params, r0k = s28._rc_from_vec(vec)
    n = len(p)
    dt = float(p.dt)
    t = np.arange(n) * dt
    soc = np.asarray(p.soc_ref, float)
    i = np.asarray(p.current_a, float)
    v = np.asarray(p.voltage_v, float)
    r0_per = np.interp(soc, (0.1, 0.5, 0.9), r0k)
    anchor = s28._R0Anchor(p.t_s, r0_per, p.capacity_ah)
    T_use = np.asarray(p.temp_c, float)
    pf = DrivingProfile(t_s=p.t_s, current_a=i, voltage_v=v, temp_c=T_use,
                        soc_ref=soc, capacity_ah=p.capacity_ah, meta=dict(p.meta))
    cfg = FilterConfig(soc0=float(np.clip(soc[0], 0.0, 1.0)), sigma_soc0=0.05)
    out = EKF(ocv=ocv2d, params=params, capacity_ah=p.capacity_ah,
              cfg=cfg).run(pf, anchor=anchor)
    inn = np.asarray(out["innovation"], float)
    v_clean = open_loop_from_truth(vec, soc, i, ocv2d, dt, temp)
    valid = t >= BURNIN_S
    comps = s28._structure_components(inn[valid], i[valid], dt)
    struct = {
        "inn_rms_mv": float(np.sqrt(np.mean(inn[valid] ** 2)) * 1e3),
        "mean_mv": float(np.mean(inn[valid]) * 1e3),
    }
    for k in ("slow", "odd", "even", "white"):
        struct[f"{k}_mv"] = float(np.sqrt(np.mean(comps[k] ** 2)) * 1e3)
    return inn, struct, v_clean, valid, s28


def stage_probe(dataset: str) -> dict:
    from bmsoc.ecm import fit_cycle_params, open_loop_from_truth

    s20 = _load("s20", os.path.join(ROOT, "scripts", "20_model_upgrade.py"))
    s23, folds, best, ocv2d = load_folds(dataset)
    s12 = _load("s12", os.path.join(ROOT, "scripts", "12_calibration_stress.py"))
    _, profiles, _ = (s12.load_calce() if dataset == "calce"
                      else s12.load_panasonic())
    prof_of = {(float(p.meta["temp_c"]), str(p.meta["profile"])): p for p in profiles}

    # 基线 2rc 与对照 3rc_slow 的逐折 vec（脚本 20 LOO 产物）
    up = pd.read_parquet(os.path.join(MET, f"model_upgrade_{dataset}_folds.parquet"))
    vec_of = {}
    for _, r in up.iterrows():
        vec_of[(str(r.variant), float(r.temp_c), str(r.held_out))] = np.asarray(
            json.loads(r.vec_json), float)
    if best != "2rc":
        print(f"[warn] best={best}（预期 2rc）——加快支路的对照口径仍按 n_rc=3 执行")

    fold_rows, struct_rows, fit_rows, lag_rows = [], [], [], []
    align_rows = []
    inn_frames = []
    insample_tau = {}   # (temp, profile) -> best τ（供 LOO-τ 用）

    for temp, held, train in s20.folds_for(profiles):
        prof_name = str(held.meta["profile"])
        key = (temp, prof_name)
        if key not in folds:
            continue
        vec_base = vec_of.get(("2rc", temp, prof_name))
        if vec_base is None:
            print(f"[skip] {key} 无 2rc 基线 vec")
            continue
        vec_slow = vec_of.get(("3rc", temp, prof_name))   # 慢端对照（可能缺失）
        f = folds[key]
        p = prof_of[key]
        dt = float(p.dt)

        # ---- (A) 模型级：3rc_fast LOO 重拟合（tau3 钉快端）----
        wins = [(pp.soc_ref, pp.current_a, pp.voltage_v, pp.dt, ocv2d, temp)
                for pp in train]
        best_fit = None
        for x0 in _make_inits(vec_base):
            vec_try, info = fit_cycle_params(
                wins, x0=x0, max_nfev=MAX_NFEV, dv_bound=0.15,
                tau3_bounds=TAU3_FAST, **_variant_kwargs())
            if best_fit is None or info["rmse_v"] < best_fit[1]["rmse_v"]:
                best_fit = (vec_try, info)
        vec_fast, info_fast = best_fit
        r0k_f, r1_f, tau1_f, r2_f, tau2_f, r3_f, tau3_f, _c, dv_f = _unpack(vec_fast)
        if not (TAU3_FAST[0] <= tau3_f <= TAU3_FAST[1]):
            raise SystemExit(f"[前提失败] {key} 重拟合 tau3={tau3_f} 不在快端区间")

        inn_b, st_b, _, valid, s28 = _run_ekf_innovation(vec_base, p, ocv2d, temp)
        inn_f, st_f, _, _, _ = _run_ekf_innovation(vec_fast, p, ocv2d, temp)
        if vec_slow is not None:
            inn_s, st_s, _, _, _ = _run_ekf_innovation(vec_slow, p, ocv2d, temp)
        else:
            inn_s, st_s = np.full_like(inn_b, np.nan), {k: float("nan") for k in st_b}

        # 开环残差口径（对照：滤波器吸收前后）
        e_b = open_loop_from_truth(vec_base, p.soc_ref, p.current_a, ocv2d, dt, temp) \
            - np.asarray(p.voltage_v, float)
        e_f = open_loop_from_truth(vec_fast, p.soc_ref, p.current_a, ocv2d, dt, temp) \
            - np.asarray(p.voltage_v, float)

        fold_rows.append({
            "dataset": dataset, "temp_c": temp, "profile": prof_name,
            "train_rmse_mv": info_fast["rmse_v"] * 1e3,
            "tau1_fast": tau1_f, "tau2_fast": tau2_f, "tau3_fast": tau3_f,
            "r3_fast": r3_f, "dv_fast_mv": dv_f * 1e3,
            "ol_rms_base_mv": float(np.sqrt(np.mean(e_b[valid] ** 2)) * 1e3),
            "ol_rms_fast_mv": float(np.sqrt(np.mean(e_f[valid] ** 2)) * 1e3),
            **{f"inn_{k}": v for k, v in st_b.items()},
            **{f"innfast_{k}": v for k, v in st_f.items()},
            **{f"inns3_{k}": v for k, v in st_s.items()},
        })

        # ---- (B) 结构级：加支路两段式（in-sample τ）----
        i_arr = np.asarray(p.current_a, float)
        white_b = s28._structure_components(inn_b[valid], i_arr[valid], dt)["white"]
        fit = _fit_fast_two_stage(white_b, i_arr[valid], dt)
        insample_tau[key] = fit["tau_best"]
        fit_rows.append({
            "dataset": dataset, "temp_c": temp, "profile": prof_name,
            "protocol": "insample",
            "tau_best_s": fit["tau_best"], "beta_mv_per_a": fit["beta"] * 1e3,
            "r2": fit["r2"], "rms_cut": fit["cut"],
            "white_rms_mv": float(np.sqrt(np.mean(white_b ** 2)) * 1e3),
        })

        # ---- 滞后互相关（v/i 采样对齐摸底，辅助口径）----
        didt = np.zeros_like(i_arr)
        didt[1:-1] = (i_arr[2:] - i_arr[:-2]) / (2.0 * dt)
        didt[0], didt[-1] = (i_arr[1] - i_arr[0]) / dt, (i_arr[-1] - i_arr[-2]) / dt
        for cov_name, cov in (("i", i_arr[valid]), ("didt", didt[valid])):
            lag = _lag_profile(white_b, cov, LAGS_S)
            for k in LAGS_S:
                lag_rows.append({"dataset": dataset, "temp_c": temp,
                                 "profile": prof_name, "cov": cov_name,
                                 "lag_s": k, "rho": lag["rho"][k],
                                 "peak_lag": lag["peak_lag"],
                                 "peak_rho": lag["peak_rho"],
                                 "rho0": lag["rho0"]})

        # ---- v/i 对齐审计（主口径：i 平移 RMSE 扫描）----
        from bmsoc.ecm import _ocv_at
        v_resid = np.asarray(p.voltage_v, float) - _ocv_at(
            ocv2d, np.asarray(p.soc_ref, float), temp)
        al = _align_sweep(v_resid, i_arr, dt, LAGS_S)
        for k in LAGS_S:
            lag_rows.append({"dataset": dataset, "temp_c": temp,
                             "profile": prof_name, "cov": "align_rmse",
                             "lag_s": k, "rho": al["rmse"][k],
                             "peak_lag": al["delta_hat"],
                             "peak_rho": al["peak_rmse"],
                             "rho0": al["rmse"].get(0, float("nan"))})
        align_rows.append({"dataset": dataset, "temp_c": temp, "profile": prof_name,
                           "delta_hat": al["delta_hat"], "signal": al["signal"],
                           "rel_depth": al["rel_depth"], "peak_rmse_mv":
                               al["peak_rmse"] * 1e3})

        # ---- 逐样本落盘（h30 注入源 + 图）----
        inn_frames.append(pd.DataFrame({
            "dataset": dataset, "temp_c": temp, "profile": prof_name,
            "t_s": np.asarray(p.t_s, float),
            "i_a": i_arr,
            "inn_base_mv": inn_b * 1e3,
            "inn_fast_mv": inn_f * 1e3,
            "inn_slow3_mv": inn_s * 1e3,
        }))
        print(f"[{temp:5.0f}] {prof_name:<5s} ol {fold_rows[-1]['ol_rms_base_mv']:5.1f}→"
              f"{fold_rows[-1]['ol_rms_fast_mv']:5.1f} mV  "
              f"inn white {st_b['white_mv']:5.1f}→{st_f['white_mv']:5.1f}  "
              f"τ3 {tau3_f:5.1f}s  快拟合 cut {fit['cut']:.0%} τ {fit['tau_best']:g}s",
              flush=True)

    if not fold_rows:
        raise SystemExit("probe 空结果")

    # ---- LOO-τ：留出折用同温度其他折 best τ 的中位数，只在留出折拟合 β ----
    for (temp, prof_name), g in pd.DataFrame(fit_rows).groupby(["temp_c", "profile"]):
        key = (float(temp), str(prof_name))
        others = [insample_tau[k] for k in insample_tau
                  if k[0] == float(temp) and k[1] != str(prof_name)
                  and np.isfinite(insample_tau[k])]
        if not others:
            others = [v for v in insample_tau.values() if np.isfinite(v)]
        tau_loo = float(np.median(others)) if others else float("nan")
        p = prof_of[key]
        i_arr = np.asarray(p.current_a, float)
        # 白色序列重取（与上面同口径）
        vec_base = vec_of[("2rc", *key)]
        inn_b, _, _, valid, s28 = _run_ekf_innovation(vec_base, p, ocv2d, float(temp))
        white_b = s28._structure_components(inn_b[valid], i_arr[valid], float(p.dt))["white"]
        fit_loo = _fit_fast_two_stage(white_b, i_arr[valid], float(p.dt), tau=tau_loo)
        fit_rows.append({
            "dataset": dataset, "temp_c": float(temp), "profile": str(prof_name),
            "protocol": "loo_tau",
            "tau_best_s": tau_loo, "beta_mv_per_a": fit_loo["beta"] * 1e3,
            "r2": fit_loo["r2"], "rms_cut": fit_loo["cut"],
            "white_rms_mv": float(np.sqrt(np.mean(white_b ** 2)) * 1e3),
        })
        # 逐样本快分量（LOO-τ）写回 innovation 帧
        comp_full = np.zeros(len(p))
        comp_full[valid] = fit_loo["component"]
        for fr in inn_frames:
            if (fr.temp_c.iloc[0] == float(temp) and str(fr.profile.iloc[0]) == str(prof_name)):
                fr["fast_loo_mv"] = comp_full * 1e3
                fr["white_base_mv"] = 0.0
                white_full = np.zeros(len(p))
                white_full[valid] = white_b
                fr["white_base_mv"] = white_full * 1e3
                break

    fdf = pd.DataFrame(fold_rows)
    ffdf = pd.DataFrame(fit_rows)
    ldf = pd.DataFrame(lag_rows)
    adf = pd.DataFrame(align_rows)
    for fr in inn_frames:
        if "fast_loo_mv" not in fr.columns:
            fr["fast_loo_mv"] = 0.0
        if "white_base_mv" not in fr.columns:
            fr["white_base_mv"] = 0.0
    idf = pd.concat(inn_frames, ignore_index=True)
    if fdf.empty:
        raise SystemExit("probe 空结果（fold）")
    fdf.to_parquet(os.path.join(MET, f"fastrc_r6_{dataset}_refit.parquet"), index=False)
    ffdf.to_parquet(os.path.join(MET, f"fastrc_r6_{dataset}_fastfit.parquet"), index=False)
    ldf.to_parquet(os.path.join(MET, f"fastrc_r6_{dataset}_lag.parquet"), index=False)
    adf.to_parquet(os.path.join(MET, f"fastrc_r6_{dataset}_align.parquet"), index=False)
    idf.to_parquet(os.path.join(MET, f"fastrc_r6_{dataset}_innovation.parquet"), index=False)

    # ---- 汇总（per_temp 中位）----
    per_temp = {}
    for temp in sorted(fdf.temp_c.unique()):
        sub = fdf[fdf.temp_c == temp]
        ins = ffdf[(ffdf.temp_c == temp) & (ffdf.protocol == "insample")]
        loo = ffdf[(ffdf.temp_c == temp) & (ffdf.protocol == "loo_tau")]
        al = adf[adf.temp_c == temp]
        per_temp[f"{temp:g}"] = {
            "white_base_mv": float(sub.inn_white_mv.median()),
            "white_fast_mv": float(sub.innfast_white_mv.median()),
            "white_slow3_mv": float(sub.inns3_white_mv.median()),
            "cut_model": float((1 - sub.innfast_white_mv / sub.inn_white_mv).median()),
            "cut_slow3": float((1 - sub.inns3_white_mv / sub.inn_white_mv).median()),
            "cut_insample": float(ins.rms_cut.median()) if len(ins) else float("nan"),
            "cut_loo": float(loo.rms_cut.median()) if len(loo) else float("nan"),
            "tau_insample_med_s": float(ins.tau_best_s.median()) if len(ins) else float("nan"),
            "slow_base_mv": float(sub.inn_slow_mv.median()),
            "slow_fast_mv": float(sub.innfast_slow_mv.median()),
            "odd_base_mv": float(sub.inn_odd_mv.median()),
            "odd_fast_mv": float(sub.innfast_odd_mv.median()),
            "inn_rms_base_mv": float(sub.inn_inn_rms_mv.median()),
            "inn_rms_fast_mv": float(sub.innfast_inn_rms_mv.median()),
            "tau3_med_s": float(sub.tau3_fast.median()),
            "delta_hat_med": float(al.delta_hat.median()) if len(al) else float("nan"),
            "align_signal_rate": float(al.signal.mean()) if len(al) else float("nan"),
        }

    # ---- 对齐审计合成对照（检出力）----
    rng = np.random.default_rng(20261102)
    i_syn = np.zeros(2400)
    i_syn[200:400] = -4.0
    i_syn[500:560] = np.linspace(0, -6, 60)
    i_syn[560:800] = -6.0
    i_syn[1000:1100] = np.linspace(0, -5, 100)
    i_syn[1100:1400] = -5.0
    i_syn[1600:1700] = np.linspace(0, -3, 100)
    i_syn[1700:1900] = -3.0
    i_syn += rng.normal(0, 0.05, i_syn.size)
    base_v = 0.05 * i_syn + 0.02 * _rc_branch(i_syn, 1.0, 5.0)

    def _align_ctrl(v):
        return {k: v for k, v in _align_sweep(v, i_syn, 1.0, LAGS_S).items()
                if k != "rmse"}

    ctrl = {
        "aligned": _align_ctrl(base_v + rng.normal(0, 3e-4, i_syn.size)),
        "delay2": _align_ctrl(
            0.05 * np.roll(i_syn, 2) + 0.02 * _rc_branch(np.roll(i_syn, 2), 1.0, 5.0)
            + rng.normal(0, 3e-4, i_syn.size)),
        "noise": _align_ctrl(rng.normal(0, 1e-3, i_syn.size)),
    }
    # 延迟对照剔除 roll 的回卷段（首尾各 2 样本）
    out = {
        "dataset": dataset, "best_variant": best, "n_folds": int(len(fdf)),
        "tau3_fast_bounds": list(TAU3_FAST), "tau_grid_s": list(TAU_GRID_S),
        "per_temp": per_temp,
        "align_controls": ctrl,
        "refit_tau3_range": [float(fdf.tau3_fast.min()), float(fdf.tau3_fast.max())],
    }
    with open(os.path.join(MET, f"fastrc_r6_{dataset}_probe.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("=== probe per_temp ===")
    for k, v in per_temp.items():
        print(f"  {k:>5s}°C  white {v['white_base_mv']:5.1f}→{v['white_fast_mv']:5.1f} mV "
              f"(model cut {v['cut_model']:+.0%}, slow3 cut {v['cut_slow3']:+.0%}, "
              f"insample {v['cut_insample']:+.0%}, loo {v['cut_loo']:+.0%})  "
              f"τ3 {v['tau3_med_s']:5.1f}s  slow {v['slow_base_mv']:4.1f}→{v['slow_fast_mv']:4.1f}  "
              f"δ̂ {v['delta_hat_med']:+.2f} (signal {v['align_signal_rate']:.0%})")
    print("=== align 合成对照 ===",
          json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                          for kk, vv in v.items()}
                      for k, v in ctrl.items()}, ensure_ascii=False))
    return out


# =====================================================================
# 判决线：工作记录 §48.2（只读，不回溯修改）
# =====================================================================

H29_L1_CUT = 0.25       # 模型级（3rc_fast LOO）冷温 innovation white 削减中位下限
H29_L2_CUT = 0.15       # 结构级（加支路 LOO-τ）white 削减中位下限
H29_L3_GROW_MV = 5.0    # 护栏：slow+even 允许恶化（mV）
H29_L3_RMSE = 0.10      # 护栏：innovation RMSE 允许恶化份额

H30_L1_QOK = 0.30       # 联动：minus_fast 或 refit_full 的 q 收敛率下限（full 基线 0%）
H30_L2_QOK = 0.30       # 充分性：fast_only 的 q 收敛率上限

# H31 仪器为探针阶段校准版：原 white×i 滞后互相关经合成对照验证无检出力
# （延迟签名 ∝ di̇ 不产生移位-i 峰，delay2 对照漏检）——判决改用
# i 平移 RMSE 扫描（_align_sweep，登记前定型，常量只读）。
H31_L1_DELTA = 1.0      # 冷温延迟估计中位 |δ̂| 下限（样本，1 Hz）
H31_L1_SIGNAL = 0.6     # 冷温有信号折占比下限（rel_depth≥5%）
H31_L2_REC = 1.0        # 合成 delay=2 的恢复容差 |δ̂−2| ≤ 1
H31_L2_FP = 0.5         # aligned 对照误检上界 |δ̂| ≤ 0.5


def _per_temp_med(df: pd.DataFrame, temp: float, col: str) -> float:
    sub = df[df.temp_c == temp]
    return float(sub[col].median()) if len(sub) else float("nan")


def stage_h29(dataset: str) -> dict:
    """H29：快支路重拟合压 white（模型级 3rc_fast vs 结构级 LOO-τ）。"""
    pp = os.path.join(MET, f"fastrc_r6_{dataset}_probe.json")
    if not os.path.exists(pp):
        raise SystemExit("缺 probe 产物 —— 先跑 --stage probe")
    with open(pp, encoding="utf-8") as fh:
        probe = json.load(fh)
    fdf = pd.read_parquet(os.path.join(MET, f"fastrc_r6_{dataset}_refit.parquet"))
    ffdf = pd.read_parquet(os.path.join(MET, f"fastrc_r6_{dataset}_fastfit.parquet"))
    cold = -20.0

    cut_model_cold = probe["per_temp"].get(f"{cold:g}", {}).get("cut_model", float("nan"))
    cut_loo_cold = probe["per_temp"].get(f"{cold:g}", {}).get("cut_loo", float("nan"))
    sub = fdf[fdf.temp_c == cold]
    grow = float(((sub.innfast_slow_mv + sub.innfast_even_mv)
                  - (sub.inn_slow_mv + sub.inn_even_mv)).median()) if len(sub) else float("nan")
    rmse_grow = float((sub.innfast_inn_rms_mv / sub.inn_inn_rms_mv - 1.0).median()) \
        if len(sub) else float("nan")

    l1 = bool(np.isfinite(cut_model_cold) and cut_model_cold >= H29_L1_CUT)
    l2 = bool(np.isfinite(cut_loo_cold) and cut_loo_cold >= H29_L2_CUT)
    l3 = bool((np.isfinite(grow) and grow <= H29_L3_GROW_MV)
              and (np.isfinite(rmse_grow) and rmse_grow <= H29_L3_RMSE))

    if l1 and l3:
        verdict = (f"成立：模型级快支路（tau3∈{TAU3_FAST}）冷温 white 削减 "
                   f"{cut_model_cold:.0%} ≥{H29_L1_CUT:.0%}，护栏未破"
                   f"（slow+even {grow:+.1f} mV，inn RMSE {rmse_grow:+.0%}）"
                   + (f"；结构级 LOO-τ 削减 {cut_loo_cold:.0%} 亦过线" if l2
                      else f"；结构级 LOO-τ 削减 {cut_loo_cold:.0%} 未过 {H29_L2_CUT:.0%} 线"))
    elif l2 and l3:
        verdict = (f"部分成立：结构级加支路 LOO-τ 削减 {cut_loo_cold:.0%} 过线，"
                   f"但模型级 3rc_fast 重拟合只削 {cut_model_cold:.0%} <{H29_L1_CUT:.0%}"
                   "——快模态形状存在，模型族重分配吸收不掉")
    elif l1 and not l3:
        verdict = (f"被否（护栏）：削减 {cut_model_cold:.0%} 达标但以 slow/even "
                   f"{grow:+.1f} mV 或 inn RMSE {rmse_grow:+.0%} 恶化为代价——拆东补西")
    else:
        verdict = (f"被否：模型级削减 {cut_model_cold:.0%}、结构级 {cut_loo_cold:.0%} "
                   f"均未过线（{H29_L1_CUT:.0%}/{H29_L2_CUT:.0%}）")

    out = {
        "dataset": dataset,
        "lines": {"L1_cut_model": H29_L1_CUT, "L2_cut_loo": H29_L2_CUT,
                  "L3_grow_mv": H29_L3_GROW_MV, "L3_rmse_grow": H29_L3_RMSE},
        "checks": {"L1_model_cut": l1, "L2_loo_cut": l2, "L3_guard": l3},
        "cold": {"cut_model": cut_model_cold, "cut_loo": cut_loo_cold,
                 "cut_insample": probe["per_temp"].get(f"{cold:g}", {}).get("cut_insample"),
                 "cut_slow3": probe["per_temp"].get(f"{cold:g}", {}).get("cut_slow3"),
                 "slow_even_grow_mv": grow, "inn_rmse_grow": rmse_grow},
        "per_temp": probe["per_temp"],
        "verdict": verdict,
    }
    with open(os.path.join(MET, f"fastrc_r6_{dataset}_h29.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"=== h29 {dataset}：{verdict}")
    return out


# =====================================================================
# stage h30 —— EKFQ 联动复测（回灌第八轮阈值）
# =====================================================================

def _run_replay_r6(dataset: str) -> pd.DataFrame:
    """零残差合成植物 + 分量注入 → EKFQ 收敛率（第八轮台架 + 快支路注入变体）。"""
    from bmsoc.ecm import open_loop_from_truth
    from bmsoc.filters import EKFQ, FilterConfig
    from bmsoc.profiles import DrivingProfile

    s28 = _load("s28", os.path.join(ROOT, "scripts", "28_ekfq_round4.py"))
    s23, folds, best, ocv2d = load_folds(dataset)
    s12 = _load("s12", os.path.join(ROOT, "scripts", "12_calibration_stress.py"))
    _, profiles, _ = (s12.load_calce() if dataset == "calce"
                      else s12.load_panasonic())
    prof_of = {(float(p.meta["temp_c"]), str(p.meta["profile"])): p for p in profiles}
    idf = pd.read_parquet(os.path.join(MET, f"fastrc_r6_{dataset}_innovation.parquet"))

    rows = []
    for temp, prof_name in REPLAY_FOLDS:
        key = (temp, prof_name)
        f = folds.get(key)
        p = prof_of.get(key)
        if f is None or p is None:
            print(f"[skip] {key} 不在 folds/profiles")
            continue
        g = idf[(idf.temp_c == temp) & (idf.profile == prof_name)].sort_values("t_s")
        if len(g) != len(p):
            raise SystemExit(f"[前提失败] {key} innovation 行数 {len(g)} != profile {len(p)}")
        inn_base = g["inn_base_mv"].to_numpy(float) * 1e-3
        inn_fast = g["inn_fast_mv"].to_numpy(float) * 1e-3
        fast_loo = g["fast_loo_mv"].to_numpy(float) * 1e-3
        n = len(p)
        dt = float(p.dt)
        t = np.arange(n) * dt
        soc = np.asarray(p.soc_ref, float)
        i = np.asarray(p.current_a, float)
        v_clean = open_loop_from_truth(f["vec"], soc, i, ocv2d, dt, temp)
        params, r0k = s28._rc_from_vec(f["vec"])
        r0_per = np.interp(soc, (0.1, 0.5, 0.9), r0k)
        anchor = s28._R0Anchor(p.t_s, r0_per, p.capacity_ah)
        valid = t >= BURNIN_S
        T_use = np.full(n, temp)
        cfg = FilterConfig(soc0=float(np.clip(soc[0], 0.0, 1.0)), sigma_soc0=0.05)

        def _demean(x):
            return x - np.mean(x[valid]) if valid.any() else x

        inject_of = {
            "none": np.zeros(n),
            "full": _demean(inn_base),
            "fast_only": _demean(fast_loo),
            "minus_fast": _demean(inn_base - fast_loo),
            "refit_full": _demean(inn_fast),
        }
        for inj in INJECT_SET:
            sig = inject_of[inj]
            for scale in REPLAY_SCALES:
                v = v_clean + scale * sig
                pf = DrivingProfile(t_s=p.t_s, current_a=i, voltage_v=v, temp_c=T_use,
                                    soc_ref=soc, capacity_ah=p.capacity_ah,
                                    meta=dict(p.meta))
                for s in CAP_SCALES:
                    out = EKFQ(ocv=ocv2d, params=params,
                               capacity_ah=p.capacity_ah * s,
                               cfg=cfg).run(pf, anchor=anchor)
                    qh = float(np.median(out["q_hat"][-max(1, n // 10):]))
                    q_rel = qh / p.capacity_ah - 1.0
                    rows.append({
                        "dataset": dataset, "temp_c": temp, "profile": prof_name,
                        "inject": inj, "scale": scale, "cap_scale": s,
                        "q_rel_err": float(q_rel),
                        "q_ok": bool(abs(q_rel) < 0.02),
                        "rmse_soc": float(np.sqrt(np.mean(
                            (out["soc_hat"] - soc) ** 2))),
                    })
        print(f"[{temp:5.0f}] {prof_name:<5s} replay r6 完成", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("replay 空结果")
    df.to_parquet(os.path.join(MET, f"fastrc_r6_{dataset}_replay.parquet"), index=False)
    return df


def _replay_summary_r6(df: pd.DataFrame) -> dict:
    tbl = {}
    for inj in INJECT_SET:
        sub = df[(df.inject == inj) & (df.scale == 1.0)]
        by_fold = {}
        for (temp, prof), g in sub.groupby(["temp_c", "profile"]):
            by_fold[f"{temp:g}_{prof}"] = {
                "q_ok_rate": float(g.q_ok.mean()),
                "q_rel_err_med": float(g.q_rel_err.median()),
                "rmse_soc_med": float(g.rmse_soc.median()),
            }
        tbl[inj] = {
            "q_ok_rate": float(sub.q_ok.mean()),
            "q_ok_rate_x0.5": float(df[(df.inject == inj) & (df.scale == 0.5)].q_ok.mean()),
            "q_ok_rate_x1.5": float(df[(df.inject == inj) & (df.scale == 1.5)].q_ok.mean()),
            "q_rel_err_med": float(sub.q_rel_err.median()),
            "rmse_soc_med": float(sub.rmse_soc.median()),
            "by_fold": by_fold,
        }
    return tbl


def stage_h30(dataset: str) -> dict:
    """H30：white 快支路成分被吸收后，EKFQ 的 q 收敛率是否回升。"""
    rp = os.path.join(MET, f"fastrc_r6_{dataset}_replay.parquet")
    df = pd.read_parquet(rp) if os.path.exists(rp) else _run_replay_r6(dataset)
    tbl = _replay_summary_r6(df)
    if tbl["none"]["q_ok_rate"] < 0.9:
        raise SystemExit(f"[前提失败] 零残差对照收敛率 "
                         f"{tbl['none']['q_ok_rate']:.0%} < 90% —— 台架前提死了")
    l1 = bool(max(tbl["minus_fast"]["q_ok_rate"], tbl["refit_full"]["q_ok_rate"])
              >= H30_L1_QOK)
    l2 = bool(tbl["fast_only"]["q_ok_rate"] <= H30_L2_QOK)
    best_inj = max(("minus_fast", "refit_full"),
                   key=lambda k: tbl[k]["q_ok_rate"])
    if l1 and l2:
        verdict = (f"成立：快支路成分是 white 中毒的主载体（只注快成分收敛 "
                   f"{tbl['fast_only']['q_ok_rate']:.0%} ≤{H30_L2_QOK:.0%}，"
                   f"剥/吸收后 {best_inj} 收敛 {tbl[best_inj]['q_ok_rate']:.0%} "
                   f"≥{H30_L1_QOK:.0%}）")
    elif l1 and not l2:
        verdict = (f"部分成立：吸收后收敛回升（{best_inj} "
                   f"{tbl[best_inj]['q_ok_rate']:.0%}）但快成分单独不充分毒"
                   f"（fast_only {tbl['fast_only']['q_ok_rate']:.0%}>{H30_L2_QOK:.0%}）")
    elif l2 and not l1:
        verdict = (f"部分成立：快成分充分毒（fast_only 收敛 "
                   f"{tbl['fast_only']['q_ok_rate']:.0%}）但吸收后收敛率仅 "
                   f"minus_fast {tbl['minus_fast']['q_ok_rate']:.0%} / "
                   f"refit_full {tbl['refit_full']['q_ok_rate']:.0%} "
                   f"<{H30_L1_QOK:.0%}——white 其余成分仍在灌 q")
    else:
        verdict = (f"被否：吸收后收敛率 minus_fast {tbl['minus_fast']['q_ok_rate']:.0%} / "
                   f"refit_full {tbl['refit_full']['q_ok_rate']:.0%}，"
                   f"fast_only {tbl['fast_only']['q_ok_rate']:.0%}——快支路不是联动瓶颈")
    out = {
        "dataset": dataset,
        "lines": {"L1_qok_recovery": H30_L1_QOK, "L2_qok_suff": H30_L2_QOK},
        "checks": {"L1_recovery": l1, "L2_fast_sufficiency": l2},
        "table": tbl,
        "verdict": verdict,
    }
    with open(os.path.join(MET, f"fastrc_r6_{dataset}_h30.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"=== h30 {dataset}：{verdict}")
    for inj in INJECT_SET:
        v = tbl[inj]
        print(f"  {inj:<11s} q收敛 {v['q_ok_rate']:.0%} "
              f"(×0.5 {v['q_ok_rate_x0.5']:.0%} / ×1.5 {v['q_ok_rate_x1.5']:.0%})  "
              f"q̂ {v['q_rel_err_med']:+.2%}  soc {v['rmse_soc_med']*100:.2f} pp")
    return out


# =====================================================================
# stage h31 —— v/i 采样对齐
# =====================================================================

def stage_h31(dataset: str) -> dict:
    """H31：v 通道相对 i 通道存在固定延迟/平均窗错位（数据侧，平移扫描口径）。"""
    pp = os.path.join(MET, f"fastrc_r6_{dataset}_probe.json")
    if not os.path.exists(pp):
        raise SystemExit("缺 probe 产物 —— 先跑 --stage probe")
    with open(pp, encoding="utf-8") as fh:
        probe = json.load(fh)
    adf = pd.read_parquet(os.path.join(MET, f"fastrc_r6_{dataset}_align.parquet"))
    cold = -20.0

    sub = adf[adf.temp_c == cold]
    med_delta = float(sub.delta_hat.median()) if len(sub) else float("nan")
    sig_rate = float(sub.signal.mean()) if len(sub) else float("nan")
    n_sig = int(sub.signal.sum()) if len(sub) else 0
    # 同号一致性：在有信号折里 δ̂ 与中位数同号的占比
    sig = sub[sub.signal & sub.delta_hat.notna()]
    sign_agree = float((np.sign(sig.delta_hat) == np.sign(med_delta)).mean()) \
        if len(sig) and np.isfinite(med_delta) else float("nan")

    l1 = bool(np.isfinite(med_delta) and abs(med_delta) >= H31_L1_DELTA
              and sig_rate >= H31_L1_SIGNAL)

    ctrl = probe.get("align_controls", {})
    rec = ctrl.get("delay2", {}).get("delta_hat", float("nan"))
    fp = ctrl.get("aligned", {}).get("delta_hat", float("nan"))
    noise_sig = bool(ctrl.get("noise", {}).get("signal", False))
    l2 = bool(np.isfinite(rec) and abs(rec - 2.0) <= H31_L2_REC
              and np.isfinite(fp) and abs(fp) <= H31_L2_FP and not noise_sig)

    if l1 and l2:
        verdict = (f"成立：冷温 δ̂ 中位 {med_delta:+.2f} 样本"
                   f"（|δ̂|≥{H31_L1_DELTA}，有信号折 {n_sig}/5，同号 {sign_agree:.0%}）"
                   "——数据侧 v/i 时序错位是 white 的成分之一")
    elif l2 and not l1:
        verdict = (f"被否：冷温 δ̂ 中位 {med_delta:+.2f} 样本"
                   f"（有信号折 {sig_rate:.0%}）未过 |δ̂|≥{H31_L1_DELTA} 线——"
                   "数据侧无固定延迟证据，white 出口在模型侧"
                   f"（合成对照 delay2 恢复 {rec:+.2f}、aligned 误检 {fp:+.2f}）")
    else:
        verdict = (f"被否（度量无检出力）：合成 delay2 恢复 {rec:+.2f}（容差 ±{H31_L2_REC}）、"
                   f"aligned 误检 {fp:+.2f}（上界 {H31_L2_FP}）、噪声误报 {noise_sig}"
                   "——L1 读数不能当证据")
    out = {
        "dataset": dataset,
        "lines": {"L1_delta": H31_L1_DELTA, "L1_signal": H31_L1_SIGNAL,
                  "L2_rec": H31_L2_REC, "L2_fp": H31_L2_FP},
        "checks": {"L1_delay_signature": l1, "L2_detection_power": l2},
        "cold": {"delta_hat_med": med_delta, "signal_rate": sig_rate,
                 "n_signal": n_sig, "sign_agree": sign_agree,
                 "per_fold": sub[["profile", "delta_hat", "signal",
                                  "rel_depth"]].to_dict("records")},
        "align_controls": ctrl,
        "verdict": verdict,
    }
    with open(os.path.join(MET, f"fastrc_r6_{dataset}_h31.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"=== h31 {dataset}：{verdict}")
    for r in out["cold"]["per_fold"]:
        print(f"  {r['profile']:<5s} δ̂ {r['delta_hat']:+.2f}  "
              f"signal {r['signal']}  depth {r['rel_depth']:.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["calce", "panasonic"], default="panasonic")
    ap.add_argument("--stage", choices=["probe", "h29", "h30", "h31"], required=True)
    args = ap.parse_args()
    fn = {"probe": stage_probe, "h29": stage_h29, "h30": stage_h30,
          "h31": stage_h31}[args.stage]
    fn(args.dataset)


if __name__ == "__main__":
    main()
