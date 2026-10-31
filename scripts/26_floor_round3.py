#!/usr/bin/env python
"""26_floor_round3.py — 第七轮：冷温小电流地板的机制判决（v1.6）。

「冷温 0–1 A 也 42 mV」是第六轮收尾后 NCA 残差中位的最大头，热修正（R(T_meas)）
不碰它。本脚本按工作记录 §42 的假设做摸底与判决实验：

- ``--stage probe``  摸底（写假设前的勘察，不算判决）：逐温度低电流残差、
  电流/记忆/SOC 分箱、OCV(T_meas) 修正、HPPC 长静置尾部外推、**regime 分解**
  （起始静置 / 工况内静置 / 动态低电流 / 动态大电流）、lead/mid 分解、
  u2 相关、四变体低电流地板对照。
- ``--stage h20``    H20 静态表差判决：OCV 表冷温不自洽假设——起始静置纯表差 +
  尾段静置（约束-τ 外推）表差 + 合成正对照（已知表偏/残余极化注入）。
- ``--stage h21``    H21 口径混合判决：dv 的 regime 不一致（静态 vs 动态基准不自洽
  的真身）+ 尾段静置未收敛残余极化 + regime-aware 重定基准后的地板重构。
- ``--stage h22``    H22 动态残差共享结构判决：动态低电流残差对 (u1,u2) 核与
  逐温度偏置/SOC 形状的同温 LOO 迁移（train→heldout，v1.4 方法）。

判决阈值见工作记录 §42（事前固定）。输出 ``results/metrics/floor_r3_*``。
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

IBIN_EDGES = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 30.0])
TSINCE_EDGES = np.array([0.0, 10.0, 60.0, 300.0, 1200.0, np.inf])
SOC_BINS = np.linspace(0.0, 1.0, 11)
LOWI_A = 1.0          # 「小电流」定义：|i| ≤ 1 A
BIGI_A = 2.0          # 「大电流」记忆源：|i| > 2 A
LEAD_S = 400.0        # 头段边界（t < LEAD_S）
REST_THR = 0.05       # 静置电流阈值（A）
REST_MIN_S = 60.0     # 静置段最短时长
TAIL_SKIP = 0.4       # 尾部外推跳过前 40%，只用后 60% 段（与前轮证据链一致）

# ---- §42 事前登记的判决线（实现只读，不再调）----
H20_L_START_MV = 10.0     # 已歇透静置纯表差 |med| 严格自洽线
H20_L_START_MAJOR_MV = 30.0  # 超过此幅度才算「表差是地板主因」
H20_L_EQ_MV = 20.0        # 尾段平衡表差区间整段超此线 ⇒ 表差成立
H20_L_CTRL_RECOV_MV = 5.0 # 合成正对照回收误差上限
H21_L_DDV_COLD_MV = 80.0  # Δdv @ −20 °C 下限（不一致幅度）
H21_L_DDV_WARM_MV = 15.0  # Δdv @ 25 °C 上限（冷温特有）
H21_L_ARTIFACT_SHARE = 0.30   # 口径重定移除的 SSE 份额下限
H21_L_SLOPE_MV300 = 20.0  # 尾段静置末斜率下限（未收敛证据）
SETTLED_SLOPE_MV300 = 25.0  # 末斜率 ≤ 此值视作已歇透（静置读数可用）
H22_L_CUT = 0.25          # LOO 迁移 RMSE 降幅「成立」线
H22_L_WEAK = 0.10         # LOO 迁移 RMSE 降幅「被否」线


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_folds(dataset: str):
    s23 = _load_module("s23", os.path.join(ROOT, "scripts", "23_residual_round2.py"))
    return s23.fold_residuals(dataset)


def _time_since_big(i: np.ndarray, dt: float) -> np.ndarray:
    """每个样本距上一次 |i| > BIGI_A 的秒数（之前没有则 NaN）。"""
    big = np.flatnonzero(np.abs(i) > BIGI_A)
    idx = np.arange(len(i))
    if big.size == 0:
        return np.full(len(i), np.nan)
    pos = np.searchsorted(big, idx, side="right") - 1
    out = np.where(pos >= 0, (idx - big[np.clip(pos, 0, None)]) * dt, np.nan)
    return out.astype(float)


def _stats(e: np.ndarray) -> dict:
    if e.size == 0:
        return {"n": 0, "med_mv": float("nan"), "med_abs_mv": float("nan"),
                "rmse_mv": float("nan"), "bias_mv": float("nan")}
    return {"n": int(e.size), "med_mv": float(np.median(e) * 1e3),
            "med_abs_mv": float(np.median(np.abs(e)) * 1e3),
            "rmse_mv": float(np.sqrt(np.mean(e ** 2)) * 1e3),
            "bias_mv": float(np.mean(e) * 1e3)}


def _tail_equilibrium(t: np.ndarray, v: np.ndarray, i0: int, i1: int):
    """静置尾部指数外推到平衡：v(t) ≈ c + a·exp(−(t−t0)/τ)。

    只用后 ``1 − TAIL_SKIP`` 段拟合；τ 在 10–20000 s 对数网格上选 SSE 最小。
    返回 None 表示样本不足。gap = (c − v[i1])·1e3：>0 表示末电压仍在平衡之下
    （静置没歇透，OCV 点带残余极化）。
    """
    k0 = i0 + int(TAIL_SKIP * (i1 - i0))
    if i1 - k0 < 20:
        return None
    tt = t[k0:i1 + 1] - t[k0]
    vv = v[k0:i1 + 1]
    best = None
    for tau in np.logspace(1.0, 4.3, 40):
        x = np.exp(-tt / max(tau, 1.0))
        A = np.column_stack([x, np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(A, vv, rcond=None)
        sse = float(np.sum((vv - A @ coef) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(tau), coef)
    sse, tau, coef = best
    # 末 300 s 斜率（电压还在往哪边走）
    j = int(np.searchsorted(t, t[i1] - 300.0, side="right")) - 1
    j = max(j, k0)
    slope = float((v[i1] - v[j]) / max(t[i1] - t[j], 1.0) * 300.0 * 1e3) \
        if i1 > j else 0.0
    return {"gap_mv": float((coef[1] - v[i1]) * 1e3), "tau_s": tau,
            "sse_mv2": sse * 1e6, "end_slope_mv300": slope,
            "dur_s": float(t[i1] - t[i0])}


def _tail_equilibrium_fixedtau(t: np.ndarray, v: np.ndarray, i0: int, i1: int,
                              tau: float):
    """同 :func:`_tail_equilibrium`，但 τ 固定（HPPC 静置 τ_med 先验）。

    单-τ 自由拟合在 300 s 短静置上会把 τ 拉到快模态、低估残余 gap；
    约束 τ 只拟合 (a, c)，给出「慢模态口径」的外推。"""
    k0 = i0 + int(TAIL_SKIP * (i1 - i0))
    if i1 - k0 < 20:
        return None
    tt = t[k0:i1 + 1] - t[k0]
    vv = v[k0:i1 + 1]
    x = np.exp(-tt / max(float(tau), 1.0))
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, vv, rcond=None)
    return {"gap_mv": float((coef[1] - v[i1]) * 1e3), "a_mv": float(coef[0] * 1e3)}


def _rc_states(vec: np.ndarray, i: np.ndarray, dt: float):
    """零状态 RC 响应 (u1, u2)（与 open_loop_from_truth 同相位）。"""
    from bmsoc.ecm import _rc_lfilter, _unpack_vec
    r0k, r1, tau1, r2, tau2, r3, tau3, r0_cur, dv = _unpack_vec(vec)
    u1 = _rc_lfilter(r1, tau1, i, dt)
    u2 = _rc_lfilter(r2, tau2, i, dt)
    return np.asarray(u1, float), np.asarray(u2, float)


def _regime_masks(i: np.ndarray, t: np.ndarray):
    """样本级 regime 掩码。

    返回 ``(mask_start, mask_other, dyn_low, dyn_high, segs)``：
    起始静置（上电后、首次大电流前的长静置）/ 其余长静置 /
    动态低电流（|i|≤LOWI_A 不在长静置内）/ 动态大电流。
    """
    from bmsoc.profiles import rest_segments
    n = len(i)
    segs = rest_segments(t, i, thr=REST_THR, min_dur_s=REST_MIN_S)
    big = np.flatnonzero(np.abs(i) > BIGI_A)
    t_first_big = float(t[big[0]]) if big.size else np.inf
    mask_start = np.zeros(n, bool)
    mask_other = np.zeros(n, bool)
    for i0, i1, dur in segs:
        is_start = t[i0] < LEAD_S and t[i0] <= t_first_big
        (mask_start if is_start else mask_other)[i0:i1 + 1] = True
    rest = mask_start | mask_other
    dyn_low = (~rest) & (np.abs(i) <= LOWI_A)
    dyn_high = (~rest) & (np.abs(i) > LOWI_A)
    return mask_start, mask_other, dyn_low, dyn_high, segs


def _fold_vmeas(f: dict, ocv2d):
    """重算 v_meas 与 OCV 表值（e = open_loop(vec) − v_meas）。"""
    from bmsoc.ecm import open_loop_from_truth
    temp, dt = float(f["temp_c"]), float(f["dt"])
    vmod = open_loop_from_truth(f["vec"], f["soc"], f["i"], ocv2d, dt, temp)
    vmeas = vmod - f["e"]
    ocv_tab = np.asarray(ocv2d.batch(f["soc"], np.full(len(f["e"]), temp)), float)
    return vmeas, ocv_tab


# ---------------------------------------------------------------- stage probe

def stage_probe(dataset: str) -> dict:
    folds, best, ocv2d = load_folds(dataset)
    if not folds:
        raise SystemExit("fold_residuals 返回空 —— 先跑 scripts/20/23")
    keys = sorted(folds)
    bin_rows, sum_rows = [], []
    hist_rows, soc_rows = [], []
    regime_rows, restseg_rows = [], []

    # ---- ⑤ HPPC 长静置尾部外推 gap（仅 panasonic 有多倍率 HPPC；先算，
    #      其 τ_med 作为 profile 尾段约束-τ 外推的先验） ----
    rest_rows = []
    if dataset == "panasonic":
        from bmsoc import io_panasonic as io
        from bmsoc.profiles import rest_segments
        for path, temp in io.list_hppc():
            m = io.load_meas(path)
            t_h, i_h, v_h = m["t_s"], m["current_a"], m["voltage_v"]
            for i0, i1, dur in rest_segments(t_h, i_h, thr=REST_THR, min_dur_s=600.0):
                fit = _tail_equilibrium(t_h, v_h, i0, i1)
                if fit is None:
                    continue
                rest_rows.append({"temp_c": float(temp), "i0": int(i0), "i1": int(i1),
                                  **fit})
    else:
        print(f"[{dataset}] 无多倍率 HPPC，跳过静置尾部外推（记 0 行）")
    rdf = pd.DataFrame(rest_rows)
    rdf.to_parquet(os.path.join(MET, f"floor_r3_{dataset}_rests.parquet"), index=False)
    if dataset == "panasonic" and rdf.empty:
        raise SystemExit("HPPC 静置段 0 条 —— 检查 rest_segments 阈值/时长")
    tau_by_temp = {}
    if not rdf.empty:
        for temp in rdf.temp_c.unique():
            tau_by_temp[float(temp)] = float(rdf[rdf.temp_c == temp].tau_s.median())

    for key in keys:
        f = folds[key]
        temp, prof, dt = f["temp_c"], f["profile"], float(f["dt"])
        e, i, soc, T = f["e"], f["i"], f["soc"], f["T"]
        n = len(e)
        t = np.arange(n) * dt
        t_since = _time_since_big(i, dt)
        low = np.abs(i) <= LOWI_A
        dv = float(f["vec"][-1])
        mask_start, mask_other, dyn_low, dyn_high, segs = _regime_masks(i, t)

        # ---- ① 电流分箱（确认地板位置） ----
        ai = np.abs(i)
        for b0, b1 in zip(IBIN_EDGES[:-1], IBIN_EDGES[1:]):
            m = (ai > b0) & (ai <= b1)
            if not m.any():
                continue
            st = _stats(e[m])
            bin_rows.append({"dataset": dataset, "temp_c": temp, "profile": prof,
                             "ibin": f"({b0:g},{b1:g}]", **st})

        # ---- ② 温度 × 记忆分箱（仅低电流） ----
        for s0, s1 in zip(TSINCE_EDGES[:-1], TSINCE_EDGES[1:]):
            m = low & (t_since >= s0) & (t_since < s1) & np.isfinite(t_since)
            if not m.any():
                continue
            st = _stats(e[m])
            hist_rows.append({"dataset": dataset, "temp_c": temp, "profile": prof,
                              "tbin": f"[{s0:g},{s1:g})", **st})

        # ---- ③ 温度 × SOC 分箱（仅低电流） ----
        for s0, s1 in zip(SOC_BINS[:-1], SOC_BINS[1:]):
            m = low & (soc >= s0) & (soc < s1 + (1e-9 if s1 >= 1.0 else 0.0))
            if not m.any():
                continue
            st = _stats(e[m])
            soc_rows.append({"dataset": dataset, "temp_c": temp, "profile": prof,
                             "socbin": f"[{s0:.1f},{s1:.1f})", **st})

        # ---- ④ 低电流残差：OCV(T_meas) 修正前后 + lead/mid + u2 相关 ----
        e_low = e[low]
        doccv = (np.asarray(ocv2d.batch(soc[low], T[low]), float)
                 - np.asarray(ocv2d.batch(soc[low], np.full(low.sum(), temp)), float))
        st_raw, st_t = _stats(e_low), _stats(e_low + doccv)
        lead = t < LEAD_S
        el, em = e[low & lead], e[low & ~lead]
        sse_l = float(np.sum(el ** 2)) if el.size else 0.0
        sse_m = float(np.sum(em ** 2)) if em.size else 0.0
        u1, u2 = _rc_states(f["vec"], i, dt)
        rho_u2 = (float(np.corrcoef(e[low], u2[low])[0, 1])
                  if low.sum() > 10 and np.std(u2[low]) > 0 else float("nan"))
        sum_rows.append({
            "dataset": dataset, "temp_c": temp, "profile": prof, "n": n,
            "n_low": int(low.sum()),
            "all_rmse_mv": float(np.sqrt(np.mean(e ** 2)) * 1e3),
            "low_rmse_mv": st_raw["rmse_mv"], "low_med_abs_mv": st_raw["med_abs_mv"],
            "low_med_mv": st_raw["med_mv"], "low_bias_mv": st_raw["bias_mv"],
            "low_rmse_ocvt_mv": st_t["rmse_mv"], "low_med_abs_ocvt_mv": st_t["med_abs_mv"],
            "t_swing_low_k": float(T[low].max() - T[low].min()) if low.any() else 0.0,
            "dv_mv": dv * 1e3,
            "n_lead": int(el.size), "n_mid": int(em.size),
            "rmse_lead_mv": float(np.sqrt(np.mean(el ** 2)) * 1e3) if el.size else float("nan"),
            "rmse_mid_mv": float(np.sqrt(np.mean(em ** 2)) * 1e3) if em.size else float("nan"),
            "sse_share_lead": (sse_l / (sse_l + sse_m)) if (sse_l + sse_m) > 0 else float("nan"),
            "bias_lead_mv": float(np.median(el) * 1e3) if el.size else float("nan"),
            "bias_mid_mv": float(np.median(em) * 1e3) if em.size else float("nan"),
            "rho_u2": rho_u2,
        })

        # ---- ⑥ regime 分解（起始静置 / 工况静置 / 动态低 / 动态高）----
        for name, m in (("rest_start", mask_start), ("rest_other", mask_other),
                        ("dyn_low", dyn_low), ("dyn_high", dyn_high)):
            if not m.any():
                continue
            st = _stats(e[m])
            st_nd = _stats(e[m] - dv)
            regime_rows.append({
                "dataset": dataset, "temp_c": temp, "profile": prof, "regime": name,
                "n": st["n"], "rmse_mv": st["rmse_mv"], "med_mv": st["med_mv"],
                "med_abs_mv": st["med_abs_mv"], "bias_mv": st["bias_mv"],
                "rmse_nodv_mv": st_nd["rmse_mv"], "med_nodv_mv": st_nd["med_mv"],
                "bias_nodv_mv": st_nd["bias_mv"],
                "sse_mv2": float(np.sum(e[m] ** 2)),
            })

        # ---- ⑦ 静置段逐段：纯表差 + 尾部外推 ----
        if segs:
            vmeas, ocv_tab = _fold_vmeas(f, ocv2d)
            big = np.flatnonzero(np.abs(i) > BIGI_A)
            t_first_big = float(t[big[0]]) if big.size else np.inf
            for i0, i1, dur in segs:
                is_start = int(t[i0] < LEAD_S and t[i0] <= t_first_big)
                j0 = max(i0, i1 - int(20.0 / dt))
                fit = _tail_equilibrium(t, vmeas, i0, i1)
                if fit is None:
                    continue
                soc_med = float(np.median(soc[i0:i1 + 1]))
                ocv_at = float(np.median(ocv_tab[i0:i1 + 1]))
                c_mv = fit["gap_mv"] + vmeas[i1] * 1e3
                tau_fix = tau_by_temp.get(float(temp))
                fitf = (_tail_equilibrium_fixedtau(t, vmeas, i0, i1, tau_fix)
                        if tau_fix else None)
                restseg_rows.append({
                    "dataset": dataset, "temp_c": temp, "profile": prof,
                    "is_start": is_start, "dur_s": float(dur),
                    "t_frac": float(t[i0] / max(t[-1], 1.0)),
                    "soc_med": soc_med,
                    "T_med": float(np.median(T[i0:i1 + 1])),
                    "table_gap_mv": float(np.median(vmeas[j0:i1 + 1] - ocv_at) * 1e3),
                    "gap_end_mv": float((vmeas[i1] - ocv_at) * 1e3),
                    "gap_eq_mv": float(c_mv - ocv_at * 1e3),
                    "gap_eq_fixtau_mv": (float(vmeas[i1] * 1e3 + fitf["gap_mv"]
                                                - ocv_at * 1e3)
                                         if fitf else float("nan")),
                    "tau_s": fit["tau_s"],
                    "tau_fix_s": float(tau_fix) if tau_fix else float("nan"),
                    "end_slope_mv300": fit["end_slope_mv300"],
                })

    bdf = pd.DataFrame(bin_rows)
    hdf = pd.DataFrame(hist_rows)
    sdf = pd.DataFrame(soc_rows)
    fdf = pd.DataFrame(sum_rows)
    rgdf = pd.DataFrame(regime_rows)
    rsdf = pd.DataFrame(restseg_rows)
    if bdf.empty or hdf.empty or sdf.empty or fdf.empty or rgdf.empty:
        raise SystemExit(f"探针空结果 bin={len(bdf)} hist={len(hdf)} "
                         f"soc={len(sdf)} regime={len(rgdf)}")
    bdf.to_parquet(os.path.join(MET, f"floor_r3_{dataset}_ibins.parquet"), index=False)
    hdf.to_parquet(os.path.join(MET, f"floor_r3_{dataset}_hist.parquet"), index=False)
    sdf.to_parquet(os.path.join(MET, f"floor_r3_{dataset}_socbins.parquet"), index=False)
    fdf.to_parquet(os.path.join(MET, f"floor_r3_{dataset}_folds.parquet"), index=False)
    rgdf.to_parquet(os.path.join(MET, f"floor_r3_{dataset}_regimes.parquet"), index=False)
    rsdf.to_parquet(os.path.join(MET, f"floor_r3_{dataset}_restsegs.parquet"), index=False)

    # ---- ⑧ 四变体低电流地板对照（加阶/电流内阻是否碰地板）----
    variants_low = {}
    up_path = os.path.join(MET, f"model_upgrade_{dataset}_folds.parquet")
    if os.path.exists(up_path):
        from bmsoc.ecm import open_loop_from_truth
        up = pd.read_parquet(up_path)
        ocv2d_only = ocv2d
        profiles_needed = {key: folds[key] for key in keys}
        for variant in sorted(up.variant.unique()):
            params_of = {}
            for _, r in up[up.variant == variant].iterrows():
                params_of[(float(r.temp_c), str(r.held_out))] = np.asarray(
                    json.loads(r.vec_json), float)
            per_temp_v = {}
            for key, f in profiles_needed.items():
                vec = params_of.get((float(f["temp_c"]), str(f["profile"])))
                if vec is None:
                    continue
                vh = open_loop_from_truth(vec, f["soc"], f["i"], ocv2d_only,
                                          float(f["dt"]), float(f["temp_c"]))
                ev = vh - (open_loop_from_truth(f["vec"], f["soc"], f["i"],
                                                ocv2d_only, float(f["dt"]),
                                                float(f["temp_c"])) - f["e"])
                low = np.abs(f["i"]) <= LOWI_A
                if not low.any():
                    continue
                per_temp_v.setdefault(f"{key[0]:g}", []).append(
                    float(np.sqrt(np.mean(ev[low] ** 2)) * 1e3))
            variants_low[variant] = {k: float(np.median(v))
                                     for k, v in sorted(per_temp_v.items())}

    # ---- 汇总 ----
    per_temp = {}
    for temp in sorted(fdf.temp_c.unique()):
        sub = fdf[fdf.temp_c == temp]
        per_temp[f"{temp:g}"] = {
            "n_low": int(sub.n_low.sum()),
            "low_rmse_mv": float(sub.low_rmse_mv.median()),
            "low_med_abs_mv": float(sub.low_med_abs_mv.median()),
            "low_bias_mv": float(sub.low_bias_mv.median()),
            "low_rmse_ocvt_mv": float(sub.low_rmse_ocvt_mv.median()),
            "all_rmse_mv": float(sub.all_rmse_mv.median()),
            "dv_med_mv": float(sub.dv_mv.median()),
            "dv_spread_mv": float(sub.dv_mv.max() - sub.dv_mv.min()),
            "t_swing_low_k": float(sub.t_swing_low_k.median()),
            "rmse_lead_mv": float(sub.rmse_lead_mv.median()),
            "rmse_mid_mv": float(sub.rmse_mid_mv.median()),
            "sse_share_lead": float(sub.sse_share_lead.median()),
            "bias_lead_mv": float(sub.bias_lead_mv.median()),
            "bias_mid_mv": float(sub.bias_mid_mv.median()),
            "rho_u2": float(sub.rho_u2.median()),
        }
    hist_by_temp = {}
    for temp in sorted(hdf.temp_c.unique()):
        sub = hdf[hdf.temp_c == temp]
        hist_by_temp[f"{temp:g}"] = {
            tbin: {"n": int(sub[sub.tbin == tbin].n.sum()),
                   "med_abs_mv": float(sub[sub.tbin == tbin].med_abs_mv.median()),
                   "med_mv": float(sub[sub.tbin == tbin].med_mv.median())}
            for tbin in [f"[{s0:g},{s1:g})" for s0, s1
                         in zip(TSINCE_EDGES[:-1], TSINCE_EDGES[1:])]
            if (sub.tbin == tbin).any()
        }
    soc_shape = {}
    for temp in sorted(sdf.temp_c.unique()):
        sub = sdf[sdf.temp_c == temp]
        g = sub.groupby("socbin").med_mv.median()
        soc_shape[f"{temp:g}"] = {
            "med_by_bin": {str(k): float(v) for k, v in g.items()},
            "shape_pp_mv": float(g.max() - g.min()) if len(g) else float("nan"),
        }
    rest_by_temp = {}
    for temp in sorted(rdf.temp_c.unique()) if not rdf.empty else []:
        sub = rdf[rdf.temp_c == temp]
        rest_by_temp[f"{temp:g}"] = {
            "n_rests": int(len(sub)),
            "gap_med_mv": float(sub.gap_mv.median()),
            "gap_p90_mv": float(np.quantile(sub.gap_mv, 0.9)),
            "gap_min_mv": float(sub.gap_mv.min()),
            "gap_max_mv": float(sub.gap_mv.max()),
            "end_slope_med_mv300": float(sub.end_slope_mv300.median()),
            "tau_med_s": float(sub.tau_s.median()),
            "gap_vs_soc_rho": (float(np.corrcoef(
                [(i0 + i1) / 2 for i0, i1 in zip(sub.i0, sub.i1)], sub.gap_mv)[0, 1])
                if len(sub) >= 4 else float("nan")),
        }
    regime_by_temp = {}
    for temp in sorted(rgdf.temp_c.unique()):
        sub = rgdf[rgdf.temp_c == temp]
        regime_by_temp[f"{temp:g}"] = {
            reg: {"n": int(sub[sub.regime == reg].n.sum()),
                  "rmse_mv": float(sub[sub.regime == reg].rmse_mv.median()),
                  "med_mv": float(sub[sub.regime == reg].med_mv.median()),
                  "med_nodv_mv": float(sub[sub.regime == reg].med_nodv_mv.median()),
                  "bias_nodv_mv": float(sub[sub.regime == reg].bias_nodv_mv.median())}
            for reg in ["rest_start", "rest_other", "dyn_low", "dyn_high"]
            if (sub.regime == reg).any()
        }
    restseg_by_temp = {}
    for temp in sorted(rsdf.temp_c.unique()) if not rsdf.empty else []:
        sub = rsdf[rsdf.temp_c == temp]
        restseg_by_temp[f"{temp:g}"] = {
            "n_segs": int(len(sub)),
            "n_start": int(sub.is_start.sum()),
            "table_gap_start_mv": (float(sub[sub.is_start == 1].table_gap_mv.median())
                                   if (sub.is_start == 1).any() else float("nan")),
            "table_gap_other_mv": (float(sub[sub.is_start == 0].table_gap_mv.median())
                                   if (sub.is_start == 0).any() else float("nan")),
            "gap_end_other_mv": (float(sub[sub.is_start == 0].gap_end_mv.median())
                                 if (sub.is_start == 0).any() else float("nan")),
            "gap_eq_other_mv": (float(sub[sub.is_start == 0].gap_eq_mv.median())
                                if (sub.is_start == 0).any() else float("nan")),
            "gap_eq_fixtau_other_mv": (
                float(sub[sub.is_start == 0].gap_eq_fixtau_mv.median())
                if (sub.is_start == 0).any() else float("nan")),
            "slope_other_mv300": (float(sub[sub.is_start == 0].end_slope_mv300.median())
                                  if (sub.is_start == 0).any() else float("nan")),
            "soc_other_med": (float(sub[sub.is_start == 0].soc_med.median())
                              if (sub.is_start == 0).any() else float("nan")),
        }
    summary = {
        "dataset": dataset, "best_variant": best, "n_folds": len(fdf),
        "lowi_a": LOWI_A, "bigi_a": BIGI_A, "lead_s": LEAD_S,
        "per_temp": per_temp, "hist_by_temp": hist_by_temp,
        "soc_shape": soc_shape, "rests_by_temp": rest_by_temp,
        "regime_by_temp": regime_by_temp, "restsegs_by_temp": restseg_by_temp,
        "variants_low_rmse_mv": variants_low,
    }
    with open(os.path.join(MET, f"floor_r3_{dataset}_probe.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    print(f"=== probe {dataset}：低电流折中位 "
          f"{fdf.low_rmse_mv.median():.1f} mV（全电流 {fdf.all_rmse_mv.median():.1f}） ===")
    for tk, tv in per_temp.items():
        print(f"  {tk:>5s}°C  n_low={tv['n_low']:6d}  低电流 {tv['low_rmse_mv']:5.1f} mV  "
              f"OCV(T)修后 {tv['low_rmse_ocvt_mv']:5.1f}  dv {tv['dv_med_mv']:+7.1f} "
              f"±{tv['dv_spread_mv']:.0f}  lead/mid {tv['rmse_lead_mv']:5.1f}/"
              f"{tv['rmse_mid_mv']:5.1f}  rho_u2 {tv['rho_u2']:+.2f}")
    for tk, tv in regime_by_temp.items():
        rs = tv.get("rest_start", {}); ro = tv.get("rest_other", {})
        dl = tv.get("dyn_low", {})
        print(f"  {tk:>5s}°C  rest_start e_med {rs.get('med_mv', float('nan')):+7.1f} "
              f"(nodv {rs.get('med_nodv_mv', float('nan')):+6.1f})  "
              f"rest_other {ro.get('med_mv', float('nan')):+7.1f}  "
              f"dyn_low {dl.get('med_mv', float('nan')):+6.1f}  "
              f"dyn_low_rmse {dl.get('rmse_mv', float('nan')):5.1f}")
    for tk, tv in restseg_by_temp.items():
        print(f"  {tk:>5s}°C  静置段 n={tv['n_segs']} (start {tv['n_start']})  "
              f"表差 start {tv['table_gap_start_mv']:+6.1f}  "
              f"other {tv['table_gap_other_mv']:+6.1f}  "
              f"gap_end {tv['gap_end_other_mv']:+6.1f}  "
              f"gap_eq {tv['gap_eq_other_mv']:+6.1f}  "
              f"slope {tv['slope_other_mv300']:+5.1f} mV/300s")
    for vk, vv in variants_low.items():
        print(f"  变体 {vk:>8s}  低电流地板 {vv}")
    return summary


# ---------------------------------------------------------------- stage h20

def _ctrl_key(tau: float, dur: float, tag: str) -> str:
    return f"tau{tau:.4g}_dur{dur:.4g}_{tag}"


def _synthetic_rest_control(rng: np.random.Generator, taus) -> dict:
    """合成正对照：已知表偏 δ 与已知残余极化 a·exp(−t/τ) 能否被读数分开。

    - settled 静置（a=0）：table_gap（= gap_end）应 ≈ δ —— 直接读数的回收检验
    - polarized 静置：gap_end ≈ δ + a·exp(−dur/τ)（未收敛污染），
      外推 gap_eq 应 ≈ δ —— 平衡点估计器的回收检验（naive 自由-τ vs 约束-τ）
    - twomode 静置（50 mV/30 s + 80 mV/τ）：现实情形的下界标定 ——
      约束-τ 估计器在快模态残留下的偏差有多大，就是真实数据读数的不确定度
    """
    dt = 1.0
    out = {}
    for tau_true in taus:
        for dur in (300.0, 600.0):
            for tag, amps in (
                    ("settled", ((0.0, tau_true),)),
                    ("polarized", ((80.0, tau_true),)),
                    ("twomode", ((50.0, 30.0), (80.0, tau_true)))):
                n = int(dur / dt) + 1
                t = np.arange(n) * dt
                delta = 0.020  # +20 mV 表偏
                v = np.full(n, 4.0 + delta)
                for amp_mv, tau in amps:
                    v += (amp_mv * 1e-3) * np.exp(-t / tau)
                v += rng.normal(0.0, 0.001, n)
                table = 4.0
                gap_end = float((v[-1] - table) * 1e3)
                fit = _tail_equilibrium(t, v, 0, n - 1)
                fitf = _tail_equilibrium_fixedtau(t, v, 0, n - 1, tau_true)
                out[_ctrl_key(tau_true, dur, tag)] = {
                    "delta_mv": delta * 1e3,
                    "gap_end_mv": gap_end,
                    "gap_eq_naive_mv": (float(v[-1] * 1e3 + fit["gap_mv"] - table * 1e3)
                                        if fit else float("nan")),
                    "gap_eq_fixtau_mv": (float(v[-1] * 1e3 + fitf["gap_mv"] - table * 1e3)
                                         if fitf else float("nan")),
                    "tau_naive_s": fit["tau_s"] if fit else float("nan"),
                    "end_slope_mv300": fit["end_slope_mv300"] if fit else float("nan"),
                }
    return out


def stage_h20(dataset: str) -> dict:
    """H20：OCV 表冷温不自洽（静态表差）判决。"""
    probe_path = os.path.join(MET, f"floor_r3_{dataset}_probe.json")
    rs_path = os.path.join(MET, f"floor_r3_{dataset}_restsegs.parquet")
    if not os.path.exists(probe_path) or not os.path.exists(rs_path):
        raise SystemExit("缺 probe/restsegs 产物 —— 先跑 --stage probe")
    with open(probe_path, encoding="utf-8") as fh:
        probe = json.load(fh)
    rs = pd.read_parquet(rs_path)

    tau_priors = sorted({round(v["tau_med_s"], 1)
                         for v in probe["rests_by_temp"].values()
                         if np.isfinite(v.get("tau_med_s", float("nan")))})
    taus = sorted({120.0, 500.0} | set(tau_priors))
    rng = np.random.default_rng(20261031)
    control = _synthetic_rest_control(rng, taus)
    ctrl_settled = [v for k, v in control.items() if k.endswith("settled")]
    ctrl_recov_err = max(abs(v["gap_end_mv"] - v["delta_mv"]) for v in ctrl_settled)
    # 平衡点估计器回收偏差：单模态（方法本征）与双模态（现实下界）
    ctrl_single_bias = max(abs(v["gap_eq_fixtau_mv"] - v["delta_mv"])
                           for k, v in control.items()
                           if k.endswith("polarized") and "dur300" in k)
    ctrl_two_bias = max(abs(v["gap_eq_fixtau_mv"] - v["delta_mv"])
                        for k, v in control.items()
                        if k.endswith("twomode") and "dur300" in k)

    per_temp = {}
    verdict_rows = []
    for temp in sorted(rs.temp_c.unique()):
        sub = rs[rs.temp_c == temp]
        settled = sub[(sub.is_start == 1)
                      | ((sub.is_start == 0)
                         & (sub.end_slope_mv300 <= SETTLED_SLOPE_MV300))]
        unsettled = sub[(sub.is_start == 0)
                        & (sub.end_slope_mv300 > SETTLED_SLOPE_MV300)]
        # L1：已歇透静置的纯表差（直接测）
        table_gap = float(settled.table_gap_mv.median()) if len(settled) else float("nan")
        # L2：未歇透尾段的平衡表差两估计器区间（naive 自由-τ vs 约束-τ）
        if len(unsettled):
            g_naive = float(unsettled.gap_eq_mv.median())
            g_fix = float(unsettled.gap_eq_fixtau_mv.median())
            lo, hi = (g_naive, g_fix) if g_naive <= g_fix else (g_fix, g_naive)
        else:
            g_naive = g_fix = lo = hi = float("nan")
        gap_end = float(sub[sub.is_start == 0].gap_end_mv.median()) \
            if (sub.is_start == 0).any() else float("nan")
        slope = float(sub[sub.is_start == 0].end_slope_mv300.median()) \
            if (sub.is_start == 0).any() else float("nan")
        per_temp[f"{temp:g}"] = {
            "n_settled": int(len(settled)), "n_unsettled": int(len(unsettled)),
            "table_gap_settled_mv": table_gap,
            "gap_end_other_mv": gap_end,
            "gap_eq_naive_mv": g_naive, "gap_eq_fixtau_mv": g_fix,
            "eq_bracket_lo_mv": lo, "eq_bracket_hi_mv": hi,
            "slope_other_mv300": slope,
            "tau_prior_s": float(sub.tau_fix_s.median()) if len(sub) else float("nan"),
            "soc_unsettled_med": (float(unsettled.soc_med.median())
                                  if len(unsettled) else float("nan")),
        }
        ok_strict = bool(np.isfinite(table_gap) and abs(table_gap) <= H20_L_START_MV)
        # L2 区间规则：整段区间落在 ±H20_L_EQ_MV 之外（同号）⇒ 低 SOC 表差成立
        if np.isfinite(lo):
            eq_outside = bool(lo > H20_L_EQ_MV or hi < -H20_L_EQ_MV)
            eq_major = bool(lo > H20_L_START_MAJOR_MV or hi < -H20_L_START_MAJOR_MV)
        else:
            eq_outside = eq_major = False  # 无未歇透段：L2 空过
        verdict_rows.append({"temp_c": temp, "table_gap": table_gap,
                             "ok_strict": ok_strict, "eq_outside": eq_outside,
                             "eq_major": eq_major,
                             "major": bool((np.isfinite(table_gap)
                                            and abs(table_gap) > H20_L_START_MAJOR_MV)
                                           or eq_major)})
    vdf = pd.DataFrame(verdict_rows)
    ctrl_ok = bool(ctrl_recov_err <= H20_L_CTRL_RECOV_MV
                   and ctrl_single_bias <= H20_L_CTRL_RECOV_MV
                   and ctrl_two_bias <= H20_L_CTRL_RECOV_MV)
    major_any = bool(vdf.major.any())
    strict_fail = vdf[~vdf.ok_strict].temp_c.tolist()
    eq_bad = vdf[vdf.eq_outside].temp_c.tolist()

    if major_any:
        verdict = (f"成立（部分）：温度 {vdf[vdf.major].temp_c.tolist()} 表差超 "
                   f"{H20_L_START_MAJOR_MV:g} mV —— 表差量化进误差预算")
    elif eq_bad:
        verdict = f"成立（部分）：温度 {eq_bad} 尾段平衡表差区间整段超线 —— 低 SOC 表差"
    elif strict_fail:
        verdict = (f"被否（主因）：{strict_fail} 已歇透静置表差超严格线 "
                   f"{H20_L_START_MV:g} mV 但 ≤{H20_L_START_MAJOR_MV:g} mV"
                   "（单点偏差记预算）；其余温度表自洽 —— 表差不是地板来源")
    else:
        verdict = ("被否：静态表差全线 ≤ 线（已歇透 ≤{s:g} mV、尾段区间含 0）—— "
                   "OCV 表自洽，地板另有来源（见 H21/H22）").format(s=H20_L_START_MV)
    out = {
        "dataset": dataset,
        "lines": {
            "L1_strict_mv": H20_L_START_MV,
            "L1_major_mv": H20_L_START_MAJOR_MV,
            "L2_eq_mv": H20_L_EQ_MV,
            "L3_ctrl_recov_mv": H20_L_CTRL_RECOV_MV,
        },
        "per_temp": per_temp,
        "synthetic_control": control,
        "ctrl_recov_err_mv": float(ctrl_recov_err),
        "ctrl_fixtau_bias_single_mv": float(ctrl_single_bias),
        "ctrl_fixtau_bias_two_mv": float(ctrl_two_bias),
        "ctrl_ok": ctrl_ok,
        "strict_fail_temps": strict_fail,
        "eq_outside_temps": eq_bad,
        "major_temps": vdf[vdf.major].temp_c.tolist(),
        "verdict": verdict,
    }
    with open(os.path.join(MET, f"floor_r3_{dataset}_h20.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"=== h20 {dataset}：{verdict}")
    for tk, tv in per_temp.items():
        print(f"  {tk:>5s}°C  已歇透表差 {tv['table_gap_settled_mv']:+6.1f} mV "
              f"(n={tv['n_settled']})  尾段区间 "
              f"[{tv['eq_bracket_lo_mv']:+6.1f},{tv['eq_bracket_hi_mv']:+6.1f}] "
              f"(naive/fix {tv['gap_eq_naive_mv']:+6.1f}/{tv['gap_eq_fixtau_mv']:+6.1f})  "
              f"slope {tv['slope_other_mv300']:+5.1f}")
    print(f"  合成正对照：settled 表偏回收误差 {ctrl_recov_err:.2f} mV；"
          f"约束-τ 平衡点偏差 单模态 {ctrl_single_bias:.1f} / "
          f"双模态 {ctrl_two_bias:.1f} mV")
    return out


# ---------------------------------------------------------------- stage h21

def stage_h21(dataset: str) -> dict:
    """H21：dv 的 regime 不一致 + 尾段未收敛 = 地板大头（口径混合）。"""
    folds, best, ocv2d = load_folds(dataset)
    if not folds:
        raise SystemExit("fold_residuals 返回空")
    keys = sorted(folds)
    rows = []
    for key in keys:
        f = folds[key]
        temp, prof, dt = f["temp_c"], f["profile"], float(f["dt"])
        e, i, soc, T = f["e"], f["i"], f["soc"], f["T"]
        n = len(e)
        t = np.arange(n) * dt
        dv = float(f["vec"][-1])
        e_nodv = e - dv
        mask_start, mask_other, dyn_low, dyn_high, segs = _regime_masks(i, t)
        dyn = dyn_low | dyn_high
        low = np.abs(i) <= LOWI_A
        if not dyn.any():
            continue
        # dv_static：起始静置（协议歇透、双侧极化为零）；无则用「已歇透」静置段尾 20 s
        if mask_start.any():
            dv_static = -float(np.mean(e_nodv[mask_start]))
            static_src = "start"
        else:
            vmeas, _ = _fold_vmeas(f, ocv2d)
            tails = []
            for i0, i1, dur in segs:
                fit = _tail_equilibrium(t, vmeas, i0, i1)
                if fit is None or abs(fit["end_slope_mv300"]) > SETTLED_SLOPE_MV300:
                    continue
                j0 = max(i0, i1 - int(20.0 / dt))
                tails.append(e_nodv[j0:i1 + 1])
            if not tails:
                continue
            dv_static = -float(np.mean(np.concatenate(tails)))
            static_src = "settled_tail"
        dv_dyn = -float(np.mean(e_nodv[dyn]))
        ddv = dv_dyn - dv_static
        # regime-aware 重定基准：静置样本用 dv_static，动态样本用 dv_dyn
        dv_reg = np.where(mask_start | mask_other, dv_static, dv_dyn)
        e2 = e_nodv + dv_reg
        low_rmse0 = float(np.sqrt(np.mean(e[low] ** 2)) * 1e3)
        low_rmse2 = float(np.sqrt(np.mean(e2[low] ** 2)) * 1e3)
        sse0 = float(np.sum(e[low] ** 2))
        sse2 = float(np.sum(e2[low] ** 2))
        artifact_share = 1.0 - sse2 / sse0 if sse0 > 0 else float("nan")
        sse_rest2 = float(np.sum(e2[low & (mask_start | mask_other)] ** 2))
        # 尾段静置未收敛证据
        ot = mask_other
        end_slope = float("nan")
        u_res_med = float("nan")
        if ot.any():
            vmeas, ocv_tab = _fold_vmeas(f, ocv2d)
            segs_ot = [s for s in segs if mask_other[s[0]]]
            slopes, ures = [], []
            for i0, i1, dur in segs_ot:
                fit = _tail_equilibrium(t, vmeas, i0, i1)
                if fit is None:
                    continue
                slopes.append(fit["end_slope_mv300"])
                c = vmeas[i1] + fit["gap_mv"] * 1e-3
                ures.append(float(np.mean(c - vmeas[i0:i1 + 1])) * 1e3)
            end_slope = float(np.median(slopes)) if slopes else float("nan")
            u_res_med = float(np.median(ures)) if ures else float("nan")
        # 源探针：动态低电流 e_nodv vs T_meas（自热偶型线索）
        t_swing = float(T[dyn].max() - T[dyn].min()) if dyn.any() else 0.0
        rho_t = (float(np.corrcoef(e_nodv[dyn_low], T[dyn_low])[0, 1])
                 if dyn_low.sum() > 20 and np.std(T[dyn_low]) > 0 else float("nan"))
        rows.append({
            "dataset": dataset, "temp_c": temp, "profile": prof,
            "static_src": static_src,
            "dv_fit_mv": dv * 1e3, "dv_static_mv": dv_static * 1e3,
            "dv_dyn_mv": dv_dyn * 1e3, "ddv_mv": ddv * 1e3,
            "low_rmse0_mv": low_rmse0, "low_rmse_rebased_mv": low_rmse2,
            "artifact_share": artifact_share,
            "rest_sse_share_rebased": (sse_rest2 / sse2) if sse2 > 0 else float("nan"),
            "end_slope_mv300": end_slope, "u_res_med_mv": u_res_med,
            "t_swing_dyn_k": t_swing, "rho_enodv_T": rho_t,
        })
    fdf = pd.DataFrame(rows)
    if fdf.empty:
        raise SystemExit("h21 空结果")
    fdf.to_parquet(os.path.join(MET, f"floor_r3_{dataset}_h21_folds.parquet"), index=False)

    per_temp = {}
    verdict_rows = []
    for temp in sorted(fdf.temp_c.unique()):
        sub = fdf[fdf.temp_c == temp]
        ddv_med = float(sub.ddv_mv.median())
        rebase = float(sub.low_rmse_rebased_mv.median())
        share = float(sub.artifact_share.median())
        slope = float(sub.end_slope_mv300.median())
        per_temp[f"{temp:g}"] = {
            "ddv_med_mv": ddv_med,
            "dv_static_med_mv": float(sub.dv_static_mv.median()),
            "dv_dyn_med_mv": float(sub.dv_dyn_mv.median()),
            "dv_fit_med_mv": float(sub.dv_fit_mv.median()),
            "low_rmse0_mv": float(sub.low_rmse0_mv.median()),
            "low_rmse_rebased_mv": rebase,
            "artifact_share": share,
            "rest_sse_share_rebased": float(sub.rest_sse_share_rebased.median()),
            "end_slope_mv300": slope,
            "u_res_med_mv": float(sub.u_res_med_mv.median()),
            "rho_enodv_T": float(sub.rho_enodv_T.median()),
            "t_swing_dyn_k": float(sub.t_swing_dyn_k.median()),
        }
        verdict_rows.append({
            "temp_c": temp, "ddv_med_mv": ddv_med, "share": share, "slope": slope,
        })
    vdf = pd.DataFrame(verdict_rows)
    cold = vdf[vdf.temp_c == -20.0]
    warm = vdf[vdf.temp_c == 25.0]
    l1 = bool(len(cold) and abs(cold.ddv_med_mv.iloc[0]) >= H21_L_DDV_COLD_MV
              and len(warm) and abs(warm.ddv_med_mv.iloc[0]) <= H21_L_DDV_WARM_MV)
    l2 = bool(len(cold) and cold.share.iloc[0] >= H21_L_ARTIFACT_SHARE)
    l3 = bool(len(cold) and cold.slope.iloc[0] >= H21_L_SLOPE_MV300)
    if l1 and l2 and l3:
        verdict = ("成立：地板大头 = dv regime 混合 + 尾段未收敛（口径/状态问题）；"
                   f"口径重定移除 {per_temp['-20']['artifact_share']:.0%} SSE，"
                   f"−20 °C 低电流地板 {per_temp['-20']['low_rmse0_mv']:.1f} → "
                   f"{per_temp['-20']['low_rmse_rebased_mv']:.1f} mV")
    elif l1 and l2:
        verdict = "部分成立：口径混合确认，未收敛证据不足（L3 未过）"
    elif l1:
        verdict = "部分成立：regime 不一致确认，但口径伪影份额不足（L2 未过）"
    else:
        verdict = "被否：口径混合解释不足，动态模型误差为主（转 H22）"
    out = {
        "dataset": dataset,
        "lines": {
            "L1_ddv_cold_mv": H21_L_DDV_COLD_MV,
            "L1_ddv_warm_mv": H21_L_DDV_WARM_MV,
            "L2_artifact_share": H21_L_ARTIFACT_SHARE,
            "L3_slope_mv300": H21_L_SLOPE_MV300,
        },
        "checks": {"L1_ddv": l1, "L2_artifact": l2, "L3_unconverged": l3},
        "per_temp": per_temp,
        "verdict": verdict,
    }
    with open(os.path.join(MET, f"floor_r3_{dataset}_h21.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"=== h21 {dataset}：{verdict}")
    for tk, tv in per_temp.items():
        print(f"  {tk:>5s}°C  Δdv {tv['ddv_med_mv']:+7.1f} mV "
              f"(static {tv['dv_static_med_mv']:+6.1f} / dyn {tv['dv_dyn_med_mv']:+6.1f})  "
              f"地板 {tv['low_rmse0_mv']:5.1f}→{tv['low_rmse_rebased_mv']:5.1f}  "
              f"伪影份额 {tv['artifact_share']:.0%}  "
              f"slope {tv['end_slope_mv300']:+5.1f}  u_res {tv['u_res_med_mv']:+5.1f}  "
              f"rho(e,T) {tv['rho_enodv_T']:+.2f}")
    return out


# ---------------------------------------------------------------- stage h22

def stage_h22(dataset: str) -> dict:
    """H22：动态低电流残差的共享结构（(u1,u2) 核 / 逐温度偏置+SOC 形状）LOO 迁移。

    训练只用**动态 regime**（长静置剥掉，避免把口径伪影喂给核）低电流样本；
    held-out 折评估同口径动态低电流 RMSE。v1.4 train→heldout 方法。
    """
    folds, best, ocv2d = load_folds(dataset)
    if not folds:
        raise SystemExit("fold_residuals 返回空")
    keys = sorted(folds)
    rows = []
    for key in keys:
        f = folds[key]
        temp, prof, dt = f["temp_c"], f["profile"], float(f["dt"])
        e, i, soc = f["e"], f["i"], f["soc"]
        n = len(e)
        t = np.arange(n) * dt
        mask_start, mask_other, dyn_low, dyn_high, _ = _regime_masks(i, t)
        m_h = dyn_low  # held-out 评估集：动态低电流
        if not m_h.any():
            continue
        u1_h, u2_h = _rc_states(f["vec"], i, dt)
        train = [folds[k] for k in keys if k != key and folds[k]["temp_c"] == temp]
        if not train:
            continue
        # ---- (a) (1, u1, u2) 核 ----
        X_parts, y_parts = [], []
        for g in train:
            gi, gt = g["i"], np.arange(len(g["e"])) * float(g["dt"])
            _, _, gdl, _, _ = _regime_masks(gi, gt)
            if not gdl.any():
                continue
            g1, g2 = _rc_states(g["vec"], gi, float(g["dt"]))
            X_parts.append(np.column_stack([np.ones(gdl.sum()), g1[gdl], g2[gdl]]))
            y_parts.append(g["e"][gdl])
        if X_parts:
            X = np.vstack(X_parts)
            y = np.concatenate(y_parts)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            e_corr_kernel = e[m_h] - (coef[0] + coef[1] * u1_h[m_h] + coef[2] * u2_h[m_h])
        else:
            coef = np.zeros(3)
            e_corr_kernel = e[m_h].copy()
        # ---- (b) 逐温度偏置 + SOC 形状（10 bins，动态低电流训练）----
        c_bin = np.zeros(len(SOC_BINS) - 1)
        pool_e, pool_s = [], []
        for g in train:
            gi, gt = g["i"], np.arange(len(g["e"])) * float(g["dt"])
            _, _, gdl, _, _ = _regime_masks(gi, gt)
            if not gdl.any():
                continue
            pool_e.append(g["e"][gdl])
            pool_s.append(g["soc"][gdl])
        if pool_e:
            pe = np.concatenate(pool_e)
            ps = np.concatenate(pool_s)
            idx = np.clip(np.digitize(ps, SOC_BINS) - 1, 0, len(c_bin) - 1)
            for b in range(len(c_bin)):
                mb = idx == b
                c_bin[b] = float(np.median(pe[mb])) if mb.any() else 0.0
            # 空箱回落到全局偏置
            glob = float(np.median(pe))
            for b in range(len(c_bin)):
                if not (idx == b).any():
                    c_bin[b] = glob
            idx_h = np.clip(np.digitize(soc[m_h], SOC_BINS) - 1, 0, len(c_bin) - 1)
            e_corr_shape = e[m_h] - c_bin[idx_h]
        else:
            e_corr_shape = e[m_h].copy()
        base = float(np.sqrt(np.mean(e[m_h] ** 2)) * 1e3)
        rmse_k = float(np.sqrt(np.mean(e_corr_kernel ** 2)) * 1e3)
        rmse_s = float(np.sqrt(np.mean(e_corr_shape ** 2)) * 1e3)
        rows.append({
            "dataset": dataset, "temp_c": temp, "profile": prof,
            "n_eval": int(m_h.sum()), "n_train": len(train),
            "rmse_base_mv": base, "rmse_kernel_mv": rmse_k, "rmse_shape_mv": rmse_s,
            "cut_kernel": 1.0 - rmse_k / base if base > 0 else float("nan"),
            "cut_shape": 1.0 - rmse_s / base if base > 0 else float("nan"),
            "k0_mv": coef[0] * 1e3, "k1": coef[1], "k2": coef[2],
        })
    fdf = pd.DataFrame(rows)
    if fdf.empty:
        raise SystemExit("h22 空结果")
    fdf.to_parquet(os.path.join(MET, f"floor_r3_{dataset}_h22_folds.parquet"), index=False)

    per_temp = {}
    for temp in sorted(fdf.temp_c.unique()):
        sub = fdf[fdf.temp_c == temp]
        per_temp[f"{temp:g}"] = {
            "rmse_base_mv": float(sub.rmse_base_mv.median()),
            "rmse_kernel_mv": float(sub.rmse_kernel_mv.median()),
            "rmse_shape_mv": float(sub.rmse_shape_mv.median()),
            "cut_kernel": float(sub.cut_kernel.median()),
            "cut_shape": float(sub.cut_shape.median()),
            "k1_med": float(sub.k1.median()),
            "k2_med": float(sub.k2.median()),
        }
    cold = per_temp.get("-20", {})
    ck, cs = cold.get("cut_kernel", float("nan")), cold.get("cut_shape", float("nan"))
    l1 = bool(ck >= H22_L_CUT)
    l2 = bool(cs >= H22_L_CUT)
    if l1 and l2:
        verdict = "成立：(u 核) 与 (偏置+形状) 双双可迁移 —— 共享结构进修正路线"
    elif l1:
        verdict = ("成立（u 核）：动态残差是慢极化状态共享结构（模型族欠拟合）；"
                   f"形状迁移弱（{cs:+.0%}）")
    elif l2:
        verdict = f"成立（形状）：逐温度偏置/形状可迁移（{cs:+.0%}）；u 核弱（{ck:+.0%}）"
    elif ck < H22_L_WEAK and cs < H22_L_WEAK:
        verdict = ("被否：u 核与形状迁移均 <10% —— 动态地板非共享结构"
                   "（滤波器在线状态可吸收的部分不在此口径）")
    else:
        verdict = f"部分成立：迁移 10–25%（u 核 {ck:+.0%} / 形状 {cs:+.0%}），量化进预算"
    out = {
        "dataset": dataset,
        "lines": {"L_cut_pass": H22_L_CUT, "L_cut_weak": H22_L_WEAK},
        "per_temp": per_temp,
        "cold_cut_kernel": float(ck), "cold_cut_shape": float(cs),
        "verdict": verdict,
    }
    with open(os.path.join(MET, f"floor_r3_{dataset}_h22.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"=== h22 {dataset}：{verdict}")
    for tk, tv in per_temp.items():
        print(f"  {tk:>5s}°C  动态低电流 {tv['rmse_base_mv']:5.1f} → "
              f"u核 {tv['rmse_kernel_mv']:5.1f} ({tv['cut_kernel']:+.0%})  "
              f"形状 {tv['rmse_shape_mv']:5.1f} ({tv['cut_shape']:+.0%})  "
              f"k1 {tv['k1_med']:+.2f} k2 {tv['k2_med']:+.2f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["probe", "h20", "h21", "h22"], required=True)
    ap.add_argument("--dataset", choices=["calce", "panasonic"], default="panasonic")
    args = ap.parse_args()
    if args.stage == "probe":
        stage_probe(args.dataset)
    elif args.stage == "h20":
        stage_h20(args.dataset)
    elif args.stage == "h21":
        stage_h21(args.dataset)
    else:
        stage_h22(args.dataset)


if __name__ == "__main__":
    main()
