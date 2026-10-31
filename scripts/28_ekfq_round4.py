#!/usr/bin/env python
"""28_ekfq_round4.py — 第八轮：滤波后残差口径的 EKFQ 复验（v1.7）。

问题（记录 §43.7 首选）
---------------------
第七轮把冷温低电流开环地板（59.2 mV @ −20 °C）拆成三块后，进滤波器预算的
只有 C 块（动态模型残差 45.9 mV）。但 v1.2/v1.5 的 EKFQ 判词
（「残差 >20 mV 时不行」，奇型 ≤10–15 mV）用的是**开环**残差口径。
EKF 的 u/b 状态在线吸收一部分残差后，EKFQ 的 q 实际看到的是
**滤波后残差**（innovation）——冷温的 45.9 mV 经状态吸收后还剩多少、
什么结构，直接决定 EKFQ 在 NCA 冷温的真实可行性边界。

阶段
----
- ``probe``：逐折跑 EKF/EKFQ，滤波后残差（innovation）按 regime 与
  结构（mean/slow/odd/even/fast）分解，对照开环残差与 v1.5 结构阈值；
  EKFQ 真实复验读数（q̂ 误差、收敛率、SOC RMSE 对照 EKF）。
- ``h23``/``h24``/``h25``：判决实验（假设登记后补）。

输出：``results/metrics/ekfq_r4_{dataset}_{probe.json,folds,ekfq,structure}.*``
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

# ---- 口径常量（与脚本 26 对齐）----
LOWI_A = 1.0           # 小电流 |i| ≤ 1 A
BIGI_A = 2.0           # 大电流 |i| > 2 A
LEAD_S = 400.0         # 头段边界
REST_THR = 0.05        # 静置电流阈值（A）
BURNIN_S = 200.0       # 滤波 burn-in（状态收敛窗）
SLOW_WIN_S = 300.0     # 慢变成分滚动窗

# ---- v1.5 结构阈值（引用，不重调）----
TH_SLOW_MV = 30.0      # 慢/偶型上界（20–30 mV 带的上端）
TH_ODD_MV = 15.0       # 奇型（∝ i）上界（10–15 mV 带的上端）

CAP_SCALES = (0.95, 1.05)


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_folds(dataset: str):
    s23 = _load("s23", os.path.join(ROOT, "scripts", "23_residual_round2.py"))
    folds, best, ocv2d = s23.fold_residuals(dataset)
    return s23, folds, best, ocv2d


def _rc_from_vec(vec: np.ndarray):
    from bmsoc.ecm import RCParams, _unpack_vec
    r0k, r1, tau1, r2, tau2, r3, tau3, r0_cur, dv = _unpack_vec(vec)
    return RCParams(r0=float(np.mean(r0k)), r1=r1, tau1=tau1, r2=r2, tau2=tau2,
                    dv=dv, r3=r3, tau3=tau3, r0_cur=r0_cur), np.asarray(r0k, float)


class _R0Anchor:
    """鸭子类型 SoHSchedule：逐步 r0(soc_ref)（与开环口径的 r0k 插值一致）。"""

    def __init__(self, t_s, r0_arr, q_ah):
        self.t_s = np.asarray(t_s, float)
        self.r0_arr = np.asarray(r0_arr, float)
        self.q_ah = float(q_ah)

    def q_at(self, t):
        return np.full_like(np.asarray(t, float), self.q_ah)

    def r0_at(self, t, default=None):
        return np.interp(np.asarray(t, float), self.t_s, self.r0_arr)


def _structure_components(r_v: np.ndarray, i: np.ndarray, dt: float,
                          window_s: float = SLOW_WIN_S) -> dict:
    """滤波后残差结构分解（顺序固定，报告口径随序），返回**分量序列**：

    1. ``mean``：常值（b 状态的领地）
    2. ``slow``：滚动中位（窗 ``window_s``）− mean 的漂移
    3. ``fast = r − mean − slow``：快变
    4. ``odd``：fast 对中心化 i 的最小二乘投影；``even``：对中心化 |i|（先剥 odd）
    5. ``white``：余项

    ``r ≈ mean + slow + odd + even + white``（逐点）。另附 odd/even 斜率（mV/A）。
    """
    r = np.asarray(r_v, float)
    n = r.size
    mean = float(np.mean(r)) if n else float("nan")
    if n < 10:
        z = np.zeros(n)
        return {"mean": mean * np.ones(n), "slow": z, "odd": z, "even": z,
                "white": r - mean if n else z,
                "odd_slope_mv_a": float("nan"), "even_slope_mv_a": float("nan")}
    ser = pd.Series(r)
    w = max(3, int(window_s / max(dt, 1e-9)))
    slow_raw = ser.rolling(w, center=True, min_periods=max(3, w // 4)).median().to_numpy()
    slow_raw = np.where(np.isnan(slow_raw), mean, slow_raw)
    slow_dev = slow_raw - mean
    fast = r - slow_raw
    gi = i - np.mean(i)
    den_i = float(np.dot(gi, gi))
    beta_i = float(np.dot(gi, fast) / den_i) if den_i > 1e-12 else 0.0
    odd = beta_i * gi
    res1 = fast - odd
    ga = np.abs(i) - np.mean(np.abs(i))
    den_a = float(np.dot(ga, ga))
    beta_a = float(np.dot(ga, res1) / den_a) if den_a > 1e-12 else 0.0
    even = beta_a * ga
    white = res1 - even
    return {"mean": mean * np.ones(n), "slow": slow_dev, "odd": odd, "even": even,
            "white": white,
            "odd_slope_mv_a": float(beta_i * 1e3), "even_slope_mv_a": float(beta_a * 1e3)}


def _structure(r_v: np.ndarray, i: np.ndarray, dt: float,
               window_s: float = SLOW_WIN_S) -> dict:
    """分量 RMS 口径（见 :func:`_structure_components`）。"""
    r = np.asarray(r_v, float)
    comps = _structure_components(r, i, dt, window_s)
    out = {"mean_mv": float(np.mean(r) * 1e3) if r.size else float("nan"),
           "odd_slope_mv_a": comps["odd_slope_mv_a"],
           "even_slope_mv_a": comps["even_slope_mv_a"]}
    for k in ("slow", "odd", "even", "white"):
        out[f"{k}_mv"] = float(np.sqrt(np.mean(comps[k] ** 2)) * 1e3) if r.size \
            else float("nan")
    return out


# =====================================================================
# stage probe
# =====================================================================

def stage_probe(dataset: str) -> dict:
    from bmsoc.filters import EKF, EKFQ, FilterConfig

    s26 = _load("s26", os.path.join(ROOT, "scripts", "26_floor_round3.py"))
    s23, folds, best, ocv2d = load_folds(dataset)
    s12 = _load("s12", os.path.join(ROOT, "scripts", "12_calibration_stress.py"))
    _, profiles, _ = (s12.load_calce() if dataset == "calce"
                      else s12.load_panasonic())
    prof_of = {}
    for p in profiles:
        prof_of[(float(p.meta["temp_c"]), str(p.meta["profile"]))] = p

    fold_rows, struct_rows, ekfq_rows = [], [], []
    for key in sorted(folds):
        f = folds[key]
        temp, prof_name = f["temp_c"], f["profile"]
        p = prof_of.get(key)
        if p is None:
            continue
        params, r0k = _rc_from_vec(f["vec"])
        n = len(p)
        dt = float(p.dt)
        t = np.arange(n) * dt
        soc = np.asarray(p.soc_ref, float)
        i = np.asarray(p.current_a, float)
        v = np.asarray(p.voltage_v, float)
        r0_per = np.interp(soc, (0.1, 0.5, 0.9), r0k)
        anchor = _R0Anchor(p.t_s, r0_per, p.capacity_ah)
        e_ol = np.asarray(f["e"], float)          # v_model − v_meas（开环，含 dv）
        mask_start, mask_other, dyn_low, dyn_high, segs = s26._regime_masks(i, t)
        regimes = {"rest_start": mask_start, "rest_other": mask_other,
                   "dyn_low": dyn_low, "dyn_high": dyn_high}
        valid = t >= BURNIN_S

        for tmode in ("matched", "real"):
            T_use = np.full(n, temp) if tmode == "matched" else np.asarray(p.temp_c, float)
            from bmsoc.profiles import DrivingProfile
            pf = DrivingProfile(t_s=p.t_s, current_a=i, voltage_v=v, temp_c=T_use,
                                soc_ref=soc, capacity_ah=p.capacity_ah,
                                meta=dict(p.meta))
            cfg = FilterConfig(soc0=float(np.clip(soc[0], 0.0, 1.0)), sigma_soc0=0.05)
            est = EKF(ocv=ocv2d, params=params, capacity_ah=p.capacity_ah,
                      cfg=cfg)
            out = est.run(pf, anchor=anchor)
            inn = np.asarray(out["innovation"], float)   # v_meas − y_hat
            row = {"dataset": dataset, "temp_c": temp, "profile": prof_name,
                   "tmode": tmode, "estimator": "ekf", "cap_scale": 1.0,
                   "n": n,
                   "rmse_ol_mv": float(np.sqrt(np.mean(e_ol[valid] ** 2)) * 1e3),
                   "rmse_inn_mv": float(np.sqrt(np.mean(inn[valid] ** 2)) * 1e3),
                   "med_inn_mv": float(np.median(inn[valid]) * 1e3),
                   "absorb_frac": float(1.0 - np.sqrt(np.mean(inn[valid] ** 2))
                                       / max(np.sqrt(np.mean(e_ol[valid] ** 2)), 1e-12)),
                   "rmse_soc": float(np.sqrt(np.mean(
                       (out["soc_hat"] - soc) ** 2))),
                   "bias_end_mv": float(out["bias_hat"][-1] * 1e3),
                   }
            for rname, m in regimes.items():
                mm = m & valid
                if mm.any():
                    row[f"rmse_inn_{rname}_mv"] = float(
                        np.sqrt(np.mean(inn[mm] ** 2)) * 1e3)
                    row[f"med_inn_{rname}_mv"] = float(np.median(inn[mm]) * 1e3)
                    row[f"rmse_ol_{rname}_mv"] = float(
                        np.sqrt(np.mean(e_ol[mm] ** 2)) * 1e3)
            fold_rows.append(row)
            st = _structure(inn[valid], i[valid], dt)
            struct_rows.append({"dataset": dataset, "temp_c": temp,
                                "profile": prof_name, "tmode": tmode,
                                "estimator": "ekf", **st})

        # ---- EKFQ 真实复验（real 温度口径，q0 两个方向）----
        T_real = np.asarray(p.temp_c, float)
        pf = DrivingProfile(t_s=p.t_s, current_a=i, voltage_v=v, temp_c=T_real,
                            soc_ref=soc, capacity_ah=p.capacity_ah,
                            meta=dict(p.meta))
        cfg = FilterConfig(soc0=float(np.clip(soc[0], 0.0, 1.0)), sigma_soc0=0.05)
        ekf_out = EKF(ocv=ocv2d, params=params, capacity_ah=p.capacity_ah,
                      cfg=cfg).run(pf, anchor=anchor)
        ekf_rmse = float(np.sqrt(np.mean((ekf_out["soc_hat"] - soc) ** 2)))
        for s in CAP_SCALES:
            q0 = p.capacity_ah * s
            qest = EKFQ(ocv=ocv2d, params=params, capacity_ah=q0, cfg=cfg)
            qout = qest.run(pf, anchor=anchor)
            qh_end = float(np.median(qout["q_hat"][-max(1, n // 10):]))
            inn_q = np.asarray(qout["innovation"], float)
            qst = _structure(inn_q[valid], i[valid], dt)
            ekfq_rows.append({
                "dataset": dataset, "temp_c": temp, "profile": prof_name,
                "cap_scale": s, "n": n,
                "rmse_soc": float(np.sqrt(np.mean((qout["soc_hat"] - soc) ** 2))),
                "rmse_ekf": ekf_rmse,
                "q_rel_err": float(qh_end / p.capacity_ah - 1.0),
                "q_ok": bool(abs(qh_end / p.capacity_ah - 1.0) < 0.02),
                "rmse_inn_mv": float(np.sqrt(np.mean(inn_q[valid] ** 2)) * 1e3),
                "odd_mv": qst["odd_mv"], "slow_mv": qst["slow_mv"],
                "even_mv": qst["even_mv"], "mean_mv": qst["mean_mv"],
            })
        print(f"[{temp:5.0f}] {prof_name:<5s} ol {row['rmse_ol_mv']:5.1f} → "
              f"inn {row['rmse_inn_mv']:5.1f} mV 吸收 {row['absorb_frac']:.0%} 完成")

    fdf = pd.DataFrame(fold_rows)
    sdf = pd.DataFrame(struct_rows)
    qdf = pd.DataFrame(ekfq_rows)
    if fdf.empty:
        raise SystemExit("probe 空结果")
    fdf.to_parquet(os.path.join(MET, f"ekfq_r4_{dataset}_folds.parquet"), index=False)
    sdf.to_parquet(os.path.join(MET, f"ekfq_r4_{dataset}_structure.parquet"), index=False)
    qdf.to_parquet(os.path.join(MET, f"ekfq_r4_{dataset}_ekfq.parquet"), index=False)

    per_temp = {}
    for temp in sorted(fdf.temp_c.unique()):
        for tmode in ("matched", "real"):
            sub = fdf[(fdf.temp_c == temp) & (fdf.tmode == tmode)]
            ss = sdf[(sdf.temp_c == temp) & (sdf.tmode == tmode)]
            per_temp[f"{temp:g}_{tmode}"] = {
                "rmse_ol_mv": float(sub.rmse_ol_mv.median()),
                "rmse_inn_mv": float(sub.rmse_inn_mv.median()),
                "absorb_frac": float(sub.absorb_frac.median()),
                "med_inn_mv": float(sub.med_inn_mv.median()),
                "rmse_inn_rest_start_mv": float(sub.rmse_inn_rest_start_mv.median()),
                "rmse_inn_rest_other_mv": float(sub.rmse_inn_rest_other_mv.median()),
                "rmse_inn_dyn_low_mv": float(sub.rmse_inn_dyn_low_mv.median()),
                "rmse_inn_dyn_high_mv": float(sub.rmse_inn_dyn_high_mv.median()),
                "rmse_ol_dyn_low_mv": float(sub.rmse_ol_dyn_low_mv.median()),
                "mean_mv": float(ss.mean_mv.median()),
                "slow_mv": float(ss.slow_mv.median()),
                "odd_mv": float(ss.odd_mv.median()),
                "even_mv": float(ss.even_mv.median()),
                "white_mv": float(ss.white_mv.median()),
                "odd_slope_mv_a": float(ss.odd_slope_mv_a.median()),
                "rmse_soc": float(sub.rmse_soc.median()),
            }
    ekfq_sum = {}
    for temp in sorted(qdf.temp_c.unique()):
        for s in CAP_SCALES:
            sub = qdf[(qdf.temp_c == temp) & (qdf.cap_scale == s)]
            ekfq_sum[f"{temp:g}_x{s:g}"] = {
                "rmse_soc_med": float(sub.rmse_soc.median()),
                "rmse_ekf_med": float(sub.rmse_ekf.median()),
                "q_ok_rate": float(sub.q_ok.mean()),
                "q_rel_err_med": float(sub.q_rel_err.median()),
                "inn_mv_med": float(sub.rmse_inn_mv.median()),
                "odd_mv_med": float(sub.odd_mv.median()),
            }
    out = {
        "dataset": dataset, "best_variant": best, "n_folds": int(len(fdf)) // 2,
        "burnin_s": BURNIN_S,
        "th_slow_mv": TH_SLOW_MV, "th_odd_mv": TH_ODD_MV,
        "per_temp": per_temp, "ekfq": ekfq_sum,
    }
    with open(os.path.join(MET, f"ekfq_r4_{dataset}_probe.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("=== probe per_temp（matched/real）===")
    for k, v in per_temp.items():
        print(f"  {k:<16s} ol {v['rmse_ol_mv']:5.1f} → inn {v['rmse_inn_mv']:5.1f} mV "
              f"({v['absorb_frac']:+.0%})  mean {v['mean_mv']:+6.1f} "
              f"slow {v['slow_mv']:5.1f} odd {v['odd_mv']:5.1f} even {v['even_mv']:5.1f}")
    print("=== ekfq 真实复验（median）===")
    for k, v in ekfq_sum.items():
        print(f"  {k:<12s} EKFQ {v['rmse_soc_med']*100:5.2f} pp vs "
              f"EKF {v['rmse_ekf_med']*100:5.2f} pp  q收敛 {v['q_ok_rate']:.0%}  "
              f"q̂ {v['q_rel_err_med']:+.2%}")
    return out


# =====================================================================
# stage h23 —— 判决线见工作记录 §44.2（只读，不调）
# =====================================================================

H23_L_SLOW_MV = 30.0     # 全温度 slow/even 上界（v1.5 慢/偶上界，引用）
H23_L_ABSORB = 0.25      # −20 °C 状态吸收份额下限


def stage_h23(dataset: str) -> dict:
    probe_path = os.path.join(MET, f"ekfq_r4_{dataset}_probe.json")
    if not os.path.exists(probe_path):
        raise SystemExit("缺 probe 产物 —— 先跑 --stage probe")
    with open(probe_path, encoding="utf-8") as fh:
        probe = json.load(fh)
    per_temp = probe["per_temp"]
    rows = []
    for key, v in per_temp.items():
        if not key.endswith("_matched"):
            continue
        rows.append({"temp": key, "slow_mv": v["slow_mv"], "even_mv": v["even_mv"],
                     "absorb_frac": v["absorb_frac"],
                     "ok_slow": bool(v["slow_mv"] <= H23_L_SLOW_MV),
                     "ok_even": bool(v["even_mv"] <= H23_L_SLOW_MV)})
    l1 = bool(all(r["ok_slow"] and r["ok_even"] for r in rows))
    absorb_cold = per_temp.get("-20_matched", {}).get("absorb_frac", float("nan"))
    l2 = bool(absorb_cold >= H23_L_ABSORB)
    if l1 and l2:
        verdict = ("成立：滤波后 slow/even 全温度过 v1.5 上界（slow ≤ "
                   f"{max(r['slow_mv'] for r in rows):.1f} mV），−20 °C 吸收份额 "
                   f"{absorb_cold:.0%} —— v1.2「慢变残差 >20–30 mV 不可行」的前提"
                   "不适用于滤波口径；EKFQ 失败另有通道（H24/H25）")
    else:
        verdict = f"被否：L1_slow/even {l1} / L2_absorb {l2}（见逐温度表）"
    out = {"dataset": dataset,
           "lines": {"L1_slow_mv": H23_L_SLOW_MV, "L2_absorb": H23_L_ABSORB},
           "per_temp": rows, "checks": {"L1": l1, "L2": l2}, "verdict": verdict}
    with open(os.path.join(MET, f"ekfq_r4_{dataset}_h23.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"=== h23 {dataset}：{verdict}")
    for r in rows:
        print(f"  {r['temp']:<16s} slow {r['slow_mv']:5.2f} even {r['even_mv']:5.2f} "
              f"absorb {r['absorb_frac']:.0%}")
    return out


# =====================================================================
# stage h24/h25 —— 选择性注入台架（判决线见 §44.2）
# =====================================================================

INJECT_SET = ["none", "full", "odd_only", "white_only", "minus_odd",
              "minus_white", "slow_only"]
REPLAY_FOLDS = [(-20.0, "hwfet"), (-20.0, "la92"), (25.0, "nn"), (25.0, "udds")]
REPLAY_SCALES = [0.5, 1.0, 1.5]
H24_L_SUFF = 0.30        # 充分性：只注该分量 ⇒ 收敛率 ≤ 30%
H24_L_NECC = 0.70        # 必要性：剥该分量 ⇒ 收敛率 ≥ 70%


def _run_replay(dataset: str) -> pd.DataFrame:
    """零残差合成植物 + 分量选择性注入 → EKFQ 收敛率（H24/H25 共用台架）。"""
    from bmsoc.ecm import open_loop_from_truth
    from bmsoc.filters import EKF, EKFQ, FilterConfig
    from bmsoc.profiles import DrivingProfile

    s26 = _load("s26", os.path.join(ROOT, "scripts", "26_floor_round3.py"))
    s23, folds, best, ocv2d = load_folds(dataset)
    s12 = _load("s12", os.path.join(ROOT, "scripts", "12_calibration_stress.py"))
    _, profiles, _ = (s12.load_calce() if dataset == "calce"
                      else s12.load_panasonic())
    prof_of = {(float(p.meta["temp_c"]), str(p.meta["profile"])): p for p in profiles}

    rows = []
    for temp, prof_name in REPLAY_FOLDS:
        key = (temp, prof_name)
        f = folds.get(key)
        p = prof_of.get(key)
        if f is None or p is None:
            print(f"[skip] {key} 不在 folds/profiles")
            continue
        params, r0k = _rc_from_vec(f["vec"])
        n = len(p)
        dt = float(p.dt)
        t = np.arange(n) * dt
        soc = np.asarray(p.soc_ref, float)
        i = np.asarray(p.current_a, float)
        v_clean = open_loop_from_truth(f["vec"], soc, i, ocv2d, dt, temp)
        r0_per = np.interp(soc, (0.1, 0.5, 0.9), r0k)
        anchor = _R0Anchor(p.t_s, r0_per, p.capacity_ah)
        valid = t >= BURNIN_S

        # 真实折的 EKF innovation = 注入源
        T_use = np.full(n, temp)
        pf_meas = DrivingProfile(t_s=p.t_s, current_a=i, voltage_v=np.asarray(p.voltage_v, float),
                                 temp_c=T_use, soc_ref=soc, capacity_ah=p.capacity_ah,
                                 meta=dict(p.meta))
        cfg = FilterConfig(soc0=float(np.clip(soc[0], 0.0, 1.0)), sigma_soc0=0.05)
        ekf_out = EKF(ocv=ocv2d, params=params, capacity_ah=p.capacity_ah,
                      cfg=cfg).run(pf_meas, anchor=anchor)
        inn = np.asarray(ekf_out["innovation"], float)
        comps = _structure_components(inn, i, dt)
        inject_of = {
            "none": np.zeros(n),
            "full": inn - np.mean(inn[valid]) if valid.any() else inn,
            "odd_only": comps["odd"],
            "white_only": comps["white"],
            "minus_odd": inn - comps["odd"] - np.mean(inn[valid]) if valid.any() else inn - comps["odd"],
            "minus_white": inn - comps["white"] - np.mean(inn[valid]) if valid.any() else inn - comps["white"],
            "slow_only": comps["slow"],
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
        print(f"[{temp:5.0f}] {prof_name:<5s} replay 台架完成", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("replay 空结果")
    df.to_parquet(os.path.join(MET, f"ekfq_r4_{dataset}_replay.parquet"), index=False)
    return df


def _replay_summary(df: pd.DataFrame) -> dict:
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


def stage_h24(dataset: str) -> dict:
    rp = os.path.join(MET, f"ekfq_r4_{dataset}_replay.parquet")
    df = pd.read_parquet(rp) if os.path.exists(rp) else _run_replay(dataset)
    tbl = _replay_summary(df)
    odd_suff = tbl["odd_only"]["q_ok_rate"] <= H24_L_SUFF
    minus_odd_necc = tbl["minus_odd"]["q_ok_rate"] >= H24_L_NECC
    slow_ok = tbl["slow_only"]["q_ok_rate"] >= H24_L_NECC
    if odd_suff and minus_odd_necc:
        verdict = ("成立：奇型是 q 中毒主通道（只注奇型收敛 "
                   f"{tbl['odd_only']['q_ok_rate']:.0%} ≤30%，剥奇型 "
                   f"{tbl['minus_odd']['q_ok_rate']:.0%} ≥70%）")
    elif odd_suff and not minus_odd_necc:
        verdict = (f"部分成立：奇型充分（只注收敛 {tbl['odd_only']['q_ok_rate']:.0%}）"
                   f"但非必要（剥奇型仅 {tbl['minus_odd']['q_ok_rate']:.0%}）——通道之一")
    else:
        verdict = (f"被否：只注奇型收敛率 {tbl['odd_only']['q_ok_rate']:.0%} > 30%"
                   "——奇型不是充分毒源")
    out = {"dataset": dataset,
           "lines": {"L1_suff": H24_L_SUFF, "L2_necc": H24_L_NECC},
           "table": tbl,
           "checks": {"L1_odd_suff": bool(odd_suff), "L2_minus_odd_necc": bool(minus_odd_necc),
                      "L3_slow_innocent": bool(slow_ok)},
           "verdict": verdict}
    with open(os.path.join(MET, f"ekfq_r4_{dataset}_h24.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"=== h24 {dataset}：{verdict}")
    for inj in INJECT_SET:
        v = tbl[inj]
        print(f"  {inj:<11s} q收敛 {v['q_ok_rate']:.0%} "
              f"(×0.5 {v['q_ok_rate_x0.5']:.0%} / ×1.5 {v['q_ok_rate_x1.5']:.0%})  "
              f"q̂ {v['q_rel_err_med']:+.2%}  soc {v['rmse_soc_med']*100:.2f} pp")
    return out


def stage_h25(dataset: str) -> dict:
    rp = os.path.join(MET, f"ekfq_r4_{dataset}_replay.parquet")
    df = pd.read_parquet(rp) if os.path.exists(rp) else _run_replay(dataset)
    tbl = _replay_summary(df)
    wh_suff = tbl["white_only"]["q_ok_rate"] <= H24_L_SUFF
    minus_wh_necc = tbl["minus_white"]["q_ok_rate"] >= H24_L_NECC
    if wh_suff and minus_wh_necc:
        verdict = ("成立：white（快变非线性）是 q 中毒主通道（只注 white 收敛 "
                   f"{tbl['white_only']['q_ok_rate']:.0%} ≤30%，剥 white "
                   f"{tbl['minus_white']['q_ok_rate']:.0%} ≥70%）")
    elif wh_suff and not minus_wh_necc:
        verdict = (f"部分成立：white 充分（只注收敛 {tbl['white_only']['q_ok_rate']:.0%}）"
                   f"但非必要（剥 white 仅 {tbl['minus_white']['q_ok_rate']:.0%}）")
    else:
        verdict = (f"被否：只注 white 收敛率 {tbl['white_only']['q_ok_rate']:.0%} > 30%"
                   "——white 不是充分毒源（若 H24 亦被否 ⇒ 毒源在闭环放大/未分解结构）")
    out = {"dataset": dataset,
           "lines": {"L1_suff": H24_L_SUFF, "L2_necc": H24_L_NECC},
           "table": tbl,
           "checks": {"L1_white_suff": bool(wh_suff), "L2_minus_white_necc": bool(minus_wh_necc)},
           "verdict": verdict}
    with open(os.path.join(MET, f"ekfq_r4_{dataset}_h25.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"=== h25 {dataset}：{verdict}")
    for inj in INJECT_SET:
        v = tbl[inj]
        print(f"  {inj:<11s} q收敛 {v['q_ok_rate']:.0%}  q̂ {v['q_rel_err_med']:+.2%}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["probe", "h23", "h24", "h25"], required=True)
    ap.add_argument("--dataset", choices=["calce", "panasonic"], default="panasonic")
    args = ap.parse_args()
    fn = {"probe": stage_probe, "h23": stage_h23, "h24": stage_h24,
          "h25": stage_h25}[args.stage]
    fn(args.dataset)


if __name__ == "__main__":
    main()
