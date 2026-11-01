#!/usr/bin/env python
"""34_robust_round7.py — 第十一轮：鲁棒 EKFQ 对 white 重尾的对症测试（v1.10）。

问题（记录 §49.5 首选）
---------------------
第十轮证明 white 的主体不是缺失快 RC（模型级削减 0%），且 fast_only 只灌
q̂ +3.4%（full +35.5%）——white/odd 的其余成分仍在灌 q。第九轮已知 white
**重尾**（kurt 7–30，didt 集中、记忆 +7.7~+30.3）。毒既不可共享建模又是
充分毒，滤波器侧 innovation 鲁棒化（Huber/限幅/门控）是直接对症。

阶段
----
- ``probe``：真实折 innovation 尾部统计（kurt/超限占比/尾部能量份额）；
  鲁棒变体（none/clip/huber/gate × κ∈{2,3,5}）在 full replay 上的扫描；
  真实数据 EKFQ 基线 vs 鲁棒的 q̂/SOC 对照。
- ``h32``/``h33``/``h34``：判决实验（登记线见工作记录 §50.2，实现只读常量）。

输出：``results/metrics/robust_r7_{dataset}_{probe.json,replay.parquet,tails.parquet,
real.parquet,h32.json,h33.json,h34.json}``
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

# ---- 口径常量（与脚本 28 对齐）----
BURNIN_S = 200.0
SLOW_WIN_S = 300.0
CAP_SCALES = (0.95, 1.05)
REPLAY_FOLDS = [(-20.0, "hwfet"), (-20.0, "la92"), (25.0, "nn"), (25.0, "udds")]

# ---- 鲁棒变体（预登记网格；判决线见 §50.2）----
FILTER_VARIANTS = {
    "base": dict(innov_robust="none"),
    "clip2": dict(innov_robust="clip", innov_kappa=2.0),
    "clip3": dict(innov_robust="clip", innov_kappa=3.0),
    "clip5": dict(innov_robust="clip", innov_kappa=5.0),
    "huber2": dict(innov_robust="huber", innov_kappa=2.0),
    "huber3": dict(innov_robust="huber", innov_kappa=3.0),
    "huber5": dict(innov_robust="huber", innov_kappa=5.0),
    "gate3": dict(innov_robust="gate", innov_kappa=3.0),
}
ROBUST_SET = [k for k in FILTER_VARIANTS if k != "base"]

# ---- 判决线（§50.2 事前登记；只读，不回溯修改）----
H32_L1_DROP = 0.50       # full replay |q̂| 中位相对下降下限
H32_L2_QOK = 0.30        # q_ok 收敛率下限（回灌 H30 口径）
H32_L3_CLEAN = 0.90      # 零残差对照收敛率下限（护栏）
H32_L3_SOC = 0.7         # full SOC RMSE 中位 / 基线 上限（护栏，须改善）

H33_L1_WINSOR = 0.20     # 剥 white 5% 重尾后 |q̂| 相对下降下限
H33_L2_GAUSS = 0.40      # 高斯化 white 后 |q̂| 相对下降下限

H34_L1_QREL = 0.20       # 真实数据 |q̂| 中位相对下降下限
H34_L2_SOC = 0.10        # 真实数据 SOC RMSE 中位相对恶化上限
H34_L2_SOC_MAX = 0.25    # 单折最大恶化上限


def _load(name: str, path: str):
    key = f"_mod_{name}"
    if key in globals():
        return globals()[key]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    globals()[key] = mod
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
    def __init__(self, t_s, r0_arr, q_ah):
        self.t_s = np.asarray(t_s, float)
        self.r0_arr = np.asarray(r0_arr, float)
        self.q_ah = float(q_ah)

    def q_at(self, t):
        return np.full_like(np.asarray(t, float), self.q_ah)

    def r0_at(self, t, default=None):
        return np.interp(np.asarray(t, float), self.t_s, self.r0_arr)


def _variant_cfg(name: str, soc0: float):
    from bmsoc.filters import FilterConfig
    kw = dict(FILTER_VARIANTS[name])
    return FilterConfig(soc0=float(np.clip(soc0, 0.0, 1.0)), sigma_soc0=0.05, **kw)


def _winsor(x: np.ndarray, frac: float = 0.05) -> np.ndarray:
    """对 |x| 尾部 winsorize（保留 RMS 的缩放在调用侧做）。"""
    x = np.asarray(x, float)
    if x.size == 0:
        return x.copy()
    lim = float(np.quantile(np.abs(x), 1.0 - frac))
    return np.clip(x, -lim, lim)


def _rescale_to_std(x: np.ndarray, std_target: float) -> np.ndarray:
    s = float(np.std(x))
    if s < 1e-15:
        return x.copy()
    return x * (std_target / s)


# =====================================================================
# stage probe
# =====================================================================

def _tail_stats(e: np.ndarray) -> dict:
    e = np.asarray(e, float)
    if e.size < 10:
        return {"kurt": float("nan"), "frac_2s": float("nan"), "frac_3s": float("nan"),
                "frac_5s": float("nan"), "tail_energy_share": float("nan")}
    mu = float(np.mean(e))
    sd = float(np.std(e))
    z = np.abs(e - mu) / max(sd, 1e-15)
    e2 = (e - mu) ** 2
    tot = float(np.sum(e2))
    tail = z >= 3.0
    return {
        "kurt": float(pd.Series(e).kurt()),
        "frac_2s": float(np.mean(z >= 2.0)),
        "frac_3s": float(np.mean(z >= 3.0)),
        "frac_5s": float(np.mean(z >= 5.0)),
        "tail_energy_share": float(np.sum(e2[tail]) / tot) if tot > 0 else float("nan"),
    }


def _run_real_ekf_innovations(dataset: str) -> pd.DataFrame:
    """真实折 EKF 滤波后残差逐样本（尾部统计与注入源共用）。"""
    from bmsoc.filters import EKF, FilterConfig
    from bmsoc.profiles import DrivingProfile

    s23, folds, best, ocv2d = load_folds(dataset)
    s12 = _load("s12", os.path.join(ROOT, "scripts", "12_calibration_stress.py"))
    _, profiles, _ = (s12.load_calce() if dataset == "calce"
                      else s12.load_panasonic())
    prof_of = {(float(p.meta["temp_c"]), str(p.meta["profile"])): p for p in profiles}

    frames = []
    for key in sorted(folds):
        f = folds[key]
        p = prof_of.get(key)
        if p is None:
            continue
        temp = f["temp_c"]
        params, r0k = _rc_from_vec(f["vec"])
        n = len(p)
        dt = float(p.dt)
        t = np.arange(n) * dt
        soc = np.asarray(p.soc_ref, float)
        i = np.asarray(p.current_a, float)
        v = np.asarray(p.voltage_v, float)
        r0_per = np.interp(soc, (0.1, 0.5, 0.9), r0k)
        anchor = _R0Anchor(p.t_s, r0_per, p.capacity_ah)
        T_use = np.full(n, temp)
        pf = DrivingProfile(t_s=p.t_s, current_a=i, voltage_v=v, temp_c=T_use,
                            soc_ref=soc, capacity_ah=p.capacity_ah, meta=dict(p.meta))
        cfg = FilterConfig(soc0=float(np.clip(soc[0], 0.0, 1.0)), sigma_soc0=0.05)
        out = EKF(ocv=ocv2d, params=params, capacity_ah=p.capacity_ah,
                  cfg=cfg).run(pf, anchor=anchor)
        inn = np.asarray(out["innovation"], float)
        valid = t >= BURNIN_S
        frames.append(pd.DataFrame({
            "dataset": dataset, "temp_c": temp, "profile": f["profile"],
            "t_s": t, "inn_mv": inn * 1e3, "current_a": i,
            "valid": valid,
        }))
        print(f"[inn] {temp:5.0f} {f['profile']:<5s} n={n}", flush=True)
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise SystemExit("innovation 空结果")
    return df


def stage_probe(dataset: str) -> dict:
    from bmsoc.ecm import open_loop_from_truth
    from bmsoc.filters import EKFQ
    from bmsoc.profiles import DrivingProfile

    s28 = _load("s28", os.path.join(ROOT, "scripts", "28_ekfq_round4.py"))
    s23, folds, best, ocv2d = load_folds(dataset)
    s12 = _load("s12", os.path.join(ROOT, "scripts", "12_calibration_stress.py"))
    _, profiles, _ = (s12.load_calce() if dataset == "calce"
                      else s12.load_panasonic())
    prof_of = {(float(p.meta["temp_c"]), str(p.meta["profile"])): p for p in profiles}

    # ---- A) 尾部统计 ----
    idf = _run_real_ekf_innovations(dataset)
    idf.to_parquet(os.path.join(MET, f"robust_r7_{dataset}_innovation.parquet"),
                   index=False)
    tail_rows = []
    for (temp, prof), g in idf.groupby(["temp_c", "profile"]):
        gv = g[g.valid]
        st = _tail_stats(gv["inn_mv"].to_numpy(float) * 1e-3)
        white = s28._structure_components(
            gv["inn_mv"].to_numpy(float) * 1e-3,
            gv["current_a"].to_numpy(float), 1.0)["white"]
        stw = _tail_stats(white)
        tail_rows.append({"dataset": dataset, "temp_c": temp, "profile": prof,
                          **{f"inn_{k}": v for k, v in st.items()},
                          **{f"white_{k}": v for k, v in stw.items()}})
    tdf = pd.DataFrame(tail_rows)
    tdf.to_parquet(os.path.join(MET, f"robust_r7_{dataset}_tails.parquet"), index=False)

    # ---- B) replay 网格（含机制对照）----
    rp = os.path.join(MET, f"robust_r7_{dataset}_replay.parquet")
    rdf = _run_replay(dataset, idf)
    rdf.to_parquet(rp, index=False)

    # ---- C) 真实数据 EKFQ 基线 vs 鲁棒 ----
    real_rows = []
    for key in sorted(folds):
        f = folds[key]
        p = prof_of.get(key)
        if p is None:
            continue
        temp, prof_name = f["temp_c"], f["profile"]
        params, r0k = _rc_from_vec(f["vec"])
        n = len(p)
        dt = float(p.dt)
        t = np.arange(n) * dt
        soc = np.asarray(p.soc_ref, float)
        i = np.asarray(p.current_a, float)
        v = np.asarray(p.voltage_v, float)
        r0_per = np.interp(soc, (0.1, 0.5, 0.9), r0k)
        anchor = _R0Anchor(p.t_s, r0_per, p.capacity_ah)
        T_use = np.full(n, temp)
        pf = DrivingProfile(t_s=p.t_s, current_a=i, voltage_v=v, temp_c=T_use,
                            soc_ref=soc, capacity_ah=p.capacity_ah, meta=dict(p.meta))
        for vname in ("base", "clip3", "huber3"):
            for s in CAP_SCALES:
                cfg = _variant_cfg(vname, float(soc[0]))
                out = EKFQ(ocv=ocv2d, params=params, capacity_ah=p.capacity_ah * s,
                           cfg=cfg).run(pf, anchor=anchor)
                qh = float(np.median(out["q_hat"][-max(1, n // 10):]))
                real_rows.append({
                    "dataset": dataset, "temp_c": temp, "profile": prof_name,
                    "variant": vname, "cap_scale": s,
                    "q_rel_err": qh / p.capacity_ah - 1.0,
                    "q_ok": bool(abs(qh / p.capacity_ah - 1.0) < 0.02),
                    "rmse_soc": float(np.sqrt(np.mean((out["soc_hat"] - soc) ** 2))),
                    "rmse_inn_mv": float(np.sqrt(np.mean(
                        np.asarray(out["innovation"])[t >= BURNIN_S] ** 2)) * 1e3),
                    "robust_frac": float(np.mean(np.asarray(out["robust_w"])[t >= BURNIN_S] < 1.0)),
                })
        print(f"[real] {temp:5.0f} {prof_name:<5s} 完成", flush=True)
    real_df = pd.DataFrame(real_rows)
    if real_df.empty:
        raise SystemExit("real 空结果")
    real_df.to_parquet(os.path.join(MET, f"robust_r7_{dataset}_real.parquet"), index=False)

    # ---- 汇总 ----
    sweep = _replay_summary(rdf, inject="full", scale=1.0)
    clean = _replay_summary(rdf, inject="none", scale=1.0)
    per_temp_tail = {}
    for temp, g in tdf.groupby("temp_c"):
        per_temp_tail[f"{temp:g}"] = {
            "inn_kurt": float(g.inn_kurt.median()),
            "inn_frac_3s": float(g.inn_frac_3s.median()),
            "inn_tail_energy_share": float(g.inn_tail_energy_share.median()),
            "white_kurt": float(g.white_kurt.median()),
            "white_frac_3s": float(g.white_frac_3s.median()),
            "white_tail_energy_share": float(g.white_tail_energy_share.median()),
        }
    real_sum = {}
    for vname, g in real_df.groupby("variant"):
        real_sum[vname] = {
            "q_rel_abs_med": float(np.abs(g.q_rel_err).median()),
            "q_ok_rate": float(g.q_ok.mean()),
            "rmse_soc_med": float(g.rmse_soc.median()),
            "robust_frac_med": float(g.robust_frac.median()),
        }
    out = {
        "dataset": dataset, "n_folds": int(len(tdf)),
        "tail_per_temp": per_temp_tail,
        "sweep_full": sweep, "sweep_clean": clean,
        "real": real_sum,
    }
    with open(os.path.join(MET, f"robust_r7_{dataset}_probe.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("=== probe 尾部（inn / white）===")
    for k, v in per_temp_tail.items():
        print(f"  {k:>5s}°C  kurt {v['inn_kurt']:5.1f}/{v['white_kurt']:5.1f}  "
              f">3σ {v['inn_frac_3s']:.3%}/{v['white_frac_3s']:.3%}  "
              f"尾能 {v['inn_tail_energy_share']:.1%}/{v['white_tail_energy_share']:.1%}")
    print("=== probe full replay 扫描（q_ok）===")
    for vname in FILTER_VARIANTS:
        v = sweep.get(vname, {})
        c = clean.get(vname, {})
        print(f"  {vname:<7s} full {v.get('q_ok_rate', float('nan')):.0%}  "
              f"clean {c.get('q_ok_rate', float('nan')):.0%}  "
              f"q̂ {v.get('q_rel_err_med', float('nan')):+.2%}  "
              f"soc {v.get('rmse_soc_med', float('nan'))*100:.2f}pp")
    print("=== probe 真实数据 ===")
    for vname, v in real_sum.items():
        print(f"  {vname:<7s} |q̂| {v['q_rel_abs_med']:+.2%}  q_ok {v['q_ok_rate']:.0%}  "
              f"soc {v['rmse_soc_med']*100:.2f}pp  鲁棒触发 {v['robust_frac_med']:.1%}")
    return out


# =====================================================================
# replay 台架（注入源 = 真实折 EKF innovation）
# =====================================================================

def _build_injects(inn: np.ndarray, i: np.ndarray, dt: float) -> dict:
    """机制对照注入组。

    - ``full``        ：真实滤波后残差去均值（重尾结构完整）
    - ``full_winsor5``：仅 white 分量剥 5% 重尾并回配 RMS（形状对照）
    - ``full_gauss``  ：white 换成同方差高斯（轻尾对照）
    """
    s28 = _load("s28", os.path.join(ROOT, "scripts", "28_ekfq_round4.py"))
    n = inn.size
    comps = s28._structure_components(inn, i, dt)
    mean = float(np.mean(inn))
    full = inn - mean
    white = comps["white"]
    slow_oe = comps["slow"] + comps["odd"] + comps["even"]
    w_std = float(np.std(white))
    w_win = _winsor(white, 0.05)
    w_win = _rescale_to_std(w_win, w_std)
    rng = np.random.default_rng(20261103)
    w_gauss = rng.normal(0.0, w_std, n) if w_std > 0 else np.zeros(n)
    return {
        "none": np.zeros(n),
        "full": full,
        "full_winsor5": slow_oe + w_win,
        "full_gauss": slow_oe + w_gauss,
    }


def _replay_specs() -> list:
    """(inject, variant, scale) 网格：扫描 + 机制 + 剂量。"""
    specs = []
    for inj in ("none", "full"):
        for v in FILTER_VARIANTS:
            specs.append((inj, v, 1.0))
    for inj in ("full_winsor5", "full_gauss"):
        for v in ("base", "clip3", "huber3"):
            specs.append((inj, v, 1.0))
    for v in ("base", "clip3", "huber3"):
        for sc in (0.5, 1.5):
            specs.append(("full", v, sc))
    return specs


def _run_replay(dataset: str, idf: pd.DataFrame) -> pd.DataFrame:
    from bmsoc.ecm import open_loop_from_truth
    from bmsoc.filters import EKFQ
    from bmsoc.profiles import DrivingProfile

    s28 = _load("s28", os.path.join(ROOT, "scripts", "28_ekfq_round4.py"))
    s23, folds, best, ocv2d = load_folds(dataset)
    s12 = _load("s12", os.path.join(ROOT, "scripts", "12_calibration_stress.py"))
    _, profiles, _ = (s12.load_calce() if dataset == "calce"
                      else s12.load_panasonic())
    prof_of = {(float(p.meta["temp_c"]), str(p.meta["profile"])): p for p in profiles}

    specs = _replay_specs()
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
        inn = g["inn_mv"].to_numpy(float) * 1e-3
        n = len(p)
        dt = float(p.dt)
        t = np.arange(n) * dt
        soc = np.asarray(p.soc_ref, float)
        i = np.asarray(p.current_a, float)
        v_clean = open_loop_from_truth(f["vec"], soc, i, ocv2d, dt, temp)
        params, r0k = _rc_from_vec(f["vec"])
        r0_per = np.interp(soc, (0.1, 0.5, 0.9), r0k)
        anchor = _R0Anchor(p.t_s, r0_per, p.capacity_ah)
        T_use = np.full(n, temp)
        inject_of = _build_injects(inn, i, dt)

        for inj, vname, scale in specs:
            sig = inject_of[inj]
            v = v_clean + scale * sig
            pf = DrivingProfile(t_s=p.t_s, current_a=i, voltage_v=v, temp_c=T_use,
                                soc_ref=soc, capacity_ah=p.capacity_ah,
                                meta=dict(p.meta))
            for s in CAP_SCALES:
                cfg = _variant_cfg(vname, float(soc[0]))
                out = EKFQ(ocv=ocv2d, params=params, capacity_ah=p.capacity_ah * s,
                           cfg=cfg).run(pf, anchor=anchor)
                qh = float(np.median(out["q_hat"][-max(1, n // 10):]))
                q_rel = qh / p.capacity_ah - 1.0
                rows.append({
                    "dataset": dataset, "temp_c": temp, "profile": prof_name,
                    "inject": inj, "variant": vname, "scale": scale,
                    "cap_scale": s,
                    "q_rel_err": float(q_rel),
                    "q_ok": bool(abs(q_rel) < 0.02),
                    "rmse_soc": float(np.sqrt(np.mean((out["soc_hat"] - soc) ** 2))),
                })
        print(f"[{temp:5.0f}] {prof_name:<5s} replay {len(specs)} 规格完成", flush=True)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("replay 空结果")
    return df


def _replay_summary(df: pd.DataFrame, inject: str, scale: float = 1.0) -> dict:
    tbl = {}
    sub = df[(df.inject == inject) & (df.scale == scale)]
    for vname, g in sub.groupby("variant"):
        tbl[vname] = {
            "q_ok_rate": float(g.q_ok.mean()),
            "q_rel_err_med": float(g.q_rel_err.median()),
            "q_rel_abs_med": float(np.abs(g.q_rel_err).median()),
            "rmse_soc_med": float(g.rmse_soc.median()),
            "n": int(len(g)),
        }
    return tbl


def _cell(df: pd.DataFrame, inject: str, variant: str, scale: float = 1.0) -> dict:
    g = df[(df.inject == inject) & (df.variant == variant) & (df.scale == scale)]
    if g.empty:
        return {"q_ok_rate": float("nan"), "q_rel_err_med": float("nan"),
                "rmse_soc_med": float("nan"), "n": 0}
    return {"q_ok_rate": float(g.q_ok.mean()), "q_rel_err_med": float(g.q_rel_err.median()),
            "q_rel_abs_med": float(np.abs(g.q_rel_err).median()),
            "rmse_soc_med": float(g.rmse_soc.median()), "n": int(len(g))}


# =====================================================================
# 判决阶段
# =====================================================================

def stage_h32(dataset: str) -> dict:
    """H32：鲁棒 innovation 更新缓解 full replay 的 q 中毒（幅度+收敛口径）。"""
    rp = os.path.join(MET, f"robust_r7_{dataset}_replay.parquet")
    if not os.path.exists(rp):
        raise SystemExit("缺 replay 产物 —— 先跑 --stage probe")
    df = pd.read_parquet(rp)
    base_full = _cell(df, "full", "base")
    base_clean = _cell(df, "none", "base")

    # 预登记选择规则：L1 幅度下降最大 → q_ok 高 → SOC 低
    cands = []
    for vname in ROBUST_SET:
        full = _cell(df, "full", vname)
        clean = _cell(df, "none", vname)
        drop = ((base_full["q_rel_abs_med"] - full["q_rel_abs_med"])
                / max(base_full["q_rel_abs_med"], 1e-9))
        soc_ratio = (full["rmse_soc_med"] / base_full["rmse_soc_med"]
                     if base_full["rmse_soc_med"] > 0 else float("inf"))
        cands.append((vname, full, clean, float(drop), float(soc_ratio)))
    cands.sort(key=lambda z: (-z[3], -z[1]["q_ok_rate"], z[1]["rmse_soc_med"]))
    best_name, best_full, best_clean, best_drop, soc_ratio = cands[0]

    l1 = bool(best_drop >= H32_L1_DROP)
    l2 = bool(best_full["q_ok_rate"] >= H32_L2_QOK)
    l3 = bool(best_clean["q_ok_rate"] >= H32_L3_CLEAN
              and soc_ratio <= H32_L3_SOC)

    if l1 and l2 and l3:
        verdict = (f"成立：{best_name} |q̂| 中位相对下降 {best_drop:.0%} "
                   f"≥{H32_L1_DROP:.0%}，q_ok {best_full['q_ok_rate']:.0%} "
                   f"≥{H32_L2_QOK:.0%}，clean {best_clean['q_ok_rate']:.0%}、"
                   f"SOC 比 {soc_ratio:.2f} —— 幅度与收敛口径双过")
    elif l1 and l3 and not l2:
        verdict = (f"部分成立：{best_name} |q̂| 相对下降 {best_drop:.0%} 过线、"
                   f"护栏未破（clean {best_clean['q_ok_rate']:.0%}、SOC 比 "
                   f"{soc_ratio:.2f}），但 q_ok {best_full['q_ok_rate']:.0%} "
                   f"<{H32_L2_QOK:.0%} —— 幅度缓解、2% 收敛口径未达（贴线如实记）")
    elif not l3:
        verdict = (f"被否（护栏）：{best_name} 幅度下降 {best_drop:.0%} 但"
                   f"clean {best_clean['q_ok_rate']:.0%} 或 SOC 比 {soc_ratio:.2f} "
                   "破线——拆东补西")
    else:
        verdict = (f"被否：{best_name} |q̂| 中位相对下降 {best_drop:.0%} "
                   f"<{H32_L1_DROP:.0%} —— 鲁棒更新救不动 full 结构")

    out = {
        "dataset": dataset,
        "lines": {"L1_drop": H32_L1_DROP, "L2_qok": H32_L2_QOK,
                  "L3_clean": H32_L3_CLEAN, "L3_soc_ratio": H32_L3_SOC},
        "best": best_name,
        "checks": {"L1_drop": l1, "L2_qok": l2, "L3_guard": l3},
        "cells": {v: {"full": _cell(df, "full", v), "none": _cell(df, "none", v),
                      "drop": d, "soc_ratio": r}
                  for (v, _f, _c, d, r) in cands},
        "base_full": base_full,
        "best_drop": float(best_drop), "soc_ratio": float(soc_ratio),
        "verdict": verdict,
    }
    with open(os.path.join(MET, f"robust_r7_{dataset}_h32.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"=== h32 {dataset}：{verdict}")
    for vname, full, clean, drop, sr in cands:
        print(f"  {vname:<7s} drop {drop:+.0%}  q_ok {full['q_ok_rate']:.0%}  "
              f"|q̂| {full['q_rel_abs_med']:.2%}  clean {clean['q_ok_rate']:.0%}  "
              f"soc比 {sr:.2f}")
    return out


def stage_h33(dataset: str) -> dict:
    """H33：毒的机制在 white 重尾形状（剥尾/高斯化应当显著减毒）。"""
    rp = os.path.join(MET, f"robust_r7_{dataset}_replay.parquet")
    if not os.path.exists(rp):
        raise SystemExit("缺 replay 产物 —— 先跑 --stage probe")
    df = pd.read_parquet(rp)

    base_full = _cell(df, "full", "base")
    base_win = _cell(df, "full_winsor5", "base")
    base_gauss = _cell(df, "full_gauss", "base")
    drop_win = ((base_full["q_rel_abs_med"] - base_win["q_rel_abs_med"])
                / max(base_full["q_rel_abs_med"], 1e-9))
    drop_gauss = ((base_full["q_rel_abs_med"] - base_gauss["q_rel_abs_med"])
                  / max(base_full["q_rel_abs_med"], 1e-9))
    l1 = bool(drop_win >= H33_L1_WINSOR)
    l2 = bool(drop_gauss >= H33_L2_GAUSS)

    # 一致性（不进线）：鲁棒收益在重尾存在时应更大
    rob = "huber3"
    full_r = _cell(df, "full", rob)
    win_r = _cell(df, "full_winsor5", rob)
    gain_heavy = base_full["q_rel_abs_med"] - full_r["q_rel_abs_med"]
    gain_win = base_win["q_rel_abs_med"] - win_r["q_rel_abs_med"]
    consistent = bool(gain_heavy >= gain_win)

    if l1 and l2:
        verdict = (f"成立：剥 white 5% 重尾后 |q̂| 相对下降 {drop_win:.0%} "
                   f"≥{H33_L1_WINSOR:.0%}，高斯化后 {drop_gauss:.0%} "
                   f"≥{H33_L2_GAUSS:.0%} —— 毒在重尾形状而非方差")
    elif l1 or l2:
        verdict = (f"部分成立：剥尾 {drop_win:.0%}（线 {H33_L1_WINSOR:.0%}）、"
                   f"高斯化 {drop_gauss:.0%}（线 {H33_L2_GAUSS:.0%}）只过一条"
                   "——尾部有份但不是全部")
    else:
        verdict = (f"被否：剥尾仅 {drop_win:.0%}、高斯化 {drop_gauss:.0%}，"
                   "均未过线——white 重尾不是 q 中毒的主因（毒在慢/奇结构或闭环耦合）")

    out = {
        "dataset": dataset,
        "lines": {"L1_winsor_drop": H33_L1_WINSOR, "L2_gauss_drop": H33_L2_GAUSS},
        "checks": {"L1_winsor": l1, "L2_gauss": l2,
                   "L3_consistency": consistent},
        "cells": {"base_full": base_full, "base_winsor5": base_win,
                  "base_gauss": base_gauss,
                  "huber3_full": full_r, "huber3_winsor5": win_r,
                  "huber3_gauss": _cell(df, "full_gauss", "huber3")},
        "drop_winsor": float(drop_win), "drop_gauss": float(drop_gauss),
        "gain_heavy": float(gain_heavy), "gain_win": float(gain_win),
        "verdict": verdict,
    }
    with open(os.path.join(MET, f"robust_r7_{dataset}_h33.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"=== h33 {dataset}：{verdict}")
    print(f"  base  |q̂| {base_full['q_rel_abs_med']:.2%} → winsor5 "
          f"{base_win['q_rel_abs_med']:.2%} ({drop_win:+.0%}) → gauss "
          f"{base_gauss['q_rel_abs_med']:.2%} ({drop_gauss:+.0%})")
    print(f"  huber3 收益：full {gain_heavy:.2%} vs winsor5 {gain_win:.2%} "
          f"（一致性 {consistent}）")
    return out


def stage_h34(dataset: str) -> dict:
    """H34：鲁棒收益迁移到真实数据（非 replay）。"""
    rp = os.path.join(MET, f"robust_r7_{dataset}_real.parquet")
    if not os.path.exists(rp):
        raise SystemExit("缺 real 产物 —— 先跑 --stage probe")
    df = pd.read_parquet(rp)

    base = df[df.variant == "base"]
    results = {}
    best_name, best = None, None
    for vname in ("clip3", "huber3"):
        g = df[df.variant == vname]
        # 逐折配对中位
        merged = base.merge(g, on=["temp_c", "profile", "cap_scale"],
                            suffixes=("_b", "_r"))
        q_rel_drop = (float(np.median(np.abs(merged.q_rel_err_b)))
                      - float(np.median(np.abs(merged.q_rel_err_r))))
        q_rel_rel = q_rel_drop / max(float(np.median(np.abs(merged.q_rel_err_b))), 1e-9)
        soc_grow = (float(np.median(merged.rmse_soc_r))
                    - float(np.median(merged.rmse_soc_b))) \
            / max(float(np.median(merged.rmse_soc_b)), 1e-9)
        soc_grow_max = float(np.max(
            (merged.rmse_soc_r - merged.rmse_soc_b)
            / merged.rmse_soc_b.clip(lower=1e-9)))
        results[vname] = {
            "q_rel_abs_med_base": float(np.median(np.abs(merged.q_rel_err_b))),
            "q_rel_abs_med_rob": float(np.median(np.abs(merged.q_rel_err_r))),
            "q_rel_rel_drop": float(q_rel_rel),
            "q_ok_base": float(base.q_ok.mean()), "q_ok_rob": float(g.q_ok.mean()),
            "soc_grow": float(soc_grow), "soc_grow_max": float(soc_grow_max),
            "robust_frac_med": float(g.robust_frac.median()),
        }
        if best_name is None or results[vname]["q_rel_rel_drop"] > results[best_name]["q_rel_rel_drop"]:
            best_name, best = vname, results[vname]

    l1 = bool(best["q_rel_rel_drop"] >= H34_L1_QREL)
    l2 = bool(best["soc_grow"] <= H34_L2_SOC
              and best["soc_grow_max"] <= H34_L2_SOC_MAX)

    if l1 and l2:
        verdict = (f"成立：{best_name} 真实数据 |q̂| 中位相对下降 "
                   f"{best['q_rel_rel_drop']:.0%} ≥{H34_L1_QREL:.0%}，"
                   f"SOC 中位恶化 {best['soc_grow']:+.0%} ≤{H34_L2_SOC:.0%} "
                   f"（单折最大 {best['soc_grow_max']:+.0%}）——机制迁移到真实残差")
    elif l2 and not l1:
        verdict = (f"被否：{best_name} 真实数据 |q̂| 仅相对下降 "
                   f"{best['q_rel_rel_drop']:.0%} <{H34_L1_QREL:.0%}"
                   "——对症但真实残差的主量不在尾部（或 q 毒不止尾部）")
    else:
        verdict = (f"被否（护栏）：{best_name} SOC 恶化 {best['soc_grow']:+.0%} "
                   f"/ 单折 {best['soc_grow_max']:+.0%} —— 拆东补西")

    out = {
        "dataset": dataset,
        "lines": {"L1_qrel_drop": H34_L1_QREL, "L2_soc_grow": H34_L2_SOC,
                  "L2_soc_grow_max": H34_L2_SOC_MAX},
        "best": best_name,
        "checks": {"L1_qrel_drop": l1, "L2_soc_guard": l2},
        "per_variant": results,
        "verdict": verdict,
    }
    with open(os.path.join(MET, f"robust_r7_{dataset}_h34.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"=== h34 {dataset}：{verdict}")
    for vname, v in results.items():
        print(f"  {vname:<7s} |q̂| {v['q_rel_abs_med_base']:+.2%}→{v['q_rel_abs_med_rob']:+.2%} "
              f"(rel {v['q_rel_rel_drop']:+.0%})  q_ok {v['q_ok_base']:.0%}→{v['q_ok_rob']:.0%}  "
              f"soc {v['soc_grow']:+.0%}  触发 {v['robust_frac_med']:.1%}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["probe", "h32", "h33", "h34"], required=True)
    ap.add_argument("--dataset", choices=["calce", "panasonic"], default="panasonic")
    args = ap.parse_args()
    fn = {"probe": stage_probe, "h32": stage_h32, "h33": stage_h33,
          "h34": stage_h34}[args.stage]
    fn(args.dataset)


if __name__ == "__main__":
    main()
