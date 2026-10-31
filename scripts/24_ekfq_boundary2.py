#!/usr/bin/env python
"""24_ekfq_boundary2.py — 第六轮：EKFQ 边界细化 + 热修正模型复验（v1.5）。

Part A（合成）：v1.4 的 20 mV 阈值来自 70/30 慢快混合残差；本轮把注入残差按
**结构**分层 —— 慢漂移 / 快奇（∝ i，热失配的签名）/ 快偶（∝ |i|），
回答「同样 RMS 下哪种结构对 q 杀伤最大」。

Part B（真实，NCA）：热修正模型（逐样本 R(T_meas)，训练折归一化 r_ref）
重跑 EKFQ 复验，对照 v1.4 恒参数基线。LFP 无逐样本温度数据，沿用 v1.4 结论。

输出：``results/metrics/ekfq_boundary2.*`` / ``ekfq_therm_reverify_*``。
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

RESIDUAL_MV = [0.0, 10.0, 20.0, 30.0, 50.0]
STRUCTURES = ["slow", "odd", "even"]
CAP_ERR = [0.95, 1.05]
SEEDS = [0, 1, 2, 3, 4]


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =====================================================================
# Part A 合成：残差结构分层的边界
# =====================================================================

def _inject(n: int, dt: float, rms_mv: float, seed: int, structure: str, i: np.ndarray):
    """按结构注入给定 RMS 的残差（V）。

    - ``slow``：70/30 慢正弦 + 白噪声（v1.4 协议，对照基准）
    - ``odd``：∝ i 的快分量（奇函数 —— 热电阻失配 R·ΔT·i 的签名）
    - ``even``：∝ |i|−mean 的快分量（偶函数 —— 电流幅值型误差）
    """
    if rms_mv <= 0:
        return np.zeros(n)
    rng = np.random.default_rng(seed + 991)
    rms = rms_mv / 1000.0
    t = np.arange(n) * dt
    if structure == "slow":
        slow_rms = rms * 0.7 ** 0.5
        fast_rms = rms * 0.3 ** 0.5
        phase = rng.uniform(0, 2 * np.pi)
        slow = slow_rms * np.sqrt(2) * np.sin(2 * np.pi * t / 2400.0 + phase)
        return slow + rng.normal(0.0, fast_rms, n)
    if structure == "odd":
        g = i - i.mean()
        return rms * g / max(np.sqrt(np.mean(g ** 2)), 1e-9)
    if structure == "even":
        g = np.abs(i) - np.abs(i).mean()
        return rms * g / max(np.sqrt(np.mean(g ** 2)), 1e-9)
    raise ValueError(structure)


def run_a() -> pd.DataFrame:
    from bmsoc.ecm import RCParams, simulate
    from bmsoc.filters import EKF, EKFQ, FilterConfig
    from bmsoc.profiles import DrivingProfile
    s21 = _load_module("s21", os.path.join(ROOT, "scripts", "21_ekfq_boundary.py"))
    dt = 1.0
    p_true = RCParams(r0=0.030, r1=0.012, tau1=12.0, r2=0.045, tau2=450.0)
    cfg = FilterConfig(soc0=0.95, sigma_soc0=0.02)
    rows = []
    for seed in SEEDS:
        i = s21._synth_profile(seed=seed)
        n = i.size
        sim = simulate(p_true, s21._curve(), i, dt, soc0=0.95,
                       capacity_ah=s21.Q_TRUE)
        if float(sim["soc"].min()) < 0.02 or float(sim["soc"].max()) > 1.0:
            raise SystemExit(f"seed={seed} 植物 SOC 越界")
        for structure in STRUCTURES:
            for rms_mv in RESIDUAL_MV:
                v = sim["voltage"] + _inject(n, dt, rms_mv, seed, structure, i)
                prof = DrivingProfile(t_s=np.arange(n) * dt, current_a=i,
                                      voltage_v=v, temp_c=np.full(n, 25.0),
                                      soc_ref=sim["soc"], capacity_ah=s21.Q_TRUE)
                for s in CAP_ERR:
                    q0 = s21.Q_TRUE * s
                    ests = {
                        "ekf": EKF(ocv=s21._curve(), params=p_true,
                                   capacity_ah=q0, cfg=cfg),
                        "ekfq": EKFQ(ocv=s21._curve(), params=p_true,
                                     capacity_ah=q0, cfg=cfg),
                        "ekf_oracle": EKF(ocv=s21._curve(), params=p_true,
                                          capacity_ah=s21.Q_TRUE, cfg=cfg),
                    }
                    for name, est in ests.items():
                        out = est.run(prof)
                        row = {"seed": seed, "structure": structure,
                               "resid_mv": rms_mv, "cap_scale": s,
                               "estimator": name,
                               "rmse": float(np.sqrt(np.mean(
                                   (out["soc_hat"] - prof.soc_ref) ** 2))),
                               "n": n}
                        if name == "ekfq":
                            qh = float(np.median(out["q_hat"][-n // 10:]))
                            row["q_rel_err"] = qh / s21.Q_TRUE - 1.0
                            row["q_ok"] = abs(row["q_rel_err"]) < 0.02
                        rows.append(row)
    df = pd.DataFrame(rows)
    oracle0 = df[(df.resid_mv == 0) & (df.estimator == "ekf_oracle")]
    if float(oracle0.rmse.median()) > 0.002:
        raise SystemExit(f"台架失配：oracle 零残差 RMSE "
                         f"{oracle0.rmse.median()*100:.2f} pp")
    return df


# =====================================================================
# Part B 真实：热修正模型下的 EKFQ 复验（NCA）
# =====================================================================

class ThermalAnchor:
    """鸭子类型 SoHSchedule（q_at/r0_at）：逐步热电阻 R(T_meas)。

    容量恒定传入（EKFQ 忽略 q_arr、用自身 q 状态；EKF/cc 用这个名义容量）。
    """

    def __init__(self, t_s: np.ndarray, r0_arr: np.ndarray, q_ah: float):
        self.t_s = np.asarray(t_s, float)
        self.r0_arr = np.asarray(r0_arr, float)
        self.q_ah = float(q_ah)

    def q_at(self, t):
        return np.full_like(np.asarray(t, float), self.q_ah)

    def r0_at(self, t, default=None):
        return np.interp(np.asarray(t, float), self.t_s, self.r0_arr)


def run_b(dataset: str = "panasonic") -> pd.DataFrame:
    from bmsoc.ecm import _unpack_vec
    from bmsoc.filters import CoulombCounter, EKF, EKFQ, FilterConfig
    s21 = _load_module("s21", os.path.join(ROOT, "scripts", "21_ekfq_boundary.py"))
    s23 = _load_module("s23", os.path.join(ROOT, "scripts", "23_residual_round2.py"))
    s12 = _load_module("s12", os.path.join(ROOT, "scripts", "12_calibration_stress.py"))

    ocv2d, profiles, _fold = (s12.load_calce() if dataset == "calce"
                              else s12.load_panasonic())
    up_params, best_variant, docv = s21.load_upgraded_params(dataset)
    temps_r, r0_med = s23._r0_curve(dataset)

    # 训练折归一化的 r_ref（脚本 23 热判决阶段逐折算好，只用了训练折）
    rref_of = {}
    tp = os.path.join(MET, f"residual_r2_{dataset}_therm_folds.parquet")
    if os.path.exists(tp):
        td = pd.read_parquet(tp)
        for _, r in td.iterrows():
            if "r_ref" in td.columns and not pd.isna(r.get("r_ref")):
                rref_of[(float(r.temp_c), str(r.held_out))] = float(r.r_ref)

    rows = []
    for p in profiles:
        temp = float(p.meta["temp_c"])
        prof_name = str(p.meta["profile"])
        params = up_params.get((temp, prof_name))
        if params is None:
            continue
        import bmsoc.ocv as ocvmod
        ocv_use = (ocvmod.ShapeCorrected(ocv2d, *docv[(temp, prof_name)])
                   if (temp, prof_name) in docv else ocv2d)
        T = np.asarray(p.temp_c, float)
        t_nom = temp
        r_nom = float(np.interp(t_nom, temps_r, r0_med))
        ratio = np.interp(T, temps_r, r0_med) / max(r_nom, 1e-9)
        r_ref = rref_of.get((temp, prof_name), 1.0)
        r0_arr = float(params.r0) * ratio / max(r_ref, 1e-9)
        anchor = ThermalAnchor(p.t_s, r0_arr, p.capacity_ah)

        cfg = FilterConfig(soc0=float(np.clip(p.soc_ref[0], 0.0, 1.0)), sigma_soc0=0.05)
        for model_tag, anchor_use in (("upgraded", None),
                                      ("upgraded_thermal", anchor)):
            for s in (0.95, 1.05):
                q0 = p.capacity_ah * s
                ests = {
                    "cc": CoulombCounter(ocv=ocv_use, params=params,
                                         capacity_ah=q0, cfg=cfg),
                    "ekf": EKF(ocv=ocv_use, params=params, capacity_ah=q0, cfg=cfg),
                    "ekfq": EKFQ(ocv=ocv_use, params=params, capacity_ah=q0, cfg=cfg),
                }
                for name, est in ests.items():
                    out = est.run(p, anchor=anchor_use)
                    rmse = float(np.sqrt(np.mean((out["soc_hat"] - p.soc_ref) ** 2)))
                    row = {"dataset": dataset, "model": model_tag,
                           "temp_c": temp, "profile": prof_name,
                           "estimator": name, "cap_scale": s,
                           "n": len(p), "rmse": rmse,
                           "err_end": float(out["soc_hat"][-1] - p.soc_ref[-1]),
                           "t_swing_k": float(T.max() - T.min())}
                    if name == "ekfq":
                        qh = float(np.median(out["q_hat"][-max(1, len(p) // 10):]))
                        row["q_rel_err"] = qh / p.capacity_ah - 1.0
                        row["q_ok"] = abs(row["q_rel_err"]) < 0.02
                    rows.append(row)
        print(f"[{temp:5.0f}] {prof_name:<5s} 摆幅 {T.max()-T.min():4.1f}K "
              f"r_ref {r_ref:.3f} 完成")
    return pd.DataFrame(rows)


def summarize_a(df: pd.DataFrame) -> dict:
    out = {"protocol": {"resid_mv": RESIDUAL_MV, "structures": STRUCTURES,
                        "cap_err": CAP_ERR, "seeds": SEEDS}}
    tbl = {}
    for st in STRUCTURES:
        sub = df[(df.structure == st) & (df.estimator == "ekfq")]
        ekf = df[(df.structure == st) & (df.estimator == "ekf")]
        tbl[st] = {
            str(rv): {
                "ekfq_rmse": float(sub[sub.resid_mv == rv].rmse.mean()),
                "ekf_rmse": float(ekf[ekf.resid_mv == rv].rmse.mean()),
                "q_ok_rate": float(sub[sub.resid_mv == rv].q_ok.mean()),
                "q_rel_err": float(sub[sub.resid_mv == rv].q_rel_err.mean()),
            } for rv in sorted(sub.resid_mv.unique())}
    out["by_structure"] = tbl
    with open(os.path.join(MET, "ekfq_boundary2.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for st in STRUCTURES:
        print(f"--- {st} ---")
        for rv, d in tbl[st].items():
            print(f"  {rv:>4} mV: EKFQ {d['ekfq_rmse']*100:5.2f} pp  "
                  f"EKF {d['ekf_rmse']*100:5.2f} pp  "
                  f"q收敛 {d['q_ok_rate']:.0%}  q̂误差 {d['q_rel_err']:+.2%}")
    return out


def summarize_b(df: pd.DataFrame, dataset: str) -> dict:
    out = {"dataset": dataset, "n_rows": int(len(df))}
    for model in df.model.unique():
        for est in ("ekf", "ekfq"):
            sub = df[(df.model == model) & (df.estimator == est)]
            out[f"{model}_{est}_rmse"] = float(sub.rmse.median())
            if est == "ekfq":
                out[f"{model}_q_ok_rate"] = float(sub.q_ok.mean())
    # us06 尾折单列（热修正的主战场）
    u = df[df.profile == "us06"]
    for model in u.model.unique():
        for est in ("ekf", "ekfq"):
            sub = u[(u.model == model) & (u.estimator == est)]
            out[f"us06_{model}_{est}_rmse"] = float(sub.rmse.median())
    with open(os.path.join(MET, f"ekfq_therm_reverify_{dataset}.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a", "b"], required=True)
    ap.add_argument("--dataset", choices=["calce", "panasonic"], default="panasonic")
    args = ap.parse_args()
    if args.stage == "a":
        df = run_a()
        df.to_parquet(os.path.join(MET, "ekfq_boundary2_runs.parquet"), index=False)
        summarize_a(df)
    else:
        df = run_b(args.dataset)
        df.to_parquet(os.path.join(MET, f"ekfq_therm_reverify_{args.dataset}_runs.parquet"),
                      index=False)
        summarize_b(df, args.dataset)


if __name__ == "__main__":
    main()
