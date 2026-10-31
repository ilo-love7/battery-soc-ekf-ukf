"""EKFQ 可行性边界：模型残差低到多少，在线容量辨识才能活过来。

v1.2 §10.8 的负结果：EKFQ（容量联合估计）在真实数据上全线劣于固定容量的 EKF，
机制是 38–53 mV 的模型残差与容量漂移在电压上同构，``q`` 把残差当容量吸收。
但「多低才够」一直没量化 —— 本脚本把这个负结果变成一条**规格线**。

两个子实验
----------
**A 边界扫描（合成）**：精确模型 + 注入已知水平的残差（慢正弦 + 白噪声 70/30，
对齐脚本 19 的慢/快结构），容量从 ±5% 误差起步。看 EKFQ 在每个残差水平上
能否把 ``q`` 收到 ±2% 且 SOC 优于「固定错容量」的 EKF。输出可行性阈值。

**B 真实数据复验**：用模型升级（脚本 20）后的最优变体参数，重跑 v1.2 的协议
（容量 0.95/1.00/1.05 × cc/ekf/ekfq，两数据集 LOO 参数）。与 v1.2 的 2RC
基线并排 —— 负结果是否翻案，直接看这两列。

用法：
    python scripts/21_ekfq_boundary.py --part a
    python scripts/21_ekfq_boundary.py --part b --dataset calce
    python scripts/21_ekfq_boundary.py --part b --dataset panasonic
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from bmsoc import metrics as met                              # noqa: E402
from bmsoc import ocv as ocvmod                               # noqa: E402
from bmsoc.ecm import RCParams, _unpack_vec, simulate         # noqa: E402
from bmsoc.filters import CoulombCounter, EKF, EKFQ, FilterConfig  # noqa: E402
from bmsoc.profiles import DrivingProfile                     # noqa: E402

MET = os.path.join(ROOT, "results", "metrics")

# ---- A：边界扫描 --------------------------------------------------------
RESIDUAL_MV = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0]   # 注入残差总量（RMS）
SLOW_FRAC = 0.70                                    # 慢分量占比（脚本 19：58–85%）
CAP_ERR = [0.95, 1.05]                              # 起步容量 q0 = s·Q_true
SEEDS = [0, 1, 2, 3, 4]
Q_TRUE = 2.5
NOISE_V = 0.005

# ---- B：真实复验 --------------------------------------------------------
CAP_SCALES = [0.95, 1.00, 1.05]


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =====================================================================
# A 合成边界
# =====================================================================

def _curve(slope: float = 1.0, base: float = 3.1):
    soc = np.linspace(0.0, 1.0, 21)
    return ocvmod.from_rest_points(soc, base + slope * soc, temp_c=25.0)


def _synth_profile(n: int = 12000, seed: int = 0):
    """富激励工况：随机电流 + 少量方波段，保证 q 可辨识（v1.2：可辨识性来自激励）。

    两个硬约束（都踩过坑）：
    1. ``i[0] = 0`` —— 滤波器极化初值约定 ``u_j = R_j·i0``，植物 ``u=0`` 起步；
       电流非零起步会把初值失配灌进 ``b``。
    2. **电流按目标放电深度 75% 归一化** —— 第一版直接叠方波把电芯放到了
       SOC −0.30，滤波器裁剪在 −0.05，末段 25 pp 误差全是裁剪伪影。
       归一化后末端 SOC ≈ 0.20，全程不出 [0, 1]。
    """
    rng = np.random.default_rng(seed)
    i = rng.normal(-0.6, 1.6, n)
    # 叠两段恒流平台与一段充电，模拟真实负载节奏
    i[n // 5:n // 5 + 800] = -3.0
    i[2 * n // 5:2 * n // 5 + 600] = 1.5
    i[3 * n // 5:3 * n // 5 + 900] = -4.5
    ah = float(-i.sum()) / 3600.0          # 放出的安时（正）
    i *= (0.75 * Q_TRUE) / ah              # 目标 DOD 75%
    i[0] = 0.0
    return i


def _inject_residual(n: int, dt: float, rms_mv: float, seed: int):
    """慢正弦 + 白噪声，慢占比 SLOW_FRAC（波形结构对齐脚本 19 的实测残差）。"""
    if rms_mv <= 0:
        return np.zeros(n)
    rng = np.random.default_rng(seed + 991)
    rms = rms_mv / 1000.0
    slow_rms = rms * SLOW_FRAC ** 0.5
    fast_rms = rms * (1 - SLOW_FRAC) ** 0.5
    t = np.arange(n) * dt
    phase = rng.uniform(0, 2 * np.pi)
    slow = slow_rms * np.sqrt(2) * np.sin(2 * np.pi * t / 2400.0 + phase)
    fast = rng.normal(0.0, fast_rms, n)
    return slow + fast


def run_a() -> pd.DataFrame:
    dt = 1.0
    p_true = RCParams(r0=0.030, r1=0.012, tau1=12.0, r2=0.045, tau2=450.0)
    cfg = FilterConfig(soc0=0.95, sigma_soc0=0.02)
    rows = []
    for seed in SEEDS:
        i = _synth_profile(seed=seed)
        n = i.size
        sim = simulate(p_true, _curve(), i, dt, soc0=0.95, capacity_ah=Q_TRUE)
        # 前提断言：植物 SOC 不得出 [0, 1]（出界会被滤波器裁剪伪影污染）
        if float(sim["soc"].min()) < 0.02 or float(sim["soc"].max()) > 1.0:
            raise SystemExit(f"seed={seed} 植物 SOC 越界 "
                             f"[{sim['soc'].min():.3f}, {sim['soc'].max():.3f}]")
        for rms_mv in RESIDUAL_MV:
            v = sim["voltage"] + _inject_residual(n, dt, rms_mv, seed)
            prof = DrivingProfile(t_s=np.arange(n) * dt, current_a=i, voltage_v=v,
                                  temp_c=np.full(n, 25.0), soc_ref=sim["soc"],
                                  capacity_ah=Q_TRUE)
            for s in CAP_ERR:
                q0 = Q_TRUE * s
                ests = {
                    "ekf": EKF(ocv=_curve(), params=p_true, capacity_ah=q0, cfg=cfg),
                    "ekfq": EKFQ(ocv=_curve(), params=p_true, capacity_ah=q0, cfg=cfg),
                    "ekf_oracle": EKF(ocv=_curve(), params=p_true, capacity_ah=Q_TRUE, cfg=cfg),
                }
                for name, est in ests.items():
                    out = est.run(prof)
                    m = met.soc_metrics(out["soc_hat"], prof.soc_ref, prof.t_s, tol=0.02)
                    row = {"seed": seed, "resid_mv": rms_mv, "cap_scale": s,
                           "estimator": name,
                           "rmse": m["rmse"], "err_end": float(out["soc_hat"][-1] - prof.soc_ref[-1]),
                           "n": n}
                    if name == "ekfq":
                        qh = float(np.median(out["q_hat"][-n // 10:]))
                        row["q_rel_err"] = qh / Q_TRUE - 1.0
                        row["q_ok"] = abs(row["q_rel_err"]) < 0.02
                    rows.append(row)
    df = pd.DataFrame(rows)
    # 台架自检：零残差、真容量下 oracle 的 RMSE 必须近零，否则整条边界无效
    oracle0 = df[(df.resid_mv == 0) & (df.estimator == "ekf_oracle")]
    if float(oracle0.rmse.median()) > 0.002:
        raise SystemExit(f"台架失配：oracle 在零残差下 RMSE 中位 "
                         f"{oracle0.rmse.median()*100:.2f} pp，先查植物/滤波器初值约定")
    return df


# =====================================================================
# B 真实数据复验
# =====================================================================

def load_upgraded_params(dataset: str):
    """从脚本 20 的 parquet 重建每折的最优变体 RCParams + ΔOCV 结点。

    升级模型 = 最优 RC 变体 + ΔOCV(SOC) 形状重标定（脚本 20 第二阶段）。
    LFP 上这是 LOO 残差 37.4 → 17.6 mV 的那一步；NCA 上 ΔOCV 幅度只有 ~22 mV。
    """
    fp = os.path.join(MET, f"model_upgrade_{dataset}_folds.parquet")
    if not os.path.exists(fp):
        raise SystemExit(f"缺 {fp}：先跑 scripts/20_model_upgrade.py --dataset {dataset}")
    df = pd.read_parquet(fp)
    with open(os.path.join(MET, f"model_upgrade_{dataset}.json"), encoding="utf-8") as f:
        best = json.load(f)["best"]
    sub = df[df.variant == best]
    out = {}
    for _, r in sub.iterrows():
        vec = np.asarray(json.loads(r.vec_json), float)
        r0k, r1, tau1, r2, tau2, r3, tau3, r0_cur, dv = _unpack_vec(vec)
        p = RCParams(r0=float(np.mean(r0k)), r1=r1, tau1=tau1, r2=r2, tau2=tau2,
                     dv=dv, r3=r3, tau3=tau3, r0_cur=r0_cur)
        out[(float(r.temp_c), str(r.held_out))] = p
    docv = {}
    kp = os.path.join(MET, f"model_upgrade_{dataset}_docv_knots.parquet")
    if os.path.exists(kp):
        kd = pd.read_parquet(kp)
        for _, r in kd.iterrows():
            docv[(float(r.temp_c), str(r.held_out))] = (
                np.asarray(json.loads(r.soc_centers), float),
                np.asarray(json.loads(r.docv_mv), float))
    return out, best, docv


def load_base_params(dataset: str):
    """v1.2 基线：脚本 03/10 的二阶 RC 折参数。"""
    s12 = _load_module("s12", os.path.join(ROOT, "scripts", "12_calibration_stress.py"))
    _ocv, _profs, fold = (s12.load_calce() if dataset == "calce" else s12.load_panasonic())
    return fold


def run_b(dataset: str) -> pd.DataFrame:
    s12 = _load_module("s12", os.path.join(ROOT, "scripts", "12_calibration_stress.py"))
    ocv2d, profiles, base_fold = (s12.load_calce() if dataset == "calce"
                                  else s12.load_panasonic())
    up_params, best_variant, docv = load_upgraded_params(dataset)
    print(f"{dataset}：升级变体 = {best_variant}，参数 {len(up_params)} 折，"
          f"ΔOCV {len(docv)} 折")

    rows = []
    for p in profiles:
        temp = float(p.meta["temp_c"])
        prof_name = str(p.meta["profile"])
        for model_tag, params, ocv_use in (
            ("base2rc", base_fold(temp, prof_name), ocv2d),
            ("upgraded", up_params.get((temp, prof_name)),
             ocvmod.ShapeCorrected(ocv2d, *docv[(temp, prof_name)])
             if (temp, prof_name) in docv else ocv2d),
        ):
            if params is None:
                continue
            cfg = FilterConfig(soc0=float(np.clip(p.soc_ref[0], 0.0, 1.0)), sigma_soc0=0.05)
            for s in CAP_SCALES:
                q0 = p.capacity_ah * s
                ests = {
                    "cc": CoulombCounter(ocv=ocv_use, params=params, capacity_ah=q0, cfg=cfg),
                    "ekf": EKF(ocv=ocv_use, params=params, capacity_ah=q0, cfg=cfg),
                    "ekfq": EKFQ(ocv=ocv_use, params=params, capacity_ah=q0, cfg=cfg),
                }
                for name, est in ests.items():
                    out = est.run(p)
                    m = met.soc_metrics(out["soc_hat"], p.soc_ref, p.t_s, tol=0.02)
                    row = {"dataset": dataset, "model": model_tag,
                           "temp_c": temp, "profile": prof_name,
                           "estimator": name, "cap_scale": s,
                           "n": len(p), "soc_end": float(p.soc_ref[-1]),
                           "err_end": float(out["soc_hat"][-1] - p.soc_ref[-1]),
                           **{f"soc_{k}": v for k, v in m.items()}}
                    if name == "ekfq":
                        qh = float(np.median(out["q_hat"][-max(1, len(p) // 10):]))
                        row["q_rel_err"] = qh / p.capacity_ah - 1.0
                        row["q_ok"] = abs(row["q_rel_err"]) < 0.02
                    rows.append(row)
    return pd.DataFrame(rows)


def summarize_a(df: pd.DataFrame) -> dict:
    out = {"protocol": {"resid_mv": RESIDUAL_MV, "slow_frac": SLOW_FRAC,
                        "cap_err": CAP_ERR, "seeds": SEEDS},
           "by_resid": {}}
    for rms in RESIDUAL_MV:
        g = df[(df.resid_mv == rms) & (df.cap_scale == 1.05)]
        ekfq = g[g.estimator == "ekfq"]
        ekf = g[g.estimator == "ekf"]
        oracle = g[g.estimator == "ekf_oracle"]
        out["by_resid"][f"{rms:g}"] = {
            "ekfq_rmse": float(ekfq.rmse.median()),
            "ekf_rmse": float(ekf.rmse.median()),
            "oracle_rmse": float(oracle.rmse.median()),
            "ekfq_q_rel_err_med": float(ekfq.q_rel_err.median()),
            "ekfq_q_ok_rate": float(ekfq.q_ok.mean()),
            "ekfq_beats_ekf": bool(ekfq.rmse.median() < ekf.rmse.median()),
        }
    # 阈值：q 收敛率 ≥ 80% 的最大残差
    ok = [float(k) for k, v in out["by_resid"].items() if v["ekfq_q_ok_rate"] >= 0.8]
    out["viability_threshold_mv"] = max(ok) if ok else 0.0
    return out


def summarize_b(df: pd.DataFrame, dataset: str) -> dict:
    out = {"dataset": dataset, "by_model": {}}
    for model, gm in df.groupby("model"):
        out["by_model"][model] = {}
        for s, g in gm.groupby("cap_scale"):
            rec = {}
            for est in ("cc", "ekf", "ekfq"):
                sub = g[g.estimator == est]
                rec[est] = {
                    "rmse_med_pp": float(sub.soc_rmse.median() * 100),
                    "err_end_med_pp": float(sub.err_end.median() * 100),
                    "tail_med_pp": float(sub.soc_tail_rmse.median() * 100) if "soc_tail_rmse" in sub else None,
                }
                if est == "ekfq":
                    rec[est]["q_rel_err_med"] = float(sub.q_rel_err.median())
                    rec[est]["q_ok_rate"] = float(sub.q_ok.mean())
            out["by_model"][model][f"{s:g}"] = rec
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b"], required=True)
    ap.add_argument("--dataset", choices=["calce", "panasonic"], default="calce")
    args = ap.parse_args()

    if args.part == "a":
        t0 = time.time()
        df = run_a()
        df.to_parquet(os.path.join(MET, "ekfq_boundary_runs.parquet"), index=False)
        summary = summarize_a(df)
        summary["elapsed_s"] = round(time.time() - t0, 1)
        with open(os.path.join(MET, "ekfq_boundary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print(f"\n=== EKFQ 可行性边界（cap_scale=1.05，RMSE 中位 pp） ===")
        print(f"{'resid_mV':>8} {'ekfq':>8} {'ekf固定':>8} {'oracle':>8} {'q_err%':>8} {'q收率':>6}")
        for k, v in summary["by_resid"].items():
            print(f"{k:>8} {v['ekfq_rmse']*100:8.2f} {v['ekf_rmse']*100:8.2f} "
                  f"{v['oracle_rmse']*100:8.2f} {v['ekfq_q_rel_err_med']*100:8.2f} "
                  f"{v['ekfq_q_ok_rate']*100:5.0f}%")
        print(f"可行性阈值：慢变残差 ≤ {summary['viability_threshold_mv']:.0f} mV "
              f"（q 收敛率 ≥ 80%）")
    else:
        t0 = time.time()
        df = run_b(args.dataset)
        if df.empty:
            raise SystemExit("没有任何 run——查数据根目录与脚本 20 输出")
        df.to_parquet(os.path.join(MET, f"ekfq_reverify_{args.dataset}_runs.parquet"),
                      index=False)
        summary = summarize_b(df, args.dataset)
        summary["best_variant"] = load_upgraded_params(args.dataset)[1]
        summary["elapsed_s"] = round(time.time() - t0, 1)
        with open(os.path.join(MET, f"ekfq_reverify_{args.dataset}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print(f"\n=== {args.dataset} EKFQ 复验（RMSE 中位 pp / 段末中位 pp） ===")
        for model, ms in summary["by_model"].items():
            for s, rec in ms.items():
                e = rec["ekfq"]; f_ = rec["ekf"]
                print(f"  {model:<8s} s={s:<5s} ekf {f_['rmse_med_pp']:5.2f}/{f_['err_end_med_pp']:+6.2f}"
                      f"  ekfq {e['rmse_med_pp']:5.2f}/{e['err_end_med_pp']:+6.2f}"
                      f"  q_err {e['q_rel_err_med']*100:+6.2f}% 收率 {e['q_ok_rate']*100:3.0f}%")


if __name__ == "__main__":
    main()
