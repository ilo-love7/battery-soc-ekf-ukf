"""开环电压残差摸底：38–53 mV 里到底装着什么。

v1.2 §10.8 把 EKFQ 的失败归因于「模型残差与容量漂移在电压上同构」，但残差本身
一直没有解剖过。本脚本对两个数据集的**留一工况**残差做四个维度的分解，直接回答
「模型该往哪里改」：

1. **充放方向**（hysteresis 特征）：充电偏正、放电偏负 ⇒ 充放滞后未建模；
   两个方向各自 rms 是误差的可回收上限。
2. **电流幅值**（R0 特征）：残差随 |i| 线性增长 ⇒ 欧姆内阻标定不足；
   回归斜率就是 R0 误差（mV/A）。
3. **慢/快频带**（极化阶数特征）：残差的低通分量（滑动均值）占主导 ⇒
   慢极化/扩散未被 RC 支路吃掉（与 τ2 顶界 17/24、24/24 一致）；
   快分量占主导 ⇒ 噪声/OCV 细节，不是模型阶数问题。
4. **滞回状态相关性**：残差与一阶滞回状态 h（对 sign(i) 低通）的相关系数，
   |ρ| 高 ⇒ 加滞回模型能直接吃掉这么多。

输出
----
- results/metrics/residual_autopsy_{dataset}_samples.parquet  逐样本残差（可复算）
- results/metrics/residual_autopsy_{dataset}.json            分解汇总
- results/metrics/residual_autopsy_summary.json              两数据集对照

用法：
    python scripts/19_residual_autopsy.py --dataset calce
    python scripts/19_residual_autopsy.py --dataset panasonic
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

from bmsoc.ecm import open_loop_from_truth  # noqa: E402

MET = os.path.join(ROOT, "results", "metrics")

REST_A = 0.15            # |i| < 0.15 A 视为静置
HYST_TAU_S = 600.0       # 滞回状态的时间常数（LFP/NCA 的滞后都是分钟级）
SLOW_WINDOWS_S = (60.0, 300.0, 900.0)
I_BINS = (0.0, 0.5, 1.0, 2.0, 4.0)


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_dataset(dataset: str):
    s12 = _load_module("s12", os.path.join(ROOT, "scripts", "12_calibration_stress.py"))
    return s12.load_calce() if dataset == "calce" else s12.load_panasonic()


def hyst_state(i: np.ndarray, dt: float, tau: float = HYST_TAU_S) -> np.ndarray:
    """一阶滞回状态：对 sign(i) 低通。h→+1 表示近期以充电为主。"""
    a = float(np.exp(-dt / max(tau, 1e-6)))
    s = np.sign(i)
    h = np.empty_like(i)
    acc = 0.0
    for k in range(i.size):
        acc = a * acc + (1.0 - a) * s[k]
        h[k] = acc
    return h


def slow_split(resid: np.ndarray, dt: float, window_s: float):
    """滑动均值低通 → (slow, fast)。min_periods=1 避免两端 NaN。"""
    w = max(3, int(round(window_s / dt)))
    slow = pd.Series(resid).rolling(w, center=True, min_periods=1).mean().to_numpy()
    return slow, resid - slow


def decompose(p, params, ocv2d) -> tuple:
    """一段的残差分解。返回 (stats dict, samples DataFrame)。"""
    i = np.asarray(p.current_a, float)
    v = np.asarray(p.voltage_v, float)
    dt = float(p.dt)
    # fold_params 给出的是「留出该段」的 LOO 参数
    vec = np.array([params.r0, params.r0, params.r0,
                    params.r1, params.tau1, params.r2, params.tau2,
                    getattr(params, "dv", 0.0)], float)
    v_hat = open_loop_from_truth(vec, p.soc_ref, i, ocv2d, dt, float(p.meta["temp_c"]))
    e = v_hat - v

    chg = i > REST_A
    dis = i < -REST_A
    rst = ~chg & ~dis

    def bd(mask):
        if not np.any(mask):
            return float("nan"), float("nan"), 0
        return float(np.mean(e[mask])), float(np.sqrt(np.mean(e[mask] ** 2))), int(mask.sum())

    chg_bias, chg_rmse, n_chg = bd(chg)
    dis_bias, dis_rmse, n_dis = bd(dis)
    rst_bias, rst_rmse, n_rst = bd(rst)

    # R0 特征：残差对 i 的回归斜率（mV/A）
    slope_i = float(np.polyfit(i, e, 1)[0] * 1e3) if i.std() > 1e-6 else float("nan")
    # 滞回特征：残差与 h 的相关
    h = hyst_state(i, dt)
    corr_h = float(np.corrcoef(h, e)[0, 1]) if e.std() > 1e-9 else float("nan")
    # 慢/快分解（三个窗口看谱）
    slow = {}
    for w in SLOW_WINDOWS_S:
        s, f = slow_split(e, dt, w)
        slow[f"w{int(w)}"] = {"slow_std_mv": float(np.std(s) * 1e3),
                             "fast_std_mv": float(np.std(f) * 1e3),
                             "slow_rms_mv": float(np.sqrt(np.mean(s ** 2)) * 1e3),
                             "fast_rms_mv": float(np.sqrt(np.mean(f ** 2)) * 1e3)}
    # 大电流子集（高功率工况的可回收部分）
    big = np.abs(i) >= 2.0
    big_rmse = float(np.sqrt(np.mean(e[big] ** 2)) * 1e3) if np.any(big) else float("nan")

    stats = {
        "temp_c": float(p.meta["temp_c"]), "profile": str(p.meta["profile"]),
        "n": int(e.size), "dt": dt, "capacity_ah": p.capacity_ah,
        "soc_end": float(p.soc_ref[-1]),
        "rmse_mv": float(np.sqrt(np.mean(e ** 2)) * 1e3),
        "bias_mv": float(np.mean(e) * 1e3),
        "p90_mv": float(np.quantile(np.abs(e), 0.9) * 1e3),
        "chg_bias_mv": chg_bias * 1e3, "chg_rmse_mv": chg_rmse * 1e3, "n_chg": n_chg,
        "dis_bias_mv": dis_bias * 1e3, "dis_rmse_mv": dis_rmse * 1e3, "n_dis": n_dis,
        "rst_bias_mv": rst_bias * 1e3, "rst_rmse_mv": rst_rmse * 1e3, "n_rst": n_rst,
        "hyst_gap_mv": (chg_bias - dis_bias) * 1e3,   # 充放偏置差 = 滞后可回收上限
        "slope_i_mv_a": slope_i,
        "corr_hyst": corr_h,
        "big_i_rmse_mv": big_rmse,
        "dv_mv": float(getattr(params, "dv", 0.0)) * 1e3,
        "tau2": float(params.tau2),
        **{f"{k1}_{k2}": v2 for k1, d in slow.items() for k2, v2 in d.items()},
    }
    samples = pd.DataFrame({
        "t_s": p.t_s, "i": i, "v": v, "resid_v": e, "hyst": h,
        "temp_c": float(p.meta["temp_c"]), "profile": str(p.meta["profile"]),
    })
    return stats, samples


def summarize(df: pd.DataFrame, dataset: str) -> dict:
    def med(col):
        return float(df[col].median() * 1) if df[col].notna().any() else float("nan")

    g = df.groupby("temp_c")
    return {
        "dataset": dataset,
        "n_folds": int(len(df)),
        "rmse_median_mv": float(df.rmse_mv.median()),
        "rmse_p90_mv": float(df.rmse_mv.quantile(0.9)),
        "bias_median_mv": float(df.bias_mv.median()),
        "by_temp_rmse_mv": {f"{t:g}": float(x.rmse_mv.median()) for t, x in g},
        # ① 充放方向
        "chg_bias_median_mv": float(df.chg_bias_mv.median()),
        "dis_bias_median_mv": float(df.dis_bias_mv.median()),
        "hyst_gap_median_mv": float(df.hyst_gap_mv.median()),
        "hyst_gap_p90_mv": float(df.hyst_gap_mv.quantile(0.9)),
        # ② 电流幅值
        "slope_i_median_mv_a": float(df.slope_i_mv_a.median()),
        "big_i_rmse_median_mv": float(df.big_i_rmse_mv.median()),
        # ③ 慢/快
        "slow300_rms_median_mv": float(df.w300_slow_rms_mv.median()),
        "fast300_rms_median_mv": float(df.w300_fast_rms_mv.median()),
        "slow300_frac": float(df.w300_slow_rms_mv.median()
                             / max(df.rmse_mv.median(), 1e-9)),
        # ④ 滞回相关
        "corr_hyst_median": float(df.corr_hyst.median()),
        "corr_hyst_abs_p90": float(df.corr_hyst.abs().quantile(0.9)),
        "worst_folds": df.nlargest(5, "rmse_mv")[
            ["temp_c", "profile", "rmse_mv", "bias_mv", "hyst_gap_mv",
             "slope_i_mv_a", "w300_slow_rms_mv", "corr_hyst"]].round(3).to_dict("records"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["calce", "panasonic"], required=True)
    args = ap.parse_args()

    ocv2d, profiles, fold_params = load_dataset(args.dataset)
    rows, samp_frames = [], []
    for p in profiles:
        params = fold_params(float(p.meta["temp_c"]), str(p.meta["profile"]))
        stats, samples = decompose(p, params, ocv2d)
        rows.append(stats)
        samp_frames.append(samples)
    if not rows:
        raise SystemExit("没有任何段可分解——查数据根目录（CALCE_UMD / PANASONIC_ROOT）")

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(MET, f"residual_autopsy_{args.dataset}_folds.parquet"),
                  index=False)
    pd.concat(samp_frames, ignore_index=True).to_parquet(
        os.path.join(MET, f"residual_autopsy_{args.dataset}_samples.parquet"), index=False)
    summary = summarize(df, args.dataset)
    with open(os.path.join(MET, f"residual_autopsy_{args.dataset}.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    # 与另一数据集合并
    other = "panasonic" if args.dataset == "calce" else "calce"
    both = {}
    for ds in (args.dataset, other):
        fp = os.path.join(MET, f"residual_autopsy_{ds}.json")
        if os.path.exists(fp):
            with open(fp, encoding="utf-8") as f:
                both[ds] = json.load(f)
    with open(os.path.join(MET, "residual_autopsy_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(both, f, ensure_ascii=False, indent=1)

    print(f"\n=== {args.dataset} 残差摸底（LOO，{len(df)} 折） ===")
    print(f"RMSE 中位 {summary['rmse_median_mv']:.1f} mV，P90 {summary['rmse_p90_mv']:.1f} mV")
    print(f"① 充放偏置差（滞后上限）中位 {summary['hyst_gap_median_mv']:+.1f} mV "
          f"（充 {summary['chg_bias_median_mv']:+.1f} / 放 {summary['dis_bias_median_mv']:+.1f}）")
    print(f"② 残差-i 斜率中位 {summary['slope_i_median_mv_a']:+.1f} mV/A，"
          f"|i|≥2A 子集 RMSE {summary['big_i_rmse_median_mv']:.1f} mV")
    print(f"③ 慢(300s) RMS {summary['slow300_rms_median_mv']:.1f} mV vs "
          f"快 {summary['fast300_rms_median_mv']:.1f} mV "
          f"（慢占 {summary['slow300_frac']*100:.0f}%）")
    print(f"④ 滞回相关 ρ 中位 {summary['corr_hyst_median']:+.2f}，|ρ| P90 {summary['corr_hyst_abs_p90']:.2f}")
    print(f"按温度: {summary['by_temp_rmse_mv']}")


if __name__ == "__main__":
    main()
