"""模型升级对照：二阶 RC 基线 vs 三阶 RC vs 电流相关 R0（留一工况）。

残差摸底（脚本 19）给出的两条证据：
① LOO 残差的 58–85%（LFP 85%、NCA 58%）在 >60 s 的慢频带 ⇒ 慢极化欠拟合；
② NCA 大电流子集（|i|≥2 A）RMSE 77.9 mV ≫ 整体 ⇒ 高倍率下内阻随电流上升。
另有一条反证：充放偏置差只有 4–5 mV，「LFP 充放滞后」假设**不成立**
（v1.2 把 dv +37 mV 读成滞后是误判，dv 是常值基准偏，已被现有模型吸收）。

据此比较四个变体（每个温度留一工况，训练拟合、留出验证，与 03/10 同协议）：

    2rc        基线（v1.x 全部数字的模型）
    2rc_cur    2rc + r0_cur·i|i|      （电流相关内阻）
    3rc        2rc + r3/tau3         （第三条慢极化支路，tau3 ≤ 50000 s）
    3rc_cur    两者都要

输出
----
- results/metrics/model_upgrade_{dataset}_folds.parquet   逐折 × 逐变体
- results/metrics/model_upgrade_{dataset}.json            对比汇总（含慢/快分解）
- results/metrics/model_upgrade_summary.json              两数据集对照

用法：
    python scripts/20_model_upgrade.py --dataset calce
    python scripts/20_model_upgrade.py --dataset panasonic --variants 3rc,3rc_cur --limit 4
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

from bmsoc.ecm import _unpack_vec, fit_cycle_params, open_loop_from_truth  # noqa: E402

MET = os.path.join(ROOT, "results", "metrics")

VARIANTS = {
    "2rc":     dict(n_rc=2, fit_r0_cur=False),
    "2rc_cur": dict(n_rc=2, fit_r0_cur=True),
    "3rc":     dict(n_rc=3, fit_r0_cur=False),
    "3rc_cur": dict(n_rc=3, fit_r0_cur=True),
}
NCA_DT = 2.0
CALCE_DT = 1.0
MAX_NFEV = 300


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_dataset(dataset: str, dt: float):
    s12 = _load_module("s12", os.path.join(ROOT, "scripts", "12_calibration_stress.py"))
    ocv2d, profiles, _fold = s12.load_calce() if dataset == "calce" else s12.load_panasonic()
    return ocv2d, profiles


def folds_for(profiles):
    """按 (温度) 分组建 LOO 折：同温度下逐段留出。"""
    by_temp = {}
    for p in profiles:
        by_temp.setdefault(float(p.meta["temp_c"]), []).append(p)
    for temp in sorted(by_temp):
        sub = by_temp[temp]
        for held in sub:
            train = [p for p in sub if p is not held]
            if not train:
                continue
            yield temp, held, train


def slow_fast_rmse(resid: np.ndarray, dt: float, window_s: float = 300.0):
    w = max(3, int(round(window_s / dt)))
    slow = pd.Series(resid).rolling(w, center=True, min_periods=1).mean().to_numpy()
    fast = resid - slow
    return (float(np.sqrt(np.mean(slow ** 2)) * 1e3),
            float(np.sqrt(np.mean(fast ** 2)) * 1e3))


def multi_start_fit(wins, kwargs, inits):
    best = None
    for x0 in inits:
        vec, info = fit_cycle_params(wins, x0=x0, max_nfev=MAX_NFEV, dv_bound=0.15, **kwargs)
        if best is None or info["rmse_v"] < best[1]["rmse_v"]:
            best = (vec, info)
    return best


def make_inits(variant: str, r0_med: float):
    """每变体 3 组初值：慢支路的时间常数跨 1–2 个数量级。"""
    kw = VARIANTS[variant]
    heads = [
        np.array([r0_med, r0_med, r0_med * 1.1, 0.008, 10.0, 0.05, 800.0]),
        np.array([r0_med * 0.7, r0_med * 0.7, r0_med * 0.7, 0.004, 5.0, 0.02, 300.0]),
        np.array([r0_med * 1.4, r0_med * 1.4, r0_med * 1.4, 0.015, 25.0, 0.08, 2000.0]),
    ]
    out = []
    for h in heads:
        tail = []
        if kw["n_rc"] == 3:
            tail += [0.015, 2500.0]
        if kw["fit_r0_cur"]:
            tail += [0.003]
        tail += [0.0]
        out.append(np.concatenate([h, tail]))
    return out


def run(args) -> pd.DataFrame:
    ocv2d, profiles = load_dataset(args.dataset, args.dt)
    # 容量基准给 fit 用不到（开环真值 SOC），但记录上便于核对
    rows = []
    n_folds = 0
    for temp, held, train in folds_for(profiles):
        n_folds += 1
        if args.limit and n_folds > args.limit:
            break
        wins = [(p.soc_ref, p.current_a, p.voltage_v, p.dt, ocv2d, temp) for p in train]
        r0_med = 0.03 if args.dataset == "calce" else 0.05
        held_s, held_i, held_v = held.soc_ref, held.current_a, held.voltage_v
        for variant in args.variants:
            t0 = time.time()
            vec, info = multi_start_fit(wins, VARIANTS[variant], make_inits(variant, r0_med))
            v_hat = open_loop_from_truth(vec, held_s, held_i, ocv2d, held.dt, temp)
            e = v_hat - held_v
            r0k, r1, tau1, r2, tau2, r3, tau3, r0_cur, dv = _unpack_vec(vec)
            slow_mv, fast_mv = slow_fast_rmse(e, held.dt)
            rows.append({
                "dataset": args.dataset, "variant": variant,
                "temp_c": temp, "held_out": str(held.meta["profile"]),
                "n_val": len(e), "dt": held.dt,
                "train_rmse_mv": info["rmse_v"] * 1e3,
                "val_rmse_mv": float(np.sqrt(np.mean(e ** 2)) * 1e3),
                "val_bias_mv": float(np.mean(e) * 1e3),
                "val_p90_mv": float(np.quantile(np.abs(e), 0.9) * 1e3),
                "slow300_mv": slow_mv, "fast300_mv": fast_mv,
                "r0_01": float(r0k[0]), "r0_05": float(r0k[1]), "r0_09": float(r0k[2]),
                "r1": r1, "tau1": tau1, "r2": r2, "tau2": tau2,
                "r3": r3, "tau3": tau3, "r0_cur": r0_cur, "dv_mv": dv * 1e3,
                "vec_json": json.dumps([float(x) for x in vec]),
                "fit_s": time.time() - t0,
            })
            print(f"[{temp:>5.0f}°C] {held.meta['profile']:<5s} {variant:<8s} "
                  f"训练 {info['rmse_v']*1e3:5.1f} → 验证 {np.sqrt(np.mean(e**2))*1e3:5.1f} mV "
                  f"(bias {np.mean(e)*1e3:+6.1f})  慢 {slow_mv:5.1f} / 快 {fast_mv:5.1f}  "
                  f"τ3 {tau3:6.0f}  r0' {r0_cur*1e3:5.2f} mΩ/A  {time.time()-t0:4.0f}s")
    return pd.DataFrame(rows)


def run_docv(dataset: str, n_bins: int = 20) -> pd.DataFrame:
    """第二阶段：ΔOCV(SOC) 形状重标定（残差摸底的「慢残差 = OCV 形状误差」假设）。

    流程（每折独立，无泄漏）：用**训练折**的开环残差按 SOC 分箱取均值 →
    ΔOCV(soc) 插值曲线 → 从**留出折**的模型电压里减掉 → 看 LOO 残差变化。

    为什么用残差而不是重拟合 OCV：ΔOCV 是对既有 OCV 表的加性修正，
    滤波器侧只需 OCV 包装类，不用动结构。
    """
    fp = os.path.join(MET, f"model_upgrade_{dataset}_folds.parquet")
    if not os.path.exists(fp):
        raise SystemExit(f"缺 {fp}：先跑 --stage variants")
    up = pd.read_parquet(fp)
    with open(os.path.join(MET, f"model_upgrade_{dataset}.json"), encoding="utf-8") as f:
        best = json.load(f)["best"]
    s12 = _load_module("s12", os.path.join(ROOT, "scripts", "12_calibration_stress.py"))
    ocv2d, profiles, _base = (s12.load_calce() if dataset == "calce"
                              else s12.load_panasonic())
    params_of = {}
    for _, r in up[up.variant == best].iterrows():
        params_of[(float(r.temp_c), str(r.held_out))] = np.asarray(
            json.loads(r.vec_json), float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    rows, knot_rows = [], []
    for temp in sorted({float(p.meta["temp_c"]) for p in profiles}):
        sub = [p for p in profiles if float(p.meta["temp_c"]) == temp]
        for held in sub:
            key = (temp, str(held.meta["profile"]))
            vec = params_of.get(key)
            if vec is None:
                continue
            train = [p for p in sub if p is not held]
            xs, es = [], []
            for p in train:
                vh = open_loop_from_truth(vec, p.soc_ref, p.current_a, ocv2d, p.dt, temp)
                xs.append(p.soc_ref)
                es.append(vh - p.voltage_v)
            x, e = np.concatenate(xs), np.concatenate(es)
            idx = np.clip(np.digitize(x, bins) - 1, 0, n_bins - 1)
            d = np.array([e[idx == k].mean() if np.any(idx == k) else 0.0
                          for k in range(n_bins)])
            vh = open_loop_from_truth(vec, held.soc_ref, held.current_a, ocv2d,
                                      held.dt, temp)
            e0 = vh - held.voltage_v
            e1 = e0 - np.interp(held.soc_ref, centers, d)
            rows.append({
                "dataset": dataset, "variant": best, "temp_c": temp,
                "held_out": key[1], "n_val": len(e0),
                "before_rmse_mv": float(np.sqrt(np.mean(e0 ** 2)) * 1e3),
                "after_rmse_mv": float(np.sqrt(np.mean(e1 ** 2)) * 1e3),
                "before_bias_mv": float(np.mean(e0) * 1e3),
                "after_bias_mv": float(np.mean(e1) * 1e3),
                "docv_max_mv": float(np.max(np.abs(d)) * 1e3),
            })
            knot_rows.append({"temp_c": temp, "held_out": key[1],
                              "soc_centers": json.dumps([float(c) for c in centers]),
                              "docv_mv": json.dumps([float(v * 1e3) for v in d])})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("没有折可标定")
    df.to_parquet(os.path.join(MET, f"model_upgrade_{dataset}_docv_folds.parquet"),
                  index=False)
    pd.DataFrame(knot_rows).to_parquet(
        os.path.join(MET, f"model_upgrade_{dataset}_docv_knots.parquet"), index=False)
    summary = {
        "dataset": dataset, "best_variant": best,
        "before_median_mv": float(df.before_rmse_mv.median()),
        "after_median_mv": float(df.after_rmse_mv.median()),
        "gain_median_mv": float((df.before_rmse_mv - df.after_rmse_mv).median()),
        "after_p90_mv": float(df.after_rmse_mv.quantile(0.9)),
        "docv_max_median_mv": float(df.docv_max_mv.median()),
        "n_folds": int(len(df)),
    }
    with open(os.path.join(MET, f"model_upgrade_{dataset}_docv.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\n=== {dataset} ΔOCV 形状重标定（LOO，变体 {best}） ===")
    print(f"  RMSE 中位 {summary['before_median_mv']:.1f} → {summary['after_median_mv']:.1f} mV"
          f"（改善 {summary['gain_median_mv']:+.1f}，ΔOCV 幅度中位 "
          f"{summary['docv_max_median_mv']:.0f} mV）")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["calce", "panasonic"], required=True)
    ap.add_argument("--stage", choices=["variants", "docv"], default="variants")
    ap.add_argument("--variants", default="2rc,2rc_cur,3rc,3rc_cur")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 折（冒烟）")
    ap.add_argument("--dt", type=float, default=0.0, help="0 = 按数据集默认")
    args = ap.parse_args()

    if args.stage == "docv":
        run_docv(args.dataset)
        return

    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in args.variants:
        if v not in VARIANTS:
            raise SystemExit(f"未知变体 {v}，可选 {sorted(VARIANTS)}")
    if not args.dt:
        args.dt = CALCE_DT if args.dataset == "calce" else NCA_DT

    df = run(args)
    if df.empty:
        raise SystemExit("没有任何折跑完——查数据根目录（CALCE_UMD / PANASONIC_ROOT）")
    out_p = os.path.join(MET, f"model_upgrade_{args.dataset}_folds.parquet")
    df.to_parquet(out_p, index=False)

    summary = {"dataset": args.dataset, "dt": args.dt,
               "n_folds": int(df.held_out.nunique() * df.variant.nunique()),
               "by_variant": {}}
    for v, g in df.groupby("variant"):
        summary["by_variant"][v] = {
            "val_rmse_median_mv": float(g.val_rmse_mv.median()),
            "val_rmse_p90_mv": float(g.val_rmse_mv.quantile(0.9)),
            "val_bias_median_mv": float(g.val_bias_mv.median()),
            "slow300_median_mv": float(g.slow300_mv.median()),
            "fast300_median_mv": float(g.fast300_mv.median()),
            "tau2_at_bound": int((g.tau2 > 5900).sum()),
            "tau3_median_s": float(g.tau3.median()) if "3rc" in v else None,
            "r0_cur_median_ma": float(g.r0_cur.median() * 1e3) if "cur" in v else None,
        }
    # 最优变体（按验证 RMSE 中位）
    order = sorted(summary["by_variant"],
                   key=lambda k: summary["by_variant"][k]["val_rmse_median_mv"])
    summary["ranking"] = order
    summary["best"] = order[0]
    base = summary["by_variant"].get("2rc", {}).get("val_rmse_median_mv")
    if base:
        summary["improvement_pct"] = float(
            (base - summary["by_variant"][order[0]]["val_rmse_median_mv"]) / base * 100.0)
    with open(os.path.join(MET, f"model_upgrade_{args.dataset}.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    other = "panasonic" if args.dataset == "calce" else "calce"
    both = {}
    for ds in (args.dataset, other):
        fp = os.path.join(MET, f"model_upgrade_{ds}.json")
        if os.path.exists(fp):
            with open(fp, encoding="utf-8") as f:
                both[ds] = json.load(f)
    with open(os.path.join(MET, "model_upgrade_summary.json"), "w", encoding="utf-8") as f:
        json.dump(both, f, ensure_ascii=False, indent=1)

    print(f"\n=== {args.dataset} 变体对比（验证 RMSE 中位，mV） ===")
    for v in order:
        s = summary["by_variant"][v]
        print(f"  {v:<8s} {s['val_rmse_median_mv']:5.1f}  (P90 {s['val_rmse_p90_mv']:5.1f}, "
              f"慢 {s['slow300_median_mv']:4.1f}/快 {s['fast300_median_mv']:4.1f})")
    if base:
        print(f"最优 {order[0]}，相对基线改善 {summary['improvement_pct']:.1f}%")


if __name__ == "__main__":
    main()
