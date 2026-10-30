"""在各温度的动态工况上标定二阶 RC 电气参数（留一工况，LOO）。

为什么不在 HPPC 上标定就够
--------------------------
HPPC 是「10 s 脉冲 + 20 min 静置」，它看到的是短时激励下的弛豫；把它外推到
UDDS/US06 这种**连续半小时到数小时**的负载时，慢极化支路（R2、τ2）会被系统性放大
（实测开环电压偏置 −30 ~ −70 mV，US06 低 SOC 处到 −128 mV），滤波器会把这个偏置
当成 SOC 偏差去消化。因此：**OCV 用 HPPC 静置点，电气参数在动态工况上标定**。

协议：对每个温度，把该温度的全部工况轮留一个，用其余工况拟合，再在被留出的工况上
评估开环电压残差。报告里给出的 SOC 估计指标，用的都是「没见过该工况」的参数。

用法：
    python scripts/03_fit_ecm_cycle.py [--dt 2] [--stride 2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bmsoc import io_panasonic as io          # noqa: E402
from bmsoc import ocv as ocvmod               # noqa: E402
from bmsoc.ecm import fit_cycle_params, open_loop_from_truth  # noqa: E402
from bmsoc.profiles import build_profile      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET = os.path.join(ROOT, "results", "metrics")


def load_ocv2d():
    """OCV(SOC, T)：温度方向线性插值，供逐样本查表。"""
    with open(os.path.join(MET, "ocv_panasonic.json"), encoding="utf-8") as f:
        raw = json.load(f)
    # 落盘的曲线若已平滑过（smoothed=True），这里必须关掉平滑，否则会二次平滑
    curves = {float(k): ocvmod.from_rest_points(
                  v["soc"], v["ocv"], temp_c=float(k), source="hppc",
                  smooth_mv=0.0 if v.get("smoothed") else 1.5)
              for k, v in raw.items()}
    return ocvmod.OCV2D(ocvmod.OCVTable(curves))


def load_cell_capacity():
    """每温度的电芯容量基准：取 HPPC 一次完整放电的电量。

    这一基准同时用于 OCV 的 SOC 轴与真值 SOC，两者必须同源，否则会出现
    「模型电压系统性偏低」的假象（本仓库早期版本就踩过这个坑）。
    """
    with open(os.path.join(MET, "identify_summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    return {float(k): float(v["q_cap_ah"]) for k, v in s.items()}


def build_all(dt: float, qcell: dict, keep_rest_s: float = 300.0):
    out, seen = [], set()
    for path, temp, prof in io.list_files():
        if temp not in qcell:
            continue
        if (temp, prof) in seen:      # -20 °C 有两个 US06 文件，只留第一个
            continue
        seen.add((temp, prof))
        df = io.to_frame(path)
        p = build_profile(df.t_s, df.current_a, df.voltage_v, df.temp_c, qcell[temp],
                          soc0=1.0, dt=dt, meta={"profile": prof, "temp_c": temp,
                                                 "source": os.path.basename(path),
                                                 "q_test_ah": io.capacity_ah(df)})
        # 低温文件开头有 2 h 温箱静置，占样本数一半以上；只留 300 s 作为上电窗口
        p = p.trim_leading_rest(keep_s=keep_rest_s)
        p.meta["soc0"] = 1.0
        out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=2.0, help="标定用采样步长（秒）")
    ap.add_argument("--out", default=os.path.join(MET, "ecm_folds.parquet"))
    args = ap.parse_args()

    ocv2d = load_ocv2d()
    rc = pd.read_parquet(os.path.join(MET, "rc_params.parquet"))
    hppc_r0 = rc.groupby("temp_c")["r0"].median().to_dict()
    qcell = load_cell_capacity()
    profs = build_all(args.dt, qcell)
    print(f"载入 {len(profs)} 段工况，容量基准 { {k: round(v,3) for k,v in qcell.items()} }")

    rows = []
    scores = []
    for temp in sorted({p.meta["temp_c"] for p in profs}):
        sub = [p for p in profs if p.meta["temp_c"] == temp]
        r0_med = float(hppc_r0.get(temp, 0.03))
        inits = [
            np.array([r0_med, r0_med, r0_med, r0_med * 0.3, 10.0, r0_med * 1.5, 900.0, 0.0]),
            np.array([r0_med * 0.6, r0_med * 0.6, r0_med * 0.6, r0_med * 0.1, 5.0, r0_med * 0.5, 300.0, 0.0]),
            np.array([r0_med * 1.5, r0_med * 1.5, r0_med * 1.5, r0_med * 0.5, 25.0, r0_med * 3.0, 2000.0, 0.0]),
        ]
        for held in sub:
            train = [p for p in sub if p is not held]
            wins = [(p.soc_ref, p.current_a, p.voltage_v, p.dt, ocv2d, p.temp_c) for p in train]
            best = None
            for x0 in inits:
                v_try, info_try = fit_cycle_params(wins, x0=x0, dv_bound=0.15)
                if best is None or info_try["rmse_v"] < best[1]["rmse_v"]:
                    best = (v_try, info_try)
            vec, info = best
            val = open_loop_from_truth(vec, held.soc_ref, held.current_a, ocv2d, held.dt, held.temp_c)
            e = val - held.voltage_v
            row = {
                "temp_c": float(temp), "held_out": held.meta["profile"],
                "n_train": len(train), "n_val": len(held),
                "train_rmse_v": info["rmse_v"], "train_bias_v": info["bias_v"],
                "val_rmse_v": float(np.sqrt(np.mean(e ** 2))),
                "val_bias_v": float(np.mean(e)),
                "val_max_v": float(np.max(np.abs(e))),
                "dv": info["dv"],
                "r0_01": float(vec[0]), "r0_05": float(vec[1]), "r0_09": float(vec[2]),
                "r1": float(vec[3]), "tau1": float(vec[4]),
                "r2": float(vec[5]), "tau2": float(vec[6]),
            }
            rows.append(row)
            scores.append(row["val_rmse_v"])
            print(f"[{temp:>5.0f}°C] 留出 {held.meta['profile']:<5s} "
                  f"训练 {info['rmse_v']*1e3:5.1f} mV → 验证 {row['val_rmse_v']*1e3:5.1f} mV "
                  f"(bias {row['val_bias_v']*1e3:+6.1f} mV)  R0 {vec[1]*1e3:5.1f} mΩ  "
                  f"τ2 {vec[6]:5.0f}s  dv {info['dv']*1e3:+5.1f} mV")

    df = pd.DataFrame(rows)
    df.to_parquet(args.out, index=False)
    summary = {
        "n_folds": int(len(df)),
        "val_rmse_median_v": float(df["val_rmse_v"].median()),
        "val_rmse_p90_v": float(df["val_rmse_v"].quantile(0.9)),
        "val_bias_median_v": float(df["val_bias_v"].median()),
        "by_temp": {f"{t:g}": {"val_rmse_median_v": float(g["val_rmse_v"].median()),
                              "val_bias_median_v": float(g["val_bias_v"].median())}
                    for t, g in df.groupby("temp_c")},
    }
    with open(os.path.join(MET, "ecm_folds_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\n留一工况验证：RMSE 中位数 {summary['val_rmse_median_v']*1e3:.1f} mV，"
          f"P90 {summary['val_rmse_p90_v']*1e3:.1f} mV → {args.out}")


if __name__ == "__main__":
    main()
