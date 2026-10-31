"""CALCE A123（LFP）的 OCV 表与电气参数标定（留一工况）。

与 Panasonic 流水线（02/03）的差别
---------------------------------
1. **OCV 来源不同**：这里没有 HPPC，OCV 取自 C/20 慢放支路（21 h）。
   慢放本身有极化，所以曲线整体比真 OCV 低 20–40 mV，这个常数偏置由
   ``dv`` 显式吸收（它就是为此设计的），不让 SOC 去背。
2. **容量基准同源**：SOC 轴用该温度的 C/20 容量，动态段也用它归一化。
   −10 °C 时高倍率放电只放出 70% 就撞截止，SOC 终点约 0.31，这是物理事实。
3. **留一**：每温度只有 3 段工况，所以 3 折。

输出
----
- ``results/metrics/calce_ocv.json``      8 条 OCV 曲线
- ``results/metrics/calce_segments.parquet`` 24 段的元信息（时长、容量、SOC 终点）
- ``results/metrics/calce_params.parquet``   24 折的电气参数与开环残差
- ``results/metrics/calce_identify_summary.json``

用法：
    python scripts/10_calce_prepare.py [--dt 1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bmsoc import calce as ca                    # noqa: E402
from bmsoc import ocv as ocvmod                  # noqa: E402
from bmsoc.ecm import fit_cycle_params, open_loop_from_truth  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET = os.path.join(ROOT, "results", "metrics")
SMOOTH_MV = 1.5


def build_ocv():
    tab = ca.ocv_table()
    curves, payload = {}, {}
    for temp, (soc, ocv, q) in tab.items():
        c = ocvmod.from_rest_points(soc, ocv, temp_c=temp, source="c/20", smooth_mv=SMOOTH_MV)
        curves[temp] = c
        payload[f"{temp:g}"] = c.to_dict() | {"q_slow_ah": q}
    return curves, payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=1.0)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(MET, exist_ok=True)

    print("[1/3] OCV 表（C/20 慢放支路）")
    curves, payload = build_ocv()
    with open(os.path.join(MET, "calce_ocv.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    o2 = ocvmod.OCV2D(ocvmod.OCVTable(curves))
    for temp in sorted(curves):
        c = curves[temp]
        d = c.docv(np.linspace(0.05, 0.95, 181))
        print(f"    {temp:6.0f} °C  resid {c.smooth_residual_v * 1000:.2f} mV   "
              f"slope {d.min():.3f}–{d.max():.3f} (median {np.median(d):.3f})")

    print("[2/3] 24 段工况")
    profiles = ca.build_calce_profiles(dt=args.dt)
    rows = []
    for p in profiles:
        rows.append({"temp_c": float(p.meta["temp_c"]), "profile": p.meta["profile"],
                     "n": len(p), "dur_s": float(p.t_s[-1]),
                     "q_seg_ah": float(p.meta["q_seg_ah"]), "q_ref_ah": float(p.meta["q_ref_ah"]),
                     "soc_end": float(p.meta["soc_end"]),
                     "i_rms": float(np.sqrt(np.mean(p.current_a ** 2))),
                     "i_min": float(p.current_a.min()), "i_max": float(p.current_a.max()),
                     "v_min": float(p.voltage_v.min()), "v_max": float(p.voltage_v.max()),
                     "source": p.meta["source"]})
    pd.DataFrame(rows).to_parquet(os.path.join(MET, "calce_segments.parquet"), index=False)
    print(f"    {len(profiles)} 段，温度 {sorted({float(p.meta['temp_c']) for p in profiles})}")

    print("[3/3] 留一工况标定（每温度 3 折）")
    by_key = {(float(p.meta["temp_c"]), p.meta["profile"]): p for p in profiles}
    out = []
    for temp in sorted({k[0] for k in by_key}):
        for hold in ("dst", "us06", "fuds"):
            train = [by_key[(temp, k)] for k in ("dst", "us06", "fuds") if k != hold]
            win = [(p.soc_ref, p.current_a, p.voltage_v, p.dt, o2, temp) for p in train]
            vec, info = fit_cycle_params(win, dv_bound=0.15)
            ho = by_key[(temp, hold)]
            vp = open_loop_from_truth(vec, ho.soc_ref, ho.current_a, o2, ho.dt, temp)
            err = vp - ho.voltage_v
            out.append({"temp_c": temp, "held_out": hold,
                        "r0_01": float(vec[0]), "r0_05": float(vec[1]), "r0_09": float(vec[2]),
                        "r1": float(vec[3]), "tau1": float(vec[4]),
                        "r2": float(vec[5]), "tau2": float(vec[6]), "dv": float(vec[7]),
                        "fit_rmse_v": info["rmse_v"], "fit_bias_v": info["bias_v"],
                        "val_rmse_v": float(np.sqrt(np.mean(err ** 2))),
                        "val_bias_v": float(np.mean(err)),
                        "val_max_abs_v": float(np.max(np.abs(err))),
                        "n_train": int(sum(len(p) for p in train))})
            print(f"    {temp:6.0f} °C 留 {hold:5s}  拟合 {info['rmse_v'] * 1000:5.1f} mV   "
                  f"验证 {np.sqrt(np.mean(err ** 2)) * 1000:5.1f} mV（偏置 {np.mean(err) * 1000:+5.1f}）"
                  f"  dv {vec[7] * 1000:+5.1f} mV  τ2 {vec[6]:.0f} s")

    df = pd.DataFrame(out)
    df.to_parquet(os.path.join(MET, "calce_params.parquet"), index=False)
    summary = {
        "dataset": "CALCE A123 (LFP, 2.3 Ah nominal, heavily aged)",
        "ocv_source": "C/20 slow discharge branch",
        "ocv_smooth_mv": SMOOTH_MV,
        "dt": args.dt,
        "n_profiles": len(profiles),
        "temps": sorted({float(p.meta["temp_c"]) for p in profiles}),
        "val_rmse_median_mv": float(df.val_rmse_v.median() * 1000),
        "val_rmse_p90_mv": float(df.val_rmse_v.quantile(0.9) * 1000),
        "dv_median_mv": float(df.dv.median() * 1000),
        "tau2_at_bound": int((df.tau2 > 5999).sum()),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(MET, "calce_identify_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\n完成，{time.time() - t0:.0f} s。验证 RMSE 中位 {summary['val_rmse_median_mv']:.1f} mV、"
          f"P90 {summary['val_rmse_p90_mv']:.1f} mV；dv 中位 {summary['dv_median_mv']:+.1f} mV")


if __name__ == "__main__":
    main()
