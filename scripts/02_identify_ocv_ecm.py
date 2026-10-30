"""HPPC 五脉冲 → OCV(SOC) 曲线 + 二阶 RC 参数表。

输入：Panasonic 18650PF 的 5 个温度 HPPC 文件（每温度约 19 个 SOC 档 × 5 个脉冲倍率）。
输出：
- results/metrics/ocv_panasonic.json     每温度的 OCV 网格（SOC 0–1）
- results/metrics/rc_params.parquet      (温度, SOC) 上的 R0/R1/τ1/R2/τ2 与拟合误差
- results/metrics/hppc_pulses.parquet    逐脉冲明细（含倍率相关性与失败记录）
- results/metrics/identify_summary.json  汇总

方法要点：同一 SOC 档的 4–5 个倍率脉冲**联合**辨识（τ 走格点、R 用非负最小二乘），
而不是逐脉冲拟合再取中位数——后者在 1 s 采样、10 s 脉冲下不可辨识。

用法：
    python scripts/02_identify_ocv_ecm.py [--temps 25,10,0,-10,-20]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bmsoc import io_panasonic as io          # noqa: E402
from bmsoc import ocv as ocvmod               # noqa: E402
from bmsoc.ecm import identify_stack, model_voltage  # noqa: E402
from bmsoc.profiles import rest_segments      # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "metrics")

PULSE_MIN_S = 4.0
PULSE_MAX_S = 30.0
REST_FOR_OCV_S = 600.0
RELAX_AFTER_S = 1100.0
GROUP_DROP = 0.035       # 档内 5 个脉冲共约 2.2%，档间降 5–10%
MIN_I_FOR_FIT = 2.5      # 只用 ≥1C 的脉冲，0.5C 信号太弱


def find_pulses(t: np.ndarray, i: np.ndarray):
    """切出脉冲段（|I|>0.5 A 且持续 4–30 s）。"""
    mask = np.abs(i) > 0.5
    out = []
    n = mask.size
    k = 0
    while k < n:
        if not mask[k]:
            k += 1
            continue
        j = k
        while j + 1 < n and mask[j + 1]:
            j += 1
        dur = t[j] - t[k]
        if PULSE_MIN_S <= dur <= PULSE_MAX_S:
            out.append((k, j, dur, float(np.mean(i[k:j + 1]))))
        k = j + 1
    return out


def group_pulses(t, i, v, ah, q_cap, dt):
    """把脉冲按 SOC 档分组。

    档内 5 个脉冲自身只放掉约 2.2% SOC，档间实际降 5–10%，因此用
    「相对本档首个脉冲的 SOC 降幅 > GROUP_DROP」作为新档判据即可分开。
    """
    groups: List[Dict[str, object]] = []
    cur: List[Dict[str, object]] = []
    for (i0, i1, dur, i_mean) in find_pulses(t, i):
        a = max(0, i0 - 3)
        b = min(len(t), i1 + int(RELAX_AFTER_S / dt))
        soc = float(np.clip(1.0 + ah[i0] / q_cap, 0.0, 1.0))
        rec = {
            "i0": i0, "i1": i1, "dur": dur, "i_pulse": i_mean, "soc": soc,
            "ocv": float(v[a:i0].mean()) if i0 - a > 0 else float(v[i0]),
            "win": (a, min(b, len(t))),
        }
        if cur and (float(cur[0]["soc"]) - soc) > GROUP_DROP:
            groups.append({"pulses": cur})
            cur = []
        cur.append(rec)
    if cur:
        groups.append({"pulses": cur})
    return groups


def identify_file(path: str, temp_c: float, smooth_mv: float = 1.5) -> Dict[str, object]:
    d = io.load_meas(path)
    t, i, v, ah = d["t_s"], d["current_a"], d["voltage_v"], d["q_ah"]
    dt = float(np.median(np.diff(t)))
    q_cap = float(abs(np.nanmin(ah)))

    # ---- OCV：长静置段末电压 ----
    rests = rest_segments(t, i, thr=0.05, min_dur_s=REST_FOR_OCV_S)
    soc_pts, ocv_pts = [], []
    for i0, i1, _dur in rests:
        soc_pts.append(float(np.clip(1.0 + ah[i1] / q_cap, 0.0, 1.0)))
        ocv_pts.append(float(v[i1]))
    curve = ocvmod.from_rest_points(soc_pts, ocv_pts, temp_c=temp_c, smooth_mv=smooth_mv,
                                    source=f"hppc_rest:{os.path.basename(path)}")

    # ---- 参数：按 SOC 档联合辨识 ----
    groups = group_pulses(t, i, v, ah, q_cap, dt)
    grid_rows, pulse_rows = [], []
    for g in groups:
        pls = [p for p in g["pulses"] if abs(p["i_pulse"]) >= MIN_I_FOR_FIT]
        if len(pls) < 2:
            continue
        wins = [{"i": i[p["win"][0]:p["win"][1]], "v": v[p["win"][0]:p["win"][1]],
                 "ocv": p["ocv"], "dt": dt} for p in pls]
        try:
            p, info = identify_stack(wins)
        except Exception as e:  # 单档失败不影响整轮
            grid_rows.append({"soc": float(np.mean([q["soc"] for q in pls])), "ok": False, "err": str(e)})
            continue
        grid_rows.append({
            "soc": float(np.mean([q["soc"] for q in pls])), "ok": True, **p.to_dict(),
            "rmse_v": info["rmse_v"], "n_pulse": len(pls),
            "i_min": float(min(q["i_pulse"] for q in pls)),
            "i_max": float(max(q["i_pulse"] for q in pls)),
        })
        for q in pls:
            win_v = v[q["win"][0]:q["win"][1]]
            win_i = i[q["win"][0]:q["win"][1]]
            pred = model_voltage(p, q["ocv"], win_i, dt)
            pulse_rows.append({
                "temp_c": float(temp_c), "soc": q["soc"], "i_pulse": q["i_pulse"],
                "r0": p.r0, "r1": p.r1, "tau1": p.tau1, "r2": p.r2, "tau2": p.tau2,
                "rmse_v": float(np.sqrt(np.mean((pred - win_v) ** 2))), "dur_s": q["dur"],
            })

    grid = pd.DataFrame(grid_rows)
    ok = grid[grid["ok"]].copy() if "ok" in grid.columns else grid.copy()
    ok = ok.sort_values("soc").reset_index(drop=True)
    return {
        "temp_c": temp_c, "curve": curve, "grid": ok,
        "pulses": pd.DataFrame(pulse_rows), "q_cap": q_cap,
        "n_ocv": len(soc_pts), "source": os.path.basename(path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--temps", default="")
    ap.add_argument("--smooth-mv", type=float, default=1.5,
                    help="OCV 曲线平滑的残差目标（mV），0 = 关闭")
    args = ap.parse_args()
    want = [float(x) for x in args.temps.split(",") if x.strip()] if args.temps else None

    os.makedirs(OUT, exist_ok=True)
    files = io.list_hppc()
    if want:
        files = [f for f in files if f[1] in want]
    if not files:
        raise SystemExit("没有可用的 HPPC 文件")

    curves, grids, pulses, summary = {}, [], [], {}
    for path, temp in files:
        res = identify_file(path, temp, smooth_mv=args.smooth_mv)
        curves[float(temp)] = res["curve"]
        g = res["grid"].copy()
        g["temp_c"] = float(temp)
        grids.append(g)
        if len(res["pulses"]):
            pulses.append(res["pulses"])
        summary[f"{temp:g}"] = {
            "source": res["source"], "q_cap_ah": res["q_cap"], "n_ocv_points": res["n_ocv"],
            "ocv_violation_v": res["curve"].violation_v, "n_soc_levels": int(len(g)),
            "ocv_smooth_mv": float(args.smooth_mv),
            "ocv_smooth_residual_mv": float(getattr(res["curve"], "smooth_residual_v", 0.0) * 1e3),
            "params_rmse_v_median": float(g["rmse_v"].median()) if len(g) else None,
            "r0_median_mohm": float(g["r0"].median() * 1e3) if len(g) else None,
            "r1_median_mohm": float(g["r1"].median() * 1e3) if len(g) else None,
            "tau1_median_s": float(g["tau1"].median()) if len(g) else None,
            "r2_median_mohm": float(g["r2"].median() * 1e3) if len(g) else None,
            "tau2_median_s": float(g["tau2"].median()) if len(g) else None,
        }
        print(f"[{temp:>5.0f}°C] OCV {res['n_ocv']:2d} 点  档位 {len(g):2d}  "
              f"R0 {g['r0'].median()*1e3:5.2f} mΩ  τ1 {g['tau1'].median():5.1f}s  "
              f"τ2 {g['tau2'].median():6.0f}s  拟合 RMSE {g['rmse_v'].median()*1e3:4.1f} mV")

    with open(os.path.join(OUT, "ocv_panasonic.json"), "w", encoding="utf-8") as f:
        json.dump({f"{k:g}": v.to_dict() for k, v in curves.items()}, f, ensure_ascii=False, indent=1)

    grid_all = pd.concat(grids, ignore_index=True)
    grid_all.to_parquet(os.path.join(OUT, "rc_params.parquet"), index=False)
    if pulses:
        pd.concat(pulses, ignore_index=True).to_parquet(os.path.join(OUT, "hppc_pulses.parquet"), index=False)
    with open(os.path.join(OUT, "identify_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    print(f"\n写出: ocv_panasonic.json / rc_params.parquet ({len(grid_all)} 行) / hppc_pulses.parquet")


if __name__ == "__main__":
    main()
