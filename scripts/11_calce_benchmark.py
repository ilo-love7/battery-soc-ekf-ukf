"""CALCE A123（LFP）SOC 估计基准：库仑计数 / EKF / UKF / AEKF。

协议与 Panasonic 基准（``04_benchmark_soc.py``）**逐字相同**：
四个场景（初值误差 / 电流偏置）、同样的滤波器超参数、同样的指标定义，
唯一变量是化学体系与数据集。这样两边数字可以直接并排放。

工况：8 个温度 × DST / US06 / FUDS 三段，共 24 段；参数取「留一该段」折。
SOC 真值由电流积分给出（容量基准 = 该温度 C/20 容量，与 OCV 轴同源）。

用法：
    python scripts/11_calce_benchmark.py [--dt 1] [--limit 0]
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

from bmsoc import calce as ca                                    # noqa: E402
from bmsoc import metrics as met                                 # noqa: E402
from bmsoc import ocv as ocvmod                                  # noqa: E402
from bmsoc.ecm import RCParams                                   # noqa: E402
from bmsoc.filters import ESTIMATORS, FilterConfig               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET = os.path.join(ROOT, "results", "metrics")

SCENARIOS = {
    "known": (0.0, 0.0),
    "init30": (-0.30, 0.0),
    "bias20ma": (0.0, 0.02),
    "hard": (-0.30, 0.02),
}


def load_ocv2d():
    with open(os.path.join(MET, "calce_ocv.json"), encoding="utf-8") as f:
        raw = json.load(f)
    curves = {float(k): ocvmod.from_rest_points(
                  v["soc"], v["ocv"], temp_c=float(k), source="c/20",
                  smooth_mv=0.0 if v.get("smoothed") else 1.5)
              for k, v in raw.items()}
    return ocvmod.OCV2D(ocvmod.OCVTable(curves))


_FOLDS = None


def _folds():
    global _FOLDS
    if _FOLDS is None:
        _FOLDS = pd.read_parquet(os.path.join(MET, "calce_params.parquet"))
    return _FOLDS


def fold_params(temp: float, profile: str) -> RCParams:
    df = _folds()
    row = df[(df.temp_c == temp) & (df.held_out == profile)]
    if len(row) == 0:
        raise KeyError(f"没有 {temp}°C/{profile} 的折参数")
    r = row.iloc[0]
    return RCParams(r0=float(np.mean([r["r0_01"], r["r0_05"], r["r0_09"]])),
                    r1=float(r["r1"]), tau1=float(r["tau1"]),
                    r2=float(r["r2"]), tau2=float(r["tau2"]), dv=float(r["dv"]))


def summarise(res: pd.DataFrame, dt: float, n_profiles: int) -> dict:
    summary = {}
    for sname, g in res.groupby("scenario"):
        summary[sname] = met.summarise_runs(g.to_dict("records"))
    by_temp = {}
    for (sname, t), g in res.groupby(["scenario", "temp_c"]):
        by_temp.setdefault(sname, {})[f"{t:g}"] = met.summarise_runs(g.to_dict("records"))
    by_prof = {}
    for (sname, p), g in res.groupby(["scenario", "profile"]):
        by_prof.setdefault(sname, {})[p] = met.summarise_runs(g.to_dict("records"))
    scen = {s: {"d_soc0": SCENARIOS[s][0], "bias_a": SCENARIOS[s][1]}
            for s in summary if s in SCENARIOS}
    out = {"overall": summary, "by_temp": by_temp, "by_profile": by_prof,
           "config": {"dt": dt, "n_profiles": n_profiles, "keep_rest_s": 300.0,
                      "scenarios": scen, "dataset": "CALCE A123 (LFP)"}}
    with open(os.path.join(MET, "calce_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


def print_summary(summary: dict) -> None:
    print("\n=== 稳态（末 30%）/ 全程 RMSE 中位数，单位百分点 ===")
    for sname, s in summary.items():
        line = f"  {sname:<9s}"
        for name in ("cc", "ekf", "ukf", "aekf"):
            if name in s:
                line += (f"  {name}:尾{s[name]['tail_rmse_median']*100:5.2f}"
                         f"/全{s[name]['rmse_median']*100:5.2f}")
        print(line)
    print("\n=== 收敛到 5 pp 以内的中位时间（分钟）/ 60 s 内收敛占比 ===")
    for sname, s in summary.items():
        line = f"  {sname:<9s}"
        for name in ("cc", "ekf", "ukf", "aekf"):
            if name in s:
                v = s[name]["conv5_median_s"]
                line += (f"  {name}:{(v/60 if np.isfinite(v) else float('nan')):5.1f}"
                         f"/{s[name]['conv5_frac_60s']:4.2f}")
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--summarise-only", action="store_true")
    args = ap.parse_args()

    if args.summarise_only:
        res = pd.read_parquet(os.path.join(MET, "calce_benchmark_runs.parquet"))
        print_summary(summarise(res, args.dt, int(res[["temp_c", "profile"]]
                                                 .drop_duplicates().shape[0])))
        return

    ocv2d = load_ocv2d()
    profiles = ca.build_calce_profiles(dt=args.dt)
    if args.limit:
        profiles = profiles[:args.limit]
    scen = {k: SCENARIOS[k] for k in args.scenarios.split(",") if k in SCENARIOS}

    rows = []
    t_all = time.time()
    for p in profiles:
        temp = float(p.meta["temp_c"])
        prof = str(p.meta["profile"])
        params = fold_params(temp, prof)
        soc_true0 = float(p.soc_ref[0])
        parts = []
        for sname, (dsoc, bias) in scen.items():
            soc0 = float(np.clip(soc_true0 + dsoc, 0.0, 1.0))
            cfg = FilterConfig(soc0=soc0, sigma_soc0=0.20 if dsoc else 0.05)
            for name, cls in ESTIMATORS.items():
                if sname == "known" and name in ("ukf", "aekf"):
                    continue
                est = cls(ocv=ocv2d, params=params, capacity_ah=p.capacity_ah, cfg=cfg)
                out = est.run(p, current_bias_a=bias)
                m = met.soc_metrics(out["soc_hat"], p.soc_ref, p.t_s, tol=0.02)
                rows.append({"estimator": name, "scenario": sname, "temp_c": temp,
                             "profile": prof, "n": len(p),
                             "capacity_ah": p.capacity_ah, "soc0": soc0,
                             "soc_end": float(p.soc_ref[-1]),
                             "current_bias_a": bias,
                             **{f"soc_{k}": v for k, v in m.items()}})
                if sname == "hard":
                    parts.append(f"{name}:尾{m['tail_rmse']*100:4.1f}全{m['rmse']*100:4.1f}")
        print(f"[{temp:>5.0f}°C {prof:<5s}] n={len(p):5d} SOCH {p.soc_ref[0]:.2f}→{p.soc_ref[-1]:.2f}  "
              + "  ".join(parts))

    res = pd.DataFrame(rows)
    res.to_parquet(os.path.join(MET, "calce_benchmark_runs.parquet"), index=False)
    print_summary(summarise(res, args.dt, len(profiles))["overall"])
    print(f"\n耗时 {time.time() - t_all:.0f}s → calce_benchmark_runs.parquet / calce_benchmark.json")


if __name__ == "__main__":
    main()
