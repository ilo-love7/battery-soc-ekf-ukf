"""SOC 估计基准：库仑计数 / EKF / UKF / AEKF。

为什么不是「初值已知、电流完美」
-------------------------------
那样库仑计数会精确复现真值（真值本来就是同一路电流对同一容量积分），
滤波器只会往里面加噪声，基准变成循环论证。真实 BMS 的痛点是：

1. **上电时 SOC 未知**（本基准取 −30 pp，即 30 个百分点偏低）；
2. **电流传感器有偏置**（0.02 A ≈ 0.7% C，积分一小时就有 0.02 Ah 漂移）。

于是协议是：真值仍由原始电流积分（不掺假），估计器只能看到「带偏置的电流」和
「端电压」，靠电压把 SOC 拉回来。这才是 EKF/UKF/AEKF 的用武之地。

每条工况使用的电气参数来自 script 03 的**留一该工况**折（模型没见过这条工况）。
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

from bmsoc import io_panasonic as io                       # noqa: E402
from bmsoc import metrics as met                            # noqa: E402
from bmsoc import ocv as ocvmod                             # noqa: E402
from bmsoc.ecm import RCParams                              # noqa: E402
from bmsoc.filters import ESTIMATORS, FilterConfig          # noqa: E402
from bmsoc.profiles import DrivingProfile, build_profile    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET = os.path.join(ROOT, "results", "metrics")

SCENARIOS = {
    # 名称: (初始 SOC 误差, 电流偏置 A)
    "known": (0.0, 0.0),
    "init30": (-0.30, 0.0),
    "bias20ma": (0.0, 0.02),
    "hard": (-0.30, 0.02),
}


def load_ocv2d():
    with open(os.path.join(MET, "ocv_panasonic.json"), encoding="utf-8") as f:
        raw = json.load(f)
    # 落盘的曲线若已平滑过（smoothed=True），这里必须关掉平滑，否则会二次平滑
    curves = {float(k): ocvmod.from_rest_points(
                  v["soc"], v["ocv"], temp_c=float(k), source="hppc",
                  smooth_mv=0.0 if v.get("smoothed") else 1.5)
              for k, v in raw.items()}
    return ocvmod.OCV2D(ocvmod.OCVTable(curves))


def load_capacity():
    with open(os.path.join(MET, "identify_summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    return {float(k): float(v["q_cap_ah"]) for k, v in s.items()}


_FOLDS = None


def _folds():
    global _FOLDS
    if _FOLDS is None:
        _FOLDS = pd.read_parquet(os.path.join(MET, "ecm_folds.parquet"))
    return _FOLDS


def fold_params(temp: float, profile: str) -> RCParams:
    df = _folds()
    row = df[(df.temp_c == temp) & (df.held_out == profile)]
    if len(row) == 0:
        raise KeyError(f"没有 {temp}°C/{profile} 的折参数")
    r = row.iloc[0]
    r0 = float(np.mean([r["r0_01"], r["r0_05"], r["r0_09"]]))
    return RCParams(r0=r0, r1=float(r["r1"]), tau1=float(r["tau1"]),
                    r2=float(r["r2"]), tau2=float(r["tau2"]), dv=float(r["dv"]))


def _write_summary(res: pd.DataFrame, dt: float, n_profiles: int) -> None:
    """把逐条 run 的指标表聚合成报告用的 JSON（与是否刚跑过无关）。"""
    summary = {}
    for sname, g in res.groupby("scenario"):
        summary[sname] = met.summarise_runs(g.to_dict("records"))
    by_temp = {}
    for (sname, t), g in res.groupby(["scenario", "temp_c"]):
        by_temp.setdefault(sname, {})[f"{t:g}"] = met.summarise_runs(g.to_dict("records"))
    scen = {s: {"d_soc0": SCENARIOS[s][0], "bias_a": SCENARIOS[s][1]}
            for s in summary if s in SCENARIOS}
    out = {"overall": summary, "by_temp": by_temp,
           "config": {"dt": dt, "n_profiles": n_profiles,
                      "keep_rest_s": 300.0, "scenarios": scen}}
    with open(os.path.join(MET, "soc_benchmark.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    _print_summary(summary)


def _print_summary(summary: dict) -> None:
    print("\n=== 稳态（末 30%）/ 全程 RMSE 中位数，单位百分点 ===")
    for sname, s in summary.items():
        line = f"  {sname:<9s}"
        for name in ("cc", "ekf", "ukf", "aekf"):
            if name in s:
                line += (f"  {name}:尾{s[name]['tail_rmse_median']*100:5.2f}"
                         f"/全{s[name]['rmse_median']*100:5.2f}")
        print(line)
    print("\n=== 收敛到 5 个百分点以内的中位时间（分钟）/ 60 s 内收敛占比 ===")
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
    ap.add_argument("--keep-rest", type=float, default=300.0,
                    help="开头长静置只保留这么多秒（低温文件原始静置 2 h）")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--summarise-only", action="store_true",
                    help="不重跑，只用已有 soc_benchmark_runs.parquet 重算汇总")
    args = ap.parse_args()

    if args.summarise_only:
        res = pd.read_parquet(os.path.join(MET, "soc_benchmark_runs.parquet"))
        _write_summary(res, dt=args.dt, n_profiles=int(res[["temp_c", "profile"]]
                                                       .drop_duplicates().shape[0]))
        return

    ocv2d = load_ocv2d()
    qcell = load_capacity()
    seen, items = set(), []
    for path, temp, prof in io.list_files():
        if temp not in qcell or (temp, prof) in seen:
            continue
        seen.add((temp, prof))
        items.append((path, temp, prof))
    if args.limit:
        items = items[:args.limit]

    scen = {k: SCENARIOS[k] for k in args.scenarios.split(",") if k in SCENARIOS}
    rows = []
    t_all = time.time()
    for path, temp, prof in items:
        df = io.to_frame(path)
        prof_obj = build_profile(df.t_s, df.current_a, df.voltage_v, df.temp_c, qcell[temp],
                                 soc0=1.0, dt=args.dt,
                                 meta={"profile": prof, "temp_c": temp,
                                       "source": os.path.basename(path)})
        prof_obj = prof_obj.trim_leading_rest(keep_s=args.keep_rest)
        params = fold_params(temp, prof)
        soc_true0 = float(prof_obj.soc_ref[0])
        head = f"[{temp:>5.0f}°C {prof:<5s}] n={len(prof_obj):5d}"
        parts = []
        for sname, (dsoc, bias) in scen.items():
            soc0 = float(np.clip(soc_true0 + dsoc, 0.0, 1.0))
            cfg = FilterConfig(soc0=soc0, sigma_soc0=0.20 if dsoc else 0.05)
            for name, cls in ESTIMATORS.items():
                if sname == "known" and name in ("ukf", "aekf"):
                    continue          # 已知初值/无偏置只作一致性检查
                est = cls(ocv=ocv2d, params=params, capacity_ah=prof_obj.capacity_ah, cfg=cfg)
                out = est.run(prof_obj, current_bias_a=bias)
                m = met.soc_metrics(out["soc_hat"], prof_obj.soc_ref, prof_obj.t_s, tol=0.02)
                rows.append({"estimator": name, "scenario": sname, "temp_c": temp,
                             "profile": prof, "n": len(prof_obj),
                             "capacity_ah": prof_obj.capacity_ah, "soc0": soc0,
                             "current_bias_a": bias,
                             **{f"soc_{k}": v for k, v in m.items()}})
                if sname == "hard":
                    parts.append(f"{name}:尾{m['tail_rmse']*100:4.1f}全{m['rmse']*100:4.1f}")
        print(head + "  " + "  ".join(parts))

    res = pd.DataFrame(rows)
    res.to_parquet(os.path.join(MET, "soc_benchmark_runs.parquet"), index=False)
    _write_summary(res, dt=args.dt, n_profiles=len(items))
    print(f"\n耗时 {time.time()-t_all:.0f}s → soc_benchmark_runs.parquet / soc_benchmark.json")


if __name__ == "__main__":
    main()
