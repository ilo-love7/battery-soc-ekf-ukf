"""敏感性扫描：初值误差 / 电流偏置 / 电压噪声。

基准里每个场景只取一个点（−30 pp、0.02 A），看不出「多坏才是坏」。这里沿三个
轴各扫一遍，回答三个工程问题：

1. 初值偏差多大以内还能被电压拉回来（EKF/UKF/AEKF 各自的收敛边界）？
2. 电流偏置多大以内不靠电压补偿也能忍（库仑计数的可用区间）？
3. 量测噪声把精度压到什么程度（滤波器是噪声放大器还是抑制器）？

只取 6 条代表性工况（覆盖 −20 ~ 25 °C），成本可控。

用法：
    python scripts/05_sensitivity.py [--dt 1] [--out results/metrics/soc_sensitivity.parquet]
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bmsoc import io_panasonic as io                       # noqa: E402
from bmsoc import metrics as met                            # noqa: E402
from bmsoc.filters import ESTIMATORS, FilterConfig          # noqa: E402
from bmsoc.profiles import build_profile                    # noqa: E402

import importlib.util                                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET = os.path.join(ROOT, "results", "metrics")

#: 温度 → 工况，覆盖 5 个温度档
DEFAULT_CASES = [(-20.0, "la92"), (-10.0, "la92"), (0.0, "la92"),
                 (10.0, "udds"), (25.0, "us06"), (25.0, "udds")]

DSOC_GRID = [-0.40, -0.30, -0.20, -0.10, -0.05, 0.0]
# 注：网格本来还有 +5/+10/+20/+30 pp，但这批工况全部从满电起步（soc_ref[0] = 1.0），
# 正向误差会被 clip 压成同一个状态，跑出来是重复的假数据，所以删掉（见 main 里的跳过逻辑）。
BIAS_GRID = [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05]
NOISE_GRID = [0.0, 0.005, 0.010, 0.020, 0.030]

_SKIPPED: set = set()


def _bench_module():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_benchmark_soc.py")
    spec = importlib.util.spec_from_file_location("bench4", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--keep-rest", type=float, default=300.0)
    ap.add_argument("--out", default=os.path.join(MET, "soc_sensitivity.parquet"))
    args = ap.parse_args()

    b4 = _bench_module()
    ocv2d = b4.load_ocv2d()
    qcell = b4.load_capacity()

    items = []
    for temp, prof in DEFAULT_CASES:
        hits = [(p, t, pr) for p, t, pr in io.list_files() if t == temp and pr == prof]
        if not hits:
            print(f"跳过 {temp:g}°C/{prof}（无数据）")
            continue
        items.append(hits[0])

    axes = []
    for v in DSOC_GRID:
        axes.append(("dsoc0", v, {"dsoc0": v, "bias": 0.0, "noise": 0.0}))
    for v in BIAS_GRID:
        axes.append(("current_bias_a", v, {"dsoc0": 0.0, "bias": v, "noise": 0.0}))
    for v in NOISE_GRID:
        axes.append(("sigma_v", v, {"dsoc0": 0.0, "bias": 0.0, "noise": v}))
    # 电压噪声对库仑计数无影响，只跑一次，省 1/5 的算力
    skip_cc = {("sigma_v", v) for v in NOISE_GRID if v > 0}

    rows = []
    t_all = time.time()
    for path, temp, prof in items:
        df = io.to_frame(path)
        po = build_profile(df.t_s, df.current_a, df.voltage_v, df.temp_c, qcell[temp],
                           soc0=1.0, dt=args.dt).trim_leading_rest(keep_s=args.keep_rest)
        par = b4.fold_params(temp, prof)
        soc_true0 = float(po.soc_ref[0])
        for axis, value, cfgd in axes:
            soc0 = float(np.clip(soc_true0 + cfgd["dsoc0"], 0.0, 1.0))
            # 这批工况都从满电（soc_ref[0] = 1.0）起步，所以「估计值比真值高」是物理上
            # 不可达的：clip 会把 +5/+10/+20/+30 pp 全压成同一个 1.0，跑出来是四行
            # 完全重复的假数据。跳过，并在日志里说明一次。
            if abs(soc0 - (soc_true0 + cfgd["dsoc0"])) > 1e-9:
                if value not in _SKIPPED:
                    _SKIPPED.add(value)
                    print(f"跳过 dsoc0={value:+.2f} pp：soc_ref[0]={soc_true0:.3f}，"
                          f"正向初值误差会撞 1.0 限幅")
                continue
            cfg = FilterConfig(soc0=soc0, sigma_soc0=0.20 if cfgd["dsoc0"] else 0.05)
            for name, cls in ESTIMATORS.items():
                if name == "cc" and (axis, value) in skip_cc:
                    continue
                est = cls(ocv=ocv2d, params=par, capacity_ah=po.capacity_ah, cfg=cfg)
                out = est.run(po, voltage_noise=cfgd["noise"], seed=7,
                              current_bias_a=cfgd["bias"])
                m = met.soc_metrics(out["soc_hat"], po.soc_ref, po.t_s, tol=0.02)
                rows.append({"axis": axis, "value": float(value), "estimator": name,
                             "temp_c": temp, "profile": prof, "n": len(po),
                             "dsoc0": cfgd["dsoc0"], "current_bias_a": cfgd["bias"],
                             "sigma_v": cfgd["noise"],
                             **{f"soc_{k}": v for k, v in m.items()}})
        print(f"[{temp:>5.0f}°C {prof:<5s}] n={len(po):5d} 完成（累计 {time.time()-t_all:.0f}s）")

    res = pd.DataFrame(rows)
    res.to_parquet(args.out, index=False)

    agg = {}
    for (axis, value, est), g in res.groupby(["axis", "value", "estimator"]):
        agg.setdefault(axis, {}).setdefault(f"{value:g}", {})[est] = {
            "rmse_median": float(g["soc_rmse"].median()),
            "tail_rmse_median": float(g["soc_tail_rmse"].median()),
            "max_abs_p90": float(g["soc_max_abs"].quantile(0.9)),
            "converge5_median_s": float(g["soc_converge5_s"].median()),
            "n": int(len(g)),
        }
    payload = {
        "config": {"dt": args.dt, "keep_rest_s": args.keep_rest,
                   "cases": [{"temp_c": t, "profile": p} for _, t, p in items],
                   "grids": {"dsoc0": DSOC_GRID, "current_bias_a": BIAS_GRID,
                             "sigma_v": NOISE_GRID}},
        "aggregate": agg,
    }
    with open(os.path.join(MET, "soc_sensitivity.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    for axis in ("dsoc0", "current_bias_a", "sigma_v"):
        if axis not in agg:
            continue
        unit = {"dsoc0": "pp", "current_bias_a": "A", "sigma_v": "mV"}[axis]
        mult = {"dsoc0": 100.0, "current_bias_a": 1.0, "sigma_v": 1000.0}[axis]
        print(f"\n=== {axis} 扫描（全程 RMSE 中位数，单位 pp） ===")
        print(f"{axis + '(' + unit + ')':>18s} " + " ".join(f"{e:>8s}" for e in ESTIMATORS))
        for v in sorted(agg[axis], key=float):
            cells = []
            for e in ESTIMATORS:
                got = agg[axis][v].get(e)
                cells.append(f"{got['rmse_median']*100:8.2f}" if got else "       -")
            print(f"{float(v)*mult:18.3g} " + " ".join(cells))
    print(f"\n耗时 {time.time()-t_all:.0f}s → {args.out}")


if __name__ == "__main__":
    main()
