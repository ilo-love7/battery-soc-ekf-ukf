"""中段上电压力测试：起点在平台区时，滤波器还剩多少本事。

为什么要做这个（关闭 v1.1 最大的边界）
--------------------------------------
v1.1 的四个场景**全部从满电（SOC=1.0）起步**，而 LFP 恰在 SOC>0.85 处最陡
（30 pp ↔ 285 mV），初值误差一条新息就被拉回。结论「LFP 反而更好」的适用范围
因此存疑 —— 它可能只是「起点选得好」。真正难的场景是**平台上电**：

- LFP 平台区 `dOCV/dSOC` 中位 0.17 V/unit、卡尔曼增益 0.0002（NCA 的 1/23）；
- 30 pp 的初值误差在平台上几乎不产生电压信号，要等走到陡坡才能纠正；
- 低温段（−10 °C）放电只到 SOC 0.31 就截止，**根本走不到底部陡坡**
  —— 预测这类段从平台起步的初值误差将全程残留。

协议
----
- 每段工况用 ``slice_from_soc`` 从真值 SOC ∈ {0.70, 0.50, 0.30} 处切开，
  只跑后半段（起点在平台上 = 「上电即在平台」）；
- 初值误差 ``dsoc ∈ {0, −0.30, +0.30}``（正负都测：平台两侧不对称，
  +0.30 指向顶部陡坡、−0.30 指向底部陡坡，收敛速度可能完全不同）；
- 估计器：``known`` 只跑 cc/ekf（与 04/11 的约定一致），``init30±`` 跑全四版；
- 指标与 04/11 逐字相同（`bmsoc.metrics.soc_metrics`）。

预注册假设
----------
H4：LFP 从 0.5 起步的 init30 误差将**远差于** v1.1 满电起步的 0.56–1.10 pp
    （预测 3 pp 以上，甚至全程不收敛）；NCA 退化幅度小于 LFP。
H5：不对称 —— 从 0.5 起步，−0.30（走向底部陡坡）比 +0.30（走向平台深处）
    **收敛更快**，因为剩余路径上有陡坡可借力。

用法：
    python scripts/14_mid_soc_stress.py --dataset calce
    python scripts/14_mid_soc_stress.py --dataset panasonic
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

from bmsoc import metrics as met                            # noqa: E402
from bmsoc.filters import ESTIMATORS, FilterConfig          # noqa: E402

MET = os.path.join(ROOT, "results", "metrics")

START_SOCS = [0.70, 0.50, 0.30]
DSOCS = {"known": 0.0, "init30n": -0.30, "init30p": +0.30}
MIN_N = 1200          # 切片后至少 20 min，否则指标没有统计意义
SLICE_TOL = 0.02


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_dataset(dataset: str):
    """复用 12 的加载器（同一份 OCV 表 / 折参数 / profile），保证与 v1.1 同源。"""
    s12 = _load_module("s12", os.path.join(ROOT, "scripts", "12_calibration_stress.py"))
    return s12.load_calce() if dataset == "calce" else s12.load_panasonic()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["calce", "panasonic"], required=True)
    ap.add_argument("--starts", default=",".join(f"{s:g}" for s in START_SOCS))
    args = ap.parse_args()
    starts = [float(s) for s in args.starts.split(",")]

    ocv2d, profiles, fold_params = load_dataset(args.dataset)
    rows, skipped = [], []
    t_all = time.time()
    for p in profiles:
        temp = float(p.meta["temp_c"])
        prof = str(p.meta["profile"])
        params = fold_params(temp, prof)
        for s_start in starts:
            sub = p.slice_from_soc(s_start, tol=SLICE_TOL)
            if sub is None or len(sub) < MIN_N:
                skipped.append({"temp_c": temp, "profile": prof, "start": s_start,
                                "reason": "below_soc_end" if sub is None else "too_short",
                                "soc_end": float(p.soc_ref[-1]), "n": 0 if sub is None else len(sub)})
                continue
            soc_true0 = float(sub.soc_ref[0])
            for sname, dsoc in DSOCS.items():
                soc0 = float(np.clip(soc_true0 + dsoc, 0.0, 1.0))
                cfg = FilterConfig(soc0=soc0, sigma_soc0=0.20 if dsoc else 0.05)
                for name, cls in ESTIMATORS.items():
                    if sname == "known" and name in ("ukf", "aekf"):
                        continue          # 与 04/11 同约定：known 不跑 UKF/AEKF
                    est = cls(ocv=ocv2d, params=params, capacity_ah=p.capacity_ah, cfg=cfg)
                    out = est.run(sub, current_bias_a=0.0)
                    m = met.soc_metrics(out["soc_hat"], sub.soc_ref, sub.t_s, tol=0.02)
                    rows.append({"estimator": name, "scenario": sname, "start_soc": s_start,
                                 "temp_c": temp, "profile": prof, "n": len(sub),
                                 "soc_true0": soc_true0, "soc0": soc0,
                                 "soc_end": float(sub.soc_ref[-1]),
                                 "recovered_end": float(abs(out["soc_hat"][-1] - sub.soc_ref[-1]) < 0.05),
                                 **{f"soc_{k}": v for k, v in m.items()}})
            print(f"[{temp:>5.0f}°C {prof:<5s}] start={s_start:.2f} n={len(sub):5d} "
                  f"{time.time() - t_all:5.0f}s", flush=True)

    if not rows:
        raise SystemExit(f"没有任何 run：profiles={len(profiles)}，"
                         f"skipped={len(skipped)}。先查数据根目录（CALCE_ROOT/PANASONIC_ROOT）")
    res = pd.DataFrame(rows)
    res.to_parquet(os.path.join(MET, f"mid_soc_{args.dataset}_runs.parquet"), index=False)
    pd.DataFrame(skipped).to_parquet(
        os.path.join(MET, f"mid_soc_{args.dataset}_skipped.parquet"), index=False)

    agg = {}
    for (s_start, sname), g in res.groupby(["start_soc", "scenario"]):
        agg.setdefault(f"{s_start:g}", {})[sname] = met.summarise_runs(g.to_dict("records"))
        agg[f"{s_start:g}"][sname] = {**agg[f"{s_start:g}"][sname]}
        for name, gg in g.groupby("estimator"):
            agg[f"{s_start:g}"][sname][name]["recovered_end_frac"] = float(gg.recovered_end.mean())
    payload = {"dataset": args.dataset, "config": {"starts": starts, "dsoCs": DSOCS,
                                                   "n_profiles": len(profiles),
                                                   "min_n": MIN_N, "slice_tol": SLICE_TOL},
               "by_start": agg, "skipped": skipped}
    with open(os.path.join(MET, f"mid_soc_{args.dataset}.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1, default=str)

    print("\n=== 全程 RMSE 中位（pp）/ 末点进 5 pp 带内占比 ===")
    for s_start in sorted(agg):
        for sname, s in agg[s_start].items():
            line = f"  start={s_start} {sname:<9s}"
            for name in ("cc", "ekf", "ukf", "aekf"):
                if name in s:
                    v = s[name]
                    line += (f"  {name}:{v['rmse_median']*100:5.2f}"
                             f"/{v.get('recovered_end_frac', float('nan')):4.2f}")
            print(line)
    print(f"\n跳过 {len(skipped)} 条（SOC 未到起点 / 切片过短），耗时 {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
