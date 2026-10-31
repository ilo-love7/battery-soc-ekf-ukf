"""容量误差：增长律验证 + 在线纠偏（EKFQ，容量作为第五状态）。

为什么要做这个（误差预算的首位）
--------------------------------
`scripts/12` 的标定容差测出：容量是唯一没有状态能吸收的误差源——
OCV ±40 mV ⇒ 0.04 pp（被偏差状态 b 吸收），容量 5% ⇒ 2–3 pp、10% ⇒ 5 pp。
本脚本回答两个问题：

1. **增长律对不对？** 纯积分下可解析推导：``err(t) = −(1/s − 1)·(1 − soc(t))``，
   即误差随放电深度**线性增长**、与化学体系无关。库仑计数应该精确复现这条直线，
   EKF 因为有电压反馈只能部分抵消——两者一起画出来，就能看出「电压反馈到底
   抵掉了多少容量误差」。
2. **把容量建成状态能不能纠回来？** ``EKFQ``（`bmsoc.filters`）把 q 作为第五状态
   联合估计。可辨识性取决于**电流激励的丰富度**（动态工况变号、变幅 ⇒ q 与 b 可分；
   恒流平台 ⇒ 共线，估计不动）。这与偏置/SOC 共线是同一个问题的两个侧面。

协议
----
- 场景固定 ``known``（初值准、电流无偏置），误差只可能来自注入的容量误差；
- 注入：滤波器初值 ``q0 = s·Q_true``，``s ∈ {0.90, 0.95, 1.00, 1.05, 1.10}``；
- 三版对照：cc（纯积分，验证公式）、ekf（Q 固定，v1.1 基线）、ekfq（Q 在线）；
- EKFQ 的 ``sigma_q0 = 0.08``（覆盖 ±10% 注入）、``q_q = 1e-10``（慢随机游走）。

预注册假设
----------
H3（已证）：容量误差挡不住，5% ⇒ 2 pp 量级，10% ⇒ 5 pp 量级。
H6：cc 的段末误差精确符合 ``−(1/s−1)(1−soc_end)``（±0.1 pp 内）。
H7：EKFQ 在**动态工况**上能把 q 收回到 ±2% 以内、SOC 误差显著低于固定 Q 的 EKF；
    但在激励弱的段上收不动——q 的收敛速度应与电流激励强度正相关。

用法：
    python scripts/15_capacity_q.py --dataset calce
    python scripts/15_capacity_q.py --dataset panasonic
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
from bmsoc.filters import CoulombCounter, EKF, EKFQ, FilterConfig  # noqa: E402

MET = os.path.join(ROOT, "results", "metrics")

CAP_SCALES = [0.90, 0.95, 1.00, 1.05, 1.10]
DSOC0 = 0.0
BIAS_A = 0.0
SIGMA_Q0 = 0.08
Q_Q = 1e-10


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_dataset(dataset: str):
    s12 = _load_module("s12", os.path.join(ROOT, "scripts", "12_calibration_stress.py"))
    return s12.load_calce() if dataset == "calce" else s12.load_panasonic()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["calce", "panasonic"], required=True)
    ap.add_argument("--scales", default=",".join(f"{s:g}" for s in CAP_SCALES))
    args = ap.parse_args()
    scales = [float(s) for s in args.scales.split(",")]

    ocv2d, profiles, fold_params = load_dataset(args.dataset)
    rows = []
    t_all = time.time()
    for p in profiles:
        temp = float(p.meta["temp_c"])
        prof = str(p.meta["profile"])
        params = fold_params(temp, prof)
        Q = p.capacity_ah
        # 激励强度：电流标准差与均值绝对值，用来解释 q 的收敛差异
        i_std = float(np.std(p.current_a))
        i_abs = float(np.mean(np.abs(p.current_a)))
        for s in scales:
            q_use = Q * s
            cfg = FilterConfig(soc0=float(np.clip(p.soc_ref[0] + DSOC0, 0, 1)),
                               sigma_soc0=0.05)
            for name, cls in (("cc", CoulombCounter), ("ekf", EKF), ("ekfq", EKFQ)):
                kw = dict(sigma_q0=SIGMA_Q0, q_q=Q_Q) if name == "ekfq" else {}
                est = cls(ocv=ocv2d, params=params, capacity_ah=q_use, cfg=cfg, **kw)
                out = est.run(p, current_bias_a=BIAS_A)
                m = met.soc_metrics(out["soc_hat"], p.soc_ref, p.t_s, tol=0.02)
                # 理论增长律：err_end = −(1/s − 1)·(1 − soc_end)
                theory_end = -(1.0 / s - 1.0) * (1.0 - float(p.soc_ref[-1]))
                q_end = float(out["q_hat"][-1]) if "q_hat" in out else float("nan")
                rows.append({
                    "estimator": name, "cap_scale": s, "temp_c": temp, "profile": prof,
                    "capacity_ah": Q, "q_used": q_use, "n": len(p),
                    "soc_end": float(p.soc_ref[-1]), "theory_end": theory_end,
                    "err_end": float(out["soc_hat"][-1] - p.soc_ref[-1]),
                    "q_end": q_end,
                    "q_rel_err_end": (q_end / Q - 1.0) if np.isfinite(q_end) else float("nan"),
                    "i_std": i_std, "i_abs": i_abs,
                    **{f"soc_{k}": v for k, v in m.items()},
                })
        print(f"[{temp:>5.0f}°C {prof:<5s}] i_std={i_std:4.2f} {time.time() - t_all:5.0f}s",
              flush=True)

    if not rows:
        raise SystemExit(f"没有任何 run：profiles={len(profiles)}。"
                         "先查数据根目录（CALCE_ROOT/PANASONIC_ROOT）")
    res = pd.DataFrame(rows)
    res.to_parquet(os.path.join(MET, f"capacity_q_{args.dataset}_runs.parquet"), index=False)

    agg = {}
    for (s, name), g in res.groupby(["cap_scale", "estimator"]):
        agg.setdefault(f"{s:g}", {})[name] = {
            "rmse_median": float(g.soc_rmse.median()),
            "rmse_p90": float(g.soc_rmse.quantile(0.9)),
            "tail_median": float(g.soc_tail_rmse.median()),
            "err_end_median": float(g.err_end.median()),
            "theory_end_median": float(g.theory_end.median()),
            "q_rel_err_end_median": float(g.q_rel_err_end.median()),
            "q_rel_err_end_p90": float(g.q_rel_err_end.quantile(0.9)),
            "n": int(len(g)),
        }
    # H7 的检验：q 收敛误差与电流激励的相关性（逐段，cap_scale=0.90/1.10）
    exc = []
    for s in (0.90, 1.10):
        g = res[(res.cap_scale == s) & (res.estimator == "ekfq")]
        if len(g) > 2:
            exc.append({"cap_scale": s,
                        "corr_i_std_qerr": float(np.corrcoef(g.i_std, g.q_rel_err_end.abs())[0, 1]),
                        "corr_i_abs_qerr": float(np.corrcoef(g.i_abs, g.q_rel_err_end.abs())[0, 1])})
    payload = {"dataset": args.dataset, "scenario": "known", "bias_a": BIAS_A,
               "sigma_q0": SIGMA_Q0, "q_q": Q_Q, "n_profiles": len(profiles),
               "grids": {"cap_scale": scales}, "by_scale": agg,
               "excitation_corr": exc}
    with open(os.path.join(MET, f"capacity_q_{args.dataset}.json"), "w",
              encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    print("\n=== 全程 RMSE 中位（pp）/ 段末误差（pp）/ q 末相对误差（%）===")
    print(f"  {'scale':>6s} {'theory':>8s}" + "".join(f"  {n:>22s}" for n in ("cc", "ekf", "ekfq")))
    for s in sorted(agg):
        line = f"  {s:>6s}"
        cc = agg[s].get("cc", {})
        line += f" {cc.get('theory_end_median', float('nan')) * 100:8.2f}"
        for name in ("cc", "ekf", "ekfq"):
            v = agg[s].get(name, {})
            qe = v.get("q_rel_err_end_median", float("nan")) * 100
            line += f"  {v.get('rmse_median', float('nan')) * 100:6.2f}/{v.get('err_end_median', float('nan')) * 100:6.2f}/{qe:6.2f}"
        print(line)
    for e in exc:
        print(f"  corr(i_std, |q_err|) s={e['cap_scale']}: {e['corr_i_std_qerr']:+.3f}")
    print(f"\n耗时 {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
