"""标定容差压力测试：OCV 表偏置 × 容量误差，两个数据集同一套协议。

为什么要做这个
--------------
初测发现 LFP（CALCE）在「OCV 表准、容量准」时并不比 NCA 难，甚至更好 ——
因为 LFP 在 SOC>0.85 处 OCV 很陡（30 pp ≈ 285 mV），初值误差一进场就被拉回。
真正的差别在**平台区**：`dOCV/dSOC` 只有 0.17 V/unit，是 NCA 的 1/5。

于是同一个 20 mV 的 OCV 表误差，在 NCA 上折算 0.5 pp，在 LFP 上是 1–10 pp。
本脚本把这件事量化：注入已知的 OCV 偏置与容量误差，看 SOC 误差怎么长。

协议
----
- 场景固定 ``known``（初值准、电流无偏置），这样测到的误差**只可能**来自注入的标定误差；
- 只跑 EKF（UKF/AEKF 对这两类系统误差无能为力，AEKF 只会把 R 放大当成噪声）；
- 两条轴：OCV 全表平移 ``[-40,-20,0,+20,+40] mV``、滤波器容量 ``Q·[0.90..1.10]``。

用法：
    python scripts/12_calibration_stress.py --dataset calce
    python scripts/12_calibration_stress.py --dataset panasonic
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

from bmsoc import metrics as met                        # noqa: E402
from bmsoc.filters import EKF, FilterConfig             # noqa: E402

MET = os.path.join(ROOT, "results", "metrics")

OCV_OFFSETS_MV = [-40.0, -20.0, 0.0, 20.0, 40.0]
CAP_SCALES = [0.90, 0.95, 1.00, 1.05, 1.10]
# SOC 相关（形状）误差：amp·sin(π·SOC)。与常值偏置正交，`b` 吸不掉。
OCV_SHAPE_MV = [0.0, 10.0, 20.0, 30.0, 40.0]
BIAS_A = 0.0
DSOC0 = 0.0


class ShapeOCV:
    """SOC 相关误差：``OCV + dv·sin(π·soc)``（两端为 0、中段最大）。

    这是真实世界里最常见的 OCV 表误差形态（老化改变平台形状、温度外推偏差），
    它与常值偏置正交，所以偏差状态 ``b`` 无法吸收 —— 这一点是本实验的关键。
    """

    def __init__(self, base, amp: float):
        self.base = base
        self.amp = float(amp)

    def _bump(self, soc):
        return self.amp * np.sin(np.pi * np.clip(np.asarray(soc, float), 0.0, 1.0))

    def __call__(self, soc, temp):
        return self.base(soc, temp) + self._bump(soc)

    def docv(self, soc, temp, h: float = 1e-4):
        s = np.clip(np.asarray(soc, float), 0.0, 1.0)
        return self.base.docv(soc, temp, h=h) + self.amp * np.pi * np.cos(np.pi * s)


class ShiftedOCV:
    """把 OCV 表整体平移 ``dv`` 伏（模拟标定偏差），斜率不变。"""

    def __init__(self, base, dv: float):
        self.base = base
        self.dv = float(dv)

    def __call__(self, soc, temp):
        return self.base(soc, temp) + self.dv

    def docv(self, soc, temp, h: float = 1e-4):
        return self.base.docv(soc, temp, h=h)


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_calce():
    b = _load_module("b11", os.path.join(ROOT, "scripts", "11_calce_benchmark.py"))
    from bmsoc import calce as ca

    return b.load_ocv2d(), ca.build_calce_profiles(dt=1.0), b.fold_params


def load_panasonic():
    b = _load_module("b04", os.path.join(ROOT, "scripts", "04_benchmark_soc.py"))
    from bmsoc import io_panasonic as io
    from bmsoc.profiles import build_profile

    qcell = b.load_capacity()
    seen, profiles = set(), []
    for path, temp, prof in io.list_files():
        if temp not in qcell or (temp, prof) in seen:
            continue
        seen.add((temp, prof))
        df = io.to_frame(path)
        p = build_profile(df.t_s, df.current_a, df.voltage_v, df.temp_c, qcell[temp],
                          soc0=1.0, dt=1.0,
                          meta={"profile": prof, "temp_c": temp,
                                "source": os.path.basename(path)})
        profiles.append(p.trim_leading_rest(keep_s=300.0))
    return b.load_ocv2d(), profiles, b.fold_params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["calce", "panasonic"], required=True)
    args = ap.parse_args()

    ocv2d, profiles, fold_params = (load_calce() if args.dataset == "calce"
                                    else load_panasonic())
    rows = []
    t_all = time.time()
    for p in profiles:
        temp = float(p.meta["temp_c"])
        prof = str(p.meta["profile"])
        params = fold_params(temp, prof)
        Q = p.capacity_ah
        for axis, values in (("ocv_mv", OCV_OFFSETS_MV), ("cap_scale", CAP_SCALES),
                             ("ocv_shape", OCV_SHAPE_MV)):
            for v in values:
                q_use = Q
                if axis == "ocv_mv":
                    ocv = ShiftedOCV(ocv2d, v / 1000.0) if v else ocv2d
                elif axis == "ocv_shape":
                    ocv = ShapeOCV(ocv2d, v / 1000.0) if v else ocv2d
                else:
                    ocv = ocv2d
                    q_use = Q * v
                cfg = FilterConfig(soc0=float(np.clip(p.soc_ref[0] + DSOC0, 0, 1)),
                                   sigma_soc0=0.05)
                out = EKF(ocv=ocv, params=params, capacity_ah=q_use, cfg=cfg).run(
                    p, current_bias_a=BIAS_A)
                m = met.soc_metrics(out["soc_hat"], p.soc_ref, p.t_s, tol=0.02)
                rows.append({"axis": axis, "value": v, "temp_c": temp, "profile": prof,
                             "estimator": "ekf", "capacity_ah": Q, "q_used": q_use,
                             "soc_rmse": m["rmse"], "soc_tail_rmse": m["tail_rmse"],
                             "soc_max_abs": m["max_abs"]})
        print(f"[{args.dataset} {temp:>5.0f}°C {prof:<5s}] done  {time.time() - t_all:.0f}s",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(MET, f"calib_stress_{args.dataset}.parquet"), index=False)
    agg = {}
    for (axis, value), g in df.groupby(["axis", "value"]):
        agg.setdefault(axis, {})[f"{value:g}"] = {
            "rmse_median": float(g.soc_rmse.median()),
            "rmse_p90": float(g.soc_rmse.quantile(0.9)),
            "tail_median": float(g.soc_tail_rmse.median()),
            "n": int(len(g)),
        }
    payload = {"dataset": args.dataset, "scenario": "known", "bias_a": BIAS_A,
               "n_profiles": len(profiles), "axes": agg,
               "grids": {"ocv_mv": OCV_OFFSETS_MV, "cap_scale": CAP_SCALES,
                         "ocv_shape": OCV_SHAPE_MV}}
    with open(os.path.join(MET, f"calib_stress_{args.dataset}.json"), "w",
              encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    for axis in ("ocv_mv", "ocv_shape", "cap_scale"):
        print(f"\n=== {axis} ===")
        for k, v in sorted(agg[axis].items(), key=lambda kv: float(kv[0])):
            print(f"  {float(k):+7.2f}  rmse {v['rmse_median'] * 100:6.2f} pp   "
                  f"p90 {v['rmse_p90'] * 100:6.2f}   tail {v['tail_median'] * 100:6.2f}")
    print(f"\n耗时 {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
