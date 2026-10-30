"""UKF 边界行为 A/B 对照：OCV 表「夹紧」vs「线性外推」。

用途：给报告 §8.2 提供同源对照（同一 OCV 表、同一折参数、同一噪声种子）。
输出 results/metrics/ukf_clip_ab.json。

    python scripts/07_ukf_clip_ab.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bmsoc import io_panasonic as io                       # noqa: E402
from bmsoc import metrics as met                            # noqa: E402
from bmsoc.filters import EKF, UKF, FilterConfig            # noqa: E402
from bmsoc.profiles import build_profile                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module                         # noqa: E402

m04 = import_module("04_benchmark_soc") if False else None
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASES = [
    # (temp_c, profile, scenario, d_soc0, bias_a)
    (-20.0, "nn", "known", 0.0, 0.0),
    (-20.0, "nn", "init30", -0.30, 0.0),
    (-20.0, "la92", "bias20ma", 0.0, 0.02),
    (0.0, "nn", "known", 0.0, 0.0),
]


def _load_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bench04", os.path.join(ROOT, "scripts", "04_benchmark_soc.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ClipOCV2D:
    """模拟修复前的行为：SOC 越界即夹成端点常数。"""

    def __init__(self, inner):
        self.inner = inner

    def __call__(self, soc, temp):
        s = np.asarray(soc, dtype=float)
        return self.inner(np.clip(s, 0.0, 1.0), temp)

    def docv(self, soc, temp):
        s = np.asarray(soc, dtype=float)
        return self.inner.docv(np.clip(s, 0.0, 1.0), temp)


def main() -> None:
    b = _load_module()
    ocv2d = b.load_ocv2d()
    clipped = ClipOCV2D(ocv2d)
    qcell = b.load_capacity()

    files = {(t, p): path for path, t, p in io.list_files()}
    print(f"{'case':<24s}{'clip,b=2':>10s}{'clip,b=0':>10s}{'ext,b=2':>10s}{'ext,b=0':>10s}{'ekf':>8s}")
    out = []
    for temp, prof, scen, dsoc, bias in CASES:
        path = files[(temp, prof)]
        df = io.to_frame(path)
        po = build_profile(df.t_s, df.current_a, df.voltage_v, df.temp_c, qcell[temp],
                           soc0=1.0, dt=1.0)
        po = po.trim_leading_rest(keep_s=300.0)
        params = b.fold_params(temp, prof)
        soc0 = float(np.clip(float(po.soc_ref[0]) + dsoc, 0.0, 1.0))
        cfg = FilterConfig(soc0=soc0, sigma_soc0=0.20 if dsoc else 0.05)

        r = {}
        for tag, ocv, beta in (("clip_b2", clipped, 2.0), ("clip_b0", clipped, 0.0),
                               ("ext_b2", ocv2d, 2.0), ("ext_b0", ocv2d, 0.0)):
            u = UKF(ocv=ocv, params=params, capacity_ah=po.capacity_ah, cfg=cfg, beta=beta)
            o = u.run(po, current_bias_a=bias)
            r[tag] = met.soc_metrics(o["soc_hat"], po.soc_ref, po.t_s, tol=0.02)["rmse"]
        e = EKF(ocv=ocv2d, params=params, capacity_ah=po.capacity_ah, cfg=cfg)
        oe = e.run(po, current_bias_a=bias)
        r["ekf"] = met.soc_metrics(oe["soc_hat"], po.soc_ref, po.t_s, tol=0.02)["rmse"]

        name = f"{temp:g}C {prof} {scen}"
        print(f"{name:<24s}" + "".join(f"{r[k]*100:10.2f}" for k in
              ("clip_b2", "clip_b0", "ext_b2", "ext_b0", "ekf")))
        out.append({"case": name, **{k: v * 100 for k, v in r.items()}})

    with open(os.path.join(ROOT, "results", "metrics", "ukf_clip_ab.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n→ results/metrics/ukf_clip_ab.json")


if __name__ == "__main__":
    main()
