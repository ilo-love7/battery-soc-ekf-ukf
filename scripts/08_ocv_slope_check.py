"""OCV 平滑前后的 dOCV/dSOC 对照（报告 §5.1 的数字来源）。

    python scripts/08_ocv_slope_check.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from bmsoc import io_panasonic as io      # noqa: E402

spec = importlib.util.spec_from_file_location("s02", os.path.join(ROOT, "scripts", "02_identify_ocv_ecm.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

files = {t: p for p, t in io.list_hppc()}
s = np.linspace(0.02, 0.98, 201)
print(f"{'T':>6s}{'npts':>6s}{'lim':>5s}{'min':>7s}{'max':>7s}{'ratio':>7s}")
for t in (-20.0, -10.0, 0.0, 10.0, 25.0):
    for sm, tag in ((0.0, "pre"), (1.5, "post")):
        r = m.identify_file(files[t], t, smooth_mv=sm)
        c = r["curve"]
        d = c.docv(s)
        ratio = d[1:] / np.maximum(d[:-1], 1e-9)
        rr = np.maximum(ratio, 1.0 / np.maximum(ratio, 1e-9))
        print(f"{t:6.0f}{r['n_ocv']:6d}{tag:>5s}{d.min():7.2f}{d.max():7.2f}"
              f"{float(np.median(rr)):7.1f}")
