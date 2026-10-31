"""跨化学体系（NCA vs LFP）的三张图。

1. ``lfp_ocv.png``         OCV 曲线与可观测性对比（同一张图上放 NCA 与 LFP）
2. ``lfp_benchmark.png``   四个场景下的 SOC RMSE，两个数据集并排
3. ``calibration_margin.png`` OCV 表偏置 / 容量误差的敏感度，两个数据集并排

用法：
    python scripts/13_calce_figures.py [--only ocv,bench,margin]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from bmsoc import ocv as ocvmod                              # noqa: E402
from bmsoc.plotting import (C_ACC, C_GRAY, C_MAIN, C_RED,     # noqa: E402
                            EST_COLOR, EST_LABEL, savefig)

OUT = os.path.join(ROOT, "docs", "figures")
MET = os.path.join(ROOT, "results", "metrics")

C_LFP = "#d1743a"
NCA_LABEL = "NCA (Panasonic NCR18650PF)"
LFP_LABEL = "LFP (CALCE A123)"


def _load_ocv(name: str, source: str):
    with open(os.path.join(MET, name), encoding="utf-8") as f:
        raw = json.load(f)
    return ocvmod.OCV2D(ocvmod.OCVTable(
        {float(k): ocvmod.from_rest_points(v["soc"], v["ocv"], temp_c=float(k),
                                           source=source,
                                           smooth_mv=0.0 if v.get("smoothed") else 1.5)
         for k, v in raw.items()}))


def _load_json(name: str):
    with open(os.path.join(MET, name), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. OCV 与可观测性
# ---------------------------------------------------------------------------
def fig_ocv():
    nca = _load_ocv("ocv_panasonic.json", "hppc")
    lfp = _load_ocv("calce_ocv.json", "c/20")
    pairs = [(nca, 25.0, NCA_LABEL, C_MAIN), (nca, -20.0, None, C_MAIN),
             (lfp, 25.0, LFP_LABEL, C_LFP), (lfp, -10.0, None, C_LFP)]
    soc = np.linspace(0.0, 1.0, 401)
    # 0.5 个百分点宽的窗口差分：与滤波器里 h=1e-4 的解析导数不同，
    # 这里要的是「局部斜率」，1e-4 会落在单个 PCHIP 区间里、显出三段样条本身的锯齿。
    h = 0.005

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    for i, (o, t, label, col) in enumerate(pairs):
        ls = "-" if i % 2 == 0 else "--"
        lw = 1.8 if i % 2 == 0 else 1.3
        lab = label or f"{t:g} °C"
        axes[0].plot(soc, o(soc, t), color=col, ls=ls, lw=lw, label=lab)
        d = (o(np.minimum(soc + h, 1.0), t) - o(np.maximum(soc - h, 0.0), t)) / (
            np.minimum(soc + h, 1.0) - np.maximum(soc - h, 0.0))
        axes[1].plot(soc, d, color=col, ls=ls, lw=lw, label=lab)
    axes[0].set_xlabel("SOC (–)")
    axes[0].set_ylabel("OCV (V)")
    axes[0].set_title("(a) OCV–SOC: two chemistries")
    axes[0].legend(fontsize=8, loc="lower right")

    axes[1].axhspan(0.0, 0.10, color=C_RED, alpha=0.10,
                    label="LFP plateau (0.01–0.10 V/unit)")
    axes[1].axhline(0.10, color=C_RED, ls=":", lw=1.0)
    fig.text(0.735, 0.012,
             "the 0.1-SOC ripple on the NCA curves is a residue of the 10 % HPPC step grid",
             fontsize=6.8, color="#888888")
    axes[1].set_ylim(0, 1.6)
    axes[1].set_xlim(0.04, 0.96)
    axes[1].set_xlabel("SOC (–)")
    axes[1].set_ylabel("dOCV/dSOC (V per unit SOC)")
    axes[1].set_title("(b) SOC observability: LFP is 4–20× weaker")
    axes[1].legend(fontsize=7.5, loc="upper center", ncol=2, framealpha=0.92,
                   handlelength=1.8, columnspacing=1.0)
    fig.suptitle("Same ECM, same filter — the difference is the slope, not the code",
                 fontsize=11)
    return savefig(fig, os.path.join(OUT, "lfp_ocv.png"))


# ---------------------------------------------------------------------------
# 2. 基准对比
# ---------------------------------------------------------------------------
def fig_bench():
    nca = _load_json("soc_benchmark.json")["overall"]
    lfp = _load_json("calce_benchmark.json")["overall"]
    scen = ["known", "bias20ma", "init30", "hard"]
    names = ["ekf", "ukf", "aekf"]
    xlab = ["known\n(0, 0 A)", "bias\n(0, +20 mA)", "init\n(−30 pp, 0 A)",
            "both\n(−30 pp, +20 mA)"]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    width = 0.24
    top = 0.0
    panels = [(axes[0], nca, "(a) NCA (24 profiles, −20…25 °C)"),
              (axes[1], lfp, "(b) LFP (24 segments, −10…50 °C)")]
    for ax, block, title in panels:
        ax.set_axisbelow(True)          # 网格画在柱子后面，否则会切出白线
        for i, sc in enumerate(scen):
            for j, nm in enumerate(names):
                if nm not in block.get(sc, {}):
                    continue
                v = block[sc][nm]["rmse_median"] * 100
                top = max(top, v)
                ax.bar(i + (j - 1) * width, v, width * 0.88,
                       color=EST_COLOR[nm], edgecolor="none", zorder=3)
                ax.text(i + (j - 1) * width, v + 0.07, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(range(len(scen)))
        ax.set_xticklabels(xlab, fontsize=8)
        ax.set_ylabel("SOC RMSE, full profile (pp)")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 4.3)             # 两侧同尺度，柱高才能直接比
    axes[0].legend(handles=[Patch(facecolor=EST_COLOR[n], edgecolor="none",
                                  label=EST_LABEL[n]) for n in names],
                   fontsize=8, ncol=3, loc="upper left")
    cc_n = [nca[s]["cc"]["rmse_median"] * 100 for s in scen if "cc" in nca.get(s, {})]
    cc_l = [lfp[s]["cc"]["rmse_median"] * 100 for s in scen if "cc" in lfp.get(s, {})]
    fig.text(0.5, 0.012,
             f"coulomb counting omitted (off the scale): {min(cc_n):.1f}–{max(cc_n):.1f} pp "
             f"(NCA) / {min(cc_l):.1f}–{max(cc_l):.1f} pp (LFP)",
             ha="center", fontsize=7.8, color="#777777")
    fig.suptitle("Identical protocol, identical hyper-parameters — the LFP column is not worse",
                 fontsize=10.5)
    return savefig(fig, os.path.join(OUT, "lfp_benchmark.png"))


# ---------------------------------------------------------------------------
# 3. 标定容差
# ---------------------------------------------------------------------------
def fig_margin():
    nca = _load_json("calib_stress_panasonic.json")["axes"]
    lfp = _load_json("calib_stress_calce.json")["axes"]

    panels = [("ocv_mv", "OCV table offset (mV)",
               "(a) constant OCV error: absorbed by the bias state"),
              ("ocv_shape", "OCV shape error amplitude (mV)",
               "(b) SOC-dependent error: still no response"),
              ("cap_scale", "assumed capacity / true",
               "(c) capacity error: the one that bites")]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    for ax, (axis, xlabel, title) in zip(axes, panels):
        for block, col, lab in ((nca, C_MAIN, NCA_LABEL), (lfp, C_LFP, LFP_LABEL)):
            xs, ys = [], []
            for k, v in block.get(axis, {}).items():
                xs.append(float(k))
                ys.append(v["rmse_median"] * 100)
            if not xs:
                continue
            o = np.argsort(xs)
            ax.plot(np.array(xs)[o], np.array(ys)[o], "o-", color=col, lw=1.5, ms=5,
                    label=lab)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=9.5)
        ax.grid(True, color="#f0f0f0", lw=0.7)
        if axis == "cap_scale":
            ax.set_ylabel("SOC RMSE (pp)", fontsize=9)
            ax.set_ylim(0, 8.4)
        elif axis == "ocv_mv":
            ax.set_ylabel("SOC RMSE (pp)", fontsize=9)
        ax.legend(fontsize=7.5)
    fig.suptitle("Calibration margin, scenario `known` — only the gain (capacity) error bites",
                 fontsize=10.5)
    return savefig(fig, os.path.join(OUT, "calibration_margin.png"))


FIGS = {"ocv": fig_ocv, "bench": fig_bench, "margin": fig_margin}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    want = [k.strip() for k in args.only.split(",") if k.strip()] or list(FIGS)
    for k in want:
        if k not in FIGS:
            raise SystemExit(f"未知图 {k}，可选 {sorted(FIGS)}")
        print("→", FIGS[k]())


if __name__ == "__main__":
    main()
