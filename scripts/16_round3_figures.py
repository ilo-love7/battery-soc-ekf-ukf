"""第三轮的两张图：中段上电退化 + 容量误差增长律与在线纠偏。

1. ``mid_soc.png``       起点 SOC ∈ {0.3, 0.5, 0.7} 时的 init30 误差（两数据集并排），
                         虚线是 v1.1 满电起步的同指标 —— 一眼看出「起点选得好」的红利有多大
2. ``capacity_q.png``    左：cc 段末误差 vs 解析增长律（对角线 = H6 成立）；
                         右：固定 Q 的 EKF vs EKFQ（Q 在线）—— EKFQ 在真实数据上更差

用法：
    python scripts/16_round3_figures.py [--only mid,cap]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib.pyplot as plt  # noqa: E402

from bmsoc.plotting import (C_GRAY, C_GREEN, C_MAIN, C_RED,  # noqa: E402
                            EST_COLOR, savefig)

OUT = os.path.join(ROOT, "docs", "figures")
MET = os.path.join(ROOT, "results", "metrics")

C_LFP = "#d1743a"
DS_COLOR = {"calce": C_LFP, "panasonic": C_MAIN}
DS_LABEL = {"calce": "LFP (CALCE A123)", "panasonic": "NCA (Panasonic NCR18650PF)"}
# v1.1 满电起步的 init30 EKF 中位（pp），作为「起点红利」的参照线
V11_INIT30 = {"calce": 0.99, "panasonic": 3.31}


def _load_json(name: str):
    with open(os.path.join(MET, name), encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. 中段上电：起点越靠平台，init30 越收不回来
# ---------------------------------------------------------------------------
def fig_mid():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    starts = [0.30, 0.50, 0.70]
    for ax, ds in zip(axes, ("calce", "panasonic")):
        data = _load_json(f"mid_soc_{ds}.json")["by_start"]
        for sname, style in (("init30n", "-"), ("init30p", "--")):
            for est in ("ekf", "ukf", "aekf"):
                ys = []
                for s in starts:
                    cell = data[f"{s:g}"].get(sname, {}).get(est)
                    ys.append(cell["rmse_median"] * 100 if cell else np.nan)
                ax.plot(starts, ys, style, marker="o", ms=4.5,
                        color=EST_COLOR[est], lw=1.6,
                        label=f"{est}  {sname}")
        v11 = V11_INIT30[ds]
        ax.axhline(v11, color=C_GRAY, ls=":", lw=1.4)
        ax.text(0.31, v11 + 0.6, f"v1.1 full-start init30 = {v11:.2f}",
                fontsize=8, color=C_GRAY)
        ax.set_title(DS_LABEL[ds])
        ax.set_xlabel("Start SOC (slice point)")
        ax.set_xticks(starts)
        ax.set_xlim(0.24, 0.76)
        ax.set_ylim(0, 36)
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("SOC RMSE median, full run (pp)")
    leg = axes[1].legend(loc="upper left", fontsize=8, ncol=1)
    leg.set_zorder(5)
    fig.text(0.5, -0.03,
             "init30n = initial SOC 30 pp too low; init30p = 30 pp too high. "
             "Each run starts where true SOC first reaches the start value.",
             ha="center", fontsize=8, color="#555555")
    return savefig(fig, os.path.join(OUT, "mid_soc.png"))


# ---------------------------------------------------------------------------
# 2. 容量：增长律（左）与在线纠偏失败（右）
# ---------------------------------------------------------------------------
def fig_cap():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax = axes[0]
    for ds in ("calce", "panasonic"):
        runs = _load_json(f"capacity_q_{ds}.json")["by_scale"]
        th, ob = [], []
        for s in sorted(runs, key=float):
            cc = runs[s]["cc"]
            th.append(cc["theory_end_median"] * 100)
            ob.append(cc["err_end_median"] * 100)
        ax.plot(th, ob, "o", ms=5, color=DS_COLOR[ds], label=DS_LABEL[ds])
    lim = (-11.5, 11.5)
    ax.plot(lim, lim, color=C_GRAY, ls="--", lw=1.2, label="y = x (analytic)")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("Predicted end error  $-(1/s-1)(1-SOC_{end})$  (pp)")
    ax.set_ylabel("Coulomb-counting end error (pp)")
    ax.set_title("Growth law: 5 capacity ratios x 24 segments")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    for ds in ("calce", "panasonic"):
        runs = _load_json(f"capacity_q_{ds}.json")["by_scale"]
        ss = sorted(runs, key=float)
        for est, style in (("ekf", "-"), ("ekfq", "--")):
            ys = [runs[s][est]["rmse_median"] * 100 for s in ss]
            ax.plot([float(s) for s in ss], ys, style, marker="o", ms=4.5,
                    color=DS_COLOR[ds], lw=1.6,
                    label=f"{DS_LABEL[ds]}  {est}")
    ax.axvline(1.0, color=C_GRAY, ls=":", lw=1.2)
    ax.set_xticks([0.90, 0.95, 1.00, 1.05, 1.10])
    ax.set_xlabel("Filter capacity / true capacity  s")
    ax.set_ylabel("SOC RMSE median, full run (pp)")
    ax.set_title("Online Q (ekfq) fails to recover; it degrades")
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=7.5)
    fig.text(0.5, -0.03,
             "Left: points on the diagonal => end error grows linearly with depth of "
             "discharge (H6). Right: ekfq (dashed) above ekf (solid) => adding Q as a "
             "5th state absorbs model residual into q on real data.",
             ha="center", fontsize=8, color="#555555")
    return savefig(fig, os.path.join(OUT, "capacity_q.png"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="mid,cap")
    args = ap.parse_args()
    made = []
    if "mid" in args.only:
        made.append(fig_mid())
    if "cap" in args.only:
        made.append(fig_cap())
    for p in made:
        print("saved", p)


if __name__ == "__main__":
    main()
