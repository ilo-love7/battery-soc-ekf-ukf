#!/usr/bin/env python
"""29_round8_figures.py — 第八轮图（v1.7，图内英文）。

1. ekfq_r4_filtered_residual.png  滤波后残差：吸收 / 结构分解 / 折×阈值象限
2. ekfq_r4_replay.png             选择性注入：收敛率 / 幅度扫描 / q 漂移方向签名
3. ekfq_r4_verdicts.png           判决面板：充分性 / 可行区 / 真实复验
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET = os.path.join(ROOT, "results", "metrics")
FIG = os.path.join(ROOT, "docs", "figures")

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
TEMPS = ["-20", "-10", "0", "10", "25"]
TLAB = ["-20", "-10", "0", "10", "25"]


def _load(name: str) -> dict:
    with open(os.path.join(MET, name), encoding="utf-8") as fh:
        return json.load(fh)


def fig_filtered() -> None:
    probe = _load("ekfq_r4_panasonic_probe.json")
    st = pd.read_parquet(os.path.join(MET, "ekfq_r4_panasonic_structure.parquet"))
    st = st[st.tmode == "matched"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

    # (a) open-loop vs filtered RMSE by temperature
    ax = axes[0]
    ol = [probe["per_temp"][f"{t}_matched"]["rmse_ol_mv"] for t in TEMPS]
    inn = [probe["per_temp"][f"{t}_matched"]["rmse_inn_mv"] for t in TEMPS]
    x = np.arange(len(TEMPS))
    ax.bar(x - 0.18, ol, 0.36, color="#8c8c8c", label="open-loop e")
    ax.bar(x + 0.18, inn, 0.36, color="#4c72b0", label="filtered innovation")
    for j, (a, b) in enumerate(zip(ol, inn)):
        ax.text(j, max(a, b) + 2, f"-{100*(1-b/a):.0f}%", ha="center", fontsize=7)
    ax.set_xticks(x, TLAB)
    ax.set_xlabel("Temperature (deg C)")
    ax.set_ylabel("RMSE (mV)")
    ax.set_title("(a) State absorption: e -> innovation")
    ax.legend(fontsize=7)

    # (b) structure decomposition medians by temperature
    ax = axes[1]
    comps = ["slow_mv", "odd_mv", "even_mv", "white_mv"]
    colors = ["#55a868", "#c44e52", "#dd8452", "#4c72b0"]
    width = 0.2
    for k, (cname, col) in enumerate(zip(comps, colors)):
        vals = [probe["per_temp"][f"{t}_matched"][cname] for t in TEMPS]
        ax.bar(x + (k - 1.5) * width, vals, width, color=col, label=cname[:-3])
    ax.axhline(15, color="#c44e52", lw=0.9, ls="--")
    ax.axhline(30, color="#55a868", lw=0.9, ls=":")
    ax.set_xticks(x, TLAB)
    ax.set_xlabel("Temperature (deg C)")
    ax.set_ylabel("Component RMS (mV)")
    ax.set_title("(b) Filtered structure (dashed: v1.5 limits)")
    ax.legend(fontsize=7)

    # (c) per-fold odd vs white with new feasible zone
    ax = axes[2]
    for temp, mk, col in ((-20, "o", "#c44e52"), (-10, "s", "#dd8452"),
                          (0, "^", "#8172b3"), (10, "D", "#55a868"),
                          (25, "v", "#4c72b0")):
        sub = st[st.temp_c == temp]
        if not len(sub):
            continue
        ax.scatter(sub.odd_mv, sub.white_mv, marker=mk, color=col, s=42,
                   label=f"{temp} deg C")
    lim = 70
    ax.axhspan(0, 10, color="green", alpha=0.10)
    ax.axvspan(0, 10, color="green", alpha=0.10)
    ax.axhline(10, color="green", lw=0.9)
    ax.axvline(10, color="green", lw=0.9)
    ax.set_xlim(-2, lim)
    ax.set_ylim(-2, lim)
    ax.set_xlabel("odd component RMS (mV)")
    ax.set_ylabel("white component RMS (mV)")
    ax.set_title("(c) Folds vs filtered feasible zone (green)")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ekfq_r4_filtered_residual.png"), dpi=150)
    plt.close(fig)


def fig_replay() -> None:
    rp = pd.read_parquet(os.path.join(MET, "ekfq_r4_panasonic_replay.parquet"))
    q = pd.read_parquet(os.path.join(MET, "ekfq_r4_panasonic_ekfq.parquet"))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    order = ["none", "slow_only", "odd_only", "white_only", "minus_white",
             "minus_odd", "full"]

    # (a) convergence rate by injection (scale 1.0)
    ax = axes[0]
    rates = [rp[(rp.inject == inj) & (rp.scale == 1.0)].q_ok.mean() for inj in order]
    cols = ["#55a868", "#55a868", "#dd8452", "#c44e52", "#8172b3",
            "#937860", "#c44e52"]
    ax.bar(np.arange(len(order)), rates, 0.6, color=cols)
    ax.axhline(0.30, color="k", lw=0.9, ls="--")
    ax.axhline(0.70, color="k", lw=0.9, ls=":")
    ax.set_xticks(np.arange(len(order)),
                  ["none", "slow", "odd", "white", "-white", "-odd", "full"],
                  rotation=30)
    ax.set_ylabel("q convergence rate")
    ax.set_title("(a) Selective injection (scale 1.0)")
    ax.text(5.6, 0.33, "30% sufficiency", fontsize=7, ha="right")
    ax.text(5.6, 0.73, "70% necessity", fontsize=7, ha="right")

    # (b) scale sweep
    ax = axes[1]
    for inj, col, mk in (("full", "#c44e52", "o"), ("white_only", "#4c72b0", "s"),
                         ("odd_only", "#dd8452", "^")):
        sub = rp[rp.inject == inj].groupby("scale").q_ok.mean()
        ax.plot(sub.index, sub.values, mk + "-", color=col, label=inj)
    ax.axhline(0.30, color="k", lw=0.9, ls="--")
    ax.set_xticks([0.5, 1.0, 1.5])
    ax.set_xlabel("amplitude scale (x measured RMS)")
    ax.set_ylabel("q convergence rate")
    ax.set_title("(b) Amplitude sweep locates the boundary")
    ax.legend(fontsize=7)

    # (c) q drift direction signature
    ax = axes[2]
    s1 = rp[(rp.scale == 1.0)]
    folds = [(-20.0, "hwfet"), (-20.0, "la92"), (25.0, "nn"), (25.0, "udds")]
    x = np.arange(len(folds))
    width = 0.22
    for k, (inj, col) in enumerate((("odd_only", "#dd8452"),
                                    ("white_only", "#4c72b0"), ("full", "#c44e52"))):
        vals = []
        for temp, prof in folds:
            g = s1[(s1.temp_c == temp) & (s1.profile == prof) & (s1.inject == inj)]
            vals.append(float(g.q_rel_err.median()) * 100)
        ax.bar(x + (k - 1) * width, vals, width, color=col, label=inj)
    real = []
    for temp, prof in folds:
        g = q[(q.temp_c == temp) & (q.profile == prof)]
        real.append(float(g.q_rel_err.median()) * 100)
    ax.plot(x, real, "k*", ms=11, label="real data (median)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, [f"{t:g} {p}" for t, p in folds], rotation=20)
    ax.set_ylabel("q drift at end (%)")
    ax.set_title("(c) Direction: odd down, white up")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ekfq_r4_replay.png"), dpi=150)
    plt.close(fig)


def fig_verdicts() -> None:
    rp = pd.read_parquet(os.path.join(MET, "ekfq_r4_panasonic_replay.parquet"))
    st = pd.read_parquet(os.path.join(MET, "ekfq_r4_panasonic_structure.parquet"))
    st = st[st.tmode == "matched"]
    q = pd.read_parquet(os.path.join(MET, "ekfq_r4_panasonic_ekfq.parquet"))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

    # (a) sufficiency demo: clean vs full replay
    ax = axes[0]
    labels = ["clean plant", "full replay"]
    rates = [rp[(rp.inject == "none") & (rp.scale == 1.0)].q_ok.mean(),
             rp[(rp.inject == "full") & (rp.scale == 1.0)].q_ok.mean()]
    soc = [rp[(rp.inject == "none") & (rp.scale == 1.0)].rmse_soc.median() * 100,
           rp[(rp.inject == "full") & (rp.scale == 1.0)].rmse_soc.median() * 100]
    ax.bar([0, 1], rates, 0.5, color=["#55a868", "#c44e52"])
    ax.set_xticks([0, 1], labels)
    ax.set_ylabel("q convergence rate")
    ax.set_ylim(0, 1.15)
    for j, (r, s) in enumerate(zip(rates, soc)):
        ax.text(j, r + 0.05, f"{r:.0%}\nSOC {s:.1f} pp", ha="center", fontsize=8)
    ax.set_title("(a) Residual structure is sufficient")

    # (b) feasible zone map (white vs odd), all profiles all temps
    ax = axes[1]
    for temp, mk, col in ((-20, "o", "#c44e52"), (-10, "s", "#dd8452"),
                          (0, "^", "#8172b3"), (10, "D", "#55a868"),
                          (25, "v", "#4c72b0")):
        sub = st[st.temp_c == temp]
        if not len(sub):
            continue
        ax.scatter(sub.odd_mv, sub.white_mv, marker=mk, color=col, s=42,
                   label=f"{temp} deg C")
    ax.add_patch(plt.Rectangle((0, 0), 10, 10, color="green", alpha=0.15))
    ax.axhline(10, color="green", lw=0.9)
    ax.axvline(10, color="green", lw=0.9)
    ax.set_xlim(-2, 70)
    ax.set_ylim(-2, 70)
    ax.set_xlabel("odd RMS (mV)")
    ax.set_ylabel("white RMS (mV)")
    ax.set_title("(b) Feasible zone: no NCA fold enters")
    ax.legend(fontsize=7, loc="upper left")

    # (c) real EKFQ vs EKF SOC RMSE by temperature
    ax = axes[2]
    x = np.arange(len(TEMPS))
    ekf = []
    ekfq = []
    for t in TEMPS:
        sub = q[q.temp_c == float(t)]
        ekf.append(float(sub.rmse_ekf.median()) * 100)
        ekfq.append(float(sub.rmse_soc.median()) * 100)
    ax.bar(x - 0.18, ekf, 0.36, color="#4c72b0", label="EKF (fixed Q)")
    ax.bar(x + 0.18, ekfq, 0.36, color="#c44e52", label="EKFQ")
    ax.set_xticks(x, TLAB)
    ax.set_xlabel("Temperature (deg C)")
    ax.set_ylabel("SOC RMSE (pp)")
    ax.set_title("(c) Real-data re-verification: EKFQ still worse")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ekfq_r4_verdicts.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_filtered()
    fig_replay()
    fig_verdicts()
    print("figures written:", sorted(f for f in os.listdir(FIG) if f.startswith("ekfq_r4")))
