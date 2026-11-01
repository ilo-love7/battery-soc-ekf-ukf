#!/usr/bin/env python
"""35_round11_figures.py — 第十一轮图（v1.10，图内英文）。

1. robust_r7_tails.png   innovation/white 尾部统计（kurt、超限率、尾能份额）
2. robust_r7_replay.png  鲁棒扫描 q 收敛率 + 机制对照（剥尾/高斯化）
3. robust_r7_real.png    真实数据 q̂/SOC 基线 vs 鲁棒
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
TEMPS = [-20.0, -10.0, 0.0, 10.0, 25.0]
TLAB = ["-20", "-10", "0", "10", "25"]


def _load(name: str) -> dict:
    with open(os.path.join(MET, name), encoding="utf-8") as fh:
        return json.load(fh)


def fig_tails() -> None:
    tdf = pd.read_parquet(os.path.join(MET, "robust_r7_panasonic_tails.parquet"))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))

    ax = axes[0]
    for temp, lab, col in zip(TEMPS, TLAB, ["#4878CF", "#EE854A", "#6ACC64",
                                            "#D65F5F", "#9B6AC4"]):
        g = tdf[tdf.temp_c == temp]
        if g.empty:
            continue
        ax.scatter(g.inn_kurt, g.white_kurt, s=28, color=col, label=f"{lab} deg C",
                   zorder=3)
    lim = max(20.0, float(np.nanmax(tdf.inn_kurt)) * 1.1)
    ax.plot([0, lim], [0, lim], "k:", lw=0.8, label="y = x")
    ax.axhline(7.0, color="#6ACC64", ls="--", lw=0.9, label="kurt = 7 (round-9 line)")
    ax.set_xlabel("innovation kurtosis")
    ax.set_ylabel("white-component kurtosis")
    ax.set_title("(a) Heavy tails live in white")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(fontsize=6.5, loc="upper left")

    ax = axes[1]
    x = np.arange(len(TEMPS))
    inn3 = [float(tdf[tdf.temp_c == t].inn_frac_3s.median()) for t in TEMPS]
    wh3 = [float(tdf[tdf.temp_c == t].white_frac_3s.median()) for t in TEMPS]
    ax.bar(x - 0.18, np.array(inn3) * 100, 0.36, label="innovation", color="#4878CF")
    ax.bar(x + 0.18, np.array(wh3) * 100, 0.36, label="white", color="#EE854A")
    ax.axhline(0.27, color="k", ls=":", lw=0.9, label="Gaussian 0.27%")
    ax.set_xticks(x, TLAB)
    ax.set_xlabel("temperature (deg C)")
    ax.set_ylabel("fraction beyond 3 sigma (%)")
    ax.set_title("(b) Tail exceedance rate")
    ax.legend(fontsize=7)

    ax = axes[2]
    inn_e = [float(tdf[tdf.temp_c == t].inn_tail_energy_share.median()) for t in TEMPS]
    wh_e = [float(tdf[tdf.temp_c == t].white_tail_energy_share.median()) for t in TEMPS]
    ax.bar(x - 0.18, np.array(inn_e) * 100, 0.36, label="innovation", color="#4878CF")
    ax.bar(x + 0.18, np.array(wh_e) * 100, 0.36, label="white", color="#EE854A")
    ax.axhline(2.9, color="k", ls=":", lw=0.9, label="Gaussian 2.9%")
    ax.set_xticks(x, TLAB)
    ax.set_xlabel("temperature (deg C)")
    ax.set_ylabel("energy share of |z| >= 3 (%)")
    ax.set_title("(c) Tail energy share")
    ax.legend(fontsize=7)

    fig.tight_layout()
    out = os.path.join(FIG, "robust_r7_tails.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("saved", out)


def fig_replay() -> None:
    rdf = pd.read_parquet(os.path.join(MET, "robust_r7_panasonic_replay.parquet"))
    h32 = _load("robust_r7_panasonic_h32.json")
    h33 = _load("robust_r7_panasonic_h33.json")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))

    variants = ["base", "clip2", "clip3", "clip5", "huber2", "huber3", "huber5", "gate3"]
    labs = ["base", "clip2", "clip3", "clip5", "huber2", "huber3", "huber5", "gate3"]
    colors = ["#9B9B9B"] + ["#4878CF"] * 3 + ["#EE854A"] * 3 + ["#D65F5F"]

    ax = axes[0]
    x = np.arange(len(variants))
    full_q = []
    clean_q = []
    for v in variants:
        gf = rdf[(rdf.inject == "full") & (rdf.variant == v) & (rdf.scale == 1.0)]
        gc = rdf[(rdf.inject == "none") & (rdf.variant == v) & (rdf.scale == 1.0)]
        full_q.append(float(gf.q_ok.mean()) * 100)
        clean_q.append(float(gc.q_ok.mean()) * 100)
    ax.bar(x - 0.18, full_q, 0.36, label="full replay", color="#4878CF")
    ax.bar(x + 0.18, clean_q, 0.36, label="zero-residual control", color="#6ACC64")
    ax.axhline(30.0, color="#D65F5F", ls="--", lw=1.0, label="q_ok line 30%")
    ax.axhline(90.0, color="#6ACC64", ls=":", lw=1.0, label="clean line 90%")
    ax.set_xticks(x, labs, rotation=35, fontsize=7)
    ax.set_ylabel("q convergence rate (%)")
    ax.set_title("(a) Robust sweep on full replay")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=6.5, loc="upper right")

    ax = axes[1]
    cells = h33["cells"]
    names = ["base_full", "base_winsor5", "base_gauss"]
    labs2 = ["full\n(heavy tails)", "winsor 5%\n(same RMS)", "Gaussian white\n(same var)"]
    qabs = [cells[n]["q_rel_abs_med"] * 100 for n in names]
    cols = ["#D65F5F", "#EE854A", "#6ACC64"]
    xb = np.arange(3)
    ax.bar(xb, qabs, 0.55, color=cols)
    for xi, v in zip(xb, qabs):
        ax.text(xi, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
    d1 = h33["drop_winsor"] * 100
    d2 = h33["drop_gauss"] * 100
    ax.annotate(f"drop {d1:.0f}% (line 20%)", xy=(1, qabs[1]),
                xytext=(0.35, qabs[1] + 14), fontsize=7, color="#B85C00",
                arrowprops=dict(arrowstyle="->", color="#B85C00", lw=0.8))
    ax.annotate(f"drop {d2:.0f}% (line 40%)", xy=(2, qabs[2]),
                xytext=(1.15, qabs[2] + 6), fontsize=7, color="#2E7D32",
                arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=0.8))
    ax.set_xticks(xb, labs2, fontsize=7)
    ax.set_ylabel("median |q err| (%)")
    ax.set_title("(b) Tail shape is the poison (baseline filter)")
    ax.set_ylim(0, max(qabs) + 12)

    ax = axes[2]
    for inj, col, mk, lab in (("full", "#4878CF", "o", "full"),
                              ("full_winsor5", "#EE854A", "s", "winsor 5%"),
                              ("full_gauss", "#6ACC64", "^", "Gaussian white")):
        qs, ss = [], []
        for v in ("base", "clip3", "huber3"):
            g = rdf[(rdf.inject == inj) & (rdf.variant == v) & (rdf.scale == 1.0)]
            qs.append(float(g.q_ok.mean()) * 100)
            ss.append(float(g.rmse_soc.median()) * 100)
        ax.plot(qs, ss, mk + "-", color=col, label=lab, ms=6)
    ax.set_xlabel("q convergence rate (%)")
    ax.set_ylabel("SOC RMSE (pp)")
    ax.set_title("(c) q lift vs SOC cost")
    ax.legend(fontsize=7)

    fig.tight_layout()
    out = os.path.join(FIG, "robust_r7_replay.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("saved", out)


def fig_real() -> None:
    df = pd.read_parquet(os.path.join(MET, "robust_r7_panasonic_real.parquet"))
    h34 = _load("robust_r7_panasonic_h34.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3))

    ax = axes[0]
    variants = ["base", "clip3", "huber3"]
    cols = ["#9B9B9B", "#4878CF", "#EE854A"]
    x = np.arange(len(variants))
    qmed = [float(np.abs(df[df.variant == v].q_rel_err).median()) * 100
            for v in variants]
    qok = [float(df[df.variant == v].q_ok.mean()) * 100 for v in variants]
    ax.bar(x - 0.18, qmed, 0.36, label="median |q err| (%)", color="#4878CF")
    ax.bar(x + 0.18, qok, 0.36, label="q convergence (%)", color="#6ACC64")
    ax.set_xticks(x, variants)
    ax.set_ylabel("percent")
    ax.set_title("(a) Real folds: capacity estimate")
    ax.legend(fontsize=7)

    ax = axes[1]
    soc_b = (df[df.variant == "base"].groupby(["temp_c", "profile"])
             .rmse_soc.median().rename("base"))
    soc_r = (df[df.variant == h34["best"]].groupby(["temp_c", "profile"])
             .rmse_soc.median().rename("rob"))
    m = pd.concat([soc_b, soc_r], axis=1).dropna().reset_index()
    m = m.sort_values(["temp_c", "profile"])
    xx = np.arange(len(m))
    ax.bar(xx - 0.18, m.base * 100, 0.36, label="baseline", color="#9B9B9B")
    ax.bar(xx + 0.18, m.rob * 100, 0.36, label=h34["best"], color="#EE854A")
    ax.set_xticks(xx, [f"{r.temp_c:g}C\n{r.profile}" for _, r in m.iterrows()],
                  fontsize=5.5, rotation=90)
    ax.set_ylabel("SOC RMSE (pp)")
    ax.set_title(f"(b) Real folds: SOC cost (best = {h34['best']})")
    ax.legend(fontsize=7)

    fig.tight_layout()
    out = os.path.join(FIG, "robust_r7_real.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    fig_tails()
    fig_replay()
    fig_real()
