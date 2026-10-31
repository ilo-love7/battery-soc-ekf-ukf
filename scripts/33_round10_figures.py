#!/usr/bin/env python
"""33_round10_figures.py — 第十轮图（v1.9，图内英文）。

1. fastrc_r6_reduction.png   三路线 white 削减：逐折对照 / 削减份额 vs 登记线
2. fastrc_r6_replay.png      EKFQ 联动：q 收敛率与 q̂ 误差（注入变体）
3. fastrc_r6_align.png       v/i 对齐审计：δ̂ 分布 / 平移 RMSE 扫描 + 合成对照
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


def fig_reduction() -> None:
    rf = pd.read_parquet(os.path.join(MET, "fastrc_r6_panasonic_refit.parquet"))
    ff = pd.read_parquet(os.path.join(MET, "fastrc_r6_panasonic_fastfit.parquet"))
    h29 = _load("fastrc_r6_panasonic_h29.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))

    # (a) baseline vs 3rc_fast innovation white per fold
    ax = axes[0]
    cold = rf[rf.temp_c == -20.0].sort_values("profile")
    x = np.arange(len(cold))
    w = 0.36
    ax.bar(x - w / 2, cold.inn_white_mv, w, label="2rc baseline", color="#4878CF")
    ax.bar(x + w / 2, cold.innfast_white_mv, w, label="3rc_fast refit", color="#EE854A")
    ax.plot(x, cold.inns3_white_mv, "k^--", ms=5, label="3rc_slow control")
    ax.set_xticks(x, cold.profile.str.upper(), fontsize=8)
    ax.set_ylabel("innovation white RMS (mV)")
    ax.set_title("(a) Cold (-20 deg C): model-level refit absorbs ~0")
    ax.legend(fontsize=7, loc="upper left")

    # (b) cut fraction by route and temperature vs registered lines
    ax = axes[1]
    routes = [("cut_model", "model 3rc_fast", "#EE854A", "o"),
              ("cut_slow3", "model 3rc_slow", "#9B9B9B", "s"),
              ("cut_insample", "struct in-sample", "#6ACC65", "^"),
              ("cut_loo", "struct LOO-tau", "#4878CF", "D")]
    probe = _load("fastrc_r6_panasonic_probe.json")
    temps = [float(k) for k in probe["per_temp"]]
    for col, lab, color, mk in routes:
        vals = [probe["per_temp"][f"{t:g}"][col] for t in temps]
        ax.plot(temps, vals, mk + "-", color=color, label=lab)
    ax.axhline(h29["lines"]["L1_cut_model"], color="#EE854A", ls=":", lw=1)
    ax.axhline(h29["lines"]["L2_cut_loo"], color="#4878CF", ls=":", lw=1)
    ax.text(25.4, h29["lines"]["L1_cut_model"], "L1 0.25", fontsize=7,
            color="#EE854A", va="bottom")
    ax.text(25.4, h29["lines"]["L2_cut_loo"], "L2 0.15", fontsize=7,
            color="#4878CF", va="bottom")
    ax.set_xlabel("temperature (deg C)")
    ax.set_ylabel("white RMS cut fraction (median)")
    ax.set_title("(b) Registered lines: model 0% / struct 15-20%")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(-24, 29)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fastrc_r6_reduction.png"), dpi=150)
    plt.close(fig)


def fig_replay() -> None:
    h30 = _load("fastrc_r6_panasonic_h30.json")
    tbl = h30["table"]
    inj_order = ["none", "full", "fast_only", "minus_fast", "refit_full"]
    labs = ["none\n(0-resid)", "full\n(round-8)", "fast_only", "minus_fast",
            "refit_full"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))

    # (a) q_ok rate by injection
    ax = axes[0]
    x = np.arange(len(inj_order))
    vals = [tbl[j]["q_ok_rate"] for j in inj_order]
    lo = [tbl[j]["q_ok_rate_x0.5"] for j in inj_order]
    hi = [tbl[j]["q_ok_rate_x1.5"] for j in inj_order]
    colors = ["#6ACC65", "#D65F5F", "#EE854A", "#4878CF", "#9B9B9B"]
    ax.bar(x, vals, 0.55, color=colors)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.03, f"{v:.0%}", ha="center", fontsize=8)
    ax.plot(x, lo, "v", color="k", ms=5, label="scale 0.5")
    ax.plot(x, hi, "^", color="k", ms=5, label="scale 1.5")
    ax.axhline(h30["lines"]["L1_qok_recovery"], color="#D65F5F", ls=":", lw=1)
    ax.text(4.4, h30["lines"]["L1_qok_recovery"] + 0.02, "L1 0.30", fontsize=7,
            color="#D65F5F")
    ax.set_xticks(x, labs, fontsize=7.5)
    ax.set_ylabel("q convergence rate")
    ax.set_ylim(0, 1.12)
    ax.set_title("(a) Removing fast component recovers only 25%")
    ax.legend(fontsize=7, loc="upper right")

    # (b) median q_rel_err by injection
    ax = axes[1]
    qerrs = [100 * tbl[j]["q_rel_err_med"] for j in inj_order]
    ax.bar(x, qerrs, 0.55, color=colors)
    for xi, v in zip(x, qerrs):
        off = 1.5 if (v >= 0 or abs(v) < 1.0) else -4.5
        ax.text(xi, v + off, f"{v:+.1f}%", ha="center", fontsize=8)
    ax.axhline(2.0, color="k", ls="--", lw=1)
    ax.text(4.4, 3.5, "+/-2% ok band", fontsize=7)
    ax.set_xticks(x, labs, fontsize=7.5)
    ax.set_ylabel("median q_hat error (%)")
    ax.set_title("(b) Fast component poisons +3.4%; rest of white +18%")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fastrc_r6_replay.png"), dpi=150)
    plt.close(fig)


def fig_align() -> None:
    adf = pd.read_parquet(os.path.join(MET, "fastrc_r6_panasonic_align.parquet"))
    ldf = pd.read_parquet(os.path.join(MET, "fastrc_r6_panasonic_lag.parquet"))
    h31 = _load("fastrc_r6_panasonic_h31.json")
    ctrl = h31["align_controls"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))

    # (a) delta_hat per fold per temperature + controls
    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    for j, temp in enumerate(TEMPS):
        sub = adf[adf.temp_c == temp].sort_values("profile")
        if sub.empty:
            continue
        ax.plot(np.full(len(sub), temp), sub.delta_hat, "o",
                color=cmap(j / 4), ms=6,
                label=f"{TLAB[j]} deg C" if j == 0 else None)
        ax.plot([temp], [sub.delta_hat.median()], "k_", ms=14, mew=2)
    ax.axhspan(-1, 1, color="#6ACC65", alpha=0.12, label="|delta| < 1 sample")
    ax.axhline(0, color="k", lw=0.6)
    for tag, mark, col in (("delay2", "s", "#D65F5F"),
                           ("aligned", "D", "#4878CF"),
                           ("noise", "x", "#9B9B9B")):
        val = ctrl.get(tag, {}).get("delta_hat")
        if val is not None and np.isfinite(val):
            ax.plot([30.5], [val], mark, color=col, ms=7,
                    label=f"ctrl {tag} {val:+.2f}")
    ax.set_xlabel("temperature (deg C)   [right: synthetic controls]")
    ax.set_ylabel("lag estimate delta_hat (samples)")
    ax.set_title("(a) All folds sub-sample: no v/i channel delay")
    ax.set_xticks(TEMPS + [30.5], TLAB + ["ctrl"], fontsize=8)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.set_xlim(-25, 33)

    # (b) RMSE sweep curves (cold folds)
    ax = axes[1]
    for j, prof in enumerate(sorted(adf[adf.temp_c == -20.0].profile.unique())):
        g = ldf[(ldf.temp_c == -20.0) & (ldf.profile == prof)
                & (ldf["cov"] == "align_rmse")].sort_values("lag_s")
        if g.empty:
            continue
        r = g.rho.to_numpy(float)
        ax.plot(g.lag_s, 1000 * (r - r.min()), "o-", ms=3,
                label=prof.upper())
    ax.set_xlabel("lag k applied to current (samples)")
    ax.set_ylabel("RMSE(k) - min (mV)")
    ax.set_title("(b) Shift sweep: minima at k ~ 0 (cold folds)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fastrc_r6_align.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_reduction()
    fig_replay()
    fig_align()
    print("figures ->", FIG)
