#!/usr/bin/env python
"""27_round7_figures.py — 第七轮图（v1.6，图内英文）。

1. floor_r3_decomposition.png   地板温度分层 + regime 分解 + A/B/C 归属
2. floor_r3_rest_regimes.png    静置读数：起始表差 / 尾段两估计器区间 / 合成对照
3. floor_r3_verdicts.png        H20/H21/H22 判决面板
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


def fig_decomposition() -> None:
    probe = _load("floor_r3_panasonic_probe.json")
    rg = pd.read_parquet(os.path.join(MET, "floor_r3_panasonic_regimes.parquet"))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

    # (a) low-i RMSE + dv by temperature
    ax = axes[0]
    rmse = [probe["per_temp"][t]["low_rmse_mv"] for t in TEMPS]
    dv = [probe["per_temp"][t]["dv_med_mv"] for t in TEMPS]
    x = np.arange(len(TEMPS))
    ax.bar(x, rmse, 0.55, color="#4c72b0", label="low-i RMSE (mV)")
    ax.set_xticks(x, TLAB)
    ax.set_xlabel("Temperature (deg C)")
    ax.set_ylabel("Low-current RMSE (mV)", color="#4c72b0")
    ax.tick_params(axis="y", labelcolor="#4c72b0")
    ax2 = ax.twinx()
    ax2.plot(x, dv, "o-", color="#c44e52", label="fitted dv (mV)")
    ax2.set_ylabel("dv (mV)", color="#c44e52")
    ax2.tick_params(axis="y", labelcolor="#c44e52")
    ax2.grid(False)
    ax.set_title("(a) Cold-concentrated floor tracks dv")

    # (b) regime medians at -20 and 25
    ax = axes[1]
    regs = ["rest_start", "rest_other", "dyn_low"]
    labels = ["start rest", "end rest", "dynamic low-i"]
    colors = ["#dd8452", "#937860", "#55a868"]
    for j, (tk, off) in enumerate([("-20", -0.2), ("25", 0.2)]):
        vals = []
        for reg in regs:
            v = probe["regime_by_temp"].get(tk, {}).get(reg, {}).get("med_mv")
            vals.append(v if v is not None else np.nan)
        ax.bar(np.arange(3) + off, vals, 0.36, label=f"{tk} deg C",
               color=colors if j == 0 else [c + "" for c in colors],
               alpha=1.0 if j == 0 else 0.45, hatch="" if j == 0 else "//")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel("Median residual e (mV)")
    ax.set_title("(b) Regime split: rest reads dv, not physics")
    ax.legend(fontsize=8)

    # (c) A/B/C SSE shares per temperature (low-i = rest_start + rest_other + dyn_low)
    ax = axes[2]
    shares = {reg: [] for reg in regs}
    for t in TEMPS:
        sub = rg[rg.temp_c == float(t)]
        s = {reg: float(sub[sub.regime == reg].sse_mv2.sum()) for reg in regs}
        tot = sum(s.values()) or 1.0
        for reg in regs:
            shares[reg].append(100.0 * s[reg] / tot)
    bottom = np.zeros(len(TEMPS))
    for reg, lab, col in zip(regs, ["A: dv-mix artifact", "B: unconverged state",
                                    "C: model residual"], colors):
        ax.bar(np.arange(len(TEMPS)), shares[reg], 0.6, bottom=bottom,
               label=lab, color=col)
        bottom += np.array(shares[reg])
    ax.set_xticks(np.arange(len(TEMPS)), TLAB)
    ax.set_xlabel("Temperature (deg C)")
    ax.set_ylabel("Share of low-i SSE (%)")
    ax.set_title("(c) Floor attribution A/B/C")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "floor_r3_decomposition.png"), dpi=150)
    plt.close(fig)


def fig_rest_regimes() -> None:
    probe = _load("floor_r3_panasonic_probe.json")
    h20 = _load("floor_r3_panasonic_h20.json")
    rs = pd.read_parquet(os.path.join(MET, "floor_r3_panasonic_restsegs.parquet"))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

    # (a) settled-rest table gap by temperature
    ax = axes[0]
    vals = [probe["restsegs_by_temp"].get(t, {}).get("table_gap_start_mv", np.nan)
            for t in TEMPS]
    vals2 = [h20["per_temp"][t]["table_gap_settled_mv"] for t in TEMPS]
    x = np.arange(len(TEMPS))
    ax.bar(x - 0.18, vals, 0.36, label="start rest (soc=1)", color="#4c72b0")
    ax.bar(x + 0.18, vals2, 0.36, label="settled rest (median)", color="#8172b3")
    ax.axhspan(-10, 10, color="green", alpha=0.12, label="strict line +/-10 mV")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, TLAB)
    ax.set_xlabel("Temperature (deg C)")
    ax.set_ylabel("v_meas - OCV_table (mV)")
    ax.set_title("(a) Settled rests: table is self-consistent")
    ax.legend(fontsize=7)

    # (b) end-rest equilibrium brackets
    ax = axes[1]
    for j, t in enumerate(TEMPS):
        pt = h20["per_temp"][t]
        lo, hi = pt["eq_bracket_lo_mv"], pt["eq_bracket_hi_mv"]
        if not np.isfinite(lo):
            continue
        ax.plot([j, j], [lo, hi], "-", color="#c44e52", lw=3, solid_capstyle="round",
                label="naive/fix-tau bracket" if j == 0 else None)
        ax.plot(j, pt["gap_eq_naive_mv"], "v", color="#8c8c8c", ms=6,
                label="naive free-tau" if j == 0 else None)
        ax.plot(j, pt["gap_eq_fixtau_mv"], "^", color="#222222", ms=6,
                label="fix-tau (HPPC prior)" if j == 0 else None)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhspan(-20, 20, color="green", alpha=0.12, label="+/-20 mV line")
    ax.set_xticks(np.arange(len(TEMPS)), TLAB)
    ax.set_xlabel("Temperature (deg C)")
    ax.set_ylabel("Equilibrium table gap (mV)")
    ax.set_title("(b) End-rest bracket (naive v / fix-tau ^)")
    ax.legend(fontsize=7)

    # (c) synthetic control readings
    ax = axes[2]
    ctrl = h20["synthetic_control"]
    keys = [k for k in ctrl if "dur300" in k and k.split("_")[-1] != "settled"]
    keys.sort()
    xs = np.arange(len(keys))
    for j, k in enumerate(keys):
        v = ctrl[k]
        ax.plot([j - 0.2, j + 0.2], [v["gap_end_mv"], v["gap_eq_fixtau_mv"]],
                "-", color="#4c72b0", lw=1.5)
        ax.plot(j - 0.2, v["gap_end_mv"], "o", color="#dd8452", ms=5)
        ax.plot(j + 0.2, v["gap_eq_fixtau_mv"], "^", color="#222222", ms=5)
    ax.axhline(20.0, color="k", lw=0.8, ls="--", label="true delta = 20 mV")
    ax.set_xticks(xs, [k.replace("tau", "").replace("_dur300", "\n")
                       .replace("_polarized", "pol").replace("_twomode", "2m")
                       for k in keys], fontsize=7)
    ax.set_ylabel("Table-gap reading (mV)")
    ax.set_title("(c) Synthetic control: raw vs fix-tau (300 s)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "floor_r3_rest_regimes.png"), dpi=150)
    plt.close(fig)


def fig_verdicts() -> None:
    h20 = _load("floor_r3_panasonic_h20.json")
    h21 = _load("floor_r3_panasonic_h21.json")
    h22 = _load("floor_r3_panasonic_h22.json")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

    # (a) H20: settled table gap vs lines
    ax = axes[0]
    vals = [h20["per_temp"][t]["table_gap_settled_mv"] for t in TEMPS]
    x = np.arange(len(TEMPS))
    cols = ["#c44e52" if abs(v) > 10 else "#55a868" for v in vals]
    ax.bar(x, vals, 0.55, color=cols)
    ax.axhline(10, color="k", lw=0.7, ls="--")
    ax.axhline(-10, color="k", lw=0.7, ls="--")
    ax.axhline(30, color="k", lw=0.7, ls=":", label="major line 30 mV")
    ax.axhline(-30, color="k", lw=0.7, ls=":")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, TLAB)
    ax.set_xlabel("Temperature (deg C)")
    ax.set_ylabel("Settled table gap (mV)")
    ax.set_title("(a) H20 rejected: table not the floor")
    ax.legend(fontsize=7)

    # (b) H21: ddv + artifact share
    ax = axes[1]
    ddv = [abs(h21["per_temp"][t]["ddv_med_mv"]) for t in TEMPS]
    share = [100.0 * h21["per_temp"][t]["artifact_share"] for t in TEMPS]
    ax.bar(x - 0.18, ddv, 0.36, color="#4c72b0", label="|Delta dv| (mV)")
    ax.set_xticks(x, TLAB)
    ax.set_xlabel("Temperature (deg C)")
    ax.set_ylabel("|Delta dv| (mV)", color="#4c72b0")
    ax.tick_params(axis="y", labelcolor="#4c72b0")
    ax2 = ax.twinx()
    ax2.plot(x, share, "s-", color="#dd8452", label="net artifact share (%)")
    ax2.axhline(30, color="#dd8452", lw=0.8, ls="--")
    ax2.set_ylabel("Net artifact share (%)", color="#dd8452")
    ax2.tick_params(axis="y", labelcolor="#dd8452")
    ax2.grid(False)
    ax.set_title("(b) H21 partial: Delta dv real, net 18% < 30%")

    # (c) H22: LOO transfer cuts
    ax = axes[2]
    ck = [100.0 * h22["per_temp"][t]["cut_kernel"] for t in TEMPS]
    cs = [100.0 * h22["per_temp"][t]["cut_shape"] for t in TEMPS]
    ax.bar(x - 0.18, ck, 0.36, color="#55a868", label="u-kernel transfer")
    ax.bar(x + 0.18, cs, 0.36, color="#8172b3", label="bias+shape transfer")
    ax.axhline(25, color="k", lw=0.8, ls="--", label="pass line 25%")
    ax.axhline(10, color="k", lw=0.8, ls=":", label="weak line 10%")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, TLAB)
    ax.set_xlabel("Temperature (deg C)")
    ax.set_ylabel("Held-out RMSE cut (%)")
    ax.set_title("(c) H22 rejected: no shared structure")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "floor_r3_verdicts.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_decomposition()
    fig_rest_regimes()
    fig_verdicts()
    print("figures written to docs/figures/")
