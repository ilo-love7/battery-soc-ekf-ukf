#!/usr/bin/env python
"""31_round9_figures.py — 第九轮图（v1.8，图内英文）。

1. white_r5_signature.png   条件统计签名：di/dt 分箱 / 边沿 ETA / ACF·kurt
2. white_r5_waveform.png    波形与表观电阻：斜坡段 i-v / 表观 dV/di vs r0 / white·didt
3. white_r5_verdicts.png    判决面板：H26 结构读数 / 快模态 R² / H27·H28·对照
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
DIDT_LAB = ["[0,.05)", "[.05,.15)", "[.15,.4)", "[.4,1)", "[1,+)"]


def _load(name: str) -> dict:
    with open(os.path.join(MET, name), encoding="utf-8") as fh:
        return json.load(fh)


def fig_signature() -> None:
    probe = _load("white_r5_panasonic_probe.json")
    s = pd.read_parquet(os.path.join(MET, "white_r5_panasonic_samples.parquet"))
    eta = pd.read_parquet(os.path.join(MET, "white_r5_panasonic_eta.parquet"))
    cond = pd.read_parquet(os.path.join(MET, "white_r5_panasonic_cond.parquet"))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

    # (a) white RMS by |di/dt| bin, per temperature
    ax = axes[0]
    bins = [0, 0.05, 0.15, 0.4, 1.0, 99]
    width = 0.15
    x = np.arange(len(DIDT_LAB))
    cmap = plt.get_cmap("viridis")
    for j, temp in enumerate(TEMPS):
        sub = s[s.temp_c == temp].copy()
        if sub.empty:
            continue
        sub["db"] = pd.cut(sub.abs_didt, bins=bins, labels=DIDT_LAB, right=False)
        g = sub.groupby("db", observed=True).white_mv.apply(
            lambda v: float(np.sqrt(np.mean(v ** 2))))
        vals = [g.get(lab, np.nan) for lab in DIDT_LAB]
        ax.bar(x + (j - 2) * width, vals, width,
               color=cmap(j / 4), label=f"{TLAB[j]} deg C")
    ax.set_xticks(x, DIDT_LAB, fontsize=7)
    ax.set_xlabel("|di/dt| bin (A/s)")
    ax.set_ylabel("white RMS (mV)")
    ax.set_title("(a) Variance concentrates at current ramps")
    ax.legend(fontsize=6)

    # (b) edge-triggered average, onset (cold vs warm)
    ax = axes[1]
    for temp, col in ((-20.0, "#c44e52"), (25.0, "#4c72b0")):
        sub = eta[(eta.temp_c == temp) & (eta.etype == "onset")]
        if sub.empty:
            continue
        g = sub.groupby("rel_t_s").white_mv.mean()
        ax.plot(g.index, g.values, "-", color=col, lw=1.8,
                label=f"onset, {temp:g} deg C")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("Time from edge (s)")
    ax.set_ylabel("Mean white (mV)")
    ax.set_title("(b) Edge-triggered average (onset)")
    ax.legend(fontsize=7)

    # (c) ACF1 + excess kurtosis by temperature
    ax = axes[2]
    pt = probe["per_temp"]
    acf1 = [pt[f"{t:g}"]["acf_1s"] for t in TEMPS]
    kurt = [pt[f"{t:g}"]["white_kurt"] for t in TEMPS]
    x = np.arange(len(TEMPS))
    ax.bar(x - 0.18, acf1, 0.36, color="#55a868", label="ACF lag-1 s")
    ax.axhline(0.3, color="#55a868", ls="--", lw=1)
    ax.set_xticks(x, TLAB)
    ax.set_xlabel("Temperature (deg C)")
    ax.set_ylabel("ACF lag-1 s", color="#55a868")
    ax.set_title("(c) Memory (cold) and heavy tails")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, kurt, 0.36, color="#8172b2", label="excess kurtosis")
    ax2.set_ylabel("Excess kurtosis", color="#8172b2")
    ax2.axhline(1.0, color="#8172b2", ls="--", lw=1)
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "white_r5_signature.png"), dpi=150)
    plt.close(fig)


def fig_waveform() -> None:
    import importlib.util
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))

    def _load_mod(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    s12 = _load_mod("s12", os.path.join(ROOT, "scripts", "12_calibration_stress.py"))
    _, profiles, _ = s12.load_panasonic()
    prof_of = {(float(p.meta["temp_c"]), str(p.meta["profile"])): p for p in profiles}
    p = prof_of[(-20.0, "hwfet")]
    i = np.asarray(p.current_a, float)
    v = np.asarray(p.voltage_v, float)
    t = np.asarray(p.t_s, float)

    s = pd.read_parquet(os.path.join(MET, "white_r5_panasonic_samples.parquet"))
    cond = pd.read_parquet(os.path.join(MET, "white_r5_panasonic_cond.parquet"))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

    # (a) waveform around a representative ramp (hwfet -20, t≈300-330 s)
    ax = axes[0]
    m = (t >= 295) & (t <= 335)
    ax.plot(t[m], i[m], "-", color="#4c72b0", lw=1.5, label="current (A)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Current (A)", color="#4c72b0")
    ax.set_title("(a) Drive current is a ramp, not a step")
    ax2 = ax.twinx()
    ax2.plot(t[m], v[m], "-", color="#c44e52", lw=1.5, label="voltage (V)")
    ax2.set_ylabel("Voltage (V)", color="#c44e52")
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=7)

    # (b) apparent dV/di during ramps vs steady r0
    ax = axes[1]
    # 用斜坡段差分估计表观 dV/di（|di|≥0.05 A 且 |di/dt|≥0.15 A/s）
    didt = np.gradient(i, 1.0)
    di = np.diff(i)
    dv = np.diff(v)
    ramp = (np.abs(di) >= 0.05) & (np.abs(didt[1:]) >= 0.15)
    app = (dv[ramp] / di[ramp]) * 1e3  # mV/A（负号保留）
    r0_cold = 126.2
    r0_warm = 34.1
    ax.hist(np.abs(app), bins=np.linspace(0, 400, 21), color="#8172b2",
            alpha=0.75, label="|apparent dV/di| (ramp)")
    ax.axvline(r0_cold, color="#c44e52", lw=2, label=f"steady r0, -20 deg C ({r0_cold:.0f} mOhm)")
    ax.axvline(r0_cold * 2, color="#c44e52", lw=1.2, ls="--", label="2 x r0")
    ax.set_xlabel("Apparent |dV/di| (mV/A)")
    ax.set_ylabel("Count")
    ax.set_title("(b) Ramp response exceeds steady r0")
    ax.legend(fontsize=6)

    # (c) white RMS vs |di/dt| (per temp)
    ax = axes[2]
    bins = [0, 0.05, 0.15, 0.4, 1.0, 99]
    cmap = plt.get_cmap("viridis")
    for j, temp in enumerate(TEMPS):
        sub = s[s.temp_c == temp].copy()
        if sub.empty:
            continue
        sub["db"] = pd.cut(sub.abs_didt, bins=bins, labels=DIDT_LAB, right=False)
        g = sub.groupby("db", observed=True).white_mv.apply(
            lambda v_: float(np.sqrt(np.mean(v_ ** 2))))
        vals = [g.get(lab, np.nan) for lab in DIDT_LAB]
        ax.plot(np.arange(len(DIDT_LAB)), vals, "o-",
                color=cmap(j / 4), label=f"{TLAB[j]} deg C")
    ax.set_xticks(np.arange(len(DIDT_LAB)), DIDT_LAB, fontsize=7)
    ax.set_xlabel("|di/dt| bin (A/s)")
    ax.set_ylabel("white RMS (mV)")
    ax.set_title("(c) white RMS scales with ramp rate")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "white_r5_waveform.png"), dpi=150)
    plt.close(fig)


def fig_verdicts() -> None:
    probe = _load("white_r5_panasonic_probe.json")
    h26 = _load("white_r5_panasonic_h26.json")
    ff = pd.read_parquet(os.path.join(MET, "white_r5_panasonic_fastfit.parquet"))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))

    # (a) H26 structure lines: didt ratio / edge ratio by temp
    ax = axes[0]
    pt = probe["per_temp"]
    didt = [pt[f"{t:g}"]["didt_hi_lo_ratio"] for t in TEMPS]
    edge = [pt[f"{t:g}"]["edge_early_late_ratio"] for t in TEMPS]
    x = np.arange(len(TEMPS))
    ax.bar(x - 0.18, didt, 0.36, color="#4c72b0", label="high/low |di/dt| ratio")
    ax.bar(x + 0.18, edge, 0.36, color="#c44e52", label="edge 0-2s / 20-60s ratio")
    ax.axhline(2.0, color="#4c72b0", ls="--", lw=1, label="L2 line (2.0)")
    ax.axhline(1.5, color="#c44e52", ls="--", lw=1, label="L3 line (1.5)")
    ax.set_xticks(x, TLAB)
    ax.set_xlabel("Temperature (deg C)")
    ax.set_ylabel("Ratio")
    ax.set_title("(a) H26 structural signatures")
    ax.legend(fontsize=6)

    # (b) fast-RC fit R2 by temp x profile
    ax = axes[1]
    profs = sorted(ff.profile.unique())
    width = 0.15
    x = np.arange(len(TEMPS))
    cmap = plt.get_cmap("tab10")
    for j, pr in enumerate(profs):
        vals = []
        for temp in TEMPS:
            sub = ff[(ff.temp_c == temp) & (ff.profile == pr)]
            vals.append(float(sub.r2_best.iloc[0]) if len(sub) else np.nan)
        ax.bar(x + (j - len(profs) / 2) * width, vals, width,
               color=cmap(j % 10), label=pr)
    ax.axhline(0.20, color="k", ls="--", lw=1, label="L4 line (0.20)")
    ax.set_xticks(x, TLAB)
    ax.set_xlabel("Temperature (deg C)")
    ax.set_ylabel("fast-RC regression R$^2$")
    ax.set_title("(b) Explained white variance (tau=0.5-20 s)")
    ax.legend(fontsize=6, ncol=2)

    # (c) verdict summary + synthetic control
    ax = axes[2]
    labels = ["noise", "fast_rc", "hyst_slow", "hyst_fast",
              "NCA -20C", "NCA 25C"]
    ctrl = h26.get("ctrl_recov", {})
    didt_vals = [ctrl.get("noise", {}).get("didt_ratio", np.nan),
                 ctrl.get("fast_rc", {}).get("didt_ratio", np.nan),
                 ctrl.get("hyst_slow", {}).get("didt_ratio", np.nan),
                 ctrl.get("hyst_fast", {}).get("didt_ratio", np.nan),
                 probe["per_temp"]["-20"]["didt_hi_lo_ratio"],
                 probe["per_temp"]["25"]["didt_hi_lo_ratio"]]
    acf_vals = [ctrl.get("noise", {}).get("acf1", np.nan),
                ctrl.get("fast_rc", {}).get("acf1", np.nan),
                ctrl.get("hyst_slow", {}).get("acf1", np.nan),
                ctrl.get("hyst_fast", {}).get("acf1", np.nan),
                probe["per_temp"]["-20"]["acf_1s"],
                probe["per_temp"]["25"]["acf_1s"]]
    cols = ["#8c8c8c"] * 4 + ["#c44e52", "#4c72b0"]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, didt_vals, 0.36, color=cols, alpha=0.85)
    ax.axhline(1.3, color="k", ls="--", lw=1, label="noise line (1.3)")
    ax.set_xticks(x, labels, fontsize=6.5, rotation=12)
    ax.set_ylabel("|di/dt| hi/lo ratio")
    ax.set_title("(c) Controls vs real data (H28 rejected)")
    ax2 = ax.twinx()
    ax2.plot(x, acf_vals, "D", color="#55a868", ms=6, label="ACF1")
    ax2.axhline(0.3, color="#55a868", ls=":", lw=1)
    ax2.set_ylabel("ACF lag-1 s", color="#55a868")
    ax2.set_ylim(-0.5, 1.1)
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "white_r5_verdicts.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    fig_signature()
    fig_waveform()
    fig_verdicts()
    print("figures written:", os.listdir(FIG)[-3:])


if __name__ == "__main__":
    main()
