#!/usr/bin/env python
"""25_round6_figures.py — 第六轮图（v1.5，图内英文）。

1. residual_r2_mechanism.png   NCA 残差温度×电流结构 + us06 −20 热关联
2. residual_r2_hppc_saturation.png  HPPC 组内起点电阻饱和曲线
3. residual_r2_verdicts.png    三个判决实验（Δ(i) / 全局Δ(soc) / 热修正）对照
4. ekfq_boundary2.png          残差结构分层边界 + 热修正 EKFQ 复验
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


def fig_mechanism() -> None:
    s = pd.read_parquet(os.path.join(MET, "residual_autopsy_panasonic_samples.parquet"))
    s["abs_i"] = s.i.abs()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))

    # (a) RMSE by temp x |i|
    ax = axes[0]
    bins = [0, 1, 2, 4, 8, 30]
    labels = ["0-1", "1-2", "2-4", "4-8", "8+"]
    s["ibin"] = pd.cut(s.abs_i, bins, labels=labels)
    piv = s.groupby(["temp_c", "ibin"], observed=True).resid_v.agg(
        lambda x: np.sqrt((x ** 2).mean()) * 1000).unstack()
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="magma",
                   extent=[-0.5, 4.5, -22.5, 27.5], origin="lower")
    ax.set_xticks(range(5), labels)
    ax.set_xlabel("|i| bin (A)")
    ax.set_ylabel("Temperature (deg C)")
    ax.set_title("(a) NCA residual RMSE (mV)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85)

    # (b) bias by dir x |i|
    ax = axes[1]
    s["dir"] = np.where(s.i > 0, "chg", "dis")
    piv2 = s.groupby(["ibin", "dir"], observed=True).resid_v.mean().unstack() * 1000
    x = np.arange(len(piv2))
    ax.bar(x - 0.18, piv2.get("chg", pd.Series(0, index=piv2.index)).fillna(0),
           0.36, label="charge", color="#d95f02")
    ax.bar(x + 0.18, piv2.get("dis", pd.Series(0, index=piv2.index)).fillna(0),
           0.36, label="discharge", color="#1b9e77")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, labels)
    ax.set_xlabel("|i| bin (A)")
    ax.set_ylabel("Mean residual (mV)")
    ax.set_title("(b) odd-in-i signature at high |i|")
    ax.legend(fontsize=8)

    # (c) us06 -20: residual vs dT*i (thermal signature) — 逐样本实测温度
    ax = axes[2]
    cache = os.path.join(MET, "residual_r2_us06n20_thermal.parquet")
    if os.path.exists(cache):
        u = pd.read_parquet(cache)
    else:
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(
            "s12", os.path.join(ROOT, "scripts", "12_calibration_stress.py"))
        s12 = _ilu.module_from_spec(spec)
        spec.loader.exec_module(s12)
        from bmsoc.ecm import open_loop_from_truth, _unpack_vec  # noqa: F401
        ocv2d, profiles, _f = s12.load_panasonic()
        up = pd.read_parquet(os.path.join(MET, "model_upgrade_panasonic_folds.parquet"))
        with open(os.path.join(MET, "model_upgrade_panasonic.json"), encoding="utf-8") as f:
            best = json.load(f)["best"]
        vec = None
        for _, r in up[up.variant == best].iterrows():
            if float(r.temp_c) == -20.0 and str(r.held_out) == "us06":
                vec = np.asarray(json.loads(r.vec_json), float)
        rec = []
        for p in profiles:
            if str(p.meta["profile"]) == "us06" and float(p.meta["temp_c"]) == -20.0:
                vh = open_loop_from_truth(vec, p.soc_ref, p.current_a, ocv2d, p.dt, -20.0)
                u = pd.DataFrame({"resid_v": vh - p.voltage_v, "i": p.current_a,
                                  "temp_c": np.asarray(p.temp_c, float)})
                u["dT"] = u.temp_c - u.temp_c.min()
                rec.append(u)
        u = pd.concat(rec, ignore_index=True)
        u.to_parquet(cache, index=False)
    g = u.dT.values * u.i.values
    ax.scatter(g, u.resid_v * 1000, s=3, alpha=0.35, color="#7570b3")
    if np.std(g) > 0:
        b = np.polyfit(g, u.resid_v.values, 1)
        xs = np.linspace(g.min(), g.max(), 50)
        ax.plot(xs, np.polyval(b, xs) * 1000, "k-", lw=1.5,
                label=f"slope {b[0]*1e3:.2f} mV/(K$\\cdot$A)")
    ax.set_xlabel("dT $\\times$ i (K$\\cdot$A)")
    ax.set_ylabel("Residual (mV)")
    ax.set_title("(c) us06 -20 deg C: thermal coupling")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "residual_r2_mechanism.png"), dpi=150)
    plt.close(fig)


def fig_hppc_saturation() -> None:
    onset = pd.read_parquet(os.path.join(MET, "residual_r2_hppc_onset.parquet"))
    shape = pd.read_parquet(os.path.join(MET, "residual_r2_hppc_shape.parquet"))
    with open(os.path.join(MET, "residual_r2_hppc.json"), encoding="utf-8") as f:
        summ = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    temps = sorted(onset.temp_c.unique())
    for idx, T in enumerate(temps):
        sub = onset[onset.temp_c == T]
        c = cmap(idx / max(1, len(temps) - 1))
        ax.scatter(sub.abs_i, sub.ratio, s=12, alpha=0.5, color=c)
        sh = shape[shape.temp_c == T].sort_values("abs_i_med")
        ax.plot(sh.abs_i_med, sh.ratio_med, "o-", color=c, ms=4, lw=1.4,
                label=f"{T:.0f} deg C")
        k = summ["k_per_temp"].get(str(T), summ["k_per_temp"].get(f"{T:.1f}"))
        if k is not None:
            xs = np.linspace(1.2, 18, 40)
            ax.plot(xs, 1.0 / (1.0 + k * xs), "-", color=c, lw=0.9, alpha=0.7)
    ax.set_xlabel("|i| (A)")
    ax.set_ylabel("R_onset / R_onset(low-rate)")
    ax.set_title("(a) within-group onset resistance")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    ks = [(float(k), v) for k, v in summ["k_per_temp"].items()]
    ks.sort()
    ax.plot([k for k, _ in ks], [v for _, v in ks], "o-", color="#d95f02")
    ax.set_xlabel("Temperature (deg C)")
    ax.set_ylabel("saturation k (1/A)")
    ax.set_title("(b) BV-type saturation vs temperature")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "residual_r2_hppc_saturation.png"), dpi=150)
    plt.close(fig)


def fig_verdicts() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    # (a) NCA di / therm
    ax = axes[0]
    nca_di = pd.read_parquet(os.path.join(MET, "residual_r2_panasonic_di_folds.parquet"))
    nca_th = pd.read_parquet(os.path.join(MET, "residual_r2_panasonic_therm_folds.parquet"))
    cats = ["base", "d(i)\nsame-T", "d(i)\nall-T", "HPPC\nshape", "R(T)\nraw", "R(T)\nnorm"]
    vals = [nca_di.base_rmse_mv.median(), nca_di.same_rmse_mv.median(),
            nca_di.all_rmse_mv.median(), nca_di.hppc_rmse_mv.median(),
            nca_th.rt_rmse_mv.median(), nca_th.rtnorm_rmse_mv.median()]
    colors = ["#7570b3"] + ["#d95f02"] * 3 + ["#1b9e77"] * 2
    ax.bar(range(len(cats)), vals, color=colors)
    ax.axhline(32, color="k", ls="--", lw=1, label="H16/H19 target 32 mV")
    ax.axhline(42.1, color="gray", ls=":", lw=1)
    ax.set_xticks(range(len(cats)), cats, fontsize=7)
    ax.set_ylabel("LOO median RMSE (mV)")
    ax.set_title("(a) NCA verdicts")
    ax.legend(fontsize=7)

    # (b) LFP dst
    ax = axes[1]
    lfp = pd.read_parquet(os.path.join(MET, "residual_r2_calce_dst_folds.parquet"))
    cats = ["base", "global\nd(soc)", "global+\nT-bias", "same-T\nd(soc)"]
    vals = [lfp.base_rmse_mv.median(), lfp.global_rmse_mv.median(),
            lfp.global_tbias_rmse_mv.median(), lfp.same_rmse_mv.median()]
    ax.bar(range(len(cats)), vals, color=["#7570b3", "#d95f02", "#d95f02", "#1b9e77"])
    ax.axhline(14, color="k", ls="--", lw=1, label="H17 target 14 mV")
    ax.set_xticks(range(len(cats)), cats, fontsize=7)
    ax.set_title("(b) LFP verdict (H17 dead)")
    ax.legend(fontsize=7)

    # (c) us06 tail: R(T) raw
    ax = axes[2]
    u = nca_th[nca_th.held_out == "us06"].sort_values("temp_c")
    x = np.arange(len(u))
    ax.bar(x - 0.2, u.base_rmse_mv, 0.4, label="base", color="#7570b3")
    ax.bar(x + 0.2, u.rt_rmse_mv, 0.4, label="R(T) raw", color="#1b9e77")
    ax.axhline(80, color="k", ls="--", lw=1, label="H19 us06 target 80 mV")
    ax.set_xticks(x, [f"{t:.0f}" for t in u.temp_c])
    ax.set_xlabel("Temperature (deg C)")
    ax.set_title("(c) us06 folds: thermal tail fixed")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "residual_r2_verdicts.png"), dpi=150)
    plt.close(fig)


def fig_boundary2() -> None:
    b2 = pd.read_parquet(os.path.join(MET, "ekfq_boundary2_runs.parquet"))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    ax = axes[0]
    styles = {"slow": ("o-", "#7570b3"), "odd": ("s-", "#1b9e77"), "even": ("^-", "#d95f02")}
    for st, (fmt, c) in styles.items():
        sub = b2[(b2.structure == st) & (b2.estimator == "ekfq")]
        g = sub.groupby("resid_mv").agg(rmse=("rmse", "mean"),
                                        qok=("q_ok", "mean")).reset_index()
        ax.plot(g.resid_mv, g.rmse * 100, fmt, color=c, label=st)
    ax.axhline(1.95, color="gray", ls=":", lw=1, label="EKF fixed-wrong-Q")
    ax.set_xlabel("Injected residual RMS (mV)")
    ax.set_ylabel("EKFQ SOC RMSE (pp)")
    ax.set_title("(a) residual structure vs EKFQ")
    ax.legend(fontsize=8)

    ax = axes[1]
    with open(os.path.join(MET, "ekfq_therm_reverify_panasonic.json"), encoding="utf-8") as f:
        rb = json.load(f)
    keys = ["upgraded_ekf_rmse", "upgraded_thermal_ekf_rmse",
            "upgraded_ekfq_rmse", "upgraded_thermal_ekfq_rmse"]
    labs = ["EKF\nv1.4", "EKF\nthermal", "EKFQ\nv1.4", "EKFQ\nthermal"]
    ax.bar(range(4), [rb[k] * 100 for k in keys],
           color=["#7570b3", "#1b9e77", "#7570b3", "#1b9e77"])
    ax.set_xticks(range(4), labs, fontsize=8)
    ax.set_ylabel("Median SOC RMSE (pp)")
    ax.set_title("(b) NCA real reverify (thermal model)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "ekfq_boundary2.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    fig_mechanism()
    fig_hppc_saturation()
    fig_verdicts()
    fig_boundary2()
    print("figures written:", sorted(f for f in os.listdir(FIG) if f.startswith(("residual_r2", "ekfq_boundary2"))))


if __name__ == "__main__":
    main()
