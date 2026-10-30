"""生成 README 与报告用的图（英文标注，输出到 docs/figures/）。

只依赖 results/metrics/ 下的落盘结果；少数需要时序的图（跟踪轨迹、内部状态、
误差 CDF）会用 04 的同一套参数重跑单条工况，成本很低。

用法：
    python scripts/06_figures.py [--only ocv,tracking]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt  # noqa: E402

from bmsoc import io_panasonic as io                        # noqa: E402
from bmsoc import metrics as met                             # noqa: E402
from bmsoc import ocv as ocvmod                              # noqa: E402
from bmsoc.filters import ESTIMATORS, FilterConfig, AEKF      # noqa: E402
from bmsoc.plotting import (C_ACC, C_GRAY, C_MAIN, C_RED,     # noqa: E402
                            EST_COLOR, EST_LABEL, savefig)
from bmsoc.profiles import build_profile                     # noqa: E402

OUT = os.path.join(ROOT, "docs", "figures")
MET = os.path.join(ROOT, "results", "metrics")
ORDER = ["cc", "ekf", "ukf", "aekf"]


def _bench():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_benchmark_soc.py")
    spec = importlib.util.spec_from_file_location("bench4", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B4 = _bench()


def _load_profile(temp: float, profile: str, dt: float = 1.0):
    hits = [p for p, t, pr in io.list_files() if t == temp and pr == profile]
    if not hits:
        raise SystemExit(f"缺少 {temp}°C/{profile}")
    df = io.to_frame(hits[0])
    q = B4.load_capacity()[temp]
    po = build_profile(df.t_s, df.current_a, df.voltage_v, df.temp_c, q, soc0=1.0, dt=dt)
    return po.trim_leading_rest(keep_s=300.0), B4.fold_params(temp, profile)


# ---------------------------------------------------------------------------
# 1. OCV(SOC, T) 与可观测性
# ---------------------------------------------------------------------------
def fig_ocv():
    ocv2d = B4.load_ocv2d()
    # 端点（SOC→0）的斜率会冲到 4 V/unit 以上，画进去会把中间的平台压扁，
    # 所以只画 0.03–0.97，并在标题里给出全区间的最小/最大斜率。
    soc = np.linspace(0.03, 0.97, 200)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    temps = ocv2d.temps
    cmap = plt.get_cmap("viridis")
    h = 0.005
    d_all = []
    for i, t in enumerate(temps):
        col = cmap(i / max(len(temps) - 1, 1) * 0.85)
        axes[0].plot(soc, ocv2d(soc, float(t)), color=col, lw=1.6, label=f"{t:g} °C")
        d = (ocv2d(np.minimum(soc + h, 1.0), float(t))
             - ocv2d(np.maximum(soc - h, 0.0), float(t))) / (
                 np.minimum(soc + h, 1.0) - np.maximum(soc - h, 0.0))
        d_all.append(d)
        axes[1].plot(soc, d, color=col, lw=1.8)
        if t == 25.0:
            k = int(np.argmin(d))
            axes[1].plot(soc[k], d[k], "o", ms=4, color=col)
            axes[1].annotate(f"min {d[k]:.2f} @ SOC {soc[k]:.2f}",
                             xy=(soc[k], d[k]), xytext=(soc[k] + 0.05, d[k] - 0.10),
                             fontsize=8, color=col,
                             arrowprops=dict(arrowstyle="-", color=col, lw=0.8))
    lo = min(float(d.min()) for d in d_all)
    hi = max(float(d.max()) for d in d_all)
    axes[0].set_xlabel("SOC (–)")
    axes[0].set_ylabel("OCV (V)")
    axes[0].set_title("(a) OCV–SOC from HPPC rest points")
    axes[0].legend(title="cell temperature", ncol=2)
    axes[1].set_xlabel("SOC (–)")
    axes[1].set_ylabel("dOCV/dSOC (V per unit SOC)")
    axes[1].set_title(f"(b) Observability: dOCV/dSOC ∈ [{lo:.2f}, {hi:.2f}]")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(0.2, 1.40)
    fig.suptitle("Panasonic NCR18650PF · OCV table (PAVA-monotone, PCHIP)", fontsize=11)
    return savefig(fig, os.path.join(OUT, "ocv_curves.png"))


# ---------------------------------------------------------------------------
# 2. 留一工况开环电压残差
# ---------------------------------------------------------------------------
def fig_openloop():
    df = pd.read_parquet(os.path.join(MET, "ecm_folds.parquet"))
    temps = sorted(df["temp_c"].unique())
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for i, t in enumerate(temps):
        sub = df[df["temp_c"] == t]
        x = np.full(len(sub), i, dtype=float) + np.linspace(-0.13, 0.13, len(sub))
        ax.scatter(x, sub["val_rmse_v"] * 1e3, s=34, color=plt.get_cmap("viridis")(
            i / max(len(temps) - 1, 1) * 0.85), zorder=3, alpha=0.9,
            label=f"{t:g} °C (median {sub['val_rmse_v'].median()*1e3:.0f} mV)")
        ax.hlines(sub["val_rmse_v"].median() * 1e3, i - 0.28, i + 0.28,
                  color=C_MAIN, lw=1.8)
    ax.axhline(30.0, color=C_GRAY, ls=":", lw=1.2)
    ax.text(-0.45, 33, "30 mV reference", color=C_GRAY, fontsize=8)
    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([f"{t:g} °C" for t in temps])
    ax.set_ylim(0, 215)
    ax.set_ylabel("open-loop voltage RMSE on held-out profile (mV)")
    ax.set_title("Leave-one-profile-out calibration\n(OCV fixed from HPPC, 6 electrical parameters fitted)")
    ax.legend(fontsize=8, loc="upper right")
    return savefig(fig, os.path.join(OUT, "openloop_residual.png"))


# ---------------------------------------------------------------------------
# 3. SOC 轨迹
# ---------------------------------------------------------------------------
def fig_tracking():
    cases = [(-20.0, "la92"), (0.0, "la92"), (25.0, "udds")]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), sharey=True)
    for ax, (temp, prof) in zip(axes, cases):
        po, par = _load_profile(temp, prof)
        cfg = FilterConfig(soc0=float(np.clip(po.soc_ref[0] - 0.30, 0, 1)), sigma_soc0=0.20)
        ax.plot(po.t_s / 60, po.soc_ref * 100, color="black", lw=2.0,
                label="reference", zorder=5)
        for name in ORDER:
            est = ESTIMATORS[name](ocv=B4.load_ocv2d(), params=par,
                                   capacity_ah=po.capacity_ah, cfg=cfg)
            out = est.run(po, current_bias_a=0.02)
            m = met.soc_metrics(out["soc_hat"], po.soc_ref, po.t_s, tol=0.02)
            ax.plot(po.t_s / 60, out["soc_hat"] * 100, color=EST_COLOR[name], lw=1.2,
                    label=f"{EST_LABEL[name]} ({m['rmse']*100:.1f} pp)")
        ax.set_xlabel("time (min)")
        ax.set_title(f"{temp:g} °C · {prof.upper()} · init −30 pp, bias +20 mA")
    axes[0].set_ylabel("SOC (%)")
    axes[0].legend(loc="lower left", fontsize=8)
    return savefig(fig, os.path.join(OUT, "soc_tracking.png"))


# ---------------------------------------------------------------------------
# 4. 误差汇总（按场景）
# ---------------------------------------------------------------------------
def fig_error_summary():
    df = pd.read_parquet(os.path.join(MET, "soc_benchmark_runs.parquet"))
    scen = ["known", "bias20ma", "init30", "hard"]
    df = df[df["scenario"].isin(scen)]
    # 库仑计数在 0–30 pp 之间乱跳，跟滤波器放同一根 y 轴会把 1–4 pp 的差别压成一条线，
    # 所以拆两栏：左栏只看三个滤波器（0–10 pp），右栏单独放库仑计数（0–31 pp）。
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    width_f, width_c = 0.20, 0.16

    def draw(ax, names, width):
        for i, name in enumerate(names):
            xs, med = [], []
            for j, s in enumerate(scen):
                sub = df[(df["scenario"] == s) & (df["estimator"] == name)]
                if sub.empty:
                    continue
                x = j + (i - (len(names) - 1) / 2) * width
                ax.scatter(np.full(len(sub), x), sub["soc_rmse"] * 100, s=13,
                           color=EST_COLOR[name], alpha=0.35, zorder=2, lw=0)
                xs.append(x)
                med.append(sub["soc_rmse"].median() * 100)
            ax.plot(xs, med, color=EST_COLOR[name], lw=1.4, zorder=3)
            ax.scatter(xs, med, s=46, color=EST_COLOR[name], zorder=4,
                       label=EST_LABEL[name], edgecolor="white", linewidth=0.8)
        ax.set_xticks(range(len(scen)))
        ax.set_xticklabels(["known\n(0, 0)", "bias\n(0, +20 mA)", "init\n(−30 pp, 0)",
                            "hard\n(−30 pp, +20 mA)"])
        ax.grid(alpha=0.25)

    draw(axes[0], [n for n in ORDER if n != "cc"], width_f)
    draw(axes[1], ["cc"], width_c)
    axes[0].set_ylabel("SOC RMSE over full profile (pp)")
    axes[0].set_ylim(0, 10)
    axes[0].set_title("(a) the three filters")
    n_big = int(((df["estimator"] != "cc") & (df["scenario"] != "known")
                 & (df["soc_rmse"] > 0.10)).sum())
    n_tot = int(((df["estimator"] != "cc") & (df["scenario"] != "known")).sum())
    axes[0].text(0.985, 0.03,
                 f"{n_big}/{n_tot} runs above 10 pp are clipped (max 18.1)",
                 transform=axes[0].transAxes, ha="right", fontsize=7.5, color="#666666")
    axes[1].set_ylim(0, 31)
    axes[1].set_title("(b) coulomb counting")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.suptitle("24 driving profiles × 4 scenarios — points: individual runs, line: median",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return savefig(fig, os.path.join(OUT, "error_summary.png"))


# ---------------------------------------------------------------------------
# 5. 内部状态：偏置状态与协方差
# ---------------------------------------------------------------------------
def fig_internal():
    po, par = _load_profile(-20.0, "la92")
    cfg = FilterConfig(soc0=float(np.clip(po.soc_ref[0] - 0.30, 0, 1)), sigma_soc0=0.20)
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.4), sharex=True)
    tails = {}
    for name in ("ekf", "aekf"):
        out = ESTIMATORS[name](ocv=B4.load_ocv2d(), params=par,
                               capacity_ah=po.capacity_ah, cfg=cfg).run(po, current_bias_a=0.02)
        axes[0].plot(po.t_s / 60, out["bias_hat"] * 1000, color=EST_COLOR[name], lw=1.3,
                     label=EST_LABEL[name])
        axes[1].plot(po.t_s / 60, out["sigma_soc"] * 100, color=EST_COLOR[name], lw=1.3,
                     label=EST_LABEL[name])
        tails[name] = float(out["bias_hat"][-int(0.3 * len(po)):].mean() * 1000)
        # 稳态值写进图例，避免在曲线上叠字
        axes[0].lines[-1].set_label(f"{EST_LABEL[name]}  (b → {tails[name]:+.0f} mV)")
    axes[0].axhline(-par.dv * 1000, color=C_RED, ls="--", lw=1.1)
    axes[0].annotate(f"−dv = {-par.dv*1000:+.0f} mV (calibrated OCV offset)",
                     xy=(po.t_s[-1] / 60, -par.dv * 1000), xytext=(0.12, 0.44),
                     textcoords="axes fraction", fontsize=8, color=C_RED,
                     arrowprops=dict(arrowstyle="->", color=C_RED, lw=0.9))
    # 两条曲线的稳态值不一样，这本身就是结论：EKF 把偏移全压在 b 上，
    # AEKF 分了一部分到 R，所以 b 停在更低的位置。
    axes[0].axhline(0.0, color=C_GRAY, lw=0.8)
    axes[0].set_ylabel("bias state b (mV)")
    axes[0].set_title("Voltage-bias state absorbs model offset instead of corrupting SOC")
    axes[0].legend(loc="lower right")
    axes[1].set_ylabel(r"$\sigma_{SOC}$ (pp)")
    axes[1].set_xlabel("time (min)")
    axes[1].set_title("State uncertainty collapses during the initial rest, then stays bounded")
    axes[1].legend(loc="upper right")
    fig.suptitle("−20 °C · LA92 · init −30 pp, bias +20 mA", fontsize=10)
    return savefig(fig, os.path.join(OUT, "bias_state.png"))


# ---------------------------------------------------------------------------
# 6. 敏感性
# ---------------------------------------------------------------------------
def fig_sensitivity():
    path = os.path.join(MET, "soc_sensitivity.json")
    if not os.path.exists(path):
        print("跳过 sensitivity（缺 soc_sensitivity.json）")
        return None
    with open(path, encoding="utf-8") as f:
        agg = json.load(f)["aggregate"]
    spec = [("dsoc0", 100.0, "initial SOC error (pp)", 100.0,
             "the filters pull it back"),
            ("current_bias_a", 1000.0, "current-sensor bias (mA)", 1000.0,
             "cc grows linearly, filters don't"),
            ("sigma_v", 1000.0, "voltage noise (mV)", 1000.0,
             "UKF is the most noise-sensitive")]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    for ax, (axis, mult, xlabel, div, subtitle) in zip(axes, spec):
        block = agg.get(axis, {})
        for name in ORDER:
            xs, ys = [], []
            for v, d in block.items():
                got = d.get(name)
                if got is None:
                    continue
                xs.append(float(v) * div)
                ys.append(got["rmse_median"] * 100)
            if xs:
                o = np.argsort(xs)
                ax.plot(np.array(xs)[o], np.array(ys)[o], "o-", ms=4, lw=1.3,
                        color=EST_COLOR[name], label=EST_LABEL[name])
        ax.set_xlabel(xlabel)
        ax.set_title(subtitle, fontsize=9.5)
        ax.grid(alpha=0.25)
    # 第一栏要让库仑计数的 0→36 pp 爬坡可见；后两栏只看滤波器（0–8 pp）才有分辨率。
    axes[0].set_ylim(0, 40)
    axes[1].set_ylim(0, 9)
    axes[2].set_ylim(0, 9)
    axes[0].set_ylabel("median SOC RMSE (pp)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Sensitivity over 6 profiles spanning −20…25 °C "
                 "(one axis at a time, others at their baseline)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return savefig(fig, os.path.join(OUT, "sensitivity.png"))


# ---------------------------------------------------------------------------
# 7. 误差 CDF
# ---------------------------------------------------------------------------
def fig_error_cdf():
    po, par = _load_profile(0.0, "la92")
    cfg = FilterConfig(soc0=float(np.clip(po.soc_ref[0] - 0.30, 0, 1)), sigma_soc0=0.20)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name in ORDER:
        out = ESTIMATORS[name](ocv=B4.load_ocv2d(), params=par,
                               capacity_ah=po.capacity_ah, cfg=cfg).run(po, current_bias_a=0.02)
        e = np.sort(np.abs(out["soc_hat"] - po.soc_ref)) * 100
        ax.plot(e, np.linspace(0, 100, e.size), color=EST_COLOR[name], lw=1.5,
                label=EST_LABEL[name])
    ax.axvline(2.0, color=C_RED, ls="--", lw=1.1)
    ax.text(2.2, 12, "2 pp", color=C_RED, fontsize=8)
    ax.set_xlim(0, 12)
    ax.set_xlabel("|SOC error| (pp)")
    ax.set_ylabel("share of samples below (%)")
    ax.set_title("0 °C · LA92 · init −30 pp, bias +20 mA")
    ax.legend(loc="lower right")
    return savefig(fig, os.path.join(OUT, "error_cdf.png"))


FIGS = {"ocv": fig_ocv, "openloop": fig_openloop, "tracking": fig_tracking,
        "summary": fig_error_summary, "internal": fig_internal,
        "sensitivity": fig_sensitivity, "cdf": fig_error_cdf}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="逗号分隔，缺省全跑")
    args = ap.parse_args()
    want = [k.strip() for k in args.only.split(",") if k.strip()] or list(FIGS)
    for k in want:
        if k not in FIGS:
            raise SystemExit(f"未知图 {k}，可选 {sorted(FIGS)}")
        print("→", FIGS[k]())


if __name__ == "__main__":
    main()
