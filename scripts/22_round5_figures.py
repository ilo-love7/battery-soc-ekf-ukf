"""第五轮（v1.4）图：残差摸底、模型升级、EKFQ 可行性边界。

三张图：
1. residual_autopsy.png  残差四维分解（慢/快频带、充放差、|i| 斜率）+ 机制归因
2. model_upgrade.png     变体对比（2rc/3rc/r0_cur）+ ΔOCV 形状重标定前后
3. ekfq_boundary.png     EKFQ 可行性边界（残差水平扫描）+ 真实数据复验

图内文字统一英文（DejaVu 无 CJK 字形，中文只写在 markdown 里）。
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

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 130, "savefig.dpi": 130, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_BASE, C_3RC, C_DOCV = "#8c8c8c", "#4477aa", "#cc6677"
C_SLOW, C_FAST = "#4477aa", "#ee9944"


def _save(fig, name: str):
    path = os.path.join(FIG, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"→ {path}")


def fig_autopsy():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    data = {}
    for ds, label in (("calce", "CALCE LFP"), ("panasonic", "Panasonic NCA")):
        with open(os.path.join(MET, f"residual_autopsy_{ds}.json"), encoding="utf-8") as f:
            data[ds] = json.load(f)

    # ① 慢/快分解
    ax = axes[0]
    x = np.arange(2)
    slow = [data["calce"]["slow300_rms_median_mv"], data["panasonic"]["slow300_rms_median_mv"]]
    fast = [data["calce"]["fast300_rms_median_mv"], data["panasonic"]["fast300_rms_median_mv"]]
    ax.bar(x, slow, 0.55, label="slow (>300 s)", color=C_SLOW, edgecolor="none")
    ax.bar(x, fast, 0.55, bottom=slow, label="fast", color=C_FAST, edgecolor="none")
    for k, (s, f) in enumerate(zip(slow, fast)):
        ax.text(k, s + f + 1.5, f"{s:.0f}/{f:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x, ["CALCE LFP", "Panasonic NCA"])
    ax.set_ylabel("LOO residual RMS (mV)")
    ax.set_title("(a) Slow vs fast band")
    ax.legend(frameon=False, fontsize=8)
    ax.set_axisbelow(True)

    # ② 充放偏置差（被否掉的滞后假设）
    ax = axes[1]
    gap = [data["calce"]["hyst_gap_median_mv"], data["panasonic"]["hyst_gap_median_mv"]]
    chg = [data["calce"]["chg_bias_median_mv"], data["panasonic"]["chg_bias_median_mv"]]
    dis = [data["calce"]["dis_bias_median_mv"], data["panasonic"]["dis_bias_median_mv"]]
    ax.bar(x - 0.18, chg, 0.3, label="charge bias", color="#aa3377", edgecolor="none")
    ax.bar(x + 0.18, dis, 0.3, label="discharge bias", color="#66aacc", edgecolor="none")
    ax.axhline(0, color="k", lw=0.6)
    for k, g in enumerate(gap):
        ax.text(k, 11, f"gap\n{g:+.1f}", ha="center", fontsize=8, color="#666666")
    ax.set_xticks(x, ["CALCE LFP", "Panasonic NCA"])
    ax.set_ylabel("Residual bias (mV)")
    ax.set_title("(b) Charge/discharge gap small:\nhysteresis hypothesis refuted")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(-8, 16)

    # ③ 慢残差与极化代理的 r²（机制归因）
    ax = axes[2]
    r2 = [0.02, 0.06]     # 慢残差 ~ 极化低通的 R²（分析快检中位）
    r2b = [0.02, 0.06]
    ax.bar(x, [0.85, 0.58], 0.55, label="slow share of residual", color="#dddddd",
           edgecolor="none")
    ax.bar(x, [0.02, 0.06], 0.55, label=r"$R^2$ slow~polarization", color=C_3RC,
           edgecolor="none")
    ax.set_xticks(x, ["CALCE LFP", "Panasonic NCA"])
    ax.set_title("(c) Slow residual is NOT polarization\n"
                 r"($R^2\approx0.05$ vs low-passed $i$)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, "residual_autopsy.png")


def fig_model_upgrade():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))

    # ① 变体对比
    ax = axes[0]
    variants = ["2rc", "2rc_cur", "3rc", "3rc_cur"]
    labels = ["2RC\n(baseline)", "2RC\n+r0(i)", "3RC", "3RC\n+r0(i)"]
    for ds, color, name in (("calce", "#cc6677", "CALCE LFP"),
                            ("panasonic", "#4477aa", "Panasonic NCA")):
        with open(os.path.join(MET, f"model_upgrade_{ds}.json"), encoding="utf-8") as f:
            s = json.load(f)
        vals = [s["by_variant"][v]["val_rmse_median_mv"] for v in variants]
        ax.plot(labels, vals, "o-", color=color, label=name, lw=1.8, ms=5)
        for xx, vv in zip(labels, vals):
            ax.annotate(f"{vv:.0f}", (xx, vv), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=8, color=color)
    ax.set_ylabel("LOO residual RMSE (mV)")
    ax.set_title("(a) RC order / r0(i) alone gain little")
    ax.legend(frameon=False, fontsize=8)
    ax.set_axisbelow(True)

    # ② ΔOCV 前后
    ax = axes[1]
    bars = []
    for ds, color, name in (("calce", "#cc6677", "CALCE LFP"),
                            ("panasonic", "#4477aa", "Panasonic NCA")):
        with open(os.path.join(MET, f"model_upgrade_{ds}_docv.json"), encoding="utf-8") as f:
            s = json.load(f)
        bars.append((name, color, s["before_median_mv"], s["after_median_mv"],
                     s["docv_max_median_mv"]))
    xx = np.arange(len(bars))
    for k, (name, color, before, after, dampl) in enumerate(bars):
        ax.bar(k - 0.18, before, 0.34, color="#bbbbbb", edgecolor="none",
               label="before" if k == 0 else None)
        ax.bar(k + 0.18, after, 0.34, color=color, edgecolor="none",
               label=r"after $\Delta$OCV" if k == 0 else None)
        ax.text(k - 0.18, before + 1, f"{before:.0f}", ha="center", fontsize=8)
        ax.text(k + 0.18, after + 1, f"{after:.0f}", ha="center", fontsize=8)
        ax.text(k, -6, f"ΔAmp≈{dampl:.0f} mV", ha="center", fontsize=8, color="#666666")
    ax.set_xticks(xx, [b[0] for b in bars])
    ax.set_ylabel("LOO residual RMSE (mV)")
    ax.set_title(r"(b) $\Delta$OCV shape recalibration: LFP 37$\to$18 mV")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(-10, 55)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, "model_upgrade.png")


def fig_ekfq_boundary():
    fp = os.path.join(MET, "ekfq_boundary.json")
    if not os.path.exists(fp):
        print("跳过 ekfq_boundary.png（边界扫描结果未落盘）")
        return
    with open(fp, encoding="utf-8") as f:
        b = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))

    ax = axes[0]
    resid = np.array([float(k) for k in b["by_resid"]])
    order = np.argsort(resid)
    ekfq = np.array([b["by_resid"][k]["ekfq_rmse"] for k in b["by_resid"]])[order] * 100
    ekf = np.array([b["by_resid"][k]["ekf_rmse"] for k in b["by_resid"]])[order] * 100
    orc = np.array([b["by_resid"][k]["oracle_rmse"] for k in b["by_resid"]])[order] * 100
    qerr = np.array([abs(b["by_resid"][k]["ekfq_q_rel_err_med"]) for k in b["by_resid"]])[order] * 100
    ax.plot(resid, ekfq, "o-", color=C_DOCV, label="EKFQ (joint q)", lw=1.8, ms=5)
    ax.plot(resid, ekf, "s--", color=C_BASE, label="EKF (fixed wrong q)", lw=1.5, ms=4)
    ax.plot(resid, orc, "^:", color="#228833", label="EKF (oracle q)", lw=1.5, ms=4)
    ax.set_xlabel("Injected slow residual RMS (mV)")
    ax.set_ylabel("SOC RMSE (pp)")
    ax.set_title("(a) Viability boundary (synthetic)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_axisbelow(True)

    ax2 = axes[1]
    ax2.plot(resid, qerr, "o-", color="#aa3377", lw=1.8, ms=5)
    ax2.axhline(2.0, color="k", lw=0.8, ls=":")
    ax2.text(resid[-1], 2.4, "2% target", ha="right", fontsize=8)
    ax2.set_xlabel("Injected slow residual RMS (mV)")
    ax2.set_ylabel(r"|$\hat q/q_{true}-1$| at segment end (%)")
    ax2.set_title("(b) Capacity identification error")
    ax2.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, "ekfq_boundary.png")


def fig_reverify():
    rows = []
    for ds, label in (("calce", "CALCE LFP"), ("panasonic", "Panasonic NCA")):
        fp = os.path.join(MET, f"ekfq_reverify_{ds}.json")
        if not os.path.exists(fp):
            continue
        with open(fp, encoding="utf-8") as f:
            s = json.load(f)
        for model, ms in s["by_model"].items():
            for scale, rec in ms.items():
                rows.append({"dataset": label, "model": model, "scale": float(scale),
                             "ekf": rec["ekf"]["rmse_med_pp"],
                             "ekfq": rec["ekfq"]["rmse_med_pp"],
                             "q_err": abs(rec["ekfq"]["q_rel_err_med"]) * 100})
    if not rows:
        print("跳过 ekfq_reverify.png（复验结果未落盘）")
        return
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for ax, scale in zip(axes, (1.05, 0.95)):
        sub = df[df.scale == scale]
        if sub.empty:
            continue
        labels = [f"{d}\n{m}" for d, m in zip(sub.dataset, sub.model)]
        x = np.arange(len(sub))
        ax.bar(x - 0.18, sub.ekf, 0.34, color=C_BASE, edgecolor="none", label="EKF fixed q")
        ax.bar(x + 0.18, sub.ekfq, 0.34, color=C_DOCV, edgecolor="none", label="EKFQ")
        for k, (_, r) in enumerate(sub.iterrows()):
            ax.text(k + 0.18, r.ekfq + 0.15, f"q{r.q_err:.1f}%", ha="center", fontsize=7,
                    color=C_DOCV)
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylabel("SOC RMSE median (pp)")
        ax.set_title(f"Real-data reverify, q0 = {scale:.2f}·Q")
        ax.legend(frameon=False, fontsize=8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, "ekfq_reverify.png")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_autopsy()
    fig_model_upgrade()
    fig_ekfq_boundary()
    fig_reverify()
    print("完成")
