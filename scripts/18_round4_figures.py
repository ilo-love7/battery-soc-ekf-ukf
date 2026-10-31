"""第四轮（v1.3）图：外部 SOH 锚点规格 + 老化漂移增长律。

两张图：
- ``aging_anchor.png``：A2 接口规格。上排 E1（R0 老化耦合：RMSE vs R̂0 倍率，
  每条线一个 Q̂ 倍率），下排 E2（锚点漂移 + 刷新周期）。两列 = 两数据集。
- ``aging_drift.png``：E3 半合成老化植物。左 = cc 段末误差 vs 广义增长律
  理论值（对角线 = 精确），右 = oracle / refresh600 / stale 的段末误差
  （mild/severe 两档老化，两数据集并排）。

图内文字统一英文（DejaVu 无 CJK 字形，中文留 markdown 脚注）。
用法：
    python scripts/18_round4_figures.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MET = os.path.join(ROOT, "results", "metrics")
FIG = os.path.join(ROOT, "docs", "figures")

DS = {"calce": "CALCE A123 (LFP)", "panasonic": "Panasonic NCR18650PF (NCA)"}
PP = 100.0                      # SOC 小数 → 百分点


def load(ds: str) -> pd.DataFrame:
    p = os.path.join(MET, f"aging_anchor_{ds}_runs.parquet")
    if not os.path.exists(p):
        raise SystemExit(f"缺少 {p}，先跑 scripts/17_aging_anchor.py")
    return pd.read_parquet(p)


def fig_anchor(dfs) -> None:
    """A2 规格图：上排 E1（R0 耦合），下排 E2（漂移+刷新）。"""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2))
    for col, (ds, df) in enumerate(dfs.items()):
        # ---- E1：RMSE vs R̂0 倍率 ----
        ax = axes[0][col]
        g = df[(df.exp == "r0") & (df.estimator == "ekf")]
        for s, ls in ((0.95, "--"), (1.00, "-"), (1.05, ":")):
            gg = g[g.q_scale == s].groupby("r0_scale").soc_rmse.median() * PP
            ax.plot(gg.index, gg.values, ls, marker="o", ms=4,
                    label=f"Q-hat/Q = {s:.2f}")
        ax.set_xlabel(r"R0-hat / R0  (aging factor)")
        ax.set_ylabel("SOC RMSE median (pp)")
        ax.set_title(f"{DS[ds]}: R0 coupling (E1)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # ---- E2：RMSE vs 漂移速率 ----
        ax = axes[1][col]
        g = df[(df.exp == "walk") & (df.estimator == "ekf") & (df.q_scale == 1.0)]
        for t_upd, ls, lab in ((np.nan, "-", "no refresh"),
                               (900.0, "--", "refresh 15 min"),
                               (300.0, ":", "refresh 5 min")):
            gg = g[g.t_upd_s.fillna(-1) == (t_upd if not np.isnan(t_upd) else -1)]
            gg = gg.groupby("walk_per_hour").soc_rmse.median() * PP
            ax.plot(gg.index * 100.0, gg.values, ls, marker="s", ms=4, label=lab)
        ax.set_xlabel("SOH estimate walk (%/h)")
        ax.set_ylabel("SOC RMSE median (pp)")
        ax.set_title(f"{DS[ds]}: anchor staleness (E2)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    fig.tight_layout()
    out = os.path.join(FIG, "aging_anchor.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("→", out)


def fig_drift(dfs) -> None:
    """E3：左 = 增长律散点，右 = oracle/stale 对照。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    # ---- 左：广义增长律（cc，stale 锚点，时变 Q_true）----
    lo, hi = 0.0, 0.0
    for ds, df in dfs.items():
        g = df[(df.exp == "drift") & (df.estimator == "cc") & (df["mode"] == "stale")]
        x, y = g.theory_end * PP, g.err_end * PP
        ax1.scatter(x, y, s=22, alpha=0.75, label=DS[ds])
        lo = min(lo, x.min(), y.min())
        hi = max(hi, x.max(), y.max())
    pad = 0.05 * (hi - lo + 1e-9)
    ax1.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="y = x")
    ax1.set_xlabel("theory end error (pp)")
    ax1.set_ylabel("measured end error (pp)")
    ax1.set_title("Generalized growth law under drift (E3, cc)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    # ---- 右：oracle / refresh600 / stale ----
    modes = ["oracle", "refresh600", "stale"]
    agings = ["mild", "severe"]
    width = 0.12
    xticks = np.arange(len(agings))
    colors = {"calce": "#1f77b4", "panasonic": "#d62728"}
    for k, (ds, df) in enumerate(dfs.items()):
        g = df[(df.exp == "drift") & (df.estimator == "ekf")]
        for j, mode in enumerate(modes):
            vals = [g[(g.aging == a) & (g["mode"] == mode)].err_end.median() * PP
                    for a in agings]
            off = (j - 1) * width + (k - 0.5) * width * 0.5
            ax2.bar(xticks + off, vals, width * 0.9,
                    label=f"{mode} ({'LFP' if ds == 'calce' else 'NCA'})",
                    color=colors[ds], alpha=0.35 + 0.25 * j,
                    edgecolor="none", hatch="/" if mode == "stale" else None)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(["mild: Q-2%, R0+10%", "severe: Q-5%, R0+50%"])
    ax2.set_ylabel("end error median (pp)")
    ax2.set_title("Aging drift: anchor modes (E3, ekf)")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_axisbelow(True)

    fig.tight_layout()
    out = os.path.join(FIG, "aging_drift.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("→", out)


def main() -> None:
    dfs = {ds: load(ds) for ds in DS}
    fig_anchor(dfs)
    fig_drift(dfs)


if __name__ == "__main__":
    main()
