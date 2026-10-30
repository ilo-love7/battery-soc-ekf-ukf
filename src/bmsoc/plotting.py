"""统一绘图风格（浅色主题 + 与 A1/A2/A3 项目一致的色板）。"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.unicode_minus": False,
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.grid": True,
    "grid.color": "#e8e8e8",
    "grid.linewidth": 0.8,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
})

C_MAIN = "#2f6fd0"
C_ACC = "#e08b2e"
C_RED = "#c0392b"
C_GREEN = "#2e8b57"
C_GRAY = "#8a8a8a"
C_PURPLE = "#7a5ea8"
PALETTE = [C_MAIN, C_ACC, C_RED, C_GREEN, C_PURPLE, C_GRAY]

#: 估计器固定配色（全项目统一，图例顺序也固定）
EST_COLOR = {"cc": C_GRAY, "ekf": C_MAIN, "ukf": C_ACC, "aekf": C_GREEN}
EST_LABEL = {"cc": "Coulomb counting", "ekf": "EKF", "ukf": "UKF", "aekf": "AEKF"}


def savefig(fig, path: str, tight: bool = True) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
