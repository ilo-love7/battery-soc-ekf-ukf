"""SOC 误差指标。"""
from __future__ import annotations

from typing import Dict

import numpy as np


def convergence_time(t_s: np.ndarray, err: np.ndarray, tol: float = 0.02) -> float:
    """首次进入并**一直保持**在 ``tol`` 以内的时间（秒）。

    判据取「此后所有点都在带内」，避免把一次偶然穿带算成收敛。
    """
    t = np.asarray(t_s, dtype=float)
    e = np.abs(np.asarray(err, dtype=float))
    inside = e <= tol
    n = inside.size
    if n == 0 or not inside[-1]:
        return float("nan")      # 末尾都没进带内 → 未收敛
    k = n - 1
    while k >= 0 and inside[k]:
        k -= 1
    return float(t[k + 1] - t[0])


def soc_metrics(soc_hat: np.ndarray, soc_ref: np.ndarray, t_s: np.ndarray,
                tol: float = 0.02, warmup_s: float = 0.0) -> Dict[str, float]:
    """一套 SOC 指标（单位：SOC 为 0–1 小数，pp = 百分点×100）。"""
    sh = np.asarray(soc_hat, dtype=float)
    sr = np.asarray(soc_ref, dtype=float)
    t = np.asarray(t_s, dtype=float)
    if sh.size != sr.size:
        raise ValueError("长度不一致")
    keep = t >= (t[0] + warmup_s) if warmup_s > 0 else np.ones(t.size, dtype=bool)
    e = sh[keep] - sr[keep]
    tk = t[keep]
    n_tail = max(1, int(0.3 * e.size))          # 末 30% 视为稳态
    tail = e[-n_tail:]
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "max_abs": float(np.max(np.abs(e))),
        "p95_abs": float(np.percentile(np.abs(e), 95)),
        "bias": float(np.mean(e)),
        "final_abs": float(abs(e[-1])),
        "tail_rmse": float(np.sqrt(np.mean(tail ** 2))),
        "tail_mae": float(np.mean(np.abs(tail))),
        "converge_s": convergence_time(tk, e, tol=tol),
        "converge5_s": convergence_time(tk, e, tol=0.05),
        "frac_out_2pp": float(np.mean(np.abs(e) > 0.02)),
        "frac_out_5pp": float(np.mean(np.abs(e) > 0.05)),
        "n": int(e.size),
    }


def summarise_runs(rows) -> Dict[str, Dict[str, float]]:
    """把逐条 run 的指标按估计器聚合（中位数 + 分位）。"""
    import collections

    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["estimator"]].append(r)
    out = {}
    for name, rs in groups.items():
        k = "soc_" if "soc_rmse" in rs[0] else ""
        rmse = np.array([r[k + "rmse"] for r in rs], dtype=float)
        mx = np.array([r[k + "max_abs"] for r in rs], dtype=float)
        conv = np.array([r[k + "converge_s"] for r in rs], dtype=float)
        tail = np.array([r.get(k + "tail_rmse", np.nan) for r in rs], dtype=float)
        conv5 = np.array([r.get(k + "converge5_s", np.nan) for r in rs], dtype=float)
        out[name] = {
            "n_runs": int(len(rs)),
            "rmse_median": float(np.median(rmse)),
            "rmse_mean": float(np.mean(rmse)),
            "rmse_p90": float(np.percentile(rmse, 90)),
            "max_abs_median": float(np.median(mx)),
            "tail_rmse_median": float(np.nanmedian(tail)) if np.isfinite(tail).any() else float("nan"),
            "conv_median_s": float(np.nanmedian(conv)) if np.isfinite(conv).any() else float("nan"),
            "conv5_median_s": float(np.nanmedian(conv5)) if np.isfinite(conv5).any() else float("nan"),
            # 全部未收敛时占比应为 0（而不是 NaN），否则聚合表里看不出「一条都没收敛」
            "conv5_frac_60s": float(np.mean(conv5 <= 60.0)) if conv5.size else float("nan"),
        }
    return out
