"""容量/内阻老化耦合：外部 SOH 锚点（A2 接口）压力测试。

为什么要做这个（误差预算的首位）
--------------------------------
§23 已证在线联合估计容量（EKFQ）在真实数据上结构性失败 —— 38–53 mV 的模型
残差与容量误差在电压上同构，``q`` 把残差当容量吸收。容量只能来自**外部锚点**
（A2 的 SOH 输出），本脚本量化「A2 要交付多准、多频」以及老化耦合的代价：

1. **E1 r0**（真实数据）：老化使 R0 上升 20–100%，``R̂0`` 错了会怎样？
   R0 误差是 i 相关的，偏置状态 ``b`` 吸不掉 —— 此前从未量化。
2. **E2 walk**（真实数据）：两次刷新之间锚点会漂（SOH 估计器自身的漂移），
   刷新只修正**往后的积分速率**，已积累的 SOC 误差不会被撤销 —— 刷新周期
   到底值多少？
3. **E3 drift**（半合成植物）：``Q_ref``/``R0`` 真的时变化（老化）——
   真实电流驱动 ECM，``Q_true(t)``/``R0_true(t)`` 线性漂移 + 50 mV 慢变残差，
   验证广义增长律 ``err(k) = Σ_{j<k} (1/q̂ − 1/q_true(j))·i_j·dt/3600``，
   并对比 oracle（锚点实时准确）/ stale（从不刷新）/ refresh600。

协议
----
- 场景固定 ``known``（初值准、电流无偏置），误差只可能来自注入的锚点/老化；
- 估计器：cc（增长律核对）+ ekf（电压反馈能挡掉多少）；
- E1 网格：``Q̂ = s·Q``（s ∈ {0.95, 1.00, 1.05}）× ``R̂0 = ρ·R0``
  （ρ ∈ {0.85, 1.00, 1.25, 1.50, 2.00}）；
- E2 网格：漂移速率 w ∈ {0, 0.06, 0.18}/h × 刷新周期 T ∈ {段长(从不), 900, 300 s}，
  刷新点锚点回到真值、周期内线性漂走；外加一角（偏置 +5% + w=0.06 + T=900）；
- E3 网格：aging ∈ {mild: Q −2%/R0 +10%, severe: Q −5%/R0 +50%} ×
  {oracle, stale, refresh600}，植物残差 = 50 mV 正弦 + 10 mV 噪声。

预注册假设
----------
H8：cc 在时变容量下的逐点误差精确符合广义增长律（数值差 < 1e-6 pp）。
H9：段末误差由锚点**偏置**主导；刷新周期本身几乎不值钱 —— 刷新只拉回速率、
    不撤销存量误差，staleness 的均值贡献才是本质（→ A2 的规格是「准」，不是「勤」）。
H10：R̂0 误差对 SOC 的泄漏远小于同百分比的容量误差（u/b 吸大部分），
     但 severe（+50%）下不可忽略。
H11：E3 的 stale − oracle 差 ≈ 广义增长律代入 q̂ = q_true(0) 的预测。

用法：
    python scripts/17_aging_anchor.py --dataset calce
    python scripts/17_aging_anchor.py --dataset panasonic
    python scripts/17_aging_anchor.py --dataset calce --exps drift --limit 3   # 冒烟
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from bmsoc import metrics as met                                    # noqa: E402
from bmsoc.ecm import simulate_tv                                   # noqa: E402
from bmsoc.filters import CoulombCounter, EKF, FilterConfig, SoHSchedule  # noqa: E402
from bmsoc.profiles import DrivingProfile                           # noqa: E402

MET = os.path.join(ROOT, "results", "metrics")

R0_SCALES = [0.85, 1.00, 1.25, 1.50, 2.00]
Q_SCALES = [0.95, 1.00, 1.05]
WALK_PER_HOUR = [0.0, 0.06, 0.18]
T_UPD_S = [None, 900.0, 300.0]          # None = 从不刷新（段内一直漂）
WALK_CORNER = {"q_bias": 0.05, "walk_per_hour": 0.06, "t_upd_s": 900.0}
AGING = {"mild": {"dq": 0.02, "dr": 0.10}, "severe": {"dq": 0.05, "dr": 0.50}}
DRIFT_MODES = ["oracle", "stale", "refresh600"]
RESIDUAL_MV = 50.0                      # 慢变残差幅度（正弦），对齐真实开环残差量级
NOISE_V = 0.010
ESTIMATORS = [("cc", CoulombCounter), ("ekf", EKF)]


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_dataset(dataset: str):
    s12 = _load_module("s12", os.path.join(ROOT, "scripts", "12_calibration_stress.py"))
    return s12.load_calce() if dataset == "calce" else s12.load_panasonic()


# ---- 锚点构造（A2 输出的三种形态） ------------------------------------

def anchor_static(q_true: float, s: float, r0_true: float, rho: float) -> SoHSchedule:
    """E1：静态锚点 —— BMS 持旧标定（容量 s 倍、内阻 ρ 倍）。"""
    return SoHSchedule([0.0], [q_true * s], [r0_true * rho])


def anchor_walk(q_true: float, walk_per_hour: float, t_upd_s, t_end: float,
                knot_dt: float = 60.0) -> SoHSchedule:
    """E2：A2 锚点 —— 刷新点回真值，之后以 w/h 线性漂走（零阶保持近似）。

    刷新只修正往后的积分速率；已积累的 SOC 误差不在修正范围内（BMS 侧
    没有新的 SOC 参照点就无法重算），这正是本实验要量化的代价。
    """
    ts, qs = [], []
    period = float(t_end) if t_upd_s is None else float(t_upd_s)
    w = float(walk_per_hour) / 3600.0
    t = 0.0
    while t < t_end - 1e-9:
        seg = min(period, t_end - t)
        m = max(2, int(seg / knot_dt) + 1)
        for frac in np.linspace(0.0, 1.0, m, endpoint=False):
            ts.append(t + frac * seg)
            qs.append(q_true * (1.0 + w * frac * seg))
        t += period
    return SoHSchedule(ts, qs)


def anchor_tv(t_knots: np.ndarray, q_at, r0_at) -> SoHSchedule:
    """E3：时变锚点 —— 在给定节点上取当时的 Q̂/R̂0（oracle/refresh 共用）。"""
    return SoHSchedule(t_knots, [q_at(t) for t in t_knots],
                       [r0_at(t) for t in t_knots])


def realized_staleness(anchor: SoHSchedule, q_true: float, t_s: np.ndarray):
    q = anchor.q_at(t_s)
    d = np.abs(q / q_true - 1.0)
    return float(d.max()), float(d.mean())


def theory_end_cc(q_hat_arr: np.ndarray, q_true_arr: np.ndarray,
                  i: np.ndarray, dt: float) -> float:
    """广义增长律的段末值：Σ (1/q̂(j) − 1/q_true(j))·i_j·dt/3600。

    q 全程恒定时化回 §23 的静态式 ``−(1/s−1)(1−SOC_end)``。
    对 cc 是精确式；EKF 偏离它的程度就是「电压反馈挡掉了多少」。
    """
    term = (1.0 / q_hat_arr - 1.0 / q_true_arr) * i * dt / 3600.0
    return float(np.sum(term))


# ---- 三个子实验 -------------------------------------------------------

def run_r0(ocv2d, profiles, fold_params) -> list:
    """E1：静态 Q̂/R̂0 网格（真实数据）。"""
    rows = []
    for p in profiles:
        temp = float(p.meta["temp_c"])
        prof = str(p.meta["profile"])
        params = fold_params(temp, prof)
        Q, r0 = p.capacity_ah, params.r0
        q_true_arr = np.full(len(p), Q)
        for s in Q_SCALES:
            for rho in R0_SCALES:
                anchor = anchor_static(Q, s, r0, rho)
                st_max, st_mean = realized_staleness(anchor, Q, p.t_s)
                theory = theory_end_cc(anchor.q_at(p.t_s), q_true_arr,
                                       p.current_a, p.dt)
                cfg = FilterConfig(soc0=float(np.clip(p.soc_ref[0], 0.0, 1.0)),
                                   sigma_soc0=0.05)
                for name, cls in ESTIMATORS:
                    est = cls(ocv=ocv2d, params=params, capacity_ah=Q, cfg=cfg)
                    out = est.run(p, anchor=anchor)
                    rows.append(_row("r0", name, p, s=s, rho=rho, out=out,
                                     st_max=st_max, st_mean=st_mean, theory=theory))
    return rows


def run_walk(ocv2d, profiles, fold_params) -> list:
    """E2：锚点漂移 + 刷新周期（真实数据）。"""
    combos = [{"q_bias": 0.0, "walk_per_hour": w, "t_upd_s": t}
              for w in WALK_PER_HOUR for t in T_UPD_S]
    combos.append(dict(WALK_CORNER))
    rows = []
    for p in profiles:
        temp = float(p.meta["temp_c"])
        prof = str(p.meta["profile"])
        params = fold_params(temp, prof)
        Q = p.capacity_ah
        t_end = float(p.t_s[-1])
        q_true_arr = np.full(len(p), Q)
        cfg = FilterConfig(soc0=float(np.clip(p.soc_ref[0], 0.0, 1.0)), sigma_soc0=0.05)
        for c in combos:
            anchor = anchor_walk(Q, c["walk_per_hour"], c["t_upd_s"], t_end)
            if c["q_bias"]:
                # 偏置叠加在漂移上：刷新点回 Q·(1+bias)，漂移轨迹整体乘 (1+bias)
                anchor = SoHSchedule(anchor.t_upd, anchor.q_ah * (1.0 + c["q_bias"]))
            st_max, st_mean = realized_staleness(anchor, Q, p.t_s)
            theory = theory_end_cc(anchor.q_at(p.t_s), q_true_arr, p.current_a, p.dt)
            for name, cls in ESTIMATORS:
                est = cls(ocv=ocv2d, params=params, capacity_ah=Q, cfg=cfg)
                out = est.run(p, anchor=anchor)
                rows.append(_row("walk", name, p,
                                 s=float(anchor.q(0.0) / Q), rho=1.0,
                                 walk=c["walk_per_hour"], t_upd=c["t_upd_s"],
                                 out=out, st_max=st_max, st_mean=st_mean, theory=theory))
    return rows


def run_drift(ocv2d, profiles, fold_params) -> list:
    """E3：半合成老化植物 —— Q_ref/R0 时变化。"""
    rows = []
    for p in profiles:
        temp = float(p.meta["temp_c"])
        prof = str(p.meta["profile"])
        params = fold_params(temp, prof)
        n = len(p)
        t_s = p.t_s
        dt = p.dt
        i = p.current_a
        soc0 = float(np.clip(p.soc_ref[0], 0.05, 1.0))
        # 确定性种子（不能用 hash()：字符串 hash 跨进程不稳定）
        seed = zlib.crc32(f"{temp}|{prof}".encode()) % (2 ** 31)
        rng = np.random.default_rng(seed)
        x = np.linspace(0.0, 1.0, n)
        resid = (RESIDUAL_MV / 1000.0) * np.sin(2.0 * np.pi * x)
        noise = rng.normal(0.0, NOISE_V, size=n)
        for aging, ab in AGING.items():
            q_true = p.capacity_ah * (1.0 - ab["dq"] * x)
            r0_true = params.r0 * (1.0 + ab["dr"] * x)
            sim = simulate_tv(params, ocv2d, i, dt, q_arr=q_true, r0_arr=r0_true,
                              soc0=soc0, temp_c=p.temp_c)
            prof_tv = DrivingProfile(
                t_s=t_s, current_a=i, voltage_v=sim["voltage"] + resid + noise,
                temp_c=p.temp_c, soc_ref=sim["soc"], capacity_ah=p.capacity_ah,
                meta=dict(p.meta))
            for mode in DRIFT_MODES:
                if mode == "stale":
                    anchor = SoHSchedule([0.0], [q_true[0]], [r0_true[0]])
                elif mode == "refresh600":
                    knots = np.arange(0.0, t_s[-1] + 1.0, 600.0)
                    anchor = anchor_tv(knots, lambda t: _at(t_s, q_true, t),
                                       lambda t: _at(t_s, r0_true, t))
                else:   # oracle：60 s 节点上的真值
                    knots = np.arange(0.0, t_s[-1] + 1.0, 60.0)
                    anchor = anchor_tv(knots, lambda t: _at(t_s, q_true, t),
                                       lambda t: _at(t_s, r0_true, t))
                st_max, st_mean = realized_staleness(anchor, float(q_true.mean()), t_s)
                theory = theory_end_cc(anchor.q_at(t_s), q_true, i, dt)
                for name, cls in ESTIMATORS:
                    est = cls(ocv=ocv2d, params=params, capacity_ah=p.capacity_ah,
                              cfg=FilterConfig(soc0=soc0, sigma_soc0=0.05))
                    out = est.run(prof_tv, anchor=anchor)
                    # 指标参照必须是植物真值（prof_tv.soc_ref = sim["soc"]），
                    # 不是原真实段的 soc_ref —— 两者在老化漂移下不同
                    r = _row("drift", name, prof_tv,
                             s=float(anchor.q(0.0) / p.capacity_ah),
                             rho=float(anchor.r0(0.0, params.r0) / params.r0),
                             out=out, st_max=st_max, st_mean=st_mean, theory=theory,
                             aging=aging, mode=mode)
                    if name == "cc" and mode == "stale":
                        r["theory_gap_max"] = _growth_gap(out["soc_hat"], sim["soc"],
                                                          q_true, i, dt)
                    rows.append(r)
    return rows


def _at(t_s: np.ndarray, arr: np.ndarray, t: float) -> float:
    j = min(max(np.searchsorted(t_s, t, side="right") - 1, 0), arr.size - 1)
    return float(arr[j])


def _growth_gap(soc_hat, soc_true, q_true, i, dt) -> float:
    """广义增长律的逐点核对：err(k) = Σ_{j<k} (1/q̂0 − 1/q_true(j))·i_j·dt/3600。"""
    q0 = float(q_true[0])
    term = (1.0 / q0 - 1.0 / q_true) * i * dt / 3600.0
    theory = np.zeros(term.size)
    theory[1:] = np.cumsum(term[:-1])
    err = soc_hat - soc_true
    return float(np.max(np.abs(err - theory)))


def _row(exp, name, p, out, st_max, st_mean, theory, s=1.0, rho=1.0,
         walk=float("nan"), t_upd=float("nan"), aging="", mode=""):
    m = met.soc_metrics(out["soc_hat"], p.soc_ref, p.t_s, tol=0.02)
    return {"exp": exp, "estimator": name,
            "temp_c": float(p.meta["temp_c"]), "profile": str(p.meta["profile"]),
            "n": len(p), "capacity_ah": p.capacity_ah, "soc_end": float(p.soc_ref[-1]),
            "q_scale": float(s), "r0_scale": float(rho),
            "walk_per_hour": walk, "t_upd_s": t_upd,
            "aging": aging, "mode": mode,
            "stale_max": st_max, "stale_mean": st_mean,
            "err_end": float(out["soc_hat"][-1] - p.soc_ref[-1]),
            "theory_end": float(theory),
            **{f"soc_{k}": v for k, v in m.items()}}


def summarise(res: pd.DataFrame, n_profiles: int) -> dict:
    out = {"config": {"n_profiles": n_profiles, "r0_scales": R0_SCALES,
                      "q_scales": Q_SCALES, "walk_per_hour": WALK_PER_HOUR,
                      "t_upd_s": T_UPD_S, "aging": AGING, "modes": DRIFT_MODES,
                      "residual_mv": RESIDUAL_MV, "noise_v": NOISE_V}}
    for exp, g in res.groupby("exp"):
        key_cols = {"r0": ["q_scale", "r0_scale"],
                    "walk": ["walk_per_hour", "t_upd_s", "q_scale"],
                    "drift": ["aging", "mode"]}[exp]
        blocks = []
        for keys, gg in g.groupby(key_cols, dropna=False):
            keys = keys if isinstance(keys, tuple) else (keys,)
            rec = {k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                   for k, v in zip(key_cols, keys)}
            for est, ge in gg.groupby("estimator"):
                rec[est] = {"rmse_median": float(ge.soc_rmse.median()),
                            "tail_median": float(ge.soc_tail_rmse.median()),
                            "err_end_median": float(ge.err_end.median()),
                            "err_end_p90": float(ge.err_end.quantile(0.9)),
                            "stale_max_median": float(ge.stale_max.median()),
                            "n": int(len(ge))}
            blocks.append(rec)
        out[exp] = blocks
    dr = res[(res.exp == "drift") & (res.estimator == "cc") & (res["mode"] == "stale")]
    if len(dr) and "theory_gap_max" in dr:
        out["growth_law_gap_max"] = float(dr.theory_gap_max.max())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["calce", "panasonic"], required=True)
    ap.add_argument("--exps", default="r0,walk,drift")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    exps = [e.strip() for e in args.exps.split(",") if e.strip() in ("r0", "walk", "drift")]
    if not exps:
        raise SystemExit("--exps 至少要选一个：r0 / walk / drift")

    ocv2d, profiles, fold_params = load_dataset(args.dataset)
    if args.limit:
        profiles = profiles[:args.limit]

    t_all = time.time()
    rows = []
    runners = {"r0": run_r0, "walk": run_walk, "drift": run_drift}
    for exp in exps:
        part = runners[exp](ocv2d, profiles, fold_params)
        print(f"[{exp}] {len(part)} runs, 累计 {time.time() - t_all:.0f}s", flush=True)
        rows.extend(part)

    if not rows:
        raise SystemExit(f"没有任何 run：profiles={len(profiles)}。"
                         "先查数据根目录（CALCE_UMD / PANASONIC_ROOT）")
    res = pd.DataFrame(rows)
    out_pq = os.path.join(MET, f"aging_anchor_{args.dataset}_runs.parquet")
    if os.path.exists(out_pq):
        old = pd.read_parquet(out_pq)
        old = old[~old.exp.isin(exps)]        # 被重跑的实验整块替换，其余保留
        res = pd.concat([old, res], ignore_index=True)
    res.to_parquet(out_pq, index=False)
    payload = {"dataset": args.dataset, "exps": exps, **summarise(res, len(profiles))}
    with open(os.path.join(MET, f"aging_anchor_{args.dataset}.json"), "w",
              encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    print("\n=== E1 r0：ekf 全程 RMSE 中位（pp），行=Q̂/Q，列=R̂0/R0 ===")
    g = res[(res.exp == "r0") & (res.estimator == "ekf")]
    if len(g):
        piv = g.pivot_table(index="q_scale", columns="r0_scale",
                            values="soc_rmse", aggfunc="median") * 100
        print(piv.to_string(float_format=lambda v: f"{v:6.2f}"))
    print("\n=== E2 walk：ekf 全程 RMSE 中位（pp），仅 w/o 偏置行 ===")
    g = res[(res.exp == "walk") & (res.estimator == "ekf") & (res.q_scale == 1.0)]
    if len(g):
        piv = g.pivot_table(index="walk_per_hour", columns="t_upd_s",
                            values="soc_rmse", aggfunc="median") * 100
        print(piv.to_string(float_format=lambda v: f"{v:6.2f}"))
    print("\n=== E3 drift：ekf 尾 RMSE 中位 / 段末误差中位（pp） ===")
    g = res[(res.exp == "drift") & (res.estimator == "ekf")]
    if len(g):
        piv = g.pivot_table(index="aging", columns="mode",
                            values=["soc_tail_rmse", "err_end"], aggfunc="median") * 100
        print(piv.to_string(float_format=lambda v: f"{v:6.2f}"))
    if "growth_law_gap_max" in payload:
        print(f"\n广义增长律逐点最大缺口：{payload['growth_law_gap_max'] * 100:.6f} pp")
    print(f"\n耗时 {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
