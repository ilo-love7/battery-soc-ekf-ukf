#!/usr/bin/env python
"""23_residual_round2.py — 第六轮：残差二压的判决实验（v1.5）。

三个阶段（对应工作记录 §37 的 H16/H17）：

- ``--stage hppc``   （仅 panasonic）HPPC 逐脉冲起点电阻 → **组内**倍率归一化
  的饱和曲线 s_T(|i|) = R_onset(|i|)/R_onset(低倍率)。组内比消除 SOC 混淆。
- ``--stage di``     电流形状修正 Δ(i) 的 LOO 判决：① 从训练折残差学 dir×|i| 分箱
  均值（纯数据）；② HPPC 饱和形状 + 单参数尺度 c（物理形状、训练折定标）。
  留出折的分箱无训练支撑时单独标注覆盖率，外推不算学习。
- ``--stage dst``    （仅 calce）全局 Δ(soc)：跨温度学习、严格 LOO，对照同温度
  2 折学习的 v1.4 基线（17.6 mV）。

输出：``results/metrics/residual_r2_*``；判决阈值见 §37（H16 ≤32 mV / H17 ≤14 mV）。
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
MET = os.path.join(ROOT, "results", "metrics")

IBIN_EDGES = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 30.0])
SOC_BINS = np.linspace(0.0, 1.0, 21)


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_dataset(dataset: str):
    s12 = _load_module("s12", os.path.join(ROOT, "scripts", "12_calibration_stress.py"))
    ocv2d, profiles, _f = s12.load_calce() if dataset == "calce" else s12.load_panasonic()
    return ocv2d, profiles


def fold_residuals(dataset: str):
    """重算每折开环残差（best 变体参数，来自脚本 20 的 vec_json）。

    返回 (folds, best, ocv2d)；每折含逐样本实测温度 ``T``（自热修正要用）。
    """
    from bmsoc.ecm import open_loop_from_truth, _unpack_vec  # noqa: F401
    ocv2d, profiles = load_dataset(dataset)
    fp = os.path.join(MET, f"model_upgrade_{dataset}_folds.parquet")
    if not os.path.exists(fp):
        raise SystemExit(f"缺 {fp}：先跑 scripts/20 --stage variants")
    up = pd.read_parquet(fp)
    with open(os.path.join(MET, f"model_upgrade_{dataset}.json"), encoding="utf-8") as f:
        best = json.load(f)["best"]
    params_of = {}
    for _, r in up[up.variant == best].iterrows():
        params_of[(float(r.temp_c), str(r.held_out))] = np.asarray(
            json.loads(r.vec_json), float)
    out = {}
    for p in profiles:
        temp = float(p.meta["temp_c"])
        key = (temp, str(p.meta["profile"]))
        vec = params_of.get(key)
        if vec is None:
            continue
        vh = open_loop_from_truth(vec, p.soc_ref, p.current_a, ocv2d, p.dt, temp)
        out[key] = {
            "temp_c": temp, "profile": str(p.meta["profile"]),
            "soc": p.soc_ref, "i": p.current_a,
            "e": vh - p.voltage_v, "dt": p.dt,
            "T": np.asarray(p.temp_c, dtype=float), "vec": vec,
        }
    return out, best, ocv2d


# ---------------------------------------------------------------- stage hppc

def stage_hppc() -> dict:
    s02 = _load_module("s02", os.path.join(ROOT, "scripts", "02_identify_ocv_ecm.py"))
    import bmsoc.io_panasonic as io

    pulse_rows = []
    for path, temp in io.list_hppc():
        m = io.load_meas(path)
        t, i, v, ah = m["t_s"], m["current_a"], m["voltage_v"], m["q_ah"]
        q_cap = float(abs(np.nanmin(ah)))
        dt = float(np.median(np.diff(t)))
        groups = s02.group_pulses(t, i, v, ah, q_cap, dt)
        for gi, g in enumerate(groups):
            recs = []
            for q in g["pulses"]:
                if abs(q["i_pulse"]) < 1.2:
                    continue
                i0, i1 = int(q["i0"]), int(q["i1"])
                dt_p = (t[i1] - t[i0]) / max(1, i1 - i0)
                if dt_p <= 0:
                    continue
                n_on = max(3, int(2.0 / dt_p))
                pre = slice(max(0, i0 - int(3.0 / dt_p)), i0)
                on = slice(i0 + 2, i0 + 2 + n_on)
                if pre.stop - pre.start < 5 or on.stop - on.start < 3:
                    continue
                di = float(np.median(i[on]) - np.median(i[pre]))
                if abs(di) <= 0.3:
                    continue
                r_on = float(np.median(v[on]) - np.median(v[pre])) / di
                recs.append({"temp_c": float(temp), "group": gi, "soc": float(q["soc"]),
                             "i_pulse": float(q["i_pulse"]), "r_onset": r_on,
                             "abs_i": abs(float(q["i_pulse"])), "dt_p": dt_p})
            # 组内归一化：比值 = R_onset / 组内最低倍率 R_onset
            if len(recs) >= 2:
                base = min(recs, key=lambda r: r["abs_i"])
                if base["r_onset"] > 1e-4:
                    for r in recs:
                        r["ratio"] = r["r_onset"] / base["r_onset"]
                        r["i_base"] = base["abs_i"]
                        pulse_rows.append(r)
    pdf = pd.DataFrame(pulse_rows)
    if pdf.empty:
        raise SystemExit("HPPC 组内起点电阻为空 —— 检查 group_pulses / 温度覆盖")
    pdf.to_parquet(os.path.join(MET, "residual_r2_hppc_onset.parquet"), index=False)

    # 每温度饱和形状：s = 1/(1+k·|i|) 拟合组内比值
    shape_rows, k_of = [], {}
    for temp in sorted(pdf.temp_c.unique()):
        sub = pdf[pdf.temp_c == temp]
        # 组内比值的分箱中位（展示用）
        sub = sub.assign(ibin=pd.cut(sub.abs_i, IBIN_EDGES))
        for b, g in sub.groupby("ibin", observed=True):
            shape_rows.append({"temp_c": temp, "ibin": str(b),
                               "abs_i_med": float(g.abs_i.median()),
                               "ratio_med": float(g.ratio.median()),
                               "n": int(len(g))})
        # k 拟合（非负）
        from scipy.optimize import minimize_scalar
        x, y = sub.abs_i.to_numpy(), sub.ratio.to_numpy()

        def loss(k):
            return float(np.mean((1.0 / (1.0 + k * x) - y) ** 2))

        res = minimize_scalar(loss, bounds=(0.0, 0.5), method="bounded")
        k_of[float(temp)] = float(res.x)
    sdf = pd.DataFrame(shape_rows)
    sdf.to_parquet(os.path.join(MET, "residual_r2_hppc_shape.parquet"), index=False)

    # H16 机制半场判决：≥3 温度单调下降、低温比值 <0.85
    mono_ok, cold_ok = 0, False
    for temp in sorted(pdf.temp_c.unique()):
        s = sdf[sdf.temp_c == temp].sort_values("abs_i_med")
        rr = s.ratio_med.to_numpy()
        if len(rr) >= 3 and np.all(np.diff(rr) <= 0.02):
            mono_ok += 1
        if temp <= -10 and len(rr) and float(rr.min()) < 0.85:
            cold_ok = True
    summary = {
        "n_pulses": int(len(pdf)), "n_groups": int(pdf.group.nunique()),
        "k_per_temp": {str(k): v for k, v in k_of.items()},
        "ratio_at_rate": {
            str(t): {str(e): float(sdf[(sdf.temp_c == t) &
                                       (sdf.abs_i_med >= e - 1.0) &
                                       (sdf.abs_i_med < e + 1.5)].ratio_med.iloc[0])
                     if len(sdf[(sdf.temp_c == t) & (sdf.abs_i_med >= e - 1.0) &
                                (sdf.abs_i_med < e + 1.5)]) else None}
            for t in sorted(pdf.temp_c.unique()) for e in (2.0, 6.0, 12.0)},
        "monotonic_temps": mono_ok, "cold_ratio_below_085": bool(cold_ok),
    }
    with open(os.path.join(MET, "residual_r2_hppc.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"HPPC 起点电阻 {len(pdf)} 脉冲 / {pdf.group.nunique()} 组；"
          f"单调温度 {mono_ok}，低温比值<0.85: {cold_ok}")
    for t, k in k_of.items():
        print(f"  T={t:5.0f}  k={k:.4f} 1/A  →  s(4A)={1/(1+k*4):.3f}  s(10A)={1/(1+k*10):.3f}")
    return summary


# ------------------------------------------------------------------ stage di

def _bin_index(abs_i: np.ndarray, i: np.ndarray):
    idx = np.clip(np.digitize(abs_i, IBIN_EDGES) - 1, 0, len(IBIN_EDGES) - 2)
    return idx * 2 + (i > 0).astype(int)   # 2(dir)×8(|i|) 格


def _learn_delta_di(es, is_, dirs_ok=True):
    """训练残差 → dir×|i| 格均值（空格 NaN）。"""
    grid = np.full(2 * (len(IBIN_EDGES) - 1), np.nan)
    idx = _bin_index(np.abs(is_), is_)
    e = np.asarray(es)
    for k in range(grid.size):
        m = idx == k
        if np.any(m):
            grid[k] = e[m].mean()
    return grid


def _apply_delta_di(grid, i):
    idx = _bin_index(np.abs(i), i)
    out = np.empty(len(i))
    for k in range(grid.size):
        m = idx == k
        if not np.any(m):
            continue
        g = grid[k]
        if np.isnan(g):
            # 空格：沿 |i| 最近的同方向非空格
            d, b = k % 2, k // 2
            cand = [bb for bb in range(grid.size // 2)
                    if not np.isnan(grid[bb * 2 + d])]
            if not cand:
                g = 0.0
            else:
                g = grid[min(cand, key=lambda bb: abs(bb - b)) * 2 + d]
        out[m] = g
    return out


def stage_di(dataset: str) -> dict:
    from bmsoc.ecm import open_loop_from_truth  # noqa: F401
    folds, best, _ocv = fold_residuals(dataset)
    keys = sorted(folds)
    rows = []
    # HPPC 饱和形状（仅 panasonic）
    k_of = None
    if dataset == "panasonic":
        hp = os.path.join(MET, "residual_r2_hppc.json")
        if os.path.exists(hp):
            with open(hp, encoding="utf-8") as f:
                k_of = {float(k): v for k, v in json.load(f)["k_per_temp"].items()}
    for key in keys:
        f = folds[key]
        temp, prof = f["temp_c"], f["profile"]
        train_same = [folds[k] for k in keys if k != key and folds[k]["temp_c"] == temp]
        train_all = [folds[k] for k in keys if k != key]
        e0 = f["e"]
        rec = {"dataset": dataset, "variant": best, "temp_c": temp, "held_out": prof,
               "n_val": int(len(e0)),
               "base_rmse_mv": float(np.sqrt(np.mean(e0 ** 2)) * 1e3),
               "base_bias_mv": float(np.mean(e0) * 1e3)}
        # ① 同温度训练折 Δ(i)
        if train_same:
            g = _learn_delta_di(np.concatenate([t["e"] for t in train_same]),
                                np.concatenate([t["i"] for t in train_same]))
            d = _apply_delta_di(g, f["i"])
            e1 = e0 - d
            # 覆盖率：留出样本所在格在训练格里非空的占比
            idx = _bin_index(np.abs(f["i"]), f["i"])
            hit = np.mean(~np.isnan(g[idx])) if g.size else 0.0
            rec.update({
                "same_rmse_mv": float(np.sqrt(np.mean(e1 ** 2)) * 1e3),
                "same_bias_mv": float(np.mean(e1) * 1e3),
                "same_bin_hit": float(hit),
            })
        # ② 全温度训练折 Δ(i)（跨温度迁移）
        g2 = _learn_delta_di(np.concatenate([t["e"] for t in train_all]),
                             np.concatenate([t["i"] for t in train_all]))
        d2 = _apply_delta_di(g2, f["i"])
        e2 = e0 - d2
        idx = _bin_index(np.abs(f["i"]), f["i"])
        rec["all_rmse_mv"] = float(np.sqrt(np.mean(e2 ** 2)) * 1e3)
        rec["all_bin_hit"] = float(np.mean(~np.isnan(g2[idx])))
        # ③ HPPC 饱和形状 + 训练折单参数尺度
        if k_of is not None:
            k_t = k_of.get(temp, min(k_of.values()))
            gvec = (1.0 / (1.0 + k_t * np.abs(f["i"])) - 1.0) * f["i"]
            # c 在同温度训练折上拟合
            if train_same:
                te = np.concatenate([t["e"] for t in train_same])
                ti = np.concatenate([t["i"] for t in train_same])
                tg = (1.0 / (1.0 + k_t * np.abs(ti)) - 1.0) * ti
                denom = float(np.dot(tg, tg))
                c = -float(np.dot(te, tg)) / denom if denom > 1e-12 else 0.0
            else:
                c = 0.0
            e3 = e0 + c * gvec
            rec.update({
                "hppc_c_mohm": c * 1e3,
                "hppc_rmse_mv": float(np.sqrt(np.mean(e3 ** 2)) * 1e3),
                "hppc_bias_mv": float(np.mean(e3) * 1e3),
            })
        rows.append(rec)
        print(f"[{temp:5.0f}] {prof:<5s} 基线 {rec['base_rmse_mv']:5.1f}  "
              f"Δ(i)同温 {rec.get('same_rmse_mv', float('nan')):5.1f} "
              f"(命中 {rec.get('same_bin_hit', 0):.0%})  "
              f"Δ(i)全温 {rec['all_rmse_mv']:5.1f}  "
              f"HPPC {rec.get('hppc_rmse_mv', float('nan')):5.1f} mV")
    df = pd.DataFrame(rows)
    out = os.path.join(MET, f"residual_r2_{dataset}_di_folds.parquet")
    df.to_parquet(out, index=False)
    methods = [c for c in ("same_rmse_mv", "all_rmse_mv", "hppc_rmse_mv") if c in df]
    best_col = min(methods, key=lambda c: df[c].median())
    summary = {
        "dataset": dataset, "n_folds": int(len(df)), "best_variant": best,
        "base_median_mv": float(df.base_rmse_mv.median()),
        "same_median_mv": float(df.same_rmse_mv.median()) if "same_rmse_mv" in df else None,
        "all_median_mv": float(df.all_rmse_mv.median()),
        "hppc_median_mv": float(df.hppc_rmse_mv.median()) if "hppc_rmse_mv" in df else None,
        "best_method": best_col,
        "best_median_mv": float(df[best_col].median()),
        "median_bin_hit_same": float(df.same_bin_hit.median()) if "same_bin_hit" in df else None,
        "median_bin_hit_all": float(df.all_bin_hit.median()),
    }
    with open(os.path.join(MET, f"residual_r2_{dataset}_di.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"=== {dataset} Δ(i) 判决：基线 {summary['base_median_mv']:.1f} → "
          f"最优 {summary['best_median_mv']:.1f} mV（{best_col}） ===")
    return summary


# ----------------------------------------------------------------- stage dst

def stage_dst(dataset: str = "calce") -> dict:
    folds, best, _ocv = fold_residuals(dataset)
    keys = sorted(folds)
    rows = []
    for key in keys:
        f = folds[key]
        temp, prof = f["temp_c"], f["profile"]
        train_all = [folds[k] for k in keys if k != key]
        train_same = [folds[k] for k in keys if k != key and folds[k]["temp_c"] == temp]
        e0 = f["e"]
        rec = {"dataset": dataset, "variant": best, "temp_c": temp, "held_out": prof,
               "n_val": int(len(e0)),
               "base_rmse_mv": float(np.sqrt(np.mean(e0 ** 2)) * 1e3)}
        # 全局 Δ(soc)：跨温度、严格 LOO
        xs = np.concatenate([t["soc"] for t in train_all])
        es = np.concatenate([t["e"] for t in train_all])
        idx = np.clip(np.digitize(xs, SOC_BINS) - 1, 0, len(SOC_BINS) - 2)
        d = np.array([es[idx == k].mean() if np.any(idx == k) else 0.0
                      for k in range(len(SOC_BINS) - 1)])
        centers = 0.5 * (SOC_BINS[:-1] + SOC_BINS[1:])
        pred = np.interp(f["soc"], centers, d)
        e1 = e0 - pred
        rec["global_rmse_mv"] = float(np.sqrt(np.mean(e1 ** 2)) * 1e3)
        rec["global_bias_mv"] = float(np.mean(e1) * 1e3)
        # 全局 Δ(soc) + 同温度偏置（训练折残差在全局修正后的均值）
        if train_same:
            off = float(np.mean(np.concatenate([t["e"] for t in train_same])
                                - np.interp(np.concatenate([t["soc"] for t in train_same]),
                                            centers, d)))
            e2 = e0 - pred - off
            rec["global_tbias_rmse_mv"] = float(np.sqrt(np.mean(e2 ** 2)) * 1e3)
            rec["tbias_mv"] = off * 1e3
        # 同温度 Δ(soc)（v1.4 机制，对照基线）
        if train_same:
            xs2 = np.concatenate([t["soc"] for t in train_same])
            es2 = np.concatenate([t["e"] for t in train_same])
            idx2 = np.clip(np.digitize(xs2, SOC_BINS) - 1, 0, len(SOC_BINS) - 2)
            d2 = np.array([es2[idx2 == k].mean() if np.any(idx2 == k) else 0.0
                           for k in range(len(SOC_BINS) - 1)])
            e3 = e0 - np.interp(f["soc"], centers, d2)
            rec["same_rmse_mv"] = float(np.sqrt(np.mean(e3 ** 2)) * 1e3)
        rows.append(rec)
        print(f"[{temp:5.0f}] {prof:<5s} 基线 {rec['base_rmse_mv']:5.1f}  "
              f"全局Δ {rec['global_rmse_mv']:5.1f}  "
              f"全局+T偏置 {rec.get('global_tbias_rmse_mv', float('nan')):5.1f}  "
              f"同温Δ {rec.get('same_rmse_mv', float('nan')):5.1f} mV")
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(MET, f"residual_r2_{dataset}_dst_folds.parquet"), index=False)
    comp = "global_tbias_rmse_mv" if "global_tbias_rmse_mv" in df else "global_rmse_mv"
    wins = int((df[comp] < df.same_rmse_mv).sum()) if "same_rmse_mv" in df else 0
    summary = {
        "dataset": dataset, "n_folds": int(len(df)), "best_variant": best,
        "base_median_mv": float(df.base_rmse_mv.median()),
        "global_median_mv": float(df.global_rmse_mv.median()),
        "global_tbias_median_mv": float(df[comp].median()),
        "same_median_mv": float(df.same_rmse_mv.median()) if "same_rmse_mv" in df else None,
        "global_wins_vs_same": wins,
    }
    with open(os.path.join(MET, f"residual_r2_{dataset}_dst.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"=== {dataset} 全局Δ(soc)：基线 {summary['base_median_mv']:.1f} → "
          f"全局 {summary['global_median_mv']:.1f} / +T偏置 {summary['global_tbias_median_mv']:.1f} mV"
          f"（同温 {summary['same_median_mv']}，胜 {wins}/{len(df)} 折） ===")
    return summary


# --------------------------------------------------------------- stage therm

def _r0_curve(dataset: str) -> tuple:
    """R(T) 相对曲线：返回 (temps, r0_medians)。NCA 用 HPPC 联合拟合 r0；
    LFP 无多倍率 HPPC，用逐温度折拟合 r0_05 中位。"""
    if dataset == "panasonic":
        hp = pd.read_parquet(os.path.join(MET, "hppc_pulses.parquet"))
        g = hp.groupby("temp_c").r0.median()
    else:
        up = pd.read_parquet(os.path.join(MET, f"model_upgrade_{dataset}_folds.parquet"))
        g = up.groupby("temp_c").r0_05.median()
    return g.index.to_numpy(float), g.to_numpy(float)


def stage_therm(dataset: str) -> dict:
    """热参数失配修正（H19）：OCV(T_meas) + R(T_meas) 逐样本替换恒定名义温度。

    残差机制：大功率段电芯自热（NCA 实测摆 20 K），模型恒参数在 T_meas 下
    过预测压降。修正零自由参数（R(T) 曲线 × 折内 r0_fit 比例缩放），
    另有训练折单参数定标变体。
    """
    from bmsoc.ecm import _unpack_vec
    folds, best, ocv2d = fold_residuals(dataset)
    keys = sorted(folds)
    temps_r, r0_med = _r0_curve(dataset)

    def ratio(t_arr, t_nom):
        """R(T_meas)/R(T_nom)，线性插值 + 端点钳制。"""
        r = np.interp(t_arr, temps_r, r0_med)
        r_nom = float(np.interp(t_nom, temps_r, r0_med))
        return r / max(r_nom, 1e-9)

    rows = []
    for key in keys:
        f = folds[key]
        temp, prof = f["temp_c"], f["profile"]
        e0, i, soc, T = f["e"], f["i"], f["soc"], f["T"]
        t_nom = temp
        r0k = _unpack_vec(f["vec"])[0]
        r0_soc = np.interp(soc, (0.1, 0.5, 0.9), r0k)
        rec = {"dataset": dataset, "variant": best, "temp_c": temp, "held_out": prof,
               "n_val": int(len(e0)),
               "base_rmse_mv": float(np.sqrt(np.mean(e0 ** 2)) * 1e3),
               "t_swing_k": float(T.max() - T.min())}
        # ① 仅 OCV(T)
        doccv = (np.asarray(ocv2d.batch(soc, T), float)
                 - np.asarray(ocv2d.batch(soc, np.full_like(T, t_nom)), float))
        e_ocv = e0 + doccv
        rec["ocvt_rmse_mv"] = float(np.sqrt(np.mean(e_ocv ** 2)) * 1e3)
        # ② 仅 R(T)（比例曲线 × 折内 r0(soc)，零自由参数）
        dr = r0_soc * (ratio(T, t_nom) - 1.0)
        e_rt = e0 + dr * i
        rec["rt_rmse_mv"] = float(np.sqrt(np.mean(e_rt ** 2)) * 1e3)
        rec["rt_bias_mv"] = float(np.mean(e_rt) * 1e3)
        # ③ OCV(T) + R(T)
        e_full = e0 + doccv + dr * i
        rec["full_rmse_mv"] = float(np.sqrt(np.mean(e_full ** 2)) * 1e3)
        rec["full_bias_mv"] = float(np.mean(e_full) * 1e3)
        # ③b 归一化 R(T)：R_ref = 训练折 i² 加权平均 R(T)，修正零加权均值
        # （r0_fit 是训练段热平均而非 T_nom 值；直接比 T_nom 会给温和折加系统偏置）
        train_same = [folds[k] for k in keys if k != key and folds[k]["temp_c"] == temp]
        if train_same:
            tw = np.concatenate([t["T"] for t in train_same])
            iw = np.concatenate([t["i"] for t in train_same])
            w = iw ** 2
            r_ref = float(np.sum(ratio(tw, t_nom) * w) / max(np.sum(w), 1e-12))
            dr_n = r0_soc * (ratio(T, t_nom) / max(r_ref, 1e-9) - 1.0)
            e_n = e0 + doccv + dr_n * i
            rec["rtnorm_rmse_mv"] = float(np.sqrt(np.mean(e_n ** 2)) * 1e3)
            rec["rtnorm_bias_mv"] = float(np.mean(e_n) * 1e3)
            rec["r_ref"] = float(r_ref)
        if train_same:
            te = np.concatenate([t["e"] for t in train_same])
            ti = np.concatenate([t["i"] for t in train_same])
            ts = np.concatenate([t["soc"] for t in train_same])
            tT = np.concatenate([t["T"] for t in train_same])
            tvec_r0 = np.concatenate([
                np.interp(t["soc"], (0.1, 0.5, 0.9), _unpack_vec(t["vec"])[0])
                for t in train_same])
            tg = tvec_r0 * (ratio(tT, t_nom) - 1.0) * ti
            tdoccv = (np.asarray(ocv2d.batch(ts, tT), float)
                      - np.asarray(ocv2d.batch(ts, np.full_like(tT, t_nom)), float))
            resid_t = te + tdoccv          # OCV(T) 修正后再定标 R 项
            denom = float(np.dot(tg, tg))
            c = -float(np.dot(resid_t, tg)) / denom if denom > 1e-12 else 0.0
            e_c = e0 + doccv + c * dr * i
            rec["c_scale"] = float(c)
            rec["scaled_rmse_mv"] = float(np.sqrt(np.mean(e_c ** 2)) * 1e3)
            rec["scaled_bias_mv"] = float(np.mean(e_c) * 1e3)
        rows.append(rec)
        print(f"[{temp:5.0f}] {prof:<5s} 摆幅 {rec['t_swing_k']:4.1f}K  基线 "
              f"{rec['base_rmse_mv']:5.1f}  R(T) {rec['rt_rmse_mv']:5.1f}  "
              f"归一 {rec.get('rtnorm_rmse_mv', float('nan')):5.1f}  "
              f"全量 {rec['full_rmse_mv']:5.1f}  "
              f"定标 {rec.get('scaled_rmse_mv', float('nan')):5.1f} mV")
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(MET, f"residual_r2_{dataset}_therm_folds.parquet"),
                  index=False)
    methods = [c for c in ("ocvt_rmse_mv", "rt_rmse_mv", "full_rmse_mv",
                           "rtnorm_rmse_mv", "scaled_rmse_mv")
               if c in df and df[c].notna().any()]
    best_col = min(methods, key=lambda c: df[c].median())
    summary = {
        "dataset": dataset, "n_folds": int(len(df)), "best_variant": best,
        "base_median_mv": float(df.base_rmse_mv.median()),
        "ocvt_median_mv": float(df.ocvt_rmse_mv.median()),
        "rt_median_mv": float(df.rt_rmse_mv.median()),
        "full_median_mv": float(df.full_rmse_mv.median()),
        "rtnorm_median_mv": float(df.rtnorm_rmse_mv.median()) if "rtnorm_rmse_mv" in df else None,
        "scaled_median_mv": float(df.scaled_rmse_mv.median()) if "scaled_rmse_mv" in df else None,
        "best_method": best_col, "best_median_mv": float(df[best_col].median()),
        "us06_base_mv": float(df[df.held_out == "us06"].base_rmse_mv.median()) if "us06" in set(df.held_out) else None,
        "us06_best_mv": float(df[df.held_out == "us06"][best_col].median()) if "us06" in set(df.held_out) else None,
        "median_c": float(df.c_scale.median()) if "c_scale" in df else None,
        "median_swing_k": float(df.t_swing_k.median()),
    }
    with open(os.path.join(MET, f"residual_r2_{dataset}_therm.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"=== {dataset} 热修正判决：基线 {summary['base_median_mv']:.1f} → "
          f"最优 {summary['best_median_mv']:.1f} mV（{best_col}）===")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["hppc", "di", "dst", "therm"], required=True)
    ap.add_argument("--dataset", choices=["calce", "panasonic"], default="panasonic")
    args = ap.parse_args()
    if args.stage == "hppc":
        stage_hppc()
    elif args.stage == "di":
        stage_di(args.dataset)
    elif args.stage == "therm":
        stage_therm(args.dataset)
    else:
        stage_dst(args.dataset)


if __name__ == "__main__":
    main()
