"""二阶/三阶 RC 等效电路（Thevenin 多极化）模型与参数辨识。

符号约定
--------
**充电电流为正、放电为负**（与两个原始数据集一致）。

状态 ``x = [soc, u1, u2(, u3)]``，``u`` 为极化过电位（充电时为正），离散化为精确零阶保持：

.. math::

    soc_{k+1} &= soc_k + \\frac{\\eta\\, i_k \\Delta t}{3600\\, Q} \\\\
    u_{j,k+1} &= e_j\\, u_{j,k} + R_j (1-e_j)\\, i_k, \\quad e_j = e^{-\\Delta t/\\tau_j} \\\\
    V_k &= OCV(soc_k) + \\sum_j u_{j,k} + R_0(i)\\, i_k

``R0(i) = r0 + r0_cur·|i|`` 是欧姆内阻（瞬时，可选电流相关项，高倍率下内阻随
电流上升）；``R1/τ1`` 是电化学极化（秒级），``R2/τ2`` 是浓差极化（分钟级）；
``r3/tau3`` 是**可选的第三条慢支路**（十分钟级），``r3=0`` 时退回二阶 RC。
残差摸底（脚本 19）显示 LOO 残差的 58–85% 在 >60 s 的慢频带、且 τ2 普遍顶在
上界（NCA 17/24、CALCE 24/24 折）——第三条慢支路就是为此加的。

参数辨识
--------
HPPC 五脉冲测试：每个 SOC 点做 1C/2C/4C/8C/12C 各 10 s 脉冲，间隔 20 min 静置。
对「脉冲 + 其后静置」窗口做最小二乘，OCV 取脉冲前的静置末电压。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from .ocv import OCVCurve


@dataclass
class RCParams:
    """单点（单 SOC、单温度）RC 参数。

    ``dv`` 是 OCV 基准偏移（V）：HPPC 静置法与动态工况之间的系统差。
    它必须一路带进滤波器，否则低温下 100 mV 级的偏移会被算成 SOC 偏差。

    ``r3/tau3``：第三条慢极化支路；``r3=0``（默认）关闭，模型退化为二阶 RC。
    ``r0_cur``：电流相关内阻系数（Ω/A），欧姆项为 ``r0·i + r0_cur·i·|i|``。
    """

    r0: float
    r1: float
    tau1: float
    r2: float
    tau2: float
    dv: float = 0.0
    r3: float = 0.0
    tau3: float = 1.0
    r0_cur: float = 0.0

    @property
    def n_rc(self) -> int:
        return 3 if self.r3 > 0 else 2

    def as_array(self) -> np.ndarray:
        """兼容旧接口的 5 元向量（前两条支路）。"""
        return np.array([self.r0, self.r1, self.tau1, self.r2, self.tau2], dtype=float)

    @staticmethod
    def from_array(a) -> "RCParams":
        vals = [float(v) for v in a]
        dv = vals[5] if len(vals) > 5 else 0.0
        r3 = vals[6] if len(vals) > 6 else 0.0
        tau3 = vals[7] if len(vals) > 7 else 1.0
        r0_cur = vals[8] if len(vals) > 8 else 0.0
        return RCParams(*vals[:5], dv=dv, r3=r3, tau3=tau3, r0_cur=r0_cur)

    def to_dict(self) -> Dict[str, float]:
        return {"r0": self.r0, "r1": self.r1, "tau1": self.tau1, "r2": self.r2,
                "tau2": self.tau2, "dv": self.dv,
                "r3": self.r3, "tau3": self.tau3, "r0_cur": self.r0_cur}


def _ocv_at(ocv, soc, temp_c):
    """统一取 OCV：支持 OCV2D（带 batch）与一维 OCVCurve。"""
    s = np.asarray(soc, dtype=float)
    t = np.asarray(temp_c, dtype=float)
    if hasattr(ocv, "batch"):
        tt = np.full(s.shape, float(t)) if t.ndim == 0 else np.broadcast_to(t, s.shape)
        return ocv.batch(s, tt)
    if t.ndim == 0:
        return ocv(s)
    return np.array([float(ocv(np.array([x]))[0]) for x in s])


def step_matrices(p: RCParams, dt: float) -> Tuple[float, ...]:
    """返回 ``(a1, b1, a2, b2[, a3, b3])``：``u_j[k+1] = a_j u_j[k] + b_j i[k]``。

    ``p.r3 > 0`` 时追加第三支路的 (a3, b3)，否则保持 4 元组（二阶兼容）。
    """
    e1 = float(np.exp(-dt / max(p.tau1, 1e-6)))
    e2 = float(np.exp(-dt / max(p.tau2, 1e-6)))
    out = [e1, p.r1 * (1 - e1), e2, p.r2 * (1 - e2)]
    if getattr(p, "r3", 0.0) > 0:
        e3 = float(np.exp(-dt / max(p.tau3, 1e-6)))
        out += [e3, p.r3 * (1 - e3)]
    return tuple(out)


def r0_eff(p: RCParams, current) -> float | np.ndarray:
    """有效欧姆内阻 ``r0 + r0_cur·|i|``（电流相关项 ``r3=0`` 时为常数）。"""
    i = np.asarray(current, dtype=float)
    return p.r0 + getattr(p, "r0_cur", 0.0) * np.abs(i)


def terminal_voltage(p: RCParams, ocv: float | np.ndarray, current: float | np.ndarray,
                     u1: float | np.ndarray, u2: float | np.ndarray,
                     u3: float | np.ndarray = 0.0) -> np.ndarray:
    i = np.asarray(current, dtype=float)
    return ocv + u1 + u2 + u3 + r0_eff(p, i) * i + getattr(p, "dv", 0.0)


def simulate(p: RCParams, ocv, current_a: np.ndarray, dt: float,
             soc0: float = 1.0, capacity_ah: float = 2.9, temp_c=25.0,
             eta: float = 1.0, u0: Optional[Tuple[float, ...]] = None) -> Dict[str, np.ndarray]:
    """用真值电流驱动模型，返回 soc/u1/u2(/u3)/电压预测（开环）。"""
    i = np.asarray(current_a, dtype=float)
    n = i.size
    ab = step_matrices(p, dt)
    nr = getattr(p, "n_rc", 2)
    soc = np.empty(n)
    us = [np.empty(n) for _ in range(nr)]
    soc[0] = soc0
    u_init = u0 if u0 is not None else tuple(0.0 for _ in range(nr))
    for j in range(nr):
        us[j][0] = float(u_init[j])
    for k in range(n - 1):
        soc[k + 1] = soc[k] + eta * i[k] * dt / (3600.0 * capacity_ah)
        for j in range(nr):
            a, b = ab[2 * j], ab[2 * j + 1]
            us[j][k + 1] = a * us[j][k] + b * i[k]
    v = terminal_voltage(p, _ocv_at(ocv, soc, temp_c), i, us[0], us[1],
                         us[2] if nr > 2 else 0.0)
    out = {"soc": soc, "u1": us[0], "u2": us[1], "voltage": v}
    if nr > 2:
        out["u3"] = us[2]
    return out


def simulate_tv(p: RCParams, ocv, current_a: np.ndarray, dt: float,
                q_arr, r0_arr, soc0: float = 1.0, temp_c=25.0,
                eta: float = 1.0, u0: Optional[Tuple[float, float]] = None,
                residual_v: float | np.ndarray = 0.0) -> Dict[str, np.ndarray]:
    """**时变**容量/内阻的前向仿真 —— 老化耦合实验（脚本 17）的「植物」。

    与 :func:`simulate` 只有两处不同：

    1. SOC 积分用逐步容量 ``q_arr[k]``（对应 ``Q_ref`` 随老化下降），
       真值 SOC = ``soc0 + ∫i dτ / (3600·q(τ))``；
    2. 端电压欧姆项用逐步内阻 ``r0_arr[k]``（老化使 R0 上升）。

    ``q_arr``/``r0_arr`` 可以是与 ``current_a`` 同长的数组，也可以是标量。
    ``residual_v`` 注入慢变电压残差（V），模拟真实开环残差（38–53 mV 量级），
    避免「模型精确」的温室让滤波器表现虚高。
    """
    i = np.asarray(current_a, dtype=float)
    n = i.size
    q = np.broadcast_to(np.asarray(q_arr, dtype=float), (n,)).astype(float).copy()
    r0 = np.broadcast_to(np.asarray(r0_arr, dtype=float), (n,)).astype(float).copy()
    if np.any(q <= 0):
        raise ValueError("q_arr 必须为正")
    ab = step_matrices(p, dt)
    nr = getattr(p, "n_rc", 2)
    soc = np.empty(n)
    us = [np.empty(n) for _ in range(nr)]
    soc[0] = soc0
    u_init = u0 if u0 is not None else tuple(0.0 for _ in range(nr))
    for j in range(nr):
        us[j][0] = float(u_init[j])
    for k in range(n - 1):
        soc[k + 1] = soc[k] + eta * i[k] * dt / (3600.0 * q[k])
        for j in range(nr):
            a, b = ab[2 * j], ab[2 * j + 1]
            us[j][k + 1] = a * us[j][k] + b * i[k]
    # 时变 R0（数组）时忽略 r0_cur：r0_arr 已含逐步总内阻；常数 r0 才叠加电流项
    r0_use = r0 * i
    if getattr(p, "r0_cur", 0.0) > 0 and np.isscalar(r0_arr):
        r0_use = r0 * i + p.r0_cur * i * np.abs(i)
    v = (_ocv_at(ocv, soc, temp_c) + us[0] + us[1] + (us[2] if nr > 2 else 0.0) + r0_use
         + getattr(p, "dv", 0.0) + np.broadcast_to(np.asarray(residual_v, float), (n,)))
    out = {"soc": soc, "u1": us[0], "u2": us[1], "voltage": v, "q": q, "r0": r0}
    if nr > 2:
        out["u3"] = us[2]
    return out


def model_voltage(p: RCParams, ocv: float, current_a: np.ndarray, dt: float) -> np.ndarray:
    """OCV 固定时，脉冲窗口内的电压轨迹（辨识用）。

    与 :func:`terminal_voltage` 保持同一套电压构成（含 ``dv`` 与 ``r0_cur``），
    否则同一组参数在两条路径上给出的电压会差一个常数。
    """
    i = np.asarray(current_a, dtype=float)
    ab = step_matrices(p, dt)
    nr = getattr(p, "n_rc", 2)
    dv = getattr(p, "dv", 0.0)
    n = i.size
    us = [0.0] * nr
    v = np.empty(n)
    for k in range(n):
        v[k] = ocv + sum(us) + r0_eff(p, i[k]) * i[k] + dv
        for j in range(nr):
            a, b = ab[2 * j], ab[2 * j + 1]
            us[j] = a * us[j] + b * i[k]
    return v


_DEFAULT0 = np.array([0.05, 0.02, 20.0, 0.02, 400.0])  # R0, R1, tau1, R2, tau2
_LB = np.array([1e-4, 1e-5, 0.5, 1e-5, 10.0])
_UB = np.array([2.0, 2.0, 200.0, 2.0, 5000.0])


def identify_pulse(t_s: np.ndarray, current_a: np.ndarray, voltage_v: np.ndarray,
                   ocv_value: float, dt: Optional[float] = None,
                   x0: Optional[np.ndarray] = None) -> Tuple[RCParams, Dict[str, float]]:
    """对「脉冲 + 静置」窗口辨识二阶 RC 参数。

    ``ocv_value`` 取脉冲前的静置末电压（假定已弛豫）。
    """
    t = np.asarray(t_s, dtype=float)
    i = np.asarray(current_a, dtype=float)
    v = np.asarray(voltage_v, dtype=float)
    dt = float(dt if dt is not None else np.median(np.diff(t)))
    x0 = _DEFAULT0 if x0 is None else np.clip(np.asarray(x0, float), _LB, _UB)

    def resid(theta: np.ndarray) -> np.ndarray:
        p = RCParams.from_array(theta)
        return model_voltage(p, ocv_value, i, dt) - v

    sol = least_squares(resid, x0, bounds=(_LB, _UB), loss="soft_l1", f_scale=0.01,
                        max_nfev=400)
    p = RCParams.from_array(sol.x)
    r = resid(sol.x)
    info = {
        "rmse_v": float(np.sqrt(np.mean(r ** 2))),
        "max_abs_v": float(np.max(np.abs(r))),
        "n": int(i.size),
        "cost": float(sol.cost),
        "success": bool(sol.success),
    }
    return p, info


def time_constant_split(p: RCParams) -> float:
    """τ1/τ2 的分离度；<3 说明两个 RC 难以区分，参数不可辨识。"""
    return float(max(p.tau1, p.tau2) / max(min(p.tau1, p.tau2), 1e-9))


# 变量投影用的时间常数格点（秒）。1 s 采样、10 s 脉冲下 τ1 分辨力有限，
# 因此不在连续空间里拟合 τ，而是格点搜索 + 线性最小二乘求 R。
TAU1_GRID = np.array([2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 27.0, 40.0, 60.0])
TAU2_GRID = np.array([80.0, 120.0, 180.0, 270.0, 400.0, 600.0, 900.0, 1300.0, 2000.0])


def _rc_regressor(i: np.ndarray, tau: float, dt: float) -> np.ndarray:
    """单位 R 的极化响应 g(k)，满足 u(k) = R·g(k)。"""
    a = float(np.exp(-dt / tau))
    b = 1.0 - a
    g = np.zeros_like(i)
    acc = 0.0
    for k in range(i.size - 1):
        acc = a * acc + b * i[k]
        g[k + 1] = acc
    return g


def identify_pulse_grid(t_s: np.ndarray, current_a: np.ndarray, voltage_v: np.ndarray,
                        ocv_value: float, dt: Optional[float] = None,
                        tau1_grid: Optional[np.ndarray] = None,
                        tau2_grid: Optional[np.ndarray] = None) -> Tuple[RCParams, Dict[str, float]]:
    """变量投影法辨识二阶 RC：τ 走格点，R0/R1/R2 用非负最小二乘。

    比 5 参数非线性最小二乘稳定得多：当脉冲只有 10 s、采样 1 s 时，
    τ1 与 R1 存在强共线性，直接拟合会让 τ 在上下界之间乱跳。
    """
    from scipy.optimize import nnls

    t = np.asarray(t_s, dtype=float)
    i = np.asarray(current_a, dtype=float)
    v = np.asarray(voltage_v, dtype=float)
    dt = float(dt if dt is not None else np.median(np.diff(t)))
    tau1_grid = TAU1_GRID if tau1_grid is None else np.asarray(tau1_grid, float)
    tau2_grid = TAU2_GRID if tau2_grid is None else np.asarray(tau2_grid, float)

    y = v - ocv_value
    best = None
    for t1 in tau1_grid:
        g1 = _rc_regressor(i, float(t1), dt)
        for t2 in tau2_grid:
            if t2 <= t1 * 1.5:
                continue
            g2 = _rc_regressor(i, float(t2), dt)
            X = np.column_stack([i, g1, g2])
            beta, rnorm = nnls(X, y)
            sse = float(rnorm ** 2)
            if best is None or sse < best[0]:
                best = (sse, beta, float(t1), float(t2))
    if best is None:
        raise RuntimeError("格点全部无效")
    sse, beta, t1, t2 = best
    p = RCParams(float(beta[0]), float(beta[1]), t1, float(beta[2]), t2)
    resid = model_voltage(p, ocv_value, i, dt) - v
    return p, {
        "rmse_v": float(np.sqrt(np.mean(resid ** 2))),
        "max_abs_v": float(np.max(np.abs(resid))),
        "n": int(i.size),
        "sse": sse,
        "method": "grid+nnls",
    }


def identify_stack(windows, tau1_grid: Optional[np.ndarray] = None,
                   tau2_grid: Optional[np.ndarray] = None) -> Tuple[RCParams, Dict[str, float]]:
    """把同一 SOC 档的多个脉冲窗口**联合**辨识（共享 R0/R1/τ1/R2/τ2）。

    单个 10 s 脉冲在 1 s 采样下信息量太小：τ1 与 R1 强共线，拟合会给出
    "τ1=2 s、R1 很大" 这类等价解，跨倍率中位数也随之乱跳。把 4–5 个倍率
    叠在一起做一次变量投影，条件数显著改善，参数才稳定。

    ``windows``：``[{"i": ..., "v": ..., "ocv": float, "dt": float}, ...]``
    """
    from scipy.optimize import nnls

    if not windows:
        raise ValueError("没有可用窗口")
    tau1_grid = TAU1_GRID if tau1_grid is None else np.asarray(tau1_grid, float)
    tau2_grid = TAU2_GRID if tau2_grid is None else np.asarray(tau2_grid, float)

    ys = [np.asarray(w["v"], float) - float(w["ocv"]) for w in windows]
    i_arrs = [np.asarray(w["i"], float) for w in windows]
    dts = [float(w["dt"]) for w in windows]

    best = None
    for t1 in tau1_grid:
        for t2 in tau2_grid:
            if t2 <= t1 * 1.5:
                continue
            cols_i, cols_g1, cols_g2 = [], [], []
            for i_k, dt_k in zip(i_arrs, dts):
                cols_i.append(i_k)
                cols_g1.append(_rc_regressor(i_k, float(t1), dt_k))
                cols_g2.append(_rc_regressor(i_k, float(t2), dt_k))
            X = np.column_stack([np.concatenate(cols_i), np.concatenate(cols_g1),
                                 np.concatenate(cols_g2)])
            y = np.concatenate(ys)
            beta, rnorm = nnls(X, y)
            sse = float(rnorm ** 2)
            if best is None or sse < best[0]:
                best = (sse, beta, float(t1), float(t2))

    sse, beta, t1, t2 = best
    p = RCParams(float(beta[0]), float(beta[1]), t1, float(beta[2]), t2)
    rmse = []
    for w in windows:
        resid = model_voltage(p, float(w["ocv"]), np.asarray(w["i"], float), float(w["dt"])) - np.asarray(w["v"], float)
        rmse.append(float(np.sqrt(np.mean(resid ** 2))))
    return p, {
        "rmse_v": float(np.mean(rmse)),
        "rmse_max_v": float(np.max(rmse)),
        "n": int(sum(i.size for i in i_arrs)),
        "n_windows": len(windows),
        "sse": sse,
        "method": "stacked-grid+nnls",
    }


# ---------------------------------------------------------------------------
# 快速仿真与整段工况拟合
# ---------------------------------------------------------------------------
def _rc_lfilter(r: float, tau: float, i: np.ndarray, dt: float) -> np.ndarray:
    """RC 支路的零阶保持递推，**与 :func:`simulate` 同相位**。

    ``simulate`` 里是 ``u[k+1] = a u[k] + b i[k]``，即 ``u[k]`` 只依赖 ``i[k-1]``。
    ``lfilter([b], [1, -a], i)`` 给出的是 ``u[k] = a u[k-1] + b i[k]``，比前者**超前
    一个采样**；在本数据 1 s 网格上相当于把 R0·Δi 量级的误差张冠李戴。把输入右移
    一位再滤波即可对齐。
    """
    from scipy.signal import lfilter

    a = float(np.exp(-dt / max(tau, 1e-6)))
    b = float(r * (1.0 - a))
    shifted = np.concatenate([[0.0], i[:-1]]) if i.size else i
    return lfilter([b], [1.0, -a], shifted)


def simulate_fast(r0, r1, tau1, r2, tau2, ocv, current_a: np.ndarray,
                  soc0: float, capacity_ah: float, dt: float, temp_c=25.0, dv: float = 0.0):
    """与 :func:`simulate` 等价，但用 ``lfilter`` 做 RC 递推（快 50 倍以上）。

    ``r0`` 可以是标量或与电流等长的数组（用于 SOC 相关的欧姆内阻）。
    """
    i = np.asarray(current_a, dtype=float)
    n = i.size
    soc = soc0 + np.concatenate([[0.0], np.cumsum(i[:-1])]) * dt / (3600.0 * capacity_ah)
    u1 = _rc_lfilter(r1, tau1, i, dt)
    u2 = _rc_lfilter(r2, tau2, i, dt)
    r0_arr = np.full(n, float(r0)) if np.isscalar(r0) else np.asarray(r0, dtype=float)
    v = _ocv_at(ocv, soc, temp_c) + u1 + u2 + r0_arr * i + float(dv)
    return soc, v


def _unpack_vec(vec) -> tuple:
    """把扁平参数向量拆成 ``(r0k, r1, tau1, r2, tau2, r3, tau3, r0_cur, dv)``。

    向量布局（``dv`` 恒在末尾，可选项在 ``dv`` 之前）::

        8 元  [r0k×3, r1, tau1, r2, tau2, dv]                     二阶 RC
        9 元  [r0k×3, r1, tau1, r2, tau2, r0_cur, dv]             二阶 + 电流相关 R0
        10 元 [r0k×3, r1, tau1, r2, tau2, r3, tau3, dv]           三阶 RC
        11 元 [r0k×3, r1, tau1, r2, tau2, r3, tau3, r0_cur, dv]   三阶 + 电流相关 R0
    """
    v = np.asarray(vec, dtype=float).reshape(-1)
    n = v.size
    if n not in (8, 9, 10, 11):
        raise ValueError(f"参数向量长度 {n} 不合法（应为 8/9/10/11）")
    r0k = v[0:3]
    r1, tau1, r2, tau2 = float(v[3]), float(v[4]), float(v[5]), float(v[6])
    has3 = n >= 10
    has_cur = n in (9, 11)
    r3 = float(v[7]) if has3 else 0.0
    tau3 = float(v[8]) if has3 else 1.0
    r0_cur = float(v[-2]) if has_cur else 0.0
    return r0k, r1, tau1, r2, tau2, r3, tau3, r0_cur, float(v[-1])


def fit_cycle_params(windows, soc_knots=(0.1, 0.5, 0.9), x0: Optional[np.ndarray] = None,
                     max_nfev: int = 400, dv_bound: float = 0.12, loss: str = "linear",
                     n_rc: int = 2, fit_r0_cur: bool = False):
    """在整段动态工况上拟合电气参数（OCV 固定）。

    ``windows``：``[(soc, current, voltage, dt, ocv_curve[, temp]), ...]``
    待拟合：``R0`` 在 ``soc_knots`` 上的三个节点、``R1/τ1``、``R2/τ2``、
    可选第三支路 ``R3/τ3``（``n_rc=3``）、可选电流相关内阻 ``r0_cur``
    （``fit_r0_cur=True``），以及 OCV 基准偏移 ``dv``。

    ``n_rc=3`` 时第三支路的时间常数上界放到 50000 s —— 残差摸底显示 τ2 长期
    顶在 6000 s 上界（慢极化想当积分器用），第三条支路要接住的就是它。

    返回的向量布局见 :func:`_unpack_vec`。
    """
    from scipy.optimize import least_squares

    if n_rc not in (2, 3):
        raise ValueError("n_rc 只支持 2 或 3")
    data = []
    for w in windows:
        soc, i, v, dt, curve = w[0], w[1], w[2], w[3], w[4]
        temp = w[5] if len(w) > 5 else 25.0
        data.append((np.asarray(soc, float), np.asarray(i, float), np.asarray(v, float),
                     float(dt), curve, temp))

    def resid(th):
        out = []
        for (soc, i, v, dt, curve, temp) in data:
            vp = open_loop_from_truth(th, soc, i, curve, dt, temp)
            out.append(vp - v)
        return np.concatenate(out)

    head_lo = [0.005, 0.005, 0.005, 1e-5, 0.5, 1e-5, 30.0]
    head_hi = [0.30, 0.30, 0.30, 0.20, 120.0, 0.40, 6000.0]
    head_x0 = [0.028, 0.028, 0.033, 0.008, 12.0, 0.05, 800.0]
    lo, hi, x0d = list(head_lo), list(head_hi), list(head_x0)
    if n_rc == 3:
        lo += [1e-5, 200.0]        # r3, tau3（tau3 下界 200 s，避开与 tau2 混叠）
        hi += [0.40, 50000.0]
        x0d += [0.02, 3000.0]
    if fit_r0_cur:
        lo += [0.0]
        hi += [0.02]               # Ω/A：4 A 时最多再加 0.32 V 的 i|i| 压降
        x0d += [0.0]
    lo += [-dv_bound]
    hi += [dv_bound]
    x0d += [0.0]
    lower, upper = np.array(lo), np.array(hi)
    x0 = np.array(x0d) if x0 is None else np.asarray(x0, float)
    sol = least_squares(resid, np.clip(x0, lower + 1e-9, upper - 1e-9), bounds=(lower, upper),
                        loss=loss, f_scale=0.02, max_nfev=max_nfev)
    r0k, r1, tau1, r2, tau2, r3, tau3, r0_cur, dv = _unpack_vec(sol.x)
    r = resid(sol.x)
    info = {
        "rmse_v": float(np.sqrt(np.mean(r ** 2))),
        "mae_v": float(np.mean(np.abs(r))),
        "bias_v": float(np.mean(r)),
        "max_abs_v": float(np.max(np.abs(r))),
        "dv": dv, "r0_knots": r0k, "n": int(r.size), "cost": float(sol.cost),
        "r3": r3, "tau3": tau3, "r0_cur": r0_cur, "n_rc": n_rc,
    }
    return sol.x, info


def open_loop_from_truth(vec: np.ndarray, soc: np.ndarray, current_a: np.ndarray,
                        ocv, dt: float, temp_c=25.0):
    """用真值 SOC 直接算电压（不做 SOC 积分），供残差诊断使用。

    ``vec`` 布局见 :func:`_unpack_vec`（8/9/10/11 元，兼容二阶旧向量）。
    """
    r0k, r1, tau1, r2, tau2, r3, tau3, r0_cur, dv = _unpack_vec(vec)
    i = np.asarray(current_a, float)
    u1 = _rc_lfilter(r1, tau1, i, dt)
    u2 = _rc_lfilter(r2, tau2, i, dt)
    u3 = _rc_lfilter(r3, tau3, i, dt) if r3 > 0 else 0.0
    r0 = np.interp(soc, (0.1, 0.5, 0.9), r0k)
    return (_ocv_at(ocv, soc, temp_c) + u1 + u2 + u3
            + r0 * i + r0_cur * i * np.abs(i) + dv)


class ParamGrid:
    """(温度, SOC) 网格上的参数表，双线性插值。"""

    def __init__(self, temps, socs, values: np.ndarray):
        self.temps = np.asarray(temps, dtype=float)
        self.socs = np.asarray(socs, dtype=float)
        self.values = np.asarray(values, dtype=float)  # (n_temp, n_soc, 5)

    def at(self, temp_c: float, soc: float) -> RCParams:
        ts = self.temps
        idx_t = int(np.clip(np.searchsorted(ts, temp_c), 0, ts.size - 1))
        if idx_t > 0 and abs(ts[idx_t - 1] - temp_c) < abs(ts[idx_t] - temp_c):
            idx_t -= 1
        v = self.values[idx_t]  # (n_soc, 5)
        out = np.array([np.interp(soc, self.socs, v[:, j]) for j in range(5)])
        return RCParams.from_array(out)

    def to_arrays(self):
        return self.temps, self.socs, self.values
