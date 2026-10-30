"""SOC 估计器：库仑计数基线、EKF、UKF、AEKF。

统一接口
--------
    est = EKF(ocv=ocv2d, params=rc, capacity_ah=2.6, cfg=FilterConfig(soc0=0.7))
    out = est.run(profile, current_bias_a=0.02)

状态 ``x = [soc, u1, u2, b]``：前三个是二阶 RC 的 SOC 与两条极化支路，
``b`` 是**电压偏置状态**（模型 OCV 与实测之间的系统差）。

为什么要有 ``b``
----------------
若只有三个状态，任何持续的模型偏差（本数据集上 20–50 mV）都会在稳态被解释成
「SOC 错了」，于是滤波器停在一个 4–5 个百分点的平台上——这是实测出来的地板，
不是收敛慢。把偏差显式建成状态、给它一个很小的过程噪声，新息里的常值部分就会
落到 ``b`` 上，SOC 得以继续保持库仑计数的短期精度。

三版滤波器的差别只在「怎么传播协方差」与「是否在线调噪声」：
- ``EKF``  ：一阶泰勒展开，``H = [dOCV/dSOC, 1, 1, 1]``
- ``UKF``  ：2n+1 个 sigma 点，无需求导
- ``AEKF`` ：EKF + 滑动窗新息协方差匹配，只调量测噪声 ``R``
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .ecm import RCParams, step_matrices
from .profiles import DrivingProfile

NDIM = 4          # soc, u1, u2, bias


def _ocv_scalar(ocv, soc: float, temp_c: float) -> float:
    """取 OCV：兼容 OCV2D(soc, T) 与 OCVCurve(soc)。"""
    try:
        return float(ocv(np.array([soc]), temp_c)[0])
    except TypeError:
        return float(ocv(np.array([soc]))[0])


def _docv_scalar(ocv, soc: float, temp_c: float) -> float:
    try:
        return float(ocv.docv(np.array([soc]), temp_c)[0])
    except TypeError:
        return float(ocv.docv(np.array([soc]))[0])


def _as_profile(obj) -> DrivingProfile:
    if isinstance(obj, DrivingProfile):
        return obj
    raise TypeError("需要 DrivingProfile")


@dataclass
class FilterConfig:
    """滤波器超参数（默认值对应 10 mV 量测噪声、1 s 采样）。"""

    soc0: float = 1.0
    sigma_soc0: float = 0.30           # 初始 SOC 标准差
    sigma_u0: float = 0.02             # 初始极化电压标准差 (V)
    sigma_b0: float = 0.05             # 初始偏置标准差 (V)
    sigma_v: float = 0.010             # 电压量测噪声 (V)
    q_soc: float = 1e-10               # SOC 过程噪声（每步方差）
    q_u: float = 1e-6                  # 极化电压过程噪声
    q_b: float = 1e-12                 # 偏置过程噪声（很慢的随机游走）
    eta: float = 1.0                   # 库仑效率
    u_clip: float = 0.6                # 极化电压限幅 (V)
    b_clip: float = 0.30               # 偏置限幅 (V)
    soc_clip: Tuple[float, float] = (-0.05, 1.05)


class _BaseFilter:
    name = "base"

    def __init__(self, ocv, params: RCParams, capacity_ah: float,
                 cfg: Optional[FilterConfig] = None):
        self.ocv = ocv
        self.p = params
        self.q_cap = float(capacity_ah)
        self.cfg = cfg or FilterConfig()

    # ---- 公共 ----
    def _ab(self, dt: float):
        if not hasattr(self, "_ab_cache") or self._ab_cache[0] != dt:
            self._ab_cache = (dt, step_matrices(self.p, dt))
        return self._ab_cache[1]

    def _predict(self, x: np.ndarray, i: float, dt: float, clip: bool = True) -> np.ndarray:
        c = self.cfg
        a1, b1, a2, b2 = self._ab(dt)
        soc = x[0] + c.eta * i * dt / (3600.0 * self.q_cap)
        u1 = a1 * x[1] + b1 * i
        u2 = a2 * x[2] + b2 * i
        bias = x[3]
        if not clip:      # sigma 点传播保持线性，裁剪会破坏对称性
            return np.array([soc, u1, u2, bias])
        return np.array([np.clip(soc, *c.soc_clip),
                         float(np.clip(u1, -c.u_clip, c.u_clip)),
                         float(np.clip(u2, -c.u_clip, c.u_clip)),
                         float(np.clip(bias, -c.b_clip, c.b_clip))])

    def _h(self, x: np.ndarray, i: float, temp_c: float = 25.0) -> float:
        return (_ocv_scalar(self.ocv, float(x[0]), temp_c) + x[1] + x[2]
                + self.p.r0 * i + getattr(self.p, "dv", 0.0) + x[3])

    def _H(self, x: np.ndarray, temp_c: float) -> np.ndarray:
        return np.array([[_docv_scalar(self.ocv, float(x[0]), temp_c), 1.0, 1.0, 1.0]])

    def _init_state(self, soc_init: Optional[float], i0: float = 0.0) -> np.ndarray:
        """初始极化取当前电流下的稳态值 R_j·i0，偏置置 0。"""
        s0 = self.cfg.soc0 if soc_init is None else float(soc_init)
        return np.array([s0, self.p.r1 * i0, self.p.r2 * i0, 0.0])

    def _init_cov(self) -> np.ndarray:
        c = self.cfg
        return np.diag([c.sigma_soc0 ** 2, c.sigma_u0 ** 2, c.sigma_u0 ** 2, c.sigma_b0 ** 2])

    def _Q(self) -> np.ndarray:
        c = self.cfg
        return np.diag([c.q_soc, c.q_u, c.q_u, c.q_b])

    def run(self, profile: DrivingProfile, voltage_noise: float = 0.0,
            seed: int = 0, soc_init: Optional[float] = None,
            current_bias_a: float = 0.0) -> Dict[str, np.ndarray]:
        """跑一段工况。

        ``current_bias_a``：电流传感器偏置（A）。只影响估计器看到的电流，
        真值 SOC 仍由原始电流积分，故库仑计数会随偏置线性漂移。
        """
        prof = _as_profile(profile)
        dt = prof.dt
        v_meas = prof.voltage_v
        if voltage_noise > 0:
            rng = np.random.default_rng(seed)
            v_meas = v_meas + rng.normal(0.0, voltage_noise, size=v_meas.size)
        if current_bias_a:
            prof = DrivingProfile(t_s=prof.t_s, current_a=prof.current_a + current_bias_a,
                                  voltage_v=prof.voltage_v, temp_c=prof.temp_c,
                                  soc_ref=prof.soc_ref, capacity_ah=prof.capacity_ah,
                                  meta=dict(prof.meta))
        return self._run(prof, dt, v_meas, soc_init)

    def _run(self, prof, dt, v_meas, soc_init):  # pragma: no cover - 抽象
        raise NotImplementedError

    @staticmethod
    def _pack(soc, inn, nis, gain, sig, vp, bias=None) -> Dict[str, np.ndarray]:
        out = {"soc_hat": soc, "volt_pred": vp, "innovation": inn, "nis": nis,
               "gain_soc": gain, "sigma_soc": sig}
        if bias is not None:
            out["bias_hat"] = bias
        return out


class CoulombCounter(_BaseFilter):
    """基线：只积分电流，不看电压。"""

    name = "cc"

    def _run(self, prof, dt, v_meas, soc_init):
        n = len(prof)
        soc = np.empty(n)
        soc[0] = self.cfg.soc0 if soc_init is None else float(soc_init)
        for k in range(n - 1):
            soc[k + 1] = soc[k] + self.cfg.eta * prof.current_a[k] * dt / (3600.0 * self.q_cap)
        return self._pack(np.clip(soc, *self.cfg.soc_clip), np.zeros(n), np.zeros(n),
                          np.zeros(n), np.full(n, np.nan), np.full(n, np.nan))


class EKF(_BaseFilter):
    name = "ekf"

    def _F(self, dt: float) -> np.ndarray:
        a1, _, a2, _ = self._ab(dt)
        F = np.eye(NDIM)
        F[1, 1], F[2, 2] = a1, a2
        return F

    def _run(self, prof, dt, v_meas, soc_init):
        c = self.cfg
        n = len(prof)
        i_arr, t_arr = prof.current_a, prof.temp_c
        x = self._init_state(soc_init, float(i_arr[0]) if n else 0.0)
        P = self._init_cov()
        F = self._F(dt)
        Q = self._Q()
        R = c.sigma_v ** 2
        soc = np.empty(n); inn = np.zeros(n); nis = np.zeros(n)
        gain = np.zeros(n); sig = np.empty(n); vp = np.empty(n); bh = np.zeros(n)
        for k in range(n):
            if k > 0:
                x = self._predict(x, i_arr[k - 1], dt)
                P = F @ P @ F.T + Q
            H = self._H(x, float(t_arr[k]))
            y_hat = self._h(x, i_arr[k], float(t_arr[k]))
            S = float((H @ P @ H.T)[0, 0]) + R
            K = (P @ H.T).reshape(-1) / S
            e = float(v_meas[k]) - y_hat
            x = x + K * e
            x[0] = float(np.clip(x[0], *c.soc_clip))
            x[3] = float(np.clip(x[3], -c.b_clip, c.b_clip))
            IKH = np.eye(NDIM) - np.outer(K, H)
            P = IKH @ P @ IKH.T + R * np.outer(K, K)      # Joseph 形式
            P = 0.5 * (P + P.T)
            soc[k] = x[0]; inn[k] = e; nis[k] = e * e / S
            gain[k] = K[0]; sig[k] = float(np.sqrt(max(P[0, 0], 0.0)))
            vp[k] = y_hat; bh[k] = x[3]
        return self._pack(soc, inn, nis, gain, sig, vp, bh)


class UKF(_BaseFilter):
    """无迹卡尔曼（对称 sigma 点，n=4）。"""

    name = "ukf"

    def __init__(self, *a, alpha: float = 1.0, beta: float = 2.0, kappa: float = 0.0, **kw):
        super().__init__(*a, **kw)
        self.alpha, self.beta, self.kappa = alpha, beta, kappa

    def _weights(self, n: int):
        lam = self.alpha ** 2 * (n + self.kappa) - n
        n1 = n + lam
        wm = np.full(2 * n + 1, 1.0 / (2 * n1))
        wc = wm.copy()
        wm[0] = lam / n1
        wc[0] = lam / n1 + (1 - self.alpha ** 2 + self.beta)
        return lam, wm, wc

    @staticmethod
    def _psd(P: np.ndarray) -> np.ndarray:
        """对称化 + 特征值截断，保证协方差半正定（4 维时负权重易把它压坏）。"""
        P = 0.5 * (P + P.T)
        w, V = np.linalg.eigh(P)
        if w.min() < 0:
            P = V @ np.diag(np.clip(w, 1e-14, None)) @ V.T
        return 0.5 * (P + P.T)

    def _sigma(self, mu: np.ndarray, cov: np.ndarray, scl: float) -> np.ndarray:
        try:
            L = np.linalg.cholesky(scl * cov)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(scl * (cov + np.eye(NDIM) * 1e-12))
        X = np.empty((2 * NDIM + 1, NDIM))
        X[0] = mu
        for j in range(NDIM):
            X[1 + j] = mu + L[:, j]
            X[1 + NDIM + j] = mu - L[:, j]
        return X

    def _run(self, prof, dt, v_meas, soc_init):
        c = self.cfg
        n = len(prof)
        i_arr, t_arr = prof.current_a, prof.temp_c
        x = self._init_state(soc_init, float(i_arr[0]) if n else 0.0)
        P = self._init_cov()
        Q = self._Q()
        R = c.sigma_v ** 2
        lam, wm, wc = self._weights(NDIM)
        scl = NDIM + lam
        soc = np.empty(n); inn = np.zeros(n); nis = np.zeros(n)
        gain = np.zeros(n); sig = np.empty(n); vp = np.empty(n); bh = np.zeros(n)
        for k in range(n):
            if k > 0:
                X = self._sigma(x, P, scl)
                Xp = np.array([self._predict(pt, i_arr[k - 1], dt, clip=False) for pt in X])
                x = wm @ Xp
                x[0] = float(np.clip(x[0], *c.soc_clip))
                dX = Xp - x
                P = (wc[:, None, None] * np.einsum("ij,ik->ijk", dX, dX)).sum(0) + Q
                P = self._psd(P)
            X = self._sigma(x, P, scl)
            Y = np.array([self._h(pt, i_arr[k], float(t_arr[k])) for pt in X])
            y_hat = float(wm @ Y)
            dY = Y - y_hat
            S = float((wc * dY * dY).sum()) + R
            dX = X - x
            Pxy = (wc[:, None] * dX * dY[:, None]).sum(0)
            K = Pxy / S
            e = float(v_meas[k]) - y_hat
            x = x + K * e
            x[0] = float(np.clip(x[0], *c.soc_clip))
            x[3] = float(np.clip(x[3], -c.b_clip, c.b_clip))
            P = self._psd(P - np.outer(K, Pxy))
            soc[k] = x[0]; inn[k] = e; nis[k] = e * e / S
            gain[k] = K[0]; sig[k] = float(np.sqrt(max(P[0, 0], 0.0)))
            vp[k] = y_hat; bh[k] = x[3]
        return self._pack(soc, inn, nis, gain, sig, vp, bh)


class AEKF(EKF):
    """自适应 EKF：滑动窗新息协方差匹配，只调量测噪声 R。"""

    name = "aekf"

    def __init__(self, *a, window: int = 60, r_min: float = 4e-4, r_max: float = 0.09,
                 adapt_every: int = 10, **kw):
        super().__init__(*a, **kw)
        self.window = int(window)
        self.r_min, self.r_max = float(r_min), float(r_max)
        self.adapt_every = int(adapt_every)

    def _run(self, prof, dt, v_meas, soc_init):
        c = self.cfg
        n = len(prof)
        i_arr, t_arr = prof.current_a, prof.temp_c
        x = self._init_state(soc_init, float(i_arr[0]) if n else 0.0)
        P = self._init_cov()
        F = self._F(dt)
        Q = self._Q()
        R = c.sigma_v ** 2
        soc = np.empty(n); inn = np.zeros(n); nis = np.zeros(n)
        gain = np.zeros(n); sig = np.empty(n); vp = np.empty(n)
        bh = np.zeros(n); r_hist = np.empty(n)
        buf = []
        for k in range(n):
            if k > 0:
                x = self._predict(x, i_arr[k - 1], dt)
                P = F @ P @ F.T + Q
            H = self._H(x, float(t_arr[k]))
            y_hat = self._h(x, i_arr[k], float(t_arr[k]))
            S = float((H @ P @ H.T)[0, 0]) + R
            K = (P @ H.T).reshape(-1) / S
            e = float(v_meas[k]) - y_hat
            x = x + K * e
            x[0] = float(np.clip(x[0], *c.soc_clip))
            x[3] = float(np.clip(x[3], -c.b_clip, c.b_clip))
            IKH = np.eye(NDIM) - np.outer(K, H)
            P = IKH @ P @ IKH.T + R * np.outer(K, K)
            P = 0.5 * (P + P.T)

            buf.append(e)
            if len(buf) > self.window:
                buf.pop(0)
            if len(buf) >= max(10, self.window // 2) and (k % self.adapt_every == 0):
                c_v = float(np.mean(np.square(buf)))
                s_pred = float((H @ P @ H.T)[0, 0])
                r_new = float(np.clip(c_v - s_pred, self.r_min, self.r_max))
                R = 0.8 * R + 0.2 * r_new        # 平滑，避免单窗抖动
            soc[k] = x[0]; inn[k] = e; nis[k] = e * e / S
            gain[k] = K[0]; sig[k] = float(np.sqrt(max(P[0, 0], 0.0)))
            vp[k] = y_hat; bh[k] = x[3]; r_hist[k] = R
        out = self._pack(soc, inn, nis, gain, sig, vp, bh)
        out["R"] = r_hist
        return out


ESTIMATORS = {"cc": CoulombCounter, "ekf": EKF, "ukf": UKF, "aekf": AEKF}


def make(name: str, **kw) -> _BaseFilter:
    if name not in ESTIMATORS:
        raise KeyError(f"未知估计器 {name}，可选 {sorted(ESTIMATORS)}")
    return ESTIMATORS[name](**kw)
