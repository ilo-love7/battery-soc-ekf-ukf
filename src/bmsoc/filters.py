"""SOC 估计器：库仑计数基线、EKF、UKF、AEKF。

统一接口
--------
    est = EKF(ocv=ocv2d, params=rc, capacity_ah=2.6, cfg=FilterConfig(soc0=0.7))
    out = est.run(profile, current_bias_a=0.02)

外部 SOH 锚点（A2 接口）
------------------------
容量不能在线联合估计（``EKFQ`` 在真实数据上已证伪，工作记录 §23），只能来自
外部锚点。``SoHSchedule`` 把 A2（实车 SOH）按周期输出的 ``Q̂(t)``/``R̂0(t)`` 传进来，
两次刷新之间零阶保持::

    anchor = SoHSchedule(t_upd=[0, 600, 1200], q_ah=[2.45, 2.47, 2.44])
    out = est.run(profile, anchor=anchor)

状态 ``x = [soc, u1, u2(, u3), b]``：SOC 与各条 RC 极化支路，``b`` 是**电压偏置
状态**（模型 OCV 与实测之间的系统差）。``params.r3 > 0`` 时自动多一条慢极化
状态 ``u3``（三阶 RC），状态维数随支路数变化 —— 残差摸底（脚本 19）显示 LOO
残差的 58–85% 落在 >60 s 的慢频带，第三条慢支路就是为它准备的。

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


class SoHSchedule:
    """外部 SOH 锚点：A2 按周期给出的 ``Q̂(t)``、``R̂0(t)``（分段常值，零阶保持）。

    为什么容量只能来自外部
    ----------------------
    ``EKFQ`` 把容量当第五状态在线估计，在合成数据（模型精确）上有效，在真实
    数据上全线劣化 —— 38–53 mV 的模型残差与容量误差在电压上同构，``q`` 把残差
    当容量吸收（工作记录 §23，超参扫描证明是结构性失败）。于是容量改由外部
    锚点（A2 的 SOH 输出）供给，本类就是那个接口。

    参数
    ----
    t_upd : (m,) 各次刷新的时刻（s，与 ``profile.t_s`` 同轴，严格递增）
    q_ah : (m,) 各次刷新的容量估计 (Ah)，须 > 0
    r0_ohm : (m,) 各次刷新的欧姆内阻估计 (Ω)，可选。缺项（None）处不覆盖，
        估计器回退到模型参数 ``params.r0``。

    早于首个刷新时刻的 ``t`` 取第一个值（上电即持有第一次 SOH 输出）。
    """

    def __init__(self, t_upd, q_ah, r0_ohm=None):
        self.t_upd = np.asarray(t_upd, dtype=float).reshape(-1)
        self.q_ah = np.asarray(q_ah, dtype=float).reshape(-1)
        self.r0_ohm = None if r0_ohm is None else np.asarray(r0_ohm, dtype=float).reshape(-1)
        if self.t_upd.size == 0:
            raise ValueError("t_upd 不能为空")
        if self.t_upd.size != self.q_ah.size:
            raise ValueError("t_upd 与 q_ah 长度不一致")
        if self.r0_ohm is not None and self.r0_ohm.size != self.t_upd.size:
            raise ValueError("r0_ohm 与 t_upd 长度不一致")
        if self.t_upd.size > 1 and np.any(np.diff(self.t_upd) <= 0):
            raise ValueError("t_upd 必须严格递增")
        if np.any(self.q_ah <= 0):
            raise ValueError("q_ah 必须为正")

    def _idx(self, t) -> np.ndarray:
        """零阶保持查找：最后一个 ``t_upd <= t`` 的下标，早于首点则取 0。"""
        t_arr = np.asarray(t, dtype=float)
        j = np.searchsorted(self.t_upd, t_arr, side="right") - 1
        return np.clip(j, 0, self.t_upd.size - 1)

    def q_at(self, t, default: Optional[float] = None) -> np.ndarray:
        """向量版容量查找。 ``default`` 仅为接口对称保留，恒用锚点值。"""
        return self.q_ah[self._idx(t)]

    def r0_at(self, t, default: float) -> np.ndarray:
        """向量版内阻查找；锚点未给 r0 时用 ``default``（模型参数）。"""
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        if self.r0_ohm is None:
            return np.full(t_arr.shape, float(default))
        out = self.r0_ohm[self._idx(t_arr)]
        return out

    def q(self, t: float) -> float:
        return float(self.q_ah[self._idx(t)])

    def r0(self, t: float, default: float) -> float:
        if self.r0_ohm is None:
            return float(default)
        return float(self.r0_ohm[self._idx(t)])


class _BaseFilter:
    name = "base"

    def __init__(self, ocv, params: RCParams, capacity_ah: float,
                 cfg: Optional[FilterConfig] = None):
        self.ocv = ocv
        self.p = params
        self.q_cap = float(capacity_ah)
        self.cfg = cfg or FilterConfig()
        # 状态布局：[soc, u1..unr, b]，EKFQ 再在末尾追加 q
        self.nr = int(getattr(params, "n_rc", 2))
        self.ndim = self.nr + 2

    @property
    def _ib(self) -> int:
        """偏置状态下标。"""
        return self.nr + 1

    # ---- 公共 ----
    def _ab(self, dt: float):
        if not hasattr(self, "_ab_cache") or self._ab_cache[0] != dt:
            self._ab_cache = (dt, step_matrices(self.p, dt))
        return self._ab_cache[1]

    def _pairs(self, dt: float):
        """``[(a1, b1), (a2, b2)(, (a3, b3))]``。"""
        ab = self._ab(dt)
        return [(ab[2 * j], ab[2 * j + 1]) for j in range(self.nr)]

    def _predict(self, x: np.ndarray, i: float, dt: float, clip: bool = True,
                 q: Optional[float] = None) -> np.ndarray:
        c = self.cfg
        q_cap = self.q_cap if q is None else max(float(q), 1e-6)
        soc = x[0] + c.eta * i * dt / (3600.0 * q_cap)
        u = [a * x[1 + j] + b * i for j, (a, b) in enumerate(self._pairs(dt))]
        bias = x[self._ib]
        if not clip:      # sigma 点传播保持线性，裁剪会破坏对称性
            return np.array([soc, *u, bias])
        return np.array([np.clip(soc, *c.soc_clip),
                         *[float(np.clip(uj, -c.u_clip, c.u_clip)) for uj in u],
                         float(np.clip(bias, -c.b_clip, c.b_clip))])

    def _h(self, x: np.ndarray, i: float, temp_c: float = 25.0,
           r0: Optional[float] = None) -> float:
        r0_use = self.p.r0 if r0 is None else float(r0)
        u_sum = float(sum(x[1 + j] for j in range(self.nr)))
        return (_ocv_scalar(self.ocv, float(x[0]), temp_c) + u_sum
                + r0_use * i + getattr(self.p, "r0_cur", 0.0) * i * abs(i)
                + getattr(self.p, "dv", 0.0) + x[self._ib])

    def _H(self, x: np.ndarray, temp_c: float) -> np.ndarray:
        return np.array([[_docv_scalar(self.ocv, float(x[0]), temp_c)]
                         + [1.0] * self.nr + [1.0]])

    def _init_state(self, soc_init: Optional[float], i0: float = 0.0) -> np.ndarray:
        """初始极化取当前电流下的稳态值 R_j·i0，偏置置 0。"""
        s0 = self.cfg.soc0 if soc_init is None else float(soc_init)
        u0 = [self.p.r1 * i0, self.p.r2 * i0]
        if self.nr > 2:
            u0.append(self.p.r3 * i0)
        return np.array([s0, *u0, 0.0])

    def _init_cov(self) -> np.ndarray:
        c = self.cfg
        return np.diag([c.sigma_soc0 ** 2] + [c.sigma_u0 ** 2] * self.nr + [c.sigma_b0 ** 2])

    def _Q(self) -> np.ndarray:
        c = self.cfg
        return np.diag([c.q_soc] + [c.q_u] * self.nr + [c.q_b])

    def run(self, profile: DrivingProfile, voltage_noise: float = 0.0,
            seed: int = 0, soc_init: Optional[float] = None,
            current_bias_a: float = 0.0,
            anchor: Optional[SoHSchedule] = None) -> Dict[str, np.ndarray]:
        """跑一段工况。

        ``current_bias_a``：电流传感器偏置（A）。只影响估计器看到的电流，
        真值 SOC 仍由原始电流积分，故库仑计数会随偏置线性漂移。

        ``anchor``：外部 SOH 锚点（A2 接口）。给出后，逐步的容量/内阻取
        ``anchor`` 的零阶保持值，覆盖构造参数 ``capacity_ah``/``params.r0``。
        与 ``EKFQ`` 不兼容（容量由状态 ``x[4]`` 自己演化，见其文档）。
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
        q_arr, r0_arr = self._anchor_arrays(prof, anchor)
        return self._run(prof, dt, v_meas, soc_init, q_arr, r0_arr)

    def _anchor_arrays(self, prof, anchor: Optional[SoHSchedule]):
        n = len(prof)
        if anchor is None:
            return np.full(n, self.q_cap), np.full(n, self.p.r0)
        return (np.asarray(anchor.q_at(prof.t_s), dtype=float),
                np.asarray(anchor.r0_at(prof.t_s, default=self.p.r0), dtype=float))

    def _run(self, prof, dt, v_meas, soc_init, q_arr=None, r0_arr=None):  # pragma: no cover - 抽象
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

    def _run(self, prof, dt, v_meas, soc_init, q_arr=None, r0_arr=None):
        n = len(prof)
        soc = np.empty(n)
        soc[0] = self.cfg.soc0 if soc_init is None else float(soc_init)
        q_arr = np.full(n, self.q_cap) if q_arr is None else np.asarray(q_arr, float)
        for k in range(n - 1):
            soc[k + 1] = soc[k] + self.cfg.eta * prof.current_a[k] * dt / (3600.0 * q_arr[k])
        return self._pack(np.clip(soc, *self.cfg.soc_clip), np.zeros(n), np.zeros(n),
                          np.zeros(n), np.full(n, np.nan), np.full(n, np.nan))


class EKF(_BaseFilter):
    name = "ekf"

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(self.ndim)
        for j, (a, _) in enumerate(self._pairs(dt)):
            F[1 + j, 1 + j] = a
        return F

    def _run(self, prof, dt, v_meas, soc_init, q_arr=None, r0_arr=None):
        c = self.cfg
        n = len(prof)
        i_arr, t_arr = prof.current_a, prof.temp_c
        q_arr = np.full(n, self.q_cap) if q_arr is None else np.asarray(q_arr, float)
        r0_arr = np.full(n, self.p.r0) if r0_arr is None else np.asarray(r0_arr, float)
        x = self._init_state(soc_init, float(i_arr[0]) if n else 0.0)
        P = self._init_cov()
        F = self._F(dt)
        Q = self._Q()
        R = c.sigma_v ** 2
        ib = self._ib
        soc = np.empty(n); inn = np.zeros(n); nis = np.zeros(n)
        gain = np.zeros(n); sig = np.empty(n); vp = np.empty(n); bh = np.zeros(n)
        for k in range(n):
            if k > 0:
                x = self._predict(x, i_arr[k - 1], dt, q=q_arr[k - 1])
                P = F @ P @ F.T + Q
            H = self._H(x, float(t_arr[k]))
            y_hat = self._h(x, i_arr[k], float(t_arr[k]), r0=r0_arr[k])
            S = float((H @ P @ H.T)[0, 0]) + R
            K = (P @ H.T).reshape(-1) / S
            e = float(v_meas[k]) - y_hat
            x = x + K * e
            x[0] = float(np.clip(x[0], *c.soc_clip))
            x[ib] = float(np.clip(x[ib], -c.b_clip, c.b_clip))
            IKH = np.eye(self.ndim) - np.outer(K, H)
            P = IKH @ P @ IKH.T + R * np.outer(K, K)      # Joseph 形式
            P = 0.5 * (P + P.T)
            soc[k] = x[0]; inn[k] = e; nis[k] = e * e / S
            gain[k] = K[0]; sig[k] = float(np.sqrt(max(P[0, 0], 0.0)))
            vp[k] = y_hat; bh[k] = x[ib]
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
        nd = self.ndim
        try:
            L = np.linalg.cholesky(scl * cov)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(scl * (cov + np.eye(nd) * 1e-12))
        X = np.empty((2 * nd + 1, nd))
        X[0] = mu
        for j in range(nd):
            X[1 + j] = mu + L[:, j]
            X[1 + nd + j] = mu - L[:, j]
        return X

    def _run(self, prof, dt, v_meas, soc_init, q_arr=None, r0_arr=None):
        c = self.cfg
        n = len(prof)
        i_arr, t_arr = prof.current_a, prof.temp_c
        q_arr = np.full(n, self.q_cap) if q_arr is None else np.asarray(q_arr, float)
        r0_arr = np.full(n, self.p.r0) if r0_arr is None else np.asarray(r0_arr, float)
        x = self._init_state(soc_init, float(i_arr[0]) if n else 0.0)
        P = self._init_cov()
        Q = self._Q()
        R = c.sigma_v ** 2
        lam, wm, wc = self._weights(self.ndim)
        scl = self.ndim + lam
        ib = self._ib
        soc = np.empty(n); inn = np.zeros(n); nis = np.zeros(n)
        gain = np.zeros(n); sig = np.empty(n); vp = np.empty(n); bh = np.zeros(n)
        for k in range(n):
            if k > 0:
                X = self._sigma(x, P, scl)
                Xp = np.array([self._predict(pt, i_arr[k - 1], dt, clip=False, q=q_arr[k - 1])
                               for pt in X])
                x = wm @ Xp
                x[0] = float(np.clip(x[0], *c.soc_clip))
                dX = Xp - x
                P = (wc[:, None, None] * np.einsum("ij,ik->ijk", dX, dX)).sum(0) + Q
                P = self._psd(P)
            X = self._sigma(x, P, scl)
            Y = np.array([self._h(pt, i_arr[k], float(t_arr[k]), r0=r0_arr[k]) for pt in X])
            y_hat = float(wm @ Y)
            dY = Y - y_hat
            S = float((wc * dY * dY).sum()) + R
            dX = X - x
            Pxy = (wc[:, None] * dX * dY[:, None]).sum(0)
            K = Pxy / S
            e = float(v_meas[k]) - y_hat
            x = x + K * e
            x[0] = float(np.clip(x[0], *c.soc_clip))
            x[ib] = float(np.clip(x[ib], -c.b_clip, c.b_clip))
            P = self._psd(P - np.outer(K, Pxy))
            soc[k] = x[0]; inn[k] = e; nis[k] = e * e / S
            gain[k] = K[0]; sig[k] = float(np.sqrt(max(P[0, 0], 0.0)))
            vp[k] = y_hat; bh[k] = x[ib]
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

    def _run(self, prof, dt, v_meas, soc_init, q_arr=None, r0_arr=None):
        c = self.cfg
        n = len(prof)
        i_arr, t_arr = prof.current_a, prof.temp_c
        q_arr = np.full(n, self.q_cap) if q_arr is None else np.asarray(q_arr, float)
        r0_arr = np.full(n, self.p.r0) if r0_arr is None else np.asarray(r0_arr, float)
        x = self._init_state(soc_init, float(i_arr[0]) if n else 0.0)
        P = self._init_cov()
        F = self._F(dt)
        Q = self._Q()
        R = c.sigma_v ** 2
        ib = self._ib
        soc = np.empty(n); inn = np.zeros(n); nis = np.zeros(n)
        gain = np.zeros(n); sig = np.empty(n); vp = np.empty(n)
        bh = np.zeros(n); r_hist = np.empty(n)
        buf = []
        for k in range(n):
            if k > 0:
                x = self._predict(x, i_arr[k - 1], dt, q=q_arr[k - 1])
                P = F @ P @ F.T + Q
            H = self._H(x, float(t_arr[k]))
            y_hat = self._h(x, i_arr[k], float(t_arr[k]), r0=r0_arr[k])
            S = float((H @ P @ H.T)[0, 0]) + R
            K = (P @ H.T).reshape(-1) / S
            e = float(v_meas[k]) - y_hat
            x = x + K * e
            x[0] = float(np.clip(x[0], *c.soc_clip))
            x[ib] = float(np.clip(x[ib], -c.b_clip, c.b_clip))
            IKH = np.eye(self.ndim) - np.outer(K, H)
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
            vp[k] = y_hat; bh[k] = x[ib]; r_hist[k] = R
        out = self._pack(soc, inn, nis, gain, sig, vp, bh)
        out["R"] = r_hist
        return out


class EKFQ(EKF):
    """EKF + **容量联合估计**：状态 ``[soc, u1, u2, b, q]``，``q`` 是容量 (Ah)。

    为什么需要
    ----------
    标定容差实验（`scripts/12`）测出：容量是误差预算里**唯一没有状态能吸收**的项。
    OCV 常值误差被 ``b`` 吃掉（±40 mV ⇒ 0.04 pp），容量 5% ⇒ 2–3 pp、10% ⇒ 5 pp，
    且随放电深度线性增长。本类把容量也建成状态，看电压反馈能不能把它拉回来。

    可辨识性（必须先说清）
    --------------------
    ``h(x)`` 里容量**不出现**，它只通过 ``soc' = soc + η·i·dt/(3600·q)`` 影响
    电压。所以 ``q`` 只在「有电流积分」且「SOC 被电压锚定」时可辨识：

    - **静置/恒流平台**：`i` 恒定 ⇒ `soc` 与 `q` 的误差以同一比例进入电压，
      完全共线，估计不动；
    - **动态工况（电流变号、幅值变化大）**：`q` 的误差使 SOC 预测以特定方式漂移，
      与 `b`（常值）可区分，此时能收。

    一句话：**容量可辨识性来自电流激励的丰富度，不是来自数据量。** 这与
    「偏置与 SOC 在平台区共线」（§8.1.3）是同一个问题的两个侧面。

    ``q`` 的过程噪声给得很小（慢随机游走），初值标准差默认 5%（真实标定误差量级）。
    """

    name = "ekfq"

    def __init__(self, *a, sigma_q0: float = 0.05, q_q: float = 1e-10, **kw):
        super().__init__(*a, **kw)
        self.sigma_q0 = float(sigma_q0)          # 相对容量（分数）
        self.q_q = float(q_q)                    # 容量过程噪声（Ah²/步）
        self.ndim = self.nr + 3                  # 追加 q 状态

    @property
    def _iq(self) -> int:
        return self.nr + 2

    def _F(self, dt: float, i: float, q: float) -> np.ndarray:
        """状态转移雅可比。``∂soc/∂q = −η·i·dt/(3600 q²)`` 依赖当前电流与容量，
        所以必须每步重算（EKF 基类里 F 是常数，这里不是）。"""
        F = np.eye(self.ndim)
        for j, (a, _) in enumerate(self._pairs(dt)):
            F[1 + j, 1 + j] = a
        q = max(float(q), 1e-6)
        F[0, self._iq] = -self.cfg.eta * float(i) * dt / (3600.0 * q * q)
        return F

    def _predict(self, x: np.ndarray, i: float, dt: float, clip: bool = True,
                 q: Optional[float] = None) -> np.ndarray:
        """``q`` 参数忽略：EKFQ 的容量来自状态 ``x[_iq]``，不接受外部锚点覆盖。"""
        c = self.cfg
        q = max(float(x[self._iq]), 1e-6)
        soc = x[0] + c.eta * i * dt / (3600.0 * q)
        u = [a * x[1 + j] + b * i for j, (a, b) in enumerate(self._pairs(dt))]
        if not clip:
            return np.array([soc, *u, x[self._ib], x[self._iq]])
        return np.array([np.clip(soc, *c.soc_clip),
                         *[float(np.clip(uj, -c.u_clip, c.u_clip)) for uj in u],
                         float(np.clip(x[self._ib], -c.b_clip, c.b_clip)), q])

    def _H(self, x: np.ndarray, temp_c: float) -> np.ndarray:
        return np.array([[_docv_scalar(self.ocv, float(x[0]), temp_c)]
                         + [1.0] * self.nr + [1.0, 0.0]])

    def _init_state(self, soc_init: Optional[float], i0: float = 0.0) -> np.ndarray:
        s0 = self.cfg.soc0 if soc_init is None else float(soc_init)
        u0 = [self.p.r1 * i0, self.p.r2 * i0]
        if self.nr > 2:
            u0.append(self.p.r3 * i0)
        return np.array([s0, *u0, 0.0, self.q_cap])

    def _init_cov(self) -> np.ndarray:
        c = self.cfg
        sq = (self.sigma_q0 * self.q_cap) ** 2
        return np.diag([c.sigma_soc0 ** 2] + [c.sigma_u0 ** 2] * self.nr
                       + [c.sigma_b0 ** 2, sq])

    def _Q(self) -> np.ndarray:
        c = self.cfg
        return np.diag([c.q_soc] + [c.q_u] * self.nr + [c.q_b, self.q_q])

    def _run(self, prof, dt, v_meas, soc_init, q_arr=None, r0_arr=None):
        c = self.cfg
        n = len(prof)
        i_arr, t_arr = prof.current_a, prof.temp_c
        # q_arr 忽略：容量由状态 x[_iq] 演化，外部锚点的 Q̂ 与 EKFQ 互斥（r0 仍生效）
        r0_arr = np.full(n, self.p.r0) if r0_arr is None else np.asarray(r0_arr, float)
        x = self._init_state(soc_init, float(i_arr[0]) if n else 0.0)
        P = self._init_cov()
        Q = self._Q()
        R = c.sigma_v ** 2
        nd = self.ndim
        ib, iq = self._ib, self._iq
        soc = np.empty(n); inn = np.zeros(n); nis = np.zeros(n)
        gain = np.zeros(n); sig = np.empty(n); vp = np.empty(n); bh = np.zeros(n)
        qh = np.empty(n)
        for k in range(n):
            if k > 0:
                F = self._F(dt, float(i_arr[k - 1]), float(x[iq]))
                x = self._predict(x, i_arr[k - 1], dt)
                P = F @ P @ F.T + Q
            # 容量下限保护：q 只允许在 [0.5, 1.5]×初值内游走，防止奇异
            x[iq] = float(np.clip(x[iq], 0.5 * self.q_cap, 1.5 * self.q_cap))
            H = self._H(x, float(t_arr[k]))
            y_hat = self._h(x, i_arr[k], float(t_arr[k]), r0=r0_arr[k])
            S = float((H @ P @ H.T)[0, 0]) + R
            K = (P @ H.T).reshape(-1) / S
            e = float(v_meas[k]) - y_hat
            x = x + K * e
            x[0] = float(np.clip(x[0], *c.soc_clip))
            x[ib] = float(np.clip(x[ib], -c.b_clip, c.b_clip))
            x[iq] = float(np.clip(x[iq], 0.5 * self.q_cap, 1.5 * self.q_cap))
            IKH = np.eye(nd) - np.outer(K, H)
            P = IKH @ P @ IKH.T + R * np.outer(K, K)
            P = 0.5 * (P + P.T)
            soc[k] = x[0]; inn[k] = e; nis[k] = e * e / S
            gain[k] = K[0]; sig[k] = float(np.sqrt(max(P[0, 0], 0.0)))
            vp[k] = y_hat; bh[k] = x[ib]; qh[k] = x[iq]
        out = self._pack(soc, inn, nis, gain, sig, vp, bh)
        out["q_hat"] = qh
        return out


ESTIMATORS = {"cc": CoulombCounter, "ekf": EKF, "ukf": UKF, "aekf": AEKF}


def make(name: str, **kw) -> _BaseFilter:
    if name not in ESTIMATORS:
        raise KeyError(f"未知估计器 {name}，可选 {sorted(ESTIMATORS)}")
    return ESTIMATORS[name](**kw)
