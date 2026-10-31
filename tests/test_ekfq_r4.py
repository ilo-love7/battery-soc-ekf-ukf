"""第八轮（脚本 28）纯函数与合成台架测试 —— 不依赖任何数据集。

覆盖：结构分解的正交性/恒等式/斜率符号、分解-注入代数、锚点插值、
零残差合成台架的 EKFQ 收敛（可辨识性正对照）。
"""
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(ROOT, "scripts", "28_ekfq_round4.py")


def _load_s28():
    spec = importlib.util.spec_from_file_location("s28_ekfq", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def s28():
    return _load_s28()


@pytest.fixture(scope="module")
def synthetic_r():
    """已知结构的残差：mean 10 mV + slow 300 s 正弦 + odd + even + 白噪。"""
    rng = np.random.default_rng(7)
    n = 3000
    dt = 1.0
    t = np.arange(n) * dt
    i = 2.0 * np.sin(2 * np.pi * t / 300.0) + rng.normal(0, 0.3, n)
    mean = 0.010
    slow = 0.008 * np.sin(2 * np.pi * t / 2400.0)
    gi = i - i.mean()
    odd = 0.015 * gi                      # 15 mV/A 量级
    ga = np.abs(i) - np.abs(i).mean()
    even = 0.006 * ga
    white = rng.normal(0.0, 0.004, n)
    r = mean + slow + odd + even + white
    return dict(r=r, i=i, dt=dt, mean=mean, slow=slow, odd=odd, even=even,
                white=white)


def test_structure_identity_reconstructs_r(s28, synthetic_r):
    comps = s28._structure_components(synthetic_r["r"], synthetic_r["i"],
                                      synthetic_r["dt"])
    recon = comps["mean"] + comps["slow"] + comps["odd"] + comps["even"] \
        + comps["white"]
    assert np.allclose(recon, synthetic_r["r"], atol=1e-12)


def test_structure_recovers_odd_slope(s28, synthetic_r):
    comps = s28._structure_components(synthetic_r["r"], synthetic_r["i"],
                                      synthetic_r["dt"])
    # odd 斜率应接近 15 mV/A（slow/white 的泄漏有限）
    assert abs(comps["odd_slope_mv_a"] - 15.0) < 4.0


def test_structure_recovers_even_slope(s28, synthetic_r):
    comps = s28._structure_components(synthetic_r["r"], synthetic_r["i"],
                                      synthetic_r["dt"])
    assert abs(comps["even_slope_mv_a"] - 6.0) < 2.5


def test_structure_white_orthogonal_to_i(s28, synthetic_r):
    comps = s28._structure_components(synthetic_r["r"], synthetic_r["i"],
                                      synthetic_r["dt"])
    gi = synthetic_r["i"] - synthetic_r["i"].mean()
    rho = float(np.dot(gi, comps["white"])) / max(
        np.linalg.norm(gi) * np.linalg.norm(comps["white"]), 1e-15)
    assert abs(rho) < 0.05


def test_structure_rms_matches_components(s28, synthetic_r):
    st = s28._structure(synthetic_r["r"], synthetic_r["i"], synthetic_r["dt"])
    comps = s28._structure_components(synthetic_r["r"], synthetic_r["i"],
                                      synthetic_r["dt"])
    for k in ("slow", "odd", "even", "white"):
        assert st[f"{k}_mv"] == pytest.approx(
            float(np.sqrt(np.mean(comps[k] ** 2)) * 1e3), rel=1e-9)


def test_structure_tiny_input_returns_nan_slopes(s28):
    out = s28._structure(np.zeros(5), np.zeros(5), 1.0)
    assert np.isnan(out["odd_slope_mv_a"])


def test_r0_anchor_interpolates(s28):
    anc = s28._R0Anchor(np.array([0.0, 10.0]), np.array([0.05, 0.10]), 2.5)
    got = anc.r0_at(np.array([0.0, 5.0, 10.0]))
    assert np.allclose(got, [0.05, 0.075, 0.10])
    assert np.allclose(anc.q_at(np.array([1.0, 2.0])), 2.5)


def test_rc_from_vec_unpacks(s28):
    from bmsoc.ecm import _unpack_vec
    vec = np.array([0.03, 0.031, 0.032, 0.012, 12.0, 0.045, 450.0, -0.02])
    p, r0k = s28._rc_from_vec(vec)
    assert p.r0 == pytest.approx(0.031)
    assert p.dv == pytest.approx(-0.02)
    assert p.r3 == 0.0
    assert r0k.shape == (3,)


def test_ekfq_converges_on_clean_synthetic_plant(s28):
    """零残差正对照：EKFQ 在自洽植物上把 q 收回 ±2%（可辨识性前提）。"""
    from bmsoc.ecm import RCParams, open_loop_from_truth, simulate
    from bmsoc.filters import EKFQ, FilterConfig
    from bmsoc.profiles import DrivingProfile

    rng = np.random.default_rng(11)
    n = 6000
    dt = 1.0
    t = np.arange(n) * dt
    # 直流放电偏置保证 DOD ~0.8（q 可辨识性来自激励，见 §23.2）
    i = -1.2 + 1.5 * np.sin(2 * np.pi * t / 240.0) + 0.8 * np.sin(2 * np.pi * t / 37.0)
    q_true = 2.5
    p_true = RCParams(r0=0.030, r1=0.012, tau1=12.0, r2=0.045, tau2=450.0,
                      dv=0.0)
    socs = np.linspace(0.05, 1.0, 201)
    ocv_v = 3.2 + 0.8 * socs + 0.1 * socs ** 2

    class _Curve:
        def batch(self, soc, temp):
            return np.interp(np.asarray(soc, float), socs, ocv_v)

        def __call__(self, soc, temp=None):
            return np.interp(np.asarray(soc, float), socs, ocv_v)

        def docv(self, soc, temp=None):
            s = np.asarray(soc, float)
            return 0.8 + 0.2 * s

    sim = simulate(p_true, _Curve(), i, dt, soc0=0.95, capacity_ah=q_true)
    soc = np.asarray(sim["soc"], float)
    assert soc.min() > 0.05
    v = open_loop_from_truth(
        np.array([0.03, 0.03, 0.03, 0.012, 12.0, 0.045, 450.0, 0.0]), soc, i,
        _Curve(), dt, 25.0)
    pf = DrivingProfile(t_s=t, current_a=i, voltage_v=v,
                        temp_c=np.full(n, 25.0), soc_ref=soc, capacity_ah=q_true)
    cfg = FilterConfig(soc0=0.95, sigma_soc0=0.05)
    out = EKFQ(ocv=_Curve(), params=p_true, capacity_ah=q_true * 1.05,
               cfg=cfg).run(pf)
    qh = float(np.median(out["q_hat"][-n // 10:]))
    assert abs(qh / q_true - 1.0) < 0.02
