"""white_r5（第九轮）合成数据单元测试 —— 全合成，不依赖数据集。"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def s30():
    return _load("s30", "scripts/30_white_round5.py")


@pytest.fixture()
def step_current():
    """阶跃+斜坡驱动：3 个脉冲，dt=1 s。"""
    dt = 1.0
    i = np.zeros(600)
    i[100:180] = -4.0
    i[250:330] = -6.0
    # 斜坡脉冲
    i[400:430] = np.linspace(0, -5, 30)
    i[430:500] = -5.0
    return i, dt


def test_edge_events_detect_onset_and_release(s30, step_current):
    i, dt = step_current
    onset, release = s30._edge_events(i, dt, thr=2.0)
    assert onset.size >= 3 and release.size >= 3
    assert onset[0] == 100
    assert release[0] == 180


def test_edge_distance_finite_and_kinds(s30, step_current):
    i, dt = step_current
    dist, kind = s30._edge_distance(i, dt, thr=2.0)
    assert np.all(np.isfinite(dist))
    assert set(np.unique(kind)).issubset({-1, 0, 1})
    # 边沿后第一个样本距离 = 0
    assert dist[100] == 0.0 and kind[100] == 1
    assert dist[180] == 0.0 and kind[180] == -1


def test_di_dt_matches_finite_difference(s30):
    i = np.array([0.0, 1.0, 2.0, 4.0, 3.0])
    d = s30._di_dt(i, 1.0)
    assert d[0] == pytest.approx(1.0)
    assert d[2] == pytest.approx(1.5)  # 中心差分 (4-1)/2
    assert d[-1] == pytest.approx(-1.0)


def test_rc_fast_reaches_steady_state(s30):
    i = np.ones(200) * 2.0
    u = s30._rc_fast(i, 1.0, tau=2.0, r=0.5)
    assert u[-1] == pytest.approx(1.0, rel=0.05)  # r*i = 0.5*2


def test_acf_white_noise_near_zero():
    s30 = _load("s30b", "scripts/30_white_round5.py")
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 8000)
    ac = s30._acf(x, (1, 5, 30), 1.0)
    assert abs(ac["acf_1s"]) < 0.05
    assert abs(ac["acf_5s"]) < 0.05


def test_acf_ar1_positive():
    s30 = _load("s30c", "scripts/30_white_round5.py")
    rng = np.random.default_rng(4)
    n = 8000
    x = np.zeros(n)
    e = rng.normal(0, 1, n)
    for k in range(1, n):
        x[k] = 0.8 * x[k - 1] + e[k]
    ac = s30._acf(x, (1, 5), 1.0)
    assert ac["acf_1s"] > 0.6


def test_fast_match_recovers_known_tau(s30):
    """已知 τ=2 s 快模态：残差化匹配应给出同量级 τ*。"""
    rng = np.random.default_rng(7)
    n = 6000
    i = np.zeros(n)
    pos = 200
    while pos < n - 300:
        i[pos:pos + int(rng.integers(20, 80))] = rng.choice([-5.0, 3.0])
        pos += int(rng.integers(40, 120))
    u = s30._rc_fast(i, 1.0, 2.0)
    # 对 [1,i,|i|] 残差化后取作 white（模拟 odd/even 已被吃掉）
    w = s30._resid_basis(u, [i, np.abs(i)])
    fm = s30._fast_match(w, i, 1.0, tau_grid=(0.5, 1.0, 2.0, 5.0, 10.0))
    assert fm["fast_tau_star_s"] in (1.0, 2.0, 5.0)
    assert abs(fm["fast_rho_max"]) > 0.3


def test_synthetic_controls_separate_noise_from_structure(s30):
    """对照检出力：noise 平坦、fast_rc 有记忆、hyst_fast 有 didt 比。"""
    rng = np.random.default_rng(20261101)
    ctrl = s30.synthetic_control(rng)
    assert ctrl["noise"]["didt_hi_lo_ratio"] < 1.3
    assert abs(ctrl["noise"]["acf"]["acf_1s"]) < 0.1
    assert ctrl["fast_rc"]["acf"]["acf_1s"] > 0.5
    assert ctrl["hyst_fast"]["didt_hi_lo_ratio"] > 1.3
    # 慢滞回应被 slow 吃掉（white 保持干净）
    assert ctrl["hyst_slow"]["white_rms_mv"] < ctrl["fast_rc"]["white_rms_mv"] * 0.5


def test_hyst_proxy_sign_matches_soc_direction(s30):
    dt = 1.0
    n = 1200
    soc = np.concatenate([np.linspace(1.0, 0.6, n // 2),
                          np.linspace(0.6, 0.9, n - n // 2)])
    i = -np.gradient(soc, dt) * 2.5 * 3600.0  # 反推电流（放电为负）
    h = s30._hyst_proxy(i, soc, dt, win_s=200.0)
    # 前半段下行（放电）→ h<0；后半段上行（充电）→ h>0
    assert h[n // 4] < 0
    assert h[3 * n // 4] > 0


def test_proj_r2_explains_linear():
    s30 = _load("s30d", "scripts/30_white_round5.py")
    rng = np.random.default_rng(9)
    x = rng.normal(0, 1, 500)
    y = 2.0 * x + rng.normal(0, 0.05, 500)
    r2 = s30._proj_r2(y, [x])
    assert r2 > 0.95
    r2_null = s30._proj_r2(rng.normal(0, 1, 500), [x])
    assert r2_null < 0.1
