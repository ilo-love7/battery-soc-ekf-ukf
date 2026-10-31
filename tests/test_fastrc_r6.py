"""fastrc_r6（第十轮）合成数据单元测试 —— 全合成，不依赖数据集。"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def s32():
    return _load("s32", "scripts/32_fastrc_round6.py")


@pytest.fixture()
def drive_i():
    """斜坡+平台驱动（与探针合成对照同构），dt=1 s。"""
    i = np.zeros(2400)
    i[200:400] = -4.0
    i[500:560] = np.linspace(0, -6, 60)
    i[560:800] = -6.0
    i[1000:1100] = np.linspace(0, -5, 100)
    i[1100:1400] = -5.0
    i[1600:1700] = np.linspace(0, -3, 100)
    i[1700:1900] = -3.0
    return i


# ---------------------------------------------------------------- 基础构件

def test_rc_branch_steady_state(s32):
    i = np.ones(300) * 2.0
    u = s32._rc_branch(i, 1.0, tau=3.0, r=0.5)
    assert u[-1] == pytest.approx(1.0, rel=0.05)     # r*i


def test_resid_basis_is_orthogonal(s32):
    rng = np.random.default_rng(2)
    n = 500
    i = rng.normal(-1, 2, n)
    y = 3.0 + 0.7 * i + 0.2 * np.abs(i) + rng.normal(0, 0.1, n)
    r = s32._resid_basis(y, [i, np.abs(i)])
    assert abs(np.mean(r)) < 1e-8
    assert abs(np.dot(r, i)) < 1e-6 * n
    assert abs(np.dot(r, np.abs(i))) < 1e-6 * n


# ---------------------------------------------------------------- 两段式快支路拟合

def test_fit_fast_recovers_known_shape(s32, drive_i):
    """white = β·rc_res(τ*) + 噪声：应恢复 τ* 且削减大部分方差。"""
    i = drive_i
    tau_true = 5.0
    u = s32._rc_branch(i, 1.0, tau_true)
    u_res = s32._resid_basis(u, [i, np.abs(i)])
    rng = np.random.default_rng(3)
    y = -0.045 * u_res + rng.normal(0, 2e-4, i.size)
    fit = s32._fit_fast_two_stage(y, i, 1.0)
    assert fit["tau_best"] == pytest.approx(tau_true)
    assert fit["cut"] > 0.8
    assert fit["r2"] > 0.8
    # 分量与真值同向
    assert np.corrcoef(fit["component"], -0.045 * u_res)[0, 1] > 0.99


def test_fit_fast_fixed_tau_honored(s32, drive_i):
    i = drive_i
    u = s32._rc_branch(i, 1.0, 10.0)
    u_res = s32._resid_basis(u, [i, np.abs(i)])
    y = 0.03 * u_res
    fit = s32._fit_fast_two_stage(y, i, 1.0, tau=10.0)
    assert fit["tau_best"] == 10.0
    assert fit["cut"] > 0.9


def test_fit_fast_noise_gives_near_zero_cut(s32, drive_i):
    rng = np.random.default_rng(4)
    y = rng.normal(0, 1e-3, drive_i.size)
    fit = s32._fit_fast_two_stage(y, drive_i, 1.0)
    assert abs(fit["cut"]) < 0.15
    assert abs(fit["beta"]) < 20.0


# ---------------------------------------------------------------- 滞后与延迟签名

def test_lag_profile_finds_shifted_peak(s32):
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 2000)
    y = np.roll(x, 3)          # y[t] = x[t−3]
    lag = s32._lag_profile(y, x, lags_s=tuple(range(-5, 6)))
    assert lag["peak_lag"] == 3
    assert lag["rho"][3] > 0.9


def test_lag_delay_like_thresholds(s32):
    rho = {k: 0.0 for k in range(-5, 6)}
    rho[0] = 0.10
    lag = {"rho": rho, "rho0": 0.10}
    assert not s32._lag_delay_like(lag, 0.30, 1.5)
    rho[2] = 0.45
    assert s32._lag_delay_like(lag, 0.30, 1.5)      # |ρ(2)|=0.45 ≥ 1.5×0.10
    rho[2] = 0.12
    assert not s32._lag_delay_like(lag, 0.30, 1.5)  # 比值不过


# ---------------------------------------------------------------- 对齐审计（H31 仪器）

def test_align_sweep_recovers_injected_delay(s32, drive_i, ):
    i = drive_i
    i_shift = np.roll(i, 2)
    i_shift[:2] = 0.0
    v = 0.05 * i_shift + 0.02 * s32._rc_branch(i_shift, 1.0, 5.0)
    rng = np.random.default_rng(6)
    v = v + rng.normal(0, 3e-4, v.size)
    al = s32._align_sweep(v, i, 1.0)
    assert al["signal"]
    assert abs(al["delta_hat"] - 2.0) <= 0.5


def test_align_sweep_aligned_near_zero(s32, drive_i):
    i = drive_i
    v = 0.05 * i + 0.02 * s32._rc_branch(i, 1.0, 5.0)
    rng = np.random.default_rng(7)
    v = v + rng.normal(0, 3e-4, v.size)
    al = s32._align_sweep(v, i, 1.0)
    assert al["signal"]
    assert abs(al["delta_hat"]) <= 0.5


def test_align_sweep_noise_no_signal(s32, drive_i):
    rng = np.random.default_rng(8)
    v = rng.normal(0, 1e-3, drive_i.size)
    al = s32._align_sweep(v, drive_i, 1.0)
    assert not al["signal"]
    assert not np.isfinite(al["delta_hat"])


# ---------------------------------------------------------------- ecm.tau3_bounds（第十轮接口）

def test_fit_cycle_params_tau3_bounds_fast_end():
    from bmsoc import ocv as ocvmod
    from bmsoc.ecm import _unpack_vec, fit_cycle_params, model_voltage, RCParams
    rng = np.random.default_rng(9)
    n, dt = 3000, 1.0
    i = rng.normal(-0.6, 1.2, n)
    soc = 1.0 + np.concatenate([[0.0], np.cumsum(i[:-1])]) * dt / (3600.0 * 2.5)
    p_true = RCParams(0.025, 0.012, 15.0, 0.02, 400.0, dv=0.002, r3=0.018, tau3=6.0)
    v = model_voltage(p_true, 3.7, i, dt)
    flat = ocvmod.from_rest_points(np.linspace(0, 1, 11), np.full(11, 3.7), temp_c=25.0)
    win = [(soc, i, v, dt, flat, 25.0)]
    vec, info = fit_cycle_params(win, n_rc=3, max_nfev=300,
                                 tau3_bounds=(0.5, 30.0))
    r0k, r1, tau1, r2, tau2, r3, tau3, _rc, _dv = _unpack_vec(vec)
    assert info["rmse_v"] < 0.004
    # API 契约：tau3 必须落在给定区间（标签可与 tau1 互换，内容看极点）
    assert 0.5 <= tau3 <= 30.0
    # 快极点（τ∈[2,30] s 且 r 显著）必须被模型族认领
    poles = [(r1, tau1), (r2, tau2), (r3, tau3)]
    assert any(r > 0.005 and 2.0 <= t <= 30.0 for r, t in poles)


def test_fit_cycle_params_rejects_bad_tau3_bounds():
    from bmsoc.ecm import fit_cycle_params
    with pytest.raises(ValueError):
        fit_cycle_params([], n_rc=3, tau3_bounds=(30.0, 0.5))
    with pytest.raises(ValueError):
        fit_cycle_params([], n_rc=3, tau3_bounds=(0.0, 30.0))


# ---------------------------------------------------------------- replay 汇总口径

def test_replay_summary_shapes(s32):
    rows = []
    for inj in s32.INJECT_SET:
        for scale in s32.REPLAY_SCALES:
            for cap in s32.CAP_SCALES:
                ok = inj in ("none", "minus_fast") and scale == 1.0
                rows.append({"dataset": "panasonic", "temp_c": -20.0,
                             "profile": "hwfet", "inject": inj, "scale": scale,
                             "cap_scale": cap, "q_rel_err": 0.0 if ok else 0.4,
                             "q_ok": ok, "rmse_soc": 0.01})
    tbl = s32._replay_summary_r6(pd.DataFrame(rows))
    assert tbl["none"]["q_ok_rate"] == 1.0
    assert tbl["full"]["q_ok_rate"] == 0.0
    assert tbl["minus_fast"]["q_ok_rate"] == 1.0
    assert "by_fold" in tbl["full"]
