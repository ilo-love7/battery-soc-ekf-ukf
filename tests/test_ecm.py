"""等效电路模型与参数辨识的测试。"""
import numpy as np
import pytest

from bmsoc import ocv as ocvmod
from bmsoc.ecm import (RCParams, ParamGrid, identify_stack, model_voltage,
                       simulate, simulate_fast, step_matrices, time_constant_split)


def _curve(slope=1.0, base=3.1):
    soc = np.linspace(0, 1, 21)
    return ocvmod.from_rest_points(soc, base + slope * soc, temp_c=25.0)


def test_step_matrices_match_exponential():
    p = RCParams(r0=0.02, r1=0.01, tau1=10.0, r2=0.05, tau2=500.0)
    a1, b1, a2, b2 = step_matrices(p, dt=2.0)
    assert abs(a1 - np.exp(-0.2)) < 1e-12
    assert abs(b1 - 0.01 * (1 - np.exp(-0.2))) < 1e-12
    assert a1 < a2                       # τ 大的衰减慢


def test_soc_integration_with_constant_current():
    p = RCParams(0.02, 0.01, 10.0, 0.05, 500.0)
    n, dt, cap = 101, 1.0, 2.5
    i = np.full(n, -2.5)                 # 1C 放电 100 s
    out = simulate(p, _curve(), i, dt, soc0=1.0, capacity_ah=cap)
    expected = 1.0 - 2.5 * 100 / 3600.0 / cap
    assert abs(out["soc"][-1] - expected) < 1e-9


def test_polarization_reaches_steady_state():
    p = RCParams(0.02, 0.01, 5.0, 0.05, 50.0)
    i = np.full(4000, -1.0)
    out = simulate(p, _curve(), i, 1.0, soc0=1.0, capacity_ah=2.5)
    # 长时间恒流后 u_j → R_j·i
    assert abs(out["u1"][-1] - (-0.01)) < 1e-3
    assert abs(out["u2"][-1] - (-0.05)) < 1e-3


def test_dv_offsets_terminal_voltage():
    p0 = RCParams(0.02, 0.01, 10.0, 0.05, 500.0, dv=0.0)
    p1 = RCParams(0.02, 0.01, 10.0, 0.05, 500.0, dv=0.05)
    i = np.zeros(10)
    v0 = model_voltage(p0, 3.7, i, 1.0)
    v1 = model_voltage(p1, 3.7, i, 1.0)
    assert np.allclose(v1 - v0, 0.05)


def test_simulate_fast_matches_simulate():
    p = RCParams(0.02, 0.01, 8.0, 0.04, 300.0, dv=0.003)
    rng = np.random.default_rng(0)
    i = rng.normal(-0.5, 1.5, 600)
    a = simulate(p, _curve(), i, 1.0, soc0=0.9, capacity_ah=2.5)
    soc_b, v_b = simulate_fast(p.r0, p.r1, p.tau1, p.r2, p.tau2, _curve(), i, 0.9, 2.5, 1.0,
                               dv=p.dv)
    assert np.max(np.abs(a["voltage"] - v_b)) < 2e-3
    assert np.max(np.abs(a["soc"] - soc_b)) < 1e-9


def test_identify_stack_recovers_known_params():
    true = RCParams(r0=0.030, r1=0.008, tau1=12.0, r2=0.060, tau2=600.0)
    dt, n = 1.0, 1200
    t = np.arange(n) * dt
    windows = []
    for amp in (-2.9, -5.8, -11.6):
        i = np.zeros(n)
        i[10:20] = amp                          # 10 s 脉冲
        ocv = 3.8
        v = model_voltage(true, ocv, i, dt)
        windows.append({"i": i, "v": v, "ocv": ocv, "dt": dt})
    p, info = identify_stack(windows)
    assert info["rmse_v"] < 2e-3
    assert abs(p.r0 - true.r0) < 0.01
    assert abs(p.tau1 - true.tau1) <= max(3.0, 0.5 * true.tau1)   # τ 在格点上
    assert abs(p.tau2 - true.tau2) <= max(120.0, 0.5 * true.tau2)


def test_time_constant_split_flags_unidentifiable():
    p = RCParams(0.02, 0.01, 10.0, 0.01, 12.0)
    assert time_constant_split(p) < 2.0
    p2 = RCParams(0.02, 0.01, 5.0, 0.01, 500.0)
    assert time_constant_split(p2) > 50


def test_param_grid_interpolates_soc():
    temps = np.array([0.0, 25.0])
    socs = np.array([0.0, 0.5, 1.0])
    vals = np.zeros((2, 3, 5))
    vals[0, :, 0] = [0.05, 0.04, 0.03]
    vals[1, :, 0] = [0.03, 0.02, 0.01]
    vals[:, :, 1] = 0.005
    vals[:, :, 2] = 10.0
    vals[:, :, 3] = 0.05
    vals[:, :, 4] = 500.0
    grid = ParamGrid(temps, socs, vals)
    assert abs(grid.at(25.0, 0.0).r0 - 0.03) < 1e-9
    assert abs(grid.at(25.0, 0.25).r0 - 0.025) < 1e-9
    assert abs(grid.at(0.0, 0.5).r0 - 0.04) < 1e-9


def test_open_loop_from_truth_uses_truth_soc():
    from bmsoc.ecm import open_loop_from_truth
    vec = np.array([0.02, 0.02, 0.02, 0.01, 10.0, 0.05, 500.0, 0.0])
    soc = np.linspace(1.0, 0.9, 50)
    i = np.zeros(50)
    v = open_loop_from_truth(vec, soc, i, _curve(), 1.0)
    assert np.allclose(v, _curve()(soc))
