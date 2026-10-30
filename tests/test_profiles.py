"""工况容器与重采样的测试。

其中 ``test_resample_holds_value_across_sparse_gaps`` 是针对一个真实 bug 的回归测试：
早先版本在填充空区间时用了「全部有效值的均值」，把温箱静置段的电压从 4.17 V
灌成 3.47 V。
"""
import numpy as np
import pytest

from bmsoc import profiles as P


def _sparse_series():
    """模拟原始文件：60 s 稀疏静置 20 个点（v 从 4.180 缓慢降到 4.170），然后 0.1 s 密采样放电。"""
    t_rest = np.arange(0.0, 1200.1, 60.0)
    v_rest = 4.180 - 0.010 * t_rest / 1200.0
    i_rest = np.zeros_like(t_rest)
    t_dyn = np.arange(1200.1, 1500.0, 0.1)
    i_dyn = -3.0 * np.ones_like(t_dyn)
    v_dyn = 4.10 - 0.25 * (t_dyn - t_dyn[0]) / (t_dyn[-1] - t_dyn[0])
    return (np.concatenate([t_rest, t_dyn]), np.concatenate([i_rest, i_dyn]),
            np.concatenate([v_rest, v_dyn]), np.concatenate([np.full(t_rest.size, 25.0),
                                                             np.full(t_dyn.size, 25.0)]))


def test_resample_holds_value_across_sparse_gaps():
    """空区间必须用前值保持，而不是整段均值。"""
    t, i, v, T = _sparse_series()
    r = P.resample(t, i, v, T, dt=1.0)
    rest = r["t_s"] < 1199.0
    assert rest.sum() > 500
    # 静置段电压必须在 4.17~4.19 之间，不能被拉到 3.x
    assert r["voltage_v"][rest].min() > 4.15
    assert r["voltage_v"][rest].max() < 4.19
    # 且随原始点单调不增（前值保持 → 阶梯）
    assert np.all(np.diff(r["voltage_v"][rest]) <= 1e-9)


def test_resample_conserves_charge():
    t, i, v, T = _sparse_series()
    r = P.resample(t, i, v, T, dt=1.0)
    q_raw = np.trapezoid(i, t) / 3600.0
    q_rs = np.trapezoid(r["current_a"], r["t_s"]) / 3600.0
    assert abs(q_rs - q_raw) < 0.01 * abs(q_raw)


def test_resample_takes_last_voltage_in_bin():
    t = np.array([0.0, 0.3, 0.7, 1.2, 1.8, 2.2])
    v = np.array([4.0, 4.1, 4.2, 4.3, 4.4, 4.5])
    i = np.zeros_like(t)
    r = P.resample(t, i, v, np.full_like(t, 20.0), dt=1.0)
    assert r["t_s"].size == 3
    assert r["voltage_v"][0] == pytest.approx(4.2)
    assert r["voltage_v"][1] == pytest.approx(4.4)


def test_resample_rejects_tiny_input():
    with pytest.raises(ValueError):
        P.resample(np.array([0.0]), np.zeros(1), np.zeros(1), np.zeros(1))


def test_sample_gaps_quantifies_sparsity():
    t, _, _, _ = _sparse_series()
    g = P.sample_gaps(t, dt=1.0)
    assert 0.5 < g["gap_frac"] < 0.9          # 大部分时间处于 60 s 稀疏段
    assert g["gap_max_s"] == pytest.approx(60.0, abs=1e-6)


def test_cum_charge_matches_trapezoid():
    t = np.linspace(0, 3600, 3601)
    i = np.full_like(t, -2.0)
    q = P.cum_charge_ah(t, i)
    assert q[0] == 0.0
    assert q[-1] == pytest.approx(-2.0, abs=1e-6)


def test_rest_segments_finds_soak():
    t, i, _, _ = _sparse_series()
    segs = P.rest_segments(t, i, thr=0.05, min_dur_s=120.0)
    assert len(segs) == 1
    assert segs[0][2] == pytest.approx(1200.0, abs=1.0)


def test_build_profile_soc_from_charge():
    t, i, v, T = _sparse_series()
    pr = P.build_profile(t, i, v, T, capacity_ah=1.0, soc0=1.0, dt=1.0)
    # 300 s × 3 A = 0.25 Ah → SOC 掉 25 pp
    assert pr.soc_ref[0] == pytest.approx(1.0)
    assert pr.soc_ref[-1] == pytest.approx(0.75, abs=0.01)


def test_trim_leading_rest_keeps_window():
    t, i, v, T = _sparse_series()
    pr = P.build_profile(t, i, v, T, capacity_ah=1.0, soc0=1.0, dt=1.0)
    short = pr.trim_leading_rest(keep_s=100.0)
    assert len(short) < len(pr)
    # 行驶起点在 1200 s，只保留前 100 s 静置 → 裁掉约 1100 个样本
    assert len(pr) - len(short) == pytest.approx(1100, abs=2)
    assert short.soc_ref[0] == pytest.approx(pr.soc_ref[0])      # 静置段无电量变化
    assert abs(short.current_a[0]) <= 0.05


def test_trim_leading_rest_is_noop_without_long_rest():
    t = np.arange(0.0, 100.0, 1.0)
    pr = P.build_profile(t, -1.0 * np.ones_like(t), np.full_like(t, 4.0),
                         np.full_like(t, 25.0), capacity_ah=1.0, dt=1.0)
    assert len(pr.trim_leading_rest(keep_s=300.0)) == len(pr)


def test_profile_dt_and_slice():
    t, i, v, T = _sparse_series()
    pr = P.build_profile(t, i, v, T, capacity_ah=1.0, dt=1.0)
    assert pr.dt == pytest.approx(1.0)
    assert len(pr.slice(10, 20)) == 10
    assert pr.slice(10, 20).t_s[0] == 0.0
