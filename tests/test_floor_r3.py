"""第七轮（脚本 26）纯函数的合成数据测试 —— 不依赖任何数据集。

覆盖：记忆时距、尾部外推（自由-τ 与约束-τ）、合成正对照的表偏/残余极化分离、
regime 掩码切分。
"""
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(ROOT, "scripts", "26_floor_round3.py")


def _load_s26():
    spec = importlib.util.spec_from_file_location("s26_floor", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def s26():
    return _load_s26()


def test_time_since_big_counts_from_last_big(s26):
    i = np.array([0.0, 0.0, 5.0, 0.1, 0.2, 0.3, -4.0, 0.0, 0.0])
    t = s26._time_since_big(i, dt=1.0)
    assert np.isnan(t[0]) and np.isnan(t[1])
    assert t[2] == 0.0
    assert t[3] == 1.0 and t[5] == 3.0
    assert t[6] == 0.0 and t[8] == 2.0


def test_time_since_big_all_small_is_nan(s26):
    t = s26._time_since_big(np.zeros(7), dt=2.0)
    assert np.all(np.isnan(t))


def test_tail_equilibrium_recovers_c_and_gap(s26):
    # v(t) = 4.0 - 0.05*exp(-t/200)：末电压在平衡点之下 50 mV 量级
    t = np.arange(0.0, 601.0, 1.0)
    c_true, a_true, tau_true = 4.0, -0.05, 200.0
    v = c_true + a_true * np.exp(-t / tau_true)
    fit = s26._tail_equilibrium(t, v, 0, len(t) - 1)
    assert fit is not None
    gap_true = (c_true - v[-1]) * 1e3
    assert fit["gap_mv"] == pytest.approx(gap_true, abs=1.0)
    assert fit["end_slope_mv300"] > 0  # 还在往平衡走


def test_tail_equilibrium_none_on_tiny_segment(s26):
    t = np.arange(0.0, 25.0, 1.0)
    v = np.full_like(t, 3.9)
    assert s26._tail_equilibrium(t, v, 0, len(t) - 1) is None


def test_tail_equilibrium_fixedtau_recovers_delta_with_polarization(s26):
    # 表偏 +20 mV，残余极化 80 mV/500 s，300 s 静置 —— 约束-τ 应把两者分开
    rng = np.random.default_rng(20261031)
    t = np.arange(0.0, 301.0, 1.0)
    delta, amp, tau = 0.020, 0.080, 500.0
    v = 4.0 + delta + amp * np.exp(-t / tau) + rng.normal(0, 0.001, t.size)
    fit = s26._tail_equilibrium_fixedtau(t, v, 0, len(t) - 1, tau)
    assert fit is not None
    gap_eq = (v[-1] + fit["gap_mv"] * 1e-3) - 4.0
    assert gap_eq * 1e3 == pytest.approx(delta * 1e3, abs=5.0)


def test_synthetic_control_separates_table_offset_and_polarization(s26):
    rng = np.random.default_rng(20261031)
    ctrl = s26._synthetic_rest_control(rng, [492.4])
    settled = ctrl[s26._ctrl_key(492.4, 300.0, "settled")]
    # settled：raw 读数直接给表偏
    assert abs(settled["gap_end_mv"] - settled["delta_mv"]) <= 5.0
    pol = ctrl[s26._ctrl_key(492.4, 300.0, "polarized")]
    two = ctrl[s26._ctrl_key(492.4, 300.0, "twomode")]
    # polarized：raw 读数被残余极化污染（>30 mV），fixτ 回收表偏
    assert pol["gap_end_mv"] - pol["delta_mv"] > 30.0
    for case in (pol, two):
        assert abs(case["gap_eq_fixtau_mv"] - case["delta_mv"]) <= 5.0


def test_regime_masks_split_start_end_and_dynamic(s26):
    # 300 s 起始静置 + 400 s 驱动 + 300 s 尾段静置
    dt = 1.0
    i = np.concatenate([np.zeros(300), np.full(400, -3.0), np.zeros(300)])
    t = np.arange(i.size) * dt
    mask_start, mask_other, dyn_low, dyn_high, segs = s26._regime_masks(i, t)
    assert mask_start.sum() == 300
    assert mask_other.sum() == 300
    assert dyn_high.sum() == 400
    assert dyn_low.sum() == 0
    assert len(segs) == 2
    assert not (mask_start & mask_other).any()
