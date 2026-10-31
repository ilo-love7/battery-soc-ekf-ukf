"""CALCE LFP 数据层测试（全部用合成表，不需要数据集）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bmsoc import calce as ca
from bmsoc import io_calce as io


# ---------------------------------------------------------------- 包名解析
@pytest.mark.parametrize("name,temp", [
    ("A123_OCV-10-20120629.zip", -10.0),
    ("A123_OCV0-20120618.zip", 0.0),
    ("A123_OCV25-20120905.zip", 25.0),
    ("A123_DST-US06-FUDS-10.zip", 10.0),
    ("A123_DST-US06-FUDS-0.zip", 0.0),
    ("A123_DST-US06-FUDS-N10.zip", -10.0),
])
def test_parse_package_temperature_quirks(name, temp):
    """两个包族的命名规则不一致：OCV 的 "-10" 是负号，DST 的 "-10" 是分隔符。

    若统一按「- 是分隔符」解析，``A123_OCV-10`` 会变成 +10 °C，与 ``A123_OCV10``
    撞车，静默丢掉一个温度档。
    """
    assert io.parse_package(name)["temp_c"] == temp


def test_parse_package_kind():
    assert io.parse_package("A123_OCV25-20120905.zip")["kind"] == "ocv"
    assert io.parse_package("A123_DST-US06-FUDS-25.zip")["kind"] == "dst"


def test_parse_package_rejects_unknown():
    with pytest.raises(ValueError):
        io.parse_package("NASA_random.mat")


# ---------------------------------------------------------------- 分段
def _frame(steps, dt: float = 1.0) -> pd.DataFrame:
    """steps: [(step_id, n, current, v0, v1)] → 合成 Arbin 表。

    ``dt`` 默认 1 s（动态段）；C/20 慢放段在原始数据里是 5 s 一个点。
    """
    rows = []
    t = 0.0
    for sid, n, cur, v0, v1 in steps:
        for k in range(n):
            rows.append({"step": sid, "t_s": t, "current_a": cur,
                         "voltage_v": v0 + (v1 - v0) * k / max(n - 1, 1),
                         "temp_c_set": 25.0})
            t += dt
    return pd.DataFrame(rows)


def test_segments_picks_three_discharges_not_charges():
    """低温包的 CC-CV 充电段也有 6.9 h，只看时长会把充电段选进来。"""
    df = _frame([
        (1, 60, 0.0, 3.3, 3.3),
        (2, 4000, +1.0, 3.3, 3.6),          # 长充电，必须被排除
        (3, 3000, -1.0, 3.6, 2.0),          # 动态放电 1
        (4, 4000, +1.0, 3.3, 3.6),          # 长充电
        (5, 3000, -1.0, 3.6, 2.0),          # 动态放电 2
        (6, 4000, +1.0, 3.3, 3.6),          # 长充电
        (7, 3000, -1.0, 3.6, 2.0),          # 动态放电 3
    ])
    segs = ca.segments(df)
    assert list(segs) == ["dst", "us06", "fuds"]
    for s in segs.values():
        assert s["current_a"].mean() < 0
        assert len(s) == 3000


def test_segments_raises_when_count_is_wrong():
    df = _frame([(1, 3000, -1.0, 3.6, 2.0), (2, 3000, -1.0, 3.6, 2.0)])
    with pytest.raises(ValueError, match="期望 3 段"):
        ca.segments(df)


# ---------------------------------------------------------------- OCV 曲线
def test_ocv_curve_points_from_slow_discharge():
    """C/20 慢放支路：SOC 从 1 降到 0，电压单调降。"""
    n, dt = 2400, 5.0
    df = _frame([(1, 60, 0.0, 3.35, 3.35),
                 (2, n, -0.05, 3.35, 2.00),
                 (3, 60, 0.0, 2.00, 2.10)], dt=dt)
    soc, ocv, q = ca.ocv_curve_points(df, n_grid=61)

    # 返回的是**已按 SOC 升序排好**的曲线（供 PCHIP / 平滑使用）
    assert q == pytest.approx(0.05 * (n - 1) * dt / 3600.0, rel=1e-6)
    assert soc.shape == ocv.shape == (61,)
    assert soc[0] == pytest.approx(0.0, abs=1e-9)
    assert soc[-1] == pytest.approx(1.0, abs=1e-9)
    assert np.all(np.diff(soc) > 0)
    assert np.all(np.diff(ocv) > 0)      # 放电电流下电压随 SOC 单调升


def test_ocv_curve_points_requires_slow_branch():
    df = _frame([(1, 3000, -1.0, 3.6, 2.0)])
    with pytest.raises(ValueError, match="C/20"):
        ca.ocv_curve_points(df)


# ---------------------------------------------------------------- 温度列
def test_temp_column_fallback():
    a = pd.DataFrame({"temp_c_set": [25.0, 25.0]})
    b = pd.DataFrame({"temp_c": [25.0, 25.0]})
    assert np.allclose(ca._temp(a), 25.0)
    assert np.allclose(ca._temp(b), 25.0)
