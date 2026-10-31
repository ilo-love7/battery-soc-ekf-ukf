"""OCV 曲线与二维表的测试。"""
import numpy as np
import pytest

from bmsoc import ocv as ocvmod


def test_pava_makes_monotone_and_reports_violation():
    soc = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    raw = np.array([3.0, 3.3, 3.25, 3.6, 3.55, 4.0])       # 两处倒挂
    c = ocvmod.from_rest_points(soc, raw, temp_c=25.0)
    assert np.all(np.diff(c.ocv) >= -1e-12)
    assert c.violation_v > 0.0


def test_curve_is_monotone_and_docv_positive():
    soc = np.linspace(0, 1, 11)
    v = 3.0 + 1.2 * soc
    c = ocvmod.from_rest_points(soc, v, temp_c=25.0)
    grid = np.linspace(0, 1, 101)
    assert np.all(np.diff(c(grid)) >= -1e-9)
    assert np.all(c.docv(grid) > 0.5)


def test_inverse_roundtrip():
    soc = np.linspace(0, 1, 11)
    c = ocvmod.from_rest_points(soc, 3.0 + 1.1 * soc ** 0.8, temp_c=25.0)
    s = np.linspace(0.05, 0.95, 19)
    back = c.inverse(c(s))
    assert np.max(np.abs(back - s)) < 5e-3


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        ocvmod.from_rest_points([0.1, 0.5], [3.3, 3.8])


def test_duplicate_soc_is_averaged():
    c = ocvmod.from_rest_points([0.5, 0.5, 0.6, 0.7], [3.5, 3.6, 3.7, 3.8], temp_c=25.0)
    assert c.soc.size == 3
    assert abs(float(c(np.array([0.5]))[0]) - 3.55) < 1e-6


def _table():
    curves = {}
    for t, off in ((0.0, -0.02), (25.0, 0.0)):
        soc = np.linspace(0, 1, 21)
        curves[t] = ocvmod.from_rest_points(soc, 3.1 + 1.0 * soc + off, temp_c=t)
    return ocvmod.OCVTable(curves)


def test_ocv2d_at_node_matches_curve():
    tab = _table()
    d2 = ocvmod.OCV2D(tab)
    s = np.array([0.2, 0.5, 0.8])
    assert np.allclose(d2(s, 25.0), tab.at(25.0)(s), atol=1e-9)
    assert np.allclose(d2(s, 0.0), tab.at(0.0)(s), atol=1e-9)


def test_ocv2d_temperature_interpolation_is_linear():
    tab = _table()
    d2 = ocvmod.OCV2D(tab)
    s = np.array([0.5])
    lo = float(d2(s, 0.0)[0])
    hi = float(d2(s, 25.0)[0])
    mid = float(d2(s, 12.5)[0])
    assert abs(mid - 0.5 * (lo + hi)) < 2e-3         # 0.5 K 量化带来的小误差


def test_ocv2d_temperature_extrapolation_clamps():
    """温度方向外推取端点曲线（正常使用范围内不会用到，但要定义清楚）。"""
    tab = _table()
    d2 = ocvmod.OCV2D(tab)
    assert abs(float(d2(np.array([0.5]), -40.0)[0]) - float(d2(np.array([0.5]), 0.0)[0])) < 1e-9
    assert abs(float(d2(np.array([0.5]), 80.0)[0]) - float(d2(np.array([0.5]), 25.0)[0])) < 1e-9


def test_ocv2d_soc_extrapolation_is_linear_not_clamped():
    """SOC 越界必须按端点斜率线性外推。

    夹紧会让 UKF 的 sigma 点均值系统性地偏向低 SOC，滤波器被推到限幅上卡死
    （-20 °C 实测发散的根因）。
    """
    d2 = ocvmod.OCV2D(_table())
    row = d2.row(25.0)
    h = float(d2.soc_grid[1] - d2.soc_grid[0])
    d_hi = (row[-1] - row[-2]) / h
    d_lo = (row[1] - row[0]) / h
    assert float(d2(np.array([1.05]), 25.0)[0]) == pytest.approx(row[-1] + 0.05 * d_hi)
    assert float(d2(np.array([-0.05]), 25.0)[0]) == pytest.approx(row[0] - 0.05 * d_lo)
    assert float(d2.docv(np.array([1.05]), 25.0)[0]) == pytest.approx(d_hi)
    assert float(d2.docv(np.array([-0.05]), 25.0)[0]) == pytest.approx(d_lo)
    # 连续：越界点与边界点之差 = 斜率 × 距离
    assert float(d2(np.array([1.0]), 25.0)[0]) == pytest.approx(row[-1])


def test_ocv_curve_soc_extrapolation_is_linear():
    soc = np.linspace(0, 1, 11)
    c = ocvmod.from_rest_points(soc, 3.0 + 1.0 * soc, temp_c=25.0)
    assert float(c(np.array([1.1]))[0]) == pytest.approx(4.1, abs=1e-6)
    assert float(c(np.array([-0.1]))[0]) == pytest.approx(2.9, abs=1e-6)
    assert float(c.docv(np.array([1.1]))[0]) == pytest.approx(1.0, abs=1e-3)


def test_ocv2d_batch_equals_scalar():
    tab = _table()
    d2 = ocvmod.OCV2D(tab)
    soc = np.array([0.1, 0.3, 0.5, 0.7])
    temp = np.array([0.0, 6.0, 25.0, 13.0])
    out = d2.batch(soc, temp)
    ref = np.array([float(d2(np.array([s]), t)[0]) for s, t in zip(soc, temp)])
    assert np.allclose(out, ref)


def test_voltage_gap_positive():
    soc = np.linspace(0, 1, 11)
    c = ocvmod.from_rest_points(soc, 3.0 + 0.5 * soc, temp_c=25.0)
    assert ocvmod.voltage_gap(c, 0.5) > 0


def _staircase():
    """模拟 HPPC：每个 SOC 档 5 个几乎同 SOC 的点，档内电压完全相同（记录分辨率所致）。"""
    levels = np.arange(0.05, 0.99, 0.09)
    soc, ocv = [], []
    for lv in levels:
        soc.extend([lv, lv + 0.002, lv + 0.004, lv + 0.006, lv + 0.008])
        ocv.extend([3.10 + 0.95 * lv] * 5)
    return np.array(soc), np.array(ocv)


def test_smoothing_restores_meaningful_slope():
    """不做平滑时 PCHIP 穿过阶梯，dOCV/dSOC 会在 0 与几十之间乱跳。"""
    soc, ocv = _staircase()
    g = np.linspace(0.06, 0.94, 200)
    raw = ocvmod.from_rest_points(soc, ocv, temp_c=25.0, smooth_mv=0.0)
    sm = ocvmod.from_rest_points(soc, ocv, temp_c=25.0, smooth_mv=1.5)
    d_raw = np.asarray(raw.docv(g))
    d_sm = np.asarray(sm.docv(g))
    assert d_raw.min() < 0.05                 # 阶梯段斜率≈0
    assert d_sm.min() > 0.3                   # 平滑后恢复为有意义的斜率（真值 0.95）
    assert abs(np.median(d_sm) - 0.95) < 0.15
    assert d_sm.max() < 1.6


def test_smoothing_keeps_endpoints_and_monotonicity():
    soc, ocv = _staircase()
    sm = ocvmod.from_rest_points(soc, ocv, temp_c=25.0, smooth_mv=1.5)
    assert sm.soc[0] == pytest.approx(soc.min())
    assert sm.soc[-1] == pytest.approx(soc.max())
    assert np.all(np.diff(sm.ocv) >= -1e-12)
    assert sm.raw_soc is not None and sm.raw_soc.size == soc.size


def test_smoothing_skipped_for_few_points():
    soc = np.linspace(0, 1, 8)
    c = ocvmod.from_rest_points(soc, 3.0 + soc, temp_c=25.0, smooth_mv=1.5)
    assert c.soc.size == 8                    # 点数不足时不重建网格
    assert c.raw_soc is None


def test_smoothed_curve_roundtrip_is_not_smoothed_twice():
    """落盘的 soc/ocv 已是最终曲线，读回来必须关掉平滑（本项目踩过二次平滑的坑）。"""
    soc, ocv = _staircase()
    sm = ocvmod.from_rest_points(soc, ocv, temp_c=25.0, smooth_mv=1.5)
    d = sm.to_dict()
    assert d["smoothed"] is True
    again = ocvmod.from_rest_points(d["soc"], d["ocv"], temp_c=25.0, smooth_mv=0.0)
    g = np.linspace(0.05, 0.95, 50)
    assert np.max(np.abs(again(g) - sm(g))) < 1e-9
    # 若错误地二次平滑，曲线会被明显压低
    twice = ocvmod.from_rest_points(d["soc"], d["ocv"], temp_c=25.0, smooth_mv=1.5)
    assert np.max(np.abs(twice(g) - sm(g))) > 1e-6


def test_shape_corrected_adds_delta_and_fixes_slope():
    """ΔOCV 形状修正：数值必须加 Δ、docv 必须叠加 Δ 斜率（滤波器 H 依赖它）。"""
    soc = np.linspace(0, 1, 21)
    base = ocvmod.from_rest_points(soc, 3.2 + soc, temp_c=25.0)
    table = ocvmod.OCVTable({25.0: base})
    b2d = ocvmod.OCV2D(table)
    knots = [0.1, 0.5, 0.9]
    delta_mv = [10.0, -20.0, 30.0]
    sc = ocvmod.ShapeCorrected(b2d, knots, delta_mv)
    v0 = b2d(np.array([0.5]), 25.0)
    v1 = sc(np.array([0.5]), 25.0)
    assert float(v1[0] - v0[0]) == pytest.approx(-0.020, abs=1e-12)
    # 线段内部斜率（0.1→0.5 段：Δ 从 +10 → −20 mV ⇒ dΔ/dsoc = −0.075 V/unit）
    # 不在结点 0.5 上测 —— 折点处导数无定义，中心差分给的是两侧平均
    d0 = b2d.docv(np.array([0.3]), 25.0)
    d1 = sc.docv(np.array([0.3]), 25.0)
    assert float(d1[0] - d0[0]) == pytest.approx(-0.075, abs=5e-3)
    # 端点形状：0.1 处 +10 mV
    assert float(sc(np.array([0.1]), 25.0)[0] - b2d(np.array([0.1]), 25.0)[0]) == pytest.approx(
        0.010, abs=1e-12)


def test_shape_corrected_batch_matches_scalar():
    soc = np.linspace(0, 1, 11)
    base = ocvmod.from_rest_points(soc, 3.1 + 0.5 * soc, temp_c=25.0)
    b2d = ocvmod.OCV2D(ocvmod.OCVTable({25.0: base}))
    sc = ocvmod.ShapeCorrected(b2d, [0.0, 0.5, 1.0], [0.0, 40.0, -10.0])
    s = np.array([0.05, 0.33, 0.66, 0.95])
    t = np.full(4, 25.0)
    assert np.allclose(sc.batch(s, t), sc(s, 25.0), atol=1e-12)
