"""三类估计器的测试。

全部用**合成工况**：给定一组已知参数与已知电流，用 :func:`bmsoc.ecm.simulate`
生成端电压，再把「电流 + 电压」交给滤波器。这样真值 SOC 是精确已知的，
可以判断估计器本身是否正确，而不是在和有模型误差的实测数据纠缠。
"""
import numpy as np
import pytest

from bmsoc import ocv as ocvmod
from bmsoc.ecm import RCParams, simulate
from bmsoc.filters import AEKF, ESTIMATORS, EKF, UKF, CoulombCounter, FilterConfig, make
from bmsoc.profiles import DrivingProfile

TRUE = RCParams(r0=0.030, r1=0.012, tau1=10.0, r2=0.040, tau2=400.0)
CAP = 2.5
DT = 1.0


def _curve(slope: float = 1.0, base: float = 3.1):
    soc = np.linspace(0.0, 1.0, 21)
    return ocvmod.from_rest_points(soc, base + slope * soc, temp_c=25.0)


def _synth(n: int = 900, slope: float = 1.0, dv: float = 0.0, seed: int = 1,
           i_scale: float = 3.0):
    """随机工况 + 用真值参数生成的电压。"""
    rng = np.random.default_rng(seed)
    i = np.clip(rng.normal(0.0, i_scale, n), -3 * i_scale, i_scale)
    i[0] = 0.0
    p = RCParams(TRUE.r0, TRUE.r1, TRUE.tau1, TRUE.r2, TRUE.tau2, dv=dv)
    out = simulate(p, _curve(slope), i, DT, soc0=0.9, capacity_ah=CAP)
    prof = DrivingProfile(t_s=np.arange(n) * DT, current_a=i, voltage_v=out["voltage"],
                          temp_c=np.full(n, 25.0), soc_ref=out["soc"], capacity_ah=CAP,
                          meta={"profile": "synth"})
    return prof


def test_cc_integrates_exactly():
    n = 361
    i = np.full(n, -2.0)
    prof = DrivingProfile(np.arange(n) * DT, i, np.zeros(n), np.full(n, 25.0),
                          np.zeros(n), CAP)
    est = CoulombCounter(ocv=_curve(), params=TRUE, capacity_ah=CAP,
                         cfg=FilterConfig(soc0=1.0))
    out = est.run(prof)
    # 2 A × 360 s = 0.2 Ah → 0.08 SOC
    assert out["soc_hat"][-1] == pytest.approx(1.0 - 0.2 / CAP, abs=1e-9)


def test_cc_drifts_with_current_bias():
    n = 3601
    i = np.zeros(n)
    prof = DrivingProfile(np.arange(n) * DT, i, np.zeros(n), np.full(n, 25.0),
                          np.zeros(n), CAP)
    est = CoulombCounter(ocv=_curve(), params=TRUE, capacity_ah=CAP,
                         cfg=FilterConfig(soc0=1.0))
    out = est.run(prof, current_bias_a=0.02)
    # 0.02 A × 3600 s = 0.02 Ah → 0.008 SOC
    assert out["soc_hat"][-1] == pytest.approx(1.0 + 0.02 / CAP, abs=1e-6)


def test_ekf_tracks_when_model_is_exact():
    prof = _synth()
    cfg = FilterConfig(soc0=0.9, sigma_soc0=0.02)
    out = EKF(ocv=_curve(), params=TRUE, capacity_ah=CAP, cfg=cfg).run(prof)
    assert np.max(np.abs(out["soc_hat"] - prof.soc_ref)) < 0.01


def test_ukf_tracks_when_model_is_exact():
    prof = _synth()
    cfg = FilterConfig(soc0=0.9, sigma_soc0=0.02)
    out = UKF(ocv=_curve(), params=TRUE, capacity_ah=CAP, cfg=cfg).run(prof)
    assert np.max(np.abs(out["soc_hat"] - prof.soc_ref)) < 0.01


def test_ukf_equals_ekf_for_linear_measurement():
    """OCV 为直线时无迹变换与线性化等价，两者结果应重合。"""
    prof = _synth(slope=1.0)
    cfg = FilterConfig(soc0=0.75, sigma_soc0=0.15)
    a = EKF(ocv=_curve(1.0), params=TRUE, capacity_ah=CAP, cfg=cfg).run(prof)
    b = UKF(ocv=_curve(1.0), params=TRUE, capacity_ah=CAP, cfg=cfg).run(prof)
    assert np.max(np.abs(a["soc_hat"] - b["soc_hat"])) < 1e-4


def test_ekf_recovers_wrong_initial_soc():
    prof = _synth()
    cfg = FilterConfig(soc0=0.60, sigma_soc0=0.25)       # 真值 0.90
    out = EKF(ocv=_curve(), params=TRUE, capacity_ah=CAP, cfg=cfg).run(prof)
    tail = out["soc_hat"][-int(0.3 * len(prof)):]
    assert abs(np.mean(tail - prof.soc_ref[-tail.size:])) < 0.02


def test_dv_must_be_passed_to_filter():
    """生成数据带 +40 mV 偏移：不给滤波器 dv，就会停在约 4 pp 的偏差上。"""
    prof = _synth(dv=0.04)
    cfg = FilterConfig(soc0=0.9, sigma_soc0=0.02)
    good = EKF(ocv=_curve(), params=RCParams(*TRUE.as_array(), dv=0.04),
               capacity_ah=CAP, cfg=cfg).run(prof)
    bad = EKF(ocv=_curve(), params=TRUE, capacity_ah=CAP, cfg=cfg).run(prof)
    e_good = np.max(np.abs(good["soc_hat"] - prof.soc_ref))
    e_bad = np.max(np.abs(bad["soc_hat"] - prof.soc_ref))
    assert e_good < 0.01
    assert e_bad > e_good


def test_init_state_uses_steady_polarization():
    est = EKF(ocv=_curve(), params=TRUE, capacity_ah=CAP, cfg=FilterConfig(soc0=0.5))
    x = est._init_state(0.5, i0=-2.0)
    assert x[1] == pytest.approx(TRUE.r1 * -2.0)
    assert x[2] == pytest.approx(TRUE.r2 * -2.0)
    assert x[3] == 0.0


def test_covariance_stays_symmetric_psd():
    prof = _synth()
    cfg = FilterConfig(soc0=0.7, sigma_soc0=0.2)
    for cls in (EKF, UKF):
        out = cls(ocv=_curve(), params=TRUE, capacity_ah=CAP, cfg=cfg).run(prof)
        assert np.all(np.isfinite(out["sigma_soc"]))
        assert np.all(out["sigma_soc"] >= 0.0)


def test_soc_is_clipped_to_configured_range():
    n = 400
    i = np.full(n, -8.0)                                  # 猛烈放电
    prof = DrivingProfile(np.arange(n) * DT, i, np.zeros(n), np.full(n, 25.0),
                          np.zeros(n), CAP)
    cfg = FilterConfig(soc0=1.0, soc_clip=(-0.05, 1.05))
    est = CoulombCounter(ocv=_curve(), params=TRUE, capacity_ah=CAP, cfg=cfg)
    out = est.run(prof)
    assert out["soc_hat"].min() >= -0.05 - 1e-12


def test_aekf_r_stays_within_bounds():
    prof = _synth()
    cfg = FilterConfig(soc0=0.7, sigma_soc0=0.2)
    est = AEKF(ocv=_curve(), params=TRUE, capacity_ah=CAP, cfg=cfg,
               r_min=4e-4, r_max=0.09)
    out = est.run(prof)
    # 前 window/2 步还没触发自适应，R 仍是初始值 sigma_v^2；之后必须落在界内
    assert out["R"][60:].min() >= 4e-4 - 1e-12
    assert out["R"][60:].max() <= 0.09 + 1e-12


def test_bias_state_absorbs_constant_offset():
    """生成数据比模型高 30 mV：偏置状态应停在 +30 mV，而不是让 SOC 去背这个锅。"""
    prof = _synth(dv=0.03)
    cfg = FilterConfig(soc0=0.9, sigma_soc0=0.02)
    out = EKF(ocv=_curve(), params=TRUE, capacity_ah=CAP, cfg=cfg).run(prof)
    assert out["bias_hat"][-1] == pytest.approx(0.03, abs=0.015)
    assert np.max(np.abs(out["soc_hat"] - prof.soc_ref)) < 0.02


def test_make_and_registry():
    assert set(ESTIMATORS) == {"cc", "ekf", "ukf", "aekf"}
    est = make("ekf", ocv=_curve(), params=TRUE, capacity_ah=CAP)
    assert isinstance(est, EKF)
    with pytest.raises(KeyError):
        make("nope", ocv=_curve(), params=TRUE, capacity_ah=CAP)


def test_run_rejects_non_profile():
    est = EKF(ocv=_curve(), params=TRUE, capacity_ah=CAP)
    with pytest.raises(TypeError):
        est.run({"t_s": np.zeros(3)})
