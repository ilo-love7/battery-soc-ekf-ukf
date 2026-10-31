"""三类估计器的测试。

全部用**合成工况**：给定一组已知参数与已知电流，用 :func:`bmsoc.ecm.simulate`
生成端电压，再把「电流 + 电压」交给滤波器。这样真值 SOC 是精确已知的，
可以判断估计器本身是否正确，而不是在和有模型误差的实测数据纠缠。
"""
import numpy as np
import pytest

from bmsoc import ocv as ocvmod
from bmsoc.ecm import RCParams, simulate, simulate_tv
from bmsoc.filters import (AEKF, ESTIMATORS, EKF, EKFQ, UKF, CoulombCounter, FilterConfig,
                           SoHSchedule, make)
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


# ---- EKFQ：容量作为第五状态的联合估计 ----
# 这些测试守的是 scripts/15 的结论依据：动态激励下 q 可辨识、恒流下不可辨识。

def test_ekfq_recovers_capacity_under_dynamic_excitation():
    """随机变号电流下，q0 错 10% 应被拉回真值附近，且 SOC 误差远小于固定 Q。

    注意放电深度必须够（这里约 70% DOD）：q 的信息来自「累积电量 vs 电压锚点」
    的差，浅放电时这个差太小，会被 b / u 状态吃掉（第一版测试只放 13% DOD，
    q 从 2.25 只动到 2.26，测试如实失败——这是可辨识性的真实约束，不是 bug）。"""
    n = 8000
    rng = np.random.default_rng(7)
    i = np.clip(rng.normal(-0.8, 1.5, n), -5.0, 2.0)   # 均值 −0.8 A ≈ 71% DOD
    i[0] = 0.0
    out = simulate(RCParams(*TRUE.as_array()), _curve(), i, DT, soc0=0.9, capacity_ah=CAP)
    prof = DrivingProfile(np.arange(n) * DT, i, out["voltage"], np.full(n, 25.0),
                          out["soc"], CAP, meta={"profile": "synth"})
    assert prof.soc_ref[-1] < 0.35            # 保证放电深度够，否则测试无效
    q_wrong = CAP * 0.90                      # 滤波器以为容量小 10%
    cfg = FilterConfig(soc0=0.9, sigma_soc0=0.02)
    bad = EKF(ocv=_curve(), params=TRUE, capacity_ah=q_wrong, cfg=cfg).run(prof)
    good = EKFQ(ocv=_curve(), params=TRUE, capacity_ah=q_wrong, cfg=cfg,
                sigma_q0=0.08).run(prof)
    e_bad = np.sqrt(np.mean((bad["soc_hat"] - prof.soc_ref) ** 2))
    e_good = np.sqrt(np.mean((good["soc_hat"] - prof.soc_ref) ** 2))
    assert good["q_hat"][-1] == pytest.approx(CAP, rel=0.05)
    assert e_good < 0.5 * e_bad               # 纠偏要起实质作用，不是挤小数点


def test_ekfq_fails_to_identify_q_under_constant_current():
    """恒流下 q 与 SOC 共线：正负误差都会被当成 SOC 漂移，q 基本不动。

    这是「容量可辨识性来自电流激励丰富度」的另一半证据——同一份代码、
    同一个 OCV，只是电流换成恒流，q 就收不回来。"""
    n = 3000
    i = np.full(n, -1.5)
    i[0] = 0.0
    out = simulate(RCParams(*TRUE.as_array()), _curve(), i, DT, soc0=0.9, capacity_ah=CAP)
    prof = DrivingProfile(np.arange(n) * DT, i, out["voltage"], np.full(n, 25.0),
                          out["soc"], CAP, meta={"profile": "synth"})
    cfg = FilterConfig(soc0=0.9, sigma_soc0=0.02)
    est = EKFQ(ocv=_curve(), params=TRUE, capacity_ah=CAP * 0.90, cfg=cfg, sigma_q0=0.02)
    out_f = est.run(prof)
    # q 收敛不到 5% 以内（与动态激励的上一个测试形成对照）
    assert abs(out_f["q_hat"][-1] / CAP - 1.0) > 0.03


def test_ekfq_neutral_when_capacity_correct():
    """容量本来就是对的时，加一个 q 状态不应把结果变差。"""
    prof = _synth()
    cfg = FilterConfig(soc0=0.7, sigma_soc0=0.2)
    base = EKF(ocv=_curve(), params=TRUE, capacity_ah=CAP, cfg=cfg).run(prof)
    ext = EKFQ(ocv=_curve(), params=TRUE, capacity_ah=CAP, cfg=cfg).run(prof)
    e_base = np.sqrt(np.mean((base["soc_hat"] - prof.soc_ref) ** 2))
    e_ext = np.sqrt(np.mean((ext["soc_hat"] - prof.soc_ref) ** 2))
    assert e_ext == pytest.approx(e_base, rel=0.25)
    assert ext["q_hat"][-1] == pytest.approx(CAP, rel=0.10)


def test_ekfq_f_jacobian_has_capacity_column():
    """F 的 ∂soc/∂q 必须是 −η·i·dt/(3600q²)（符号反了 q 会往错方向跑）。"""
    est = EKFQ(ocv=_curve(), params=TRUE, capacity_ah=CAP, cfg=FilterConfig(soc0=0.5))
    F = est._F(DT, i=-2.0, q=2.0)
    assert F[0, 4] == pytest.approx(2.0 * DT / (3600.0 * 4.0), abs=1e-12)  # η=1, −(−2)
    F2 = est._F(DT, i=2.0, q=2.0)
    assert F2[0, 4] < 0


# ---- SoHSchedule：A2 外部 SOH 锚点（Q̂/R̂0 零阶保持）----
# 这些测试守的是 §23 的路线改道：容量不在线辨识，改由外部锚点供给。

def test_soh_schedule_zoh_lookup():
    sch = SoHSchedule(t_upd=[0.0, 600.0, 1200.0], q_ah=[2.4, 2.5, 2.45],
                      r0_ohm=[0.03, 0.035, 0.04])
    assert sch.q(-10.0) == pytest.approx(2.4)       # 早于首点 → 取第一个值
    assert sch.q(599.9) == pytest.approx(2.4)       # 零阶保持
    assert sch.q(600.0) == pytest.approx(2.5)       # 刷新点当场生效
    assert sch.q(9999.0) == pytest.approx(2.45)     # 末点之后一直保持
    assert sch.r0(700.0, default=0.01) == pytest.approx(0.035)
    qa = sch.q_at(np.array([-1.0, 100.0, 600.0, 1500.0]))
    assert qa == pytest.approx([2.4, 2.4, 2.5, 2.45])
    sch2 = SoHSchedule([0.0], [2.5])                # 不给 r0 → 回退模型参数
    assert sch2.r0(100.0, default=0.0123) == pytest.approx(0.0123)


def test_soh_schedule_validation():
    with pytest.raises(ValueError):
        SoHSchedule([], [])                          # 空
    with pytest.raises(ValueError):
        SoHSchedule([0.0], [2.5, 2.6])               # 长度不一致
    with pytest.raises(ValueError):
        SoHSchedule([0.0, 0.0], [2.5, 2.6])          # t_upd 非严格递增
    with pytest.raises(ValueError):
        SoHSchedule([0.0, 1.0], [2.5, -1.0])         # q 非正
    with pytest.raises(ValueError):
        SoHSchedule([0.0, 1.0], [2.5, 2.6], r0_ohm=[0.01])   # r0 长度不一致


def test_cc_with_anchor_step_schedule_exact():
    """锚点在 t=180 s 刷新容量 → 库仑计数的积分速率当场改变（精确核对）。"""
    n = 361
    i = np.full(n, -2.0)
    prof = DrivingProfile(np.arange(n) * DT, i, np.zeros(n), np.full(n, 25.0),
                          np.zeros(n), CAP)
    anchor = SoHSchedule(t_upd=[0.0, 180.0], q_ah=[CAP, 2.0 * CAP])
    est = CoulombCounter(ocv=_curve(), params=TRUE, capacity_ah=CAP,
                         cfg=FilterConfig(soc0=1.0))
    out = est.run(prof, anchor=anchor)
    # 前 180 步按 CAP 积分、后 180 步按 2·CAP 积分（2 A 放电）
    ah = 2.0 * 180.0 / 3600.0
    expected = 1.0 - ah / CAP - ah / (2.0 * CAP)
    assert out["soc_hat"][-1] == pytest.approx(expected, abs=1e-9)


def test_cc_generalized_growth_law_under_drift():
    """时变容量下的广义增长律：err(k) = Σ_{j<k} (1/q̂ − 1/q_true(j))·i_j·dt/3600。

    §23 的静态式 err_end = −(1/s−1)(1−SOC_end) 是它 q 恒定的特例。
    这里 q_true 线性老化 −5%，q̂ 停在起点的陈旧值（A2 刷新不及时的极端情形），
    库仑计数的逐点误差必须精确落在理论轨迹上。"""
    n = 2000
    rng = np.random.default_rng(3)
    i = np.clip(rng.normal(-1.0, 0.8, n), -3.0, 0.5)
    i[0] = 0.0
    q_true = np.linspace(CAP, 0.95 * CAP, n)          # 老化：容量 −5%
    out = simulate_tv(RCParams(*TRUE.as_array()), _curve(), i, DT, q_arr=q_true,
                      r0_arr=TRUE.r0, soc0=0.9)
    prof = DrivingProfile(np.arange(n) * DT, i, out["voltage"], np.full(n, 25.0),
                          out["soc"], CAP, meta={"profile": "synth"})
    stale = SoHSchedule(t_upd=[0.0], q_ah=[CAP])       # 从不刷新
    got = CoulombCounter(ocv=_curve(), params=TRUE, capacity_ah=CAP,
                         cfg=FilterConfig(soc0=0.9)).run(prof, anchor=stale)
    term = (1.0 / CAP - 1.0 / q_true) * i * DT / 3600.0
    theory = np.zeros(n)
    theory[1:] = np.cumsum(term[:-1])
    err = got["soc_hat"] - prof.soc_ref
    assert np.max(np.abs(err - theory)) < 1e-9


def test_anchor_r0_schedule_reaches_estimator():
    """R̂0 走锚点：与真值一致时跟踪不变，放大 2 倍时跟踪显著变差。"""
    prof = _synth()
    cfg = FilterConfig(soc0=0.9, sigma_soc0=0.02)
    kw = dict(ocv=_curve(), params=TRUE, capacity_ah=CAP, cfg=cfg)
    good = EKF(**kw).run(prof, anchor=SoHSchedule([0.0], [CAP], [TRUE.r0]))
    bad = EKF(**kw).run(prof, anchor=SoHSchedule([0.0], [CAP], [2.0 * TRUE.r0]))
    e_good = np.sqrt(np.mean((good["soc_hat"] - prof.soc_ref) ** 2))
    e_bad = np.sqrt(np.mean((bad["soc_hat"] - prof.soc_ref) ** 2))
    assert e_good < 0.01
    assert e_bad > 2.0 * e_good


def test_simulate_tv_static_matches_simulate():
    """q/r0 恒定时 simulate_tv 必须与 simulate 逐点一致（回归胶水）。"""
    n = 500
    rng = np.random.default_rng(11)
    i = rng.normal(0.0, 1.5, n)
    i[0] = 0.0
    base = simulate(RCParams(*TRUE.as_array()), _curve(), i, DT, soc0=0.8, capacity_ah=CAP)
    tv = simulate_tv(RCParams(*TRUE.as_array()), _curve(), i, DT,
                     q_arr=CAP, r0_arr=TRUE.r0, soc0=0.8)
    for key in ("soc", "u1", "u2", "voltage"):
        assert np.allclose(tv[key], base[key], atol=1e-12)


def test_ekfq_ignores_anchor_q_but_uses_r0():
    """EKFQ 的容量只来自状态 x[4]：锚点 Q̂ 被忽略（文档契约），R̂0 仍生效。"""
    prof = _synth()
    cfg = FilterConfig(soc0=0.7, sigma_soc0=0.2)
    kw = dict(ocv=_curve(), params=TRUE, capacity_ah=CAP, cfg=cfg)
    plain = EKFQ(**kw).run(prof)
    same = EKFQ(**kw).run(prof, anchor=SoHSchedule([0.0], [CAP * 1.10]))
    assert np.allclose(same["soc_hat"], plain["soc_hat"], atol=1e-12)
    shifted = EKFQ(**kw).run(prof, anchor=SoHSchedule([0.0], [CAP], [2.0 * TRUE.r0]))
    assert not np.allclose(shifted["soc_hat"], plain["soc_hat"])


# ---- 三阶 RC / r0_cur 进滤波器（v1.4） -----------------------------------

TRUE3 = RCParams(r0=0.030, r1=0.012, tau1=10.0, r2=0.040, tau2=400.0,
                 r3=0.030, tau3=800.0)


def test_filters_run_with_third_branch():
    """3RC 参数下 EKF/UKF/EKFQ 都能跑且收得住；2RC 滤波器跑 3RC 植物会留系统残差。"""
    n = 8000
    rng = np.random.default_rng(7)
    i = np.clip(rng.normal(-0.5, 1.8, n), -6.0, 3.0)
    i[0] = 0.0                      # 与滤波器 u_j=r_j·i0 的初值约定对齐
    sim = simulate(TRUE3, _curve(), i, DT, soc0=0.95, capacity_ah=CAP)
    prof = DrivingProfile(t_s=np.arange(n) * DT, current_a=i,
                          voltage_v=sim["voltage"], temp_c=np.full(n, 25.0),
                          soc_ref=sim["soc"], capacity_ah=CAP)
    cfg = FilterConfig(soc0=0.95, sigma_soc0=0.05)
    errs = {}
    for cls in (EKF, UKF, EKFQ):
        est = cls(ocv=_curve(), params=TRUE3, capacity_ah=CAP, cfg=cfg)
        assert est.nr == 3
        assert est.ndim == (6 if cls is EKFQ else 5)
        out = est.run(prof)
        e = float(np.sqrt(np.mean((out["soc_hat"] - prof.soc_ref) ** 2)))
        assert e < 0.02
        errs[cls.name] = e
    # 对照：2RC 参数缺一条慢支路（u3 摆幅约 ±30 mV），开环残差显著更大。
    # 滤波器 SOC 误差可能被 b 部分吸收，所以比较**电压预测残差**而不是 SOC RMSE。
    out2 = EKF(ocv=_curve(), params=TRUE, capacity_ah=CAP, cfg=cfg).run(prof)
    out3 = EKF(ocv=_curve(), params=TRUE3, capacity_ah=CAP, cfg=cfg).run(prof)
    vres2 = float(np.sqrt(np.mean((out2["volt_pred"] - prof.voltage_v) ** 2)))
    vres3 = float(np.sqrt(np.mean((out3["volt_pred"] - prof.voltage_v) ** 2)))
    assert vres3 < vres2
    assert abs(sim["u3"]).max() > 0.015     # 前提：第三支路真的起来了


def test_ekfq_third_branch_f_jacobian_shape():
    est = EKFQ(ocv=_curve(), params=TRUE3, capacity_ah=CAP, cfg=FilterConfig(soc0=0.5))
    assert est.ndim == 6 and est._iq == 5
    F = est._F(DT, i=-2.0, q=CAP)
    assert F.shape == (6, 6)
    assert F[0, 5] == pytest.approx(2.0 * DT / (3600.0 * CAP * CAP), abs=1e-12)
    assert F[3, 3] == pytest.approx(np.exp(-DT / TRUE3.tau3), abs=1e-12)
    # 量测雅可比：三条极化支路各 1，q 列 0
    H = est._H(est._init_state(0.5), 25.0)
    assert H.shape == (1, 6)
    assert list(H[0, 1:]) == [1.0, 1.0, 1.0, 1.0, 0.0]


def test_r0_cur_enters_observation():
    """r0_cur 在 _h 里必须是 i·|i| 项（符号：放电 i<0 时压降更深）。"""
    p = RCParams(r0=0.03, r1=0.0, tau1=1.0, r2=0.0, tau2=1.0, r0_cur=0.01)
    est = EKF(ocv=_curve(), params=p, capacity_ah=CAP, cfg=FilterConfig(soc0=0.5))
    x = est._init_state(0.5, i0=0.0)
    v_dis = est._h(x, i=-4.0, temp_c=25.0)
    v_chg = est._h(x, i=4.0, temp_c=25.0)
    v_zero = est._h(x, i=0.0, temp_c=25.0)
    assert v_dis == pytest.approx(v_zero + 0.03 * (-4.0) + 0.01 * (-4.0) * 4.0, abs=1e-9)
    assert v_chg == pytest.approx(v_zero + 0.03 * 4.0 + 0.01 * 4.0 * 4.0, abs=1e-9)


# ---- 逐样本锚点：鸭子类型契约（脚本 24 ThermalAnchor 依赖）----

class _PerStepAnchor:
    """最小鸭子类型：q_at(t) / r0_at(t, default) 返回逐步数组（插值）。"""

    def __init__(self, t_knots, r0_knots, q_ah):
        self.t_knots = np.asarray(t_knots, float)
        self.r0_knots = np.asarray(r0_knots, float)
        self.q_ah = float(q_ah)

    def q_at(self, t):
        return np.full_like(np.asarray(t, float), self.q_ah)

    def r0_at(self, t, default=None):
        return np.interp(np.asarray(t, float), self.t_knots, self.r0_knots)


def test_duck_typed_per_sample_anchor_contract():
    """run(anchor=) 只要求 q_at/r0_at 两个方法；逐样本插值锚点必须可用。"""
    prof = _synth(n=600, seed=3)
    kw = dict(ocv=_curve(), params=TRUE, capacity_ah=CAP,
              cfg=FilterConfig(soc0=0.9, sigma_soc0=0.05))
    t = prof.t_s
    a1 = _PerStepAnchor([t[0], t[-1]], [TRUE.r0, TRUE.r0], CAP)
    a2 = _PerStepAnchor([t[0], t[-1]], [2.0 * TRUE.r0, 2.0 * TRUE.r0], CAP)
    out1 = EKF(**kw).run(prof, anchor=a1)
    out2 = EKF(**kw).run(prof, anchor=a2)
    # r0 翻倍必须改变观测预测与 SOC 轨迹
    assert np.max(np.abs(out2["volt_pred"] - out1["volt_pred"])) > 1e-4
    assert np.max(np.abs(out2["soc_hat"] - out1["soc_hat"])) > 1e-6


def test_per_sample_r0_anchor_flows_into_volt_pred():
    """逐步 r0(t) 必须逐点进入 volt_pred：v = ocv + u + r0(t)·i + b。"""
    prof = _synth(n=400, seed=5)
    t = prof.t_s
    r0_of_t = TRUE.r0 * (1.0 + 0.5 * (t - t[0]) / max(t[-1] - t[0], 1.0))  # 线性升温型
    anchor = _PerStepAnchor(t, r0_of_t, CAP)
    est = EKF(ocv=_curve(), params=TRUE, capacity_ah=CAP,
              cfg=FilterConfig(soc0=0.9, sigma_soc0=0.05))
    out = est.run(prof, anchor=anchor)
    # 无偏置状态时：volt_pred − (OCV+u 项) 应含 r0(t)·i；用翻倍锚点差分验证逐点性
    est2 = EKF(ocv=_curve(), params=TRUE, capacity_ah=CAP,
               cfg=FilterConfig(soc0=0.9, sigma_soc0=0.05))
    out2 = est2.run(prof, anchor=_PerStepAnchor(t, 2.0 * r0_of_t, CAP))
    dv = out2["volt_pred"] - out["volt_pred"]
    i = prof.current_a
    # 状态不同导致 u 略不同，但主项是 Δr0·i = r0(t)·i：相关性必须强
    denom = np.sqrt(np.mean((r0_of_t * i) ** 2))
    assert np.sqrt(np.mean(dv ** 2)) > 0.3 * denom
    assert np.corrcoef(dv, r0_of_t * i)[0, 1] > 0.9


def test_ekfq_ignores_duck_anchor_q_but_uses_r0():
    """EKFQ 不接受锚点容量覆盖（q 是状态），但逐步 r0 必须生效。"""
    prof = _synth(n=600, seed=7)
    t = prof.t_s
    kw = dict(ocv=_curve(), params=TRUE, capacity_ah=CAP,
              cfg=FilterConfig(soc0=0.9, sigma_soc0=0.05))
    base = EKFQ(**kw).run(prof)
    wrong_q = EKFQ(**kw).run(prof, anchor=_PerStepAnchor([t[0], t[-1]],
                                                         [TRUE.r0, TRUE.r0], 1.7 * CAP))
    assert np.allclose(base["soc_hat"], wrong_q["soc_hat"], atol=1e-12)
    shifted = EKFQ(**kw).run(prof, anchor=_PerStepAnchor([t[0], t[-1]],
                                                         [2.0 * TRUE.r0, 2.0 * TRUE.r0], CAP))
    assert np.max(np.abs(shifted["soc_hat"] - base["soc_hat"])) > 1e-6
