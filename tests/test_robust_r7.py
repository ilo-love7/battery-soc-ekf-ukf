"""第十一轮（脚本 34 + filters 鲁棒更新）合成台架测试 —— 不依赖任何数据集。

覆盖：三种鲁棒模式的影响函数行为、默认向后兼容、重尾注入下鲁棒 EKFQ 优于基线、
自洁植物上不回归、winsorize/高斯化对照代数。
"""
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(ROOT, "scripts", "34_robust_round7.py")
sys_path_src = os.path.join(ROOT, "src")
if sys_path_src not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path_src)


def _load_s34():
    spec = importlib.util.spec_from_file_location("s34_robust", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def s34():
    return _load_s34()


class _Curve:
    """合成 OCV（与 test_ekfq_r4 同型）。"""

    def __init__(self):
        self.socs = np.linspace(0.05, 1.0, 201)
        self.ocv_v = 3.2 + 0.8 * self.socs + 0.1 * self.socs ** 2

    def batch(self, soc, temp):
        return np.interp(np.asarray(soc, float), self.socs, self.ocv_v)

    def __call__(self, soc, temp=None):
        return np.interp(np.asarray(soc, float), self.socs, self.ocv_v)

    def docv(self, soc, temp=None):
        s = np.asarray(soc, float)
        return 0.8 + 0.2 * s


def _plant(n=4000, seed=11):
    """自洽植物：DOD~0.7，动态激励充分（q 可辨识）。"""
    from bmsoc.ecm import RCParams, open_loop_from_truth, simulate

    rng = np.random.default_rng(seed)
    t = np.arange(n) * 1.0
    i = -1.2 + 1.5 * np.sin(2 * np.pi * t / 240.0) + 0.8 * np.sin(2 * np.pi * t / 37.0)
    q_true = 2.5
    p_true = RCParams(r0=0.030, r1=0.012, tau1=12.0, r2=0.045, tau2=450.0, dv=0.0)
    curve = _Curve()
    sim = simulate(p_true, curve, i, 1.0, soc0=0.95, capacity_ah=q_true)
    soc = np.asarray(sim["soc"], float)
    assert soc.min() > 0.05
    v = open_loop_from_truth(
        np.array([0.03, 0.03, 0.03, 0.012, 12.0, 0.045, 450.0, 0.0]), soc, i,
        curve, 1.0, 25.0)
    return dict(t=t, i=i, soc=soc, v=v, q_true=q_true, p_true=p_true, curve=curve)


def _run_ekfq(plant, v_meas, cap_scale=1.05, robust="none", kappa=3.0):
    from bmsoc.filters import EKFQ, FilterConfig
    from bmsoc.profiles import DrivingProfile

    n = plant["t"].size
    pf = DrivingProfile(t_s=plant["t"], current_a=plant["i"], voltage_v=v_meas,
                        temp_c=np.full(n, 25.0), soc_ref=plant["soc"],
                        capacity_ah=plant["q_true"])
    cfg = FilterConfig(soc0=0.95, sigma_soc0=0.05,
                       innov_robust=robust, innov_kappa=kappa)
    out = EKFQ(ocv=plant["curve"], params=plant["p_true"],
               capacity_ah=plant["q_true"] * cap_scale, cfg=cfg).run(pf)
    qh = float(np.median(out["q_hat"][-n // 10:]))
    q_err = qh / plant["q_true"] - 1.0
    soc_rmse = float(np.sqrt(np.mean((out["soc_hat"] - plant["soc"]) ** 2)))
    return out, q_err, soc_rmse


# =====================================================================
# 影响函数行为
# =====================================================================

def test_filter_config_defaults_backward_compatible():
    from bmsoc.filters import FilterConfig
    c = FilterConfig()
    assert c.innov_robust == "none"
    assert c.innov_kappa == 3.0


def test_robust_none_passthrough(s34):
    from bmsoc.ecm import RCParams
    from bmsoc.filters import EKF, FilterConfig

    est = EKF(ocv=_Curve(), params=RCParams(0.03, 0.01, 10, 0.02, 200),
              capacity_ah=2.5, cfg=FilterConfig(innov_robust="none"))
    K = np.array([0.1, 0.01, 0.01, 0.0])
    H = np.array([[4.0, 1.0, 1.0, 1.0]])
    P = np.eye(4) * 1e-4
    K2, e2, R2, w = est._robust_correct(0.5, 1e-4, K, H, P, 1e-4)
    assert float(e2) == 0.5
    assert w == 1.0
    assert np.allclose(K2, K)


def test_robust_clip_bounds_innovation():
    from bmsoc.ecm import RCParams
    from bmsoc.filters import EKF, FilterConfig

    est = EKF(ocv=_Curve(), params=RCParams(0.03, 0.01, 10, 0.02, 200),
              capacity_ah=2.5, cfg=FilterConfig(innov_robust="clip", innov_kappa=3.0))
    K = np.array([0.1, 0.01, 0.01, 0.0])
    H = np.array([[4.0, 1.0, 1.0, 1.0]])
    P = np.eye(4) * 1e-4
    delta = 3.0 * np.sqrt(1e-4)
    K2, e2, R2, w = est._robust_correct(0.5, 1e-4, K, H, P, 1e-4)
    assert abs(float(e2)) == pytest.approx(delta)
    assert w == pytest.approx(delta / 0.5)
    assert np.allclose(K2, K)          # clip 不改增益
    # 小新息直通
    K3, e3, _, w3 = est._robust_correct(0.001, 1e-4, K, H, P, 1e-4)
    assert float(e3) == 0.001 and w3 == 1.0


def test_robust_huber_shrinks_gain_on_outlier():
    from bmsoc.ecm import RCParams
    from bmsoc.filters import EKF, FilterConfig

    est = EKF(ocv=_Curve(), params=RCParams(0.03, 0.01, 10, 0.02, 200),
              capacity_ah=2.5, cfg=FilterConfig(innov_robust="huber", innov_kappa=3.0))
    H = np.array([[4.0, 1.0, 1.0, 1.0]])
    P = np.eye(4) * 1e-4
    R = 1e-4
    s_nom = float((H @ P @ H.T)[0, 0]) + R
    K_nom = (P @ H.T).reshape(-1) / s_nom
    # 大新息：K_eff 应按 R/w 变小（相对于名义 K 的对应分量）
    K2, e2, R2, w = est._robust_correct(0.5, s_nom, K_nom, H, P, R)
    assert R2 > R
    assert w < 1.0
    assert float(e2) == 0.5            # huber 不截断 e
    s_eff = float((H @ P @ H.T)[0, 0]) + R2
    K_expect = (P @ H.T).reshape(-1) / s_eff
    assert np.allclose(K2, K_expect)
    # 影响有界：|Δx| ≤ |PH'|·δ/R（IRLS-R 膨胀的渐近界），且极大新息不再炸开
    delta = 3.0 * np.sqrt(s_nom)
    bound = float(abs((P @ H.T).reshape(-1)[0])) * delta / R
    for e_big in (5.0, 50.0, 500.0):
        K_big, e_u, _, _ = est._robust_correct(e_big, s_nom, K_nom, H, P, R)
        assert abs(float(K_big[0] * e_u)) <= bound + 1e-9
    # 对照：名义增益无界增长
    assert abs(float(K_nom[0] * 500.0)) > bound


def test_robust_gate_skips_outlier():
    from bmsoc.ecm import RCParams
    from bmsoc.filters import EKF, FilterConfig

    est = EKF(ocv=_Curve(), params=RCParams(0.03, 0.01, 10, 0.02, 200),
              capacity_ah=2.5, cfg=FilterConfig(innov_robust="gate", innov_kappa=3.0))
    K = np.array([0.1, 0.01, 0.01, 0.0])
    H = np.array([[4.0, 1.0, 1.0, 1.0]])
    P = np.eye(4) * 1e-4
    K2, e2, R2, w = est._robust_correct(0.5, 1e-4, K, H, P, 1e-4)
    assert float(e2) == 0.0 and w == 0.0
    assert np.allclose(K2, 0.0)
    K3, e3, _, w3 = est._robust_correct(0.001, 1e-4, K, H, P, 1e-4)
    assert float(e3) == 0.001 and w3 == 1.0


def test_robust_unknown_mode_raises():
    from bmsoc.ecm import RCParams
    from bmsoc.filters import EKF, FilterConfig

    est = EKF(ocv=_Curve(), params=RCParams(0.03, 0.01, 10, 0.02, 200),
              capacity_ah=2.5, cfg=FilterConfig(innov_robust="bogus"))
    with pytest.raises(ValueError):
        est._robust_correct(0.5, 1e-4, np.zeros(4), np.ones((1, 4)),
                            np.eye(4) * 1e-4, 1e-4)


def test_ekfq_diagnostics_arrays_present():
    from bmsoc.ecm import RCParams
    from bmsoc.filters import EKFQ, FilterConfig
    from bmsoc.profiles import DrivingProfile

    p = _plant(n=200, seed=3)
    n = p["t"].size
    pf = DrivingProfile(t_s=p["t"], current_a=p["i"], voltage_v=p["v"],
                        temp_c=np.full(n, 25.0), soc_ref=p["soc"],
                        capacity_ah=p["q_true"])
    out = EKFQ(ocv=p["curve"], params=p["p_true"], capacity_ah=p["q_true"] * 1.05,
               cfg=FilterConfig(soc0=0.95, sigma_soc0=0.05,
                                innov_robust="clip", innov_kappa=3.0)).run(pf)
    assert "innov_used" in out and "robust_w" in out
    assert out["innov_used"].shape == out["innovation"].shape
    assert np.all(out["robust_w"] >= 0.0) and np.all(out["robust_w"] <= 1.0)
    # 无离群时权重应多为 1
    assert np.mean(out["robust_w"] == 1.0) > 0.8


# =====================================================================
# 功能：重尾注入下的 q 恢复
# =====================================================================

def _heavy_tail_voltage(plant, sigma=0.020, seed=5):
    """Student-t(df=2) 重尾电压污染（第九轮实测 white kurt 7–30 的合成对应物）。

    这一档是探针找到的决定性对照：基线 EKFQ 的 q̂ 被灌偏 ~10%、SOC 30 pp，
    鲁棒模式全部收回 ±2% —— 直接演示「重尾灌 q、鲁棒对症」。
    """
    rng = np.random.default_rng(seed)
    n = plant["v"].size
    noise = rng.standard_t(2, size=n) * sigma
    return plant["v"] + noise


def test_robust_beats_baseline_on_heavy_tailed_injection():
    """核心功能：重尾污染下，鲁棒 EKFQ 显著压低 |q̂ 误差| 并回收 SOC。"""
    p = _plant(n=5000, seed=11)
    v_bad = _heavy_tail_voltage(p)
    _, q_base, soc_base = _run_ekfq(p, v_bad, robust="none")
    _, q_clip, soc_clip = _run_ekfq(p, v_bad, robust="clip", kappa=3.0)
    _, q_hub, soc_hub = _run_ekfq(p, v_bad, robust="huber", kappa=3.0)
    # 基线应被重尾灌出明显偏差
    assert abs(q_base) > 0.05, q_base
    # 鲁棒至少削掉一半误差
    assert abs(q_clip) < 0.5 * abs(q_base), (q_base, q_clip)
    assert abs(q_hub) < 0.5 * abs(q_base), (q_base, q_hub)
    # SOC 同步改善
    assert soc_clip < soc_base
    assert soc_hub < soc_base


def test_robust_clean_plant_no_regression():
    """自洁植物：鲁棒模式不应破坏 q 收敛（可辨识性正对照的鲁棒版）。"""
    p = _plant(n=5000, seed=11)
    for mode, kappa in (("none", 3.0), ("clip", 3.0), ("huber", 3.0), ("gate", 3.0)):
        _, q_err, _ = _run_ekfq(p, p["v"], robust=mode, kappa=kappa)
        assert abs(q_err) < 0.02, f"{mode} 在自洁植物上没收敛：{q_err}"


def test_gate_robust_beats_baseline_on_spikes():
    p = _plant(n=5000, seed=11)
    v_bad = _heavy_tail_voltage(p, sigma=0.020, seed=8)
    _, q_base, _ = _run_ekfq(p, v_bad, robust="none")
    _, q_gate, _ = _run_ekfq(p, v_bad, robust="gate", kappa=3.0)
    assert abs(q_base) > 0.02
    assert abs(q_gate) < abs(q_base)


# =====================================================================
# 脚本 34 纯函数
# =====================================================================

def test_winsor_clips_tails_only(s34):
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 10000)
    x[::1000] = 20.0
    w = s34._winsor(x, 0.05)
    assert np.max(np.abs(w)) < 20.0
    assert np.max(np.abs(w)) <= np.quantile(np.abs(x), 0.95) + 1e-9
    # 中位数附近不动
    med = np.median(x)
    assert abs(np.median(w) - med) < 0.2


def test_rescale_to_std(s34):
    x = np.array([1.0, -1.0, 2.0, -2.0])
    y = s34._rescale_to_std(x, 0.5)
    assert float(np.std(y)) == pytest.approx(0.5)
    # 恒零输入不炸
    z = s34._rescale_to_std(np.zeros(5), 0.5)
    assert np.allclose(z, 0.0)


def test_tail_stats_gaussian_reference(s34):
    rng = np.random.default_rng(4)
    e = rng.normal(0, 1, 20000)
    st = s34._tail_stats(e)
    assert abs(st["kurt"]) < 0.5
    assert 0.001 < st["frac_3s"] < 0.01
    # 高斯 3σ 尾能约占总能量 3%（理论 2.9%）
    assert 0.01 < st["tail_energy_share"] < 0.08


def test_tail_stats_heavy_tailed(s34):
    rng = np.random.default_rng(4)
    e = rng.standard_t(df=2, size=20000) * 0.5
    st = s34._tail_stats(e)
    assert st["kurt"] > 10.0
    assert st["tail_energy_share"] > 0.5


def test_build_injects_shapes_and_winsor(s34):
    rng = np.random.default_rng(6)
    n = 2000
    i = np.sin(2 * np.pi * np.arange(n) / 200.0)
    inn = 0.01 * i + rng.normal(0, 0.01, n)
    inn[::200] += 0.08          # 注入重尾尖峰
    inj = s34._build_injects(inn, i, 1.0)
    assert set(inj) == {"none", "full", "full_winsor5", "full_gauss"}
    for v in inj.values():
        assert v.shape == (n,)
    # full 去均值
    assert abs(float(np.mean(inj["full"]))) < 1e-9
    # winsor5 的尖峰应被压低
    assert np.max(np.abs(inj["full_winsor5"])) < np.max(np.abs(inj["full"]))
    # 高斯化 white 的白分量方差与原 white 相当（±30%）
    s28_path = os.path.join(ROOT, "scripts", "28_ekfq_round4.py")
    spec = importlib.util.spec_from_file_location("s28_for_test", s28_path)
    s28 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s28)
    comps = s28._structure_components(inn, i, 1.0)
    w_std = float(np.std(comps["white"]))
    gauss_white = inj["full_gauss"] - (comps["slow"] + comps["odd"] + comps["even"])
    assert abs(float(np.std(gauss_white)) / w_std - 1.0) < 0.3


def test_replay_specs_cover_registered_grid(s34):
    specs = s34._replay_specs()
    keys = {(a, b, c) for a, b, c in specs}
    # 扫描：none/full × 全变体 × 1.0
    for inj in ("none", "full"):
        for v in s34.FILTER_VARIANTS:
            assert (inj, v, 1.0) in keys
    # 机制对照
    for inj in ("full_winsor5", "full_gauss"):
        for v in ("base", "clip3", "huber3"):
            assert (inj, v, 1.0) in keys
    # 剂量
    for v in ("base", "clip3", "huber3"):
        for sc in (0.5, 1.5):
            assert ("full", v, sc) in keys
    # 判决线是只读常量（§50.2 登记）
    assert s34.H32_L1_DROP == 0.50
    assert s34.H32_L2_QOK == 0.30
    assert s34.H33_L1_WINSOR == 0.20
    assert s34.H33_L2_GAUSS == 0.40
    assert s34.H34_L1_QREL == 0.20
