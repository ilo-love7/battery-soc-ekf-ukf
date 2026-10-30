"""SOC 误差指标的测试。"""
import numpy as np
import pytest

from bmsoc import metrics as met


def test_convergence_time_requires_staying_inside():
    t = np.arange(0.0, 100.0, 1.0)
    e = np.full(t.size, 0.10)
    e[40:] = 0.001                       # 40 s 后进入带内并保持
    assert met.convergence_time(t, e, tol=0.02) == pytest.approx(40.0)


def test_convergence_time_nan_when_tail_outside():
    t = np.arange(0.0, 100.0, 1.0)
    e = np.full(t.size, 0.001)
    e[-1] = 0.5                          # 末尾又出去了
    assert np.isnan(met.convergence_time(t, e, tol=0.02))


def test_convergence_time_zero_when_always_inside():
    t = np.arange(0.0, 10.0, 1.0)
    assert met.convergence_time(t, np.zeros(10), tol=0.02) == 0.0


def test_soc_metrics_basic_values():
    t = np.arange(0.0, 100.0, 1.0)
    ref = np.linspace(1.0, 0.5, t.size)
    hat = ref + 0.01
    m = met.soc_metrics(hat, ref, t)
    assert m["rmse"] == pytest.approx(0.01, abs=1e-9)
    assert m["bias"] == pytest.approx(0.01, abs=1e-9)
    assert m["max_abs"] == pytest.approx(0.01, abs=1e-9)
    assert m["frac_out_2pp"] == 0.0
    assert m["frac_out_5pp"] == 0.0
    assert m["n"] == t.size


def test_soc_metrics_tail_uses_last_30pct():
    t = np.arange(0.0, 100.0, 1.0)
    ref = np.zeros(t.size)
    hat = np.zeros(t.size)
    hat[:70] = 0.20                      # 前 70% 大误差，末 30% 准确
    m = met.soc_metrics(hat, ref, t)
    assert m["tail_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert m["rmse"] > 0.15
    assert m["tail_rmse"] < m["rmse"]


def test_soc_metrics_warmup_excludes_head():
    t = np.arange(0.0, 100.0, 1.0)
    ref = np.zeros(t.size)
    hat = np.zeros(t.size)
    hat[:20] = 0.5
    m0 = met.soc_metrics(hat, ref, t)
    m1 = met.soc_metrics(hat, ref, t, warmup_s=25.0)
    assert m0["rmse"] > m1["rmse"]
    assert m1["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert m1["n"] == 75


def test_soc_metrics_rejects_length_mismatch():
    with pytest.raises(ValueError):
        met.soc_metrics(np.zeros(5), np.zeros(4), np.arange(5.0))


def test_summarise_runs_handles_soc_prefix():
    rows = [
        {"estimator": "ekf", "soc_rmse": 0.02, "soc_max_abs": 0.05, "soc_converge_s": 10.0,
         "soc_tail_rmse": 0.01, "soc_converge5_s": 5.0},
        {"estimator": "ekf", "soc_rmse": 0.04, "soc_max_abs": 0.09, "soc_converge_s": 30.0,
         "soc_tail_rmse": 0.03, "soc_converge5_s": 15.0},
        {"estimator": "cc", "soc_rmse": 0.30, "soc_max_abs": 0.40, "soc_converge_s": float("nan"),
         "soc_tail_rmse": 0.31, "soc_converge5_s": float("nan")},
    ]
    s = met.summarise_runs(rows)
    assert s["ekf"]["n_runs"] == 2
    assert s["ekf"]["rmse_median"] == pytest.approx(0.03)
    assert s["ekf"]["tail_rmse_median"] == pytest.approx(0.02)
    # 未收敛（NaN）不能被当成 0
    assert np.isnan(s["cc"]["conv_median_s"])
    assert s["cc"]["conv5_frac_60s"] == 0.0


def test_summarise_runs_without_prefix():
    rows = [{"estimator": "ekf", "rmse": 0.01, "max_abs": 0.02, "converge_s": 1.0}]
    s = met.summarise_runs(rows)
    assert s["ekf"]["rmse_median"] == pytest.approx(0.01)
