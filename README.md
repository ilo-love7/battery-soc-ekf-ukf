# How Wrong Is Your SOC, and Why?

**A reusable EKF / UKF / AEKF package for battery state-of-charge, benchmarked on 24 driving
profiles from −20 °C to 25 °C with leave-one-profile-out calibration.**

[中文说明](README.zh-CN.md)

```
                                tail RMSE / full RMSE  (percentage points)
  scenario (init error, current bias)    CC        EKF       UKF      AEKF
  known      (  0 pp,    0 mA)        0.01/0.01  2.80/2.62    –        –
  bias20ma   (  0 pp,  +20 mA)        1.33/0.90  1.55/1.93 3.39/3.31 1.32/1.63
  init30     (−30 pp,    0 mA)       25.37/28.68 3.89/3.31 4.18/3.81 2.95/3.11
  hard       (−30 pp,  +20 mA)       24.65/27.86 2.23/2.88 3.14/3.21 1.79/1.72
```

Four things this benchmark says that are worth knowing before you deploy a filter:

1. **Model error is the floor, not the filter.** With a perfect initial SOC and a perfect
   current sensor, the EKF still sits at 2.62 pp, while pure coulomb counting gets 0.01 pp —
   because coulomb counting never looks at the model. The filter's job is not to be accurate;
   it is to *stop model error from becoming SOC error*.
2. **The only thing a filter buys you is a wrong initial SOC.** At −30 pp the EKF is 8.7×
   better than coulomb counting. At a 20 mA current bias, plain coulomb counting (0.90 pp)
   *beats* the EKF (1.93 pp). If your shunt is good, integration is hard to beat.
3. **AEKF wins consistently** (full-run 1.63–3.11 pp, tail 1.32–2.95 pp) and converges within
   60 s on 92% of the `hard` profiles (EKF 50%, UKF 21%). But on `init30` — initial error only,
   no bias — it beats the EKF by just 0.2 pp. **Adaptive measurement noise fixes bias, not
   initialisation.**
4. **UKF has no systematic advantage**: it is slightly worse than the EKF in all three
   comparable scenarios, *after* two real bugs were found and fixed (§ *Why the UKF is not
   better*). Only then is the comparison worth publishing.

---

## Features

- **Four estimators behind one interface** — `CoulombCounter`, `EKF`, `UKF`, `AEKF`, all
  `run(profile) -> dict of arrays`. Same predict/measurement model, so the comparison is honest.
- **4th state = voltage bias.** A persistent model offset (20–50 mV on this data) is estimated
  online instead of being absorbed into SOC. This alone took the `known` floor from 3.87 → 2.15 pp.
- **Temperature-aware OCV.** `OCV(SOC, T)` with per-sample cell temperature — these tests cool
  the cell by 30 K *while driving*, so a single-temperature table is off by tens of mV.
- **Linear extrapolation outside SOC ∈ [0,1]** — required for UKF, harmless for EKF. See below.
- **Leave-one-profile-out calibration** and **four stress scenarios** instead of one flattering number.
- **78 unit tests, all on synthetic data** — they run without the datasets in CI (Python 3.10 + 3.12).

## Results

### Calibration quality (leave-one-profile-out, 24 folds)

| temperature | open-loop voltage RMSE on the held-out profile | median |
|---|---|---|
| 25 °C | 17.3 – 48.7 mV | 27.5 mV |
| 10 °C | 29.1 – 74.1 mV | 40.7 mV |
| 0 °C | 28.5 – 81.8 mV | 40.2 mV |
| −10 °C | 39.8 – 111.6 mV | 59.1 mV |
| −20 °C | 77.9 – 196.8 mV | 98.6 mV |

Median **52.6 mV**, P90 **99.8 mV**. Six electrical parameters are fitted per fold; the OCV
table is *never* refitted (it comes from HPPC rests only), so this is a genuine out-of-profile test.
The worst fold at every temperature is US06 — the most aggressive current profile.

### SOC accuracy

Full table with convergence shares and P90 in `reports/基准评测报告-A4.md` §7.
Per-temperature medians for the hardest scenario (`hard`: −30 pp init + 20 mA bias):

| | −20 °C | −10 °C | 0 °C | 10 °C | 25 °C |
|---|---|---|---|---|---|
| CC | 29.07 | 28.35 | 27.48 | 27.76 | 27.53 |
| EKF | 1.76 | 3.23 | 2.35 | 4.06 | 2.48 |
| UKF | 5.16 | 3.34 | 3.23 | 3.19 | 2.68 |
| AEKF | 1.71 | **1.31** | **1.21** | **2.91** | 2.61 |

The temperature dependence is **not monotonic**: 10 °C is the worst (EKF 4.06 pp), because the
10 °C files cool from 23.7 °C down to 12 °C — the widest span — while the fold parameters are
fitted at a single temperature. And at 10 °C *all five* profiles land in 3.4–4.3 pp, so this is a
systematic offset, not an outlier. **Accuracy is governed by model error ÷ OCV slope — and both
terms grow together in the cold, so cold is not automatically easier.**

Figures: `docs/figures/` — OCV table & observability, LOO residuals, SOC tracking, error
distribution, bias-state convergence, sensitivity, error CDF.

### Cross-chemistry: LFP (CALCE A123)

The same estimators, the same hyper-parameters and the same four scenarios, run **unchanged**
on a second dataset — 24 segments, 8 temperatures (−10…50 °C), only the data layer swapped.
Full section in `reports/基准评测报告-A4.md` §10.

| scenario | cc | EKF | UKF | AEKF |
|---|---|---|---|---|
| `known` (0, 0 A) | 0.01 | **0.24** | – | – |
| `bias20ma` | 2.16 | 1.96 | 1.87 | 1.95 |
| `init30` | 28.13 | 0.99 | **0.56** | 1.10 |
| `hard` | 26.90 | 1.35 | 1.98 | **1.29** |

LFP is **not harder** here (0.2–2.0 pp vs 2.6–3.3 pp for NCA), and the reasons are measurable
rather than lucky:

- every segment starts at 100 % SOC, where the LFP curve is steep (30 pp ↔ 285 mV);
- the plateau's SOC gain is tiny — `K_soc` 0.0002 vs 0.0046 for NCA, 23× — so neither model
  error nor an OCV-table error moves the estimate.

What does bite, on both chemistries, is a **capacity** error: 5 % → 2–4.5 pp, 10 % → 5–7.4 pp,
and no filter state can absorb a *gain* error. By contrast a ±40 mV OCV-table error (constant or
SOC-shaped) costs ≤ 0.6 pp, because the bias state absorbs it.

### Two stress tests (v1.2): the table above has a precondition

**Mid-SOC power-up.** The benchmark above starts every segment at 100 % SOC — which happens to
be the steepest part of the LFP curve. Slicing each segment to start at SOC 0.7 / 0.5 / 0.3
(`slice_from_soc`) changes the picture completely: LFP `init30` degrades from 0.56–1.10 pp to
**12–36 pp** (NCA: 3.3 → 10–25 pp). Even with a *correct* initial SOC (`known`) the error grows
2–3× — a pure observability floor of plateau start-up. Worse, the error is asymmetric: an
estimate 30 pp too **high** is ~2× harder to recover than one too low (LFP 30.8 vs 14.0 pp at
start 0.5), and those runs almost never reach the 5 pp band (0–5 %). Conclusion: the
"LFP is not harder" result is a property of **starting full**, not of the chemistry.
Real BMS wakes up mid-plateau far more often than at 100 %.

**Online capacity estimation.** The capacity error growth law is exactly
`err_end = −(1/s−1)(1−SOC_end)` — verified point-for-point on 24 segments × 5 capacity ratios
(median deviation 0.05 pp). So the natural fix, adding capacity as a 5th filter state (`EKFQ`),
should work. On synthetic data it does (10 % error → recovered to ±2 %). **On real data it is
worse than doing nothing**: 12 pp vs 2.6 pp, because the 38–53 mV model residual is
voltage-indistinguishable from a slow capacity drift and gets absorbed into `q` (it drifts −3 to
−6 % even when the true capacity is correct). Hyper-parameter sweeps confirm this is structural.
Takeaway: capacity must come from an **external anchor** (SOH estimation, project A2), not from
a single discharge segment.

## Install

```bash
git clone <this-repo> && cd SOC在线估计-A4
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows
# numpy pandas pyarrow scipy openpyxl matplotlib pytest
```

Datasets are read from `PANASONIC_ROOT` (default: a local path, see `src/bmsoc/io_panasonic.py`).

## Quick start

```python
import sys; sys.path.insert(0, "src")
import numpy as np, pandas as pd
from bmsoc.ecm import RCParams
from bmsoc.filters import EKF, AEKF, FilterConfig
from bmsoc.ocv import OCV2D, OCVCurve, OCVTable, from_rest_points
from bmsoc.profiles import DrivingProfile

# 1) an OCV table (or load results/metrics/ocv_panasonic.json)
soc = np.linspace(0, 1, 11)
curve = from_rest_points(soc, 3.1 + 1.0 * soc, temp_c=25.0)
ocv2d = OCV2D(OCVTable({25.0: curve}))

# 2) a driving profile: t_s / current_a / voltage_v / temp_c / soc_ref / capacity_ah
prof = DrivingProfile(t_s=np.arange(n) * 1.0, current_a=i, voltage_v=v,
                      temp_c=np.full(n, 25.0), soc_ref=soc_ref, capacity_ah=2.77)

# 3) estimate
est = EKF(ocv=ocv2d, params=RCParams(r0=0.030, r1=0.008, tau1=12.0,
                                     r2=0.060, tau2=2000.0, dv=0.0),
          capacity_ah=2.77, cfg=FilterConfig(soc0=0.70, sigma_soc0=0.20))
out = est.run(prof, current_bias_a=0.02)
out["soc_hat"], out["bias_hat"], out["sigma_soc"]
```

## Bring your own data

You only need a `DrivingProfile`. `bmsoc.profiles.build_profile(t_s, current_a, voltage_v, temp_c,
capacity_ah, soc0=1.0, dt=1.0)` resamples any irregular log to a fixed grid and builds the
reference SOC by coulomb counting.

**Sign convention: charge positive, discharge negative.** If your logger is the other way round,
flip the current *and* the OCV curve's direction, or the filter will happily converge to garbage.

## How it works

```
HPPC pulses ──► OCV(SOC,T) ─┐
                            ├──► 2-RC parameters ──► LOO calibration ──► filters ──► SOC
driving cycle ──────────────┘  (ECM-RUL project)      (this repo)      (this repo)
```

State `x = [soc, u1, u2, b]`:

```
soc[k+1] = soc[k] + η·i[k]·Δt / (3600·Q)
u_j[k+1] = e_j·u_j[k] + R_j(1−e_j)·i[k],   e_j = exp(−Δt/τ_j)
V[k]     = OCV(soc[k], T[k]) + u1[k] + u2[k] + R0·i[k] + dv + b[k]
```

OCV comes from HPPC rest-end voltages, PAVA-monotonised and then **smoothed with a 1.5 mV
residual target**. Smoothing is not cosmetic: each HPPC SOC level contributes 5 pulses at
almost the same SOC (0.005 apart) whose rest-end voltages differ by only 0.6–8 mV — replicates,
not slope. At −20 °C, 25 of 35 adjacent pairs are *identical* (0.1 mV logger resolution), so PAVA
turns the curve into a staircase and PCHIP through it gives `dOCV/dSOC` oscillating between
**0.00 and 2.91 V/unit** (flat between steps means zero Kalman gain — the filter simply stops
updating). After smoothing: **0.43–2.95**, median/min ratio down from 10⁵ to 2.1.

RC parameters are identified with **variable projection** (τ on a grid,
`[R0,R1,R2]` by NNLS) over **5 stacked C-rates per SOC level** — a single 10 s pulse cannot
separate two time constants (`τ2` pinned to the 5000 s bound, `τ1` jumping 5–67 s).

## Three traps in this dataset

**1. Resampling filled 55% of the voltage samples with the wrong value.**
Low-temperature files start with a 2 h chamber soak logged at 60 s. Resampling to 1 s leaves
59 of every 60 bins empty, and the fill logic used *the mean of all valid voltages* instead of
the previous value. The 4.17 V rest became 3.47 V for 7080 of 12850 samples, and the EKF
diverged to 86.9 pp. Fixed to zero-order hold; the same profile is now 1.77 pp, and the
`known` floor dropped 7.05 → 2.62 pp. Regression test:
`tests/test_profiles.py::test_resample_holds_value_across_sparse_gaps`.

**2. The nominal temperature is not the cell temperature.** The 10 °C group runs from 23.7 → 12 °C;
the −20 °C group from 16.1 → −16.3 °C. Hence `OCV(SOC, T)` queried with the measured cell
temperature at every step.

**3. The capacity basis must be shared.** The OCV SOC axis and the reference SOC must use the
same `Q_ref` (HPPC full discharge: 2.773 / 2.622 / 2.476 / 2.330 / 2.182 Ah at
25/10/0/−10/−20 °C). Mixing them produced a 133 mV "model error" that was pure bookkeeping.

Also: profiles are trimmed to keep at most 300 s of leading rest, otherwise a 55%-of-samples
flat soak dilutes every RMSE.

## Why the UKF is not better

Two real bugs, both worth knowing if you implement a UKF yourself.

**(a) `wm[0] = 0` but `wc[0] = β = 2`.** With `alpha=1, kappa=0, n=4` you get `λ = 0`, so the
centre sigma point has zero weight in the mean and weight 2 in the covariance. For a linear
measurement this cancels (`dY[0] = 0`, and the unit test passes even with the default `β=2`);
for a nonlinear one `dY[0]` *is* the unscented-transform mean bias, and `wc[0]·dY[0]²` inflates `S`:

| SOC | σ=0.05 | σ=0.20 | σ=0.30 |
|---|---|---|---|
| 0.50 | +1.4 mV | +9.4 mV | +10.8 mV |
| 0.90 | −1.1 mV | +13.5 mV | **+26.5 mV** |

At σ=0.30 that is `2·0.027² = 1.5e-3` against `R = 1e-4`: `S` inflated ~15×, gain pushed down
an order of magnitude. Enough to make the UKF sluggish, **not** enough to explain a 60 pp error.
Setting `β=0` helps `init30` (62 → 36 pp) but not `known`. It was not the main cause here.

**(b) The OCV table was clamped outside SOC ∈ [0,1] — the actual culprit.** Half the sigma
points at μ=1.0, σ=0.05 land beyond 1, where `OCV` was clamped, so `E[OCV(soc)] < OCV(1.0)`;
the innovation stayed positive and the estimate was pushed into the `1.05` clip. At SOC ≈ 0 the
effect reverses and pushes it into `−0.05`. Once on a rail, `dOCV/dSOC` inside the clamped region
is 0 — the Jacobian vanishes, so the state goes from *being pushed wrong* to *being unable to
move*: a ratchet away from the truth. The EKF only differentiates at μ and never noticed.
Fixed by **linear extrapolation with the edge slope** (same OCV table, same fold parameters,
same noise seed):

| profile | scenario | clamped (before) | extrapolated (after) | EKF |
|---|---|---|---|---|
| −20 °C NN | known | 58.97 | **2.19** | 3.52 |
| −20 °C NN | init30 | 62.21 | **2.56** | 2.58 |
| −20 °C LA92 | bias20ma | 40.18 | **6.94** | 4.18 |
| 0 °C NN | known | 17.28 | **6.01** | 2.82 |

Reproduce with `python scripts/07_ukf_clip_ab.py`.

> A UKF's selling point is "no Jacobian". Its cost is sensitivity to the *global* shape of the
> measurement function. On an OCV table — smooth locally, hard-clamped at the boundary — the
> EKF is simply more robust. If you use a UKF, keep `h` smooth over the whole sigma support.

## Reproduce

```bash
export PANASONIC_ROOT=".../Panasonic_18650PF"
python scripts/00_recon_data.py --which pan      # data survey
python scripts/02_identify_ocv_ecm.py            # OCV + RC from HPPC
python scripts/03_fit_ecm_cycle.py --dt 2        # LOO calibration, ~40 s
python scripts/04_benchmark_soc.py --dt 1        # 24 profiles × 4 scenarios, ~9 min
python scripts/05_sensitivity.py --dt 1          # init / bias / noise sweeps, ~20 min
python scripts/06_figures.py                     # docs/figures/*.png

# LFP (CALCE A123) — set CALCE_ROOT to the folder holding A123_*.zip
python scripts/10_calce_prepare.py               # C/20 OCV + 24 leave-one-out folds, ~5 s
python scripts/11_calce_benchmark.py --dt 1      # 24 segments × 4 scenarios, ~7 min
python scripts/12_calibration_stress.py --dataset calce      # calibration margin, ~7 min
python scripts/12_calibration_stress.py --dataset panasonic  # same probe on NCA, ~8 min
python scripts/13_calce_figures.py               # three cross-chemistry figures

# stress tests (v1.2) — mid-SOC power-up + capacity error budget
python scripts/14_mid_soc_stress.py --dataset calce       # ~7 min
python scripts/14_mid_soc_stress.py --dataset panasonic   # ~9 min
python scripts/15_capacity_q.py --dataset calce           # ~5 min
python scripts/15_capacity_q.py --dataset panasonic       # ~6 min
python scripts/16_round3_figures.py                       # two stress-test figures

python -m pytest tests -q                        # 78 tests, no dataset needed
```

## Limitations

- **Two chemistries, not a controlled comparison.** NCA (Panasonic NCR18650PF) and LFP
  (CALCE A123, aged to ~45 % SOH) differ in cell, age, profile length *and* temperature grid,
  so §10 only claims "the method transfers", never "LFP is easier". And "transfers" now comes
  with an explicit precondition: **starting at 100 % SOC** (v1.2 §10.7 shows mid-plateau
  power-up is 15–30× harder).
- **Capacity uncertainty dominates the error budget, and online identification does not work.**
  A 5–10 % capacity error costs 3–7 pp (growth law verified exactly); adding capacity as a filter
  state absorbs the model residual instead and degrades results (v1.2 §10.8). Coupling to SOH
  estimation (project A2) is the missing link — and it is not optional.
- **OCV needs rest.** The table comes from HPPC rest-end voltages; during a drive the cell never
  fully relaxes, so the cold folds need `dv` up to −138 mV (0 of 24 folds hit the −150 mV bound,
  but the sign and magnitude vary monotonically with temperature).
- **2-RC is not enough at multi-hour scale.** `τ2` hits the 6000 s bound in 17 of 24 NCA folds (24 of 24
  on LFP); the
  slow branch degenerates into a drift term. A 3rd-order or fractional model would use that DOF.
- **Current bias is given, not estimated.** The filter can only cancel it via voltage feedback
  (a 5th bias state is collinear with SOC on the plateau, hence unobservable).
- **Convergence numbers are slightly optimistic**: a 300 s rest window is kept. For a
  "correct on the first sample" variant use `--keep-rest 0`. And every segment starts at 100 %
  SOC, where the initial error lands on a steep OCV slope — see the mid-SOC stress test above
  for the plateau case (12–36 pp instead of 0.6–1.1).

## Citation & licence

MIT. If you use this code, please also cite the dataset:

> Kollmeyer, P., et al. *Panasonic 18650PF Li-ion Battery Data.* McMaster University / UW-Madison.
> https://doi.org/10.17632/wykht8y7tg.1

```bibtex
@misc{bmsoc2026,
  title  = {bmsoc: EKF/UKF/AEKF state-of-charge estimation with leave-one-profile-out benchmarking},
  year   = {2026},
  note   = {Reference implementation, 1.7--3.1 pp SOC RMSE at -30 pp initial error}
}
```
