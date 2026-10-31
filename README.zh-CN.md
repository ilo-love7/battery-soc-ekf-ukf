# 你的 SOC 到底错多少，为什么？

**电池 SOC 的 EKF / UKF / AEKF 可复用估计器包，在 24 段 −20 °C ~ 25 °C 的动态工况上做
留一工况标定与基准评测。**

[English](README.md)

```
                          末段 RMSE / 全程 RMSE（单位：百分点 pp）
  场景（初值误差，电流偏置）          库仑计数    EKF     UKF    AEKF
  known      (  0 pp,    0 mA)     0.01/0.01  2.80/2.62    –      –
  bias20ma   (  0 pp,  +20 mA)     1.33/0.90  1.55/1.93 3.39/3.31 1.32/1.63
  init30     (−30 pp,    0 mA)    25.37/28.68 3.89/3.31 4.18/3.81 2.95/3.11
  hard       (−30 pp,  +20 mA)    24.65/27.86 2.23/2.88 3.14/3.21 1.79/1.72
```

这个基准给出四条结论，都是「部署之前就应该知道」的那一类：

1. **地板是模型误差，不是滤波器。** 初值完全正确、电流完全准确时，EKF 仍有 2.62 pp，
   而纯库仑计数是 0.01 pp——因为库仑计数根本不看模型。滤波器的作用不是「变得更准」，
   而是**别让模型误差变成 SOC 误差**。
2. **滤波器唯一的价值来自「初值未知」。** 初值偏 −30 pp 时 EKF 比库仑计数好 8.7 倍；
   但电流偏置只有 20 mA 时，纯积分 0.90 pp **反而优于** EKF 的 1.93 pp。
   分流器够准的话，纯积分很难被打败。
3. **AEKF 稳定最好**（全程 1.63–3.11 pp，尾段 1.32–2.95 pp），`hard` 场景
   92% 的工况 60 s 内收敛（EKF 50%、UKF 21%）。但它在「只有初值误差」的 `init30` 上
   只比 EKF 好 0.2 pp——**自适应量测噪声治偏置，不治初值**。
4. **UKF 没有系统性优势**，三个能比的场景里每一个都略差于 EKF，而且踩过两个真 bug
   （见下文《UKF 为什么没有优势》）。修好之后仍略差，这才是可以写进报告的程度。

---

## 特性

- **四个估计器一个接口**：`CoulombCounter / EKF / UKF / AEKF`，统一
  `run(profile) -> dict[str, ndarray]`。预测/量测方程完全相同，比较才有意义。
- **第 4 个状态 = 电压偏置**：持续性的模型偏差（这批数据上 20–50 mV）在线估计，
  而不是被算成 SOC 误差。仅这一条就把 `known` 地板从 3.87 pp 降到 2.15 pp。
- **温度感知的 OCV**：`OCV(SOC, T)`，逐样本用实测电芯温度——这批测试是**边降温边跑**的，
  30 K 的温漂用单温度表会差几十毫伏。
- **SOC ∈ [0,1] 之外按端点斜率线性外推**：UKF 必须这样，EKF 无所谓。
- **留一工况标定** + **四个压力场景**，而不是一个好看的单一数字。
- **78 项单元测试全部用合成数据**，CI 里不需要数据集（Python 3.10 + 3.12）。

## 结果

### 标定质量（留一工况，24 折）

| 温度 | 被留出工况上的开环电压 RMSE | 中位数 |
|---|---|---|
| 25 °C | 17.3 – 48.7 mV | 27.5 mV |
| 10 °C | 29.1 – 74.1 mV | 40.7 mV |
| 0 °C | 28.5 – 81.8 mV | 40.2 mV |
| −10 °C | 39.8 – 111.6 mV | 59.1 mV |
| −20 °C | 77.9 – 196.8 mV | 98.6 mV |

中位数 **52.6 mV**，P90 **99.8 mV**。每折拟合 6 个电气参数；OCV 表**从不**参与重拟合
（只来自 HPPC 静置），所以这是真正的工况外测试。五个温度的最差折全是 US06
（电流最激烈的工况）。

### SOC 精度

完整表（含收敛占比与 P90）见 `reports/基准评测报告-A4.md` §7。
最难场景（`hard`：−30 pp 初值 + 20 mA 偏置）的分温度中位数：

| | −20 °C | −10 °C | 0 °C | 10 °C | 25 °C |
|---|---|---|---|---|---|
| 库仑计数 | 29.07 | 28.35 | 27.48 | 27.76 | 27.53 |
| EKF | 1.76 | 3.23 | 2.35 | 4.06 | 2.48 |
| UKF | 5.16 | 3.34 | 3.23 | 3.19 | 2.68 |
| AEKF | 1.71 | **1.31** | **1.21** | **2.91** | 2.61 |

温度的影响**不是单调的**：10 °C 最差（EKF 4.06 pp），因为 10 °C 的文件实测温度从
23.7 °C 一路降到 12 °C、跨度最大，而折参数是单温度拟合的；且该温度下 5 条工况
**全部**落在 3.4–4.3 pp，是系统性偏差而不是离群点。**精度由「模型误差 ÷ OCV 斜率」决定，
但这两项在低温是同向放大的，所以低温不一定更容易。**

图在 `docs/figures/`：OCV 表与可观测性、留一残差、SOC 轨迹、误差分布、偏置状态收敛、
敏感性、误差 CDF。

### 跨化学体系：LFP（CALCE A123）

同一套估计器、同一套超参数、同一套四场景，**一行不改**地跑第二个数据集 ——
24 段工况、8 个温度档（−10…50 °C），只有数据层换掉。全文见
`reports/基准评测报告-A4.md` §10。

| 场景 | 库仑计数 | EKF | UKF | AEKF |
|---|---|---|---|---|
| `known`（0, 0 A） | 0.01 | **0.24** | – | – |
| `bias20ma` | 2.16 | 1.96 | 1.87 | 1.95 |
| `init30` | 28.13 | 0.99 | **0.56** | 1.10 |
| `hard` | 26.90 | 1.35 | 1.98 | **1.29** |

LFP **并不更难**（0.2–2.0 pp，NCA 是 2.6–3.3 pp），而且原因可测：

- 这批工况每段都从 100 % SOC 起步，恰好是 LFP 曲线最陡的地方（30 pp ↔ 285 mV）；
- 平台区的 SOC 增益极小 —— `K_soc` 0.0002，NCA 是 0.0046（差 23 倍），
  所以模型误差和 OCV 表误差都推不动估计值。

真正咬人的是**容量（增益）误差**，两个体系都一样：5 % ⇒ 2–4.5 pp、10 % ⇒ 5–7.4 pp，
且没有任何滤波器状态能吸收它。相比之下 ±40 mV 的 OCV 表误差（常值或随 SOC 变形）
只值 ≤0.6 pp —— 因为偏置状态把它吸掉了。

### 两个压力测试（v1.2）：上面这张表有前提

**中段上电。** 上面的基准每段都从满电（100 % SOC）起步 —— 恰好是 LFP 最陡的区段。
把每段从真值 SOC 0.7 / 0.5 / 0.3 处切开再跑（`slice_from_soc`），结论完全变了：
LFP 的 `init30` 从 0.56–1.10 pp 恶化到 **12–36 pp**（NCA：3.3 → 10–25 pp）。
即使初值**正确**（`known`）误差也涨 2–3 倍 —— 这是平台上电的纯可观测性地板。
更麻烦的是不对称：估计**偏高** 30 pp 比偏低难救约 2 倍（LFP 起点 0.5：30.8 vs 14.0 pp），
且这批运行几乎永远进不了 5 pp 带（占比 0–5 %）。结论：「LFP 并不更难」是
**满电起步**的性质，不是化学体系的性质。真实 BMS 更多是在平台上醒来，不是在满电。

**容量在线辨识。** 容量误差的增长律精确成立：`err_end = −(1/s−1)(1−SOC_end)` ——
24 段 × 5 个容量比逐点验证（中位偏差 0.05 pp）。那么把容量建成滤波器第 5 个状态
（`EKFQ`）应该能纠回来。合成数据上确实能（10 % 误差收回 ±2 %）。**真实数据上
反而更差**：12 pp vs 2.6 pp —— 38–53 mV 的模型残差与容量漂移在电压上不可区分，
全被吸进 `q`（真实容量正确时 q 也漂 −3 到 −6 %）。超参扫描证实是结构性的。
结论：容量必须来自**外部锚点**（SOH 估计，A2），不能从单段放电里辨识。

## 安装

```bash
git clone <this-repo> && cd SOC在线估计-A4
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows
# numpy pandas pyarrow scipy openpyxl matplotlib pytest
```

数据集从环境变量 `PANASONIC_ROOT` 读（缺省值见 `src/bmsoc/io_panasonic.py`）。

## 快速开始

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from bmsoc.ecm import RCParams
from bmsoc.filters import EKF, AEKF, FilterConfig
from bmsoc.ocv import OCV2D, OCVTable, from_rest_points
from bmsoc.profiles import DrivingProfile

# 1) OCV 表（也可以直接读 results/metrics/ocv_panasonic.json）
soc = np.linspace(0, 1, 11)
ocv2d = OCV2D(OCVTable({25.0: from_rest_points(soc, 3.1 + 1.0 * soc, temp_c=25.0)}))

# 2) 工况：t_s / current_a / voltage_v / temp_c / soc_ref / capacity_ah
prof = DrivingProfile(t_s=np.arange(n) * 1.0, current_a=i, voltage_v=v,
                      temp_c=np.full(n, 25.0), soc_ref=soc_ref, capacity_ah=2.77)

# 3) 估计
est = EKF(ocv=ocv2d, params=RCParams(r0=0.030, r1=0.008, tau1=12.0,
                                     r2=0.060, tau2=2000.0, dv=0.0),
          capacity_ah=2.77, cfg=FilterConfig(soc0=0.70, sigma_soc0=0.20))
out = est.run(prof, current_bias_a=0.02)
out["soc_hat"], out["bias_hat"], out["sigma_soc"]
```

## 用你自己的数据

只需要一个 `DrivingProfile`。`bmsoc.profiles.build_profile(t_s, current_a, voltage_v, temp_c,
capacity_ah, soc0=1.0, dt=1.0)` 会把任意不规则日志重采样到固定步长，并用库仑计数构造真值 SOC。

**符号约定：充电为正、放电为负。** 如果你的记录仪相反，要把电流*和* OCV 曲线方向一起翻，
否则滤波器会「稳健地」收敛到错误答案。

## 方法要点

```
HPPC 脉冲 ──► OCV(SOC,T) ─┐
                          ├──► 二阶 RC 参数 ──► 留一标定 ──► 滤波器 ──► SOC
动态工况   ────────────────┘  （ECM-RUL 项目的方法） （本项目）  （本项目）
```

状态 `x = [soc, u1, u2, b]`：

```
soc[k+1] = soc[k] + η·i[k]·Δt / (3600·Q)
u_j[k+1] = e_j·u_j[k] + R_j(1−e_j)·i[k],   e_j = exp(−Δt/τ_j)
V[k]     = OCV(soc[k], T[k]) + u1[k] + u2[k] + R0·i[k] + dv + b[k]
```

OCV 来自 HPPC 静置末电压，PAVA 保序单调化，再做一次**残差目标 1.5 mV 的平滑**。
平滑不是为了好看：HPPC 每个 SOC 档有 5 个脉冲、SOC 只差 0.005，静置末电压只差 0.6–8 mV
——这是重复测量而不是斜率。−20 °C 下 35 个点里 25 对相邻点电压**完全相同**
（记录分辨率 0.1 mV），PAVA 之后整段成了阶梯，PCHIP 穿过阶梯会让 `dOCV/dSOC`
在 **0.00–2.91 V/unit** 之间乱跳（阶梯间斜率为 0，意味着卡尔曼增益为 0，滤波器停更）；
平滑后是 **0.43–2.95**，中位/最小从 10⁵ 收到 2.1。

RC 参数用**变量投影**辨识（τ 走格点、`[R0,R1,R2]` 用 NNLS），并且把**同一 SOC 档的
5 个倍率脉冲联合**求解——单个 10 s 脉冲分不开两个时间常数（实测 `τ2` 顶到 5000 s 上界、
`τ1` 在 5–67 s 乱跳）。

## 这批数据里的三个坑

**1. 重采样把 55% 的电压样本填错了。**
低温文件开头有 2 h 温箱静置、采样间隔 60 s；重采样到 1 s 后每 60 个区间里 59 个是空的，
而填充逻辑用的是**全部有效电压的均值**而不是前值。4.17 V 的静置电压被灌成 3.47 V，
12850 个样本里 7080 个错，EKF 直接发散到 86.9 pp。改成零阶保持后同一工况 1.77 pp，
`known` 地板从 7.05 降到 2.62 pp。回归测试：
`tests/test_profiles.py::test_resample_holds_value_across_sparse_gaps`。

**2. 标称温度不是电芯温度。** 10 °C 组实测 23.7 → 12 °C，−20 °C 组 16.1 → −16.3 °C。
所以 OCV 必须做成 `OCV(SOC, T)` 并按实测温度逐样本查表。

**3. 容量基准必须同源。** OCV 的 SOC 轴与真值 SOC 必须用同一个 `Q_ref`
（HPPC 一次完整放电：25/10/0/−10/−20 °C → 2.773 / 2.622 / 2.476 / 2.330 / 2.182 Ah）。
两者不一致会凭空造出 133 mV 的「模型误差」。

另外：每段工况开头的静置只保留 300 s。修复坑 1 之后静置段占样本数 55%，
留着会把 RMSE 稀释成「静置有多准」。

## UKF 为什么没有优势

两个真 bug，自己实现 UKF 的人值得看一眼。

**（a）`wm[0] = 0` 但 `wc[0] = β = 2`。**
`alpha=1, kappa=0, n=4` 时 `λ = 0`，中心 sigma 点在均值里权重为 0、在协方差里权重为 2。
线性量测下这一项自动抵消（`dY[0] = 0`，单元测试 `test_ukf_equals_ekf_for_linear_measurement`
用默认 `β=2` 也通过）；非线性量测下 `dY[0]` **就是**无迹变换的均值偏差，
`wc[0]·dY[0]²` 会把 `S` 灌大：

| SOC | σ=0.05 | σ=0.20 | σ=0.30 |
|---|---|---|---|
| 0.50 | +1.4 mV | +9.4 mV | +10.8 mV |
| 0.90 | −1.1 mV | +13.5 mV | **+26.5 mV** |

σ=0.30 时 `2·0.027² = 1.5e-3`，而 `R = 1e-4`：`S` 被抬约 15 倍，增益被压掉一个量级。
这足以让 UKF 变钝，但**不足以**造成 60 pp 的错误。它不是本次的主因。

**（b）OCV 表在 SOC ∈ [0,1] 之外被夹紧（主因）。**
μ=1.0、σ=0.05 时有一半 sigma 点越过 1，`OCV` 被夹成常数，于是
`E[OCV(soc)] < OCV(1.0)`，新息恒为正，SOC 被一路推到 `1.05` 限幅；到 SOC≈0 时对称反向，
被推下 `−0.05`。撞在轨上之后夹紧区间内 `dOCV/dSOC = 0`，雅可比消失，**从「推错」变成「推不动」**
——一条离开真值的棘轮。EKF 只在 μ 处求导，完全看不见这个夹紧。
改成**按端点斜率线性外推**后（同一 OCV 表、同一折参数、同一噪声种子）：

| 工况 | 场景 | 夹紧（修复前） | 外推（修复后） | EKF |
|---|---|---|---|---|
| −20 °C NN | known | 58.97 | **2.19** | 3.52 |
| −20 °C NN | init30 | 62.21 | **2.56** | 2.58 |
| −20 °C LA92 | bias20ma | 40.18 | **6.94** | 4.18 |
| 0 °C NN | known | 17.28 | **6.01** | 2.82 |

复现：`python scripts/07_ukf_clip_ab.py`。

> UKF 的卖点是「不用求导」，代价是对量测函数的**全局形状**敏感。OCV 表这种
> 「局部光滑、边界硬夹紧」的函数上，EKF 反而更稳。要用 UKF，必须保证 `h` 在整个
> sigma 支撑集上光滑、无夹紧。

## 复现完整流程

```bash
export PANASONIC_ROOT=".../Panasonic_18650PF"
python scripts/00_recon_data.py --which pan      # 数据勘察
python scripts/02_identify_ocv_ecm.py            # HPPC → OCV + RC 参数
python scripts/03_fit_ecm_cycle.py --dt 2        # 留一标定，约 40 s
python scripts/04_benchmark_soc.py --dt 1        # 24 工况 × 4 场景，约 9 min
python scripts/05_sensitivity.py --dt 1          # 初值/偏置/噪声扫描，约 20 min
python scripts/06_figures.py                     # docs/figures/*.png

# LFP（CALCE A123）：需要 CALCE_ROOT 指向 A123_*.zip 所在目录
python scripts/10_calce_prepare.py               # C/20 OCV + 24 段留一标定，约 5 s
python scripts/11_calce_benchmark.py --dt 1      # 24 段 × 4 场景，约 7 min
python scripts/12_calibration_stress.py --dataset calce      # 标定容差，约 7 min
python scripts/12_calibration_stress.py --dataset panasonic  # 同上（对照），约 8 min
python scripts/13_calce_figures.py               # 三张跨体系图

# 压力测试（v1.2）：中段上电 + 容量误差预算
python scripts/14_mid_soc_stress.py --dataset calce       # 约 7 min
python scripts/14_mid_soc_stress.py --dataset panasonic   # 约 9 min
python scripts/15_capacity_q.py --dataset calce           # 约 5 min
python scripts/15_capacity_q.py --dataset panasonic       # 约 6 min
python scripts/16_round3_figures.py                       # 两张压力测试图

python -m pytest tests -q                        # 78 项，不需要数据集
```

## 局限

- **两个化学体系，但不是受控对比。** NCA（Panasonic NCR18650PF）与 LFP（CALCE A123，
  已老化到约 45 % SOH）在电芯、老化、工况长度、温度网格上**全都不同**，
  所以 §10 只主张「方法可迁移」，不主张「LFP 更容易」。而「可迁移」现在带一个
  明确前提：**从满电起步**（v1.2 §10.7：平台上电难 15–30 倍）。
- **容量不确定性主导误差预算，且在线辨识走不通。** 5–10 % 的容量误差值 3–7 pp
  （增长律已精确验证）；把容量建成滤波器状态反而吸收模型残差、结果更差（v1.2 §10.8）。
  与 SOH 估计（A2）的耦合是关键缺口 —— 而且不是可选项。
- **OCV 需要静置。** 表来自 HPPC 静置末电压；工况中电芯从未真正松弛到平衡，
  所以低温标定出的 `dv` 到 −138 mV（0/24 折触及 −150 mV 边界，但随温度单调）。
- **二阶 RC 在数小时尺度上不够。** `τ2` 在 NCA 的 24 折里 17 折、LFP 的 24 折里**全部**顶到 6000 s 上界，
  慢支路退化成漂移项。三阶或分数阶模型才能用掉这个自由度。
- **电流偏置是给定的，不是估计的。** 滤波器只能靠电压反馈抵消它，不能估计它
  （作为第 5 个状态与 SOC 在平台区共线，不可辨识）。
- **收敛数字略偏乐观**：保留了 300 s 静置窗口；要「第一拍就得准」的版本用 `--keep-rest 0`。
  而且每段都从满电起步，初值误差恰好落在陡坡上 —— 平台起步的情形见上面的压力测试
  （12–36 pp 而不是 0.6–1.1 pp）。

## 引用与许可

MIT。用到本代码请一并引用数据集：

> Kollmeyer, P., et al. *Panasonic 18650PF Li-ion Battery Data.* McMaster University / UW-Madison.
> https://doi.org/10.17632/wykht8y7tg.1

```bibtex
@misc{bmsoc2026,
  title  = {bmsoc: EKF/UKF/AEKF state-of-charge estimation with leave-one-profile-out benchmarking},
  year   = {2026},
  note   = {Reference implementation, 1.7--3.1 pp SOC RMSE at -30 pp initial error}
}
```
