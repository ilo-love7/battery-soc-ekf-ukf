"""Panasonic 18650PF（Kollmeyer, UW-Madison）数据读取。

数据形态
--------
每个 ``.mat`` 是 MATLAB v7 结构体，顶层只有一个变量 ``meas``，字段为
``TimeStamp / Voltage / Current / Ah / Wh / Power / Battery_Temp_degC / Time / Chamber_Temp_degC``。
约定：**放电电流为负**，``Ah`` 为累计净电量（放电为负），时间单位秒，动态工况采样约 0.1 s。

文件名即工况，格式 ``<日期>_<时刻> <温度>degC_<工况>_Pan18650PF.mat``，
温度用 ``n`` 表示负号（``n20degC`` = -20 °C）。单文件可能包含多个工况串联
（如 ``10degC_US06_HWFET_UDDS_LA92_NN``），这类组合文件默认不进入基准。

对应数据目录由环境变量 ``PANASONIC_ROOT`` 指定，缺省为本地 02_数据集 路径。
"""
from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DEFAULT_ROOT = r"D:/项目/02_数据集/01_国际公开集/Panasonic_18650PF"

#: 单工况文件名里的工况代号（组合文件含多个代号，另行标记）
KNOWN_PROFILES = ("US06", "UDDS", "LA92", "NN", "HWFET")

_NAME_RE = re.compile(r"^(?P<date>[\d-]+)_(?P<time>[\d.]+)\s+(?P<temp>n?\d+)degC_(?P<rest>.+)_Pan18650PF\.mat$")


def data_root(root: str | None = None) -> str:
    return root or os.environ.get("PANASONIC_ROOT", DEFAULT_ROOT)


def parse_name(filename: str) -> Dict[str, object]:
    """从文件名解析温度与实际包含的工况。

    Returns
    -------
    dict
        ``temp_c``（float）、``profiles``（list[str]）、``combined``（bool）
    """
    m = _NAME_RE.match(os.path.basename(filename))
    if not m:
        raise ValueError(f"无法解析文件名: {filename}")
    raw_temp = m.group("temp")
    temp_c = float(-int(raw_temp[1:]) if raw_temp.startswith("n") else int(raw_temp))
    rest = m.group("rest")
    profiles = [p for p in KNOWN_PROFILES if p in rest]
    if not profiles:
        return {"temp_c": temp_c, "profiles": [], "combined": False}
    combined = len(profiles) > 1
    return {"temp_c": temp_c, "profiles": profiles, "combined": combined}


def list_files(root: str | None = None, combined: bool = False) -> List[Tuple[str, float, str]]:
    """列出可用的单工况动态测试文件。

    Returns
    -------
    list of (path, temp_c, profile)
    """
    root = data_root(root)
    out: List[Tuple[str, float, str]] = []
    for path in sorted(glob.glob(os.path.join(root, "*degC_*_Pan18650PF.mat"))):
        if "Pulse" in path or "pulse" in path or "Cycle" in path or "Charge" in path:
            continue
        meta = parse_name(path)
        if not meta["profiles"]:
            continue
        if meta["combined"] and not combined:
            continue
        out.append((path, float(meta["temp_c"]), str(meta["profiles"][0]).lower()))
    return out


def list_hppc(root: str | None = None) -> List[Tuple[str, float]]:
    """列出 HPPC 五脉冲文件（用于 OCV 与 RC 参数辨识）。"""
    root = data_root(root)
    out = []
    for path in sorted(glob.glob(os.path.join(root, "*degC_*[Pp]ulse_HPPC_Pan18650PF.mat"))):
        if "DisPulse" in path:
            continue
        out.append((path, float(parse_name(path)["temp_c"])))
    return out


def load_meas(path: str) -> Dict[str, np.ndarray]:
    """读取 ``meas`` 结构体为 float 数组字典（不保留 TimeStamp）。"""
    from scipy.io import loadmat

    raw = loadmat(path, squeeze_me=True, struct_as_record=False)["meas"]
    return {
        "t_s": np.asarray(raw.Time, dtype=float),
        "current_a": np.asarray(raw.Current, dtype=float),
        "voltage_v": np.asarray(raw.Voltage, dtype=float),
        "q_ah": np.asarray(raw.Ah, dtype=float),
        "temp_c": np.asarray(raw.Battery_Temp_degC, dtype=float),
        "chamber_c": np.asarray(getattr(raw, "Chamber_Temp_degC", np.full(len(raw.Time), np.nan)), dtype=float),
    }


def to_frame(path: str) -> pd.DataFrame:
    """单文件 → DataFrame（附带温度、工况、容量与真值 SOC）。"""
    meta = parse_name(path)
    d = load_meas(path)
    df = pd.DataFrame(d)
    df["dataset"] = "panasonic_18650pf"
    df["cell"] = "Pan18650PF"
    df["temp_c_set"] = float(meta["temp_c"])
    df["profile"] = meta["profiles"][0].lower() if meta["profiles"] else "other"
    df["combined"] = bool(meta["combined"])
    df["source"] = os.path.basename(path)
    return df


def capacity_ah(df: pd.DataFrame, tail_s: float = 60.0) -> float:
    """本次测试实际放出的容量（Ah）。

    取累计电量 ``q_ah`` 的最小绝对值；末段若有回充（割断电压后静置回升不算），
    用最后 ``tail_s`` 秒内最负值更稳。实测两者差异 <0.2%，故直接用全局最小值。
    """
    q = np.asarray(df["q_ah"], dtype=float)
    return float(abs(np.nanmin(q)))


def attach_soc_ref(df: pd.DataFrame, capacity: float | None = None) -> pd.DataFrame:
    """按「满充出发 → 库仑计数」构造真值 SOC。

    SOC_ref(t) = 1 + q(t)/Q_ref，q 为数据自带累计电量（放电为负），
    Q_ref 为该次测试的放电容量，故放电终止时 SOC_ref≈0。
    该定义只用测试自身的电量与容量，不需要模型参与。
    """
    df = df.copy()
    cap = float(capacity if capacity is not None else capacity_ah(df))
    df["capacity_ah"] = cap
    df["soc_ref"] = np.clip(1.0 + np.asarray(df["q_ah"], dtype=float) / cap, 0.0, 1.0)
    return df


def summarize(path: str) -> Dict[str, object]:
    """单文件速览（供勘察与单元测试使用）。"""
    df = to_frame(path)
    meta = parse_name(path)
    return {
        "source": os.path.basename(path),
        "temp_c": meta["temp_c"],
        "profile": str(meta["profiles"][0]).lower() if meta["profiles"] else None,
        "n": int(len(df)),
        "duration_min": float((df["t_s"].iloc[-1] - df["t_s"].iloc[0]) / 60.0),
        "dt_median_s": float(np.median(np.diff(df["t_s"]))),
        "capacity_ah": capacity_ah(df),
        "v_min": float(df["voltage_v"].min()),
        "v_max": float(df["voltage_v"].max()),
        "i_min": float(df["current_a"].min()),
        "i_max": float(df["current_a"].max()),
    }
