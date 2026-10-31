"""CALCE（马里兰大学）A123 数据读取。

数据形态
--------
官方按测试包分发 zip，内含 Arbin 导出的 ``.xlsx``（一电芯一文件，另有 ``.csv`` 版本）。
数据表在 ``Channel_1-006`` 工作表，表头为
``Data_Point / Test_Time(s) / Date_Time / Step_Time(s) / Step_Index / Cycle_Index /
Current(A) / Voltage(V) / Charge_Capacity(Ah) / Discharge_Capacity(Ah) / ...``。
约定：**充电电流为正、放电为负**，容量列为各自单调累计。

本项目用两类包：
- ``A123_OCV<温度>``：小电流分段充放 + 长静置，给出 OCV(SOC)；
- ``A123_DST-US06-FUDS<温度>``：DST / US06 / FUDS 三种动态工况，给出 SOC 估计验证。

对应数据目录由环境变量 ``CALCE_ROOT`` 指定。
"""
from __future__ import annotations

import glob
import os
import re
import tempfile
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_ROOT = r"D:/项目/02_数据集/01_国际公开集/CALCE_UMD"

# Arbin 列名 → 内部名
_COLMAP = {
    "Data_Point": "data_point",
    "Test_Time(s)": "t_s",
    "Step_Time(s)": "step_t_s",
    "Step_Index": "step",
    "Cycle_Index": "cycle",
    "Current(A)": "current_a",
    "Voltage(V)": "voltage_v",
    "Charge_Capacity(Ah)": "q_chg_ah",
    "Discharge_Capacity(Ah)": "q_dis_ah",
    "Internal_Resistance(Ohm)": "r_ohm",
}

_KEEP = list(_COLMAP.keys())

_NAME_RE = re.compile(r"^A123_(?P<kind>OCV|DST-US06-FUDS)(?P<rest>.*)$", re.I)
# 温度字段的三种写法：``25``、``-10``、``N10``（或 ``-N10``）。
# 不能用「分隔符 + 数字」的写法：``A123_OCV-10`` 里的 "-" 是负号而不是分隔符，
# 那样会被读成 +10 °C，与 ``A123_OCV10`` 撞车（本仓库踩过）。
_TEMP_RE = re.compile(r"(?P<sign>-)?(?P<n>N)?-?(?P<num>\d+)", re.I)


def data_root(root: str | None = None) -> str:
    return root or os.environ.get("CALCE_ROOT", DEFAULT_ROOT)


def parse_package(path: str) -> Dict[str, object]:
    """解析包名 → 测试类型与温度。

    两个包族的命名规则**不一致**，必须分开处理（这是数据集的坑，不是笔误）：

    ==========================  ==================  ==================
    包名                        含义                解析
    ==========================  ==================  ==================
    ``A123_OCV-10-20120629``    −10 °C              ``-`` 是负号
    ``A123_OCV25-20120905``     +25 °C              无分隔符
    ``A123_DST-US06-FUDS-10``   +10 °C              ``-`` 是分隔符
    ``A123_DST-US06-FUDS-N10``  −10 °C              ``N`` 表示负
    ==========================  ==================  ==================

    若按「``-`` 一律当分隔符」解析，``A123_OCV-10`` 会变成 +10 °C，与
    ``A123_OCV10`` 撞车、静默丢掉一个温度档（本仓库踩过）。
    """
    base = os.path.basename(path)
    m = _NAME_RE.match(base)
    if not m:
        raise ValueError(f"无法解析 CALCE 包名: {base}")
    kind = "ocv" if m.group("kind").upper() == "OCV" else "dst"
    rest = m.group("rest").lstrip("_")
    if kind == "dst":
        rest = rest.lstrip("-")          # DST 的正温度写成 "-10"，此处 "-" 只是分隔
    tm = _TEMP_RE.match(rest)
    if tm is None:
        temp_c = float("nan")
    else:
        neg = bool(tm.group("sign")) or bool(tm.group("n"))
        temp_c = float(-int(tm.group("num")) if neg else int(tm.group("num")))
    return {"kind": kind, "temp_c": temp_c, "package": base}


def list_packages(root: str | None = None, kind: Optional[str] = None) -> List[Tuple[str, Dict[str, object]]]:
    """列出 A123 的 OCV / DST 包，可按 ``kind`` 过滤（"ocv" / "dst"）。"""
    root = data_root(root)
    out = []
    for path in sorted(glob.glob(os.path.join(root, "A123_*.zip"))):
        meta = parse_package(path)
        if kind and meta["kind"] != kind:
            continue
        out.append((path, meta))
    return out


def _pick_member(zip_path: str, pattern: str = ".xlsx") -> str:
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(pattern)]
        # 排除 Excel 的临时锁文件
        names = [n for n in names if not os.path.basename(n).startswith("~$")]
        if not names:
            raise FileNotFoundError(f"{zip_path} 内无 {pattern}")
        return sorted(names)[0]


def _extract_member(zip_path: str, member: str, cache_dir: Optional[str] = None) -> str:
    cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "calce_xlsx")
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, f"{os.path.basename(zip_path)[:-4]}__{os.path.basename(member)}")
    if os.path.exists(dst) and os.path.getsize(dst) == zipfile.ZipFile(zip_path).getinfo(member).file_size:
        return dst
    with zipfile.ZipFile(zip_path) as z, z.open(member) as f, open(dst, "wb") as out:
        out.write(f.read())
    return dst


def read_sheet(path: str, sheet: str = "Channel_1-006",
               limit: Optional[int] = None) -> pd.DataFrame:
    """读取一个 Arbin 导出的 ``.xlsx`` 数据表。

    表头之前有几行测试报告信息，用「包含 Data_Point 的行」定位表头。
    """
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    try:
        # 有的包里 'Channel_Chart' 是 Chartsheet（没有 iter_rows），必须跳过
        def _has_cells(title: str) -> bool:
            return hasattr(wb[title], "iter_rows")

        if sheet not in wb.sheetnames or not _has_cells(sheet):
            pref = [s for s in wb.sheetnames if s.startswith("Channel") and _has_cells(s)]
            if not pref:
                raise ValueError(f"{path} 无可读数据表（{wb.sheetnames}）")
            sheet = pref[0]
        ws = wb[sheet]
        rows = ws.iter_rows(values_only=True)
        header = None
        for row in rows:
            if row and any(isinstance(v, str) and v.strip() == "Data_Point" for v in row if v):
                header = [str(v).strip() if v is not None else "" for v in row]
                break
        if header is None:
            raise ValueError(f"{path} 未找到表头")
        idx = {name: header.index(name) for name in _KEEP if name in header}
        missing = [c for c in _KEEP if c not in idx]
        if "Test_Time(s)" in missing or "Current(A)" in missing or "Voltage(V)" in missing:
            raise ValueError(f"{path} 缺少必要列: {missing}")

        cols = {_COLMAP[c]: [] for c in idx}
        order = list(idx.items())
        n = 0
        for row in rows:
            if row is None:
                continue
            if limit is not None and n >= limit:
                break
            if len(row) <= max(idx.values()):
                continue
            for name, pos in order:
                cols[_COLMAP[name]].append(row[pos])
            n += 1
    finally:
        wb.close()

    df = pd.DataFrame(cols)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["t_s", "current_a", "voltage_v"]).reset_index(drop=True)
    return df


def load_package(zip_path: str, sheet: str = "Channel_1-006",
                 limit: Optional[int] = None) -> pd.DataFrame:
    """读取一个包（取第一个电芯），补齐温度/工况/来源等元信息。"""
    meta = parse_package(zip_path)
    member = _pick_member(zip_path)
    local = _extract_member(zip_path, member)
    df = read_sheet(local, sheet=sheet, limit=limit)
    df["dataset"] = "calce_a123"
    df["cell"] = os.path.basename(member).split("-")[0] or "A123"
    df["temp_c_set"] = float(meta["temp_c"])
    df["kind"] = meta["kind"]
    df["profile"] = "ocv" if meta["kind"] == "ocv" else "dst_us06_fuds"
    df["source"] = f"{os.path.basename(zip_path)}::{os.path.basename(member)}"
    if "q_dis_ah" in df:
        df["q_ah"] = -df["q_dis_ah"].fillna(0.0) + df["q_chg_ah"].fillna(0.0)
    return df


def capacity_ah(df: pd.DataFrame) -> float:
    """本包总放电容量（Ah），用库仑积分与容量列两种估计取较大者。"""
    from .profiles import cum_charge_ah

    q_int = abs(np.nanmin(cum_charge_ah(df["t_s"].to_numpy(float), df["current_a"].to_numpy(float))))
    q_col = float(np.nanmax(df["q_dis_ah"].to_numpy(float))) if "q_dis_ah" in df else 0.0
    return float(max(q_int, q_col))
