"""数据格式勘察：Panasonic 18650PF (.mat) 与 CALCE A123 (.xlsx)。

只打印结构，不做任何写盘。用来确认字段名、采样步长、容量真值可算性。
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

PAN = r"D:/项目/02_数据集/01_国际公开集/Panasonic_18650PF"
CAL = r"D:/项目/02_数据集/01_国际公开集/CALCE_UMD"


def probe_mat(path: str) -> None:
    from scipy.io import loadmat

    print(f"\n=== {os.path.basename(path)} ({os.path.getsize(path)/1e6:.1f} MB) ===")
    try:
        m = loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as e:
        print("  loadmat 失败（可能是 v7.3/HDF5）:", e)
        return
    keys = [k for k in m if not k.startswith("__")]
    print("  top keys:", keys)
    for k in keys:
        v = m[k]
        if hasattr(v, "_fieldnames"):
            print(f"  {k}: struct fields = {list(v._fieldnames)}")
            for f in v._fieldnames:
                fv = getattr(v, f)
                arr = np.atleast_1d(fv)
                kind = "numeric" if np.issubdtype(arr.dtype, np.number) else arr.dtype
                print(f"    - {f}: type={kind} shape={arr.shape}"
                      + (f" first3={arr[:3]}" if arr.size >= 3 else f" vals={arr}"))
        else:
            arr = np.atleast_1d(v)
            print(f"  {k}: type={arr.dtype} shape={arr.shape}"
                  + (f" first3={arr[:3]}" if arr.size >= 3 else f" vals={arr}"))


def probe_xlsx(path: str, nrow: int = 6) -> None:
    from openpyxl import load_workbook

    print(f"\n=== {os.path.basename(path)} ({os.path.getsize(path)/1e6:.1f} MB) ===")
    wb = load_workbook(path, read_only=True, data_only=True)
    print("  sheets:", wb.sheetnames)
    ws = wb[wb.sheetnames[0]]
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i >= nrow:
            break
        print("   ", [None if v is None else (round(v, 4) if isinstance(v, float) else v) for v in row[:12]])
    wb.close()


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "pan":
        # 文件名带日期前缀与空格，例如 "03-20-17_01.43 25degC_US06_Pan18650PF.mat"
        for pat in ["* 25degC_US06_Pan18650PF.mat", "* 25degC_5Pulse_HPPC_Pan18650PF.mat",
                    "* 25degC_Cycle_1_Pan18650PF.mat"]:
            hits = sorted(glob.glob(os.path.join(PAN, pat)))
            if not hits:
                print("未命中:", pat)
                continue
            probe_mat(hits[0])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "calce":
        import tempfile
        import zipfile

        for zname, inner in [
            ("A123_DST-US06-FUDS-25.zip", "A1-007-DST-US06-FUDS-25-20120827.xlsx"),
            ("A123_OCV25-20120905.zip", "A1-007-OCV-25-20120905.xlsx"),
        ]:
            zp = os.path.join(CAL, zname)
            with zipfile.ZipFile(zp) as z:
                names = [n for n in z.namelist() if n.endswith(inner)]
                if not names:
                    print(zp, "内无", inner, "有:", z.namelist()[:5])
                    continue
                tmp = os.path.join(tempfile.gettempdir(), os.path.basename(names[0]))
                with z.open(names[0]) as f, open(tmp, "wb") as out:
                    out.write(f.read())
            probe_xlsx(tmp)
        return

    pats = sorted(glob.glob(os.path.join(PAN, "*US06*.mat")))
    print("Panasonic US06 文件数:", len(pats))
    print("\n".join("  " + os.path.basename(p) for p in pats))
    print("\nCALCE A123 包:", len(glob.glob(os.path.join(CAL, "A123_*.zip"))))


if __name__ == "__main__":
    main()
