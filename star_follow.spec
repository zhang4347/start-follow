# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包設定：產生免裝 Python 的單一資料夾版。

打包內容：
  - Python 直譯器 + 所有套件（mss/opencv/numpy/Pillow/pytesseract/pywin32/PyYAML...）
  - Tesseract 執行檔 + 所有 DLL（內建於 _internal/tesseract）
  - 語言檔 chi_tra / eng（內建於 _internal/tessdata）

對外可編輯檔（config.yaml、data/follow_list.json、logs/）由打包腳本放在 exe 旁。
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

PROJECT = Path(SPECPATH)
TESS_SRC = Path(r"C:\Program Files\Tesseract-OCR")

# --- 一起打包的資料檔（verbatim 複製，不做相依分析） ---
datas = []

# Tesseract 主程式 + 執行所需 DLL（只取 tesseract.exe，不含訓練工具）
datas.append((str(TESS_SRC / "tesseract.exe"), "tesseract"))
for dll in TESS_SRC.glob("*.dll"):
    datas.append((str(dll), "tesseract"))

# 中文/英文語言檔
for td in (PROJECT / "star_follow" / "tessdata").glob("*.traineddata"):
    datas.append((str(td), "tessdata"))

hiddenimports = [
    "win32gui",
    "win32con",
    "win32api",
    "win32process",
    "pytesseract",
    "mss",
    "cv2",
    "numpy",
    "yaml",
    "PIL",
    "pydirectinput",
    "star_follow.monitor.sheet_uploader",
    "star_follow.update.updater",
]

# Google 試算表上傳（gspread + google-auth）：namespace 套件需明確收集
hiddenimports += ["cachetools"]
for _pkg in ("gspread", "google.auth", "google.oauth2", "requests", "cachetools"):
    hiddenimports += collect_submodules(_pkg)
for _pkg in ("gspread", "google-auth", "requests", "google-api-core"):
    try:
        datas += copy_metadata(_pkg)
    except Exception:
        pass
datas += collect_data_files("gspread")
datas += collect_data_files("certifi")

a = Analysis(
    ["star_follow_app.py"],
    pathex=[str(PROJECT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PyQt5", "tkinter", "matplotlib"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StarFollow",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    uac_admin=True,  # exe 啟動即要求系統管理員（manifest），UAC 由 Windows 跳出
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="StarFollow",
)
