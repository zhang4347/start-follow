# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包設定：產生免裝 Python 的單一資料夾版。

打包內容：
  - Python 直譯器 + 所有套件（mss/opencv/numpy/Pillow/pytesseract/pywin32/PyYAML...）
  - Tesseract 執行檔 + 所有 DLL（內建於 _internal/tesseract）
  - 語言檔 chi_tra / eng（內建於 _internal/tessdata）

對外可編輯檔（config.yaml、data/follow_list.json、logs/）由打包腳本放在 exe 旁。
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

PROJECT = Path(SPECPATH)
TESS_SRC = Path(r"C:\Program Files\Tesseract-OCR")

# --- 一起打包的資料檔（verbatim 複製，不做相依分析） ---
datas = []

# --- OpenSSL DLL（HTTPS 必需） ---
# Anaconda 的 OpenSSL DLL 放在 Library\bin，PyInstaller 預設抓不到，導致 _ssl
# 無法載入 → 整支 exe 不能連 HTTPS（自動更新、Google 試算表上傳全部失敗）。
# 這裡明確把 libssl/libcrypto 收進來放到根目錄，並補上 _ssl/_hashlib 擴充。
binaries = []
# venv 的 sys.prefix 沒有 Library\bin；OpenSSL DLL 在 base_prefix（Anaconda）下，
# 所以兩個 prefix 都要找。
_PREFIXES = [Path(sys.prefix)]
_base = Path(getattr(sys, "base_prefix", sys.prefix))
if _base not in _PREFIXES:
    _PREFIXES.append(_base)
_ssl_names = ("libssl-3-x64.dll", "libcrypto-3-x64.dll",
              "libssl-3.dll", "libcrypto-3.dll")
_seen_dll = set()
for _pref in _PREFIXES:
    for _ssl_dir in (_pref / "Library" / "bin", _pref / "DLLs", _pref):
        for _name in _ssl_names:
            _p = _ssl_dir / _name
            if _p.is_file() and _name not in _seen_dll:
                binaries.append((str(_p), "."))
                _seen_dll.add(_name)
    for _pyd in ("_ssl.pyd", "_hashlib.pyd"):
        _p = _pref / "DLLs" / _pyd
        if _p.is_file() and _pyd not in _seen_dll:
            binaries.append((str(_p), "."))
            _seen_dll.add(_pyd)
if not any(n.startswith("libssl") for n in _seen_dll):
    raise SystemExit("打包中止：找不到 OpenSSL DLL（libssl/libcrypto），HTTPS 會壞。請確認 Anaconda Library\\bin。")

# Tesseract 主程式 + 執行所需 DLL（只取 tesseract.exe，不含訓練工具）
datas.append((str(TESS_SRC / "tesseract.exe"), "tesseract"))
for dll in TESS_SRC.glob("*.dll"):
    datas.append((str(dll), "tesseract"))

# 中文/英文語言檔
for td in (PROJECT / "star_follow" / "tessdata").glob("*.traineddata"):
    datas.append((str(td), "tessdata"))

# 內建一份「參考 config.yaml」到 _ref/：新版啟動時用它覆蓋外部 config.yaml，
# 確保自動更新（換檔腳本會保留舊 config）的舊使用者也能拿到新版座標/門檻。
datas.append((str(PROJECT / "star_follow" / "config.yaml"), "_ref"))

# match_template 用的模板圖：務必打包，否則凍結後讀不到、所有模板分數恆為 0
# （導覽/五局提示確定鈕全失效）。放到資源根的 templates/，對應 paths.templates_dir()。
for _tpl in (PROJECT / "star_follow" / "vision" / "templates").glob("*.png"):
    datas.append((str(_tpl), "templates"))

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
    # HTTPS 相關（自動更新／Google 試算表上傳都需要）
    "ssl",
    "_ssl",
    "_hashlib",
    "certifi",
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
    binaries=binaries,
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
