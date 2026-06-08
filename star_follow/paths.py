"""執行路徑解析：同時支援開發模式與 PyInstaller 打包（frozen）。

打包成 exe 後：
  - 可編輯檔（config.yaml、data/follow_list.json、logs/）放在「exe 所在資料夾」，
    方便客戶用記事本直接修改、查看 log。
  - 內建唯讀資源（tessdata 語言檔、bundled tesseract）放在 _MEIPASS / exe 旁。

開發模式（直接跑 python -m star_follow.tools.run）：
  - 維持原本 star_follow/ 套件內的相對路徑，行為與打包前完全一致。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent  # .../star_follow

# Tesseract 對「非 ASCII 路徑」（中文、特殊字元）會讀不到語言檔，
# 因此打包後若 exe 落在中文/含空白路徑，會把 OCR 檔複製到固定 ASCII 目錄再用。
_ASCII_OCR_CANDIDATES = (
    Path(r"C:\ProgramData\StarFollow\ocr"),
    Path(r"C:\star_follow_ocr"),
)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """可編輯檔（config.yaml、data、logs）所在資料夾。"""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return _PKG_DIR


def resource_dir() -> Path:
    """內建唯讀資源（tessdata、tesseract）所在資料夾。"""
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return _PKG_DIR


def config_path() -> Path:
    return app_dir() / "config.yaml"


def follow_list_path() -> Path:
    return app_dir() / "data" / "follow_list.json"


def logs_dir() -> Path:
    return app_dir() / "logs"


def launch_settings_path() -> Path:
    """雙擊 exe 時讀取的啟動設定（模式/下注），放在 exe 旁邊方便記事本修改。"""
    return app_dir() / "啟動設定.txt"


def balance_state_path() -> Path:
    """記錄上次餘額，用來比較賺賠。"""
    return app_dir() / "data" / "balance_state.json"


def tessdata_dir() -> Path:
    """內建 tessdata（chi_tra/eng）。開發與打包優先用 star_follow/tessdata。"""
    for base in (_PKG_DIR, resource_dir()):
        p = base / "tessdata"
        if p.is_dir() and any(p.glob("*.traineddata")):
            return p
    return _PKG_DIR / "tessdata"


def tesseract_exe() -> Path:
    """Tesseract 執行檔。

    開發與客戶端必須同一套：優先 star_follow/tesseract（build 前從本機
    C:\\Program Files\\Tesseract-OCR 同步）。只有缺檔時才退回系統安裝路徑。
    """
    for base in (_PKG_DIR, resource_dir()):
        p = base / "tesseract" / "tesseract.exe"
        if p.is_file():
            return p
    env = os.environ.get("TESSERACT_CMD")
    if env and Path(env).is_file():
        return Path(env)
    return Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


def ocr_runtime() -> tuple[Path, Path]:
    """回傳 (tesseract_exe, tessdata_dir)。打包模式會走 staged_ocr（ASCII 路徑）。"""
    staged = staged_ocr()
    if staged is not None:
        return staged
    return tesseract_exe(), tessdata_dir()


def templates_dir() -> Path:
    """模板圖（match_template 比對用）所在資料夾。

    凍結（打包）後放在打包資源根的 templates/；開發模式用套件內的
    star_follow/vision/templates/。先前用 Path(__file__).parent/templates，
    凍結後 __file__ 指向 PYZ 內虛擬路徑而讀不到模板（分數恆為 0）。
    """
    if is_frozen():
        p = resource_dir() / "templates"
        if p.is_dir():
            return p
    return _PKG_DIR / "vision" / "templates"


def _is_ascii(p: Path) -> bool:
    return all(ord(c) < 128 for c in str(p))


def _sync_dir(src: Path, dst: Path) -> None:
    """把 src 內所有檔複製到 dst（大小不同或不存在才複製）。"""
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*"):
        if not f.is_file():
            continue
        target = dst / f.name
        if target.is_file() and target.stat().st_size == f.stat().st_size:
            continue
        shutil.copy2(f, target)


def staged_ocr() -> tuple[Path, Path] | None:
    """確保 OCR（tesseract + tessdata）位於 ASCII 路徑，回傳 (tesseract_exe, tessdata_dir)。

    - 非打包模式：回傳 None（沿用開發環境設定）。
    - 打包後若內建路徑本身已是純 ASCII：直接用內建，不複製。
    - 否則把內建 tesseract/ 與 tessdata/ 複製到第一個可寫入的 ASCII 目錄後使用。
    """
    if not is_frozen():
        return None

    src_tess_dir = resource_dir() / "tesseract"
    src_data_dir = resource_dir() / "tessdata"
    bundled_exe = src_tess_dir / "tesseract.exe"

    # 內建路徑已是 ASCII，直接用，省去複製
    if _is_ascii(bundled_exe) and _is_ascii(src_data_dir):
        return bundled_exe, src_data_dir

    for root in _ASCII_OCR_CANDIDATES:
        try:
            dst_tess = root / "tesseract"
            dst_data = root / "tessdata"
            _sync_dir(src_tess_dir, dst_tess)
            _sync_dir(src_data_dir, dst_data)
            exe = dst_tess / "tesseract.exe"
            if exe.is_file() and any(dst_data.glob("*.traineddata")):
                return exe, dst_data
        except Exception:
            continue
    # 全部失敗就退回內建（至少嘗試），讓上層自行處理
    return bundled_exe, src_data_dir
