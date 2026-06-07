from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

from star_follow import paths as _paths
from star_follow.paths import tessdata_dir as _bundled_tessdata
from star_follow.paths import tesseract_exe as _resolve_tess_exe

_PKG = Path(__file__).resolve().parents[1]
_TESS_EXE = _resolve_tess_exe()
_ASCII_TESS_PREFIX = Path(os.environ.get("TESSDATA_PREFIX_ASCII", r"C:\star_follow_ocr"))
_BUNDLED_TESSDATA = _bundled_tessdata()  # 打包內建的 tessdata（chi_tra/eng）
_TESSDATA = _PKG / "tessdata"


def _tess_prefix(base: Path) -> Path:
    inner = base / "tessdata"
    return inner if inner.is_dir() else base


def _has_data(d: Path) -> bool:
    return d.is_dir() and any(d.glob("*.traineddata"))


# 打包模式：把 OCR 搬到 ASCII 路徑（避開中文/空白路徑讓 tesseract 讀不到語言檔）
_staged = _paths.staged_ocr()
if _staged is not None:
    _TESS_EXE, _staged_prefix = _staged
    os.environ["TESSDATA_PREFIX"] = str(_staged_prefix)
elif _has_data(_BUNDLED_TESSDATA):
    os.environ["TESSDATA_PREFIX"] = str(_BUNDLED_TESSDATA)
elif (_ASCII_TESS_PREFIX / "tessdata").is_dir():
    os.environ["TESSDATA_PREFIX"] = str(_tess_prefix(_ASCII_TESS_PREFIX))
elif _has_data(_TESSDATA):
    os.environ["TESSDATA_PREFIX"] = str(_tess_prefix(_PKG))

if _TESS_EXE.is_file():
    pytesseract.pytesseract.tesseract_cmd = str(_TESS_EXE)

_DIGITS = re.compile(r"[^\d]")
_OCR_SCALE = 1
_FAST = True


def set_ocr_options(*, fast: bool = True, scale: int = 1) -> None:
    global _FAST, _OCR_SCALE
    _FAST = fast
    _OCR_SCALE = max(1, scale)


def preprocess_stats_cell(img: np.ndarray, scale: int | None = None) -> np.ndarray:
    sc = scale if scale is not None else _OCR_SCALE
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if sc > 1:
        gray = cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
    _, th = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    return th


def preprocess_text(img: np.ndarray, scale: int = 2) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return gray


def ocr_digits(img: np.ndarray, psm: int = 8) -> tuple[str, float]:
    proc = preprocess_stats_cell(img, 2 if not _FAST else 1)
    config = f"--psm {psm} -c tessedit_char_whitelist=0123456789"
    raw = pytesseract.image_to_string(Image.fromarray(proc), config=config).strip()
    digits = _DIGITS.sub("", raw)
    return digits, 0.8 if digits else 0.0


def _cell_may_have_digits(img: np.ndarray) -> bool:
    """快速判斷格子是否可能有數字，跳過空白格不呼叫 Tesseract。"""
    if img.size == 0:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return int((gray < 115).sum()) >= 25


def ocr_amount(img: np.ndarray) -> tuple[int, float]:
    if img.size == 0:
        return 0, 0.0
    if not _cell_may_have_digits(img):
        return 0, 0.0
    proc = preprocess_stats_cell(img)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(Image.fromarray(proc), config=config).strip()
    text = _DIGITS.sub("", text.replace(",", "").replace("，", ""))
    if not text:
        return 0, 0.0
    try:
        return int(text), 0.85
    except ValueError:
        return 0, 0.0


def ocr_balance(img: np.ndarray) -> tuple[int, float]:
    """讀左下角餘額（亮黃數字、深色底，格式像 183,256）。

    回傳 (金額, 信心)。讀不到回 (0, 0.0)。
    """
    if img is None or img.size == 0:
        return 0, 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    # 亮字深底：取亮的像素為前景，再反白成「黑字白底」給 Tesseract
    _, th = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
    th = cv2.bitwise_not(th)
    # 不再「取最長」（會偏向把逗號誤讀成數字而多一位）；改用單行 psm 7 為主，
    # 只有 psm 7 讀不到時才退而用 psm 6。多一位/少一位的偶發誤差由上層多次取多數決處理。
    best = ""
    for psm in (7, 6):
        config = f"--psm {psm} -c tessedit_char_whitelist=0123456789,"
        try:
            raw = pytesseract.image_to_string(Image.fromarray(th), config=config)
        except pytesseract.TesseractError:
            continue
        digits = _DIGITS.sub("", raw)
        if digits:
            best = digits
            break
    if not best:
        return 0, 0.0
    try:
        return int(best), 0.85
    except ValueError:
        return 0, 0.0


def ocr_account_name(img: np.ndarray) -> tuple[str, float]:
    """讀左下角遊戲帳號名（淺色字、深底，中文）。

    回傳 (名稱, 信心)。讀不到回 ("", 0.0)。
    """
    if img is None or img.size == 0:
        return "", 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    th = cv2.bitwise_not(th)  # 亮字深底 → 黑字白底
    config = "--psm 7 -l chi_tra"
    try:
        raw = pytesseract.image_to_string(Image.fromarray(th), config=config)
    except pytesseract.TesseractError:
        return "", 0.0
    name = "".join(raw.split())  # 去掉所有空白
    return name, 0.7 if name else 0.0


def preprocess_name_cell(img: np.ndarray, scale: int = 3) -> np.ndarray:
    """玩家暱稱（黑字、淺色背景）：放大 + Otsu 二值化，必要時反白。"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Tesseract 偏好黑字白底；暱稱是黑字，若前景偏黑則維持，否則反白
    if float(th.mean()) < 127:
        th = cv2.bitwise_not(th)
    return th


def ocr_name_cell(img: np.ndarray) -> tuple[str, float]:
    """讀玩家暱稱：放大+Otsu，psm 7 單行 chi_tra。"""
    if img.size == 0:
        return "", 0.0
    proc = preprocess_name_cell(img)
    config = "--psm 7 -l chi_tra"
    try:
        raw = pytesseract.image_to_string(Image.fromarray(proc), config=config).strip()
        return raw, 0.7 if raw else 0.0
    except pytesseract.TesseractError:
        return "", 0.0


def warmup_ocr() -> None:
    """啟動時先載入 Tesseract（中文+數字），避免第一局冷啟動卡 ~4 秒。"""
    dummy = np.full((40, 120, 3), 255, dtype=np.uint8)
    # 用純 numpy 畫幾條黑色直條當假字即可暖機 Tesseract；不呼叫 cv2 繪圖函式，
    # 避免凍結環境下 cv2 native 綁定偶發載入不全（缺 putText）而讓整支程式崩潰。
    dummy[8:32, 20:28] = 0
    dummy[8:32, 44:52] = 0
    dummy[8:32, 68:76] = 0
    try:
        ocr_digits(dummy)
        ocr_name_cell(dummy)
    except Exception:
        pass


def ocr_chinese_line(img: np.ndarray, *, stats: bool = False) -> tuple[str, float]:
    proc = preprocess_stats_cell(img) if stats else preprocess_text(img)
    config = "--psm 7 -l chi_tra"
    try:
        raw = pytesseract.image_to_string(Image.fromarray(proc), config=config).strip()
        return raw, 0.7 if raw else 0.0
    except pytesseract.TesseractError:
        return "", 0.0


@lru_cache(maxsize=1)
def _paddle_reader():
    from paddleocr import PaddleOCR

    return PaddleOCR(use_angle_cls=False, lang="ch", show_log=False)


def ocr_chinese_paddle(img: np.ndarray) -> tuple[str, float]:
    try:
        reader = _paddle_reader()
    except Exception:
        return ocr_chinese_line(img)
    result = reader.ocr(img, cls=False)
    if not result or not result[0]:
        return "", 0.0
    texts: list[str] = []
    confs: list[float] = []
    for line in result[0]:
        texts.append(line[1][0])
        confs.append(float(line[1][1]))
    return "".join(texts), sum(confs) / len(confs)
