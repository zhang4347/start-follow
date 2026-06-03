"""大廳導覽用 OCR 輔助：在固定區塊找關鍵字，當模板的第二證據。"""

from __future__ import annotations

import re

import numpy as np

from star_follow.vision.ocr import ocr_chinese_line


def _crop(frame: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return frame[:0, :0]
    return frame[y : y + h, x : x + w]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def region_has_keywords(
    frame: np.ndarray,
    rect: tuple[int, int, int, int],
    keywords: tuple[str, ...],
) -> tuple[bool, str, float]:
    """裁切區域 OCR，檢查是否含任一關鍵字。回傳 (命中, 原文, 信心)。"""
    crop = _crop(frame, rect)
    if crop.size == 0:
        return False, "", 0.0
    text, conf = ocr_chinese_line(crop)
    norm = _norm(text)
    if not norm:
        return False, text, conf
    hit = any(k in norm for k in keywords)
    return hit, text, conf
