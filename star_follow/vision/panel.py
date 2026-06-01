from __future__ import annotations

import cv2
import numpy as np

from .ocr import ocr_chinese_line

_ROW_LABELS = ("莊家", "閒家", "和局", "莊對", "閒對")
_MIN_ROW_HITS = 2

_MENU_HINTS = ("統計", "設定", "離開", "規則", "說明", "押注", "返回", "大廳")
_MENU_STATS_KEYS = ("統計", "押注統計", "押注", "圖表")


def _ocr_text(roi: np.ndarray) -> str:
    if roi.size == 0:
        return ""
    text, _ = ocr_chinese_line(roi)
    return text or ""


def _header_brown_ratio(header: np.ndarray) -> float:
    if header.size == 0:
        return 0.0
    hsv = cv2.cvtColor(header, cv2.COLOR_RGB2HSV)
    brown = cv2.inRange(hsv, np.array([8, 30, 40]), np.array([25, 255, 220]))
    return float(brown.sum() / 255) / max(1, header.shape[0] * header.shape[1])


def _count_row_labels(text: str) -> int:
    return sum(1 for label in _ROW_LABELS if label in text)


def stats_panel_debug(
    frame: np.ndarray,
    panel_rect: list[int],
    *,
    table_rect: list[int] | None = None,
    close_rect: list[int] | None = None,
) -> dict:
    """除錯：回傳各判斷條件結果。"""
    x, y, w, h = panel_rect
    roi = frame[y : y + h, x : x + w]
    header = roi[: min(72, roi.shape[0]), :]
    title = _ocr_text(header)
    info: dict = {
        "title": title,
        "header_brown": round(_header_brown_ratio(header), 3),
        "row_labels": 0,
        "row_text": "",
        "close_mean": None,
        "close_std": None,
        "open": False,
    }
    if table_rect is not None:
        tx, ty, tw, th = table_rect
        left = frame[ty : ty + th, tx : tx + int(tw * 0.18)]
        row_text = _ocr_text(left)
        info["row_text"] = row_text
        info["row_labels"] = _count_row_labels(row_text)
    if close_rect is not None:
        cx, cy, cw, ch = close_rect
        g = cv2.cvtColor(frame[cy : cy + ch, cx : cx + cw], cv2.COLOR_RGB2GRAY)
        info["close_mean"] = round(float(g.mean()), 1)
        info["close_std"] = round(float(g.std()), 1)
    info["open"] = stats_panel_open(
        frame, panel_rect, table_rect=table_rect, close_rect=close_rect
    )
    return info


def stats_table_visible(frame: np.ndarray, table_rect: list[int] | None) -> tuple[bool, float]:
    """以統計表底色（淺米色）判斷是否開啟。

    統計表格底為淺米色（高明度、低飽和）；下注區為紅棕色（高飽和）。
    回傳 (是否開啟, 米色比例)。此法不需 OCR，快且不會被下注區紅棕色誤判。
    """
    if not table_rect:
        return False, 0.0
    tx, ty, tw, th = table_rect
    if tw <= 0 or th <= 0:
        return False, 0.0
    roi = frame[ty : ty + th, tx : tx + tw]
    if roi.size == 0:
        return False, 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    beige = float(((v > 150) & (s < 90)).mean())
    return beige > 0.40, beige


def stats_panel_visible_fast(
    frame: np.ndarray,
    panel_rect: list[int],
    *,
    close_rect: list[int] | None = None,
) -> bool:
    """快速視覺判斷（不跑 OCR），用於統計表剛彈出時搶時間偵測。"""
    x, y, w, h = panel_rect
    if w <= 0 or h <= 0:
        return False
    roi = frame[y : y + h, x : x + w]
    header = roi[: min(72, roi.shape[0]), :]
    if _header_brown_ratio(header) > 0.22:
        return True
    if close_rect is not None:
        cx, cy, cw, ch = close_rect
        close_roi = frame[cy : cy + ch, cx : cx + cw]
        if close_roi.size > 0:
            gray = cv2.cvtColor(close_roi, cv2.COLOR_RGB2GRAY)
            if float(gray.mean()) > 130 and float(gray.std()) > 35:
                return True
    return False


def stats_panel_open(
    frame: np.ndarray,
    panel_rect: list[int],
    *,
    table_rect: list[int] | None = None,
    close_rect: list[int] | None = None,
    strict: bool = False,
) -> bool:
    """
    押注統計窗是否開啟。
    須有標題「押注統計」，或表格左欄同時出現多個列名（莊家/閒家…）。
    strict=True 時只認標題與列名，不用棕色/關閉鈕啟發式
    （下注區本身就是紅棕色，會讓非嚴格判斷誤判成開啟）。
    """
    x, y, w, h = panel_rect
    if w <= 0 or h <= 0:
        return False
    roi = frame[y : y + h, x : x + w]
    header = roi[: min(72, roi.shape[0]), :]
    title = _ocr_text(header)

    if "押注統計" in title:
        return True
    if "押注" in title and "統計" in title:
        return True

    row_text = ""
    if table_rect is not None:
        tx, ty, tw, th = table_rect
        t_roi = frame[ty : ty + th, tx : tx + tw]
        if t_roi.size > 0:
            left = t_roi[:, : max(8, int(tw * 0.18))]
            row_text = _ocr_text(left)
            if _count_row_labels(row_text) >= _MIN_ROW_HITS:
                return True

    if strict:
        return False

    header_brown = _header_brown_ratio(header) > 0.32
    if close_rect is not None and header_brown:
        cx, cy, cw, ch = close_rect
        close_roi = frame[cy : cy + ch, cx : cx + cw]
        if close_roi.size > 0:
            gray = cv2.cvtColor(close_roi, cv2.COLOR_RGB2GRAY)
            if float(gray.mean()) > 145 and float(gray.std()) > 42:
                return True

    return False


def menu_dropdown_open(frame: np.ndarray, menu_panel_rect: list[int]) -> bool:
    """相容舊呼叫；請改傳 menu_button ROI 或 tuple。"""
    from star_follow.vision.menu_match import menu_dropdown_open as _open

    return _open(frame, tuple(menu_panel_rect))


def find_stats_menu_click(
    frame: np.ndarray,
    menu_panel_rect: list[int],
) -> tuple[int, int] | None:
    x, y, w, h = menu_panel_rect
    if w <= 0 or h <= 0:
        return None
    roi = frame[y : y + h, x : x + w]
    row_h = max(28, h // 4)
    n_rows = max(2, (h + row_h - 1) // row_h)

    for i in range(n_rows):
        ry0 = min(i * row_h, h - 1)
        ry1 = min(h, ry0 + row_h)
        band = roi[ry0:ry1, :]
        text = _ocr_text(band)
        if any(k in text for k in _MENU_STATS_KEYS):
            return x + w // 2, y + ry0 + (ry1 - ry0) // 2

    full = _ocr_text(roi)
    if "統計" in full or "押注" in full:
        return x + w // 2, y + h // 3

    return None


def scan_menu_rows(
    frame: np.ndarray,
    menu_panel_rect: list[int],
) -> list[tuple[int, str]]:
    x, y, w, h = menu_panel_rect
    roi = frame[y : y + h, x : x + w]
    row_h = max(28, h // 4)
    n_rows = max(2, (h + row_h - 1) // row_h)
    out: list[tuple[int, str]] = []
    for i in range(n_rows):
        ry0 = min(i * row_h, h - 1)
        ry1 = min(h, ry0 + row_h)
        text = _ocr_text(roi[ry0:ry1, :])
        cy = y + ry0 + (ry1 - ry0) // 2
        out.append((cy, text))
    return out
