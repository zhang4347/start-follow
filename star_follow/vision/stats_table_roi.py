from __future__ import annotations

import cv2
import numpy as np


def find_stats_table_in_panel(frame: np.ndarray, panel_rect: list[int]) -> tuple[int, int, int, int] | None:
    """在 stats_panel 內找棕色表格內緣。"""
    x, y, w, h = panel_rect
    panel = frame[y : y + h, x : x + w]
    hsv = cv2.cvtColor(panel, cv2.COLOR_RGB2HSV)
    brown = cv2.inRange(hsv, np.array([8, 30, 40]), np.array([28, 255, 220]))
    brown = cv2.morphologyEx(brown, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(brown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(c)
    if bw < 200 or bh < 150:
        return None
    # 內縮一點避開邊框
    m = 8
    return [x + bx + m, y + by + m, bw - 2 * m, bh - 2 * m]
