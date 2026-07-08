"""珠盤路（bead plate）偵測：判斷牌桌下方路子是否為空盤 → 新的一條牌（新靴）。

原理：牌局進行中珠盤路滿是紅/藍珠子（實測顏色佔比 ~33-37%）；一開新靴時整塊
全白、沒有任何珠子（顏色佔比 ≈ 0%）。兩者差距極大，故以「顏色佔比低於門檻」即可
穩定判為新靴。

注意：ROI 需對準預設「全路」檢視左下的珠盤路格子（避開上方『莊贏/閒贏…』標題列，
那些字永遠有色會誤判）。frame 為 capture_client 的 RGB numpy 陣列。
"""

from __future__ import annotations

import numpy as np


def bead_colored_fraction(frame: np.ndarray, rect: tuple[int, int, int, int]) -> float:
    """回傳 ROI 內紅/藍珠子像素佔比（0.0~1.0）。"""
    x, y, w, h = rect
    hgt, wid = frame.shape[:2]
    x = max(0, min(int(x), wid - 1))
    y = max(0, min(int(y), hgt - 1))
    w = max(1, min(int(w), wid - x))
    h = max(1, min(int(h), hgt - y))
    c = frame[y : y + h, x : x + w].astype(np.int16)
    r = c[..., 0]
    g = c[..., 1]
    b = c[..., 2]
    red = (r > 120) & (r - g > 50) & (r - b > 50)
    blue = (b > 110) & (b - r > 40) & (b - g > 20)
    col = red | blue
    return float(col.mean()) if col.size else 0.0


def is_new_shoe(frame: np.ndarray, rect: tuple[int, int, int, int], threshold: float) -> bool:
    """珠盤路顏色佔比低於門檻 → 判為新的一條牌（空盤）。"""
    return bead_colored_fraction(frame, rect) < threshold
