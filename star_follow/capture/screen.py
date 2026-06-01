from __future__ import annotations

import logging
import time

import numpy as np
from mss import mss
from PIL import Image

from .window import GameWindow

logger = logging.getLogger(__name__)

# 重用單一 mss 實例（每次 with mss() 都重建會多花數十毫秒）
_SCT = None
_slow_warned = False


def _sct():
    global _SCT
    if _SCT is None:
        _SCT = mss()
    return _SCT


def capture_region(left: int, top: int, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid capture size: {width}x{height}")
    t0 = time.perf_counter()
    try:
        shot = _sct().grab({"left": left, "top": top, "width": width, "height": height})
    except Exception:
        # mss 實例可能因螢幕變更失效，重建一次再試
        global _SCT
        _SCT = None
        shot = _sct().grab({"left": left, "top": top, "width": width, "height": height})
    img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    arr = np.array(img)
    elapsed = (time.perf_counter() - t0) * 1000
    global _slow_warned
    if elapsed > 400 and not _slow_warned:
        logger.warning("截圖耗時 %.0fms（偏慢，可能是多螢幕/DPI 影響速度）", elapsed)
        _slow_warned = True
    return arr


def capture_client(win: GameWindow) -> np.ndarray:
    from star_follow.capture.window import refresh_game_window

    win = refresh_game_window(win.hwnd, win.title)
    return capture_region(win.client_left, win.client_top, win.client_width, win.client_height)


def capture_roi(win: GameWindow, rect: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    return capture_region(win.client_left + x, win.client_top + y, w, h)
