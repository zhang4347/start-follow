from __future__ import annotations

import logging
import time

import numpy as np
from mss import mss
from PIL import Image

from .window import GameWindow

logger = logging.getLogger(__name__)


class CaptureUnavailable(RuntimeError):
    """螢幕暫時無法擷取：遊戲視窗最小化、螢幕休眠/鎖定、遠端斷線或 DWM 重置時，
    mss 的 BitBlt 會失敗。這是「可重試」的暫時狀態，呼叫端應放慢輪詢稍後再試，
    不要當成程式錯誤狂噴 traceback。"""


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
    global _SCT
    try:
        shot = _sct().grab({"left": left, "top": top, "width": width, "height": height})
    except Exception:
        # mss 實例可能因螢幕變更失效，重建一次再試；再失敗就視為「暫時抓不到」
        _SCT = None
        try:
            shot = _sct().grab(
                {"left": left, "top": top, "width": width, "height": height}
            )
        except Exception as exc:
            _SCT = None  # 下次重建
            raise CaptureUnavailable(str(exc) or "screen grab failed") from exc
    img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    arr = np.array(img)
    elapsed = (time.perf_counter() - t0) * 1000
    global _slow_warned
    if elapsed > 400 and not _slow_warned:
        logger.warning("截圖耗時 %.0fms（偏慢，可能是多螢幕/DPI 影響速度）", elapsed)
        _slow_warned = True
    return arr


def capture_client(win: GameWindow) -> np.ndarray:
    import win32con
    import win32gui

    from star_follow.capture.window import refresh_game_window

    # 視窗最小化時 BitBlt 必失敗：先嘗試還原，並回報「暫時抓不到」讓上層放慢重試。
    try:
        if win32gui.IsIconic(win.hwnd):
            try:
                win32gui.ShowWindow(win.hwnd, win32con.SW_RESTORE)
            except Exception:  # noqa: BLE001
                pass
            raise CaptureUnavailable("game window minimized")
        win = refresh_game_window(win.hwnd, win.title)
    except CaptureUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001  視窗 handle 失效等
        raise CaptureUnavailable(f"window not ready: {exc}") from exc
    if win.client_width < 50 or win.client_height < 50:
        raise CaptureUnavailable("game window client area too small")
    return capture_region(win.client_left, win.client_top, win.client_width, win.client_height)


def capture_roi(win: GameWindow, rect: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    return capture_region(win.client_left + x, win.client_top + y, w, h)
