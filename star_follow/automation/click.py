from __future__ import annotations

import logging
import random
import time

from star_follow.capture.dpi import (
    click_screen,
    ensure_dpi_aware,
    get_cursor_pos,
    set_cursor_pos,
)
from star_follow.capture.window import GameWindow, focus_window, refresh_game_window

logger = logging.getLogger(__name__)

ensure_dpi_aware()

def get_click_backend(explicit: str = "auto") -> str:
    """相容舊參數；auto = ui_click_backend。"""
    if explicit != "auto":
        return explicit
    return get_ui_click_backend()


def get_menu_click_backend() -> str:
    try:
        from star_follow.config import load_config

        return load_config().automation.menu_click_backend
    except Exception:
        return "postmessage"


def get_ui_click_backend() -> str:
    try:
        from star_follow.config import load_config

        return load_config().automation.ui_click_backend
    except Exception:
        return "win32"


def screen_point_from_client(win: GameWindow, x: int, y: int) -> tuple[int, int]:
    import win32gui

    win = refresh_game_window(win.hwnd, win.title)
    return win32gui.ClientToScreen(win.hwnd, (int(x), int(y)))


def _client_point_for_hwnd(root_hwnd: int, cx: int, cy: int) -> tuple[int, int, int]:
    """將 root client 座標轉成實際接收訊息的 hwnd 與其 client 座標。"""
    import win32gui

    cx_i, cy_i = int(cx), int(cy)
    pt_screen = win32gui.ClientToScreen(root_hwnd, (cx_i, cy_i))
    target = root_hwnd
    try:
        child = win32gui.ChildWindowFromPoint(root_hwnd, (cx_i, cy_i))
        if child:
            target = child
    except Exception:
        pass
    tx, ty = win32gui.ScreenToClient(target, pt_screen)
    return target, tx, ty


def click_at(
    win: GameWindow,
    x: int,
    y: int,
    *,
    jitter: tuple[int, int] = (0, 0),
    delay_ms: tuple[int, int] = (30, 80),
    backend: str = "auto",
    refocus: bool = True,
) -> tuple[int, int]:
    win = refresh_game_window(win.hwnd, win.title)
    jx = random.randint(-jitter[0], jitter[0]) if jitter[0] else 0
    jy = random.randint(-jitter[1], jitter[1]) if jitter[1] else 0
    cx, cy = x + jx, y + jy
    sx, sy = screen_point_from_client(win, cx, cy)
    mode = get_click_backend(backend)

    if refocus:
        focus_window(win.hwnd)
        time.sleep(0.22)

    ok = False
    if mode == "postmessage":
        ok = _click_client_message(win.hwnd, cx, cy)
    elif mode == "direct":
        ok = _click_pydirectinput(sx, sy)
    elif mode == "win32":
        ok = click_screen(sx, sy)
    else:  # auto: 先試滑鼠，失敗改 postmessage
        ok = click_screen(sx, sy)
        if not ok:
            ok = _click_client_message(win.hwnd, cx, cy)

    if ok:
        logger.info("點擊 client=(%d,%d) screen=(%d,%d) backend=%s", cx, cy, sx, sy, mode)
    else:
        logger.warning("點擊可能失敗 client=(%d,%d) backend=%s", cx, cy, mode)

    lo, hi = delay_ms
    time.sleep(random.randint(lo, hi) / 1000.0)
    return sx, sy


def _click_pydirectinput(sx: int, sy: int) -> bool:
    try:
        import pydirectinput

        pydirectinput.PAUSE = 0
        pydirectinput.FAILSAFE = False
        pydirectinput.moveTo(sx, sy)
        time.sleep(0.02)
        pydirectinput.click()
        return True
    except Exception:
        return False


def _click_client_message(hwnd: int, cx: int, cy: int) -> bool:
    """SendMessage 點擊（client 座標；子視窗會自動換算）。"""
    try:
        import win32api
        import win32con
        import win32gui

        target, tx, ty = _client_point_for_hwnd(hwnd, cx, cy)
        lparam = win32api.MAKELONG(tx & 0xFFFF, ty & 0xFFFF)

        for msg, wp in (
            (win32con.WM_MOUSEMOVE, 0),
            (win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON),
            (win32con.WM_LBUTTONUP, 0),
        ):
            win32gui.SendMessage(target, msg, wp, lparam)
            time.sleep(0.03)
        return True
    except Exception as exc:
        logger.debug("SendMessage click failed: %s", exc)
        return False


def wheel_at_client(
    win: GameWindow,
    x: int,
    y: int,
    *,
    clicks: int = 1,
    delta: int = -120,
    delay_ms: tuple[int, int] = (50, 100),
) -> None:
    """在 client 座標滾輪（delta<0 往下）。"""
    win = refresh_game_window(win.hwnd, win.title)
    focus_window(win.hwnd)
    sx, sy = screen_point_from_client(win, x, y)
    set_cursor_pos(sx, sy)
    time.sleep(0.05)
    try:
        import win32api
        import win32con

        for _ in range(max(1, clicks)):
            win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, sx, sy, int(delta), 0)
            lo, hi = delay_ms
            time.sleep(random.randint(lo, hi) / 1000.0)
    except Exception as exc:
        logger.warning("滾輪失敗 (%d,%d): %s", x, y, exc)


def drag_client(
    win: GameWindow,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    steps: int = 16,
    hold_ms: int = 60,
) -> None:
    """在 client 座標按住左鍵從 (x0,y0) 拖到 (x1,y1)（大廳橫向滑動換頁用）。

    用實體滑鼠（win32），分段移動模擬人手拖曳；遊戲對瞬間跳點的拖曳常不認。
    """
    win = refresh_game_window(win.hwnd, win.title)
    focus_window(win.hwnd)
    sx0, sy0 = screen_point_from_client(win, x0, y0)
    sx1, sy1 = screen_point_from_client(win, x1, y1)
    import win32api
    import win32con

    set_cursor_pos(sx0, sy0)
    time.sleep(0.08)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(hold_ms / 1000.0)
    for i in range(1, steps + 1):
        ix = int(sx0 + (sx1 - sx0) * i / steps)
        iy = int(sy0 + (sy1 - sy0) * i / steps)
        set_cursor_pos(ix, iy)
        time.sleep(0.012)
    time.sleep(0.05)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    time.sleep(0.15)


def move_to_client(win: GameWindow, x: int, y: int) -> tuple[int, int]:
    win = refresh_game_window(win.hwnd, win.title)
    focus_window(win.hwnd)
    time.sleep(0.05)
    sx, sy = screen_point_from_client(win, x, y)
    before = get_cursor_pos()
    ok = set_cursor_pos(sx, sy)
    after = get_cursor_pos()
    logger.info(
        "滑鼠 %s → 目標(%d,%d) 實際(%s) ok=%s",
        before,
        sx,
        sy,
        after,
        ok,
    )
    return sx, sy
