"""Windows DPI 與滑鼠座標。"""

from __future__ import annotations

import ctypes
import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)

_dpi_done = False

# win32con metrics
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79


def ensure_dpi_aware() -> None:
    global _dpi_done
    if _dpi_done:
        return
    _dpi_done = True
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def virtual_screen() -> tuple[int, int, int, int]:
    import win32api

    u = win32api
    return (
        u.GetSystemMetrics(SM_XVIRTUALSCREEN),
        u.GetSystemMetrics(SM_YVIRTUALSCREEN),
        u.GetSystemMetrics(SM_CXVIRTUALSCREEN),
        u.GetSystemMetrics(SM_CYVIRTUALSCREEN),
    )


def clamp_screen(sx: int, sy: int) -> tuple[int, int]:
    vx, vy, vw, vh = virtual_screen()
    sx = max(vx, min(vx + vw - 1, int(sx)))
    sy = max(vy, min(vy + vh - 1, int(sy)))
    return sx, sy


def get_cursor_pos() -> tuple[int, int]:
    import win32api

    return win32api.GetCursorPos()


def _try(name: str, fn: Callable[[], None]) -> bool:
    try:
        fn()
        return True
    except Exception as exc:
        logger.debug("%s failed: %s", name, exc)
        return False


def _set_cursor_win32(sx: int, sy: int) -> None:
    import win32api

    win32api.SetCursorPos((sx, sy))


def _set_cursor_ctypes(sx: int, sy: int) -> None:
    if not ctypes.windll.user32.SetCursorPos(int(sx), int(sy)):
        raise OSError("SetCursorPos returned 0")


def _set_cursor_sendinput(sx: int, sy: int) -> None:
    user32 = ctypes.windll.user32
    vx, vy, vw, vh = virtual_screen()
    ax = int((sx - vx) * 65535 / max(vw - 1, 1))
    ay = int((sy - vy) * 65535 / max(vh - 1, 1))

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.c_ulong),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class INPUT(ctypes.Structure):
        class _U(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT)]

        _anonymous_ = ("u",)
        _fields_ = [("type", ctypes.c_ulong), ("u", _U)]

    def _inp(flags: int, dx: int, dy: int) -> INPUT:
        i = INPUT(type=0)
        i.mi = MOUSEINPUT(dx, dy, 0, flags, 0, None)
        return i

    move = _inp(0x8001, ax, ay)  # MOVE | ABSOLUTE
    if user32.SendInput(1, ctypes.byref(move), ctypes.sizeof(INPUT)) != 1:
        raise OSError("SendInput move failed")


def set_cursor_pos(sx: int, sy: int) -> bool:
    """移動滑鼠；成功回傳 True。"""
    sx, sy = clamp_screen(sx, sy)
    for name, fn in (
        ("win32", lambda: _set_cursor_win32(sx, sy)),
        ("ctypes", lambda: _set_cursor_ctypes(sx, sy)),
        ("SendInput", lambda: _set_cursor_sendinput(sx, sy)),
    ):
        if _try(name, fn):
            return True
    return False


def click_screen(sx: int, sy: int) -> bool:
    """移動並左鍵點擊。成功回傳 True。"""
    sx, sy = clamp_screen(sx, sy)
    if not set_cursor_pos(sx, sy):
        return False
    import win32api
    import win32con

    time.sleep(0.03)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
    time.sleep(0.04)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    return True
