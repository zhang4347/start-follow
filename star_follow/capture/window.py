from __future__ import annotations

import logging
from dataclasses import dataclass

import win32api
import win32con
import win32gui
import win32process

from star_follow.capture.dpi import ensure_dpi_aware

ensure_dpi_aware()

logger = logging.getLogger(__name__)

# 標題比對時一併嘗試的別名（config 未填 title_aliases 時使用）
_DEFAULT_TITLE_ALIASES = (
    "星城Online",
    "星城 Online",
    "星城ONLINE",
    "星城 ONLINE",
    "星城",
)

_MIN_CLIENT_W = 800
_MIN_CLIENT_H = 500

# 一律排除的視窗類別：檔案總管、桌面、工作列等系統殼層。
# （交付資料夾常叫「星城跟注」，檔案總管視窗標題會含「星城」而誤判，必須擋掉。）
_EXCLUDE_CLASSES = {
    "CabinetWClass",          # 檔案總管
    "ExploreWClass",          # 舊式檔案總管
    "Progman",                # 桌面
    "WorkerW",                # 桌面底圖
    "Shell_TrayWnd",          # 工作列
    "Shell_SecondaryTrayWnd",  # 次螢幕工作列
}

# 標題（正規化後：小寫、去空白）含這些片段者排除：編輯器/終端機/本程式自己。
_EXCLUDE_TITLE_FRAGMENTS = (
    "檔案總管",
    "fileexplorer",
    "記事本",
    "notepad",
    "cursor",
    "visualstudiocode",
    "powershell",
    "命令提示字元",
    "windowsterminal",
    "starfollow",
)

# 程序執行檔名（小寫）在此清單者排除（盡力而為，取不到就不擋）。
_EXCLUDE_PROCESSES = {
    "explorer.exe",
    "cursor.exe",
    "code.exe",
    "notepad.exe",
    "powershell.exe",
    "cmd.exe",
    "conhost.exe",
    "windowsterminal.exe",
    "starfollow.exe",
    "python.exe",
    "pythonw.exe",
}


def _window_class(hwnd: int) -> str:
    try:
        return win32gui.GetClassName(hwnd) or ""
    except win32gui.error:
        return ""


def _process_name(hwnd: int) -> str:
    """盡力取得視窗所屬程序的執行檔名（小寫）；失敗回空字串。"""
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        handle = win32api.OpenProcess(
            win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid
        )
        try:
            path = win32process.GetModuleFileNameEx(handle, 0)
        finally:
            win32api.CloseHandle(handle)
        return path.rsplit("\\", 1)[-1].lower()
    except Exception:  # noqa: BLE001  權限不足等情況不擋
        return ""


def _is_excluded_window(hwnd: int, norm_title: str) -> bool:
    if _window_class(hwnd) in _EXCLUDE_CLASSES:
        return True
    if any(frag in norm_title for frag in _EXCLUDE_TITLE_FRAGMENTS):
        return True
    proc = _process_name(hwnd)
    if proc and proc in _EXCLUDE_PROCESSES:
        return True
    return False


@dataclass(frozen=True)
class GameWindow:
    hwnd: int
    title: str
    client_left: int
    client_top: int
    client_width: int
    client_height: int

    @property
    def client_rect_screen(self) -> tuple[int, int, int, int]:
        return (
            self.client_left,
            self.client_top,
            self.client_width,
            self.client_height,
        )


# 全形 ASCII（U+FF01～U+FF5E）對應半形（U+0021～U+007E）
_FULLWIDTH_TO_HALF = {cp: cp - 0xFEE0 for cp in range(0xFF01, 0xFF5F)}


def _normalize_title(s: str) -> str:
    """比對用：去空白、全形英數轉半形、小寫。"""
    s = s.strip().replace("\u3000", " ").replace(" ", "")
    return s.translate(_FULLWIDTH_TO_HALF).lower()


def _title_patterns(primary: str, extra: list[str] | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in [primary, *(extra or []), *_DEFAULT_TITLE_ALIASES]:
        p = (p or "").strip()
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _title_matches(title: str, patterns: list[str]) -> bool:
    if not title.strip():
        return False
    norm = _normalize_title(title)
    for p in patterns:
        if p in title:
            return True
        if _normalize_title(p) in norm:
            return True
    return False


def list_candidate_windows(
    title_substring: str = "星城Online",
    *,
    title_aliases: list[str] | None = None,
    min_w: int = _MIN_CLIENT_W,
    min_h: int = _MIN_CLIENT_H,
) -> list[tuple[int, int, int, bool, bool, str]]:
    """列出可能為遊戲的視窗：(client_w, client_h, hwnd, visible, minimized, title)。"""
    patterns = _title_patterns(title_substring, title_aliases)
    rows: list[tuple[int, int, int, bool, bool, str]] = []

    def handler(hwnd: int, _: object) -> None:
        title = win32gui.GetWindowText(hwnd)
        if not _title_matches(title, patterns):
            return
        if _is_excluded_window(hwnd, _normalize_title(title)):
            return
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        w, h = right - left, bottom - top
        if w < min_w or h < min_h:
            return
        rows.append(
            (
                w,
                h,
                hwnd,
                bool(win32gui.IsWindowVisible(hwnd)),
                bool(win32gui.IsIconic(hwnd)),
                title,
            )
        )

    win32gui.EnumWindows(handler, None)
    rows.sort(key=lambda r: -(r[0] * r[1]))
    return rows


def list_large_windows(
    min_w: int = 600,
    min_h: int = 400,
    limit: int = 25,
) -> list[tuple[int, int, bool, bool, str, str]]:
    """列出目前較大的視窗標題（診斷用：找不到星城時看實際標題列寫什麼）。"""
    rows: list[tuple[int, int, bool, bool, str, str]] = []

    def handler(hwnd: int, _: object) -> None:
        title = win32gui.GetWindowText(hwnd)
        if not title.strip():
            return
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        w, h = right - left, bottom - top
        if w < min_w or h < min_h:
            return
        rows.append(
            (
                w,
                h,
                bool(win32gui.IsWindowVisible(hwnd)),
                bool(win32gui.IsIconic(hwnd)),
                title,
                _window_class(hwnd),
            )
        )

    win32gui.EnumWindows(handler, None)
    rows.sort(key=lambda r: -(r[0] * r[1]))
    return rows[:limit]


def find_game_window(
    title_substring: str = "星城Online",
    *,
    title_aliases: list[str] | None = None,
) -> GameWindow | None:
    rows = list_candidate_windows(title_substring, title_aliases=title_aliases)
    if not rows:
        return None
    w, h, hwnd, _vis, iconic, title = rows[0]
    if iconic:
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        except win32gui.error:
            pass
    win = refresh_game_window(hwnd, title)
    if win.client_width < _MIN_CLIENT_W or win.client_height < _MIN_CLIENT_H:
        return None
    return win


def refresh_game_window(hwnd: int, title: str | None = None) -> GameWindow:
    """重新讀取 client 幾何（視窗移動／縮放後必須更新）。"""
    if title is None:
        title = win32gui.GetWindowText(hwnd)
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    w, h = right - left, bottom - top
    sx, sy = win32gui.ClientToScreen(hwnd, (0, 0))
    return GameWindow(
        hwnd=hwnd,
        title=title,
        client_left=sx,
        client_top=sy,
        client_width=w,
        client_height=h,
    )


def focus_window(hwnd: int) -> None:
    """盡量把遊戲視窗拉到前景（含 AttachThreadInput）。"""
    try:
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        fg = win32gui.GetForegroundWindow()
        if fg == hwnd:
            return
        fg_thread = win32process.GetWindowThreadProcessId(fg)[0]
        cur_thread = win32api.GetCurrentThreadId()
        attached = False
        if fg_thread != cur_thread:
            try:
                win32api.AttachThreadInput(cur_thread, fg_thread, True)
                attached = True
            except Exception:
                pass
        win32gui.SetForegroundWindow(hwnd)
        win32gui.BringWindowToTop(hwnd)
        if attached:
            try:
                win32api.AttachThreadInput(cur_thread, fg_thread, False)
            except Exception:
                pass
    except win32gui.error:
        pass
