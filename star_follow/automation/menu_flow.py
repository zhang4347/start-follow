"""開統計：有手動標記時就是固定兩下，不做自動偵測。"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from star_follow.automation.click import click_at
from star_follow.capture.window import GameWindow, focus_window
from star_follow.vision.panel import stats_table_visible

logger = logging.getLogger(__name__)

_CLICK_BACKENDS = ("win32", "direct", "postmessage")


def _stats_open(
    capture_fn: Callable[[], object],
    panel_rect: list[int],
    table_rect: list[int] | None,
    close_rect: list[int] | None,
) -> bool:
    """以統計表淺米色底判斷（最可靠且快）。

    輪詢迴圈每 80ms 會呼叫一次；改為純色彩判斷，避免在面板尚未渲染時
    反覆跑中文 OCR（每次數百 ms）拖慢開表。
    """
    frame = capture_fn()
    visible, _ = stats_table_visible(frame, table_rect)
    return visible


def wait_stats_open(
    capture_fn: Callable[[], object],
    panel_rect: list[int],
    table_rect: list[int] | None,
    close_rect: list[int] | None,
    *,
    max_wait_s: float = 2.0,
    interval_s: float = 0.08,
) -> bool:
    """統計表可能稍晚才渲染；輪詢直到偵測到或逾時。"""
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        if _stats_open(capture_fn, panel_rect, table_rect, close_rect):
            return True
        time.sleep(interval_s)
    return False


def open_stats_with_marks(
    win: GameWindow,
    menu_pt: tuple[int, int],
    chart_pt: tuple[int, int],
    *,
    panel_rect: list[int],
    table_rect: list[int] | None,
    close_rect: list[int] | None,
    capture_fn: Callable[[], object],
    menu_delay_s: float = 0.55,
    stats_open_wait_s: float = 2.5,
    backends: tuple[str, ...] = _CLICK_BACKENDS,
) -> tuple[bool, str | None]:
    """
    固定流程：點 ☰ → 等 → 點柱狀圖 → 輪詢統計表。
    一旦偵測到開啟就立刻回傳，不再試下一個 backend（避免再點 ☰ 把表關掉）。
    """
    mx, my = menu_pt
    cx, cy = chart_pt
    focus_window(win.hwnd)
    time.sleep(0.15)

    for backend in backends:
        logger.info(
            "開統計：☰(%d,%d) → 柱狀圖(%d,%d) backend=%s",
            mx, my, cx, cy, backend,
        )
        click_at(win, mx, my, backend=backend)
        time.sleep(menu_delay_s)
        click_at(win, cx, cy, backend=backend)
        if wait_stats_open(
            capture_fn, panel_rect, table_rect, close_rect,
            max_wait_s=stats_open_wait_s,
            interval_s=0.08,
        ):
            logger.info("統計表已偵測到開啟 backend=%s", backend)
            return True, backend

    return False, None


def retry_stats_chart_click(
    win: GameWindow,
    chart_pt: tuple[int, int],
    *,
    panel_rect: list[int],
    table_rect: list[int] | None,
    close_rect: list[int] | None,
    capture_fn: Callable[[], object],
    backend: str = "win32",
    max_wait_s: float = 2.5,
) -> bool:
    """選單已展開但統計表未開時，只補點柱狀圖。"""
    cx, cy = chart_pt
    focus_window(win.hwnd)
    logger.info("補點柱狀圖 (%d,%d)", cx, cy)
    click_at(win, cx, cy, backend=backend)
    return wait_stats_open(
        capture_fn,
        panel_rect,
        table_rect,
        close_rect,
        max_wait_s=max_wait_s,
        interval_s=0.08,
    )
