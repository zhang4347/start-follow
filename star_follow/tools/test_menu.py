"""測試：☰ → 柱狀圖 → 押注統計。用法: python -m star_follow.tools.test_menu [--move-only]"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
from PIL import Image

from star_follow.automation.click import move_to_client, screen_point_from_client
from star_follow.automation.menu_flow import open_stats_with_marks
from star_follow.capture.dpi import get_cursor_pos
from star_follow.capture.screen import capture_client
from star_follow.capture.window import find_game_window, focus_window, refresh_game_window
from star_follow.config import load_config
from star_follow.vision.panel import stats_panel_debug
from star_follow.vision.roi import scale_point, scale_rect

LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def _scale(cfg, win, key: str) -> tuple[int, int, int, int]:
    r = cfg.roi[key]
    return scale_rect(
        r,
        cfg.window.reference_width,
        cfg.window.reference_height,
        win.client_width,
        win.client_height,
    )


def _scale_pt(cfg, win, key: str) -> tuple[int, int] | None:
    pt = cfg.click_points.get(key)
    if not pt or len(pt) != 2:
        return None
    return scale_point(
        pt,
        cfg.window.reference_width,
        cfg.window.reference_height,
        win.client_width,
        win.client_height,
    )


def _panel_args(cfg, win):
    panel = list(_scale(cfg, win, "stats_panel"))
    table = list(_scale(cfg, win, "stats_table")) if "stats_table" in cfg.roi else None
    close = list(_scale(cfg, win, "stats_close")) if "stats_close" in cfg.roi else None
    return panel, table, close


def _save_markers(frame, menu_pt, chart_pt, path: Path) -> None:
    img = frame.copy()
    for pt, color, label in (
        (menu_pt, (0, 180, 255), "menu"),
        (chart_pt, (0, 255, 0), "chart"),
    ):
        if pt:
            cv2.circle(img, pt, 12, color, 2)
            cv2.putText(img, label, (pt[0] - 20, pt[1] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    Image.fromarray(img).save(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--move-only", action="store_true", help="只移滑鼠到柱狀圖（確認位置）")
    parser.add_argument("--chart-at", nargs=2, type=int, metavar=("X", "Y"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    win = find_game_window(cfg.window.title_substring)
    if not win:
        print("找不到遊戲視窗")
        return 1

    print(f"視窗 client {win.client_width}×{win.client_height}")
    print("固定座標模式：有 mark_menu 標記就只點那兩個位置（請用管理員 PowerShell）")

    focus_window(win.hwnd)
    time.sleep(0.3)
    frame = capture_client(win)
    panel, table, close = _panel_args(cfg, win)
    if stats_panel_debug(frame, panel, table_rect=table, close_rect=close)["open"]:
        print("統計表已是開啟狀態")
        return 0

    menu_pt = _scale_pt(cfg, win, "menu_button")
    chart_pt = tuple(args.chart_at) if args.chart_at else _scale_pt(cfg, win, "menu_chart")

    if not chart_pt:
        print("請先標記：python -m star_follow.tools.mark_menu")
        return 2
    if not menu_pt:
        print("config 缺少 menu_button，請跑 mark_menu 完整兩步")
        return 2

    print(f"☰ client={menu_pt}  柱狀圖 client={chart_pt}")

    if args.move_only:
        sx, sy = screen_point_from_client(win, chart_pt[0], chart_pt[1])
        print(f"5 秒後移到柱狀圖 screen=({sx},{sy})…")
        time.sleep(5)
        move_to_client(win, chart_pt[0], chart_pt[1])
        print(f"滑鼠 {get_cursor_pos()}")
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    _save_markers(frame, menu_pt, chart_pt, LOG_DIR / f"menu_marks_{ts}.png")
    print(f"標記位置圖 logs/menu_marks_{ts}.png")

    ok, backend = open_stats_with_marks(
        win,
        menu_pt,
        chart_pt,
        panel_rect=panel,
        table_rect=table,
        close_rect=close,
        capture_fn=lambda: capture_client(win),
        backends=("win32",),
    )
    if ok:
        print(f"統計表開啟: True（backend={backend}）")
        return 0

    print("統計表開啟: False")
    print("若你剛才看到統計表有閃一下：代表座標正確，可能是 OCR 來不及判斷。")
    print("請再跑一次；若仍 False 把畫面狀況告訴我。")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
