"""診斷滑鼠 / PostMessage 點擊。用法: python -m star_follow.tools.test_mouse"""

from __future__ import annotations

import time

from star_follow.automation.click import (
    click_at,
    get_menu_click_backend,
    get_ui_click_backend,
    move_to_client,
    screen_point_from_client,
)
from star_follow.capture.dpi import ensure_dpi_aware, get_cursor_pos, virtual_screen
from star_follow.capture.screen import capture_client
from star_follow.capture.window import find_game_window, focus_window
from star_follow.config import load_config
from star_follow.vision.menu_match import menu_dropdown_open
from star_follow.vision.roi import scale_rect

ensure_dpi_aware()


def main() -> int:
    cfg = load_config()
    win = find_game_window(cfg.window.title_substring)
    if not win:
        print("找不到遊戲視窗")
        return 1

    menu_btn = scale_rect(
        cfg.roi["menu_button"],
        cfg.window.reference_width,
        cfg.window.reference_height,
        win.client_width,
        win.client_height,
    )
    menu_panel = scale_rect(
        cfg.roi["menu_panel"],
        cfg.window.reference_width,
        cfg.window.reference_height,
        win.client_width,
        win.client_height,
    )
    cx = menu_btn[0] + menu_btn[2] // 2
    cy = menu_btn[1] + menu_btn[3] // 2
    sx, sy = screen_point_from_client(win, cx, cy)
    vs = virtual_screen()
    menu_b = get_menu_click_backend()
    ui_b = get_ui_click_backend()

    print("=== 滑鼠 / 點擊診斷 ===")
    print(f"menu_click={menu_b}  ui_click={ui_b}")
    print(f"遊戲: {win.title}")
    print(f"client {win.client_width}×{win.client_height} @ 螢幕 ({win.client_left},{win.client_top})")
    print(f"虛擬螢幕 origin=({vs[0]},{vs[1]}) size={vs[2]}×{vs[3]}")
    print(f"☰ client=({cx},{cy}) → 螢幕=({sx},{sy})")
    print(f"目前滑鼠: {get_cursor_pos()}")
    print()

    focus_window(win.hwnd)
    time.sleep(0.3)
    frame0 = capture_client(win)
    menu_before = menu_dropdown_open(frame0, menu_btn)

    print("① 測試滑鼠移動（3 秒後）…")
    time.sleep(3)
    before = get_cursor_pos()
    move_to_client(win, cx, cy)
    after = get_cursor_pos()
    moved = abs(after[0] - sx) <= 10 and abs(after[1] - sy) <= 10
    print(f"   移動前 {before} → 移動後 {after}，到位={moved}")

    print(f"\n② PostMessage/SendMessage 點擊 ☰（backend={menu_b}）…")
    time.sleep(1)
    click_at(win, cx, cy, backend=menu_b)
    time.sleep(0.5)
    frame1 = capture_client(win)
    menu_after = menu_dropdown_open(frame1, menu_btn)
    print(f"   選單展開: {menu_before} → {menu_after}")

    if menu_after and not menu_before:
        print("\n成功：PostMessage 可開選單。請用管理員 PowerShell 跑 test_menu（柱狀圖用 win32 真點擊）")
        return 0

    if moved:
        print("\n滑鼠可動但選單沒開 — 座標可能不準，或遊戲不吃 SendMessage")
    else:
        print("\n滑鼠被擋（常見：遊戲用管理員執行）。")
        print("若選單仍沒開，請以系統管理員開 PowerShell 再試，或回報結果。")
    return 2 if not menu_after else 0


if __name__ == "__main__":
    raise SystemExit(main())
