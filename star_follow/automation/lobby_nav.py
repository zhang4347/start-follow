"""被踢回大廳時，自動導覽回到百家樂桌。

畫面狀態與動作（偵測現在在哪一頁 → 做對應的一步，迴圈到坐進牌桌）：
  1) 已在牌桌            → 完成（呼叫端決定要不要再 goto 指定桌號）
  2) 百家樂入口          → 有「隨機選台」鈕 → 點它隨機進桌
  3) 棋牌大廳            → 看得到百家樂卡片 → 點卡片
  4) 其他大廳/棋牌大廳   → 看不到百家樂卡片 → 點右側「棋牌」選單，
     但卡片在最右                再往右滑（拖曳）直到百家樂卡片出現

模板（vision/templates）：
  lobby_random_select.png  入口頁「隨機選台」鈕
  lobby_baccarat_card.png  棋牌大廳的百家樂卡片（閒/莊撲克牌圖）
  lobby_qipai_menu.png     右側選單「棋牌」鈕
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np

from star_follow.automation.click import click_at, drag_client
from star_follow.capture.screen import capture_client
from star_follow.capture.window import GameWindow, focus_window
from star_follow.config import AppConfig
from star_follow.vision.game_detect import is_baccarat_table
from star_follow.vision.menu_match import match_template_in_region

logger = logging.getLogger(__name__)

_T_RANDOM = "lobby_random_select.png"
_T_CARD = "lobby_baccarat_card.png"
_T_QIPAI = "lobby_qipai_menu.png"
_T_CONFIRM = "lobby_confirm_button.png"  # 「超過五局未押注，已退出」提示的「確定」鈕

# 提示對話框的「確定」鈕大致落在畫面中央偏下；只在這一帶找。
_RECT_CONFIRM = (380, 370, 520, 290)

# 固定位置（1280×720 client，遊戲視窗大小固定）。每一輪只在這些小範圍比對，
# 避免整張畫面多尺度比對拖慢下注時序。百家樂卡片會隨橫向捲動位移，所以卡片
# 用整張畫面找，但「卡片」只在回桌導覽時才搜尋（不在每輪檢查）。
_RECT_RANDOM = (985, 580, 295, 140)   # 入口「隨機選台」鈕
_RECT_QIPAI = (1130, 335, 150, 95)    # 右側選單「棋牌」鈕


def _full_rect(win: GameWindow) -> tuple[int, int, int, int]:
    return (0, 0, win.client_width, win.client_height)


def _find(frame: np.ndarray, win: GameWindow, name: str, thr: float) -> tuple[int, int, float] | None:
    return match_template_in_region(frame, name, _full_rect(win), threshold=thr)


def _find_rect(
    frame: np.ndarray, name: str, rect: tuple[int, int, int, int], thr: float
) -> tuple[int, int, float] | None:
    return match_template_in_region(frame, name, rect, threshold=thr)


def dismiss_popup_if_any(
    win: GameWindow, cfg: AppConfig, capture_fn: Callable[[], np.ndarray]
) -> bool:
    """若畫面中央有「確定」提示鈕（例：超過五局未押注已退出），點掉它。回傳是否有點。"""
    frame = capture_fn()
    hit = _find_rect(frame, _T_CONFIRM, _RECT_CONFIRM, 0.6)
    if hit is None:
        return False
    logger.info("偵測到提示對話框，點『確定』(%d,%d) 關閉", hit[0], hit[1])
    click_at(win, hit[0], hit[1], backend=cfg.automation.ui_click_backend)
    time.sleep(0.6)
    return True


def screen_state_fast(frame: np.ndarray, cfg: AppConfig, win: GameWindow) -> str:
    """每輪用的輕量判斷：只在固定小範圍比對『隨機選台』與『棋牌』鈕。

    回傳 'table' | 'entry' | 'lobby' | 'unknown'。
    （不細分棋牌大廳，因為對「是否被踢出」的判斷而言，entry/lobby 都代表要回桌。）
    """
    # 提示對話框（被踢退出）也算「要回桌」，且要優先處理，否則背景的彩色卡片列
    # 可能讓 is_baccarat_table 誤判成牌桌而不去關掉它。
    if _find_rect(frame, _T_CONFIRM, _RECT_CONFIRM, 0.6) is not None:
        return "lobby"
    if _find_rect(frame, _T_RANDOM, _RECT_RANDOM, 0.6) is not None:
        return "entry"
    if _find_rect(frame, _T_QIPAI, _RECT_QIPAI, 0.55) is not None:
        return "lobby"
    if is_baccarat_table(frame, cfg):
        return "table"
    return "unknown"


def detect_screen(frame: np.ndarray, cfg: AppConfig, win: GameWindow) -> str:
    """回傳 'table' | 'entry' | 'qipai' | 'lobby' | 'unknown'（回桌導覽用，含整張找卡片）。

    先比對大廳/入口的專屬模板（分數很高、很可靠），因為 is_baccarat_table 會被
    大廳底部一整排彩色遊戲卡誤判成牌桌；只有都不像大廳時，才當作牌桌判斷。
    """
    if _find_rect(frame, _T_RANDOM, _RECT_RANDOM, 0.6) is not None:
        return "entry"
    if _find(frame, win, _T_CARD, 0.62) is not None:
        return "qipai"
    if _find_rect(frame, _T_QIPAI, _RECT_QIPAI, 0.55) is not None:
        return "lobby"
    if is_baccarat_table(frame, cfg):
        return "table"
    return "unknown"


def _scroll_right_find_card(
    win: GameWindow, cfg: AppConfig, capture_fn: Callable[[], np.ndarray], *, max_drags: int = 8
) -> tuple[int, int, float] | None:
    """在棋牌大廳往右滑（拖曳），直到百家樂卡片出現；回傳卡片中心或 None。"""
    w, h = win.client_width, win.client_height
    y = int(h * 0.42)
    x_from, x_to = int(w * 0.72), int(w * 0.24)  # 把畫面往左拖 → 露出右邊
    prev_hit = -1.0
    for _ in range(max_drags):
        frame = capture_fn()
        card = _find(frame, win, _T_CARD, 0.62)
        if card is not None:
            return card
        # 也可能滑過頭直接到入口；交給上層 loop 處理
        if _find(frame, win, _T_RANDOM, 0.6) is not None:
            return None
        drag_client(win, x_from, y, x_to, y)
        time.sleep(0.35)
    # 最後再讀一次
    return _find(capture_fn(), win, _T_CARD, 0.62)


def return_to_baccarat_table(
    win: GameWindow,
    cfg: AppConfig,
    capture_fn: Callable[[], np.ndarray] | None = None,
    *,
    timeout_s: float = 90.0,
) -> bool:
    """從大廳/入口自動導覽回到「坐進某張百家樂桌」。成功回傳 True。

    不負責換到特定桌號（坐進任一桌即算成功）；掛桌要回固定桌號，由呼叫端在成功後
    再用 room_nav.switch_to_table(target) 達成。
    """
    cap = capture_fn or (lambda: capture_client(win))
    try:
        focus_window(win.hwnd)
    except Exception:
        pass
    deadline = time.monotonic() + timeout_s
    last_action = ""
    stuck = 0
    while time.monotonic() < deadline:
        # 先關掉可能擋住的提示對話框（例：超過五局未押注已退出），否則點不到底下的按鈕
        if dismiss_popup_if_any(win, cfg, cap):
            continue
        frame = cap()
        state = detect_screen(frame, cfg, win)
        if state == "table":
            logger.info("已回到牌桌")
            return True
        if state == "entry":
            hit = _find(frame, win, _T_RANDOM, 0.6)
            if hit:
                logger.info("百家樂入口 → 點『隨機選台』(%d,%d)", hit[0], hit[1])
                click_at(win, hit[0], hit[1], backend=cfg.automation.ui_click_backend)
                time.sleep(2.0)
                last_action = "entry"
                continue
        if state == "qipai":
            card = _find(frame, win, _T_CARD, 0.62)
            if card:
                logger.info("棋牌大廳 → 點百家樂卡片 (%d,%d)", card[0], card[1])
                click_at(win, card[0], card[1], backend=cfg.automation.ui_click_backend)
                time.sleep(1.8)
                last_action = "qipai"
                continue
        # lobby / unknown：先點右側「棋牌」進棋牌大廳，再往右滑找百家樂卡片
        menu = _find(frame, win, _T_QIPAI, 0.55)
        if menu:
            logger.info("大廳 → 點右側『棋牌』(%d,%d)", menu[0], menu[1])
            click_at(win, menu[0], menu[1], backend=cfg.automation.ui_click_backend)
            time.sleep(1.2)
            card = _scroll_right_find_card(win, cfg, cap)
            if card:
                logger.info("往右滑後找到百家樂卡片 (%d,%d)，點入", card[0], card[1])
                click_at(win, card[0], card[1], backend=cfg.automation.ui_click_backend)
                time.sleep(1.8)
            last_action = "lobby"
            continue
        # 完全認不出畫面：可能在過場/載入，稍等再看
        stuck += 1
        if stuck % 5 == 0:
            logger.info("回桌：暫時認不出畫面（state=%s, last=%s），等待…", state, last_action)
        time.sleep(0.8)
    logger.warning("回桌逾時（%.0fs）", timeout_s)
    return False
