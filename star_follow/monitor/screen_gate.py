"""背景監控（餘額回報／上傳）共用的「是否在牌桌內」判斷。

餘額 ROI 是相對牌桌版面定義的；不在房內（大廳／百家樂入口／棋牌大廳）時，該位置
可能顯示別的數字，OCR 會讀成錯誤餘額。故上傳／回報前先用此 gate 擋掉非牌桌畫面。

只「讀畫面」不點擊，並用自己的 mss 實例（mss 非執行緒安全，不共用主程式實例）。
"""

from __future__ import annotations

import logging

import numpy as np
from mss import mss

from star_follow.capture.window import find_game_window, refresh_game_window
from star_follow.config import AppConfig

logger = logging.getLogger(__name__)


def in_baccarat_room(cfg: AppConfig) -> bool | None:
    """目前是否在百家樂牌桌內。

    回傳 True／False；找不到遊戲視窗或判斷時出例外回 None（呼叫端視為「不確定」，
    通常與 False 一樣不上傳，等下次重試）。
    """
    win = find_game_window(
        cfg.window.title_substring, title_aliases=cfg.window.title_aliases or None
    )
    if win is None:
        return None
    try:
        win = refresh_game_window(win.hwnd, win.title)
        from PIL import Image

        with mss() as sct:
            shot = sct.grab(
                {
                    "left": win.client_left,
                    "top": win.client_top,
                    "width": win.client_width,
                    "height": win.client_height,
                }
            )
        frame = np.array(Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX"))
        from star_follow.vision.game_detect import detect_in_baccarat_room

        ok, _meta = detect_in_baccarat_room(frame, cfg, win)
        return bool(ok)
    except Exception as exc:  # noqa: BLE001
        logger.debug("在房判斷失敗：%s", exc)
        return None
