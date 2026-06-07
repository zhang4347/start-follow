"""背景監控（餘額回報／上傳）共用的「是否在牌桌內」判斷。

餘額 ROI 是相對牌桌版面定義的；不在房內（大廳／百家樂入口／棋牌大廳）時，該位置
可能顯示別的數字，OCR 會讀成錯誤餘額。故上傳／回報前先用此 gate 擋掉非牌桌畫面。

只「讀畫面」不點擊，並用自己的 mss 實例（mss 非執行緒安全，不共用主程式實例）。
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable

import numpy as np
from mss import mss

from star_follow.capture.window import find_game_window, refresh_game_window
from star_follow.config import AppConfig

logger = logging.getLogger(__name__)


def stable_balance(
    read_once: Callable[[], int],
    *,
    samples: int = 5,
    min_agree: int = 3,
) -> int:
    """餘額多次讀取取多數決，過濾偶發辨識誤差（5/8/9 看錯、逗號誤讀成多/少一位）。

    餘額在整點當下是靜止的，正確值會在多次讀取中勝出；偶發錯誤被洗掉。
    讀 samples 次，取出現最多的非零值；達到 min_agree 次一致才採信，否則回 0
    （視為這次讀取不可靠，交由整點補上傳機制稍後重試）。
    """
    import time

    vals: list[int] = []
    for i in range(max(1, samples)):
        try:
            v = read_once()
        except Exception as exc:  # noqa: BLE001
            logger.debug("餘額單次讀取例外：%s", exc)
            v = 0
        if v > 0:
            vals.append(v)
        if i + 1 < samples:
            time.sleep(0.12)
    if not vals:
        return 0
    val, cnt = Counter(vals).most_common(1)[0]
    if cnt < min_agree:
        logger.info(
            "餘額多次讀取無共識（樣本=%s，最高一致=%d<%d），本次不採信、稍後重試",
            vals, cnt, min_agree,
        )
        return 0
    return int(val)


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
