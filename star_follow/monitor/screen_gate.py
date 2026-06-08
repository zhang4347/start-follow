"""背景監控（餘額回報／上傳）共用的「是否在牌桌內」判斷。

餘額 ROI 是相對牌桌版面定義的；不在房內（大廳／百家樂入口／棋牌大廳）時，該位置
可能顯示別的數字，OCR 會讀成錯誤餘額。故上傳／回報前先用此 gate 擋掉非牌桌畫面。

只「讀畫面」不點擊，並用自己的 mss 實例（mss 非執行緒安全，不共用主程式實例）。
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable

import numpy as np
from mss import mss

from star_follow.capture.window import find_game_window, refresh_game_window
from star_follow.config import AppConfig

logger = logging.getLogger(__name__)


def _capture_client_frame(cfg: AppConfig) -> np.ndarray | None:
    """用自己的 mss 實例擷取整個遊戲 client 畫面（背景執行緒安全）。找不到視窗回 None。"""
    win = find_game_window(
        cfg.window.title_substring, title_aliases=cfg.window.title_aliases or None
    )
    if win is None:
        return None
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
    return np.array(Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX"))


def _balance_crop(frame: np.ndarray, cfg: AppConfig) -> np.ndarray | None:
    """從整張畫面裁出餘額 ROI。"""
    from star_follow.vision.roi import scale_rect

    rect_ref = cfg.roi.get("balance")
    if not rect_ref:
        return None
    h, w = frame.shape[:2]
    x, y, bw, bh = scale_rect(
        list(rect_ref),
        cfg.window.reference_width,
        cfg.window.reference_height,
        w,
        h,
    )
    crop = frame[y : y + bh, x : x + bw]
    return crop if crop.size else None


def _prune_dir(d, keep: int) -> None:
    try:
        files = sorted(d.glob("*.png"), key=lambda p: p.stat().st_mtime)
        for p in files[: max(0, len(files) - keep)]:
            p.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def _save_balance_debug(
    crops: list[tuple[int, np.ndarray | None]],
    frame: np.ndarray | None,
    chosen: int,
) -> None:
    """把這批餘額讀取的每張裁切圖（檔名標 OCR 值）＋一張全畫面存到 logs/balance_debug，
    供事後診斷餘額讀錯的原因（框位/壓到逗號/字體糊/5-8-9 混淆）。自動只留最近數百張。"""
    try:
        from PIL import Image

        from star_follow.paths import logs_dir

        d = logs_dir() / "balance_debug"
        d.mkdir(parents=True, exist_ok=True)
        _prune_dir(d, keep=400)
        ts = time.strftime("%Y%m%d_%H%M%S")
        for i, (v, crop) in enumerate(crops):
            if crop is None or not getattr(crop, "size", 0):
                continue
            tag = "chosen" if (v == chosen and chosen > 0) else "ocr"
            Image.fromarray(np.asarray(crop)).save(
                str(d / f"{ts}_{i}_{tag}-{v}.png")
            )
        if frame is not None and getattr(frame, "size", 0):
            Image.fromarray(np.asarray(frame)).save(
                str(d / f"{ts}_full_chosen-{chosen}.png")
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("存餘額診斷圖失敗：%s", exc)


def stable_balance(
    read_once: Callable[[], int] | None = None,
    *,
    cfg: AppConfig | None = None,
    samples: int = 5,
    min_agree: int = 3,
    debug: bool = True,
) -> int:
    """餘額多次讀取取多數決，過濾偶發辨識誤差（5/8/9 看錯、逗號誤讀成多/少一位）。

    餘額在整點當下是靜止的，正確值會在多次讀取中勝出；偶發錯誤被洗掉。
    讀 samples 次，取出現最多的非零值；達到 min_agree 次一致才採信，否則回 0
    （視為這次讀取不可靠，交由整點補上傳機制稍後重試）。

    用法：
      - 傳 cfg：自己擷取畫面＋裁切＋OCR，並在 debug=True 時存診斷圖（推薦，可蒐集資料）。
      - 傳 read_once（相容舊用法）：回傳單次餘額整數的 callback，不存診斷圖。
    """
    vals: list[int] = []
    crops: list[tuple[int, np.ndarray | None]] = []
    first_frame: np.ndarray | None = None

    for i in range(max(1, samples)):
        v = 0
        crop = None
        try:
            if cfg is not None:
                frame = _capture_client_frame(cfg)
                if frame is not None:
                    if first_frame is None:
                        first_frame = frame
                    crop = _balance_crop(frame, cfg)
                    if crop is not None:
                        from star_follow.vision.ocr import ocr_balance

                        v, _ = ocr_balance(crop)
            elif read_once is not None:
                v = read_once()
        except Exception as exc:  # noqa: BLE001
            logger.debug("餘額單次讀取例外：%s", exc)
            v = 0
        if v > 0:
            vals.append(v)
        crops.append((v, crop))
        if i + 1 < samples:
            time.sleep(0.12)

    chosen = 0
    if vals:
        val, cnt = Counter(vals).most_common(1)[0]
        if cnt >= min_agree:
            chosen = int(val)
        else:
            logger.info(
                "餘額多次讀取無共識（樣本=%s，最高一致=%d<%d），本次不採信、稍後重試",
                vals, cnt, min_agree,
            )
    if cfg is not None and debug:
        logger.info("餘額讀取樣本=%s → 採用=%s（診斷圖存 logs\\balance_debug）", vals, chosen)
        _save_balance_debug(crops, first_frame, chosen)
    return chosen


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
