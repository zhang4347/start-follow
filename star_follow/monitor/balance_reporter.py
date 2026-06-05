"""定時讀取帳號餘額，透過 Telegram 回報，並與上次比較賺賠。

設計重點：
  - 跑在獨立背景執行緒，只「讀畫面」不點擊，完全不干擾下注流程。
  - 用自己的 mss 實例擷取（mss 非執行緒安全，不共用主程式的實例）。
  - 上次餘額存到 data/balance_state.json，重開程式也記得。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime

import numpy as np
from mss import mss

from star_follow import paths
from star_follow.config import AppConfig
from star_follow.capture.window import find_game_window, refresh_game_window
from star_follow.notify.telegram import send_message
from star_follow.vision.ocr import ocr_balance
from star_follow.vision.roi import scale_rect

logger = logging.getLogger(__name__)


def _capture_region_own(left: int, top: int, width: int, height: int) -> np.ndarray:
    """用獨立 mss 實例擷取（供背景執行緒安全使用）。"""
    from PIL import Image

    with mss() as sct:
        shot = sct.grab({"left": left, "top": top, "width": width, "height": height})
    return np.array(Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX"))


def read_balance_once(cfg: AppConfig) -> tuple[int, np.ndarray | None]:
    """擷取餘額區塊並 OCR。回傳 (金額, 截到的影像)；找不到視窗或讀不到回 (0, None/影像)。"""
    rect_ref = cfg.roi.get("balance")
    if not rect_ref:
        logger.warning("config 沒有 roi.balance，無法讀餘額")
        return 0, None
    win = find_game_window(cfg.window.title_substring, title_aliases=cfg.window.title_aliases or None)
    if win is None:
        return 0, None
    win = refresh_game_window(win.hwnd, win.title)
    x, y, w, h = scale_rect(
        rect_ref,
        cfg.window.reference_width,
        cfg.window.reference_height,
        win.client_width,
        win.client_height,
    )
    img = _capture_region_own(win.client_left + x, win.client_top + y, w, h)
    amount, _ = ocr_balance(img)
    return amount, img


def _load_state() -> dict:
    p = paths.balance_state_path()
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save_state(state: dict) -> None:
    p = paths.balance_state_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("寫入餘額狀態失敗：%s", exc)


def _format_report(now_bal: int, last_bal: int | None, last_ts: str | None) -> str:
    lines = ["💰 帳號餘額回報", f"目前餘額：{now_bal:,}"]
    if last_bal is None:
        lines.append("（這是啟動後第一次回報，作為基準）")
    else:
        diff = now_bal - last_bal
        if diff > 0:
            tag = f"+{diff:,}（賺 🟢）"
        elif diff < 0:
            tag = f"{diff:,}（虧 🔴）"
        else:
            tag = "±0（持平）"
        lines.append(f"與上次相比：{tag}")
        if last_ts:
            lines.append(f"（上次 {last_ts}）")
    lines.append(f"時間：{datetime.now().strftime('%m/%d %H:%M:%S')}")
    return "\n".join(lines)


class BalanceReporter:
    """背景執行緒：每隔 interval_min 分鐘讀餘額並回報 Telegram。"""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.tg = cfg.telegram
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.tg.enabled:
            return
        if not self.tg.bot_token or not self.tg.chat_id:
            logger.warning("Telegram 已啟用但缺 bot_token / chat_id，餘額回報不啟動")
            return
        self._thread = threading.Thread(target=self._run, name="BalanceReporter", daemon=True)
        self._thread.start()
        logger.info(
            "餘額回報已啟動：每 %s 分鐘一次（Telegram chat=%s）",
            self.tg.interval_min,
            self.tg.chat_id,
        )

    def stop(self) -> None:
        self._stop.set()

    def _report_once(self) -> None:
        # 不在牌桌內時餘額位置會顯示別的數字，OCR 會讀錯 → 本次跳過，等下個間隔再報。
        from star_follow.monitor.screen_gate import in_baccarat_room

        if in_baccarat_room(self.cfg) is not True:
            logger.info("餘額回報：目前不在牌桌內，暫不辨識餘額（下次再報）")
            return
        amount, _img = read_balance_once(self.cfg)
        if amount <= 0:
            logger.info("餘額回報：暫時讀不到餘額（視窗未開或 OCR 失敗），稍後再試")
            return
        state = _load_state()
        last_bal = state.get("last_balance")
        last_ts = state.get("last_ts")
        text = _format_report(amount, last_bal, last_ts)
        if send_message(self.tg.bot_token, self.tg.chat_id, text):
            logger.info("餘額回報已送出：%s（上次 %s）", f"{amount:,}", last_bal)
            state["last_balance"] = amount
            state["last_ts"] = datetime.now().strftime("%m/%d %H:%M:%S")
            _save_state(state)
        else:
            logger.warning("餘額回報送出失敗（不更新基準，下次再試）")

    def _run(self) -> None:
        # 啟動後先等視窗就緒並報一次基準
        if self.tg.report_on_start:
            # 給遊戲視窗/OCR 一點暖機時間
            for _ in range(20):
                if self._stop.is_set():
                    return
                if find_game_window(
                    self.cfg.window.title_substring,
                    title_aliases=self.cfg.window.title_aliases or None,
                ) is not None:
                    break
                time.sleep(1.0)
            self._report_once()

        interval_s = max(10.0, float(self.tg.interval_min) * 60.0)
        while not self._stop.wait(interval_s):
            try:
                self._report_once()
            except Exception as exc:  # noqa: BLE001
                logger.warning("餘額回報迴圈例外：%s", exc)
