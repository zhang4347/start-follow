"""每整點把本機帳號餘額上傳到 Google 試算表（集中統計用）。

設計重點：
  - 跑在獨立背景執行緒，只「讀畫面」不點擊，完全不干擾下注流程。
  - 上傳時間「對齊主機系統時鐘的整點」（例如 01:00、02:00…），不是倒數。
  - 帳號名稱：優先用 config 的 sheet.account_name；留空才用 OCR 讀桌內帳號。
  - 用服務帳戶金鑰（service_account.json）連 Google Sheets，免使用者登入。
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta

import numpy as np
from mss import mss

from star_follow import paths
from star_follow.capture.window import find_game_window, refresh_game_window
from star_follow.config import AppConfig
from star_follow.vision.ocr import ocr_account_name, ocr_balance
from star_follow.vision.roi import scale_rect

logger = logging.getLogger(__name__)

_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
_HEADER = ["帳號", "餘額", "更新時間"]


def _capture_region_own(left: int, top: int, width: int, height: int) -> np.ndarray:
    from PIL import Image

    with mss() as sct:
        shot = sct.grab({"left": left, "top": top, "width": width, "height": height})
    return np.array(Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX"))


def _read_roi(cfg: AppConfig, roi_key: str):
    """擷取指定 roi 區塊影像。回傳 (影像 or None)。"""
    rect_ref = cfg.roi.get(roi_key)
    if not rect_ref:
        return None
    win = find_game_window(cfg.window.title_substring, title_aliases=cfg.window.title_aliases or None)
    if win is None:
        return None
    win = refresh_game_window(win.hwnd, win.title)
    x, y, w, h = scale_rect(
        rect_ref,
        cfg.window.reference_width,
        cfg.window.reference_height,
        win.client_width,
        win.client_height,
    )
    return _capture_region_own(win.client_left + x, win.client_top + y, w, h)


def read_balance_once(cfg: AppConfig) -> int:
    img = _read_roi(cfg, "balance")
    if img is None:
        return 0
    amount, _ = ocr_balance(img)
    return amount


def read_account_name_once(cfg: AppConfig) -> str:
    img = _read_roi(cfg, "account_name")
    if img is None:
        return ""
    name, _ = ocr_account_name(img)
    return name


def _seconds_to_next_hour() -> float:
    now = datetime.now()
    nxt = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return max(1.0, (nxt - now).total_seconds())


class SheetUploader:
    """背景執行緒：每整點讀餘額並上傳 Google 試算表。"""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.sh = cfg.sheet
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ws = None  # 快取的工作表物件

    def start(self) -> None:
        if not self.sh.enabled:
            return
        if not self.sh.spreadsheet_id:
            logger.warning("sheet 已啟用但缺 spreadsheet_id，餘額上傳不啟動")
            return
        key = paths.app_dir() / self.sh.service_account_file
        if not key.is_file():
            logger.warning("找不到服務帳戶金鑰檔 %s，餘額上傳不啟動", key)
            return
        self._thread = threading.Thread(target=self._run, name="SheetUploader", daemon=True)
        self._thread.start()
        logger.info("餘額上傳已啟動：每整點上傳到 Google 試算表（%s）", self.sh.spreadsheet_id)

    def stop(self) -> None:
        self._stop.set()

    def _worksheet(self):
        if self._ws is not None:
            return self._ws
        import gspread
        from google.oauth2.service_account import Credentials

        key = str(paths.app_dir() / self.sh.service_account_file)
        creds = Credentials.from_service_account_file(key, scopes=_SCOPES)
        gc = gspread.authorize(creds)
        book = gc.open_by_key(self.sh.spreadsheet_id)
        try:
            ws = book.worksheet(self.sh.worksheet)
        except Exception:  # noqa: BLE001  工作表不存在就建立
            ws = book.add_worksheet(title=self.sh.worksheet, rows=1000, cols=3)
            ws.append_row(_HEADER)
        self._ws = ws
        return ws

    def _upsert(self, ws, account: str, balance: int, ts: str) -> str:
        """一帳號一列：找到自己那列就覆蓋，找不到就新增。回傳動作描述。"""
        rows = ws.get_all_values()
        if not rows:  # 全空表（沒表頭）→ 先補表頭
            ws.append_row(_HEADER)
            rows = [_HEADER]
        for i, row in enumerate(rows):
            if i == 0:  # 表頭
                continue
            if row and row[0].strip() == account:
                r = i + 1  # 試算表列號（1-based）
                ws.update_cell(r, 2, balance)   # B 欄：餘額
                ws.update_cell(r, 3, ts)        # C 欄：更新時間
                return f"更新第 {r} 列"
        ws.append_row([account, balance, ts], value_input_option="USER_ENTERED")
        return "新增一列"

    def _resolve_account(self) -> str:
        if self.sh.account_name:
            return self.sh.account_name
        name = read_account_name_once(self.cfg)
        if name:
            logger.info("OCR 讀到帳號名：%s", name)
        return name or "未知帳號"

    def _upload_once(self) -> None:
        balance = read_balance_once(self.cfg)
        if balance <= 0:
            logger.info("餘額上傳：暫時讀不到餘額（視窗未開或 OCR 失敗），跳過本次")
            return
        account = self._resolve_account()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            ws = self._worksheet()
            action = self._upsert(ws, account, balance, ts)
            logger.info("餘額已上傳（%s）：帳號=%s 餘額=%s @ %s", action, account, f"{balance:,}", ts)
        except Exception as exc:  # noqa: BLE001
            logger.warning("餘額上傳失敗：%s（下次整點再試）", exc)
            self._ws = None  # 連線可能失效，下次重建

    def _wait_window_ready(self, tries: int = 20) -> None:
        for _ in range(tries):
            if self._stop.is_set():
                return
            if find_game_window(
                self.cfg.window.title_substring,
                title_aliases=self.cfg.window.title_aliases or None,
            ) is not None:
                return
            time.sleep(1.0)

    def _run(self) -> None:
        if self.sh.upload_on_start:
            self._wait_window_ready()
            self._upload_once()
        while not self._stop.is_set():
            wait_s = _seconds_to_next_hour()
            logger.info("下次餘額上傳於整點（約 %.0f 分鐘後）", wait_s / 60.0)
            if self._stop.wait(wait_s):
                return
            try:
                self._upload_once()
            except Exception as exc:  # noqa: BLE001
                logger.warning("餘額上傳迴圈例外：%s", exc)
