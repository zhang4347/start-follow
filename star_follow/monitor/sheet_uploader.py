"""每個整點把本機帳號餘額上傳到 Google 試算表（集中統計用）。

設計重點：
  - 跑在獨立背景執行緒，只「讀畫面」不點擊，完全不干擾下注流程。
  - 上傳對齊主機系統時鐘的整點，但「不必剛好在整點」：到了某整點若一時上傳不成
    （在辨識別的東西／視窗沒開／OCR 讀不到），會每隔一小段時間補上傳，直到成功；
    成功後該整點不再重複，等下一個整點。
  - 帳號名稱：優先用 config 的 sheet.account_name（由啟動設定手動輸入）；留空才退而用 OCR。
  - 用服務帳戶金鑰（service_account.json）連 Google Sheets，免使用者登入。
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

_RETRY_INTERVAL_S = 25.0  # 輪詢／補上傳間隔（秒）：越小越貼近整點、失敗補上傳越快

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


class SheetUploader:
    """背景執行緒：每個整點讀餘額並上傳 Google 試算表（含補上傳）。"""

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
        logger.info("餘額上傳已啟動：每個整點上傳到 Google 試算表（含補上傳，%s）", self.sh.spreadsheet_id)

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

    def _upload_once(self) -> bool:
        """讀餘額並上傳。成功回傳 True；讀不到餘額或上傳失敗回傳 False（之後會重試補上傳）。"""
        # 不在牌桌內時餘額位置會顯示別的數字，OCR 會讀錯 → 不上傳，等下次補上傳。
        from star_follow.monitor.screen_gate import in_baccarat_room

        if in_baccarat_room(self.cfg) is not True:
            logger.info("餘額上傳：目前不在牌桌內，暫不辨識餘額（稍後重試）")
            return False
        from star_follow.monitor.screen_gate import stable_balance

        balance = stable_balance(lambda: read_balance_once(self.cfg))
        if balance <= 0:
            logger.info("餘額上傳：暫時讀不到餘額（視窗未開或 OCR 失敗），稍後重試")
            return False
        account = self._resolve_account()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            ws = self._worksheet()
            action = self._upsert(ws, account, balance, ts)
            logger.info("餘額已上傳（%s）：帳號=%s 餘額=%s @ %s", action, account, f"{balance:,}", ts)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("餘額上傳失敗：%s（稍後重試）", exc)
            self._ws = None  # 連線可能失效，下次重建
            return False

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
        """每個整點各上傳一次，但「不必剛好在整點」：

        到了某個整點後，只要當下還沒上傳成功（可能剛好在辨識別的東西、或視窗沒開、
        OCR 一時讀不到），就每隔一小段時間重試，直到該整點的餘額成功補上傳為止；
        成功後該整點就不再重複上傳，等下一個整點。
        """
        uploaded_hour: datetime | None = None  # 已成功上傳的「整點」

        if self.sh.upload_on_start:
            self._wait_window_ready()
            if self._upload_once():
                uploaded_hour = self._current_hour()

        while not self._stop.is_set():
            try:
                cur_hour = self._current_hour()
                if uploaded_hour != cur_hour:
                    # 這個整點還沒成功上傳 → 嘗試（含補上傳：整點過了也照上）
                    if self._upload_once():
                        uploaded_hour = cur_hour
                    else:
                        logger.info("本整點（%s）尚未成功上傳，稍後重試", cur_hour.strftime("%H:%M"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("餘額上傳迴圈例外：%s", exc)
            # 短間隔輪詢：盡量貼近整點，且失敗時能很快補上傳
            if self._stop.wait(_RETRY_INTERVAL_S):
                return

    @staticmethod
    def _current_hour() -> datetime:
        return datetime.now().replace(minute=0, second=0, microsecond=0)
