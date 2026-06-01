"""換房（巡房）導覽：開桌號清單、OCR 桌號、前往指定桌、確認目前桌號。

座標來源（config，皆為參考解析度，執行時自動依視窗大小縮放）：
  click_points.room_switch_button  開/關桌號清單的圖示
  click_points.room_list_scroll    清單內捲動點
  roi.room_no_col                  清單左側桌號數字欄（OCR 桌號 + 取列 y）
  roi.room_current_table           左下角目前桌號
  room.goto_x                      「前往」鈕欄位中心 x
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable

import cv2
import numpy as np
import pytesseract

from star_follow.automation.click import click_at, wheel_at_client
from star_follow.capture.screen import capture_client
from star_follow.capture.window import GameWindow, focus_window
from star_follow.config import AppConfig
from star_follow.vision.roi import scale_point, scale_rect

logger = logging.getLogger(__name__)

_DIGITS = re.compile(r"\d+")


def _save_debug(frame: np.ndarray, cfg: AppConfig, win: GameWindow, name: str) -> None:
    """把換房失敗當下的畫面存到 logs/room_debug，供校正清單 ROI 用。"""
    try:
        from star_follow.paths import logs_dir

        d = logs_dir() / "room_debug"
        d.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%H%M%S")
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(d / f"{ts}_{name}.png"), bgr)
        # 同時把清單桌號欄裁切另存，方便確認 OCR 看到什麼
        rect = _scaled_rect(cfg, win, "room_no_col")
        if rect is not None:
            x, y, w, h = rect
            crop = bgr[y : y + h, x : x + w]
            if crop.size:
                cv2.imwrite(str(d / f"{ts}_{name}_nocol.png"), crop)
        logger.info("已存換房診斷圖：logs\\room_debug\\%s_%s.png", ts, name)
    except Exception:  # noqa: BLE001
        pass


def _scaled_rect(cfg: AppConfig, win: GameWindow, key: str) -> tuple[int, int, int, int] | None:
    r = cfg.roi.get(key)
    if not r or len(r) != 4:
        return None
    return scale_rect(
        list(r), cfg.window.reference_width, cfg.window.reference_height,
        win.client_width, win.client_height,
    )


def _scaled_point(cfg: AppConfig, win: GameWindow, key: str) -> tuple[int, int] | None:
    p = cfg.click_points.get(key)
    if not p or len(p) != 2:
        return None
    return scale_point(
        list(p), cfg.window.reference_width, cfg.window.reference_height,
        win.client_width, win.client_height,
    )


def _scaled_goto_x(cfg: AppConfig, win: GameWindow) -> int:
    return int(round(cfg.room.goto_x * win.client_width / max(1, cfg.window.reference_width)))


_OCR_SCALE = 3.0


def _ocr_numbers_with_y(crop: np.ndarray) -> list[tuple[int, int]]:
    """回傳 [(數字, crop 內 y 中心), ...]。

    用 3x 放大 + 兩種 PSM（11 稀疏 / 6 區塊）合併，提高小數字辨識率。
    """
    if crop.size == 0:
        return []
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=_OCR_SCALE, fy=_OCR_SCALE, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(th.mean()) < 127:
        th = cv2.bitwise_not(th)
    out: list[tuple[int, int]] = []
    for psm in (11, 6):
        config = f"--psm {psm} -c tessedit_char_whitelist=0123456789"
        try:
            data = pytesseract.image_to_data(th, config=config, output_type=pytesseract.Output.DICT)
        except pytesseract.TesseractError:
            continue
        n = len(data.get("text", []))
        for i in range(n):
            txt = (data["text"][i] or "").strip()
            if not _DIGITS.fullmatch(txt):
                continue
            try:
                val = int(txt)
            except ValueError:
                continue
            y_center = data["top"][i] + data["height"][i] // 2
            out.append((val, int(y_center / _OCR_SCALE)))  # 還原縮放
    return out


def read_list_rows(frame: np.ndarray, cfg: AppConfig, win: GameWindow) -> list[tuple[int, int]]:
    """OCR 清單左側桌號欄，回傳 [(桌號, client_y), ...]，依 y 排序、去除過近重複。"""
    rect = _scaled_rect(cfg, win, "room_no_col")
    if rect is None:
        return []
    x, y, w, h = rect
    crop = frame[y : y + h, x : x + w]
    raw = _ocr_numbers_with_y(crop)
    valid = set(cfg.room.tables)
    rows: list[tuple[int, int]] = []
    for no, cy in sorted(raw, key=lambda t: t[1]):
        if no not in valid:
            continue
        if rows and abs((y + cy) - rows[-1][1]) < 18:
            continue
        rows.append((no, y + cy))
    return rows


def read_current_table(frame: np.ndarray, cfg: AppConfig, win: GameWindow) -> int | None:
    rect = _scaled_rect(cfg, win, "room_current_table")
    if rect is None:
        return None
    x, y, w, h = rect
    crop = frame[y : y + h, x : x + w]
    nums = _ocr_numbers_with_y(crop)
    valid = set(cfg.room.tables)
    for no, _cy in nums:
        if no in valid:
            return no
    return None


def list_is_open(frame: np.ndarray, cfg: AppConfig, win: GameWindow) -> bool:
    """清單開啟時會有多個不重複的有效桌號（current + 其他，由上而下）。

    要求 ≥3 個不重複桌號，避免關閉時下注賠率區零星數字（1:30、1:12.1…）誤判。
    """
    rows = read_list_rows(frame, cfg, win)
    distinct = {no for no, _ in rows}
    return len(distinct) >= 3


def open_room_list(
    win: GameWindow, cfg: AppConfig, capture_fn: Callable[[], np.ndarray], *, max_wait_s: float = 1.6
) -> bool:
    """開啟桌號清單。

    切換鈕是 toggle（開↔關），若 list_is_open 偶有誤判會卡相位，因此用
    「點一下→等偵測→沒開就再點」的方式，最多嘗試數次直到真的偵測為開。
    """
    if list_is_open(capture_fn(), cfg, win):
        return True
    btn = _scaled_point(cfg, win, "room_switch_button")
    if btn is None:
        logger.warning("缺少 room_switch_button 座標（請先跑 mark_rooms）")
        return False
    ui_b = cfg.automation.ui_click_backend
    # 此鈕是 toggle（開↔關）。為避免「偵測偶爾誤判 → 多點一下反而把清單關掉」
    # 造成奇偶相位錯亂，每輪都用「開→等→沒偵測到就再點一下關回去」重置，
    # 確保每次嘗試都從『關閉』狀態重新開，最多 3 輪。
    last_frame = capture_fn()
    for _ in range(3):
        click_at(win, btn[0], btn[1], backend=ui_b)  # 嘗試開
        deadline = time.monotonic() + max_wait_s
        while time.monotonic() < deadline:
            last_frame = capture_fn()
            if list_is_open(last_frame, cfg, win):
                return True
            time.sleep(0.12)
        click_at(win, btn[0], btn[1], backend=ui_b)  # 沒偵測到 → 點回關閉，重置相位
        time.sleep(0.25)
    _save_debug(last_frame, cfg, win, "open_fail")
    return False


def close_room_list(win: GameWindow, cfg: AppConfig, capture_fn: Callable[[], np.ndarray]) -> None:
    """確保清單關閉（回到乾淨狀態）。最多嘗試兩次。"""
    btn = _scaled_point(cfg, win, "room_switch_button")
    if btn is None:
        return
    for _ in range(2):
        if not list_is_open(capture_fn(), cfg, win):
            return
        click_at(win, btn[0], btn[1], backend=cfg.automation.ui_click_backend)
        time.sleep(0.3)


def scroll_list(win: GameWindow, cfg: AppConfig, *, up: bool = False) -> None:
    pt = _scaled_point(cfg, win, "room_list_scroll")
    if pt is None:
        return
    wheel_at_client(win, pt[0], pt[1], clicks=cfg.room.scroll_clicks, delta=120 if up else -120)


def scroll_to_top(
    win: GameWindow,
    cfg: AppConfig,
    capture_fn: Callable[[], np.ndarray] | None = None,
) -> None:
    """把清單捲回最頂，確保由上而下掃得到所有桌（含 No.1）。

    有 capture_fn 時逐步往上捲並驗證：當可見桌號連續兩次不再變動，視為已到頂；
    這比盲捲固定次數更可靠（捲輪有時要分次才吃得到、或一次捲過頭）。
    """
    pt = _scaled_point(cfg, win, "room_list_scroll")
    if pt is None:
        return
    if capture_fn is None:
        clicks = cfg.room.scroll_clicks * (cfg.room.max_scroll_pages + 2)
        wheel_at_client(win, pt[0], pt[1], clicks=clicks, delta=120)
        time.sleep(0.15)
        return
    prev: tuple[int, ...] | None = None
    for _ in range(cfg.room.max_scroll_pages + 3):
        cur = tuple(no for no, _ in read_list_rows(capture_fn(), cfg, win))
        if cur and cur == prev:
            return  # 連兩次可見桌號相同 → 已到頂
        prev = cur
        wheel_at_client(win, pt[0], pt[1], clicks=max(1, cfg.room.scroll_clicks), delta=120)
        time.sleep(0.18)


def confirm_switch(win: GameWindow, cfg: AppConfig) -> None:
    """點前往後會跳出『離開座位前往指定牌桌』確認視窗，按『確定』。"""
    pt = _scaled_point(cfg, win, "room_confirm_button")
    if pt is None:
        logger.warning("缺少 room_confirm_button 座標（請先跑 mark_rooms）")
        return
    time.sleep(0.4)  # 等確認視窗彈出
    click_at(win, pt[0], pt[1], backend=cfg.automation.ui_click_backend)


def goto_table(
    win: GameWindow, cfg: AppConfig, capture_fn: Callable[[], np.ndarray], target_no: int
) -> bool:
    """清單已開啟：往下捲找到 target_no 那列，點其「前往」→ 按彈窗「確定」。回傳是否有點到。

    是否真的換桌成功由呼叫端用 read_current_table 確認（滿桌會點不動）。
    """
    goto_x = _scaled_goto_x(cfg, win)
    if goto_x <= 0:
        logger.warning("缺少 room.goto_x（請先跑 mark_rooms）")
        return False
    ui_b = cfg.automation.ui_click_backend
    scroll_to_top(win, cfg, capture_fn)  # 先回最頂，才掃得到上半部的桌號（含 No.1）
    seen: set[int] = set()
    for _page in range(cfg.room.max_scroll_pages + 1):
        frame = capture_fn()
        rows = read_list_rows(frame, cfg, win)
        for no, cy in rows:
            seen.add(no)
            if no == target_no:
                click_at(win, goto_x, cy, backend=ui_b)
                confirm_switch(win, cfg)
                return True
        scroll_list(win, cfg)
        time.sleep(0.15)
    logger.info("找 No.%d 失敗，整輪掃到的桌號=%s", target_no, sorted(seen))
    _save_debug(capture_fn(), cfg, win, f"goto_{target_no}_fail")
    return False


def switch_to_table(
    win: GameWindow, cfg: AppConfig, target_no: int, *, capture_fn: Callable[[], np.ndarray] | None = None
) -> bool:
    """完整換桌：開清單 → 前往 target_no → 按確定。

    成功判斷＝清單已關閉（離開清單進入新桌）。滿桌時「前往」為灰、點了沒反應、
    清單仍開著 → 判定失敗。此法不依賴左下角桌號（會被對話框擋住）。
    """
    cap = capture_fn or (lambda: capture_client(win))
    try:
        focus_window(win.hwnd)  # 確保點擊落在遊戲視窗（live 防失焦）
    except Exception:
        pass
    if not open_room_list(win, cfg, cap):
        logger.warning("開桌號清單失敗")
        close_room_list(win, cfg, cap)  # 回到乾淨狀態
        return False
    if not goto_table(win, cfg, cap, target_no):
        logger.info("清單中找不到 No.%d（可能已不在或捲動範圍外）", target_no)
        close_room_list(win, cfg, cap)
        return False
    deadline = time.monotonic() + cfg.room.switch_confirm_s
    while time.monotonic() < deadline:
        if not list_is_open(cap(), cfg, win):
            return True  # 已離開清單 → 換桌成功
        time.sleep(0.12)
    # 清單仍開著 → 滿桌或前往無效；關閉清單回到乾淨狀態
    close_room_list(win, cfg, cap)
    return False
