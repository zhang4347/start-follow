"""「超過五局未押注」居中提示窗偵測（OCR + 視覺，避免牌桌／大廳底部誤判在房內）。"""

from __future__ import annotations

import time

import cv2
import numpy as np

from star_follow.config import AppConfig
from star_follow.vision.menu_match import match_template_in_region
from star_follow.vision.nav_text import region_has_keywords
from star_follow.vision.roi import scale_point, scale_rect

_T_CONFIRM = "lobby_confirm_button.png"
_REF_DIALOG = (380, 370, 520, 290)
_REF_MSG_OCR = (300, 385, 680, 150)
_MSG_KEYS = ("五局", "未押注", "退出遊戲", "退出", "未押")
_THR_TEMPLATE = 0.58
_TEAL_MIN = 0.14
_GREEN_MIN = 280
_TITLE_PURPLE_MIN = 0.032

# 點過確定仍「看得到」時：暫停視覺辨識與連點（常為牌桌 UI 誤判）
_visual_suppress_until = 0.0
_click_cooldown_until = 0.0


def suppress_kick_visual(seconds: float) -> None:
    global _visual_suppress_until
    _visual_suppress_until = max(_visual_suppress_until, time.monotonic() + seconds)


def suppress_kick_clicks(seconds: float) -> None:
    global _click_cooldown_until
    _click_cooldown_until = max(_click_cooldown_until, time.monotonic() + seconds)


def kick_click_on_cooldown() -> bool:
    return time.monotonic() < _click_cooldown_until


def _dialog_rect(frame: np.ndarray, cfg: AppConfig) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    return scale_rect(
        list(_REF_DIALOG),
        cfg.window.reference_width or 1280,
        cfg.window.reference_height or 720,
        w,
        h,
    )


def _kick_dialog_visual(frame: np.ndarray, cfg: AppConfig) -> bool:
    """居中提示窗：紫標題列 + 青綠內文區 + 下方綠色確定鈕。"""
    if time.monotonic() < _visual_suppress_until:
        return False
    x, y, bw, bh = _dialog_rect(frame, cfg)
    roi = frame[y : y + bh, x : x + bw]
    if roi.size == 0 or bh < 40 or bw < 80:
        return False
    top = roi[: max(8, int(bh * 0.22)), :]
    hsv_t = cv2.cvtColor(top, cv2.COLOR_RGB2HSV)
    purple = cv2.inRange(hsv_t, np.array([115, 35, 45]), np.array([168, 255, 255]))
    n_top = max(1, top.shape[0] * top.shape[1])
    purple_r = float(purple.sum() / 255 / n_top)
    mid = roi[int(bh * 0.10) : int(bh * 0.55), :]
    if mid.size == 0:
        return False
    hsv_m = cv2.cvtColor(mid, cv2.COLOR_RGB2HSV)
    teal = cv2.inRange(hsv_m, np.array([75, 30, 30]), np.array([108, 255, 255]))
    teal_r = float(teal.mean()) / 255.0
    if teal_r < _TEAL_MIN:
        return False
    if purple_r < _TITLE_PURPLE_MIN and teal_r < 0.18:
        return False
    sub = roi[int(bh * 0.38) :, :]
    if sub.size == 0:
        return False
    hsv_s = cv2.cvtColor(sub, cv2.COLOR_RGB2HSV)
    green = cv2.inRange(hsv_s, np.array([32, 60, 60]), np.array([95, 255, 255]))
    return int(green.sum() / 255) >= _GREEN_MIN


def kick_popup_message_ocr(frame: np.ndarray, cfg: AppConfig) -> tuple[bool, str]:
    h, w = frame.shape[:2]
    for ref in (_REF_MSG_OCR, (360, 400, 560, 130)):
        body = scale_rect(
            list(ref),
            cfg.window.reference_width or 1280,
            cfg.window.reference_height or 720,
            w,
            h,
        )
        hit, text, _conf = region_has_keywords(frame, body, _MSG_KEYS)
        if hit:
            return True, text
    return False, ""


def is_kick_idle_popup(
    frame: np.ndarray,
    cfg: AppConfig,
    *,
    win: object | None = None,
) -> bool:
    """五局未押注提示窗：OCR、紫標題+青綠視覺、或確定鈕模板。"""
    hit, _text = kick_popup_message_ocr(frame, cfg)
    if hit:
        return True
    x, y, bw, bh = _dialog_rect(frame, cfg)
    tpl = match_template_in_region(frame, _T_CONFIRM, (x, y, bw, bh), threshold=0.0)
    if tpl and float(tpl[2]) >= _THR_TEMPLATE:
        return True
    if _kick_dialog_visual(frame, cfg):
        return True
    return False


def find_kick_confirm_xy(
    frame: np.ndarray,
    cfg: AppConfig,
    win: object,
) -> tuple[int, int] | None:
    """僅在 is_kick_idle_popup 為真時呼叫。"""
    pt = cfg.click_points.get("kick_idle_confirm")
    if pt and len(pt) == 2:
        ref_w = cfg.window.reference_width or 1280
        ref_h = cfg.window.reference_height or 720
        cw = int(getattr(win, "client_width", frame.shape[1]))
        ch = int(getattr(win, "client_height", frame.shape[0]))
        return scale_point(list(pt), ref_w, ref_h, cw, ch)
    x, y, bw, bh = _dialog_rect(frame, cfg)
    tpl = match_template_in_region(frame, _T_CONFIRM, (x, y, bw, bh), threshold=0.0)
    if tpl and float(tpl[2]) >= _THR_TEMPLATE:
        return int(tpl[0]), int(tpl[1])
    roi = frame[y : y + bh, x : x + bw]
    sub = roi[int(bh * 0.38) :, :] if roi.size else roi
    if sub.size:
        hsv = cv2.cvtColor(sub, cv2.COLOR_RGB2HSV)
        green = cv2.inRange(hsv, np.array([32, 60, 60]), np.array([95, 255, 255]))
        ys, xs = np.where(green > 0)
        if xs.size >= 80:
            return x + int(xs.mean()), y + int(bh * 0.38) + int(ys.mean())
    return None
