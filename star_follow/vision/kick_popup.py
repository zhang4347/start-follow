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
# 高信心門檻：確定鈕模板達此分數即採信，即使畫面像牌桌也算彈窗。
# 實測：真實牌桌僅 ~0.38，彈窗達 1.000，故 0.80 既可靠又零誤判。
_THR_TEMPLATE_STRICT = 0.80
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


def _kick_requires_ocr_only(frame: np.ndarray, cfg: AppConfig, win: object | None = None) -> bool:
    """已在牌桌時，五局提示只用 OCR 關鍵字，避免視覺/模板誤點確定。"""
    from star_follow.vision.game_detect import _looks_like_table_surface, is_qipai_hall_frame

    if is_qipai_hall_frame(frame, cfg):
        return True
    return _looks_like_table_surface(frame, cfg)


_kick_memo: dict = {}


def _frame_fp(frame: np.ndarray) -> tuple:
    return (frame.shape, int(frame[::64, ::64].sum(dtype=np.int64)))


def is_kick_idle_popup(
    frame: np.ndarray,
    cfg: AppConfig,
    *,
    win: object | None = None,
) -> bool:
    """五局未押注提示窗。對「同一幀」去重快取（一次 classify 內會被多處重複呼叫）。"""
    key = id(frame)
    fp = _frame_fp(frame)
    cached = _kick_memo.get(key)
    if cached is not None and cached[0] == fp:
        return cached[1]
    r = _is_kick_idle_popup_impl(frame, cfg, win=win)
    if len(_kick_memo) > 24:
        _kick_memo.clear()
    _kick_memo[key] = (fp, r)
    return r


def _is_kick_idle_popup_impl(
    frame: np.ndarray,
    cfg: AppConfig,
    *,
    win: object | None = None,
) -> bool:
    """五局未押注提示窗：確定鈕模板（高信心可跨牌桌）優先，再以紫標題+青綠視覺
    與 OCR 關鍵字補強。

    速度：先算「便宜」的模板分數（~數 ms）。高信心直接採信、免 OCR；連對話框
    視覺都沒有又遠低於門檻時，必非彈窗 → 直接排除、免 OCR。只有「疑似對話框、
    模板分數中等」時才跑較慢的 OCR/牌桌守門，避免回桌途中每一幀都吃 Tesseract。
    """
    # 1) 確定鈕模板高信心 → 直接採信（最快最準，彈窗實測 1.000、真牌桌/大廳僅 ~0.38~0.46）
    x, y, bw, bh = _dialog_rect(frame, cfg)
    tpl = match_template_in_region(frame, _T_CONFIRM, (x, y, bw, bh), threshold=0.0)
    tpl_score = float(tpl[2]) if tpl else 0.0
    if tpl_score >= _THR_TEMPLATE_STRICT:
        return True

    # 2) 牌桌／棋牌大廳表面：視覺與卡片美術會干擾（百家樂卡片會誤觸發 dialog_visual），
    #    這些畫面上唯一可靠的彈窗訊號就是確定鈕模板。模板偏低 → 必為乾淨畫面，免 OCR 直接排除；
    #    只有模板落在中段（可能是非參考解析度下的彈窗）才花 OCR 關鍵字確認，保留安全網。
    if _kick_requires_ocr_only(frame, cfg, win):
        if tpl_score >= _THR_TEMPLATE:
            return kick_popup_message_ocr(frame, cfg)[0]
        return False

    # 3) 非牌桌/大廳的居中對話框：紫標題+青綠視覺 + OCR 關鍵字補強
    dialog_visual = _kick_dialog_visual(frame, cfg)
    if not dialog_visual and tpl_score < _THR_TEMPLATE:
        return False
    if kick_popup_message_ocr(frame, cfg)[0]:
        return True
    if tpl_score >= _THR_TEMPLATE:
        return True
    return dialog_visual


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
