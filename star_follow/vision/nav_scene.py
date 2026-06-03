"""大廳導覽：多特徵計分分段（非單一模板／單一顏色）。

各畫面靠多項指標加總，並有「否決規則」避免牌桌、首頁被誤判成棋牌大廳。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from star_follow.config import AppConfig
from star_follow.vision.game_detect import (
    detect_in_baccarat_room,
    detect_room_switch_button,
    has_room_switch_template,
    is_lobby,
    qipai_sidebar_tab_state,
)
from star_follow.vision.menu_match import match_template_in_region
from star_follow.vision.roi import scale_rect

# 1280×720 參考
_REF_QIPAI_TAB = (1130, 335, 150, 95)
_REF_CARD_BAND = (120, 220, 1040, 260)  # 棋牌大廳橫向遊戲卡片列
_REF_TABLE_MENU = (1175, 18, 95, 95)  # 牌桌右上 ☰（與右側 y≈335 的棋牌分頁不同）
_REF_RANDOM = (985, 580, 295, 140)

PHASE_TABLE = "table"
PHASE_ENTRY = "baccarat_entry"
PHASE_QIPAI_READY = "qipai_card_ready"
PHASE_QIPAI_SCROLL = "qipai_need_scroll"
PHASE_HOME = "home_lobby"
PHASE_UNKNOWN = "unknown"


@dataclass
class SceneFeatures:
    """各畫面特徵（0~1 或分數），診斷時全部印出。"""

    card_row: float = 0.0
    home_purple: float = 0.0
    qipai_tab: str = "unclear"
    entry_yellow: bool = False
    random_tpl: float = 0.0
    baccarat_card_tpl: float = 0.0
    table_switch: float = 0.0
    table_switch_ok: bool = False
    in_room: bool = False
    in_room_method: str = ""
    table_menu_chart: float = 0.0
    table_countdown: float = 0.0
    table_bottom_no: float = 0.0
    table_ui: float = 0.0
    chip_bar: bool = False
    entry_score: float = 0.0

    phase_scores: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def measure_center_yellow_ratio(frame: np.ndarray) -> float:
    """中央黃字說明（入口／規則頁），不套用 is_lobby 的倒數否決。"""
    h, w = frame.shape[:2]
    center = frame[int(h * 0.25) : int(h * 0.72), int(w * 0.18) : int(w * 0.72)]
    if center.size == 0:
        return 0.0
    hsv = cv2.cvtColor(center, cv2.COLOR_RGB2HSV)
    yellow = cv2.inRange(hsv, np.array([15, 70, 100]), np.array([45, 255, 255]))
    return float(yellow.sum() / 255 / max(1, center.shape[0] * center.shape[1]))


def measure_random_button_score(frame: np.ndarray, cfg: AppConfig) -> float:
    """右下角『隨機選台』按鈕區（綠／金）。"""
    band = _scale_band(frame, cfg, _REF_RANDOM)
    if band.size == 0:
        return 0.0
    hsv = cv2.cvtColor(band, cv2.COLOR_RGB2HSV)
    green = cv2.inRange(hsv, np.array([32, 50, 80]), np.array([95, 255, 255]))
    warm = cv2.inRange(hsv, np.array([12, 55, 90]), np.array([42, 255, 255]))
    r = max(float(green.mean()) / 255.0, float(warm.mean()) / 255.0)
    return min(1.0, r * 3.5)


def measure_baccarat_entry_score(
    frame: np.ndarray,
    cfg: AppConfig,
    *,
    random_tpl: float = 0.0,
) -> float:
    """百家樂入口（準備點隨機選台）：優先於首頁紫色分頁判斷。"""
    if _chip_bar(frame):
        return 0.0
    score = 0.0
    y = measure_center_yellow_ratio(frame)
    if y >= 0.016:
        score = max(score, min(1.0, y * 20.0))
    if random_tpl >= 0.30:
        score = max(score, random_tpl)
    score = max(score, measure_random_button_score(frame, cfg) * 0.9)
    t_cd = measure_table_countdown(frame, cfg)
    t_menu = measure_table_menu_chart(frame, cfg)
    # 入口說明頁頂部常讓倒數 ROI 偏高，但尚無籌碼列／牌桌選單
    if t_cd >= 0.50 and t_menu < 0.42:
        score = max(score, 0.52)
    return score


def _scale_band(frame: np.ndarray, cfg: AppConfig, ref_rect: tuple[int, int, int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    ref_w = cfg.window.reference_width or 1280
    ref_h = cfg.window.reference_height or 720
    x, y, bw, bh = scale_rect(list(ref_rect), ref_w, ref_h, w, h)
    return frame[y : y + bh, x : x + bw]


def measure_card_row_score(frame: np.ndarray, cfg: AppConfig) -> float:
    """棋牌大廳：中央一排『直向遊戲卡片』的結構分（0~1）。

    用垂直邊緣柱狀圖找 4~7 個等距峰；牌桌、首頁 SLOT 橫幅通常不符合。
    """
    band = _scale_band(frame, cfg, _REF_CARD_BAND)
    if band.size == 0:
        return 0.0
    gray = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    col_e = np.abs(sobelx).sum(axis=0).astype(np.float32)
    if col_e.max() < 1:
        return 0.0
    col_e = col_e / col_e.max()
    bw = band.shape[1]
    thr = 0.42
    mins: list[int] = []
    i = 8
    while i < bw - 8:
        if col_e[i] >= thr and (i == 8 or col_e[i - 1] < thr):
            j = i
            while j < bw and col_e[j] >= thr:
                j += 1
            if 18 <= (j - i) <= int(bw * 0.22):
                mins.append((i + j) // 2)
            i = j
        else:
            i += 1
    n = len(mins)
    if n < 4 or n > 8:
        return 0.0
    gaps = np.diff(mins)
    if gaps.size == 0:
        return 0.0
    mean_g = float(gaps.mean())
    if mean_g < bw * 0.08 or mean_g > bw * 0.28:
        return 0.0
    uniform = 1.0 - min(1.0, float(gaps.std()) / max(mean_g, 1.0))
    if uniform < 0.55:
        return 0.0
    hsv = cv2.cvtColor(band, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    color = float((sat > 75).mean())
    if color < 0.08:
        return 0.0
    return min(1.0, uniform * color * (n / 6.0))


def measure_table_countdown(frame: np.ndarray, cfg: AppConfig) -> float:
    h, w = frame.shape[:2]
    cd = cfg.roi.get("countdown")
    if not cd or len(cd) != 4:
        return 0.0
    x, y, bw, bh = scale_rect(
        cd, cfg.window.reference_width or 1280, cfg.window.reference_height or 720, w, h
    )
    top = frame[y : y + bh, x : x + bw]
    if top.size == 0:
        return 0.0
    gray = cv2.cvtColor(top, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    std = float(th.std())
    if std < 28:
        return 0.0
    return min(1.0, (std - 28) / 25.0)


def measure_table_menu_chart(frame: np.ndarray, cfg: AppConfig) -> float:
    """牌桌右上 ☰ 區（menu_button ROI），與大廳右側棋牌分頁不同高度。"""
    h, w = frame.shape[:2]
    ref_w = cfg.window.reference_width or 1280
    ref_h = cfg.window.reference_height or 720
    roi = cfg.roi.get("menu_button")
    rect = (
        scale_rect(roi, ref_w, ref_h, w, h)
        if roi and len(roi) == 4
        else scale_rect(list(_REF_TABLE_MENU), ref_w, ref_h, w, h)
    )
    best = 0.0
    for name in ("menu_chart.png", "menu_chart_crop.png"):
        hit = match_template_in_region(frame, name, rect, threshold=0.0)
        if hit:
            best = max(best, float(hit[2]))
    return best


def measure_table_bottom_no(frame: np.ndarray, cfg: AppConfig) -> float:
    """左下角目前桌號區塊（有 No. 數字時對比偏高）。"""
    rt = cfg.roi.get("room_current_table")
    if not rt or len(rt) != 4:
        return 0.0
    h, w = frame.shape[:2]
    x, y, bw, bh = scale_rect(
        rt, cfg.window.reference_width or 1280, cfg.window.reference_height or 720, w, h
    )
    patch = frame[y : y + bh, x : x + bw]
    if patch.size == 0:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    std = float(gray.std())
    return min(1.0, max(0.0, (std - 22) / 28.0))


def _chip_bar(frame: np.ndarray) -> bool:
    h, w = frame.shape[:2]
    strip = frame[int(h * 0.76) : int(h * 0.92), int(w * 0.22) : int(w * 0.78)]
    hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
    mask = (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 80)
    if mask.mean() < 0.04:
        return False
    col = mask.sum(axis=0)
    return int((col > strip.shape[0] * 0.15).sum()) >= 4


def compute_scene_features(
    frame: np.ndarray,
    cfg: AppConfig,
    *,
    baccarat_card_score: float = 0.0,
    random_score: float = 0.0,
    win: object | None = None,
) -> SceneFeatures:
    h, w = frame.shape[:2]
    ref_w = cfg.window.reference_width or 1280
    ref_h = cfg.window.reference_height or 720
    tab_rect = scale_rect(list(_REF_QIPAI_TAB), ref_w, ref_h, w, h)
    tab, purple_r, _ = qipai_sidebar_tab_state(frame, tab_rect)

    sw_ok, sw_meta = detect_room_switch_button(frame, cfg)
    in_room, room_meta = detect_in_baccarat_room(frame, cfg, win)
    t_menu = measure_table_menu_chart(frame, cfg)
    t_cd = measure_table_countdown(frame, cfg)
    t_no = measure_table_bottom_no(frame, cfg)
    card_row = measure_card_row_score(frame, cfg)
    entry_score = measure_baccarat_entry_score(frame, cfg, random_tpl=random_score)

    at_table = sw_ok or in_room
    table_ui = 1.0 if at_table else 0.0

    f = SceneFeatures(
        card_row=card_row,
        home_purple=purple_r,
        qipai_tab=tab,
        entry_yellow=is_lobby(frame, cfg),
        entry_score=entry_score,
        random_tpl=random_score,
        baccarat_card_tpl=baccarat_card_score,
        table_switch=float(sw_meta.get("switch_score", 0.0)),
        table_switch_ok=sw_ok,
        in_room=in_room,
        in_room_method=str(room_meta.get("method", "")),
        table_menu_chart=t_menu,
        table_countdown=t_cd,
        table_bottom_no=t_no,
        table_ui=min(1.0, table_ui),
        chip_bar=_chip_bar(frame),
    )
    f.phase_scores, f.reasons = _score_phases(f)
    return f


def _score_phases(f: SceneFeatures) -> tuple[dict[str, float], list[str]]:
    s: dict[str, float] = {PHASE_UNKNOWN: 0.05}
    reasons: list[str] = []
    at_table = f.table_switch_ok or f.in_room
    entry_like = f.entry_score >= 0.40 or f.entry_yellow or f.random_tpl >= 0.38

    # --- 1) 牌桌 ---
    if at_table:
        s[PHASE_TABLE] = 1.35
        if f.table_switch_ok:
            reasons.append(f"牌桌：換桌模板 score={f.table_switch:.2f}")
        else:
            reasons.append(f"牌桌：在房內 ({f.in_room_method or 'detect'})")

    # --- 2) 入口（優先於首頁；倒數 ROI 偏高但無籌碼列也算入口）---
    if not at_table and entry_like:
        s[PHASE_ENTRY] = 1.08 + f.entry_score * 0.45 + f.random_tpl * 0.25
        reasons.append(
            f"入口：entry_score={f.entry_score:.2f} 黃字={f.entry_yellow} random_tpl={f.random_tpl:.2f}"
        )

    # --- 3) 首頁：僅「未進棋牌」且不像入口 ---
    if not at_table and not entry_like and f.qipai_tab == "unselected" and f.home_purple >= 0.20:
        s[PHASE_HOME] = 0.62 + f.home_purple * 0.55
        reasons.append(f"首頁：棋牌分頁紫色 purple={f.home_purple:.2f}")

    # --- 4) 棋牌大廳：棋牌分頁已選中（金/亮）---
    if not at_table and not entry_like and f.qipai_tab == "selected":
        if f.baccarat_card_tpl >= 0.52:
            s[PHASE_QIPAI_READY] = 0.72 + f.baccarat_card_tpl * 0.55
            reasons.append(f"棋牌可點百家樂：card_tpl={f.baccarat_card_tpl:.2f}")
        else:
            s[PHASE_QIPAI_SCROLL] = 0.78
            reasons.append("棋牌大廳：分頁已選中，百家樂模板未達標→需滑動")

    if f.card_row >= 0.48 and not at_table and PHASE_QIPAI_SCROLL not in s and PHASE_QIPAI_READY not in s:
        reasons.append(f"牌列結構={f.card_row:.2f}（輔助；主判斷用棋牌分頁顏色）")

    return s, reasons


def classify_phase_from_features(f: SceneFeatures, *, min_conf: float = 0.48, min_gap: float = 0.10) -> tuple[str, float]:
    ranked = sorted(f.phase_scores.items(), key=lambda x: -x[1])
    phase, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    total = sum(f.phase_scores.values()) or 1.0
    conf = top / total
    if top < min_conf or (top - second) < min_gap:
        return PHASE_UNKNOWN, min(conf, 0.35)
    if phase == PHASE_TABLE and not (f.table_switch_ok or f.in_room):
        return PHASE_UNKNOWN, 0.3
    if phase in (PHASE_QIPAI_SCROLL, PHASE_QIPAI_READY) and f.qipai_tab != "selected":
        return PHASE_UNKNOWN, 0.3
    if phase == PHASE_HOME and f.qipai_tab != "unselected":
        return PHASE_UNKNOWN, 0.3
    if phase == PHASE_ENTRY and f.entry_score < 0.32 and f.random_tpl < 0.35 and not f.entry_yellow:
        return PHASE_UNKNOWN, 0.3
    if phase == PHASE_HOME and f.entry_score >= 0.40:
        return PHASE_ENTRY, max(conf, 0.75)
    return phase, conf
