"""被踢回大廳時，自動導覽回到百家樂桌。

畫面狀態（classify_nav_screen）分五段 + 牌桌：
  home_lobby         首頁大廳 → 點右側「棋牌」
  qipai_need_scroll  棋牌大廳、尚未滑到百家樂 → 只拖曳往右滑
  qipai_card_ready   棋牌大廳、已看到百家樂卡片 → 點卡片
  baccarat_entry     百家樂入口 → 點「隨機選台」
  table              已在牌桌
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from star_follow.automation.click import click_at, drag_client
from star_follow.capture.screen import capture_client
from star_follow.capture.window import GameWindow, focus_window
from star_follow.config import AppConfig
from star_follow.vision.game_detect import (
    detect_in_baccarat_room,
    has_room_switch_template,
    qipai_sidebar_tab_state,
)
from star_follow.vision.nav_scene import (
    SceneFeatures,
    classify_phase_from_features,
    compute_scene_features,
)
from star_follow.vision.menu_match import match_template_in_region
from star_follow.vision.roi import scale_point, scale_rect

logger = logging.getLogger(__name__)

# --- 畫面階段代碼 ---
PHASE_TABLE = "table"
PHASE_ENTRY = "baccarat_entry"
PHASE_QIPAI_READY = "qipai_card_ready"
PHASE_QIPAI_SCROLL = "qipai_need_scroll"
PHASE_HOME = "home_lobby"
PHASE_UNKNOWN = "unknown"

_PHASE_LABEL: dict[str, str] = {
    PHASE_TABLE: "已在牌桌",
    PHASE_ENTRY: "百家樂入口",
    PHASE_QIPAI_READY: "棋牌大廳(已滑動·可點百家樂)",
    PHASE_QIPAI_SCROLL: "棋牌大廳(未滑動·需往右拖曳)",
    PHASE_HOME: "首頁大廳(右側棋牌未選中·紫色)",
    PHASE_UNKNOWN: "無法確認",
}

_PHASE_ACTION: dict[str, str] = {
    PHASE_TABLE: "無需回桌",
    PHASE_ENTRY: "點『隨機選台』進桌",
    PHASE_QIPAI_READY: "點畫面上的百家樂卡片",
    PHASE_QIPAI_SCROLL: "按住拖曳往左滑，露出右側百家樂後再點",
    PHASE_HOME: "點右側選單『棋牌』",
    PHASE_UNKNOWN: "等待載入或補模板／手動確認畫面",
}

_T_RANDOM = "lobby_random_select.png"
_T_CARD = "lobby_baccarat_card.png"
_T_QIPAI = "lobby_qipai_menu.png"
_T_CONFIRM = "lobby_confirm_button.png"

_RECT_CONFIRM = (380, 370, 520, 290)
_REF_CONFIRM_PT = (640, 495)
_THR_CONFIRM = 0.52
_RECT_RANDOM = (985, 580, 295, 140)
_RECT_QIPAI = (1130, 335, 150, 95)

_REF_PT_QIPAI = (1209, 382)
_REF_PT_CARD = (529, 364)
_REF_PT_RANDOM = (1127, 649)

_THR_CARD_READY = 0.58
_THR_CARD_STRONG = 0.72
_THR_RANDOM_ENTRY = 0.55
_THR_QIPAI_MENU = 0.55
_MIN_CONFIDENCE = 0.50
_AMBIGUOUS_GAP = 0.12
_QIPAI_ENTER_GRACE_S = 14.0

# OCR 裁切區（1280×720）：關鍵字當模板外的第二證據
_RECT_OCR_CARDS = (160, 290, 960, 300)
_REF_CARD_X_MIN = 340
_REF_CARD_X_MAX = 780


@dataclass
class NavScreenResult:
    phase: str
    label: str
    action: str
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    phase_scores: dict[str, float] = field(default_factory=dict)
    qipai_hall: bool = False
    home_lobby: bool = False
    card_score: float = 0.0
    card_xy: tuple[int, int] | None = None
    random_score: float = 0.0
    menu_score: float = 0.0
    qipai_tab: str = ""
    table_hud: bool = False
    scores: dict[str, float] = field(default_factory=dict)
    ocr: dict[str, str] = field(default_factory=dict)


def _save_unknown_frame(frame: np.ndarray, tag: str) -> None:
    try:
        from star_follow.paths import logs_dir
        from PIL import Image

        d = logs_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"recovery_{tag}_{int(time.time())}.png"
        Image.fromarray(frame).save(path)
        logger.info("已存回桌診斷截圖 %s", path.name)
    except Exception:  # noqa: BLE001
        logger.exception("存回桌診斷截圖失敗")


def _full_rect(win: GameWindow) -> tuple[int, int, int, int]:
    return (0, 0, win.client_width, win.client_height)


def _base_scale(cfg: AppConfig, win: GameWindow) -> float:
    ref = cfg.window.reference_width or 1280
    if ref <= 0:
        return 1.0
    return win.client_width / ref


def _scaled_pt(cfg: AppConfig, win: GameWindow, ref_pt: tuple[int, int]) -> tuple[int, int]:
    return scale_point(
        list(ref_pt),
        cfg.window.reference_width or 1280,
        cfg.window.reference_height or 720,
        win.client_width,
        win.client_height,
    )


def _scaled_rect(cfg: AppConfig, win: GameWindow, ref_rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return scale_rect(
        list(ref_rect),
        cfg.window.reference_width or 1280,
        cfg.window.reference_height or 720,
        win.client_width,
        win.client_height,
    )


def _match_score(
    frame: np.ndarray,
    win: GameWindow,
    name: str,
    rect: tuple[int, int, int, int],
    cfg: AppConfig,
) -> tuple[int, int, float] | None:
    return match_template_in_region(
        frame, name, rect, threshold=0.0, base_scale=_base_scale(cfg, win)
    )


def _gather_signals(frame: np.ndarray, cfg: AppConfig, win: GameWindow) -> dict:
    """收集各模板最佳分數與特徵（門檻 0，僅用於分類）。"""
    full = _full_rect(win)
    random_r = _match_score(frame, win, _T_RANDOM, _scaled_rect(cfg, win, _RECT_RANDOM), cfg)
    random_w = _match_score(frame, win, _T_RANDOM, full, cfg)
    card_w = _match_score(frame, win, _T_CARD, full, cfg)
    menu_r = _match_score(frame, win, _T_QIPAI, _scaled_rect(cfg, win, _RECT_QIPAI), cfg)
    confirm = _match_score(frame, win, _T_CONFIRM, _scaled_rect(cfg, win, _RECT_CONFIRM), cfg)

    def _s(hit: tuple[int, int, float] | None) -> float:
        return float(hit[2]) if hit else 0.0

    tab_rect = _scaled_rect(cfg, win, _RECT_QIPAI)
    qipai_tab, purple_r, highlight_r = qipai_sidebar_tab_state(frame, tab_rect)
    # 隨機選台只在右下固定區比對；全畫面比對在大廳會誤觸發入口
    random_score = _s(random_r)
    card_score = _s(card_w) if qipai_tab == "selected" else 0.0
    menu_score = _s(menu_r)
    scene = compute_scene_features(
        frame,
        cfg,
        baccarat_card_score=card_score,
        random_score=random_score,
        win=win,
    )

    return {
        "random_r": random_r,
        "random_w": random_w,
        "random_score": random_score,
        "card_w": card_w,
        "card_score": card_score,
        "card_xy": (card_w[0], card_w[1]) if card_w else None,
        "menu_r": menu_r,
        "menu_score": menu_score,
        "confirm_score": _s(confirm),
        "qipai_hall": qipai_tab == "selected",
        "qipai_tab": qipai_tab,
        "purple_r": purple_r,
        "highlight_r": highlight_r,
        "table_hud": scene.table_switch_ok or scene.in_room,
        "in_room": scene.in_room,
        "in_room_method": scene.in_room_method,
        "scene": scene,
        "has_switch_tpl": has_room_switch_template(),
    }


def _ocr_signals(frame: np.ndarray, cfg: AppConfig, win: GameWindow) -> dict:
    from star_follow.vision.nav_text import region_has_keywords

    ocr_random = region_has_keywords(
        frame, _scaled_rect(cfg, win, _RECT_RANDOM), ("隨機選台", "隨機", "選台")
    )
    ocr_menu = region_has_keywords(frame, _scaled_rect(cfg, win, _RECT_QIPAI), ("棋牌",))
    ocr_card = region_has_keywords(
        frame, _scaled_rect(cfg, win, _RECT_OCR_CARDS), ("百家樂", "百家")
    )
    return {
        "ocr_random": ocr_random[0],
        "ocr_random_text": ocr_random[1],
        "ocr_menu": ocr_menu[0],
        "ocr_menu_text": ocr_menu[1],
        "ocr_baccarat": ocr_card[0],
        "ocr_baccarat_text": ocr_card[1],
    }


def _card_ready_valid(sig: dict, cfg: AppConfig, win: GameWindow) -> bool:
    """百家樂模板要比中，且位置合理（避免把左邊其他遊戲卡片誤當百家樂）。"""
    if sig["card_score"] < _THR_CARD_READY:
        return False
    xy = sig.get("card_xy")
    if not xy:
        return sig["card_score"] >= _THR_CARD_STRONG
    x_min, _ = _scaled_pt(cfg, win, (_REF_CARD_X_MIN, 0))
    x_max, _ = _scaled_pt(cfg, win, (_REF_CARD_X_MAX, 0))
    if xy[0] < x_min and sig["card_score"] < 0.68:
        return False
    if xy[0] > x_max and sig["card_score"] < 0.65:
        return False
    return True


def _needs_ocr(sig: dict) -> bool:
    """模板分數接近、易混淆時才跑 OCR（較慢）。"""
    scene: SceneFeatures | None = sig.get("scene")
    if scene and scene.table_switch_ok:
        return False
    a = sig["random_score"]
    b = sig["card_score"]
    if max(a, b) - min(a, b) < 0.15:
        return True
    return False


def classify_nav_screen(
    frame: np.ndarray,
    cfg: AppConfig,
    win: GameWindow,
    *,
    use_ocr: bool | None = None,
) -> NavScreenResult:
    """判斷目前在哪一頁。模板 + 加權分 + 可選 OCR；信心不足則 unknown。"""
    sig = _gather_signals(frame, cfg, win)
    nav_cfg = _read_nav_cfg(cfg)
    if use_ocr is None:
        use_ocr = nav_cfg["use_ocr"] and _needs_ocr(sig)
    scene: SceneFeatures = sig["scene"]
    if use_ocr:
        ocr = _ocr_signals(frame, cfg, win)
        sig.update(ocr)
        if ocr.get("ocr_random"):
            scene.phase_scores[PHASE_ENTRY] = scene.phase_scores.get(PHASE_ENTRY, 0) + 0.35
        scene = compute_scene_features(
            frame,
            cfg,
            baccarat_card_score=sig["card_score"],
            random_score=max(sig["random_score"], 0.55 if ocr.get("ocr_random") else 0),
            win=win,
        )
        sig["scene"] = scene

    phase_scores = dict(scene.phase_scores)
    phase, confidence = classify_phase_from_features(
        scene, min_conf=_MIN_CONFIDENCE, min_gap=_AMBIGUOUS_GAP
    )

    in_ok, in_meta = detect_in_baccarat_room(frame, cfg, win)
    override_note = ""
    if _is_kick_popup_visible(frame, cfg, win):
        if phase == PHASE_TABLE:
            phase = PHASE_QIPAI_SCROLL
            override_note = "五局提示窗→不算牌桌"
        in_ok = False
    elif in_ok and phase != PHASE_TABLE:
        override_note = f"在房內覆寫：{phase}→table ({in_meta.get('method', '')})"
        phase = PHASE_TABLE
        confidence = max(confidence, 0.88)

    reasons: list[str] = []
    if override_note:
        reasons.append(override_note)
    reasons.extend(
        [
            f"【計分】牌列結構={scene.card_row:.2f} 牌桌ui={scene.table_ui:.2f} "
            f"在房內={scene.in_room}({scene.in_room_method}) "
            f"倒數OCR區={scene.table_countdown:.2f} 統計圖={scene.table_menu_chart:.2f} "
            f"桌號區={scene.table_bottom_no:.2f}",
            f"模板 random={sig['random_score']:.2f} baccarat_card={sig['card_score']:.2f}",
            f"棋牌分頁={sig.get('qipai_tab')} 紫={sig.get('purple_r', 0):.2f} 換桌={scene.table_switch_ok}",
        ]
    )
    reasons.extend(scene.reasons)
    if not sig.get("has_switch_tpl"):
        reasons.append("可選：在牌桌用 mark_rooms 存檔產生 room_switch.png，牌桌判定更穩")
    if use_ocr:
        reasons.append(
            f"OCR 隨機={sig.get('ocr_random')} 棋牌={sig.get('ocr_menu')} 百家樂={sig.get('ocr_baccarat')}"
        )

    if phase == PHASE_QIPAI_READY and not _card_ready_valid(sig, cfg, win):
        phase = PHASE_QIPAI_SCROLL
        reasons.append("百家樂模板/位置未達標→改判需滑動")

    scene_feat: SceneFeatures = sig["scene"]
    if phase == PHASE_ENTRY and not scene_feat.entry_yellow and sig["random_score"] < 0.48 and not sig.get(
        "ocr_random"
    ):
        phase = PHASE_UNKNOWN
        reasons.append("入口分數偏低→暫不確認")

    ocr_out = {}
    if use_ocr:
        ocr_out = {
            "random": sig.get("ocr_random_text", ""),
            "menu": sig.get("ocr_menu_text", ""),
            "baccarat": sig.get("ocr_baccarat_text", ""),
        }

    return NavScreenResult(
        phase=phase,
        label=_PHASE_LABEL.get(phase, phase),
        action=_PHASE_ACTION.get(phase, ""),
        confidence=confidence,
        reasons=reasons,
        phase_scores=phase_scores,
        qipai_hall=scene.card_row >= 0.48,
        home_lobby=phase == PHASE_HOME,
        card_score=sig["card_score"],
        card_xy=sig["card_xy"] if _card_ready_valid(sig, cfg, win) else None,
        random_score=sig["random_score"],
        menu_score=sig["menu_score"],
        qipai_tab=sig.get("qipai_tab", ""),
        table_hud=bool(sig.get("table_hud")) or in_ok,
        scores={
            "random": sig["random_score"],
            "card": sig["card_score"],
            "menu": sig["menu_score"],
            "confirm": sig["confirm_score"],
        },
        ocr=ocr_out,
    )


def _read_nav_cfg(cfg: AppConfig) -> dict:
    raw = cfg.raw.get("nav_confirm")
    if not isinstance(raw, dict):
        raw = {}
    return {
        "samples": max(1, int(raw.get("samples", 3))),
        "gap_s": float(raw.get("gap_s", 0.12)),
        "min_agree": max(1, int(raw.get("min_agree", 2))),
        "min_confidence": float(raw.get("min_confidence", _MIN_CONFIDENCE)),
        "use_ocr": bool(raw.get("use_ocr", True)),
        "enter_table_wait_s": float(raw.get("enter_table_wait_s", 28.0)),
        "enter_table_poll_s": float(raw.get("enter_table_poll_s", 0.35)),
        "not_table_grace_s": float(raw.get("not_table_grace_s", 10.0)),
        "not_table_poll_s": float(raw.get("not_table_poll_s", 0.4)),
    }


def classify_nav_screen_confirmed(
    capture_fn: Callable[[], np.ndarray],
    cfg: AppConfig,
    win: GameWindow,
) -> NavScreenResult:
    """回桌／點擊前用：連續擷取多張，多數決 + 取信心最高者，減少一幀誤判。"""
    nc = _read_nav_cfg(cfg)
    n = nc["samples"]
    votes: list[NavScreenResult] = []
    for i in range(n):
        if i > 0:
            time.sleep(nc["gap_s"])
        votes.append(
            classify_nav_screen(
                capture_fn(),
                cfg,
                win,
                use_ocr=nc["use_ocr"] and (i == n - 1),
            )
        )
    from collections import Counter

    phases = [v.phase for v in votes]
    counts = Counter(phases)
    phase, agree = counts.most_common(1)[0]
    if agree < nc["min_agree"]:
        best = min(votes, key=lambda v: (v.confidence, -v.phase_scores.get(v.phase, 0)))
        best.reasons.append(f"多幀不一致 {dict(counts)}→不執行點擊")
        return NavScreenResult(
            phase=PHASE_UNKNOWN,
            label=_PHASE_LABEL[PHASE_UNKNOWN],
            action=_PHASE_ACTION[PHASE_UNKNOWN],
            confidence=0.2,
            reasons=best.reasons,
            phase_scores=best.phase_scores,
        )
    candidates = [v for v in votes if v.phase == phase]
    best = max(candidates, key=lambda v: v.confidence)
    if best.confidence < nc["min_confidence"]:
        best.reasons.append(f"多幀同意但信心 {best.confidence:.2f} < {nc['min_confidence']}")
        return NavScreenResult(
            phase=PHASE_UNKNOWN,
            label=_PHASE_LABEL[PHASE_UNKNOWN],
            action=_PHASE_ACTION[PHASE_UNKNOWN],
            confidence=best.confidence,
            reasons=best.reasons,
            phase_scores=best.phase_scores,
        )
    best.reasons.append(f"多幀確認 {agree}/{n} 一致→{phase}")
    return best


def screen_state_fast(frame: np.ndarray, cfg: AppConfig, win: GameWindow) -> str:
    """單幀畫面階段（回桌診斷用）。只有 table 代表可跟注。"""
    return classify_nav_screen(frame, cfg, win).phase


def try_resolve_table_phase(
    capture_fn: Callable[[], np.ndarray],
    cfg: AppConfig,
    win: GameWindow,
    *,
    max_wait_s: float,
    poll_s: float,
) -> str:
    """判成棋牌大廳／未知時，短等倒數或桌號出現，避免局間誤回大廳滑動。"""
    frame = capture_fn()
    if _is_kick_popup_visible(frame, cfg, win):
        return PHASE_QIPAI_SCROLL
    nav = classify_nav_screen(frame, cfg, win, use_ocr=False)
    if nav.phase == PHASE_TABLE:
        return PHASE_TABLE
    if nav.phase in (PHASE_HOME, PHASE_ENTRY):
        return nav.phase

    ok, meta = detect_in_baccarat_room(frame, cfg, win)
    if ok:
        logger.info("短等前已確認在牌桌（%s）", meta.get("method", ""))
        return PHASE_TABLE

    if nav.phase not in (PHASE_QIPAI_SCROLL, PHASE_QIPAI_READY, PHASE_UNKNOWN):
        return nav.phase

    logger.info(
        "畫面=%s，短等 %.0fs 確認是否在牌桌（等倒數／桌號）…",
        nav.label,
        max_wait_s,
    )
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        time.sleep(poll_s)
        frame = capture_fn()
        if _is_kick_popup_visible(frame, cfg, win):
            return PHASE_QIPAI_SCROLL
        ok, meta = detect_in_baccarat_room(frame, cfg, win)
        if ok:
            logger.info("短等後確認在牌桌（%s）", meta.get("method", ""))
            return PHASE_TABLE
        nav = classify_nav_screen(frame, cfg, win, use_ocr=False)
        if nav.phase == PHASE_TABLE:
            return PHASE_TABLE
        if nav.phase in (PHASE_HOME, PHASE_ENTRY):
            return nav.phase
    return nav.phase


def screen_state_for_engine(
    frame: np.ndarray,
    cfg: AppConfig,
    win: GameWindow,
    capture_fn: Callable[[], np.ndarray],
) -> str:
    """引擎用：先單幀判斷；若像棋牌大廳則短等再確認是否在房內。"""
    nc = _read_nav_cfg(cfg)
    phase = classify_nav_screen(frame, cfg, win, use_ocr=False).phase
    if phase == PHASE_TABLE:
        return PHASE_TABLE
    grace = float(nc.get("not_table_grace_s", 10.0))
    if grace <= 0:
        return phase
    if phase in (PHASE_QIPAI_SCROLL, PHASE_QIPAI_READY, PHASE_UNKNOWN):
        return try_resolve_table_phase(
            capture_fn, cfg, win, max_wait_s=grace, poll_s=float(nc.get("not_table_poll_s", 0.4))
        )
    return phase


def detect_screen(frame: np.ndarray, cfg: AppConfig, win: GameWindow) -> str:
    """相容舊呼叫：等同 classify_nav_screen().phase。"""
    return classify_nav_screen(frame, cfg, win).phase


def diagnose_screen(frame: np.ndarray, cfg: AppConfig, win: GameWindow) -> dict:
    sig = _gather_signals(frame, cfg, win)
    nav = classify_nav_screen(frame, cfg, win, use_ocr=True)
    scene: SceneFeatures | None = sig.get("scene")
    bs = _base_scale(cfg, win)
    full = _full_rect(win)
    items = [
        (_T_RANDOM, "隨機選台(入口)", _scaled_rect(cfg, win, _RECT_RANDOM)),
        (_T_QIPAI, "棋牌選單(右側)", _scaled_rect(cfg, win, _RECT_QIPAI)),
        (_T_CONFIRM, "確定鈕(被踢提示)", _scaled_rect(cfg, win, _RECT_CONFIRM)),
        (_T_CARD, "百家樂卡片", full),
    ]
    score_rows: list[dict] = []
    for name, label, rect in items:
        region = match_template_in_region(frame, name, rect, threshold=0.0, base_scale=bs)
        whole = match_template_in_region(frame, name, full, threshold=0.0, base_scale=bs)
        score_rows.append({"template": name, "label": label, "rect": rect, "region": region, "whole": whole})
    return {
        "client": (win.client_width, win.client_height),
        "base_scale": bs,
        "phase": nav.phase,
        "phase_label": nav.label,
        "suggested_action": nav.action,
        "is_qipai_hall": nav.qipai_hall,
        "card_score": nav.card_score,
        "random_score": nav.random_score,
        "menu_score": nav.menu_score,
        "qipai_tab": nav.qipai_tab,
        "table_hud": nav.table_hud,
        "card_row_score": scene.card_row if scene else 0,
        "table_ui_score": scene.table_ui if scene else 0,
        "confidence": nav.confidence,
        "reasons": nav.reasons,
        "phase_scores": nav.phase_scores,
        "ocr": nav.ocr,
        "detect_screen": nav.phase,
        "screen_state_fast": nav.phase,
        "scores": score_rows,
    }


def _confirm_dialog_rect(cfg: AppConfig, win: GameWindow) -> tuple[int, int, int, int]:
    return _scaled_rect(cfg, win, _RECT_CONFIRM)


def _is_kick_popup_visible(
    frame: np.ndarray, cfg: AppConfig, win: GameWindow
) -> bool:
    from star_follow.vision.kick_popup import is_kick_idle_popup

    return is_kick_idle_popup(frame, cfg, win=win)


def _find_popup_confirm_xy(
    frame: np.ndarray, cfg: AppConfig, win: GameWindow
) -> tuple[int, int] | None:
    from star_follow.vision.kick_popup import find_kick_confirm_xy

    return find_kick_confirm_xy(frame, cfg, win)


def dismiss_popup_if_any(
    win: GameWindow, cfg: AppConfig, capture_fn: Callable[[], np.ndarray]
) -> bool:
    """關閉『超過五局未押注』提示（須 OCR 關鍵字或確定鈕模板高分才點）。"""
    frame = capture_fn()
    if not _is_kick_popup_visible(frame, cfg, win):
        return False
    xy = _find_popup_confirm_xy(frame, cfg, win)
    if not xy:
        logger.warning("五局提示 OCR 已命中但找不到確定座標，略過點擊")
        return False
    logger.info("偵測到五局未押注提示，點『確定』(%d,%d)", xy[0], xy[1])
    click_at(win, xy[0], xy[1], backend=cfg.automation.ui_click_backend)
    time.sleep(0.85)
    if not _is_kick_popup_visible(capture_fn(), cfg, win):
        return True
    logger.warning("點過確定後仍偵測到五局提示，不再連點（避免誤判狂點）")
    return True


def _find_card(
    frame: np.ndarray, win: GameWindow, cfg: AppConfig, *, thr: float = _THR_CARD_READY
) -> tuple[int, int, float] | None:
    hit = _match_score(frame, win, _T_CARD, _full_rect(win), cfg)
    if hit and hit[2] >= thr:
        return hit
    return None


def _scroll_right_find_card(
    win: GameWindow, cfg: AppConfig, capture_fn: Callable[[], np.ndarray], *, max_drags: int = 12
) -> tuple[int, int, float] | None:
    w, h = win.client_width, win.client_height
    y = int(h * 0.42)
    x_from, x_to = int(w * 0.72), int(w * 0.24)
    for _ in range(max_drags):
        frame = capture_fn()
        card = _find_card(frame, win, cfg)
        if card:
            return card
        nav = classify_nav_screen(frame, cfg, win)
        if nav.phase == PHASE_ENTRY:
            return None
        if nav.phase == PHASE_TABLE:
            return None
        drag_client(win, x_from, y, x_to, y)
        time.sleep(0.35)
    frame = capture_fn()
    return _find_card(frame, win, cfg, thr=_THR_CARD_READY - 0.05)


def _wait_enter_table(
    capture_fn: Callable[[], np.ndarray],
    cfg: AppConfig,
    win: GameWindow,
    *,
    timeout_s: float,
    poll_s: float,
) -> bool:
    """點完隨機選台後等待進房：倒數 OCR／左下桌號／籌碼列／換桌模板。"""
    logger.info("進桌載入中，等待牌桌特徵（最多 %.0fs，不會去滑棋牌大廳）…", timeout_s)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = capture_fn()
        ok, meta = detect_in_baccarat_room(frame, cfg, win)
        if ok:
            logger.info("已進入牌桌（%s）", meta.get("method", ""))
            return True
        nav = classify_nav_screen(frame, cfg, win, use_ocr=False)
        if nav.phase == PHASE_TABLE:
            return True
        if nav.phase in (PHASE_HOME, PHASE_ENTRY):
            logger.warning("進桌等待中卻回到 %s，停止等待", nav.label)
            return False
        time.sleep(poll_s)
    logger.warning("進桌等待逾時（%.0fs）仍未偵測到牌桌特徵", timeout_s)
    return False


def _click_entry(win: GameWindow, cfg: AppConfig, frame: np.ndarray) -> None:
    hit = _match_score(frame, win, _T_RANDOM, _scaled_rect(cfg, win, _RECT_RANDOM), cfg)
    if hit is None or hit[2] < _THR_RANDOM_ENTRY:
        hit_w = _match_score(frame, win, _T_RANDOM, _full_rect(win), cfg)
        if hit_w and hit_w[2] >= _THR_RANDOM_ENTRY:
            hit = hit_w
    if hit is None or hit[2] < _THR_RANDOM_ENTRY:
        rx, ry = _scaled_pt(cfg, win, _REF_PT_RANDOM)
        logger.info("入口模板未中，改點備援『隨機選台』(%d,%d)", rx, ry)
        click_at(win, rx, ry, backend=cfg.automation.ui_click_backend)
    else:
        logger.info("百家樂入口 → 點『隨機選台』(%d,%d) score=%.2f", hit[0], hit[1], hit[2])
        click_at(win, hit[0], hit[1], backend=cfg.automation.ui_click_backend)
    time.sleep(2.0)


def _click_home_qipai(win: GameWindow, cfg: AppConfig, frame: np.ndarray) -> None:
    hit = _match_score(frame, win, _T_QIPAI, _scaled_rect(cfg, win, _RECT_QIPAI), cfg)
    if hit and hit[2] >= _THR_QIPAI_MENU:
        logger.info("首頁大廳 → 點『棋牌』(%d,%d) score=%.2f", hit[0], hit[1], hit[2])
        click_at(win, hit[0], hit[1], backend=cfg.automation.ui_click_backend)
    else:
        qx, qy = _scaled_pt(cfg, win, _REF_PT_QIPAI)
        logger.info("首頁大廳模板未中，改點備援『棋牌』(%d,%d)", qx, qy)
        click_at(win, qx, qy, backend=cfg.automation.ui_click_backend)
    time.sleep(1.2)


def _click_card_at(
    win: GameWindow, cfg: AppConfig, xy: tuple[int, int], *, tag: str = ""
) -> None:
    logger.info("棋牌大廳 → 點百家樂 (%d,%d) %s", xy[0], xy[1], tag)
    click_at(win, xy[0], xy[1], backend=cfg.automation.ui_click_backend)
    time.sleep(1.8)


def return_to_baccarat_table(
    win: GameWindow,
    cfg: AppConfig,
    capture_fn: Callable[[], np.ndarray] | None = None,
    *,
    timeout_s: float = 90.0,
) -> bool:
    cap = capture_fn or (lambda: capture_client(win))
    logger.info(
        "回桌開始：client=%dx%d base_scale=%.3f",
        win.client_width,
        win.client_height,
        _base_scale(cfg, win),
    )
    try:
        focus_window(win.hwnd)
    except Exception:
        pass
    deadline = time.monotonic() + timeout_s
    last_phase = ""
    qipai_enter_mono: float | None = None
    stuck = 0
    saved_unknown = False

    while time.monotonic() < deadline:
        if dismiss_popup_if_any(win, cfg, cap):
            continue
        nav = classify_nav_screen_confirmed(cap, cfg, win)
        phase = nav.phase
        frame = cap()

        if phase != last_phase:
            logger.info(
                "畫面判定：%s（%s）信心=%.0f%% card=%.2f random=%.2f menu=%.2f → %s",
                nav.label,
                phase,
                nav.confidence * 100,
                nav.card_score,
                nav.random_score,
                nav.menu_score,
                nav.action,
            )
            for r in nav.reasons:
                logger.info("  %s", r)
            last_phase = phase

        if phase == PHASE_UNKNOWN:
            stuck += 1
            if not saved_unknown:
                _save_unknown_frame(frame, "low_conf")
                saved_unknown = True
            if stuck % 3 == 0:
                logger.info("回桌：信心不足，等待畫面穩定…")
            time.sleep(0.5)
            continue

        if phase == PHASE_TABLE:
            logger.info("已回到牌桌")
            return True

        if phase == PHASE_ENTRY:
            _click_entry(win, cfg, frame)
            qipai_enter_mono = None
            nc = _read_nav_cfg(cfg)
            _wait_enter_table(
                cap,
                cfg,
                win,
                timeout_s=nc["enter_table_wait_s"],
                poll_s=nc["enter_table_poll_s"],
            )
            continue

        if phase in (PHASE_QIPAI_SCROLL, PHASE_QIPAI_READY):
            ok, meta = detect_in_baccarat_room(frame, cfg, win)
            if ok:
                logger.info("畫面雖像棋牌大廳，但已在牌桌內（%s），不滑動", meta.get("method", ""))
                continue

        if phase == PHASE_QIPAI_READY and nav.card_xy:
            _click_card_at(win, cfg, nav.card_xy, tag=f"score={nav.card_score:.2f}")
            qipai_enter_mono = None
            continue

        if phase == PHASE_QIPAI_SCROLL or (
            qipai_enter_mono is not None
            and (time.monotonic() - qipai_enter_mono) < _QIPAI_ENTER_GRACE_S
        ):
            card = _scroll_right_find_card(win, cfg, cap)
            if card:
                _click_card_at(win, cfg, (card[0], card[1]), tag=f"滑動後 score={card[2]:.2f}")
                qipai_enter_mono = None
            else:
                fx, fy = _scaled_pt(cfg, win, _REF_PT_CARD)
                logger.warning("滑動後仍找不到百家樂模板，嘗試備援座標 (%d,%d)", fx, fy)
                _click_card_at(win, cfg, (fx, fy), tag="備援")
                qipai_enter_mono = None
            continue

        if phase == PHASE_HOME:
            if nav.qipai_tab == "selected":
                logger.warning("首頁：棋牌分頁已亮起但還在大廳畫面，先等待…")
                time.sleep(0.8)
                continue
            _click_home_qipai(win, cfg, frame)
            qipai_enter_mono = time.monotonic()
            continue

        if qipai_enter_mono is not None:
            card = _scroll_right_find_card(win, cfg, cap)
            if card:
                _click_card_at(win, cfg, (card[0], card[1]))
            time.sleep(0.5)
            continue

        stuck += 1
        if stuck % 5 == 0:
            logger.info("回桌：未預期狀態 %s，等待…", nav.label)
        time.sleep(0.8)

    logger.warning("回桌逾時（%.0fs）", timeout_s)
    _save_unknown_frame(cap(), "timeout")
    return False
