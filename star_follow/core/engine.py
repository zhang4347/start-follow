from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

from PIL import Image

from star_follow.automation.click import click_at, get_menu_click_backend, get_ui_click_backend, wheel_at_client
from star_follow.automation.executor import BetExecutor
from star_follow.automation.menu_flow import open_stats_with_marks, retry_stats_chart_click
from star_follow.capture.screen import CaptureUnavailable, capture_client
from star_follow.capture.window import find_game_window, focus_window
from star_follow.config import AppConfig, load_config
from star_follow.core.countdown_tracker import CountdownTracker
from star_follow.core.follow_list import FollowList
from star_follow.core.risk import cap_plan
from star_follow.vision.menu_match import menu_dropdown_open
from star_follow.vision.panel import stats_table_visible
from star_follow.vision.roi import scale_point, scale_rect
from star_follow.vision.state import CountdownColor, CountdownState, read_countdown
from star_follow.vision.stats_parser import (
    parse_bottom_row_amount,
    parse_stats_table,
    resolve_follow_columns,
    verify_follow_columns,
)

logger = logging.getLogger(__name__)
from star_follow.paths import logs_dir as _logs_dir

LOG_DIR = _logs_dir()


class Phase(Enum):
    IDLE = auto()
    BET_OPEN = auto()
    STATS_READY = auto()
    LOCKED = auto()


@dataclass
class RoundContext:
    last_t: int | None = None
    round_start_t: int | None = None
    round_start_mono: float | None = None
    stats_opened: bool = False
    stats_closed: bool = False
    prefetch_done: bool = False
    ocr_done: bool = False
    resolved_columns: dict[str, int] = field(default_factory=dict)
    plan: dict[str, int] = field(default_factory=dict)
    ui_prepared: bool = False
    recovery_open_tried: bool = False
    # 本局統計表「表頭實際讀到幾個名字」：用來區分「真的沒對象」與「OCR 沒讀到表」
    header_name_count: int = 0
    # 最近一次 _build_plan「過濾前」對象實際下注的區域（含莊/閒），供換桌判斷
    # 「對象只下莊/閒（不是我們要跟的）→ 不黏桌」用。
    last_raw_bet_areas: set[str] = field(default_factory=set)


class FollowEngine:
    """
    掛機模式：僅在 T=19～20 錨定成功才開局；太晚（如 T=17）整局跳過。
    開局後：開統計 → 預定位 → T=finalize_at_t（預設 13）定稿 → 關表跟注。
    """

    def __init__(
        self,
        cfg: AppConfig | None = None,
        follow: FollowList | None = None,
        *,
        dry_run: bool = True,
    ) -> None:
        self.cfg = cfg or load_config()
        self.follow = follow or FollowList.load()
        self.dry_run = dry_run
        self.phase = Phase.IDLE
        self.ctx = RoundContext()
        self._running = False
        self._win = None
        tcfg = self.cfg.timing
        self._cd_tracker = CountdownTracker(
            anchor_at=tcfg.countdown_anchor_t,
            resync_tolerance=tcfg.countdown_resync_tolerance,
        )
        self._last_logged_t: int | None = None
        # 跨局欄位快取：上一局找到的「對象→欄位」，下一局只驗證該欄表頭名字（快），
        # 對不上才退回整排表頭 OCR；換桌時清空。
        self._col_cache: dict[str, int] = {}
        self._late_skip_logged = False
        self._ui_fail_streak = 0
        self._last_idle_cleanup_mono = 0.0
        self._recover_cooldown_until = 0.0
        self._engine_started_mono = time.perf_counter()
        self._last_locked_log_mono = 0.0
        self._session_rounds = 0
        self._session_stake = 0
        self._locked_saw_close = False  # LOCKED：是否已確認當局收掉（紅燈/低T），用於穩健接新局
        self._locked_since_mono = 0.0  # 進入 LOCKED 的時點（時間保險用）
        self._stay_idle_rounds = 0  # 掛房：連續沒下注的局數（防踢用）
        self._stay_first_track_done = False  # 掛房：是否已成功追到對象（暖機完成）
        # 餘額上傳協調用：是否「穩定待在（指定）牌桌」。上傳只在此為真且非下注/讀表階段才放行，
        # 避免上傳的抓圖/OCR 與主流程定位讀表搶資源造成卡頓。
        self._at_table_stable = False
        self._stay_absent_streak = 0  # 掛房：連續沒讀到任何指定對象的局數
        self._stay_paused = False  # 掛房：pause 模式下對象全離桌 → 停跟注/防踢/回桌
        self._stay_absent_active = False  # 掛房：目前是否處於「對象全離桌」狀態（各模式共用）
        self._stay_pause_notified = False  # 已對「對象全離桌」發過 TG
        self._last_stay_pause_log_mono = 0.0
        self._stopped_reason: str | None = None
        self._capture_unavail_since: float | None = None  # 畫面暫時抓不到的起始時點（None=正常）
        self._last_lobby_log_mono = 0.0
        self._stay_at_table_grace_until = 0.0
        self._last_idle_wait_log_mono = 0.0
        self._win_logged = False
        self._last_nav_check_mono = 0.0
        self._last_table_verify_mono = 0.0  # 掛房：上次核對「是否在指定桌」的時點
        self._cached_screen = "table"
        self._last_kick_check_mono = 0.0
        # 換房模式狀態
        self._patrol_current: int | None = None
        self._patrol_visited: set[int] = set()
        self._last_patrol_wait_log_mono = 0.0

    def _find_game(self):
        wcfg = self.cfg.window
        return find_game_window(wcfg.title_substring, title_aliases=wcfg.title_aliases or None)

    def _min_start_t(self) -> int:
        return self.cfg.timing.min_round_start_t

    def _stay_late_start_floor_t(self) -> int:
        """掛房錨不到時的最低可開局 T：比照巡房 min_enter_t（預設 8），
        只要還有足夠時間開表+下注就照跟，不再整局放棄。"""
        floor = getattr(self.cfg.room, "min_enter_t", 8) or 8
        return max(self.cfg.timing.finalize_at_t + 2, floor)

    def _stop_engine(self, reason: str) -> None:
        self._stopped_reason = reason
        self._running = False
        logger.error("安全停損：%s — 引擎停止，請人工確認後再啟動", reason)

    def _can_start_round(self, t: int | None, cd: CountdownState) -> bool:
        """掛機開局判斷。

        - 抓到 T=19～20：錨定成功，走精準軟體倒數（最佳）。
        - 沒錨到（中途進桌／漏拍）：只要綠燈且 T 還夠開表+下注（≥late floor），
          就改用 OCR 倒數照跟，不再因為錯過 1～2 秒的錨定窗而整局放棄。
        """
        if t is None or cd.color != CountdownColor.GREEN:
            return False
        timing = self.cfg.timing
        if t > timing.open_stats_at_t:
            return False  # 太早，等倒數進入開局窗
        if timing.software_countdown:
            self._cd_tracker.try_anchor(cd)
        if self._cd_tracker.anchored:
            return True  # 正常路徑：T=19～20 已錨定
        # 錨不到 → 比照巡房：還有足夠時間就用 OCR 倒數跟
        return t >= self._stay_late_start_floor_t()

    def _begin_round(self, start_t: int, frame=None) -> None:
        assert self._win is not None
        t0 = time.perf_counter()
        self._late_skip_logged = False
        self.phase = Phase.BET_OPEN
        self.ctx = RoundContext(
            last_t=start_t,
            round_start_t=start_t,
            round_start_mono=time.perf_counter(),
        )
        # 半路接局（沒在 T=19~20 錨定）時，用開局 T 起算軟體倒數：否則開統計表後
        # 表會蓋住倒數數字 → OCR 讀不到 t → 沒有軟體時鐘 → 永遠走不到定稿而卡死。
        if (
            self.cfg.timing.software_countdown
            and not self._cd_tracker.anchored
            and start_t is not None
        ):
            self._cd_tracker.force_anchor(start_t)
        mode = "錨定" if self._cd_tracker.anchored else "OCR跟(未錨定)"
        logger.info("開盤 T=%s（%s）", start_t, mode)
        focus_window(self._win.hwnd)
        # 重用 run_once 已擷取的影像，省一次截圖
        if frame is None:
            frame = capture_client(self._win)
        self._prepare_round_ui(frame)
        t_prep = time.perf_counter()
        opened = self._try_open_stats(frame, start_t)
        t_open = time.perf_counter()
        if opened:
            if not self.ctx.prefetch_done:
                self._try_prefetch(capture_client(self._win), start_t)
            if self.ctx.stats_opened:
                self.phase = Phase.STATS_READY
        logger.info(
            "開盤流程耗時 準備%.0fms 開表%.0fms 共%.0fms",
            (t_prep - t0) * 1000,
            (t_open - t_prep) * 1000,
            (time.perf_counter() - t0) * 1000,
        )

    def _skip_late_round(self, frame, t: int | None) -> None:
        """本局太晚不開流程；順便關掉可能殘留的統計表。"""
        if not self._late_skip_logged and t is not None:
            logger.info(
                "本局 T=%s 已過開局窗（需 T=%s～%s 且錨定成功），跳過本局，等下一局",
                t,
                self._min_start_t(),
                self.cfg.timing.open_stats_at_t,
            )
            self._late_skip_logged = True
        if self._panel_open_confirmed(frame) or self._menu_dropdown_open(frame):
            self._recover_ui(frame, "跳過晚接局")

    def _round_active_too_long(self) -> bool:
        """看門狗：單局在下注階段卡太久（多半 OCR 太慢／畫面異常導致走不到定稿、
        統計表關不掉），回 True 代表該強制收尾，避免永遠卡住。預設 50 秒，可用
        config.yaml 的 timing.round_watchdog_s 調整（<=0 關閉）。"""
        rs = self.ctx.round_start_mono
        if rs is None:
            return False
        budget = 50.0
        raw = self.cfg.raw.get("timing")
        if isinstance(raw, dict):
            try:
                budget = float(raw.get("round_watchdog_s", 50.0))
            except Exception:  # noqa: BLE001
                budget = 50.0
        if budget <= 0:
            return False
        return (time.perf_counter() - rs) > budget

    def _menu_button_rect(self) -> tuple[int, int, int, int]:
        return self._scale_rect("menu_button")

    def _menu_dropdown_open(self, frame) -> bool:
        try:
            return menu_dropdown_open(frame, self._menu_button_rect())
        except Exception:
            return False

    def _sync_stats_flags(self, frame) -> None:
        """以畫面為準；進行中的局不因 OCR 暫時讀不到表頭就清掉 stats_opened。"""
        open_now = self._panel_open_confirmed(frame)
        if open_now:
            self.ctx.stats_opened = True
            self.ctx.stats_closed = False
        elif self.ctx.stats_opened and self.phase in (Phase.IDLE, Phase.LOCKED):
            if not self._panel_open(frame):
                self.ctx.stats_opened = False

    def _enter_locked(self, *, note: str = "本局結束，等待下一局") -> None:
        """封盤後重置軟體倒數，下一局才能重新錨定。"""
        if self.phase != Phase.LOCKED:
            logger.info(note)
        self.phase = Phase.LOCKED
        self.ctx = RoundContext()
        self._cd_tracker.reset()
        self._last_logged_t = None
        self._locked_saw_close = False
        self._locked_since_mono = time.monotonic()

    def _dismiss_menu_dropdown(self, frame) -> bool:
        """關掉卡在畫面上的 ☰ 下拉（沒開統計表時）。"""
        if not self._menu_dropdown_open(frame):
            return True
        assert self._win is not None
        mx, my = self._menu_button_point()
        focus_window(self._win.hwnd)
        logger.info("關閉殘留選單 ☰ @(%d,%d)", mx, my)
        for attempt in range(1, 4):
            click_at(self._win, mx, my, delay_ms=(80, 150), backend="win32")
            time.sleep(self.cfg.stats.close_click_wait_s)
            check = capture_client(self._win)
            if not self._menu_dropdown_open(check):
                return True
            if attempt < 3:
                logger.info("選單仍展開，再點 ☰（第%d次）", attempt + 1)
        return not self._menu_dropdown_open(capture_client(self._win))

    def _should_close_stats(self, frame) -> bool:
        """關表判斷一律用嚴格偵測：避免被下注區紅棕色騙到狂點 ☰。"""
        return self._panel_open_confirmed(frame)

    def _force_close_stats(self, frame, *, reason: str = "") -> bool:
        assert self._win is not None
        if not self._should_close_stats(frame):
            self.ctx.stats_closed = True
            self.ctx.stats_opened = False
            return True

        mx, my = self._menu_button_point()
        focus_window(self._win.hwnd)
        tag = f"（{reason}）" if reason else ""
        logger.info("關閉統計表%s：點 ☰ @(%d,%d)", tag, mx, my)
        wait_s = self.cfg.stats.close_click_wait_s
        retries = max(1, self.cfg.stats.close_retries)

        for attempt in range(1, retries + 1):
            click_at(self._win, mx, my, delay_ms=(80, 150), backend="win32")
            time.sleep(wait_s)
            check = capture_client(self._win)
            if not self._should_close_stats(check):
                self.ctx.stats_closed = True
                self.ctx.stats_opened = False
                self._dismiss_menu_dropdown(check)
                logger.info("已關閉押注統計")
                return True
            if attempt < retries:
                logger.info("統計表仍開啟，再點 ☰（第%d/%d 次）", attempt + 1, retries)
                self._dismiss_menu_dropdown(check)

        check = capture_client(self._win)
        still = self._should_close_stats(check)
        if still:
            logger.warning("統計表關閉失敗 — 將在下一輪自動清理")
            self._save_debug_frame(check, "close_stats_fail")
            self.ctx.stats_closed = False
            self.ctx.stats_opened = True
        else:
            self.ctx.stats_closed = True
            self.ctx.stats_opened = False
        return not still

    def _recover_ui(self, frame, reason: str) -> bool:
        """掛機恢復：關統計表 + 收選單，盡量回到可開新局的狀態。"""
        logger.info("UI 恢復：%s", reason)
        ok_panel = self._force_close_stats(frame, reason=reason)
        frame2 = capture_client(self._win) if self._win else frame
        ok_menu = self._dismiss_menu_dropdown(frame2)
        ok = ok_panel and ok_menu
        self._recover_cooldown_until = time.perf_counter() + self.cfg.stats.idle_cleanup_cooldown_s
        if not ok:
            self._save_debug_frame(
                capture_client(self._win) if self._win else frame2,
                "recover_ui_fail",
            )
        return ok

    def _should_idle_cleanup(
        self,
        frame,
        t: int | None,
        cd_color: CountdownColor,
    ) -> bool:
        if not self.cfg.stats.idle_cleanup:
            return False
        now = time.perf_counter()
        if now < self._recover_cooldown_until:
            return False
        if now - self._engine_started_mono < 20.0:
            return False
        if now - self._last_idle_cleanup_mono < self.cfg.stats.idle_cleanup_cooldown_s:
            return False
        if (
            cd_color == CountdownColor.GREEN
            and t is not None
            and t >= self._min_start_t() - 1
        ):
            return False
        if self._menu_dropdown_open(frame):
            return True
        return self._panel_open_confirmed(frame)

    def _prepare_round_ui(self, frame) -> None:
        if not self.cfg.stats.force_clean_on_round_start:
            self.ctx.ui_prepared = True
            return
        if self._panel_open_confirmed(frame) or self._menu_dropdown_open(frame):
            self._recover_ui(frame, "新局開局前清理")
        self.ctx.stats_opened = False
        self.ctx.stats_closed = False
        self.ctx.ui_prepared = True

    def _abort_round(self, frame, reason: str) -> None:
        logger.warning("本局中止：%s", reason)
        self._recover_ui(frame, reason)
        self._ui_fail_streak += 1
        if self._ui_fail_streak >= 3:
            logger.error(
                "連續 %d 局 UI 異常 — 請確認 menu 標記、遊戲是否當住",
                self._ui_fail_streak,
            )
        limit = self.cfg.safety.max_ui_fail_streak
        if limit > 0 and self._ui_fail_streak >= limit:
            self._stop_engine(f"連續 {self._ui_fail_streak} 局 UI 異常")
        self._enter_locked(note=f"本局中止：{reason}")

    def _abort_round_if_too_late(self, frame, t: int | None) -> bool:
        """若已誤入流程但開局太晚（連 OCR 跟都來不及），安全中止。"""
        rs = self.ctx.round_start_t
        if rs is None or t is None:
            return False
        if rs >= self._stay_late_start_floor_t():
            return False
        logger.warning("本局開在 T=%s 過晚，中止流程", rs)
        if self._panel_open(frame) or self.ctx.stats_opened:
            self._recover_ui(frame, "開局過晚")
        self._enter_locked(note="開局過晚，本局中止")
        return True

    def _scale_rect(self, key: str) -> tuple[int, int, int, int]:
        win = self._win
        assert win is not None
        r = self.cfg.roi[key]
        return scale_rect(
            r,
            self.cfg.window.reference_width,
            self.cfg.window.reference_height,
            win.client_width,
            win.client_height,
        )

    def _panel_rects(self, frame) -> tuple[list[int], list[int] | None, list[int] | None]:
        table = self._scale_rect("stats_table") if "stats_table" in self.cfg.roi else None
        close = self._scale_rect("stats_close") if "stats_close" in self.cfg.roi else None
        panel = list(self._scale_rect("stats_panel"))
        return panel, list(table) if table else None, list(close) if close else None

    def _panel_open_confirmed(self, frame) -> bool:
        """關表／閒置清理用：以統計表淺米色底偵測，可靠且不被下注區紅棕色誤判。

        色彩判斷快（~ms）且可靠；不再退回昂貴的中文 OCR（每次 ~數百 ms），
        以免開盤前累積數秒延遲。
        """
        _panel, table, _close = self._panel_rects(frame)
        visible, _ratio = stats_table_visible(frame, table)
        return visible

    def _panel_open(self, frame) -> bool:
        """開表後快速偵測：同樣以淺米色底為主訊號（不跑 OCR）。"""
        _panel, table, _close = self._panel_rects(frame)
        visible, _ratio = stats_table_visible(frame, table)
        return visible

    def _save_debug_frame(self, frame, name: str) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / f"{name}_{int(time.time())}.png"
        Image.fromarray(frame).save(path)
        logger.info("除錯截圖 %s", path.name)

    def _scale_click(self, key: str) -> tuple[int, int] | None:
        pt = self.cfg.click_points.get(key)
        if not pt or len(pt) != 2:
            return None
        win = self._win
        assert win is not None
        return scale_point(
            pt,
            self.cfg.window.reference_width,
            self.cfg.window.reference_height,
            win.client_width,
            win.client_height,
        )

    def _open_stats_menu(self) -> bool:
        """點 ☰ → 押注統計；成功以畫面偵測為準。"""
        assert self._win is not None
        if self.cfg.stats.manual_panel:
            return False
        win = self._win
        scfg = self.cfg.stats
        retries = max(1, scfg.open_retries)
        menu_button_pt = self._scale_click("menu_button")
        chart_pt = self._scale_click("menu_chart")
        panel_rect = list(self._scale_rect("stats_panel"))
        table = (
            list(self._scale_rect("stats_table")) if "stats_table" in self.cfg.roi else None
        )
        close = (
            list(self._scale_rect("stats_close")) if "stats_close" in self.cfg.roi else None
        )
        cap = lambda: capture_client(win)

        for attempt in range(1, retries + 1):
            focus_window(win.hwnd)

            if menu_button_pt and chart_pt:
                ok, backend = open_stats_with_marks(
                    win,
                    menu_button_pt,
                    chart_pt,
                    panel_rect=panel_rect,
                    table_rect=table,
                    close_rect=close,
                    capture_fn=cap,
                    menu_delay_s=scfg.menu_delay_s,
                    stats_open_wait_s=scfg.stats_open_wait_s,
                    backends=("win32",),
                )
                if ok:
                    logger.info("開統計成功 backend=%s", backend)
                    return True
                check = capture_client(win)
                if self._menu_dropdown_open(check) and not self._panel_open(check):
                    logger.warning("開統計 第%d次：選單已開、表未開，補點柱狀圖", attempt)
                    if retry_stats_chart_click(
                        win,
                        chart_pt,
                        panel_rect=panel_rect,
                        table_rect=table,
                        close_rect=close,
                        capture_fn=cap,
                        max_wait_s=scfg.stats_open_wait_s,
                    ):
                        logger.info("補點柱狀圖後統計表已開啟")
                        return True
                logger.warning("開統計 第%d次失敗（固定座標）", attempt)
                self._save_debug_frame(check, f"open_stats_fail_a{attempt}")
                self._dismiss_menu_dropdown(check)
                continue

            # 無手動標記時才走自動推算（備援）
            from star_follow.vision.menu_match import chart_icon_candidates

            menu_btn = self._scale_rect("menu_button")
            mx = menu_btn[0] + menu_btn[2] // 2
            my = menu_btn[1] + menu_btn[3] // 2
            menu_b = get_menu_click_backend()
            ui_b = get_ui_click_backend()
            click_at(win, mx, my, backend=menu_b)
            time.sleep(0.75)
            frame = capture_client(win)
            for ox, oy, _, tag in chart_icon_candidates(frame, menu_btn):
                click_at(win, ox, oy, backend=ui_b)
                time.sleep(0.9)
                if self._panel_open(capture_client(win)):
                    logger.info("開統計成功 (%s)", tag)
                    return True

            check = capture_client(win)
            self._save_debug_frame(check, f"open_stats_fail_a{attempt}")

        return False

    def _menu_button_point(self) -> tuple[int, int]:
        pt = self._scale_click("menu_button")
        if pt:
            return pt
        r = self._scale_rect("menu_button")
        return r[0] + r[2] // 2, r[1] + r[3] // 2

    def _close_stats_panel(self, frame) -> bool:
        """關統計表 = 再點 ☰；失敗不標記為已關，下一輪會自動清理。"""
        return self._force_close_stats(frame, reason="定稿後關表")

    def _save_screenshot(self, frame, t: int | None, *, suffix: str = "") -> Path:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        path = LOG_DIR / f"ocr_T{t}_{ts}{suffix}.png"
        Image.fromarray(frame).save(path)
        logger.info("已存截圖 %s", path.name)
        return path

    def _stats_scroll_point(self) -> tuple[int, int]:
        pt = self._scale_click("stats_scroll")
        if pt:
            return pt
        r = self._scale_rect("stats_table")
        return r[0] + r[2] - 20, r[1] + r[3] // 2

    def _scroll_cfg(self) -> dict:
        raw = self.cfg.raw.get("stats_scroll")
        if isinstance(raw, dict):
            return raw
        return {}

    def _scroll_stats_to_top(self) -> None:
        """開表後先往上捲，確保首列=莊家（上次若捲動後關閉會殘留）。"""
        scroll_cfg = self._scroll_cfg()
        if scroll_cfg.get("scroll_to_top") is False:
            return
        clicks = int(scroll_cfg.get("scroll_to_top_clicks", 2))
        if clicks <= 0:
            return
        assert self._win is not None
        sx, sy = self._stats_scroll_point()
        logger.info("統計表往上捲 %d 次 @(%d,%d)", clicks, sx, sy)
        wheel_at_client(
            self._win,
            sx,
            sy,
            clicks=clicks,
            delta=int(scroll_cfg.get("delta_up", 120)),
        )
        time.sleep(float(scroll_cfg.get("wait_sec", 0.15)))

    def _bottom_read_enabled(self) -> bool:
        """1b：開表後直接滑到最底、只截一張就讀完所有邊注（表頭固定、不看莊閒）。

        預設開啟（程式內建），所以舊使用者自動更新後、即使 config.yaml 沒有新鍵也會
        生效；要關閉才在 config.yaml 設 stats_scroll.read_at_bottom: false。
        """
        enabled = bool(self._scroll_cfg().get("read_at_bottom", True))
        return enabled and bool(self._stats_rows_bottom())

    def _stats_rows_bottom(self) -> list[str]:
        """滑到最底時可見的列名。優先用 config 明列的 stats_rows_bottom；沒有就由現有
        設定推導＝stats_rows 去掉最上面的莊家 + 最後一列 stats_scroll_row（閒龍寶），
        這樣舊 config.yaml 不必新增任何鍵也能用。
        """
        explicit = self.cfg.raw.get("stats_rows_bottom")
        if explicit:
            return [str(x) for x in explicit]
        rows = list(self.cfg.stats_rows or [])
        scroll_row = self.cfg.raw.get("stats_scroll_row")
        if len(rows) >= 2 and scroll_row:
            return rows[1:] + [str(scroll_row)]
        return []

    def _scroll_stats_to_bottom(self) -> None:
        """滑到最底並停住（資料列只有一頁多一點，多滑幾下會夾在最底）。"""
        scroll_cfg = self._scroll_cfg()
        clicks = int(scroll_cfg.get("scroll_to_bottom_clicks", 5))
        if clicks <= 0:
            return
        assert self._win is not None
        sx, sy = self._stats_scroll_point()
        logger.info("統計表往下捲到底 %d 次 @(%d,%d)", clicks, sx, sy)
        wheel_at_client(
            self._win, sx, sy, clicks=clicks, delta=int(scroll_cfg.get("delta_down", -120))
        )
        time.sleep(float(scroll_cfg.get("wait_sec", 0.15)))

    def _position_stats_for_read(self) -> None:
        """讀統計表前先把捲動位置擺好：1b 開→滑到底；否則維持原本的滑到頂。"""
        if self._bottom_read_enabled():
            self._scroll_stats_to_bottom()
        else:
            self._scroll_stats_to_top()

    def _scroll_stats_down(self) -> None:
        scroll_cfg = self._scroll_cfg()
        if scroll_cfg.get("enabled") is False:
            return
        rows = int(
            scroll_cfg.get("scroll_down_rows", self.cfg.raw.get("stats_scroll_rows", 2))
        )
        if rows <= 0:
            return
        assert self._win is not None
        sx, sy = self._stats_scroll_point()
        logger.info("統計表往下捲 %d 次 @(%d,%d)", rows, sx, sy)
        wheel_at_client(
            self._win,
            sx,
            sy,
            clicks=rows,
            delta=int(scroll_cfg.get("delta_down", -120)),
        )
        time.sleep(float(scroll_cfg.get("wait_sec", 0.15)))

    def _read_countdown(self) -> CountdownState:
        assert self._win is not None
        cd_rect = self._scale_rect("countdown")
        frame = capture_client(self._win)
        cd_img = frame[
            cd_rect[1] : cd_rect[1] + cd_rect[3],
            cd_rect[0] : cd_rect[0] + cd_rect[2],
        ]
        return read_countdown(cd_img)

    def _effective_t(self, cd: CountdownState | None = None) -> tuple[int | None, CountdownColor]:
        cd = cd or self._read_countdown()
        if self.cfg.timing.software_countdown and self._cd_tracker.anchored:
            t, color = self._cd_tracker.effective(cd)
            return t, color
        return cd.seconds, cd.color

    def _refresh_effective_countdown(self) -> tuple[int | None, CountdownColor, CountdownState]:
        """局內關鍵決策前重讀倒數（軟體+OCR 合併）。"""
        cd_ocr = self._read_countdown()
        t, cd_color = self._effective_t(cd_ocr)
        cd = CountdownState(
            color=cd_color,
            seconds=t,
            confidence=cd_ocr.confidence,
            status_text=cd_ocr.status_text,
        )
        return t, cd_color, cd

    def _cache_column_hints(self) -> None:
        for entry in self.follow.active_entries():
            col = self.ctx.resolved_columns.get(entry.name)
            if col is not None:
                entry.column_index = col

    def _resolve_columns(self, frame) -> None:
        entries = self.follow.active_entries()
        names = [e.name for e in entries]
        # 跨局快取：上一局已找到全部對象 → 這局只驗證那幾欄表頭名字（單欄 OCR，快），
        # 全部相符就沿用，省掉整排中文表頭 OCR（最大宗的慢動作）。
        cached = [(n, self._col_cache[n]) for n in names if n in self._col_cache]
        if cached and len(cached) == len(names):
            vr = verify_follow_columns(frame, self.cfg, cached)
            if len(vr.resolved_columns) == len(names):
                logger.info(
                    "表頭快取命中 %.0f ms 欄位=%s", vr.elapsed_ms, vr.resolved_columns
                )
                self.ctx.resolved_columns = dict(vr.resolved_columns)
                self._cache_column_hints()
                return
            logger.info("表頭快取失效（對象換位/換桌），改整排重掃")
        targets = [(e.name, e.column_index) for e in entries]
        result = resolve_follow_columns(frame, self.cfg, targets)
        if result.elapsed_ms:
            logger.info("表頭 OCR %.0f ms", result.elapsed_ms)
        self.ctx.resolved_columns = dict(result.resolved_columns)
        self._col_cache = dict(result.resolved_columns)
        self._cache_column_hints()

    def _build_plan(
        self,
        frame,
        *,
        refresh_only: bool = False,
        include_scroll: bool = False,
        ocr_t: int | None = None,
    ) -> dict[str, int]:
        entries = self.follow.active_entries()
        targets = [(e.name, e.column_index) for e in entries]
        known = self.ctx.resolved_columns if refresh_only else None
        bottom = self._bottom_read_enabled()
        rows_override = self._stats_rows_bottom() if bottom else None

        result = parse_stats_table(
            frame,
            self.cfg,
            follow_targets=targets,
            known_columns=known,
            rows_override=rows_override,
        )
        if result.elapsed_ms:
            kind = "刷新金額" if refresh_only else "表頭+欄位"
            logger.info("統計 OCR %.0f ms (%s)", result.elapsed_ms, kind)
        self.ctx.header_name_count = len(
            [n for _, n in result.header_columns if n and n.strip()]
        )
        if not refresh_only:
            self.ctx.resolved_columns = dict(result.resolved_columns)
            self._col_cache = dict(result.resolved_columns)
            self._cache_column_hints()
        plan: dict[str, int] = {}
        for entry in entries:
            col = result.resolved_columns.get(entry.name)
            if col is None:
                seen = [n for _, n in result.header_columns if n]
                logger.warning(
                    "「%s」不在本桌（表頭: %s），跳過",
                    entry.name,
                    "、".join(seen[:8]) or "無",
                )
                continue
            bets = result.bets_by_column.get(col, {})
            for stats_name, amount in bets.items():
                bet_key = self.cfg.stats_to_bet.get(stats_name, stats_name)
                plan[bet_key] = plan.get(bet_key, 0) + amount

        # 1b 模式：已滑到底、單張截圖就含最後一列（閒龍寶），不需要再往下補讀。
        scroll_row = self.cfg.raw.get("stats_scroll_row")
        if include_scroll and not bottom and scroll_row and result.resolved_columns:
            assert self._win is not None
            self._scroll_stats_down()
            frame2 = capture_client(self._win)
            if self.cfg.timing.save_screenshot_on_ocr and ocr_t is not None:
                self._save_screenshot(frame2, ocr_t, suffix="_scroll")
            bet_key = self.cfg.stats_to_bet.get(scroll_row, scroll_row)
            min_chip = min(self.cfg.chip_values) if self.cfg.chip_values else 1000
            for name, col in result.resolved_columns.items():
                amount = parse_bottom_row_amount(frame2, self.cfg, col)
                if amount < min_chip:
                    if amount > 0:
                        logger.info(
                            "最底列 %s 讀到 %d，低於最小籌碼 %d，忽略（OCR 雜訊）",
                            name,
                            amount,
                            min_chip,
                        )
                    continue
                plan[bet_key] = plan.get(bet_key, 0) + amount
                logger.info("最底列有數字 → %s：%s %d", scroll_row, name, amount)

        # 記錄「過濾前」對象實際下注的區域（含莊/閒），供換桌邏輯判斷對象是否
        # 只下莊/閒。
        self.ctx.last_raw_bet_areas = {area for area, amt in plan.items() if amt > 0}
        plan = self._filter_follow_plan(plan)
        return cap_plan(plan, self.cfg.chip_values)

    # 兩模式都「絕不跟」莊/閒（硬規則，不受 config 影響；換桌只跟特殊邊注、
    # 掛桌也不跟莊閒，掛桌防踢補注是另一條路徑不經過這裡）。
    _ALWAYS_EXCLUDE_FOLLOW = frozenset({"莊", "閒"})

    def _filter_follow_plan(self, plan: dict[str, int]) -> dict[str, int]:
        """兩模式共用：濾掉「不跟」的下注區。莊/閒一律不跟（硬規則），
        另外再加上 config 的 follow_exclude（可額外排除其他區）。"""
        exclude = set(self.cfg.betting.follow_exclude or []) | self._ALWAYS_EXCLUDE_FOLLOW
        out: dict[str, int] = {}
        for area, amount in plan.items():
            if area in exclude:
                if amount > 0:
                    logger.info("不跟 %s（莊/閒一律不跟），略過 %d", area, amount)
                continue
            out[area] = amount
        return out

    def _try_prefetch(self, frame, t: int | None) -> None:
        if self.ctx.prefetch_done:
            return
        self._resolve_columns(frame)
        self.ctx.prefetch_done = True
        logger.info("預定位 T=%s 欄位=%s", t, self.ctx.resolved_columns)

    def _should_prefetch(self, t: int | None, panel_now: bool) -> bool:
        if self.ctx.prefetch_done or not panel_now or t is None:
            return False
        timing = self.cfg.timing
        if t <= timing.prefetch_at_t and t > timing.ocr_at_t:
            return True
        if self.ctx.round_start_mono is not None:
            elapsed = time.perf_counter() - self.ctx.round_start_mono
            if elapsed >= 0.8 and t > timing.ocr_at_t:
                return True
        return False

    def _should_finalize(self, t: int | None, panel_now: bool) -> bool:
        if self.ctx.ocr_done or t is None:
            return False
        if not (panel_now or self.ctx.stats_opened):
            return False
        rs = self.ctx.round_start_t
        if rs is not None and rs < self._stay_late_start_floor_t():
            return False
        timing = self.cfg.timing
        # 主要：到達目標定稿 T（速度已足夠，可設 13/14 留大量餘裕）
        if t <= timing.finalize_at_t:
            return True
        # 後備：軟體倒數若跳過目標 T，仍在 OCR 窗口內補定稿
        if timing.ocr_retry_at_t <= t <= timing.ocr_at_t:
            return True
        return False

    def _try_open_stats(self, frame, t: int | None) -> bool:
        """開統計表；以畫面偵測為準，不依賴 stats_closed 旗標。"""
        if self.ctx.stats_opened and (
            self.ctx.prefetch_done or self._panel_open(frame)
        ):
            self.ctx.stats_closed = False
            if self.phase == Phase.BET_OPEN:
                self.phase = Phase.STATS_READY
            return True
        self._sync_stats_flags(frame)
        if self._panel_open_confirmed(frame) or (
            self.ctx.stats_opened and self._panel_open(frame)
        ):
            if not self.ctx.stats_opened:
                self._position_stats_for_read()
            self.ctx.stats_opened = True
            self.ctx.stats_closed = False
            if self.phase == Phase.BET_OPEN:
                self.phase = Phase.STATS_READY
            return True
        if self.cfg.stats.manual_panel:
            if self._panel_open(frame):
                self.ctx.stats_opened = True
                if self.phase == Phase.BET_OPEN:
                    self.phase = Phase.STATS_READY
                logger.info("統計表已開啟（手動）T=%s", t)
            return self.ctx.stats_opened
        if t is None:
            return False
        timing = self.cfg.timing
        if t <= timing.open_stats_at_t or t <= timing.open_stats_latest_t:
            if self._open_stats_menu():
                check = capture_client(self._win)
                if not self._panel_open(check):
                    logger.warning(
                        "點了選單但統計表未確認開啟 T=%s（可能點到別的螢幕或座標偏移）",
                        t,
                    )
                    self._save_debug_frame(check, "stats_not_confirmed")
                    return False
                self._position_stats_for_read()
                self.ctx.stats_opened = True
                if self.phase == Phase.BET_OPEN:
                    self.phase = Phase.STATS_READY
                logger.info("統計表已開啟 T=%s", t)
                return True
            logger.warning("自動開統計失敗 T=%s", t)
        return False

    def _safety_allows_bet(self, plan: dict[str, int]) -> bool:
        """下注前安全檢查：單局/累計上限、局數上限。超過則跳過或停引擎。"""
        sf = self.cfg.safety
        stake = sum(v for v in plan.values() if v > 0)

        if sf.max_stake_per_round > 0 and stake > sf.max_stake_per_round:
            logger.warning(
                "本局下注 %d 超過單局上限 %d，跳過本局",
                stake,
                sf.max_stake_per_round,
            )
            return False

        if sf.max_rounds > 0 and self._session_rounds >= sf.max_rounds:
            self._stop_engine(f"已達本次下注局數上限 {sf.max_rounds}")
            return False

        if sf.max_total_stake > 0 and self._session_stake + stake > sf.max_total_stake:
            self._stop_engine(
                f"再下注將超過累計上限 {sf.max_total_stake}"
                f"（目前 {self._session_stake} + {stake}）"
            )
            return False

        return True

    def _audit_after_bet(self, t_bet: int | None) -> None:
        """下注後存截圖供事後核對（不阻塞流程）。"""
        if not self.cfg.safety.audit_screenshot or self._win is None:
            return
        try:
            frame = capture_client(self._win)
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            path = LOG_DIR / f"bet_audit_T{t_bet}_{ts}.png"
            Image.fromarray(frame).save(path)
            logger.info("下注稽核截圖 %s 計畫=%s", path.name, self.ctx.plan)
        except Exception:
            logger.exception("下注稽核截圖失敗")

    def _can_bet_now(self, t: int | None, cd_color: CountdownColor) -> bool:
        """只要還在可下注窗（綠燈、T≥min）就下；不設上限（越早下越好）。"""
        if t is None:
            return False
        if cd_color == CountdownColor.RED and t <= 3:
            return False
        return t >= self.cfg.timing.min_bet_t

    def _finalize_at_ocr(self, frame, t: int | None, cd: CountdownState) -> None:
        """定稿：刷新金額 → 關表 → 跟注（完成後重新取 T）。"""
        assert self._win is not None
        win = self._win
        t0 = time.perf_counter()

        # 防踢優先（慢機器保險）：對象已確認離桌、且本局再沒下注就達補注門檻時，
        # 不做後面 4~5 秒的統計 OCR（會把下注窗吃掉→補注失敗→被踢），改為立刻關表、
        # 用還有效的倒數補一手防踢。下一局再讀統計偵測對象是否回來。
        if self._should_early_anti_kick():
            if not self._close_stats_panel(capture_client(win)):
                self._recover_ui(capture_client(win), "防踢前關表失敗")
            t_bet, cd_color = self._read_cd_real()
            bet_t = t_bet if t_bet is not None else t
            bet_color = cd_color if t_bet is not None else cd.color
            if self._should_count_stay_idle():
                self._stay_idle_rounds += 1
            logger.info(
                "對象離桌＋將達防踢門檻：跳過慢統計，提前補注（T=%s）", bet_t
            )
            if not self._maybe_anti_kick(frame, bet_t, bet_color):
                self._save_round(frame, t_bet, bet=False)
            self._enter_locked()
            return

        # 1b：定稿讀數前確保停在最底（表頭固定，一張就含所有邊注），並重取畫面
        if self._bottom_read_enabled():
            self._scroll_stats_to_bottom()
            frame = capture_client(win)

        refresh = bool(self.ctx.resolved_columns)
        self.ctx.plan = self._build_plan(
            frame,
            refresh_only=refresh,
            include_scroll=True,
            ocr_t=t if self.cfg.timing.save_screenshot_on_ocr else None,
        )
        self.ctx.ocr_done = True
        self._update_stay_presence()
        first_track = self._mark_first_track_success()

        if self.cfg.timing.save_screenshot_on_ocr:
            self._save_screenshot(frame, t)

        if not self._close_stats_panel(capture_client(win)):
            self._recover_ui(capture_client(win), "定稿後關表失敗")

        # 關表後用真實 OCR 重判時間（不用會漂移的軟體時鐘），盡快下注
        t_bet, cd_color = self._read_cd_real()
        elapsed = time.perf_counter() - t0
        logger.info("定稿耗時 %.1f 秒，關表後真實倒數 T=%s（%s）", elapsed, t_bet, cd_color.name)

        if not self.ctx.plan:
            if first_track and self._maybe_first_track_insurance(frame, t_bet, cd_color):
                self._enter_locked()
                return
            if self._should_count_stay_idle():
                self._stay_idle_rounds += 1
                logger.info("跟注計畫為空（連續未下注 %d 局）", self._stay_idle_rounds)
            else:
                logger.info(
                    "跟注計畫為空（表頭未讀穩／尚未追到對象，不計入防踢閒置；"
                    "若表頭有名字但對象不在則仍會累計）"
                )
            if not self._maybe_anti_kick(frame, t_bet, cd_color):
                self._save_round(frame, t_bet, bet=False)
            self._enter_locked()
            return

        logger.info("跟注計畫: %s", self.ctx.plan)

        if not self._bet_gate_after_close(t_bet, cd_color):
            logger.warning("關表後讀不到倒數（多半已封盤/過場），跳過執行（定稿時 T=%s）", t)
            if self._should_count_stay_idle():
                self._stay_idle_rounds += 1
            self._save_round(frame, t_bet, bet=False)
            self._enter_locked()
            return

        if not self._safety_allows_bet(self.ctx.plan):
            self._save_round(frame, t_bet, bet=False)
            self._enter_locked()
            return

        executor = BetExecutor(self.cfg, win, dry_run=self.dry_run)
        executor.execute(self.ctx.plan)
        stake = sum(v for v in self.ctx.plan.values() if v > 0)
        self._session_rounds += 1
        self._session_stake += stake
        self._stay_idle_rounds = 0  # 有下注就歸零
        logger.info(
            "本局下注 %d，累計 %d 局 / %d",
            stake,
            self._session_rounds,
            self._session_stake,
        )
        self._save_round(frame, t_bet, bet=True)
        self._audit_after_bet(t_bet)
        self._ui_fail_streak = 0
        self._enter_locked()

    def _notify_targets_gone(self, text: str) -> None:
        """固定掛桌：對象異動時用 Telegram 通知（背景執行緒，避免阻塞）。

        只要設了 bot_token / chat_id 就會送（不受餘額回報 enabled 影響）。
        """
        tg = self.cfg.telegram
        if not tg.bot_token or not tg.chat_id:
            return
        import threading

        from star_follow.notify.telegram import send_message

        threading.Thread(
            target=send_message,
            args=(tg.bot_token, tg.chat_id, text),
            daemon=True,
        ).start()

    def _stay_return_to_table(self, win) -> bool:
        """掛房被踢回大廳：自動導覽回任一百家樂桌；若有指定桌號，再切到該桌號。"""
        from star_follow.automation import lobby_nav
        from star_follow.automation.room_nav import read_current_table, switch_to_table

        ok = lobby_nav.return_to_baccarat_table(win, self.cfg, lambda: capture_client(win))
        if not ok:
            logger.warning("自動回桌失敗（暫時導覽不到牌桌），稍後重試")
            return False
        target = self.cfg.room.stay_table
        if target:
            # 隨機進桌後不一定是指定桌號，讀目前桌號；不符就切過去
            cur = read_current_table(capture_client(win), self.cfg, win)
            if cur != target:
                logger.info("回到 No.%s，需切換到指定桌號 No.%s", cur, target)
                if switch_to_table(win, self.cfg, target, capture_fn=lambda: capture_client(win)):
                    logger.info("已切換回指定桌號 No.%s", target)
                else:
                    logger.warning("切換到指定桌號 No.%s 失敗，下一輪重試", target)
                    return False
            else:
                logger.info("回到的就是指定桌號 No.%s", target)
        # 重置局狀態，下一輪正常跑
        self.phase = Phase.IDLE
        self.ctx = RoundContext()
        self._cd_tracker.reset()
        self._col_cache = {}  # 換桌後欄位版面不同，清掉跨局快取，下一局整排重掃
        self._stay_idle_rounds = 0
        self._stay_first_track_done = False
        self._stay_at_table_grace_until = time.monotonic() + 30.0
        logger.info("掛房已回到牌桌，恢復跟注（30s 內不因畫面閃爍重跑回桌）")
        return True

    def _stay_ensure_target_table(self, win) -> bool:
        """掛房固定桌號：人在牌桌、但不是指定桌（例如回桌時指定桌滿桌、落在別桌）→
        持續嘗試切回指定桌。回傳 True 表示本輪已處理（呼叫端應結束本輪）。

        關鍵：每輪都獨立用左下角桌號重新判斷，所以
          - 滿桌切不過去 → 下一輪再試（一直優先回指定桌）；
          - 切桌過程又被踢出 → 下一輪會先被「不在牌桌」分支接住、重新進場，
            不會卡在「一直換房卻沒發現又被踢」的狀態。
        讀不到桌號就不動作（避免誤切）。
        """
        target = self.cfg.room.stay_table
        if self.cfg.room.mode != "stay" or not target:
            return False
        now = time.monotonic()
        if now - self._last_table_verify_mono < 5.0:
            return False
        self._last_table_verify_mono = now
        if now < self._stay_at_table_grace_until:
            return False
        from star_follow.automation.room_nav import read_current_table, switch_to_table

        cur = read_current_table(capture_client(win), self.cfg, win)
        if cur is None or cur == target:
            return False
        logger.info("掛房：目前在 No.%s，非指定桌 No.%s → 嘗試切回", cur, target)
        if switch_to_table(win, self.cfg, target, capture_fn=lambda: capture_client(win)):
            logger.info("已切回指定桌號 No.%s", target)
            self.phase = Phase.IDLE
            self.ctx = RoundContext()
            self._cd_tracker.reset()
            self._col_cache = {}
            self._stay_first_track_done = False
            self._stay_at_table_grace_until = time.monotonic() + 30.0
        else:
            logger.warning("指定桌 No.%s 目前切不過去（可能滿桌），下一輪再試", target)
        return True

    def _update_stay_presence(self) -> None:
        """掛房：依本局是否讀到任何指定對象，更新「對象全離桌→暫停／停止」狀態。

        關鍵：只有「確實讀到統計表（表頭有名字）但沒有任何一個是追蹤對象」才算
        真的離桌；若整張表頭一個名字都讀不到（OCR 失敗／表沒開好）就不計入，避免
        單次讀取失誤誤判成對象消失而錯誤停止程式。
        """
        if self.cfg.room.mode != "stay" or not self.cfg.room.stay_pause_when_targets_absent:
            return
        on_absent = (getattr(self.cfg.room, "stay_on_absent", "keep") or "keep").lower()
        present = len(self.ctx.resolved_columns) > 0
        table_tag = f"No.{self.cfg.room.stay_table}" if self.cfg.room.stay_table else "本桌"
        if present:
            if self._stay_absent_active:
                logger.info("追蹤對象回到%s，恢復跟注", table_tag)
                self._notify_targets_gone(f"追蹤對象已回到 {table_tag}，恢復跟注")
            self._stay_absent_streak = 0
            self._stay_absent_active = False
            self._stay_paused = False
            self._stay_pause_notified = False
            return
        if self.ctx.header_name_count <= 0:
            logger.info("本局統計表沒讀到任何名字（OCR 失敗／表未開好），不計入離桌判定，重試")
            return
        self._stay_absent_streak += 1
        logger.info(
            "本局表頭有 %d 個名字但無任何追蹤對象（連續離桌 %d 局）",
            self.ctx.header_name_count,
            self._stay_absent_streak,
        )
        need = max(1, self.cfg.room.stay_absent_rounds_to_pause)
        if self._stay_absent_streak < need:
            return
        self._stay_absent_active = True
        self._stay_first_track_done = False  # 對象確認不在；回來後可再觸發首次保險閒注
        msg = f"{table_tag} 統計表未見任何追蹤對象"

        if on_absent == "stop":
            self._stay_paused = True
            if not self._stay_pause_notified:
                self._stay_pause_notified = True
                self._notify_targets_gone(f"{msg}，程式即將停止")
            logger.warning(
                "連續 %d 局都沒有任何指定對象在%s → 停止程式",
                self._stay_absent_streak, table_tag,
            )
            self.stop()
            return

        if on_absent == "pause":
            self._stay_paused = True
            if not self._stay_pause_notified:
                self._stay_pause_notified = True
                self._notify_targets_gone(f"{msg}，暫停跟注（不防踢、被踢不回桌）")
            logger.warning(
                "連續 %d 局都沒有任何指定對象在%s → 暫停跟注（不防踢、被踢也不自動回桌）",
                self._stay_absent_streak, table_tag,
            )
            return

        # keep（預設）：只通知，程式照常跑——繼續防踢補注、被踢照樣自動回桌，
        # 待對象某局又出現就自動恢復跟注。
        self._stay_paused = False
        if not self._stay_pause_notified:
            self._stay_pause_notified = True
            self._notify_targets_gone(
                f"{msg}，持續防踢並守住{table_tag}，待對象回來自動恢復跟注"
            )
        now = time.perf_counter()
        if now - self._last_stay_pause_log_mono >= 15.0:
            logger.info(
                "對象全離桌（keep）：持續防踢補注／被踢自動回桌，守住%s 待對象回來",
                table_tag,
            )
            self._last_stay_pause_log_mono = now

    def _stay_table_read_ok(self) -> bool:
        """本局統計表表頭是否確實讀到名字（非 OCR 暖機／表沒開好）。"""
        return self.ctx.header_name_count > 0

    def _stay_target_confirmed_absent(self) -> bool:
        """表頭讀得到其他玩家，但追蹤對象不在本桌（真的離桌，不是暖機讀不到）。"""
        if self.cfg.room.mode != "stay":
            return False
        if not self._stay_table_read_ok():
            return False
        return len(self.ctx.resolved_columns) == 0

    def _should_count_stay_idle(self) -> bool:
        """掛房防踢：何時累計「我方未下注」局數。

        - 已追到對象：照常累計。
        - 表頭讀得到名字但對象不在（或已確認離桌）：照常累計並防踢。
        - 表頭讀不到（OCR 暖機）：不累計，避免與「真的不在房」混淆。
        """
        if self.cfg.room.mode != "stay":
            return True
        if self._stay_first_track_done:
            return True
        if self._stay_absent_active or self._stay_target_confirmed_absent():
            return True
        return False

    def _mark_first_track_success(self) -> bool:
        """本局是否為「首次成功追到對象」。回傳 True 代表剛進入可跟注狀態。

        必須本局 resolved_columns 有值；表頭讀不到時不算追到。
        """
        if self.cfg.room.mode != "stay" or self._stay_first_track_done:
            return False
        if not self.ctx.resolved_columns:
            return False
        if not self._stay_table_read_ok():
            return False
        self._stay_first_track_done = True
        return True

    def _anti_kick_plan(self) -> dict[str, int] | None:
        bcfg = self.cfg.betting
        if not bcfg.anti_kick_enabled or self._stay_paused:
            return None
        amount = bcfg.anti_kick_amount or (min(self.cfg.chip_values) if self.cfg.chip_values else 1000)
        side = bcfg.anti_kick_side or "閒"
        return {side: amount}

    def _place_anti_kick_bet(
        self,
        frame,
        t_bet: int | None,
        cd_color: CountdownColor,
        *,
        reason: str,
    ) -> bool:
        plan = self._anti_kick_plan()
        if not plan:
            return False
        if not self._can_bet_now(t_bet, cd_color):
            logger.info("%s：已過可下注時間 T=%s，本局先不補", reason, t_bet)
            return False
        if not self._safety_allows_bet(plan):
            return False
        side, amount = next(iter(plan.items()))
        logger.info("%s %s %d", reason, side, amount)
        assert self._win is not None
        BetExecutor(self.cfg, self._win, dry_run=self.dry_run).execute(plan)
        self.ctx.plan = plan
        self._session_rounds += 1
        self._session_stake += amount
        self._stay_idle_rounds = 0
        self._save_round(frame, t_bet, bet=True)
        self._audit_after_bet(t_bet)
        self._ui_fail_streak = 0
        return True

    def _maybe_first_track_insurance(
        self, frame, t_bet: int | None, cd_color: CountdownColor
    ) -> bool:
        """首次追到對象且本局無邊注可跟：先補一手閒，抵消 OCR 暖機期遊戲端的閒置計數。"""
        bcfg = self.cfg.betting
        if not bcfg.anti_kick_first_track_bet:
            return False
        return self._place_anti_kick_bet(
            frame,
            t_bet,
            cd_color,
            reason="首次追到對象，防踢保險補注",
        )

    def _should_early_anti_kick(self) -> bool:
        """是否本局直接補防踢、跳過慢統計 OCR。

        條件：開了防踢、沒被暫停、對象已『確認離桌』(連續離桌達門檻)、且這局再沒下注
        累計就會達到補注門檻。此時沒有對象可跟，提前補注純賺（避免慢機器被踢）。
        對象在場時不走這條（仍照常讀統計跟注）。
        """
        bcfg = self.cfg.betting
        if not bcfg.anti_kick_enabled or self._stay_paused:
            return False
        if not self._stay_absent_active:
            return False
        return self._stay_idle_rounds + 1 >= bcfg.anti_kick_idle_rounds

    def _maybe_anti_kick(self, frame, t_bet: int | None, cd_color: CountdownColor) -> bool:
        """掛房防踢：連續多局沒下注時，自己補一手最小注（莊/閒）避免被系統踢出。

        回傳是否真的補注。補不成（時間已過）不歸零，下一局再試。
        """
        if not self._should_count_stay_idle():
            return False
        bcfg = self.cfg.betting
        if not bcfg.anti_kick_enabled:
            return False
        if self._stay_idle_rounds < bcfg.anti_kick_idle_rounds:
            return False
        return self._place_anti_kick_bet(
            frame,
            t_bet,
            cd_color,
            reason=f"連續 {self._stay_idle_rounds} 局未下注，防踢補注",
        )

    def _in_active_betting_phase(self) -> bool:
        return self.phase in (Phase.BET_OPEN, Phase.STATS_READY, Phase.LOCKED)

    def _lobby_nav_interval_s(self) -> float:
        raw = self.cfg.raw.get("nav_confirm")
        if isinstance(raw, dict):
            return float(raw.get("engine_nav_interval_s", 1.5))
        return 1.5

    def _engine_lobby_check(
        self, win, frame, cap: Callable
    ) -> tuple[bool, str]:
        """回傳 (是否 early-return, screen)。局內跳過大廳導覽以保 T 軸速度。"""
        from star_follow.automation import lobby_nav

        if self._in_active_betting_phase():
            return False, "table"

        now = time.monotonic()
        if now < self._stay_at_table_grace_until:
            return False, "table"

        interval = self._lobby_nav_interval_s()
        if (
            self._cached_screen == "table"
            and now - self._last_nav_check_mono < interval
        ):
            return False, "table"

        if now - self._last_kick_check_mono >= 2.0:
            self._last_kick_check_mono = now
            if lobby_nav.dismiss_popup_if_any(win, self.cfg, cap):
                return True, self._cached_screen

        frame = capture_client(win)
        screen = lobby_nav.screen_state_fast_for_engine(frame, self.cfg, win, cap)
        self._last_nav_check_mono = now
        self._cached_screen = screen
        return False, screen

    def run_once(self) -> bool:
        win = self._find_game()
        if not win:
            now = time.perf_counter()
            if now - getattr(self, "_last_nowin_log_mono", 0.0) >= 15.0:
                logger.warning(
                    "找不到視窗「%s」（請確認：星城已開、已進百家樂桌、未最小化、"
                    "且星城與本程式皆用管理員執行；可設 模式=視窗診斷 查看實際標題）",
                    self.cfg.window.title_substring,
                )
                self._last_nowin_log_mono = now
            if self.phase != Phase.IDLE:
                self.phase = Phase.IDLE
                self.ctx = RoundContext()
                self._cd_tracker.reset()
            self._win = None
            return False
        if self._win is None:
            self._win_logged = False
        self._win = win
        frame = capture_client(win)
        # 預設本圈「未確認穩定在桌」；確認在桌且不需回桌/切桌時才在下方設為 True。
        # 任何在確認前就 return 的分支（不在桌、回桌中、切桌中）都會維持 False → 擋住上傳。
        self._at_table_stable = False

        from star_follow.automation import lobby_nav

        cap = lambda: capture_client(win)
        if self._in_active_betting_phase():
            early, screen = self._engine_lobby_check(win, frame, cap)
            if early:
                return True
            if screen != "table" and time.monotonic() < self._stay_at_table_grace_until:
                return True
            if screen != "table":
                if self.phase != Phase.IDLE:
                    self.phase = Phase.IDLE
                    self.ctx = RoundContext()
                    self._cd_tracker.reset()
                    self._last_logged_t = None
                if self._stay_paused:
                    now = time.perf_counter()
                    if now - self._last_stay_pause_log_mono >= 15.0:
                        logger.info("追蹤對象已離桌，暫停中，不自動回桌（畫面=%s）", screen)
                        self._last_stay_pause_log_mono = now
                    return True
                now = time.perf_counter()
                if now - self._last_lobby_log_mono >= 10.0:
                    logger.info("掛房偵測到不在牌桌（%s），自動回桌…", screen)
                    self._last_lobby_log_mono = now
                self._stay_return_to_table(win)
                return True
        else:
            # 掛房 IDLE：節流大廳檢查，不每圈 prepare（避免誤點五局、拖慢 T 軸）
            early, screen = self._engine_lobby_check(win, frame, cap)
            if early:
                return True
            if screen != "table":
                if self._stay_paused:
                    now = time.perf_counter()
                    if now - self._last_stay_pause_log_mono >= 15.0:
                        logger.info("追蹤對象已離桌，暫停中，不自動回桌（畫面=%s）", screen)
                        self._last_stay_pause_log_mono = now
                    return True
                now = time.perf_counter()
                if now - self._last_lobby_log_mono >= 10.0:
                    logger.info("掛房偵測到不在牌桌（%s），自動回桌…", screen)
                    self._last_lobby_log_mono = now
                self._stay_return_to_table(win)
                return True
            self._cached_screen = "table"
            # 在牌桌但可能不是指定桌（回桌時指定桌滿桌、落在別桌）→ 持續切回指定桌
            if self._stay_ensure_target_table(win):
                return True

        frame = capture_client(win)
        # 已確認在（指定）牌桌、不需回桌/切桌 → 允許餘額上傳在空檔（IDLE/LOCKED）進行。
        self._at_table_stable = True

        if not self._win_logged:
            focus_window(win.hwnd)
            logger.info(
                "已鎖定「%s」client=%dx%d — 請勿遮住遊戲視窗，點擊會送到此視窗",
                win.title,
                win.client_width,
                win.client_height,
            )
            self._win_logged = True

        cd_rect = self._scale_rect("countdown")
        cd_img = frame[cd_rect[1] : cd_rect[1] + cd_rect[3], cd_rect[0] : cd_rect[0] + cd_rect[2]]
        cd_ocr = read_countdown(cd_img)
        timing = self.cfg.timing
        ocr_t = cd_ocr.seconds
        ocr_color = cd_ocr.color
        cd = cd_ocr
        if timing.software_countdown:
            if self.phase != Phase.LOCKED:
                self._cd_tracker.try_anchor(cd_ocr)
            t, cd_color = self._cd_tracker.effective(cd_ocr)
            cd = CountdownState(
                color=cd_color,
                seconds=t,
                confidence=cd.confidence,
                status_text=cd.status_text,
            )
        else:
            t = ocr_t
            cd_color = ocr_color

        if (
            t is not None
            and t != self._last_logged_t
            and self.phase in (Phase.BET_OPEN, Phase.STATS_READY)
        ):
            src = "軟體" if timing.software_countdown and self._cd_tracker.anchored else "OCR"
            logger.info("T=%s (%s)", t, src)
            self._last_logged_t = t

        panel_open = self._panel_open(frame)
        anchor_t = timing.countdown_anchor_t

        # 看門狗：下注階段若卡太久（沒走到定稿），強制關表並重置，避免永遠卡死。
        if self.phase in (Phase.BET_OPEN, Phase.STATS_READY) and self._round_active_too_long():
            logger.warning("本局在下注階段逾時未定稿（卡住保護）→ 強制關表收尾、重置等下一局")
            self._recover_ui(capture_client(self._win), "單局逾時看門狗")
            self._enter_locked(note="本局逾時，重置等下一局")
            return True

        if self.phase == Phase.IDLE:
            if cd_color == CountdownColor.GREEN and t is not None:
                if self._can_start_round(t, cd):
                    self._begin_round(t, frame=frame)
                elif t > timing.open_stats_at_t:
                    self._skip_late_round(frame, t)
            else:
                now = time.perf_counter()
                if now - self._last_idle_wait_log_mono >= 20.0:
                    logger.info(
                        "等待開局（IDLE）螢幕 T=%s %s",
                        ocr_t,
                        ocr_color.name if ocr_color else "?",
                    )
                    self._last_idle_wait_log_mono = now
            if self._should_idle_cleanup(frame, t, cd_color):
                self._recover_ui(frame, "掛機閒置清理")
                self._last_idle_cleanup_mono = time.perf_counter()
            if self.phase == Phase.IDLE:
                return True

        if self.phase == Phase.BET_OPEN:
            t, cd_color, cd = self._refresh_effective_countdown()
            if not self.ctx.ui_prepared:
                self._prepare_round_ui(capture_client(self._win))
            if self._abort_round_if_too_late(frame, t):
                return True
            if self._try_open_stats(frame, t):
                if not self.ctx.prefetch_done:
                    self._try_prefetch(capture_client(self._win), t)
            if self.ctx.stats_opened:
                self.phase = Phase.STATS_READY
            if (
                cd_color == CountdownColor.RED
                and t is not None
                and t <= 3
                and not self._cd_tracker.anchored
            ):
                self._enter_locked(note="封盤（未錨定）")
            return True

        if self.phase == Phase.STATS_READY:
            t, cd_color, cd = self._refresh_effective_countdown()
            if self._abort_round_if_too_late(frame, t):
                return True
            self._sync_stats_flags(frame)
            self._try_open_stats(frame, t)

            panel_now = panel_open or self._panel_open(frame)

            if (
                not panel_now
                and not self.ctx.stats_opened
                and not self.ctx.recovery_open_tried
                and t is not None
                and t <= timing.prefetch_at_t
                and t > timing.ocr_at_t
            ):
                self.ctx.recovery_open_tried = True
                logger.warning("統計表仍未開啟 T=%s，嘗試恢復開啟", t)
                if self._open_stats_menu():
                    self._position_stats_for_read()
                    self.ctx.stats_opened = True
                    panel_now = True

            if self._should_prefetch(t, panel_now):
                self._try_prefetch(capture_client(self._win), t)

            if self._should_finalize(t, panel_now):
                logger.info("定稿 T=%s", t)
                self._finalize_at_ocr(frame, t, cd)
            elif (
                t is not None
                and t <= timing.ocr_retry_at_t
                and not self.ctx.ocr_done
                and not (panel_now or self.ctx.stats_opened)
            ):
                self._abort_round(frame, "定稿前統計表從未開啟")
            elif (
                cd_color == CountdownColor.RED
                and t is not None
                and t <= 3
                and not self._cd_tracker.anchored
            ):
                self._recover_ui(frame, "封盤前收尾")
                self._enter_locked(note="封盤前收尾")
            elif t is not None and t == 0 and not self.ctx.ocr_done and self.ctx.stats_opened:
                logger.warning("軟體 T=0 尚未定稿，緊急定稿")
                self._finalize_at_ocr(frame, t, cd)
            return True

        if self.phase == Phase.LOCKED:
            now = time.perf_counter()
            if now - self._last_locked_log_mono >= 15.0:
                logger.info(
                    "等待下一局（LOCKED）螢幕倒數=%s %s",
                    ocr_t,
                    ocr_color.name if ocr_color else "?",
                )
                self._last_locked_log_mono = now
            # 先確認當局已收掉（紅燈／倒數歸零附近／無綠燈，或鎖局已逾 12s 當局必已結束）
            # → 之後遇到綠燈新局（T≥floor）就接，避免當機漏掉 19~20 而整局不跟
            if (
                ocr_color != CountdownColor.GREEN
                or (ocr_t is not None and ocr_t <= 4)
                or (time.monotonic() - self._locked_since_mono >= 22.0)
            ):
                self._locked_saw_close = True
            new_round = (
                ocr_color == CountdownColor.GREEN
                and ocr_t is not None
                and (
                    ocr_t >= self._min_start_t()
                    or (self._locked_saw_close and ocr_t >= self._stay_late_start_floor_t())
                )
            )
            if new_round:
                frame2 = capture_client(self._win)
                if self._panel_open_confirmed(frame2) or self._menu_dropdown_open(frame2):
                    self._recover_ui(frame2, "封盤轉下一局")
                logger.info("下一局開盤窗 OCR T=%s", ocr_t)
                self.phase = Phase.IDLE
                self.ctx = RoundContext()
                self._last_logged_t = None
                self._late_skip_logged = False
                if timing.software_countdown:
                    self._cd_tracker.try_anchor(cd_ocr)
                if self._can_start_round(ocr_t, cd_ocr):
                    self._begin_round(ocr_t, frame=frame2)
            if self.phase == Phase.LOCKED:
                return True

        return True

    def _save_round(self, frame, t: int | None, *, bet: bool) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        meta = {
            "t": t,
            "plan": self.ctx.plan,
            "resolved_columns": self.ctx.resolved_columns,
            "dry_run": self.dry_run,
            "bet_executed": bet,
        }
        (LOG_DIR / f"round_{ts}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ====================== 換房（巡房）模式 ======================
    def _read_cd(self, frame) -> CountdownState:
        """直接 OCR 螢幕倒數（換房進局時間不定，不用軟體錨定）。"""
        r = self._scale_rect("countdown")
        img = frame[r[1] : r[1] + r[3], r[0] : r[0] + r[2]]
        return read_countdown(img)

    def _read_cd_stable(self, tries: int = 8, interval: float = 0.12) -> tuple[int | None, CountdownColor]:
        """重讀倒數，容忍瞬間 OCR 讀不到（剛關表時畫面可能在過場）。"""
        assert self._win is not None
        color = CountdownColor.OTHER
        for _ in range(tries):
            t, color = self._effective_t(self._read_cd(capture_client(self._win)))
            if t is not None:
                return t, color
            time.sleep(interval)
        return None, color

    def _read_cd_real(self, tries: int = 6, interval: float = 0.1) -> tuple[int | None, CountdownColor]:
        """關表後用『真實 OCR』重判倒數（刻意不走軟體時鐘，避免長時間蓋住倒數造成的
        系統時間 vs 遊戲時間漂移，害得明明還有時間卻被判成沒時間下注）。"""
        assert self._win is not None
        color = CountdownColor.OTHER
        for _ in range(tries):
            cd = self._read_cd(capture_client(self._win))
            if cd.seconds is not None:
                return cd.seconds, cd.color
            color = cd.color
            time.sleep(interval)
        return None, color

    def _bet_gate_after_close(self, t: int | None, color: CountdownColor) -> bool:
        """關表後是否還能下注（寬鬆）：只要還讀得到倒數（>=1 秒）就下，盡快下；
        只有完全讀不到倒數（多半在過場/開牌）或已歸零才放棄。"""
        if t is None:
            return False
        return t >= 1

    def _patrol_next_target(self, cur: int | None) -> int | None:
        seq = sorted(self.cfg.room.tables)
        if not seq:
            return None
        avail = [n for n in seq if n not in self._patrol_visited]
        if not avail:
            self._patrol_visited.clear()
            logger.info("巡房：已巡完一輪，重新開始")
            avail = list(seq)
        if cur is not None:
            ahead = [n for n in avail if n > cur]
            return ahead[0] if ahead else avail[0]
        return avail[0]

    def _patrol_advance(self, cur: int | None) -> bool:
        """換到下一桌（跳過滿桌/到不了的桌）。成功回傳 True。"""
        from star_follow.automation.room_nav import read_current_table, switch_to_table

        assert self._win is not None
        for _ in range(len(self.cfg.room.tables) + 1):
            target = self._patrol_next_target(cur)
            if target is None:
                return False
            self._patrol_visited.add(target)
            if switch_to_table(self._win, self.cfg, target):
                # 不要直接相信「目標就是落點」：用左下角桌號 OCR 確認真正進到哪一桌，
                # 避免誤判（例如點到別列、或滿桌沒換成卻以為換成了）。
                time.sleep(0.4)
                actual = read_current_table(capture_client(self._win), self.cfg, self._win)
                if actual is None:
                    self._patrol_current = target
                    logger.info("已換到 No.%d（桌號未能確認）", target)
                else:
                    self._patrol_current = actual
                    self._patrol_visited.add(actual)
                    if actual == target:
                        logger.info("已換到 No.%d", actual)
                    else:
                        logger.warning(
                            "換桌落點為 No.%d（目標 No.%d，可能滿桌或誤判）；就掃此桌",
                            actual,
                            target,
                        )
                self._col_cache = {}  # 換桌後欄位版面不同，清掉跨局快取
                return True
            logger.info("No.%d 換桌未成功（滿桌或點不到），跳過", target)
            cur = target
        return False

    def _patrol_scan_table(self) -> tuple[bool, dict[str, int]]:
        """開統計表→OCR 表頭/金額→關表。

        回傳 (本桌是否有追蹤對象, 跟注計畫)。計畫空代表對象目前未下注。
        """
        assert self._win is not None
        win = self._win
        if not self._open_stats_menu():
            logger.info("本桌開統計失敗")
            self._force_close_stats(capture_client(win), reason="巡房開統計失敗")
            return (False, {})
        self._position_stats_for_read()
        self.ctx = RoundContext()
        frame = capture_client(win)
        plan = self._build_plan(frame, refresh_only=False, include_scroll=True)
        present = bool(self.ctx.resolved_columns)
        self.ctx.plan = plan
        self.ctx.ocr_done = True
        if not self._close_stats_panel(capture_client(win)):
            self._force_close_stats(capture_client(win), reason="巡房關表")
        return (present, plan)

    def _patrol_place(self, plan: dict[str, int], t_bet: int | None) -> bool:
        """實際下注 + 記錄。"""
        assert self._win is not None
        BetExecutor(self.cfg, self._win, dry_run=self.dry_run).execute(plan)
        stake = sum(v for v in plan.values() if v > 0)
        self._session_rounds += 1
        self._session_stake += stake
        logger.info(
            "巡房下注 No.%s 共 %d，累計 %d 局 / %d",
            self._patrol_current,
            stake,
            self._session_rounds,
            self._session_stake,
        )
        self._audit_after_bet(t_bet)
        self._ui_fail_streak = 0
        return True

    def _patrol_follow_one_window(self, cur: int | None) -> str:
        """等本桌的一個下注窗口，跟注一次（與掛房一致的「晚定稿」做法）：

          1) 看到綠燈且 T>=min_enter_t（預設 8）→ 立即「提早開統計表」。
          2) 統計表會蓋住倒數，故開表前先用 OCR 讀到的 T 設一個軟體時鐘，開表後
             改用時鐘推算 T。
          3) 撐到時鐘 T<=patrol_finalize_at_t（預設 6）才讀「最終金額」並跟注，
             盡量跟到對象最後的注（避免跟到他早早下、之後又改的注）。

        回傳：'bet' 有跟到 / 'no_bet' 對象本局沒下可跟邊注 / 'absent' 對象不在 /
              'timeout' 等一局逾時。（後三者皆 → 換桌）
        """
        assert self._win is not None
        win = self._win
        rcfg = self.cfg.room
        deadline = time.monotonic() + rcfg.result_wait_timeout_s
        anchor_t: int | None = None   # 開表前 OCR 到的 T，當軟體時鐘起點
        anchor_mono = 0.0
        opened = False
        last_log = 0.0
        self.ctx = RoundContext()
        while time.monotonic() < deadline and self._running:
            if not opened:
                t, color = self._effective_t(self._read_cd(capture_client(win)))
                if color != CountdownColor.GREEN or t is None:
                    time.sleep(0.2)  # 開牌／過場：等下一個綠燈下注窗
                    continue
                if t >= rcfg.min_enter_t:
                    anchor_t = t
                    anchor_mono = time.monotonic()
                if anchor_t is None:
                    time.sleep(0.2)  # 此窗口進來時已不足 min_enter 秒 → 等下一個窗
                    continue
                # 提早開表（讓 T<=finalize 時只需快速刷新金額，不必再等開表）
                if not self._open_stats_menu():
                    self._force_close_stats(capture_client(win), reason="巡房開統計失敗")
                    return "timeout"
                self._position_stats_for_read()
                opened = True
                # 預定位：完整讀一次解析欄位（與進桌偵測同法，比只讀表頭可靠），確認對象在
                self._build_plan(capture_client(win), refresh_only=False, include_scroll=True)
                logger.info("No.%s 預定位欄位=%s", cur, self.ctx.resolved_columns)
                if not self.ctx.resolved_columns:
                    if not self._close_stats_panel(capture_client(win)):
                        self._force_close_stats(capture_client(win), reason="巡房對象不在")
                    return "absent"
                continue

            # 已開表：改用軟體時鐘推算 T（統計表蓋住倒數，無法直接 OCR）
            t_now = max(0, anchor_t - int(time.monotonic() - anchor_mono))
            if t_now <= rcfg.patrol_finalize_at_t:
                self._position_stats_for_read()
                frame = capture_client(win)
                plan = self._build_plan(frame, refresh_only=True, include_scroll=True)
                present = bool(self.ctx.resolved_columns)
                raw_areas = set(self.ctx.last_raw_bet_areas)
                if not self._close_stats_panel(capture_client(win)):
                    self._force_close_stats(capture_client(win), reason="巡房定稿關表")
                # 關表後倒數露出來，用「真實 OCR」當下注安全閘（軟體時鐘只用來決定何時讀）
                t_bet, cd_color = self._read_cd_real()
                logger.info(
                    "No.%s 定稿讀取 present=%s 可跟=%s 原始區=%s（T=%s）",
                    cur, present, plan, raw_areas, t_bet,
                )
                if not present:
                    return "absent"
                if not plan:
                    # 對象有下注、但只下莊/閒（不是我們要跟的）→ 不黏桌，直接換。
                    if raw_areas & {"莊", "閒"}:
                        return "main_only"
                    # 完全沒讀到下注（對象本局沒下 或 OCR 漏讀）→ 交由容忍機制。
                    return "no_bet"
                logger.info("No.%s 定稿跟注 %s（關表後真實 T=%s %s）", cur, plan, t_bet, cd_color.name)
                if not self._bet_gate_after_close(t_bet, cd_color):
                    logger.info("關表後讀不到倒數（多半已封盤/過場），放棄本局")
                    return "no_bet"
                if not self._safety_allows_bet(plan):
                    return "no_bet"
                if self._patrol_place(plan, t_bet):
                    return "bet"
                return "no_bet"

            now = time.monotonic()
            if now - last_log >= 5.0:
                logger.info(
                    "No.%s 已開表，撐到剩 T<=%s 才定稿（時鐘 T≈%s）",
                    cur,
                    rcfg.patrol_finalize_at_t,
                    t_now,
                )
                last_log = now
            time.sleep(0.2)
        logger.info("等下注窗口逾時（%.0fs），放棄 No.%s", rcfg.result_wait_timeout_s, cur)
        return "timeout"

    def _patrol_visit_table(self, cur: int | None) -> str:
        """進到本桌：先確認對象在不在；在就黏桌，每個下注窗口都提早開表、撐到剩
        T<=patrol_finalize_at_t 才讀最終金額跟注，直到對象某局沒下邊注／離開／逾時才換桌。
        """
        # 進桌先快速確認對象在不在（隨時可開表，不必等下注窗，空桌才能秒換）
        present, _plan = self._patrol_scan_table()
        if not present:
            logger.info("No.%s 無追蹤對象，換下一桌", cur)
            return "switch"
        leave_after = max(1, self.cfg.room.patrol_leave_after_idle)
        logger.info(
            "No.%s 有追蹤對象，黏桌跟注（撐到剩 T<=%s 定稿；連續 %d 局沒得跟才換桌）",
            cur,
            self.cfg.room.patrol_finalize_at_t,
            leave_after,
        )
        idle = 0  # 連續「沒跟到邊注」的局數（含 OCR 暫時漏讀）
        while self._running:
            result = self._patrol_follow_one_window(cur)
            if result == "bet":
                idle = 0
                self._patrol_wait_round_end()  # 等本局開完牌再進下一局，避免棄注
                continue
            if result == "timeout":
                logger.info("No.%s 等下注窗口逾時，換下一桌", cur)
                return "switch"
            if result == "main_only":
                # 對象只下莊/閒（不是我們要跟的）→ 立刻換桌，不黏、不容忍。
                logger.info("No.%s 對象只下莊/閒，不跟，換下一桌", cur)
                return "switch"
            # absent / no_bet：對象可能只是這局沒下邊注，或 OCR 暫時漏讀。
            # 不要一次就走——連續達門檻才換桌，避免對象還在下卻被提早放掉。
            idle += 1
            logger.info(
                "No.%s 本局沒跟到（%s），連續 %d/%d 局",
                cur, result, idle, leave_after,
            )
            if idle >= leave_after:
                logger.info("No.%s 連續 %d 局沒得跟，換下一桌", cur, idle)
                return "switch"
            self._patrol_wait_round_end()  # 留桌：等本局開完牌再看下一局
        return "switch"

    def _patrol_wait_round_end(self) -> None:
        """下注後等本局開完牌：等到倒數先結束（非綠燈／開牌中），再重新出現新的
        下注窗口（綠燈且 T 夠大）為止——代表本局已結算，這時離桌才不會影響本注。
        """
        assert self._win is not None
        win = self._win
        rcfg = self.cfg.room
        deadline = time.monotonic() + rcfg.result_wait_timeout_s
        saw_non_green = False  # 是否已看到本局下注結束（進入開牌）
        last_log = 0.0
        while time.monotonic() < deadline and self._running:
            t, color = self._effective_t(self._read_cd(capture_client(win)))
            if color != CountdownColor.GREEN:
                saw_non_green = True
            elif saw_non_green and t is not None and t >= rcfg.min_enter_t:
                logger.info("本局已開完牌（新下注窗口 T=%s）", t)
                return
            now = time.monotonic()
            if now - last_log >= 10.0:
                logger.info("等開牌中…（T=%s %s）", t, color.name)
                last_log = now
            time.sleep(0.5)
        logger.warning("等開牌逾時（%.0fs），仍換下一桌", rcfg.result_wait_timeout_s)

    def run_patrol_once(self) -> bool:
        win = self._find_game()
        if not win:
            now = time.perf_counter()
            if now - getattr(self, "_last_nowin_log_mono", 0.0) >= 15.0:
                logger.warning("找不到視窗「%s」，等待視窗出現", self.cfg.window.title_substring)
                self._last_nowin_log_mono = now
            self._win = None
            return False
        self._win = win
        frame = capture_client(win)

        from star_follow.automation import lobby_nav

        cap = lambda: capture_client(win)
        ready, screen = lobby_nav.prepare_for_table_play(win, self.cfg, cap)
        if not ready:
            now = time.perf_counter()
            if now - self._last_lobby_log_mono >= 8.0:
                logger.info("尚未就緒（%s），先回桌或關提示…", screen)
                self._last_lobby_log_mono = now
            if screen != lobby_nav.PHASE_TABLE:
                lobby_nav.return_to_baccarat_table(win, self.cfg, cap)
            return True

        frame = capture_client(win)
        if not self._win_logged:
            focus_window(win.hwnd)
            logger.info("巡房模式啟動，鎖定「%s」client=%dx%d", win.title, win.client_width, win.client_height)
            self._win_logged = True

        if self._patrol_current is None:
            from star_follow.automation.room_nav import read_current_table

            cur = read_current_table(frame, self.cfg, win)
            if cur is not None:
                self._patrol_current = cur

        if self._patrol_current is None:
            logger.warning("尚未進入牌桌（無桌號），執行大廳導覽…")
            lobby_nav.return_to_baccarat_table(win, self.cfg, cap)
            return True

        # 不管現在能不能下注，先開統計表看本桌有沒有追蹤對象：
        #   有 → 黏桌持續跟注，到對象某局沒下邊注才換桌（在 _patrol_visit_table 內處理）
        #   沒有 → 直接換下一桌
        logger.info("進入 No.%s，開統計表檢查追蹤對象", self._patrol_current)
        self._patrol_visit_table(self._patrol_current)

        if not self._patrol_advance(self._patrol_current):
            logger.warning("找不到可換的桌，稍候再試")
            time.sleep(1.0)
        return True

    def run_loop(self) -> None:
        self._running = True
        patrol = self.cfg.room.mode == "patrol"
        logger.info(
            "引擎啟動 (%s, %s)",
            "dry-run" if self.dry_run else "LIVE",
            "換房巡房" if patrol else "掛房",
        )
        while self._running:
            try:
                _loop_t0 = time.perf_counter()
                ok = self.run_patrol_once() if patrol else self.run_once()
                _loop_ms = (time.perf_counter() - _loop_t0) * 1000.0
                # 卡頓診斷：單圈過久（多半是某步阻塞或與背景上傳搶資源）→ 記一筆，方便定位。
                if _loop_ms >= 2500.0:
                    logger.warning("主迴圈單圈耗時 %.0f ms（phase=%s）— 偏慢，留意是否有阻塞", _loop_ms, self.phase.name)
                if ok is False and self._win is None:
                    # 視窗不在，降速輪詢避免空轉
                    time.sleep(1.0)
                    continue
                if self._capture_unavail_since is not None:
                    # 截圖恢復正常：記一筆並重置回合狀態，重新同步畫面
                    down = time.monotonic() - self._capture_unavail_since
                    logger.info("畫面已恢復擷取（暫停約 %.0f 秒）", down)
                    self._capture_unavail_since = None
                    self.phase = Phase.IDLE
                    self.ctx = RoundContext()
                    self._cd_tracker.reset()
            except CaptureUnavailable as exc:
                # 螢幕暫時抓不到（視窗最小化/螢幕休眠或鎖定/遠端斷線）：放慢重試、
                # 只在「開始」時記一筆，不狂噴 traceback，也不嘗試回桌（回桌也要截圖）。
                if self._capture_unavail_since is None:
                    self._capture_unavail_since = time.monotonic()
                    logger.warning(
                        "畫面暫時無法擷取（%s）：多半是遊戲視窗被最小化／螢幕休眠或鎖定。"
                        "已暫停動作、每秒重試，待畫面恢復自動接續。",
                        exc,
                    )
                self.phase = Phase.IDLE
                time.sleep(1.0)
                continue
            except Exception:
                logger.exception("迴圈錯誤")
                try:
                    if self._win:
                        self._recover_ui(
                            capture_client(self._win),
                            "例外後恢復",
                        )
                except CaptureUnavailable:
                    # 恢復時也抓不到畫面：當成暫時狀態，下一輪走上面的重試分支
                    self._capture_unavail_since = self._capture_unavail_since or time.monotonic()
                except Exception:
                    logger.exception("例外恢復失敗")
                self.phase = Phase.IDLE
                self.ctx = RoundContext()
                self._cd_tracker.reset()
            time.sleep(self.cfg.timing.poll_ms / 1000.0)

        if self._stopped_reason:
            logger.warning(
                "引擎已停止（%s）。本次共下注 %d 局 / 累計 %d",
                self._stopped_reason,
                self._session_rounds,
                self._session_stake,
            )

    def stop(self) -> None:
        self._running = False
