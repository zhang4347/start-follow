from __future__ import annotations

import logging

from star_follow.automation.chip_planner import count_bet_clicks, format_chip_plan, plan_chips
from star_follow.automation.click import click_at
from star_follow.capture.window import GameWindow, focus_window
from star_follow.config import AppConfig
from star_follow.vision.roi import scale_point

logger = logging.getLogger(__name__)


class BetExecutor:
    def __init__(self, cfg: AppConfig, win: GameWindow, *, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.win = win
        self.dry_run = dry_run
        ref_w = cfg.window.reference_width
        ref_h = cfg.window.reference_height
        aw, ah = win.client_width, win.client_height

        self._chips = {
            int(k): scale_point(v, ref_w, ref_h, aw, ah)
            for k, v in cfg.chips.items()
        }
        self._areas = {
            k: scale_point(v, ref_w, ref_h, aw, ah)
            for k, v in cfg.bet_areas.items()
        }
        t = cfg.timing
        self._bet_delay = (t.bet_click_delay_ms_min, t.bet_click_delay_ms_max)

    def execute(self, plan: dict[str, int]) -> None:
        if not plan:
            return

        if not self.dry_run:
            focus_window(self.win.hwnd)

        for area, amount in plan.items():
            if amount <= 0:
                continue
            if area not in self._areas:
                logger.warning("未知注區: %s", area)
                continue
            ax, ay = self._areas[area]
            try:
                chip_plan = plan_chips(amount, self.cfg.chip_values)
            except ValueError as exc:
                logger.error("金額 %s 無法湊齊: %s", amount, exc)
                continue

            logger.info("下注 %s %s：%s", area, amount, format_chip_plan(amount, self.cfg.chip_values))

            for denom, times in chip_plan:
                cx, cy = self._chips.get(denom, (0, 0))
                if cx == 0 and cy == 0:
                    logger.warning("未設定籌碼 %s 座標，請執行 mark_chips", denom)
                    continue
                if self.dry_run:
                    logger.info("[dry-run] 選籌碼 %s @(%d,%d)", denom, cx, cy)
                else:
                    click_at(
                        self.win,
                        cx,
                        cy,
                        delay_ms=self._bet_delay,
                        backend="win32",
                        refocus=False,
                    )
                for i in range(times):
                    if self.dry_run:
                        logger.info(
                            "[dry-run] 點注區 %s @(%d,%d)（%s 第 %d/%d 次）",
                            area,
                            ax,
                            ay,
                            denom,
                            i + 1,
                            times,
                        )
                    else:
                        click_at(
                            self.win,
                            ax,
                            ay,
                            jitter=(8, 12),
                            delay_ms=self._bet_delay,
                            backend="win32",
                            refocus=False,
                        )

        total_clicks = 0
        for amt in plan.values():
            if amt <= 0:
                continue
            try:
                total_clicks += count_bet_clicks(plan_chips(amt, self.cfg.chip_values))
            except ValueError:
                pass
        if total_clicks:
            logger.info("本輪下注共約 %d 次點擊", total_clicks)
