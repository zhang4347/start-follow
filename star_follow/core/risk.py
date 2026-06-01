from __future__ import annotations

import logging

from star_follow.automation.chip_planner import plan_chips

logger = logging.getLogger(__name__)


def cap_plan(plan: dict[str, int], chip_values: list[int] | None = None) -> dict[str, int]:
    """過濾 OCR 雜訊（如 2、7）與無法用籌碼湊齊的金額。"""
    if not plan:
        return {}
    if not chip_values:
        return {k: v for k, v in plan.items() if v > 0}

    min_chip = min(chip_values)
    out: dict[str, int] = {}
    for area, amount in plan.items():
        if amount <= 0:
            continue
        if amount < min_chip:
            logger.info(
                "忽略 %s %d（低於最小籌碼 %d，疑似 OCR 雜訊）",
                area,
                amount,
                min_chip,
            )
            continue
        try:
            plan_chips(amount, chip_values)
        except ValueError:
            logger.warning("忽略 %s %d（無法用籌碼面額湊齊）", area, amount)
            continue
        out[area] = amount
    return out
