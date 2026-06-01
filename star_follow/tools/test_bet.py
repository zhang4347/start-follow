"""測試籌碼組合與下注點擊。

用法:
  python -m star_follow.tools.test_bet --plan          # 顯示各金額最少點擊組合
  python -m star_follow.tools.test_bet --amount 10000 --area 閒   # dry-run 單注
  python -m star_follow.tools.test_bet --amount 10000 --area 閒 --live  # 真點擊
"""

from __future__ import annotations

import argparse
import logging

from star_follow.automation.chip_planner import format_chip_plan
from star_follow.automation.executor import BetExecutor
from star_follow.capture.window import find_game_window, focus_window
from star_follow.config import load_config

SAMPLE_AMOUNTS = [1000, 5000, 10000, 15000, 28000, 50000, 100000]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true", help="列出常用金額的籌碼組合")
    parser.add_argument("--amount", type=int, default=0)
    parser.add_argument("--area", type=str, default="閒")
    parser.add_argument("--live", action="store_true", help="真的點擊（預設 dry-run）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()

    if args.plan or args.amount <= 0:
        print("籌碼面額:", cfg.chip_values)
        print()
        for amt in SAMPLE_AMOUNTS:
            print(format_chip_plan(amt, cfg.chip_values))
        if args.amount <= 0:
            return 0

    win = find_game_window(cfg.window.title_substring)
    if not win:
        print("找不到星城視窗")
        return 1

    focus_window(win.hwnd)
    mode = "LIVE" if args.live else "dry-run"
    print(f"\n{mode}：{args.area} {args.amount}")
    print(format_chip_plan(args.amount, cfg.chip_values))

    executor = BetExecutor(cfg, win, dry_run=not args.live)
    executor.execute({args.area: args.amount})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
