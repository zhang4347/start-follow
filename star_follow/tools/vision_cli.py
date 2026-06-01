"""M2：即時印出倒數與統計表 JSON。用法: python -m star_follow.tools.vision_cli"""

from __future__ import annotations

import argparse
import json
import sys
import time

from star_follow.capture.screen import capture_client
from star_follow.capture.window import find_game_window
from star_follow.config import load_config
from star_follow.vision.panel import stats_panel_open
from star_follow.vision.roi import scale_rect
from star_follow.vision.state import read_countdown
from star_follow.vision.stats_parser import parse_stats_table


def main() -> int:
    parser = argparse.ArgumentParser(description="視覺管線測試")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--image", type=str, help="離線測試用截圖路徑")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--paddle", action="store_true", help="表頭暱稱用 PaddleOCR")
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)

    def process_frame(frame, label: str) -> None:
        win_w = cfg.window.reference_width
        win_h = cfg.window.reference_height
        if args.image:
            h, w = frame.shape[:2]
            win_w, win_h = w, h

        cd = cfg.roi.get("countdown", [462, 28, 100, 100])
        cd_r = scale_rect(cd, cfg.window.reference_width, cfg.window.reference_height, win_w, win_h)
        cd_img = frame[cd_r[1] : cd_r[1] + cd_r[3], cd_r[0] : cd_r[0] + cd_r[2]]
        state = read_countdown(cd_img)

        panel = cfg.roi.get("stats_panel", [95, 72, 835, 455])
        panel_r = scale_rect(
            panel, cfg.window.reference_width, cfg.window.reference_height, win_w, win_h
        )
        table = cfg.roi.get("stats_table")
        table_r = (
            scale_rect(
                table, cfg.window.reference_width, cfg.window.reference_height, win_w, win_h
            )
            if table
            else None
        )
        close_r = (
            scale_rect(
                cfg.roi["stats_close"],
                cfg.window.reference_width,
                cfg.window.reference_height,
                win_w,
                win_h,
            )
            if cfg.roi.get("stats_close")
            else None
        )
        open_ = stats_panel_open(
            frame,
            list(panel_r),
            table_rect=list(table_r) if table_r else None,
            close_rect=list(close_r) if close_r else None,
        )

        out: dict = {
            "source": label,
            "countdown": {
                "color": state.color.value,
                "seconds": state.seconds,
                "confidence": round(state.confidence, 3),
            },
            "stats_panel_open": open_,
        }
        if open_:
            import time as _time
            from star_follow.core.follow_list import FollowList

            fl = FollowList.load()
            targets = [(e.name, e.column_index) for e in fl.active_entries()]
            t0 = _time.perf_counter()
            stats = parse_stats_table(
                frame, cfg, use_paddle=args.paddle, follow_targets=targets
            )
            out["ocr_ms"] = round(stats.elapsed_ms or (_time.perf_counter() - t0) * 1000, 1)
            out["header_columns"] = [
                {"col": i, "name": n} for i, n in stats.header_columns if n
            ]
            out["resolved_columns"] = stats.resolved_columns
            out["stats"] = {
                "players": stats.players,
                "bets_by_column": stats.bets_by_column,
            }
            out["follow_plans"] = {}
            out["not_in_room"] = []
            from star_follow.vision.stats_parser import extract_player_bets

            for entry in fl.active_entries():
                col = stats.resolved_columns.get(entry.name)
                if col is None:
                    out["not_in_room"].append(entry.name)
                    out["follow_plans"][entry.name] = {}
                    continue
                plan = extract_player_bets(
                    stats, entry.name, cfg.stats_to_bet, column_index=col
                )
                out["follow_plans"][entry.name] = plan
        print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.image:
        import numpy as np
        from PIL import Image

        img = np.array(Image.open(args.image).convert("RGB"))
        process_frame(img, args.image)
        return 0

    while True:
        win = find_game_window(cfg.window.title_substring)
        if not win:
            print("找不到星城視窗", file=sys.stderr)
            if args.once:
                return 1
            time.sleep(1)
            continue
        frame = capture_client(win)
        # 離線 config 用參考解析度時，依實際視窗縮放
        h, w = frame.shape[:2]
        old_ref = (cfg.window.reference_width, cfg.window.reference_height)
        cfg.window.reference_width = w
        cfg.window.reference_height = h
        process_frame(frame, win.title)
        cfg.window.reference_width, cfg.window.reference_height = old_ref
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
