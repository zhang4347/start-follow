r"""手動標記統計表：兩步拉框，解決「自動偵測把最右欄(第7欄)收掉」與「欄位切歪」。

步驟一：在整張畫面框出『整個押注統計表格』（務必把最右邊第 7 欄含進來；上到表頭名字列、
        下到總計列）。→ 寫成固定的 roi.stats_table，並關掉會收窄的自動偵測。
步驟二：在裁好的表格上框出『整排玩家名字』（左貼第一欄、右貼第七欄、上下只框名字列）。
        → 等分成 7 欄，寫成 stats_layout.player_band / header_band。

用法（先進 venv 或直接用 venv 的 python，並先在遊戲打開統計表）：
    python -m star_follow.tools.mark_columns
    # 或對存好的全畫面截圖：
    python -m star_follow.tools.mark_columns --image logs\grab\frame_xxx.png
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
from PIL import Image

from star_follow.config import load_config, save_config
from star_follow.paths import logs_dir
from star_follow.vision.stats_parser import read_column_headers


def _load_frame(cfg, image: str | None) -> np.ndarray | None:
    if image:
        bgr = cv2.imread(image)
        if bgr is None:
            print(f"讀不到圖檔：{image}")
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    from star_follow.capture.screen import capture_client
    from star_follow.capture.window import find_game_window, refresh_game_window

    win = find_game_window(
        cfg.window.title_substring, title_aliases=cfg.window.title_aliases or None
    )
    if win is None:
        print("找不到遊戲視窗（請先開好星城並打開統計表，或改用 --image）。")
        return None
    win = refresh_game_window(win.hwnd, win.title)
    return np.asarray(capture_client(win))


def _select(title: str, bgr: np.ndarray) -> tuple[int, int, int, int]:
    r = cv2.selectROI(title, bgr, showCrosshair=True)
    cv2.destroyAllWindows()
    return tuple(int(v) for v in r)  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="改用存好的全畫面截圖")
    parser.add_argument("--cols", type=int, default=7, help="玩家欄數（預設 7）")
    args = parser.parse_args()

    cfg = load_config()
    frame = _load_frame(cfg, args.image)
    if frame is None:
        return 1
    H, W = frame.shape[:2]
    ref_w = cfg.window.reference_width or 1280
    ref_h = cfg.window.reference_height or 720
    out = logs_dir() / "grab"
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # 步驟一：整個表格
    print("步驟一：框住『整個押注統計表格』（務必含最右第 7 欄；上到名字列、下到總計列）。")
    full_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    tx, ty, tw, th = _select("STEP1: whole table (include 7th col)", full_bgr)
    if tw <= 0 or th <= 0:
        print("沒有框選表格（已取消）。")
        return 1
    table_ref = [
        int(round(tx * ref_w / W)),
        int(round(ty * ref_h / H)),
        int(round(tw * ref_w / W)),
        int(round(th * ref_h / H)),
    ]
    print(f"roi.stats_table（參考 {ref_w}x{ref_h}）= {table_ref}")

    table = frame[ty : ty + th, tx : tx + tw]
    Image.fromarray(table).save(str(out / f"mark_table_{ts}.png"))

    # 步驟二：名字列
    print("\n步驟二：在表格上框住『整排玩家名字』（左貼第一欄、右貼第七欄、上下只框名字列）。")
    table_bgr = cv2.cvtColor(table, cv2.COLOR_RGB2BGR)
    nx, ny, nw, nh = _select("STEP2: player names row", table_bgr)
    if nw <= 0 or nh <= 0:
        print("沒有框選名字列（已取消，未寫入）。")
        return 1
    x0f = round(nx / tw, 4)
    x1f = round((nx + nw) / tw, 4)
    y0f = round(ny / th, 4)
    hf = round(nh / th, 4)
    player_band = [x0f, x1f]
    header_band = [0.0, y0f, 1.0, hf]
    n = max(1, args.cols)
    print(f"player_band = {player_band}  header_band = {header_band}  cols = {n}")

    # 步驟三（可選）：框資料列範圍，重新對齊金額的「列」。程式實際是「往下滑到底」再讀，
    # 所以畫面第一列是『閒家』、最後一列是『閒龍寶』（莊家已滑出畫面）。
    # 框：第一列(閒家)上緣 → 最後一列(閒龍寶)下緣，不要框到「總計」列。
    # 直接按 c/Esc 跳過則沿用現有 data_top/data_bottom。
    rows = [str(x) for x in (cfg.raw.get("stats_rows_bottom") or cfg.stats_rows or [])]
    visible_rows = int(cfg.raw.get("stats_layout", {}).get("visible_rows", 7) or 7)
    print(
        "\n步驟三（可選）：框『資料列範圍』= 第一列(閒家)上緣 → 最後一列(閒龍寶)下緣，"
        "不含總計列。（程式會往下滑，莊家不在畫面內）。要跳過就直接按 c。"
    )
    rx, ry2, rw, rh = _select("STEP3 (optional): data rows region", table_bgr)
    data_top = float(cfg.raw.get("stats_layout", {}).get("data_top", 0.155))
    data_bottom = float(cfg.raw.get("stats_layout", {}).get("data_bottom", 0.88))
    if rh > 0 and rw > 0:
        data_top = round(ry2 / th, 4)
        data_bottom = round((ry2 + rh) / th, 4)
        print(f"data_top = {data_top}  data_bottom = {data_bottom}（共 {visible_rows} 列）")
    else:
        print(f"略過列校正，沿用 data_top={data_top} data_bottom={data_bottom}")

    # 組出最終 layout
    layout = dict(cfg.raw.get("stats_layout", {}))
    layout.update(
        {
            "player_band": player_band,
            "header_band": header_band,
            "player_cols": n,
            "detect_grid": False,
            "data_top": data_top,
            "data_bottom": data_bottom,
            "visible_rows": visible_rows,
        }
    )

    # 試讀名字
    try:
        from star_follow.vision.ocr import set_ocr_options

        set_ocr_options(fast=cfg.vision.fast_ocr, scale=cfg.vision.ocr_scale)
        headers = read_column_headers(table, layout)
        print("\n=== 試讀名字 ===")
        for idx, name in headers:
            print(f"  欄 {idx}: {name!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"試讀名字失敗：{exc}")

    # 驗證疊圖：畫出 n 欄 × 各列格線（金額會從這些格子讀），列號標在左邊
    col_w = nw / n
    row_h = (data_bottom - data_top) / max(1, visible_rows)
    vis = table_bgr.copy()
    # 欄分隔（沿用名字列上下，畫滿整張表高方便對齊金額）
    for i in range(n + 1):
        cx = int(round(nx + i * col_w))
        cv2.line(vis, (cx, 0), (cx, th), (0, 0, 255), 1)
    # 列分隔
    for r in range(visible_rows + 1):
        cy = int(round((data_top + r * row_h) * th))
        cv2.line(vis, (nx, cy), (int(round(nx + nw)), cy), (0, 200, 0), 1)
        if r < visible_rows:
            label = rows[r] if r < len(rows) else str(r)
            cv2.putText(
                vis, str(r), (max(2, nx - 16), cy + int(row_h * th * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
            )
    cv2.rectangle(vis, (nx, ny), (nx + nw, ny + nh), (0, 215, 255), 1)
    grid_path = out / f"mark_grid_{ts}.png"
    Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)).save(str(grid_path))
    print(f"已存驗證疊圖（紅=欄、綠=列、列號 0~{visible_rows - 1}）：{grid_path}")
    print(f"列對應：{list(enumerate(rows[:visible_rows]))}")

    try:
        ans = input("\n要把表格範圍＋欄位＋列設定寫回 config.yaml 嗎？(y/n) ").strip().lower()
    except Exception:  # noqa: BLE001
        ans = "n"
    if ans == "y":
        cfg.raw.setdefault("roi", {})["stats_table"] = table_ref
        cfg.roi["stats_table"] = table_ref
        sl = cfg.raw.setdefault("stats_layout", {})
        sl["fixed_table"] = True
        sl["player_band"] = player_band
        sl["header_band"] = header_band
        sl["player_cols"] = n
        sl["detect_grid"] = False
        sl["data_top"] = data_top
        sl["data_bottom"] = data_bottom
        sl["visible_rows"] = visible_rows
        p = save_config(cfg)
        print(f"已寫入 {p}：roi.stats_table + stats_layout（fixed_table=true，含列校正）")
    else:
        print("未寫入。可把驗證疊圖與上面數值傳回，我幫你設定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
