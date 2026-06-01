from __future__ import annotations

import difflib
import re
import time
from dataclasses import dataclass, field

import numpy as np

from star_follow.config import AppConfig

from . import ocr as ocr_mod
from .ocr import ocr_amount, ocr_chinese_line, ocr_chinese_paddle, ocr_name_cell
from .stats_table_roi import find_stats_table_in_panel


@dataclass
class StatsParseResult:
    players: list[str]
    bets_by_player: dict[str, dict[str, int]]
    bets_by_column: dict[int, dict[str, int]]
    header_columns: list[tuple[int, str]]
    confidences: dict[str, float]
    raw_header: list[tuple[str, float]]
    elapsed_ms: float = 0.0
    resolved_columns: dict[str, int] = field(default_factory=dict)


_SKIP_NAMES = {"注區", "玩家", "押注統計", "總計", ""}
_MAX_PLAYER_COLS = 8


def _crop_rel(img: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    h, w = img.shape[:2]
    return img[int(h * y0) : int(h * y1), int(w * x0) : int(w * x1)]


def _layout_metrics(layout: dict) -> tuple[float, float, int]:
    """依可見列數計算列高（固定比例，不 OCR 列名）。"""
    data_top = float(layout.get("data_top", 0.155))
    data_bottom = float(layout.get("data_bottom", 0.88))
    visible_rows = int(layout.get("visible_rows", 7))
    if layout.get("row_height") is not None:
        row_h = float(layout["row_height"])
    else:
        row_h = (data_bottom - data_top) / max(1, visible_rows)
    return data_top, row_h, visible_rows


def _row_band(layout: dict, row_index: int) -> tuple[float, float]:
    data_top, row_h, visible_rows = _layout_metrics(layout)
    if row_index < 0 or row_index >= visible_rows:
        return 0.0, 0.0
    y0 = data_top + row_index * row_h
    return y0, min(float(layout.get("data_bottom", 0.88)), y0 + row_h)


def extract_stats_table(frame: np.ndarray, cfg: AppConfig) -> tuple[np.ndarray, list[int]]:
    panel_rect = cfg.roi.get("stats_panel", [118, 86, 1043, 548])
    detected = find_stats_table_in_panel(frame, list(panel_rect))
    rect = detected or cfg.roi.get("stats_table", [110, 118, 805, 400])
    x, y, w, h = rect
    return frame[y : y + h, x : x + w].copy(), list(rect)


def _column_layout(table_w: int, layout: dict) -> tuple[int, list[tuple[int, int]]]:
    label_frac = float(layout.get("label_col_frac", 0.11))
    label_w = max(40, int(table_w * label_frac))
    remain = table_w - label_w
    col_w = max(50, remain // _MAX_PLAYER_COLS)
    cols = []
    for i in range(_MAX_PLAYER_COLS):
        x0 = label_w + i * col_w
        x1 = min(table_w, x0 + col_w)
        if x1 - x0 < 30:
            break
        cols.append((x0, x1))
    return label_w, cols


def detect_column_bounds(
    table: "np.ndarray",
    layout: dict,
) -> tuple[int, list[tuple[int, int]]] | None:
    """偵測表格垂直格線取得真實欄界；失敗回 None 由等分法接手。"""
    import cv2
    import numpy as np

    h, w = table.shape[:2]
    if w < 200 or h < 80:
        return None
    # 取資料列區域（避開上方標題/下方總計），對格線最敏感
    y0 = int(h * 0.12)
    y1 = int(h * 0.90)
    body = table[y0:y1, :]
    gray = cv2.cvtColor(body, cv2.COLOR_RGB2GRAY).astype("float32")
    col_mean = gray.mean(axis=0)  # 每個 x 的平均亮度；格線較暗 → 局部極小
    # 與局部背景（中位數平滑）比較，找明顯偏暗的縱向溝槽
    k = max(9, w // 60) | 1
    bg = cv2.medianBlur(col_mean.reshape(1, -1).astype("uint8"), 1).flatten().astype("float32")
    bg = np.convolve(col_mean, np.ones(k) / k, mode="same")
    dip = bg - col_mean  # 越大代表越暗（線）
    thr = max(8.0, float(dip.max()) * 0.4)
    xs = [x for x in range(2, w - 2) if dip[x] >= thr]
    if len(xs) < 3:
        return None
    # 把連續的 x 併成一條線，取中心
    lines: list[int] = []
    group = [xs[0]]
    for x in xs[1:]:
        if x - group[-1] <= 8:
            group.append(x)
        else:
            lines.append(sum(group) // len(group))
            group = [x]
    lines.append(sum(group) // len(group))
    if len(lines) < 3:
        return None
    # lines[0] 很小 = 表格左外框，其後第一段才是 label 欄需丟棄；
    # 否則 label 欄在 (0, lines[0])，各線之間皆為玩家欄。
    drop_first = lines[0] < 25
    label_w = lines[1] if drop_first and len(lines) > 1 else lines[0]
    cols: list[tuple[int, int]] = []
    for a, b in zip(lines, lines[1:]):
        if b - a < 40:
            continue
        cols.append((a + 2, b - 2))
    if drop_first and cols:
        cols = cols[1:]
    if not cols:
        return None
    return label_w, cols[:_MAX_PLAYER_COLS]


def _columns_for(table: "np.ndarray", layout: dict) -> tuple[int, list[tuple[int, int]]]:
    if layout.get("detect_grid", True):
        try:
            det = detect_column_bounds(table, layout)
        except Exception:
            det = None
        if det and len(det[1]) >= 1:
            return det
    return _column_layout(table.shape[1], layout)


def _clean_name(name: str) -> str:
    name = re.sub(r"\s+", "", name)
    for ch in "注區玩家|;[]()（）":
        name = name.replace(ch, "")
    return name.strip()


def _read_header_columns(
    header: np.ndarray,
    cols: list[tuple[int, int]],
    *,
    use_paddle: bool = False,
) -> list[tuple[str, float, int, int]]:
    results: list[tuple[str, float, int, int]] = []
    for x0, x1 in cols:
        cell = header[:, x0:x1]
        if use_paddle:
            name, conf = ocr_chinese_paddle(cell)
        else:
            name, conf = ocr_chinese_line(cell, stats=True)
        name = _clean_name(name)
        if not name or name in _SKIP_NAMES or len(name) < 2 or name.isdigit():
            continue
        results.append((name, conf, x0, x1))
    return results


_USE_PADDLE_HEADER = True


def set_header_ocr(use_paddle: bool) -> None:
    global _USE_PADDLE_HEADER
    _USE_PADDLE_HEADER = use_paddle


def _ocr_header_name(cell: np.ndarray) -> str:
    """玩家暱稱（中文）：可選 PaddleOCR；否則用放大+Otsu 的 Tesseract。"""
    if _USE_PADDLE_HEADER:
        try:
            name, conf = ocr_chinese_paddle(cell)
            if name:
                return _clean_name(name)
        except Exception:
            pass
    name, _ = ocr_name_cell(cell)
    return _clean_name(name)


def read_column_headers(table: np.ndarray, layout: dict) -> list[tuple[int, str]]:
    """讀取各玩家欄表頭暱稱，回傳 (欄位索引, 名稱)。"""
    band = layout.get("header_band", [0.0, 0.0, 1.0, 0.12])
    if isinstance(band, list) and len(band) == 4:
        header = _crop_rel(table, band[0], band[1], band[2], band[3])
    else:
        header = table[: max(20, int(table.shape[0] * 0.12)), :]
    _, cols = _columns_for(table, layout)
    out: list[tuple[int, str]] = []
    for idx, (x0, x1) in enumerate(cols):
        cell = header[:, x0:x1]
        out.append((idx, _ocr_header_name(cell)))
    return out


# 跟注是真金白銀，比對要「精準優先」：寧可錯過也不要跟錯路人。
_NAME_MATCH_CUTOFF = 0.8   # 模糊比對最低相似度（容忍約 1 個字的 OCR 誤差）
_NAME_MATCH_MARGIN = 0.08  # 最佳與第二名差距需大於此值，否則視為無法分辨→不跟


def _norm_name(s: str) -> str:
    """正規化暱稱：全形轉半形、去空白、去除非中英數雜訊符號，利於精準比對。"""
    if not s:
        return ""
    out = []
    for ch in s:
        o = ord(ch)
        if o == 0x3000:
            out.append(" ")
        elif 0xFF01 <= o <= 0xFF5E:
            out.append(chr(o - 0xFEE0))
        else:
            out.append(ch)
    t = "".join(out)
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", t)


def find_column_for_player(
    headers: list[tuple[int, str]],
    player_name: str,
    hint: int | None = None,
) -> int | None:
    """在所有表頭中找出「最符合」這個暱稱的欄；精準優先，找不到夠像的就回 None。"""
    target = _norm_name(player_name)
    if not target:
        return None
    named = [(idx, _norm_name(name)) for idx, name in headers if name]
    named = [(idx, n) for idx, n in named if n]
    if not named:
        return None

    # 1) 正規化後完全相同（最可靠）
    exact = [idx for idx, n in named if n == target]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        # 罕見：多欄同名 → 用 hint 才能確定，否則放棄（不亂猜）
        return hint if (hint is not None and hint in exact) else None

    # 2) 高相似度模糊比對：取最佳，且需明顯勝過第二名，避免跟到名字相近的路人
    scored = sorted(
        (
            (difflib.SequenceMatcher(None, target, n).ratio(), idx)
            for idx, n in named
        ),
        key=lambda x: x[0],
        reverse=True,
    )
    best_r, best_idx = scored[0]
    if best_r < _NAME_MATCH_CUTOFF:
        return None
    if len(scored) > 1 and scored[1][0] >= best_r - _NAME_MATCH_MARGIN:
        return None
    return best_idx


def _parse_single_column(
    table: np.ndarray,
    col_index: int,
    rows: list[str],
    layout: dict,
) -> dict[str, int]:
    """只 OCR 一欄金額（固定座標；表必須已捲到頂，首列=莊家）。"""
    _, cols = _columns_for(table, layout)
    if col_index < 0 or col_index >= len(cols):
        return {}
    x0, x1 = cols[col_index]
    _, _, visible_rows = _layout_metrics(layout)
    out: dict[str, int] = {}
    for i, row_name in enumerate(rows[:visible_rows]):
        y0, y1 = _row_band(layout, i)
        if y1 <= y0:
            continue
        row_img = _crop_rel(table, 0.0, y0, 1.0, y1)
        cell = row_img[:, x0:x1]
        amount, _ = ocr_amount(cell)
        if amount > 0:
            out[row_name] = amount
    return out


def parse_bottom_row_amount(
    frame: np.ndarray,
    cfg: AppConfig,
    col_index: int,
) -> int:
    """捲到底後讀最後一格金額（有數字 = 閒龍寶）。"""
    table, _ = extract_stats_table(frame, cfg)
    layout = cfg.raw.get("stats_layout", {})
    _, cols = _columns_for(table, layout)
    if col_index < 0 or col_index >= len(cols):
        return 0
    _, _, visible_rows = _layout_metrics(layout)
    y0, y1 = _row_band(layout, visible_rows - 1)
    if y1 <= y0:
        return 0
    x0, x1 = cols[col_index]
    cell = _crop_rel(table, 0.0, y0, 1.0, y1)[:, x0:x1]
    amount, _ = ocr_amount(cell)
    return amount if amount > 0 else 0


def match_player_name(players: list[str], target: str) -> str | None:
    """精準比對暱稱：正規化後完全相同優先，否則高相似度且明顯勝過第二名才算。
    比對不到夠像的就回 None（寧可錯過也不要跟錯路人）。"""
    target_n = _norm_name(target)
    if not target_n:
        return None
    named = [(p, _norm_name(p)) for p in players]
    named = [(p, n) for p, n in named if n]
    if not named:
        return None
    for p, n in named:
        if n == target_n:
            return p
    scored = sorted(
        ((difflib.SequenceMatcher(None, target_n, n).ratio(), p) for p, n in named),
        key=lambda x: x[0],
        reverse=True,
    )
    best_r, best_p = scored[0]
    if best_r < _NAME_MATCH_CUTOFF:
        return None
    if len(scored) > 1 and scored[1][0] >= best_r - _NAME_MATCH_MARGIN:
        return None
    return best_p


def resolve_follow_columns(
    frame: np.ndarray,
    cfg: AppConfig,
    follow_targets: list[tuple[str, int | None]],
) -> StatsParseResult:
    """只解析跟注對象欄位（不讀金額），供 T=15 預定位。"""
    t0 = time.perf_counter()
    ocr_mod.set_ocr_options(fast=cfg.vision.fast_ocr, scale=cfg.vision.ocr_scale)
    set_header_ocr(cfg.vision.header_use_paddle)
    table, _ = extract_stats_table(frame, cfg)
    layout = cfg.raw.get("stats_layout", {})

    # 一律讀表頭重新確認欄位；hint 只當提示，不直接採用（換桌後舊欄位會指到路人）。
    resolved: dict[str, int] = {}
    headers: list[tuple[int, str]] = read_column_headers(table, layout)
    for name, hint in follow_targets:
        col = find_column_for_player(headers, name, hint=hint)
        if col is not None:
            resolved[name] = col

    elapsed = (time.perf_counter() - t0) * 1000
    return StatsParseResult(
        players=[n for _, n in headers if n],
        bets_by_player={},
        bets_by_column={},
        header_columns=headers,
        confidences={},
        raw_header=[(n, 0.0) for _, n in headers if n],
        elapsed_ms=elapsed,
        resolved_columns=resolved,
    )


def parse_stats_table(
    frame: np.ndarray,
    cfg: AppConfig,
    *,
    use_paddle: bool = False,
    only_columns: list[int] | None = None,
    follow_targets: list[tuple[str, int | None]] | None = None,
    known_columns: dict[str, int] | None = None,
) -> StatsParseResult:
    t0 = time.perf_counter()
    ocr_mod.set_ocr_options(fast=cfg.vision.fast_ocr, scale=cfg.vision.ocr_scale)
    set_header_ocr(cfg.vision.header_use_paddle)

    table, _ = extract_stats_table(frame, cfg)

    layout = cfg.raw.get("stats_layout", {})
    rows = cfg.stats_rows or []

    if follow_targets is not None and cfg.vision.fast_ocr:
        if known_columns:
            headers = []
            players = []
            bets_by_column: dict[int, dict[str, int]] = {}
            resolved = dict(known_columns)
            for name, col in known_columns.items():
                bets_by_column[col] = _parse_single_column(table, col, rows, layout)
        else:
            headers = read_column_headers(table, layout)
            players = [n for _, n in headers if n]
            bets_by_column = {}
            resolved = {}
            for name, hint in follow_targets:
                col = find_column_for_player(headers, name, hint=hint)
                if col is None:
                    continue
                resolved[name] = col
                bets_by_column[col] = _parse_single_column(table, col, rows, layout)
        elapsed = (time.perf_counter() - t0) * 1000
        return StatsParseResult(
            players=players,
            bets_by_player={},
            bets_by_column=bets_by_column,
            header_columns=headers,
            confidences={},
            raw_header=[(n, 0.0) for _, n in headers if n],
            elapsed_ms=elapsed,
            resolved_columns=resolved,
        )

    if only_columns is not None and cfg.vision.fast_ocr:
        headers = read_column_headers(table, layout)
        bets_by_column: dict[int, dict[str, int]] = {}
        for col in only_columns:
            bets_by_column[col] = _parse_single_column(table, col, rows, layout)
        elapsed = (time.perf_counter() - t0) * 1000
        return StatsParseResult(
            players=[n for _, n in headers if n],
            bets_by_player={},
            bets_by_column=bets_by_column,
            header_columns=headers,
            confidences={},
            raw_header=[],
            elapsed_ms=elapsed,
        )

    band = layout.get("header_band", [0.0, 0.0, 1.0, 0.12])
    if isinstance(band, list) and len(band) == 4:
        header = _crop_rel(table, band[0], band[1], band[2], band[3])
    else:
        header = table[: max(20, int(table.shape[0] * 0.12)), :]

    _, _, visible_rows = _layout_metrics(layout)

    _, cols = _columns_for(table, layout)
    header_info = _read_header_columns(header, cols, use_paddle=use_paddle)
    players = [h[0] for h in header_info]

    bets: dict[str, dict[str, int]] = {p: {} for p in players}
    bets_by_column: dict[int, dict[str, int]] = {}
    confs: dict[str, float] = {}

    for col_idx, (name, _, x0, x1) in enumerate(header_info):
        bets_by_column[col_idx] = {}

    for i, row_name in enumerate(rows[:visible_rows]):
        y0, y1 = _row_band(layout, i)
        row_img = _crop_rel(table, 0.0, y0, 1.0, y1)
        for col_idx, (name, _, x0, x1) in enumerate(header_info):
            cell = row_img[:, x0:x1]
            amount, conf = ocr_amount(cell)
            if amount > 0:
                bets[name][row_name] = amount
                bets_by_column[col_idx][row_name] = amount
            confs[f"{name}:{row_name}"] = conf

    elapsed = (time.perf_counter() - t0) * 1000
    return StatsParseResult(
        players=players,
        bets_by_player=bets,
        bets_by_column=bets_by_column,
        header_columns=[(col_idx, name) for col_idx, (name, _, _, _) in enumerate(header_info)],
        confidences=confs,
        raw_header=[(n, c) for n, c, _, _ in header_info],
        elapsed_ms=elapsed,
    )


def extract_player_bets(
    result: StatsParseResult,
    player_name: str,
    stats_to_bet: dict[str, str],
    *,
    column_index: int | None = None,
) -> dict[str, int]:
    raw: dict[str, int] = {}
    if column_index is not None and column_index in result.bets_by_column:
        raw = result.bets_by_column[column_index]
    if not raw:
        matched = match_player_name(result.players, player_name)
        if matched is not None:
            raw = result.bets_by_player.get(matched, {})
    out: dict[str, int] = {}
    for stats_name, amount in raw.items():
        bet_key = stats_to_bet.get(stats_name, stats_name)
        out[bet_key] = out.get(bet_key, 0) + amount
    return out
