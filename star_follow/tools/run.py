"""主引擎入口。用法: python -m star_follow.tools.run [--live]"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

from star_follow.core.engine import FollowEngine
from star_follow.core.follow_list import FollowList
from star_follow.paths import logs_dir as _logs_dir
from star_follow.util.admin import is_admin, require_admin


def _prune_old_logs(log_dir: Path, days: int = 7) -> int:
    """刪掉 log_dir 內超過 days 天的舊 log 與診斷截圖，避免長期累積佔空間。

    只清 *.log / *.png / *.jpg（log 與診斷圖），保留校正工具的 *_mark 資料夾。
    回傳刪除的檔案數。任何錯誤都吞掉（清理失敗不影響主程式啟動）。
    """
    cutoff = time.time() - days * 86400
    removed = 0
    try:
        for p in log_dir.rglob("*"):
            try:
                if not p.is_file():
                    continue
                if p.suffix.lower() not in (".log", ".png", ".jpg", ".jpeg"):
                    continue
                # 保留校正/標記工具的輸入資料夾（lobby_mark、*_mark 等）
                if any(part.endswith("_mark") for part in p.parts):
                    continue
                if p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
                    removed += 1
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    return removed


def _setup_logging() -> Path:
    """同時輸出到主控台與 logs/run_*.log，方便事後/跨視窗檢視。"""
    log_dir = _logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    removed = _prune_old_logs(log_dir, days=7)
    log_path = log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.handlers = [sh, fh]
    if removed:
        logging.getLogger(__name__).info("已清理 %d 個 7 天前的舊 log／診斷圖", removed)
    return log_path


def _read_launch_settings() -> dict:
    """讀取 exe 旁的「啟動設定.txt」，決定模式與是否真下注。

    回傳 dict：{"mode": "patrol"/"stay"/None, "live": True/False/None, "selftest": bool}
    讀不到或格式不符時對應值為 None / False，由上層套用預設。
    """
    from star_follow import paths

    out: dict = {
        "mode": None,
        "live": None,
        "selftest": False,
        "balance_test": False,
        "list_windows": False,
        "nav_test": False,
        "stay_table": None,
        "stay_targets": [],
        "account_name": None,
        "follow_ratio": None,  # (lo, hi) 比例範圍；False=關閉(跟原額)；None=不指定(用 config 預設)
    }
    p = paths.launch_settings_path()
    if not p.is_file():
        return out

    raw = b""
    try:
        raw = p.read_bytes()
    except Exception:  # noqa: BLE001
        return out
    text = None
    for enc in ("utf-8-sig", "utf-8", "cp950", "big5", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except Exception:  # noqa: BLE001
            continue
    if text is None:
        return out

    for line in text.splitlines():
        s = line.strip()
        if not s or s[0] in "#；;/":
            continue
        s = s.replace("＝", "=").replace("：", "=").replace(":", "=")
        if "=" not in s:
            continue
        key, val = s.split("=", 1)
        key = key.strip().replace(" ", "")
        val = val.strip().replace(" ", "")
        if any(k in key for k in ("模式", "mode")):
            if any(v in val for v in ("換房", "巡房", "patrol")):
                out["mode"] = "patrol"
            elif any(v in val for v in ("掛房", "單桌", "stay")):
                out["mode"] = "stay"
            elif any(v in val for v in ("回桌", "導覽", "辨識", "回桌診斷", "navtest", "nav")):
                out["nav_test"] = True
            elif any(v in val for v in ("視窗", "視窗診斷", "list-windows", "windows")):
                out["list_windows"] = True
            elif any(v in val for v in ("餘額測試", "餘額", "balance")):
                out["balance_test"] = True
            elif any(v in val for v in ("檢查", "檢測", "自我", "selftest", "test")):
                out["selftest"] = True
        elif any(k in key for k in ("下注", "實戰", "bet", "live")):
            if any(v in val for v in ("開", "啟", "是", "on", "true", "yes", "1")):
                out["live"] = True
            elif any(v in val for v in ("關", "閉", "否", "off", "false", "no", "0", "測試")):
                out["live"] = False
        elif any(k in key for k in ("桌號", "桌号", "桌", "table")):
            import re as _re

            m = _re.search(r"\d+", val)
            if m:
                out["stay_table"] = int(m.group())
        elif any(k in key for k in ("對象", "对象", "跟注對象", "target")):
            # 一行可填多個（用 , 、 ， 空白分隔），也可多行累加
            parts = [p for p in _split_targets(val) if p]
            out["stay_targets"].extend(parts)
        elif any(k in key for k in ("帳號", "帐号", "帳戶", "account", "玩家")):
            if val:
                out["account_name"] = val
        elif any(k in key for k in ("跟注比例", "下注比例", "比例", "ratio")):
            out["follow_ratio"] = _parse_ratio_setting(val)
    return out


def _parse_ratio_setting(val: str):
    """解析『跟注比例』：可填範圍 0.8-0.99 / 0.8~0.99，或單一值 0.9（固定比例），
    或填 關/off/100% 代表不縮小（跟對方原額）。讀不懂回 None（沿用 config 預設）。
    """
    import re as _re

    v = val.strip().lower().replace("%", "")
    if not v:
        return None
    if any(x in v for x in ("關", "閉", "off", "no", "false", "原額", "100")):
        return False
    nums = [float(x) for x in _re.findall(r"\d+(?:\.\d+)?", v)]
    if not nums:
        return None
    # 允許用百分比寫法（如 80-99）：>1 視為百分比自動換算。
    nums = [n / 100.0 if n > 1.0 else n for n in nums]
    lo = min(nums)
    hi = max(nums)
    lo = max(0.01, min(lo, 1.0))
    hi = max(0.01, min(hi, 1.0))
    return (lo, hi)


def _split_targets(val: str) -> list[str]:
    import re as _re

    return [p.strip() for p in _re.split(r"[,，、;；\s]+", val) if p.strip()]


# 「啟動設定.txt」每個選項的別名（判斷舊檔有沒有這個選項）與要補上的範本區塊。
# 之後新增選項只要往這裡加一筆，自動更新的人就會被「保留原值、補上新選項」。
_LAUNCH_OPTION_KEYS: dict[str, tuple[str, ...]] = {
    "帳號": ("帳號", "帐号", "帳戶", "account", "玩家"),
    "桌號": ("桌號", "桌号", "桌", "table"),
    "對象": ("對象", "对象", "跟注對象", "target"),
    "跟注比例": ("跟注比例", "下注比例", "比例", "ratio"),
}
_LAUNCH_OPTION_BLOCKS: dict[str, str] = {
    "帳號": (
        "# 【帳號】這台掛機的玩家帳號名稱（餘額上傳到試算表時用）。\n"
        "#    直接打字輸入，不再用 OCR 辨識名字（OCR 容易讀錯）。\n"
        "#    例：帳號=我的帳號名\n"
        "帳號="
    ),
    "桌號": (
        "# 【桌號】掛房要固定待著的桌號（例：5）。\n"
        "#    被踢出或當機跳回大廳時，程式會自動回到這一桌。\n"
        "#    留空或填 0 = 不指定（待在當前桌；回大廳時回任一桌）。\n"
        "桌號="
    ),
    "對象": (
        "# 【對象】掛房要跟注的對象暱稱，可填多個，用「、」或逗號分隔。\n"
        "#    例：對象=阿明、小華、大雄\n"
        "#    留空 = 沿用 follow_list.json 的名單。\n"
        "#    若「指定的對象全部都不在這桌」→ 程式會暫停跟注（不補注防踢、\n"
        "#    被踢回大廳也不會自動回桌），並透過 Telegram 通知。\n"
        "對象="
    ),
    "跟注比例": (
        "# 【跟注比例】跟注金額 = 對方金額 × 隨機比例，再無條件進位到整數注（最低 1000）。\n"
        "#    避免每把都跟對方一模一樣太明顯。例：對方 10000、比例 0.83 → 進位到 9000。\n"
        "#    填範圍 = 每把在此範圍隨機（建議）：例 跟注比例=0.8-0.99\n"
        "#    填單一值 = 固定比例：例 跟注比例=0.9\n"
        "#    填『關』 = 不縮小，跟對方原額：跟注比例=關\n"
        "#    留空 = 用預設 0.8~0.99。\n"
        "跟注比例=0.8-0.99"
    ),
}


def _migrate_launch_settings() -> None:
    """智能合併『啟動設定.txt』：保留使用者已填的值，只把『缺少的新選項』以註解
    範本附加到檔尾（自動更新會保留舊檔，否則新選項不會出現在舊使用者的檔案裡）。
    """
    from star_follow import paths

    p = paths.launch_settings_path()
    if not p.is_file():
        return
    try:
        raw = p.read_bytes()
    except Exception:  # noqa: BLE001
        return
    text = None
    for enc in ("utf-8-sig", "utf-8", "cp950", "big5", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except Exception:  # noqa: BLE001
            continue
    if text is None:
        return

    present: set[str] = set()
    for line in text.splitlines():
        s = line.strip()
        if not s or s[0] in "#；;/":
            continue
        s = s.replace("＝", "=").replace("：", "=").replace(":", "=")
        if "=" not in s:
            continue
        key = s.split("=", 1)[0].strip().replace(" ", "")
        for opt, aliases in _LAUNCH_OPTION_KEYS.items():
            if any(a in key for a in aliases):
                present.add(opt)

    missing = [opt for opt in _LAUNCH_OPTION_BLOCKS if opt not in present]
    if not missing:
        return
    addition = "\n\n# === 新版自動補上的選項（可自行修改後存檔）===\n"
    addition += "\n\n".join(_LAUNCH_OPTION_BLOCKS[o] for o in missing)
    new_text = text.rstrip("\r\n") + "\n" + addition + "\n"
    try:
        p.write_text(new_text, encoding="utf-8-sig")
        logging.getLogger(__name__).info(
            "啟動設定.txt 已自動補上新選項：%s（原設定保留）", "、".join(missing)
        )
    except Exception:  # noqa: BLE001
        pass


def _sync_config_for_version() -> None:
    """新版啟動時，用打包內建的參考 config 覆蓋外部 config.yaml（每個版本只做一次）。

    自動更新的換檔腳本會「保留」使用者既有的 config.yaml，導致舊使用者更新後仍用
    舊設定（新座標／門檻沒跟上）。這裡在新 exe 啟動時依版本標記做一次覆蓋，把新版
    config 套上去。使用者專屬資料只有 data\\follow_list.json 與啟動設定.txt，不受影響。
    （開發模式不動；只有打包且內建了參考檔的版本才會生效。）
    """
    from star_follow import paths
    from star_follow.version import __version__

    if not paths.is_frozen():
        return
    ref = paths.resource_dir() / "_ref" / "config.yaml"
    if not ref.is_file():
        return
    target = paths.config_path()
    marker = paths.app_dir() / "data" / ".config_version"
    try:
        seen = marker.read_text(encoding="utf-8").strip()
    except Exception:  # noqa: BLE001
        seen = ""
    if seen == __version__ and target.is_file():
        return
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ref, target)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(__version__, encoding="utf-8")
        print(f"已套用 v{__version__} 設定檔 config.yaml（追蹤名單與啟動設定保留）")
    except Exception as exc:  # noqa: BLE001
        print(f"套用新版設定檔失敗，沿用現有 config.yaml：{exc}")


def _balance_test() -> int:
    """讀一次左下角餘額與帳號名，存下截圖供校正 roi.balance / roi.account_name。"""
    import time

    import numpy as np
    from PIL import Image

    from star_follow.config import load_config
    from star_follow.monitor.sheet_uploader import _read_roi
    from star_follow.paths import logs_dir
    from star_follow.vision.ocr import ocr_account_name, ocr_balance

    cfg = load_config()
    out = logs_dir()
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    print("=== 餘額／帳號讀取測試 ===")

    def _do(roi_key: str, label: str, ocr_fn):
        rect = cfg.roi.get(roi_key)
        print(f"\nroi.{roi_key} = {rect}")
        img = _read_roi(cfg, roi_key)
        if img is None:
            print(f"找不到遊戲視窗或未設定 roi.{roi_key}，請先把星城開好再測。")
            return
        crop_path = out / f"{roi_key}_{ts}.png"
        try:
            Image.fromarray(np.asarray(img)).save(crop_path)
            print(f"已存擷取畫面：{crop_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"存圖失敗：{exc}")
        value, _ = ocr_fn(img)
        if isinstance(value, int):
            print(f"OCR 讀到的{label}：{value:,}" if value else f"OCR 讀不到{label}（請校正 roi.{roi_key}）")
        else:
            print(f"OCR 讀到的{label}：{value!r}" if value else f"OCR 讀不到{label}（請校正 roi.{roi_key}）")

    _do("balance", "餘額", ocr_balance)
    _do("account_name", "帳號名", ocr_account_name)
    print(f"\nsheet.account_name 設定值：{cfg.sheet.account_name!r}（有填就用這個，不靠 OCR）")
    try:
        input("\n按 Enter 關閉本視窗...")
    except Exception:  # noqa: BLE001
        pass
    return 0


def _nav_test() -> int:
    """回桌辨識診斷：每 2 秒截一次當前畫面，印出它判斷你在哪一頁、各模板比對分數、
    以及它會點哪顆鈕。可一邊切換『大廳/棋牌/入口/牌桌』一邊看，方便校正門檻或補模板。
    """
    import time as _t

    import numpy as np
    from PIL import Image

    from star_follow.automation.lobby_nav import diagnose_screen
    from star_follow.capture.screen import capture_client
    from star_follow.capture.window import find_game_window
    from star_follow.config import load_config
    from star_follow.paths import logs_dir

    cfg = load_config()
    wcfg = cfg.window
    win = find_game_window(wcfg.title_substring, title_aliases=wcfg.title_aliases or None)
    if win is None:
        print("[找不到] 沒抓到星城視窗。請先開好星城，並用『系統管理員』執行。")
        try:
            input("\n按 Enter 關閉...")
        except Exception:  # noqa: BLE001
            pass
        return 1

    out = logs_dir()
    out.mkdir(parents=True, exist_ok=True)
    print("=== 回桌辨識診斷 ===")
    print(f"視窗：{win.client_width}x{win.client_height}  標題={win.title!r}")
    print("每 2 秒判斷一次。請依序停留：")
    print("  ① 首頁大廳  ② 棋牌大廳(不要滑)  ③ 棋牌大廳(滑到百家樂)  ④ 百家樂入口  ⑤ 牌桌")
    print("分數：百家樂卡片>=0.58=可點；<0.58且在棋牌區=要滑動。")
    print("牌桌：僅 room_switch.png 模板命中才算（無模板請在牌桌跑 mark_rooms 存檔）。")
    print("（按 Ctrl+C 結束）\n")

    thr = {
        "lobby_random_select.png": 0.60,
        "lobby_qipai_menu.png": 0.55,
        "lobby_confirm_button.png": 0.60,
        "lobby_baccarat_card.png": 0.62,
    }
    n = 0
    try:
        while True:
            n += 1
            frame = capture_client(win)
            info = diagnose_screen(frame, cfg, win)
            ts = _t.strftime("%H%M%S")
            shot = out / f"navtest_{ts}_{n}.png"
            try:
                Image.fromarray(np.asarray(frame)).save(shot)
            except Exception:  # noqa: BLE001
                pass
            conf = info.get("confidence", 0) * 100
            print(
                f"[{ts}] 【{info.get('phase_label', info['phase'])}】"
                f" 信心={conf:.0f}%  phase={info['phase']}"
            )
            print(
                f"    模板: 百家樂卡={info['card_score']:.2f}"
                f" 隨機選台={info['random_score']:.2f} 棋牌分頁={info.get('qipai_tab', '?')}"
            )
            print(
                f"    結構: 牌列={info.get('card_row_score', 0):.2f}(>=0.48才算棋牌大廳)"
                f"  在牌桌={info.get('table_hud')} (換桌模板/倒數OCR/左下桌號/籌碼列)"
            )
            ps = info.get("phase_scores") or {}
            if ps:
                ranked = sorted(ps.items(), key=lambda x: -x[1])
                print("    各狀態得分: " + " | ".join(f"{k}={v:.2f}" for k, v in ranked[:5]))
            ocr = info.get("ocr") or {}
            if ocr:
                print(f"    OCR: 隨機區={ocr.get('random','')!r} 棋牌鈕={ocr.get('menu','')!r}"
                      f" 卡區={ocr.get('baccarat','')!r}")
            print(f"    → {info.get('suggested_action', '')}")
            for r in info.get("reasons") or []:
                print(f"       {r}")
            for s in info["scores"]:
                name = s["template"]
                t = thr.get(name, 0.55)
                rg = s["region"]
                wh = s["whole"]
                rg_txt = f"區塊{rg[2]:.2f}@({rg[0]},{rg[1]})" if rg else "區塊:模板缺檔/區塊太小"
                wh_txt = f"整張{wh[2]:.2f}@({wh[0]},{wh[1]})" if wh else "整張:無"
                hit = "比中" if (rg and rg[2] >= t) else ("(整張有到)" if (wh and wh[2] >= t) else "未中")
                print(f"    {s['label']:<14}門檻{t:.2f}  {rg_txt}  {wh_txt}  {hit}")
            print(f"    截圖：{shot.name}\n")
            _t.sleep(2.0)
    except KeyboardInterrupt:
        print("\n結束診斷。請把上面的輸出，加上 logs 裡的 navtest_*.png 給我。")
    return 0


def _list_windows() -> int:
    """列出符合設定的星城視窗；找不到時列出其他大視窗供對照。"""
    from star_follow.capture.window import (
        find_game_window,
        list_candidate_windows,
        list_large_windows,
    )
    from star_follow.config import load_config

    cfg = load_config()
    wcfg = cfg.window
    print("=== 星城視窗診斷 ===")
    print(f"設定 title_substring: {wcfg.title_substring!r}")
    if wcfg.title_aliases:
        print(f"額外 title_aliases: {wcfg.title_aliases}")

    cands = list_candidate_windows(
        wcfg.title_substring,
        title_aliases=wcfg.title_aliases or None,
    )
    if cands:
        print("\n找到以下符合的視窗（程式會用最大那個）：")
        for w, h, hwnd, vis, mini, title in cands:
            print(f"  {w}x{h}  hwnd={hwnd}  可見={vis}  最小化={mini}")
            print(f"    標題: {title}")
        win = find_game_window(wcfg.title_substring, title_aliases=wcfg.title_aliases or None)
        if win:
            print(f"\n目前會使用: {win.client_width}x{win.client_height}  標題={win.title!r}")
        print("\n若程式仍說找不到，請確認星城沒被最小化，且建議已進入百家樂牌桌。")
        return 0

    print("\n[找不到] 沒有任何視窗標題符合「星城Online / 星城」等關鍵字。")
    print("\n請業主確認：")
    print("  1. 星城已開啟（不要只開登入器就關掉）")
    print("  2. 視窗不要最小化到工作列")
    print("  3. 建議已點進「百家樂」牌桌（不要只停在大廳選單）")
    print("  4. 星城與本程式都要用「系統管理員」執行（右鍵 → 以系統管理員身分執行）")
    print("\n以下是目前畫面上較大的視窗標題，請找哪一個是星城：")
    print("（找到後把完整標題填進 config.yaml 的 title_substring 或 title_aliases）\n")
    for w, h, vis, mini, title, cls in list_large_windows():
        print(f"  {w}x{h}  可見={vis}  最小化={mini}  類別={cls}  |  {title}")
    return 1


def _test_crop(image_path: str, target: str | None = None) -> int:
    """離線重跑單欄表頭裁切 OCR（用客戶端 stats_debug 的 colN-empty.png 查根因）。"""
    import os

    import numpy as np
    import pytesseract
    from PIL import Image

    from star_follow import paths
    from star_follow.vision import stats_parser as sp
    from star_follow.vision.ocr import ocr_name_trace

    p = Path(image_path)
    if not p.is_file():
        print(f"找不到檔案：{p}")
        return 1
    img = np.array(Image.open(p).convert("RGB"))
    print("=== StarFollow 表頭裁切 OCR 診斷 ===")
    print(f"檔案：{p}")
    print(f"shape：{img.shape}")
    print(f"frozen：{paths.is_frozen()}  app_dir：{paths.app_dir()}")
    print(f"tesseract_cmd：{pytesseract.pytesseract.tesseract_cmd}")
    print(f"TESSDATA_PREFIX：{os.environ.get('TESSDATA_PREFIX')}")
    print()

    tr = ocr_name_trace(img)
    print(f"ink：{tr.get('ink')}")
    for st in tr.get("steps", []):
        print(
            f"  [{st.get('step')}] raw={st.get('raw')!r} "
            f"cleaned={st.get('cleaned')!r} "
            f"accepted={st.get('accepted')} reason={st.get('reason')}"
        )
    print(f"candidates：{tr.get('candidates')}")
    if target:
        col = sp.find_column_for_player_cands([(0, tr.get("candidates") or [])], target)
        print(f"比對目標 {target!r} -> col={col}")
    print("=== 結束 ===")
    return 0


def _selftest() -> int:
    """打包後自我檢查：確認設定檔、語言檔、Tesseract、OCR 都能正確載入。"""
    import os

    from star_follow import paths
    from star_follow.config import load_config

    print("=== StarFollow 自我檢查 ===")
    print(f"打包模式 frozen: {paths.is_frozen()}")
    print(f"程式資料夾 app_dir: {paths.app_dir()}")
    print(f"設定檔 config.yaml: {paths.config_path()}  存在={paths.config_path().is_file()}")
    print(f"名單 follow_list.json: {paths.follow_list_path()}  存在={paths.follow_list_path().is_file()}")
    print(f"Tesseract 執行檔: {paths.tesseract_exe()}  存在={paths.tesseract_exe().is_file()}")
    print(f"語言檔資料夾 tessdata: {paths.tessdata_dir()}  存在={paths.tessdata_dir().is_dir()}")
    exe, data = paths.ocr_runtime()
    print(f"OCR 實際使用: tesseract={exe}  tessdata={data}")
    print(f"TESSDATA_PREFIX: {os.environ.get('TESSDATA_PREFIX')}")

    ok = True
    try:
        cfg = load_config()
        print(f"設定載入成功：room.mode={cfg.room.mode} tables={cfg.room.tables}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[失敗] 設定載入：{exc}")

    try:
        fl = FollowList.load()
        print(f"跟注名單：{[e.name for e in fl.entries]}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[失敗] 名單載入：{exc}")

    try:
        import numpy as np
        import pytesseract
        from PIL import Image

        print(f"Tesseract 命令：{pytesseract.pytesseract.tesseract_cmd}")
        blank = Image.fromarray(np.full((48, 160, 3), 255, dtype=np.uint8))
        # 載入 chi_tra+eng 做一次實際辨識（空白圖回空字串，但會驗證語言檔可被讀取）
        pytesseract.image_to_string(blank, lang="chi_tra+eng")
        print("Tesseract OCR（chi_tra+eng）載入成功")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"[失敗] Tesseract OCR：{exc}")

    print("=== 結果：", "全部通過 OK" if ok else "有錯誤 FAIL", "===")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--live",
        action="store_true",
        help="連下注也真點擊（預設 dry-run：會開關統計表，但不點籌碼）",
    )
    parser.add_argument("--add", type=str, help="新增跟注暱稱後結束（不啟動引擎）")
    parser.add_argument("--patrol", action="store_true", help="換房模式（巡房跟注），覆寫 config 的 room.mode")
    parser.add_argument("--stay", action="store_true", help="掛房模式（單桌跟注），覆寫 config 的 room.mode")
    parser.add_argument("--selftest", action="store_true", help="自我檢查：確認設定檔/語言檔/OCR 可正常載入後結束")
    parser.add_argument(
        "--test-crop",
        metavar="PNG",
        help="離線診斷：對 stats_debug 的 colN-empty.png 重跑表頭 OCR，查客戶端為何讀不到",
    )
    parser.add_argument(
        "--test-target",
        metavar="NAME",
        help="配合 --test-crop：指定追蹤暱稱，看 fuzzy 比對是否過門檻",
    )
    parser.add_argument("--balance-test", action="store_true", help="測試讀取左下角餘額與帳號名（校正 roi.balance / roi.account_name）")
    parser.add_argument("--list-windows", action="store_true", help="診斷：列出星城視窗（找不到時列出其他大視窗）")
    parser.add_argument("--nav-test", action="store_true", help="診斷：每2秒判斷目前在大廳/棋牌/入口/牌桌哪一頁，印出各模板分數")
    args = parser.parse_args()

    if getattr(args, "nav_test", False):
        return _nav_test()

    if getattr(args, "list_windows", False):
        return _list_windows()

    if getattr(args, "balance_test", False):
        return _balance_test()

    # 新版設定檔：依版本把內建的新 config.yaml 套上去（自動更新會保留舊 config，
    # 這裡確保新版座標/門檻能跟上；data\追蹤名單與啟動設定.txt 不受影響）
    _sync_config_for_version()
    # 啟動設定檔（雙擊 exe 時用；命令列參數優先）
    # 先做智能合併：保留舊值、補上新選項（自動更新者的舊檔才看得到新選項）
    _migrate_launch_settings()
    settings = _read_launch_settings()

    if getattr(args, "test_crop", None):
        return _test_crop(args.test_crop, getattr(args, "test_target", None))

    if args.selftest or (settings["selftest"] and not (args.patrol or args.stay)):
        return _selftest()

    if settings["nav_test"] and not (args.patrol or args.stay):
        return _nav_test()

    if settings["list_windows"] and not (args.patrol or args.stay):
        return _list_windows()

    if settings["balance_test"] and not (args.patrol or args.stay):
        return _balance_test()

    # 自動更新：偵測到新版會下載換檔並重啟（此行之後就不會再往下執行）
    try:
        from star_follow.config import load_config
        from star_follow.update.updater import maybe_update

        maybe_update(load_config().update)
    except Exception:  # noqa: BLE001  更新流程任何錯誤都不應擋住正常啟動
        pass

    # 模式：CLI > 設定檔 > config 預設
    if args.patrol:
        mode_override = "patrol"
    elif args.stay:
        mode_override = "stay"
    else:
        mode_override = settings["mode"]

    # 下注：CLI --live > 設定檔；預設為測試(不下注)以策安全
    if args.live:
        live = True
    elif settings["live"] is not None:
        live = settings["live"]
    else:
        live = False

    log_path = _setup_logging()
    fl = FollowList.load()
    if args.add:
        fl.add(args.add)
        fl.save()
        print(f"已加入跟注: {args.add}")
        return 0

    require_admin(live=live)
    engine = FollowEngine(follow=fl, dry_run=not live)
    if mode_override:
        engine.cfg.room.mode = mode_override
    # 掛房：套用啟動設定的「桌號 / 對象」
    if settings.get("stay_table") is not None:
        engine.cfg.room.stay_table = settings["stay_table"]
    if settings.get("stay_targets"):
        engine.cfg.room.stay_targets = list(settings["stay_targets"])
    # 掛房且有指定對象 → 用指定對象覆寫跟注名單（不動 follow_list.json）
    if engine.cfg.room.mode == "stay" and engine.cfg.room.stay_targets:
        engine.follow = FollowList.from_names(engine.cfg.room.stay_targets)
    # 餘額上傳用的帳號名：以啟動設定手動輸入為準（不用 OCR 讀名字，避免誤判）
    if settings.get("account_name"):
        engine.cfg.sheet.account_name = settings["account_name"]
    # 跟注比例（擬真）：啟動設定可覆寫 config 的範圍，或關閉
    fr = settings.get("follow_ratio")
    if fr is False:
        engine.cfg.betting.follow_ratio_enabled = False
    elif isinstance(fr, tuple):
        engine.cfg.betting.follow_ratio_enabled = True
        engine.cfg.betting.follow_ratio_min = fr[0]
        engine.cfg.betting.follow_ratio_max = fr[1]
    t = engine.cfg.timing
    mode = "LIVE（含下注）" if not engine.dry_run else "dry-run（開關統計+OCR，不下注）"
    room_mode = "換房巡房" if engine.cfg.room.mode == "patrol" else "掛房（單桌）"
    admin_tag = "管理員" if is_admin() else "一般使用者"
    from star_follow.version import __version__ as _ver

    print(f"StarFollow v{_ver}")
    print("引擎啟動", mode, f"[{room_mode}]", f"({admin_tag})")
    print(f"記錄檔：{log_path}")
    bset = engine.cfg.betting
    if bset.follow_ratio_enabled and not (bset.follow_ratio_min >= 1.0 and bset.follow_ratio_max >= 1.0):
        print(
            f"跟注擬真：下注額 = 對方 × {bset.follow_ratio_min:.2f}~{bset.follow_ratio_max:.2f} 隨機，"
            f"無條件進位到 {bset.follow_ratio_round_to}"
        )
    else:
        print("跟注擬真：關閉（跟對方原額）")
    from star_follow.vision.ocr import warmup_ocr

    warmup_ocr()
    if engine.cfg.room.mode == "patrol":
        print(f"換房：進房綠燈 T>={engine.cfg.room.min_enter_t} 才開統計跟注；下注後等開牌再換桌")
        print(f"桌號序列：{engine.cfg.room.tables}")
    else:
        print(
            f"時間軸：T={t.open_stats_at_t} 開統計 -> T<={t.prefetch_at_t} 預定位 -> "
            f"T={t.finalize_at_t} 定稿（僅 T>={t.min_round_start_t} 且錨定成功才開局）"
        )
        if engine.cfg.room.stay_table:
            print(f"掛房固定桌號：No.{engine.cfg.room.stay_table}（被踢/當機會自動回此桌）")
        else:
            print("掛房未指定桌號（待在當前桌；回大廳時回任一桌）。可在啟動設定填『桌號=5』")
        tgt = engine.cfg.room.stay_targets or fl.active()
        print(f"掛房追蹤對象：{('、'.join(tgt) if tgt else '（無）')}")
        if engine.cfg.room.stay_pause_when_targets_absent:
            n = engine.cfg.room.stay_absent_rounds_to_pause
            mode = (getattr(engine.cfg.room, "stay_on_absent", "keep") or "keep").lower()
            if mode == "stop":
                print(
                    f"對象全離桌偵測：連續 {n} 局統計表都讀不到任何對象 → Telegram 通知後停止程式"
                )
            elif mode == "pause":
                print(
                    f"對象全離桌偵測：連續 {n} 局都讀不到任何對象 → Telegram 通知後暫停跟注、不回桌"
                )
            else:
                print(
                    f"對象全離桌偵測：連續 {n} 局都讀不到任何對象 → Telegram 通知，但持續防踢守桌，"
                    "對象回來自動恢復跟注"
                )
    if engine.dry_run:
        print("真下注：python -m star_follow.tools.run --live")
        print("標記籌碼：python -m star_follow.tools.mark_chips")

    uploader = None
    if engine.cfg.sheet.enabled:
        from star_follow.monitor.sheet_uploader import SheetUploader, UploadGate

        # 上傳跟著引擎階段走：只在穩定在桌且空檔（等開局/下注完等開牌）才上傳，
        # 不與定位/讀統計表搶資源。
        gate = UploadGate(engine)
        uploader = SheetUploader(engine.cfg, gate=gate)
        uploader.start()
        print("餘額上傳：在牌桌空檔（避開定位/讀表）每整點上傳到 Google 試算表")

    try:
        engine.run_loop()
    except KeyboardInterrupt:
        engine.stop()
    finally:
        if uploader is not None:
            uploader.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
