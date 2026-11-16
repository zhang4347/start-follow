"""簡易操作介面：雙擊 StarFollow.exe 先開這個視窗，選模式／改設定後按「開始」。

設計重點：
  * 設定照樣存在「啟動設定.txt」（只改「鍵=值」那一行的值，註解全部保留），
    所以舊的記事本用法、自動更新的智能合併都不受影響。
  * 追蹤名單直接編輯 data\\follow_list.json。
  * 視窗只負責「選好要做什麼」：回傳一個動作 dict 給 run.main() 繼續執行，
    不另開行程（避免再跳一次 UAC）。
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# 啟動設定.txt 讀寫（保留註解、只改值）
# ---------------------------------------------------------------------------

# (標準鍵, 別名…)；順序有意義：與 run._read_launch_settings 的判斷順序一致，
# 例如「單局總下注警告」「跟注比例」要排在「下注」前面，否則會被吃掉。
_FIELDS: list[tuple[str, tuple[str, ...]]] = [
    ("模式", ("模式", "mode")),
    ("單局總下注警告", ("單局總下注警告", "单局总下注警告", "總下注警告", "总下注警告", "單局總額警告", "round_total_warn")),
    ("試探升級動作", ("試探升級動作", "試探動作", "probe_action")),
    ("連續試探升級局數", ("連續試探", "試探換桌局數", "試探升級", "probe_escalate")),
    ("反試探", ("反試探", "防試探", "試探防護", "probe_guard")),
    ("防踢局數", ("防踢局數", "防踢", "補注局數", "anti_kick")),
    ("跟上一局", ("跟上一局", "跟上局", "上局跟", "延遲跟", "follow_prev")),
    ("跟注比例", ("跟注比例", "下注比例", "比例", "倍數", "倍率", "ratio")),
    ("下注", ("下注", "實戰", "bet", "live")),
    ("搶房房號", ("搶房房號", "搶房")),
    ("巡房房間", ("巡房房間", "巡防房間", "巡房桌號", "巡防桌號", "巡房範圍", "巡防範圍", "循房", "patrol_room", "patrol_table")),
    ("影像辨識房間", ("影像辨識房間", "圖像辨識房間", "影像房間", "影像辨識", "template_room")),
    ("影像比對門檻", ("影像比對門檻", "比對門檻", "影像門檻", "match_threshold")),
    ("影像比對領先差", ("影像比對領先差", "領先差", "比對領先", "match_margin")),
    ("桌號", ("桌號", "桌号", "桌", "table")),
    ("對象", ("對象", "对象", "跟注對象", "target")),
    ("帳號", ("帳號", "帐号", "帳戶", "account", "玩家")),
    ("單注上限", ("單注上限", "单注上限", "最大跟注", "max_bet", "maxbet")),
]
_CANON = {name for name, _ in _FIELDS}


def _canonical(key: str) -> str | None:
    key = key.strip().replace(" ", "")
    if key in _CANON:
        return key
    for name, aliases in _FIELDS:
        if any(a in key for a in aliases):
            return name
    return None


def _split_kv(line: str) -> tuple[str, str] | None:
    s = line.strip()
    if not s or s[0] in "#；;/":
        return None
    for i, ch in enumerate(s):
        if ch in "=＝":
            return s[:i], s[i + 1 :]
    return None


def _read_text(p: Path) -> str:
    if not p.is_file():
        return ""
    raw = p.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp950", "big5", "latin-1"):
        try:
            return raw.decode(enc)
        except Exception:  # noqa: BLE001
            continue
    return ""


def read_values(p: Path) -> dict[str, str]:
    """{標準鍵: 原始字串值}。同鍵多行時取最後一行（與主程式一致），「對象」則合併。"""
    out: dict[str, str] = {}
    for line in _read_text(p).splitlines():
        kv = _split_kv(line)
        if not kv:
            continue
        name = _canonical(kv[0])
        if not name:
            continue
        val = kv[1].strip()
        if name == "對象" and out.get("對象") and val:
            out["對象"] = out["對象"] + "、" + val
        else:
            out[name] = val
    return out


def write_values(p: Path, updates: dict[str, str]) -> None:
    """把 updates 寫回檔案：改第一個同鍵行的值，之後重複的同鍵行改成註解；缺的鍵補在檔尾。"""
    text = _read_text(p)
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()
    written: set[str] = set()
    out_lines: list[str] = []
    for line in lines:
        kv = _split_kv(line)
        name = _canonical(kv[0]) if kv else None
        if name and name in updates:
            if name in written:
                out_lines.append("# （已改由操作介面設定）" + line.strip())
                continue
            indent = line[: len(line) - len(line.lstrip())]
            out_lines.append(f"{indent}{kv[0].strip()}={updates[name]}")
            written.add(name)
        else:
            out_lines.append(line)
    missing = [k for k in updates if k not in written]
    if missing:
        if out_lines and out_lines[-1].strip():
            out_lines.append("")
        out_lines.append("# === 操作介面補上的設定 ===")
        for k in missing:
            out_lines.append(f"{k}={updates[k]}")
    p.write_text(nl.join(out_lines) + nl, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# 介面
# ---------------------------------------------------------------------------

_MODES = [
    ("換房", "換房跟注", "一桌一桌巡，找名單上的人跟著下注"),
    ("掛房", "掛房跟注", "固定待在一桌跟注，不換桌（被踢只發 TG，不自動回桌）"),
    ("通知", "只通知", "巡各房，抓到追蹤對象下邊注就發 TG，不下注"),
    ("搶房", "搶房", "一直搶指定房號，搶到發 TG，期間補注防踢"),
]

_DEFAULTS = {
    "模式": "換房", "下注": "開", "帳號": "", "跟注比例": "0.8-0.99", "單注上限": "20萬",
    "單局總下注警告": "30萬", "跟上一局": "開", "反試探": "開", "連續試探升級局數": "2",
    "試探升級動作": "暫停", "巡房房間": "", "影像辨識房間": "", "影像比對門檻": "0.55",
    "影像比對領先差": "0.04", "桌號": "", "對象": "", "防踢局數": "2", "搶房房號": "",
}


def _is_on(v: str) -> bool:
    return any(x in (v or "") for x in ("開", "啟", "是", "on", "true", "yes", "1"))


def _mode_of(v: str) -> str:
    v = v or ""
    if any(x in v for x in ("搶房", "搶桌", "抢房", "grab")):
        return "搶房"
    if any(x in v for x in ("通知", "盯場", "通報", "雷達", "notify")):
        return "通知"
    if any(x in v for x in ("掛房", "單桌", "stay")):
        return "掛房"
    return "換房"


def run_launcher() -> dict | None:
    """顯示操作介面。回傳 {"action": "start"|"grab"|"selftest"|"list_windows"|"balance_test", ...}；
    使用者直接關掉視窗回 None。"""
    import tkinter as tk
    from tkinter import messagebox, simpledialog, ttk

    from star_follow import paths
    from star_follow.core.follow_list import FollowList
    from star_follow.version import __version__

    try:  # 高 DPI 螢幕字才不會糊
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:  # noqa: BLE001
        pass

    settings_path = paths.launch_settings_path()
    vals = dict(_DEFAULTS)
    vals.update({k: v for k, v in read_values(settings_path).items()})
    fl = FollowList.load()

    result: dict = {}
    root = tk.Tk()
    root.title(f"星城跟注 v{__version__}")
    root.resizable(False, False)
    base_font = ("Microsoft JhengHei UI", 11)
    root.option_add("*Font", base_font)
    style = ttk.Style(root)
    try:
        style.theme_use("vista")
    except Exception:  # noqa: BLE001
        pass
    style.configure(".", font=base_font)
    style.configure("Title.TLabel", font=("Microsoft JhengHei UI", 15, "bold"))
    style.configure("Hint.TLabel", foreground="#666666", font=("Microsoft JhengHei UI", 9))
    style.configure("Warn.TLabel", foreground="#c62828", font=("Microsoft JhengHei UI", 10, "bold"))
    style.configure("Big.TButton", font=("Microsoft JhengHei UI", 13, "bold"), padding=(18, 8))
    style.configure("Mode.TRadiobutton", font=("Microsoft JhengHei UI", 12, "bold"))

    v: dict[str, tk.Variable] = {}
    for key in ("帳號", "跟注比例", "單注上限", "單局總下注警告", "連續試探升級局數", "試探升級動作",
                "巡房房間", "影像辨識房間", "影像比對門檻", "影像比對領先差", "桌號", "對象", "防踢局數", "搶房房號"):
        v[key] = tk.StringVar(value=vals.get(key, ""))
    v["模式"] = tk.StringVar(value=_mode_of(vals.get("模式", "")))
    v["下注"] = tk.BooleanVar(value=_is_on(vals.get("下注", "開")))
    v["跟上一局"] = tk.BooleanVar(value=_is_on(vals.get("跟上一局", "開")))
    v["反試探"] = tk.BooleanVar(value=_is_on(vals.get("反試探", "開")))
    if v["試探升級動作"].get() not in ("暫停", "換桌"):
        v["試探升級動作"].set("換桌" if "換" in v["試探升級動作"].get() else "暫停")

    outer = ttk.Frame(root, padding=14)
    outer.pack(fill="both", expand=True)
    ttk.Label(outer, text="星城自動跟注", style="Title.TLabel").pack(anchor="w")
    ttk.Label(outer, text="選好模式、確認設定後按「儲存並開始」。設定會自動存檔，下次開啟會記住。",
              style="Hint.TLabel").pack(anchor="w", pady=(0, 8))

    nb = ttk.Notebook(outer)
    nb.pack(fill="both", expand=True)

    def row(parent, label, var, hint="", width=28):
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=4)
        ttk.Label(f, text=label, width=12).pack(side="left")
        ttk.Entry(f, textvariable=var, width=width).pack(side="left")
        if hint:
            ttk.Label(f, text=hint, style="Hint.TLabel").pack(side="left", padx=8)
        return f

    # ---------- 分頁一：主畫面 ----------
    tab_main = ttk.Frame(nb, padding=12)
    nb.add(tab_main, text="  主畫面  ")
    mode_box = ttk.LabelFrame(tab_main, text=" 要做什麼？ ", padding=10)
    mode_box.pack(fill="x")
    for key, label, desc in _MODES:
        f = ttk.Frame(mode_box)
        f.pack(fill="x", pady=2)
        ttk.Radiobutton(f, text=label, value=key, variable=v["模式"], style="Mode.TRadiobutton",
                        width=9, command=lambda: show_mode_fields()).pack(side="left")
        ttk.Label(f, text=desc, style="Hint.TLabel").pack(side="left")

    bet_box = ttk.Frame(tab_main)
    bet_box.pack(fill="x", pady=(10, 0))
    ttk.Checkbutton(bet_box, text="真的下注（實戰）", variable=v["下注"],
                    command=lambda: show_bet_warn()).pack(side="left")
    bet_warn = ttk.Label(bet_box, text="", style="Warn.TLabel")
    bet_warn.pack(side="left", padx=10)

    def show_bet_warn() -> None:
        if v["下注"].get():
            bet_warn.config(text="● 實戰：會真的下注", foreground="#c62828")
        else:
            bet_warn.config(text="○ 測試：只辨識不下注（搶房不會補注，會被踢）", foreground="#e65100")

    row(tab_main, "帳號名稱", v["帳號"], "餘額上傳試算表用")

    mode_fields = ttk.LabelFrame(tab_main, text=" 這個模式的設定 ", padding=10)
    mode_fields.pack(fill="x", pady=(10, 0))
    frames: dict[str, ttk.Frame] = {}

    f = ttk.Frame(mode_fields)
    row(f, "巡房房間", v["巡房房間"], "例：7、8、9、10　留空＝全部")
    frames["換房"] = f
    f = ttk.Frame(mode_fields)
    row(f, "巡房房間", v["巡房房間"], "例：7、8、9、10　留空＝全部")
    ttk.Label(f, text="抓到追蹤對象下邊注就發 TG，程式不會下注。", style="Hint.TLabel").pack(anchor="w")
    frames["通知"] = f
    f = ttk.Frame(mode_fields)
    row(f, "桌號", v["桌號"], "要固定待的房號，例：5", width=8)
    row(f, "對象", v["對象"], "例：阿明、小華　留空＝用追蹤名單")
    row(f, "防踢局數", v["防踢局數"], "幾局沒下注就補一手 閒1000（預設 2）", width=8)
    frames["掛房"] = f
    f = ttk.Frame(mode_fields)
    row(f, "要搶的房號", v["搶房房號"], "例：3", width=8)
    ttk.Label(f, text="① 先自己進任一間百家樂房　② 再按「儲存並開始」\n"
                      "搶到會發 TG；之後請關掉程式，改選「換房／掛房」重新開始跟注。",
              style="Hint.TLabel", justify="left").pack(anchor="w", pady=(4, 0))
    frames["搶房"] = f

    def show_mode_fields() -> None:
        for fr in frames.values():
            fr.pack_forget()
        frames[v["模式"].get()].pack(fill="x")

    # ---------- 分頁二：跟注設定 ----------
    tab_bet = ttk.Frame(nb, padding=12)
    nb.add(tab_bet, text="  跟注設定  ")
    row(tab_bet, "跟注比例", v["跟注比例"], "倍數 0.01~5；範圍如 0.8-0.99；填「關」＝跟原額", width=14)
    row(tab_bet, "單注上限", v["單注上限"], "對方單注超過就不跟；例 20萬；0＝不限", width=14)
    row(tab_bet, "單局總額警告", v["單局總下注警告"], "對方整局加總超過就發 TG；0＝不提醒", width=14)
    ttk.Checkbutton(tab_bet, text="跟上一局（對方拖到最後才下注時用；預設開）",
                    variable=v["跟上一局"]).pack(anchor="w", pady=(10, 4))
    probe = ttk.LabelFrame(tab_bet, text=" 反試探 ", padding=10)
    probe.pack(fill="x", pady=(8, 0))
    ttk.Checkbutton(probe, text="開啟反試探（對方用整數大額當餌時不跟）", variable=v["反試探"]).pack(anchor="w")
    pf = ttk.Frame(probe)
    pf.pack(fill="x", pady=(6, 0))
    ttk.Label(pf, text="連續").pack(side="left")
    ttk.Spinbox(pf, from_=0, to=9, width=4, textvariable=v["連續試探升級局數"]).pack(side="left", padx=4)
    ttk.Label(pf, text="局被試探就").pack(side="left")
    ttk.Combobox(pf, values=["暫停", "換桌"], width=6, state="readonly",
                 textvariable=v["試探升級動作"]).pack(side="left", padx=4)
    ttk.Label(pf, text="（0＝不升級）", style="Hint.TLabel").pack(side="left")

    # ---------- 分頁三：追蹤名單 ----------
    tab_list = ttk.Frame(nb, padding=12)
    nb.add(tab_list, text="  追蹤名單  ")
    ttk.Label(tab_list, text="要跟注的玩家暱稱。雙擊一列可切換「跟／不跟」。", style="Hint.TLabel").pack(anchor="w")
    tree = ttk.Treeview(tab_list, columns=("on", "name"), show="headings", height=10, selectmode="browse")
    tree.heading("on", text="跟注")
    tree.heading("name", text="暱稱")
    tree.column("on", width=70, anchor="center")
    tree.column("name", width=380)
    tree.pack(fill="both", expand=True, pady=6)

    def refresh_tree() -> None:
        tree.delete(*tree.get_children())
        for i, e in enumerate(fl.entries):
            tree.insert("", "end", iid=str(i), values=("✔ 跟" if e.enabled else "✘ 不跟", e.name))

    def selected_entry():
        sel = tree.selection()
        return fl.entries[int(sel[0])] if sel else None

    def add_name() -> None:
        name = simpledialog.askstring("新增追蹤", "輸入玩家暱稱（要跟遊戲裡完全一樣）：", parent=root)
        if name and name.strip():
            fl.add(name.strip())
            refresh_tree()

    def del_name() -> None:
        e = selected_entry()
        if e and messagebox.askyesno("刪除", f"確定從名單刪除「{e.name}」？", parent=root):
            fl.remove(e.name)
            refresh_tree()

    def toggle_name(_evt=None) -> None:
        e = selected_entry()
        if e:
            e.enabled = not e.enabled
            refresh_tree()

    tree.bind("<Double-1>", toggle_name)
    bf = ttk.Frame(tab_list)
    bf.pack(fill="x")
    ttk.Button(bf, text="＋ 新增", command=add_name).pack(side="left")
    ttk.Button(bf, text="切換跟／不跟", command=toggle_name).pack(side="left", padx=6)
    ttk.Button(bf, text="－ 刪除", command=del_name).pack(side="left")
    refresh_tree()

    # ---------- 分頁四：進階 ----------
    tab_adv = ttk.Frame(nb, padding=12)
    nb.add(tab_adv, text="  進階／工具  ")
    row(tab_adv, "影像辨識房間", v["影像辨識房間"], "格式 暱稱:房號、暱稱:房號", width=30)
    row(tab_adv, "比對門檻", v["影像比對門檻"], "0~1，跟錯人調高、比不中調低", width=8)
    row(tab_adv, "比對領先差", v["影像比對領先差"], "0~1，預設 0.04", width=8)
    tools = ttk.LabelFrame(tab_adv, text=" 診斷工具 ", padding=10)
    tools.pack(fill="x", pady=(12, 0))

    def open_path(pth: Path) -> None:
        try:
            os.startfile(str(pth))  # noqa: S606
        except Exception:  # noqa: BLE001
            pass

    def run_tool(action: str) -> None:
        if save(silent=True):
            result["action"] = action
            root.destroy()

    tf = ttk.Frame(tools)
    tf.pack(fill="x")
    ttk.Button(tf, text="自我檢查", command=lambda: run_tool("selftest")).pack(side="left")
    ttk.Button(tf, text="找不到遊戲視窗", command=lambda: run_tool("list_windows")).pack(side="left", padx=6)
    ttk.Button(tf, text="餘額讀取測試", command=lambda: run_tool("balance_test")).pack(side="left")
    tf2 = ttk.Frame(tools)
    tf2.pack(fill="x", pady=(6, 0))
    ttk.Button(tf2, text="開啟記錄資料夾", command=lambda: open_path(paths.logs_dir())).pack(side="left")
    ttk.Button(tf2, text="用記事本改設定檔",
               command=lambda: subprocess.Popen(["notepad.exe", str(settings_path)])).pack(side="left", padx=6)

    # ---------- 底部按鈕 ----------
    def collect() -> dict[str, str]:
        out = {k: (v[k].get() or "").strip() for k in v if isinstance(v[k], tk.StringVar)}
        out["模式"] = v["模式"].get()
        out["下注"] = "開" if v["下注"].get() else "關"
        out["跟上一局"] = "開" if v["跟上一局"].get() else "關"
        out["反試探"] = "開" if v["反試探"].get() else "關"
        return out

    def validate(d: dict[str, str]) -> str | None:
        from star_follow.tools import run as _run

        if d["跟注比例"] and _run._parse_ratio_setting(d["跟注比例"]) is None:
            return "「跟注比例」看不懂，請填像 0.8-0.99、0.9 或 關"
        for k in ("單注上限", "單局總下注警告"):
            if d[k] and _run._parse_amount_setting(d[k]) is None:
                return f"「{k}」看不懂，請填像 20萬、200000 或 0"
        for k in ("影像比對門檻", "影像比對領先差"):
            if d[k] and _run._parse_float_setting(d[k]) is None:
                return f"「{k}」要填 0~1 的小數"
        for k in ("桌號", "搶房房號", "防踢局數", "連續試探升級局數"):
            if d[k] and not re.fullmatch(r"\d{1,2}", d[k]):
                return f"「{k}」要填數字"
        if d["模式"] == "搶房" and not d["搶房房號"]:
            return "搶房要先填「要搶的房號」"
        return None

    def save(silent: bool = False) -> bool:
        d = collect()
        err = validate(d)
        if err:
            messagebox.showwarning("設定有誤", err, parent=root)
            return False
        try:
            write_values(settings_path, d)
            fl.save()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("存檔失敗", f"無法寫入設定檔：{exc}", parent=root)
            return False
        if not silent:
            messagebox.showinfo("已儲存", "設定已儲存。", parent=root)
        return True

    def start() -> None:
        d = collect()
        if not save(silent=True):
            return
        if d["模式"] == "搶房":
            result.update(action="grab", table=int(d["搶房房號"]))
        else:
            if d["模式"] in ("換房", "掛房") and not fl.active() and not (d["模式"] == "掛房" and d["對象"]):
                if not messagebox.askyesno("名單是空的", "追蹤名單沒有要跟的人，確定還要開始？", parent=root):
                    return
            result["action"] = "start"
        root.destroy()

    bottom = ttk.Frame(outer)
    bottom.pack(fill="x", pady=(12, 0))
    ttk.Button(bottom, text="▶ 儲存並開始", style="Big.TButton", command=start).pack(side="right")
    ttk.Button(bottom, text="只儲存", command=lambda: save()).pack(side="right", padx=8)
    ttk.Label(bottom, text=f"設定檔：{settings_path.name}", style="Hint.TLabel").pack(side="left")

    # 先用欄位最多的「掛房」量出視窗大小並固定，切換模式時視窗才不會忽大忽小
    frames["掛房"].pack(fill="x")
    root.update_idletasks()
    root.minsize(root.winfo_reqwidth(), root.winfo_reqheight())
    show_mode_fields()
    show_bet_warn()
    root.update_idletasks()
    w, h = root.winfo_reqwidth(), root.winfo_reqheight()
    root.geometry(f"+{max(0, (root.winfo_screenwidth() - w) // 2)}+{max(0, (root.winfo_screenheight() - h) // 3)}")
    root.lift()
    root.attributes("-topmost", True)
    root.after(300, lambda: root.attributes("-topmost", False))
    root.focus_force()
    root.mainloop()
    return result or None
