"""引導式標記「回房流程」需要的畫面與按鈕（一次標完，辨識最準）。

用法（遊戲開著、client 1280×720）：
  .\.venv\Scripts\python.exe -m star_follow.tools.mark_return_flow

四個步驟（每步：擷取畫面或載入截圖 → 在按鈕中央點一下）：
  ① 五局未押注提示窗 → 綠色『確定』鈕
  ② 棋牌大廳（按掉提示後）→ 『百家樂』卡片中央
  ③ 百家樂入口 → 『隨機選台/隨機入房』鈕
  ④ 首頁大廳 → 右側『棋牌』分頁（可選，掉到首頁才需要）

按『完成並套用』後會：
  - 把每個按鈕附近裁成模板圖，存到 vision/templates/（回房比對直接用）
  - 把按鈕座標寫進 config.yaml 的 click_points（模板比不到時的固定備援）
  - 原圖與疊點圖存到 logs/return_flow_mark/ 方便對照

標完務必『重新打包』，模板才會進到交付的 exe。
"""

from __future__ import annotations

import json
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from star_follow.capture.screen import capture_client
from star_follow.capture.window import find_game_window, focus_window
from star_follow.config import DEFAULT_CONFIG_PATH, load_config, save_config
from star_follow.paths import logs_dir

_TEMPLATES = Path(__file__).resolve().parents[1] / "vision" / "templates"

# 每步：config 鍵、標題、說明、模板檔名、裁切半寬半高（1280×720 基準）、是否可選
_STEPS: list[dict] = [
    {
        "key": "kick_idle_confirm",
        "title": "① 五局未押注提示窗 → 確定鈕",
        "instr": "觸發或載入『超過五局未押注，已退出遊戲』提示窗，點綠色【確定】鈕正中央。",
        "template": "lobby_confirm_button.png",
        "half": (70, 28),
        "optional": False,
    },
    {
        "key": "baccarat_card",
        "title": "② 棋牌大廳 → 百家樂卡片",
        "instr": "切到『按掉提示後』的棋牌大廳畫面，點【百家樂】那張卡片的正中央。",
        "template": "lobby_baccarat_card.png",
        "half": (62, 86),
        "optional": False,
    },
    {
        "key": "random_select",
        "title": "③ 百家樂入口 → 隨機選台/隨機入房",
        "instr": "進到百家樂入口畫面，點【隨機選台 / 隨機入房】鈕正中央。",
        "template": "lobby_random_select.png",
        "half": (92, 34),
        "optional": False,
    },
    {
        "key": "home_qipai",
        "title": "④ (可選) 首頁大廳 → 右側『棋牌』",
        "instr": "若有時會掉到首頁：切到首頁大廳，點右側【棋牌】分頁。用不到可直接略過。",
        "template": "lobby_qipai_menu.png",
        "half": (56, 36),
        "optional": True,
    },
]


class MarkReturnFlowApp:
    def __init__(self) -> None:
        self.cfg = load_config()
        self._idx = 0
        self._rgb: np.ndarray | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._scale = 1.0
        self._img_w = self.cfg.window.reference_width or 1280
        self._img_h = self.cfg.window.reference_height or 720
        # 每步資料：{key: {"rgb": np.ndarray, "pt": (x, y)}}
        self._data: dict[str, dict] = {}
        self.out_dir = logs_dir() / "return_flow_mark"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.root = tk.Tk()
        self.root.title("標記回房流程（提示窗→棋牌大廳→入口）")
        self.root.geometry("1180x900")

        bar = ttk.Frame(self.root)
        bar.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(bar, text="◀ 上一步", command=self._prev).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="下一步 ▶", command=self._next).pack(side=tk.LEFT, padx=2)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(bar, text="擷取星城視窗", command=self._capture).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="載入截圖…", command=self._load_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="清除本步的點", command=self._clear_point).pack(side=tk.LEFT, padx=2)
        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)
        ttk.Button(bar, text="完成並套用", command=self._apply).pack(side=tk.LEFT, padx=2)

        self.progress = ttk.Label(self.root, text="", foreground="#444444")
        self.progress.pack(fill=tk.X, padx=8, pady=(2, 0))
        self.title_lbl = ttk.Label(self.root, text="", font=("Microsoft JhengHei", 13, "bold"))
        self.title_lbl.pack(fill=tk.X, padx=8, pady=(4, 0))
        self.hint = ttk.Label(self.root, text="", wraplength=1130, justify=tk.LEFT)
        self.hint.pack(fill=tk.X, padx=8, pady=2)
        self.status = ttk.Label(self.root, text="", wraplength=1130, foreground="#0066cc")
        self.status.pack(fill=tk.X, padx=8)

        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.canvas = tk.Canvas(frame, bg="#222222", highlightthickness=0)
        hscroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        vscroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hscroll.set, yscrollcommand=vscroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")
        hscroll.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.canvas.bind("<Button-1>", self._on_click)
        self.root.bind("<Configure>", lambda _e: self._redraw())
        self._refresh_step_ui()

    # ---- 步驟切換 ----
    @property
    def _step(self) -> dict:
        return _STEPS[self._idx]

    def _refresh_step_ui(self) -> None:
        step = self._step
        done = sum(1 for s in _STEPS if s["key"] in self._data and "pt" in self._data[s["key"]])
        marks = " ".join(
            ("[v]" if (s["key"] in self._data and "pt" in self._data[s["key"]]) else "[ ]")
            + s["title"].split(" ")[0]
            for s in _STEPS
        )
        self.progress.config(text=f"進度 {done}/{len(_STEPS)}　{marks}")
        self.title_lbl.config(text=step["title"] + ("（可選）" if step["optional"] else ""))
        self.hint.config(text=step["instr"])
        cur = self._data.get(step["key"], {})
        self._rgb = cur.get("rgb")
        if self._rgb is not None:
            self._img_h, self._img_w = self._rgb.shape[0], self._rgb.shape[1]
        pt = cur.get("pt")
        if pt:
            self.status.config(text=f"本步已標：{step['key']} = {pt}")
        elif self._rgb is not None:
            self.status.config(text="已擷取畫面 — 請在按鈕正中央點一下")
        else:
            self.status.config(text="請按『擷取星城視窗』或『載入截圖…』")
        self._redraw()

    def _prev(self) -> None:
        if self._idx > 0:
            self._idx -= 1
            self._refresh_step_ui()

    def _next(self) -> None:
        if self._idx < len(_STEPS) - 1:
            self._idx += 1
            self._refresh_step_ui()

    # ---- 影像 ----
    def _store_rgb(self, rgb: np.ndarray) -> None:
        self._data.setdefault(self._step["key"], {})["rgb"] = rgb
        self._refresh_step_ui()

    def _capture(self) -> None:
        win = find_game_window(self.cfg.window.title_substring)
        if not win:
            messagebox.showerror("錯誤", "找不到星城視窗，請先開啟遊戲。")
            return
        try:
            focus_window(win.hwnd)
        except Exception:
            pass
        time.sleep(0.2)
        frame = capture_client(win)
        if frame.shape[1] != 1280 or frame.shape[0] != 720:
            messagebox.showwarning(
                "解析度提醒",
                f"目前擷取為 {frame.shape[1]}×{frame.shape[0]}，建議把遊戲調成 1280×720 再標，座標才一致。",
            )
        self._store_rgb(frame)

    def _load_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("圖片", "*.png *.jpg *.jpeg"), ("All", "*.*")])
        if not path:
            return
        rgb = np.array(Image.open(path).convert("RGB"))
        self._store_rgb(rgb)

    def _fit_scale(self) -> float:
        if self._rgb is None:
            return 1.0
        cw = max(400, self.canvas.winfo_width())
        ch = max(300, self.canvas.winfo_height())
        return min(1.0, cw / self._img_w, ch / self._img_h)

    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self._rgb is None:
            return
        self._scale = self._fit_scale()
        dw, dh = max(1, int(self._img_w * self._scale)), max(1, int(self._img_h * self._scale))
        img = Image.fromarray(self._rgb).resize((dw, dh), Image.Resampling.BILINEAR)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.config(scrollregion=(0, 0, dw, dh))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
        pt = self._data.get(self._step["key"], {}).get("pt")
        if pt:
            px, py = int(pt[0] * self._scale), int(pt[1] * self._scale)
            hw, hh = self._step["half"]
            rx0, ry0 = int((pt[0] - hw) * self._scale), int((pt[1] - hh) * self._scale)
            rx1, ry1 = int((pt[0] + hw) * self._scale), int((pt[1] + hh) * self._scale)
            self.canvas.create_rectangle(rx0, ry0, rx1, ry1, outline="#ffcc00", width=2)
            self.canvas.create_oval(px - 7, py - 7, px + 7, py + 7, outline="#00cc44", width=2)
            self.canvas.create_line(px - 13, py, px + 13, py, fill="#00cc44", width=2)
            self.canvas.create_line(px, py - 13, px, py + 13, fill="#00cc44", width=2)

    def _on_click(self, event) -> None:
        if self._rgb is None:
            messagebox.showinfo("提示", "請先擷取或載入截圖。")
            return
        x = int(self.canvas.canvasx(event.x) / self._scale)
        y = int(self.canvas.canvasy(event.y) / self._scale)
        self._data.setdefault(self._step["key"], {})["pt"] = (x, y)
        self.status.config(text=f"本步已標：{self._step['key']} = ({x}, {y})（黃框=模板裁切範圍）")
        self._redraw()
        self._refresh_progress_only()

    def _refresh_progress_only(self) -> None:
        done = sum(1 for s in _STEPS if s["key"] in self._data and "pt" in self._data[s["key"]])
        marks = " ".join(
            ("[v]" if (s["key"] in self._data and "pt" in self._data[s["key"]]) else "[ ]")
            + s["title"].split(" ")[0]
            for s in _STEPS
        )
        self.progress.config(text=f"進度 {done}/{len(_STEPS)}　{marks}")

    def _clear_point(self) -> None:
        cur = self._data.get(self._step["key"])
        if cur and "pt" in cur:
            del cur["pt"]
        self._refresh_step_ui()

    # ---- 套用 ----
    def _crop(self, rgb: np.ndarray, pt: tuple[int, int], half: tuple[int, int]) -> np.ndarray:
        x, y = pt
        hw, hh = half
        h, w = rgb.shape[:2]
        x0, y0 = max(0, x - hw), max(0, y - hh)
        x1, y1 = min(w, x + hw), min(h, y + hh)
        return rgb[y0:y1, x0:x1]

    def _apply(self) -> None:
        ready = [s for s in _STEPS if s["key"] in self._data and "pt" in self._data[s["key"]]]
        if not ready:
            messagebox.showwarning("沒有資料", "還沒有標記任何步驟。")
            return
        missing_required = [
            s["title"] for s in _STEPS if not s["optional"] and s not in ready
        ]
        if missing_required:
            if not messagebox.askyesno(
                "還有必填步驟未標",
                "以下步驟尚未標記：\n" + "\n".join(missing_required) + "\n\n仍要套用已標好的部分嗎？",
            ):
                return

        _TEMPLATES.mkdir(parents=True, exist_ok=True)
        ref_w = ref_h = None
        applied: list[str] = []
        points_json: dict[str, list[int]] = {}
        for step in ready:
            key = step["key"]
            rgb = self._data[key]["rgb"]
            x, y = self._data[key]["pt"]
            ref_w, ref_h = rgb.shape[1], rgb.shape[0]

            patch = self._crop(rgb, (x, y), step["half"])
            Image.fromarray(patch).save(_TEMPLATES / step["template"])

            Image.fromarray(rgb).save(self.out_dir / f"{key}.png")
            marked = Image.fromarray(rgb.copy())
            d = ImageDraw.Draw(marked)
            hw, hh = step["half"]
            d.rectangle([x - hw, y - hh, x + hw, y + hh], outline="#ffcc00", width=2)
            d.ellipse([x - 7, y - 7, x + 7, y + 7], outline="#00cc44", width=2)
            d.text((x + 12, y - 8), f"{key} ({x},{y})", fill="#00cc44")
            marked.save(self.out_dir / f"{key}_marked.png")

            self.cfg.click_points[key] = [x, y]
            points_json[key] = [x, y]
            applied.append(f"{step['title'].split(' ')[0]} {key}=({x},{y}) → {step['template']}")

        if ref_w and ref_h:
            self.cfg.window.reference_width = ref_w
            self.cfg.window.reference_height = ref_h
        save_config(self.cfg, DEFAULT_CONFIG_PATH)
        (self.out_dir / "points.json").write_text(
            json.dumps(points_json, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        messagebox.showinfo(
            "已套用",
            "已標記並套用：\n\n"
            + "\n".join(applied)
            + f"\n\n模板：vision/templates/\n設定：{DEFAULT_CONFIG_PATH}\n圖檔：{self.out_dir}\n\n"
            "請通知我重新打包，模板才會進到交付的 exe。",
        )

    def run(self) -> int:
        self.root.mainloop()
        return 0


def main() -> int:
    return MarkReturnFlowApp().run()


if __name__ == "__main__":
    raise SystemExit(main())
