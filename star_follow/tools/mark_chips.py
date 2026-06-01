"""標記籌碼與注區點擊位置。

用法:
  python -m star_follow.tools.mark_chips              # 籌碼 + 注區逐步標記
  python -m star_follow.tools.mark_chips --chips-only # 只標 6 種籌碼
"""

from __future__ import annotations

import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from star_follow.capture.screen import capture_client
from star_follow.capture.window import find_game_window, focus_window
from star_follow.config import DEFAULT_CONFIG_PATH, load_config, save_config

CHIP_STEPS = [
    ("1000", "① 1K 籌碼", "點畫面下方 1,000 籌碼正中央。"),
    ("5000", "② 5K 籌碼", "點 5,000 籌碼正中央。"),
    ("10000", "③ 10K 籌碼", "點 10,000 籌碼正中央。"),
    ("50000", "④ 50K 籌碼", "點 50,000 籌碼正中央。"),
    ("100000", "⑤ 100K 籌碼", "點 100,000 籌碼正中央。"),
    ("500000", "⑥ 500K 籌碼", "點 500,000 籌碼正中央。"),
]

BET_STEPS = [
    ("閒", "⑦ 閒", "點「閒」注區可下注位置。"),
    ("莊", "⑧ 莊", "點「莊」注區。"),
    ("和", "⑨ 和", "點「和」注區。"),
    ("閒對", "⑩ 閒對", "點「閒對」注區。"),
    ("莊對", "⑪ 莊對", "點「莊對」注區。"),
    ("幸運六", "⑫ 幸運六", "點「幸運六」注區。"),
    ("閒龍寶", "⑬ 閒龍寶", "點「閒龍寶」注區。"),
    ("莊龍寶", "⑭ 莊龍寶", "點「莊龍寶」注區。"),
]

COLORS = {
    **{k: "#ffcc00" for k, _, _ in CHIP_STEPS},
    **{k: "#66ccff" for k, _, _ in BET_STEPS},
}


class MarkChipsApp:
    def __init__(self, *, chips_only: bool = False) -> None:
        self.cfg = load_config()
        self._steps = list(CHIP_STEPS) if chips_only else CHIP_STEPS + BET_STEPS
        self._step = 0
        self._img_w = self.cfg.window.reference_width
        self._img_h = self.cfg.window.reference_height
        self._rgb: np.ndarray | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._scale = 1.0

        self.root = tk.Tk()
        self.root.title("標記籌碼 / 注區")
        self.root.geometry("1100x820")

        bar = ttk.Frame(self.root)
        bar.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(bar, text="擷取星城視窗", command=self._capture).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="載入截圖…", command=self._load_file).pack(side=tk.LEFT, padx=2)
        self.btn_next = ttk.Button(bar, text="下一步", command=self._next_step)
        self.btn_next.pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="儲存 config.yaml", command=self._save).pack(side=tk.LEFT, padx=2)

        self.hint = ttk.Label(self.root, text="", wraplength=1050, justify=tk.LEFT)
        self.hint.pack(fill=tk.X, padx=8, pady=2)
        self.status = ttk.Label(self.root, text="", wraplength=1050)
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
        self._update_hint()
        messagebox.showinfo(
            "提示",
            "請先進入百家樂牌桌（統計表關閉），\n"
            "畫面下方要能看見 6 種籌碼，再按「擷取星城視窗」。",
        )

    def _step_key(self) -> str:
        return self._steps[self._step][0]

    def _is_chip_step(self) -> bool:
        return self._step < len(CHIP_STEPS) or self._step_key() in {k for k, _, _ in CHIP_STEPS}

    def _update_hint(self) -> None:
        key, title, detail = self._steps[self._step]
        store = "chips" if key.isdigit() else "bet_areas"
        existing = self.cfg.chips.get(key) if store == "chips" else self.cfg.bet_areas.get(key)
        extra = f"\n目前 config: {key} = {existing}" if existing else ""
        self.hint.config(text=f"{title}\n{detail}{extra}")
        last = self._step >= len(self._steps) - 1
        self.btn_next.config(text="完成並儲存" if last else "下一步")

    def _set_image(self, rgb: np.ndarray) -> None:
        self._rgb = rgb
        self._img_h, self._img_w = rgb.shape[0], rgb.shape[1]
        self.cfg.window.reference_width = self._img_w
        self.cfg.window.reference_height = self._img_h
        self._redraw()

    def _fit_scale(self) -> float:
        if self._rgb is None:
            return 1.0
        cw = max(400, self.canvas.winfo_width())
        ch = max(300, self.canvas.winfo_height())
        return min(1.0, cw / self._img_w, ch / self._img_h)

    def _overlay_points(self) -> dict[str, list[int]]:
        pts: dict[str, list[int]] = {}
        for k, v in self.cfg.chips.items():
            pts[f"chip_{k}"] = v
        for k, v in self.cfg.bet_areas.items():
            pts[f"bet_{k}"] = v
        return pts

    def _redraw(self) -> None:
        if self._rgb is None:
            return
        self._scale = self._fit_scale()
        disp_w = max(1, int(self._img_w * self._scale))
        disp_h = max(1, int(self._img_h * self._scale))
        img = Image.fromarray(self._rgb).resize((disp_w, disp_h), Image.Resampling.BILINEAR)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, disp_w, disp_h))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo, tags="bg")

        for key, pt in self._overlay_points().items():
            color = "#ffcc00" if key.startswith("chip_") else "#66ccff"
            px, py = int(pt[0] * self._scale), int(pt[1] * self._scale)
            r = 8
            self.canvas.create_oval(px - r, py - r, px + r, py + r, outline=color, width=2)
            self.canvas.create_text(px + 12, py, text=key, anchor=tk.W, fill=color)

    def _on_click(self, event) -> None:
        if self._rgb is None or self._scale <= 0:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        x = int(cx / self._scale)
        y = int(cy / self._scale)
        key = self._step_key()
        if key.isdigit():
            self.cfg.chips[key] = [x, y]
        else:
            self.cfg.bet_areas[key] = [x, y]
        self.status.config(text=f"已標記 {key} = [{x}, {y}]")
        self._redraw()

    def _next_step(self) -> None:
        key = self._step_key()
        store = self.cfg.chips if key.isdigit() else self.cfg.bet_areas
        if key not in store:
            messagebox.showwarning("尚未標記", f"請先在圖上點一下：{key}")
            return
        if self._step < len(self._steps) - 1:
            self._step += 1
            self._update_hint()
        else:
            self._save()

    def _capture(self) -> None:
        win = find_game_window(self.cfg.window.title_substring)
        if not win:
            messagebox.showerror("錯誤", "找不到星城視窗，請先開啟遊戲。")
            return
        focus_window(win.hwnd)
        frame = capture_client(win)
        self._set_image(frame)
        self.status.config(
            text=f"已擷取 {win.title} ({self._img_w}×{self._img_h}) — 請點 {self._step_key()}"
        )

    def _load_file(self) -> None:
        path = filedialog.askopenfilename(
            title="選擇截圖",
            filetypes=[("Images", "*.png *.jpg *.jpeg"), ("All", "*.*")],
        )
        if not path:
            return
        rgb = np.array(Image.open(path).convert("RGB"))
        self._set_image(rgb)
        self.status.config(text=f"已載入 {path}")

    def _save(self) -> None:
        if self._img_w > 0 and self._img_h > 0:
            self.cfg.window.reference_width = self._img_w
            self.cfg.window.reference_height = self._img_h
        path = save_config(self.cfg, DEFAULT_CONFIG_PATH)
        messagebox.showinfo(
            "已儲存",
            f"{path}\n\n"
            "測試籌碼組合：python -m star_follow.tools.test_bet --plan\n"
            "實際下注：python -m star_follow.tools.run --live",
        )

    def run(self) -> int:
        self.root.mainloop()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="標記籌碼與注區")
    parser.add_argument("--chips-only", action="store_true", help="只標 6 種籌碼")
    args = parser.parse_args()
    return MarkChipsApp(chips_only=args.chips_only).run()


if __name__ == "__main__":
    raise SystemExit(main())
