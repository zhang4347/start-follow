"""標記選單點擊位置（☰ 與柱狀圖）。

用法:
  python -m star_follow.tools.mark_menu           # 完整兩步
  python -m star_follow.tools.mark_menu --chart-only   # 只重標柱狀圖（☰ 沿用 config）
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

STEPS = [
    (
        "menu_button",
        "① 點 ☰（三橫線圓鈕）的中心",
        "在「選單未展開」的畫面上，點最右上角三橫線圓鈕正中央。",
    ),
    (
        "menu_chart",
        "② 點柱狀圖圖示的中心",
        "請先在遊戲裡手動點開 ☰，再按「擷取星城視窗」，然後點第一個柱狀圖小圖示。",
    ),
]

COLORS = {"menu_button": "#00cc44", "menu_chart": "#00cccc"}


class MarkMenuApp:
    def __init__(self, *, chart_only: bool = False) -> None:
        self.cfg = load_config()
        self._chart_only = chart_only
        self._step = 1 if chart_only and self.cfg.click_points.get("menu_button") else 0
        self._img_w = self.cfg.window.reference_width
        self._img_h = self.cfg.window.reference_height
        self._rgb: np.ndarray | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._scale = 1.0

        self.root = tk.Tk()
        title = "標記柱狀圖位置" if self._step == 1 else "標記選單點擊位置"
        self.root.title(title)
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
        if chart_only and not self.cfg.click_points.get("menu_button"):
            messagebox.showinfo(
                "提示",
                "config 裡還沒有 menu_button。\n建議先跑完整標記，或先標 ☰ 再標柱狀圖。",
            )

    def _update_hint(self) -> None:
        key, title, detail = STEPS[self._step]
        existing = self.cfg.click_points.get(key)
        extra = f"\n目前 config: {key} = {existing}" if existing else ""
        self.hint.config(text=f"{title}\n{detail}{extra}")
        self.btn_next.config(text="下一步" if self._step < len(STEPS) - 1 else "完成並儲存")

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

        for key, pt in self.cfg.click_points.items():
            if key not in COLORS:
                continue
            px, py = int(pt[0] * self._scale), int(pt[1] * self._scale)
            color = COLORS[key]
            r = 10
            self.canvas.create_oval(px - r, py - r, px + r, py + r, outline=color, width=2)
            self.canvas.create_line(px - 14, py, px + 14, py, fill=color)
            self.canvas.create_line(px, py - 14, px, py + 14, fill=color)
            self.canvas.create_text(px + 16, py, text=key, anchor=tk.W, fill=color)

    def _on_click(self, event) -> None:
        if self._rgb is None or self._scale <= 0:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        x = int(cx / self._scale)
        y = int(cy / self._scale)
        key = STEPS[self._step][0]
        self.cfg.click_points[key] = [x, y]
        self.status.config(text=f"已標記 {key} = [{x}, {y}]（1280×720 參考座標，依擷取解析度自動寫入）")
        self._redraw()

    def _next_step(self) -> None:
        key = STEPS[self._step][0]
        if key not in self.cfg.click_points:
            messagebox.showwarning("尚未標記", f"請先在圖上點一下：{key}")
            return
        if self._step < len(STEPS) - 1:
            self._step += 1
            self._update_hint()
            messagebox.showinfo(
                "下一步",
                "請到遊戲裡手動點開 ☰ 選單，\n"
                "再按「擷取星城視窗」，然後點柱狀圖圖示中心。",
            )
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
        step_name = STEPS[self._step][0]
        self.status.config(
            text=f"已擷取 {win.title} ({self._img_w}×{self._img_h}) — 請點 {step_name} 位置"
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
        need = ["menu_chart"] if self._chart_only else ["menu_button", "menu_chart"]
        missing = [k for k in need if k not in self.cfg.click_points]
        if missing:
            messagebox.showwarning("未完成", f"請先標記：{', '.join(missing)}")
            return
        if self._img_w > 0 and self._img_h > 0:
            self.cfg.window.reference_width = self._img_w
            self.cfg.window.reference_height = self._img_h
        path = save_config(self.cfg, DEFAULT_CONFIG_PATH)
        pts = self.cfg.click_points
        messagebox.showinfo(
            "已儲存",
            f"{path}\n\n"
            f"menu_button: {pts.get('menu_button')}\n"
            f"menu_chart:  {pts.get('menu_chart')}\n\n"
            "測試：python -m star_follow.tools.test_menu",
        )

    def run(self) -> int:
        self.root.mainloop()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="標記 ☰ 與柱狀圖點擊位置")
    parser.add_argument(
        "--chart-only",
        action="store_true",
        help="只重標柱狀圖（需 config 已有 menu_button）",
    )
    args = parser.parse_args()
    return MarkMenuApp(chart_only=args.chart_only).run()


if __name__ == "__main__":
    raise SystemExit(main())
