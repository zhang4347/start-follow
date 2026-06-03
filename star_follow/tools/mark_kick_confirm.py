"""標記「超過五局未押注」提示窗的『確定』按鈕。

用法:
  python -m star_follow.tools.mark_kick_confirm

步驟:
  1) 在遊戲裡觸發提示窗（或載入你剛截的圖）
  2) 在綠色「確定」正中央點一下
  3) 按「存檔」→ 寫入 config.yaml 的 kick_idle_confirm，並更新模板
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from star_follow.capture.screen import capture_client
from star_follow.capture.window import find_game_window, focus_window
from star_follow.config import DEFAULT_CONFIG_PATH, load_config, save_config
from pathlib import Path

from star_follow.paths import logs_dir

_TEMPLATES = Path(__file__).resolve().parents[1] / "vision" / "templates"

_KEY = "kick_idle_confirm"
_LABEL = "五局未押注·確定"
_PATCH_HW = (70, 28)


class MarkKickConfirmApp:
    def __init__(self) -> None:
        self.cfg = load_config()
        self._rgb: np.ndarray | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._scale = 1.0
        self._img_w = self.cfg.window.reference_width or 1280
        self._img_h = self.cfg.window.reference_height or 720
        self._pt: tuple[int, int] | None = None
        self._src_name = ""
        self.out_dir = logs_dir() / "kick_popup_mark"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.root = tk.Tk()
        self.root.title("標記：五局未押注 → 確定")
        self.root.geometry("1100x820")

        bar = ttk.Frame(self.root)
        bar.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(bar, text="擷取星城視窗", command=self._capture).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="載入截圖…", command=self._load_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="存檔", command=self._save).pack(side=tk.LEFT, padx=2)
        existing = self.cfg.click_points.get(_KEY)
        ttk.Label(bar, text=f"目前 config: {_KEY}={existing or '（未設定）'}").pack(side=tk.LEFT, padx=8)

        self.hint = ttk.Label(
            self.root,
            text="請在「超過五局未押注，已退出遊戲」提示窗出現時擷取，然後點綠色『確定』按鈕正中央。",
            wraplength=1050,
            justify=tk.LEFT,
        )
        self.hint.pack(fill=tk.X, padx=8, pady=2)
        self.status = ttk.Label(self.root, text="", wraplength=1050, foreground="#0066cc")
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

    def _set_image(self, rgb: np.ndarray, src: str) -> None:
        self._rgb = rgb
        self._img_h, self._img_w = rgb.shape[0], rgb.shape[1]
        self._src_name = src
        self._pt = None
        self.status.config(text=f"已載入 {src}（{self._img_w}×{self._img_h}）— 請點『確定』中心")
        self._redraw()

    def _capture(self) -> None:
        win = find_game_window(self.cfg.window.title_substring)
        if not win:
            messagebox.showerror("錯誤", "找不到星城視窗。")
            return
        try:
            focus_window(win.hwnd)
        except Exception:
            pass
        frame = capture_client(win)
        path = self.out_dir / "kick_popup.png"
        Image.fromarray(frame).save(path)
        self._set_image(frame, path.name)

    def _load_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("圖片", "*.png *.jpg *.jpeg"), ("All", "*.*")])
        if not path:
            return
        rgb = np.array(Image.open(path).convert("RGB"))
        self._set_image(rgb, path)

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
        dw, dh = max(1, int(self._img_w * self._scale)), max(1, int(self._img_h * self._scale))
        img = Image.fromarray(self._rgb).resize((dw, dh), Image.Resampling.BILINEAR)
        if self._pt:
            d = ImageDraw.Draw(img)
            px, py = int(self._pt[0] * self._scale), int(self._pt[1] * self._scale)
            d.ellipse([px - 8, py - 8, px + 8, py + 8], outline="#00cc44", width=2)
            d.line([px - 14, py, px + 14, py], fill="#00cc44", width=2)
            d.line([px, py - 14, px, py + 14], fill="#00cc44", width=2)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, dw, dh))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

    def _on_click(self, event) -> None:
        if self._rgb is None:
            messagebox.showinfo("提示", "請先擷取或載入截圖。")
            return
        x = int(self.canvas.canvasx(event.x) / self._scale)
        y = int(self.canvas.canvasy(event.y) / self._scale)
        self._pt = (x, y)
        self.status.config(text=f"已選 ({x}, {y}) — 按『存檔』寫入 config")
        self._redraw()

    def _save(self) -> None:
        if self._rgb is None or not self._pt:
            messagebox.showwarning("未完成", "請先載入圖片並點選『確定』中心。")
            return
        x, y = self._pt
        self.cfg.click_points[_KEY] = [x, y]
        self.cfg.window.reference_width = self._img_w
        self.cfg.window.reference_height = self._img_h
        save_config(self.cfg, DEFAULT_CONFIG_PATH)

        hw, hh = _PATCH_HW
        patch = self._rgb[max(0, y - hh) : y + hh, max(0, x - hw) : x + hw]
        tpl_dir = _TEMPLATES
        tpl_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(patch).save(tpl_dir / "lobby_confirm_button.png")

        marked = Image.fromarray(self._rgb.copy())
        d = ImageDraw.Draw(marked)
        d.ellipse([x - 8, y - 8, x + 8, y + 8], outline="#00cc44", width=2)
        d.text((x + 12, y - 8), f"{_LABEL} ({x},{y})", fill="#00cc44")
        marked.save(self.out_dir / "kick_popup_marked.png")

        messagebox.showinfo(
            "已存檔",
            f"click_points.{_KEY} = [{x}, {y}]\n"
            f"已寫入 {DEFAULT_CONFIG_PATH}\n"
            f"模板：vision/templates/lobby_confirm_button.png\n\n"
            "重新啟動跟注程式即可生效。",
        )

    def run(self) -> int:
        self.root.mainloop()
        return 0


def main() -> int:
    return MarkKickConfirmApp().run()


if __name__ == "__main__":
    raise SystemExit(main())
