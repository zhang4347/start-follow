"""標記「大廳 → 進百家樂桌」流程所需的座標（給自動回桌用）。

用法（不需要管理員）：
  python -m star_follow.tools.mark_lobby

操作：
  1) 把遊戲切到要標記的畫面（例如大廳），按「擷取視窗」抓一張。
  2) 在圖上「點」你要記的位置，會跳出輸入框讓你幫它取個名字
     （例如：百家樂入口 / 進桌確定 / 返回大廳…，中英文都可）。
  3) 換下一個畫面（例如點百家樂後跳出的畫面）→ 再「擷取視窗」→ 繼續點。
  4) 都標完按「存檔」。

存檔位置（會把截圖和座標都存起來，方便對照）：
  logs/lobby_mark/scene_*.png         每次擷取的原始畫面
  logs/lobby_mark/scene_*_marked.png  疊上標記點的畫面
  logs/lobby_mark/points.json         所有標記點（client 座標）

座標是遊戲 client 座標（1280×720 基準），之後直接寫進程式的回桌流程。
"""

from __future__ import annotations

import json
import time
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from star_follow.capture.screen import capture_client
from star_follow.capture.window import find_game_window, focus_window
from star_follow.config import load_config
from star_follow.paths import logs_dir

_COLORS = ["#00cc44", "#ffcc00", "#00cccc", "#ff66ff", "#ff6600", "#66ccff", "#ff4444"]


class MarkLobbyApp:
    def __init__(self) -> None:
        self.cfg = load_config()
        self._rgb: np.ndarray | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._scale = 1.0
        self._img_w = self.cfg.window.reference_width
        self._img_h = self.cfg.window.reference_height
        self._scene = 0
        self._scene_file = ""
        # 每個點：{scene, file, label, x, y}
        self.points: list[dict] = []
        self.out_dir = logs_dir() / "lobby_mark"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.root = tk.Tk()
        self.root.title("標記大廳→進桌座標")
        self.root.geometry("1180x860")

        bar = ttk.Frame(self.root)
        bar.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(bar, text="擷取視窗", command=self._capture).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="清除本張的點", command=self._clear_scene).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="存檔", command=self._save).pack(side=tk.LEFT, padx=2)
        self.lbl_count = ttk.Label(bar, text="尚未擷取")
        self.lbl_count.pack(side=tk.LEFT, padx=10)

        self.hint = ttk.Label(
            self.root,
            text="① 按『擷取視窗』抓目前畫面　② 在圖上點要記的位置並命名　③ 換畫面再擷取繼續　④ 按『存檔』",
            wraplength=1130,
            justify=tk.LEFT,
        )
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

    # ---- 影像 ----
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
        self._rgb = frame
        self._img_h, self._img_w = frame.shape[0], frame.shape[1]
        self._scene += 1
        ts = time.strftime("%H%M%S")
        self._scene_file = f"scene_{self._scene}_{ts}.png"
        Image.fromarray(frame).save(str(self.out_dir / self._scene_file))
        self.status.config(text=f"已擷取第 {self._scene} 張：{self._scene_file}（{self._img_w}×{self._img_h}）")
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
        dw, dh = max(1, int(self._img_w * self._scale)), max(1, int(self._img_h * self._scale))
        img = Image.fromarray(self._rgb).resize((dw, dh), Image.Resampling.BILINEAR)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0, 0, dw, dh))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
        for i, p in enumerate(self.points):
            if p["scene"] != self._scene:
                continue
            px, py = int(p["x"] * self._scale), int(p["y"] * self._scale)
            c = _COLORS[i % len(_COLORS)]
            self.canvas.create_oval(px - 8, py - 8, px + 8, py + 8, outline=c, width=2)
            self.canvas.create_line(px - 12, py, px + 12, py, fill=c)
            self.canvas.create_line(px, py - 12, px, py + 12, fill=c)
            self.canvas.create_text(px + 12, py, text=f'{p["label"]} ({p["x"]},{p["y"]})',
                                    anchor=tk.W, fill=c)
        self.lbl_count.config(text=f"第 {self._scene} 張；已標 {len(self.points)} 點")

    # ---- 互動 ----
    def _on_click(self, event) -> None:
        if self._rgb is None:
            messagebox.showinfo("提示", "請先按『擷取視窗』。")
            return
        x = int(self.canvas.canvasx(event.x) / self._scale)
        y = int(self.canvas.canvasy(event.y) / self._scale)
        label = simpledialog.askstring("命名這個點", f"位置 ({x}, {y})\n幫它取個名字（例如：百家樂入口）：", parent=self.root)
        if not label:
            return
        self.points.append({"scene": self._scene, "file": self._scene_file, "label": label, "x": x, "y": y})
        self.status.config(text=f"已記 {label} = ({x}, {y})")
        self._redraw()

    def _clear_scene(self) -> None:
        self.points = [p for p in self.points if p["scene"] != self._scene]
        self._redraw()

    def _save(self) -> None:
        if not self.points:
            messagebox.showwarning("沒有資料", "還沒有標記任何點。")
            return
        (self.out_dir / "points.json").write_text(
            json.dumps(self.points, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # 各場景輸出疊點圖
        by_scene: dict[str, list[dict]] = {}
        for p in self.points:
            by_scene.setdefault(p["file"], []).append(p)
        for fname, pts in by_scene.items():
            src = self.out_dir / fname
            if not src.is_file():
                continue
            im = Image.open(src).convert("RGB")
            d = ImageDraw.Draw(im)
            for i, p in enumerate(pts):
                c = _COLORS[i % len(_COLORS)]
                x, y = p["x"], p["y"]
                d.ellipse([x - 8, y - 8, x + 8, y + 8], outline=c, width=2)
                d.line([x - 12, y, x + 12, y], fill=c, width=2)
                d.line([x, y - 12, x, y + 12], fill=c, width=2)
                d.text((x + 12, y - 6), f'{p["label"]} ({x},{y})', fill=c)
            im.save(str(self.out_dir / fname.replace(".png", "_marked.png")))
        messagebox.showinfo(
            "已存檔",
            f"座標與截圖已存到：\n{self.out_dir}\n\n共 {len(self.points)} 點。\n"
            "請把這個資料夾（或裡面的圖和 points.json）給我。",
        )

    def run(self) -> int:
        self.root.mainloop()
        return 0


def main() -> int:
    return MarkLobbyApp().run()


if __name__ == "__main__":
    raise SystemExit(main())
