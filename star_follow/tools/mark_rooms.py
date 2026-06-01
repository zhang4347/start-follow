"""標記換房模式所需座標。

分步引導，支援「點一下」與「框矩形」。流程：
  python -m star_follow.tools.mark_rooms

步驟：
  ① 點「快速換桌」圖示（右上三個圖示的中間那個）— 點
  ② 先在遊戲手動打開桌號清單，再「擷取星城視窗」
  ③ 點任一列「前往」鈕中心（決定前往欄位 x）— 點
  ④ 框出清單左側「桌號 No.X」數字欄（涵蓋上下多列）— 框
  ⑤ 點清單中央（換桌捲動用）— 點
  ⑥ 框出左下角目前桌號「No.X」— 框

會寫入 config.yaml：
  click_points.room_switch_button / room_list_scroll
  roi.room_no_col / room_current_table
  room.goto_x
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

# (key, kind, title, detail)   kind: "point" | "rect"
STEPS = [
    (
        "room_switch_button",
        "point",
        "① 點「快速換桌」圖示中心",
        "在『清單未開』的畫面，點右上三個圖示的中間那個（桌子+箭頭）中心。",
    ),
    (
        "room_goto_ref",
        "point",
        "② 點任一列『前往』鈕中心",
        "請先在遊戲手動點開桌號清單，按『擷取星城視窗』，再點某一列右側『前往』鈕中央（決定前往欄位 x）。",
    ),
    (
        "room_no_col",
        "rect",
        "③ 框出清單左側『桌號 No.X』數字欄",
        "用滑鼠拖一個矩形，涵蓋清單左側桌號數字（盡量含上下多列、左右只框數字區）。",
    ),
    (
        "room_list_scroll",
        "point",
        "④ 點清單中央（捲動用）",
        "點清單中間任意位置，換桌時會在這裡滾輪往下捲。",
    ),
    (
        "room_current_table",
        "rect",
        "⑤ 框出左下角目前桌號『No.X』",
        "框住畫面左下角顯示目前桌號的那塊（例如 No.1 上方/下方的數字）。",
    ),
    (
        "room_confirm_button",
        "point",
        "⑥ 點彈出視窗的『確定』鈕中心",
        "點某列『前往』後會跳出『離開目前座位並前往指定牌桌』提示。請在遊戲手動觸發該視窗、按『擷取星城視窗』，再點綠色『確定』鈕中央。",
    ),
]

POINT_COLOR = {
    "room_switch_button": "#00cc44",
    "room_goto_ref": "#ffcc00",
    "room_list_scroll": "#00cccc",
    "room_confirm_button": "#66ff66",
}
RECT_COLOR = {"room_no_col": "#ff66ff", "room_current_table": "#ff6600"}


class MarkRoomsApp:
    def __init__(self, only: list[str] | None = None) -> None:
        self.cfg = load_config()
        self.steps = [s for s in STEPS if not only or s[0] in only]
        if not self.steps:
            self.steps = list(STEPS)
        self._step = 0
        self._img_w = self.cfg.window.reference_width
        self._img_h = self.cfg.window.reference_height
        self._rgb: np.ndarray | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._scale = 1.0
        # 暫存結果
        self.points: dict[str, list[int]] = {}
        self.rects: dict[str, list[int]] = {}
        self._drag_start: tuple[int, int] | None = None

        self.root = tk.Tk()
        self.root.title("標記換房座標")
        self.root.geometry("1180x860")

        bar = ttk.Frame(self.root)
        bar.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(bar, text="擷取星城視窗", command=self._capture).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="載入截圖…", command=self._load_file).pack(side=tk.LEFT, padx=2)
        self.btn_next = ttk.Button(bar, text="下一步", command=self._next_step)
        self.btn_next.pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="儲存 config.yaml", command=self._save).pack(side=tk.LEFT, padx=2)

        self.hint = ttk.Label(self.root, text="", wraplength=1130, justify=tk.LEFT)
        self.hint.pack(fill=tk.X, padx=8, pady=2)
        self.status = ttk.Label(self.root, text="", wraplength=1130)
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

        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.root.bind("<Configure>", lambda _e: self._redraw())

        self._update_hint()

    # ---- 影像 ----
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

        for key, pt in self.points.items():
            px, py = int(pt[0] * self._scale), int(pt[1] * self._scale)
            c = POINT_COLOR.get(key, "#ffffff")
            self.canvas.create_oval(px - 9, py - 9, px + 9, py + 9, outline=c, width=2)
            self.canvas.create_line(px - 13, py, px + 13, py, fill=c)
            self.canvas.create_line(px, py - 13, px, py + 13, fill=c)
            self.canvas.create_text(px + 14, py, text=key, anchor=tk.W, fill=c)

        for key, r in self.rects.items():
            x, y, w, h = (int(v * self._scale) for v in r)
            c = RECT_COLOR.get(key, "#ffffff")
            self.canvas.create_rectangle(x, y, x + w, y + h, outline=c, width=2)
            self.canvas.create_text(x + 2, y - 8, text=key, anchor=tk.W, fill=c)

    # ---- 互動 ----
    def _to_img(self, event) -> tuple[int, int]:
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        return int(cx / self._scale), int(cy / self._scale)

    def _on_press(self, event) -> None:
        if self._rgb is None:
            return
        key, kind, *_ = self.steps[self._step]
        x, y = self._to_img(event)
        if kind == "point":
            self.points[key] = [x, y]
            self.status.config(text=f"已標記 {key} = [{x}, {y}]")
            self._redraw()
        else:
            self._drag_start = (x, y)

    def _on_drag(self, event) -> None:
        if self._rgb is None or self._drag_start is None:
            return
        key = self.steps[self._step][0]
        x0, y0 = self._drag_start
        x1, y1 = self._to_img(event)
        self.rects[key] = [min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)]
        self._redraw()

    def _on_release(self, event) -> None:
        if self._drag_start is None:
            return
        key = self.steps[self._step][0]
        r = self.rects.get(key)
        self._drag_start = None
        if r:
            self.status.config(text=f"已框 {key} = {r}")

    def _update_hint(self) -> None:
        key, kind, title, detail = self.steps[self._step]
        self.hint.config(text=f"{title}　[{'點' if kind == 'point' else '框'}]\n{detail}")
        self.btn_next.config(text="下一步" if self._step < len(self.steps) - 1 else "完成並儲存")

    def _next_step(self) -> None:
        key, kind, *_ = self.steps[self._step]
        if kind == "point" and key not in self.points:
            messagebox.showwarning("尚未標記", f"請先在圖上點一下：{key}")
            return
        if kind == "rect" and key not in self.rects:
            messagebox.showwarning("尚未標記", f"請先在圖上框一個矩形：{key}")
            return
        if self._step < len(self.steps) - 1:
            self._step += 1
            self._update_hint()
            if self.steps[self._step][0] == "room_goto_ref":
                messagebox.showinfo("下一步", "請到遊戲手動點開桌號清單，\n再按『擷取星城視窗』，然後點某列『前往』鈕中心。")
            elif self.steps[self._step][0] == "room_confirm_button":
                messagebox.showinfo(
                    "下一步",
                    "請在遊戲手動點某列『前往』，跳出確認視窗後，\n按『擷取星城視窗』，再點綠色『確定』鈕中心。",
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
        self.status.config(text=f"已擷取 {win.title} ({self._img_w}×{self._img_h})")

    def _load_file(self) -> None:
        path = filedialog.askopenfilename(
            title="選擇截圖", filetypes=[("Images", "*.png *.jpg *.jpeg"), ("All", "*.*")]
        )
        if not path:
            return
        self._set_image(np.array(Image.open(path).convert("RGB")))
        self.status.config(text=f"已載入 {path}")

    _POINT_KEYS = {"room_switch_button", "room_list_scroll", "room_confirm_button"}
    _RECT_KEYS = {"room_no_col", "room_current_table"}

    def _save(self) -> None:
        # 只驗證本次有顯示的步驟（支援 --only 只重標部分）
        missing = []
        for key, kind, *_ in self.steps:
            if kind == "point" and key not in self.points:
                missing.append(key)
            if kind == "rect" and key not in self.rects:
                missing.append(key)
        if missing:
            messagebox.showwarning("未完成", f"還沒標記：{', '.join(missing)}")
            return
        if self._img_w > 0 and self._img_h > 0:
            self.cfg.window.reference_width = self._img_w
            self.cfg.window.reference_height = self._img_h
        written: list[str] = []
        for key, val in self.points.items():
            if key in self._POINT_KEYS:
                self.cfg.click_points[key] = val
                written.append(f"{key}={val}")
            elif key == "room_goto_ref":
                self.cfg.room.goto_x = int(val[0])
                written.append(f"goto_x={self.cfg.room.goto_x}")
        for key, val in self.rects.items():
            if key in self._RECT_KEYS:
                self.cfg.roi[key] = val
                written.append(f"{key}={val}")
        path = save_config(self.cfg, DEFAULT_CONFIG_PATH)
        messagebox.showinfo("已儲存", f"{path}\n\n" + "\n".join(written))

    def run(self) -> int:
        self.root.mainloop()
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="標記換房模式座標")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="KEY",
        help="只重標指定項目，例如 --only room_current_table",
    )
    args = parser.parse_args()
    return MarkRoomsApp(only=args.only).run()


if __name__ == "__main__":
    raise SystemExit(main())
