"""M1：ROI 校正工具。用法: python -m star_follow.tools.calibrate [--image 截圖.png]"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from star_follow.capture.screen import capture_client
from star_follow.capture.window import find_game_window
from star_follow.config import DEFAULT_CONFIG_PATH, load_config, save_config

RECT_KEYS = [
    "countdown",
    "menu_button",
    "menu_panel",
    "menu_stats_option",
    "stats_panel",
    "stats_table",
    "stats_close",
]
POINT_GROUPS = [
    ("click_points", ["menu_button", "menu_chart"]),
    ("chips", ["1000", "5000", "10000", "50000", "100000", "500000"]),
    (
        "bet_areas",
        ["閒", "莊", "和", "閒對", "莊對", "幸運六", "閒龍寶", "莊龍寶"],
    ),
]


class ImageCanvas(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pixmap: QPixmap | None = None
        self._scale = 1.0
        self._origin = QPoint(0, 0)
        self._drag_start: QPoint | None = None
        self._drag_end: QPoint | None = None
        self._points: list[tuple[int, int]] = []
        self.mode = "rect"
        self.on_rect = None
        self.on_point = None

    def set_image(self, rgb: np.ndarray) -> None:
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg.copy())
        self.setMinimumSize(min(w, 1200), min(h, 800))
        self._update_scaled()
        self.update()

    def _update_scaled(self) -> None:
        if not self._pixmap:
            return
        self._pixmap_scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._scale = self._pixmap_scaled.width() / self._pixmap.width()
        ox = (self.width() - self._pixmap_scaled.width()) // 2
        oy = (self.height() - self._pixmap_scaled.height()) // 2
        self._origin = QPoint(ox, oy)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_scaled()
        self.update()

    def _to_image(self, pos: QPoint) -> QPoint:
        x = int((pos.x() - self._origin.x()) / self._scale)
        y = int((pos.y() - self._origin.y()) / self._scale)
        return QPoint(max(0, x), max(0, y))

    def mousePressEvent(self, event) -> None:
        if not self._pixmap:
            return
        if self.mode == "point" and event.button() == Qt.MouseButton.LeftButton:
            img_pt = self._to_image(event.position().toPoint())
            if self.on_point:
                self.on_point(img_pt.x(), img_pt.y())
            return
        if self.mode == "rect" and event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = self._to_image(event.position().toPoint())
            self._drag_end = self._drag_start
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start is not None:
            self._drag_end = self._to_image(event.position().toPoint())
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_start is None or self._drag_end is None:
            return
        x0, y0 = self._drag_start.x(), self._drag_start.y()
        x1, y1 = self._drag_end.x(), self._drag_end.y()
        x, y = min(x0, x1), min(y0, y1)
        w, h = abs(x1 - x0), abs(y1 - y0)
        self._drag_start = None
        self._drag_end = None
        if w > 4 and h > 4 and self.on_rect:
            self.on_rect(x, y, w, h)
        self.update()

    def set_overlay_rects(self, rects: dict[str, list[int]]) -> None:
        self._overlay_rects = rects
        self.update()

    def set_overlay_points(self, points: dict[str, list[int]]) -> None:
        self._overlay_points = points
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self._pixmap:
            return
        painter = QPainter(self)
        painter.drawPixmap(self._origin, self._pixmap_scaled)
        pen = QPen(Qt.GlobalColor.green, 2)
        painter.setPen(pen)
        for key, rect in getattr(self, "_overlay_rects", {}).items():
            x, y, w, h = rect
            r = QRect(
                self._origin.x() + int(x * self._scale),
                self._origin.y() + int(y * self._scale),
                int(w * self._scale),
                int(h * self._scale),
            )
            painter.drawRect(r)
            painter.drawText(r.topLeft(), key)
        pen.setColor(Qt.GlobalColor.cyan)
        painter.setPen(pen)
        for key, pt in getattr(self, "_overlay_points", {}).items():
            px = self._origin.x() + int(pt[0] * self._scale)
            py = self._origin.y() + int(pt[1] * self._scale)
            painter.drawEllipse(px - 4, py - 4, 8, 8)
            painter.drawText(px + 6, py, key)
        if self._drag_start and self._drag_end:
            pen.setColor(Qt.GlobalColor.yellow)
            painter.setPen(pen)
            x0, y0 = self._drag_start.x(), self._drag_start.y()
            x1, y1 = self._drag_end.x(), self._drag_end.y()
            r = QRect(
                self._origin.x() + int(min(x0, x1) * self._scale),
                self._origin.y() + int(min(y0, y1) * self._scale),
                int(abs(x1 - x0) * self._scale),
                int(abs(y1 - y0) * self._scale),
            )
            painter.drawRect(r)


class CalibrateWindow(QMainWindow):
    def __init__(self, image_path: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("星城跟注 — M1 ROI 校正")
        self.cfg = load_config()
        self._img_h = self.cfg.window.reference_height
        self._img_w = self.cfg.window.reference_width

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        row = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["矩形 ROI", "點座標（選單/籌碼/注區）"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode)
        row.addWidget(QLabel("模式"))
        row.addWidget(self.mode_combo)

        self.target_combo = QComboBox()
        row.addWidget(QLabel("目標"))
        row.addWidget(self.target_combo)

        btn_cap = QPushButton("擷取星城視窗")
        btn_cap.clicked.connect(self._capture)
        row.addWidget(btn_cap)

        btn_load = QPushButton("載入截圖…")
        btn_load.clicked.connect(self._load_file)
        row.addWidget(btn_load)

        btn_save = QPushButton("儲存 config.yaml")
        btn_save.clicked.connect(self._save)
        row.addWidget(btn_save)
        layout.addLayout(row)

        self.canvas = ImageCanvas()
        layout.addWidget(self.canvas, stretch=1)

        self.status = QLabel("拖曳框選矩形，或點一下標記座標")
        layout.addWidget(self.status)

        self._rebuild_targets()
        self.canvas.on_rect = self._apply_rect
        self.canvas.on_point = self._apply_point
        self._refresh_overlay()

        if image_path:
            self._load_path(Path(image_path))

    def _rebuild_targets(self) -> None:
        self.target_combo.clear()
        if self.mode_combo.currentIndex() == 0:
            self.target_combo.addItems(RECT_KEYS)
        else:
            for group, keys in POINT_GROUPS:
                for k in keys:
                    self.target_combo.addItem(f"{group}:{k}")

    def _on_mode(self) -> None:
        self.canvas.mode = "rect" if self.mode_combo.currentIndex() == 0 else "point"
        self._rebuild_targets()

    def _apply_rect(self, x: int, y: int, w: int, h: int) -> None:
        key = self.target_combo.currentText()
        self.cfg.roi[key] = [x, y, w, h]
        self._img_w = max(self._img_w, x + w)
        self._img_h = max(self._img_h, y + h)
        self.status.setText(f"{key} = [{x}, {y}, {w}, {h}]")
        self._refresh_overlay()

    def _apply_point(self, x: int, y: int) -> None:
        text = self.target_combo.currentText()
        group, key = text.split(":", 1)
        if group == "click_points":
            self.cfg.click_points[key] = [x, y]
        elif group == "chips":
            self.cfg.chips[key] = [x, y]
        else:
            self.cfg.bet_areas[key] = [x, y]
        self.status.setText(f"{text} = [{x}, {y}]")
        self._refresh_overlay()

    def _refresh_overlay(self) -> None:
        self.canvas.set_overlay_rects(dict(self.cfg.roi))
        pts = {f"click_{k}": v for k, v in self.cfg.click_points.items()}
        pts.update({f"chip_{k}": v for k, v in self.cfg.chips.items()})
        pts.update({f"bet_{k}": v for k, v in self.cfg.bet_areas.items()})
        self.canvas.set_overlay_points(pts)

    def _capture(self) -> None:
        win = find_game_window(self.cfg.window.title_substring)
        if not win:
            QMessageBox.warning(self, "錯誤", "找不到星城視窗，請先開啟遊戲。")
            return
        frame = capture_client(win)
        self._img_h, self._img_w = frame.shape[0], frame.shape[1]
        self.cfg.window.reference_width = self._img_w
        self.cfg.window.reference_height = self._img_h
        self.canvas.set_image(frame)
        self.status.setText(f"已擷取 {win.title} ({self._img_w}x{self._img_h})")
        self._refresh_overlay()

    def _load_file(self) -> None:
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(self, "選擇截圖", "", "Images (*.png *.jpg)")
        if path:
            self._load_path(Path(path))

    def _load_path(self, path: Path) -> None:
        img = np.array(Image.open(path).convert("RGB"))
        self._img_h, self._img_w = img.shape[0], img.shape[1]
        self.cfg.window.reference_width = self._img_w
        self.cfg.window.reference_height = self._img_h
        self.canvas.set_image(img)
        self.status.setText(f"已載入 {path.name} ({self._img_w}x{self._img_h})")
        self._refresh_overlay()

    def _save(self) -> None:
        self.cfg.window.reference_width = self._img_w
        self.cfg.window.reference_height = self._img_h
        path = save_config(self.cfg, DEFAULT_CONFIG_PATH)
        QMessageBox.information(self, "已儲存", str(path))


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    args, qt_args = parser.parse_known_args()
    app = QApplication([sys.argv[0]] + qt_args)
    win = CalibrateWindow(image_path=args.image)
    win.resize(1100, 750)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
