"""截圖計時探針：連續擷取遊戲視窗 N 次，印出每次毫秒數。

用法: python -m star_follow.tools.probe_capture
不需管理員，不會點任何東西，只驗證 mss 重用是否讓截圖變快。
"""

from __future__ import annotations

import time

from star_follow.capture.screen import capture_client
from star_follow.capture.window import find_game_window


def main() -> int:
    win = find_game_window("星城Online")
    if win is None:
        print("找不到遊戲視窗（請先開啟星城 Online）")
        return 1
    print(f"視窗 client={win.client_width}x{win.client_height} @({win.client_left},{win.client_top})")

    times: list[float] = []
    for i in range(10):
        t0 = time.perf_counter()
        capture_client(win)
        ms = (time.perf_counter() - t0) * 1000
        times.append(ms)
        tag = "（首次：含 mss 初始化）" if i == 0 else ""
        print(f"第 {i + 1:>2} 次截圖 {ms:7.1f} ms {tag}")

    warm = times[1:]
    print("-" * 40)
    print(f"首次  : {times[0]:7.1f} ms")
    print(f"之後平均: {sum(warm) / len(warm):7.1f} ms（最小 {min(warm):.1f} / 最大 {max(warm):.1f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
