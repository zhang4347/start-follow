"""入口: python -m star_follow.tools.calibrate | vision_cli"""

import sys

MSG = """
星城自動跟注 — 可用指令:

  python -m star_follow.tools.calibrate [--image 截圖.png]
  python -m star_follow.tools.vision_cli [--image 截圖.png] [--once]
  python -m star_follow.tools.run [--dry-run]   # 主引擎（預設 dry-run）
"""


def main() -> int:
    print(MSG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
