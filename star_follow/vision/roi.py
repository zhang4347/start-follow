from __future__ import annotations


def _scale(v: float, ref: int, actual: int) -> int:
    if ref <= 0:
        return int(v)
    return int(round(v * actual / ref))


def scale_rect(
    rect: list[int],
    ref_w: int,
    ref_h: int,
    actual_w: int,
    actual_h: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = rect
    return (
        _scale(x, ref_w, actual_w),
        _scale(y, ref_h, actual_h),
        max(1, _scale(w, ref_w, actual_w)),
        max(1, _scale(h, ref_h, actual_h)),
    )


def scale_point(
    point: list[int],
    ref_w: int,
    ref_h: int,
    actual_w: int,
    actual_h: int,
) -> tuple[int, int]:
    x, y = point
    return _scale(x, ref_w, actual_w), _scale(y, ref_h, actual_h)
