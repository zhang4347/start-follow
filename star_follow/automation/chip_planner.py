from __future__ import annotations


def plan_chips(
    amount: int,
    denominations: list[int] | None = None,
) -> list[tuple[int, int]]:
    """金額 → [(面額, 次數), ...]，貪婪用大面額、總點擊最少。"""
    if amount <= 0:
        return []
    denoms = sorted(denominations or [1000, 5000, 10000, 50000, 100000, 500000], reverse=True)
    remaining = amount
    plan: list[tuple[int, int]] = []
    for d in denoms:
        if d <= 0:
            continue
        n, remaining = divmod(remaining, d)
        if n:
            plan.append((d, n))
    if remaining > 0:
        raise ValueError(f"無法用面額湊齊: 剩餘 {remaining}")
    return plan


def count_bet_clicks(chip_plan: list[tuple[int, int]]) -> int:
    """每種面額：1 次選籌碼 + N 次點注區。"""
    return sum(1 + times for _, times in chip_plan)


def format_chip_plan(amount: int, denominations: list[int] | None = None) -> str:
    plan = plan_chips(amount, denominations)
    parts = [f"{d}×{n}" for d, n in plan]
    clicks = count_bet_clicks(plan)
    return f"{amount} → {' + '.join(parts)}（共 {clicks} 次點擊）"
