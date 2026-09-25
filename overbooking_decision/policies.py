"""
4 นโยบาย overbooking ตามสโคปโปรเจกต์:
  1. ไม่ overbook
  2. คงที่ (เช่น 5% ของ capacity)
  3. ตามอัตราเฉลี่ยของโรงแรม/เดือน
  4. ตามโมเดล (Monte Carlo จาก simulate.find_optimal_overbook)

ทุกฟังก์ชันคืนค่า o (int) = จำนวนห้องที่แนะนำให้รับจองเกิน สำหรับคืนหนึ่งๆ
"""
from __future__ import annotations

from simulate import find_optimal_overbook


def policy_no_overbook(capacity: int) -> int:
    return 0


def policy_fixed(capacity: int, fixed_pct: float = 0.05) -> int:
    return int(round(capacity * fixed_pct))


def policy_avg_rate(capacity: int, avg_cancel_rate: float) -> int:
    """
    ใช้อัตรายกเลิกเฉลี่ยของโรงแรม/เดือนนั้น แปลงเป็นจำนวนห้องที่ "คาดว่าจะว่างจากการยกเลิก"
    แล้วอนุญาตให้รับจองเกินเท่ากับจำนวนนั้น (แบบ deterministic ไม่ใช้ Monte Carlo)
    """
    avg_cancel_rate = max(0.0, min(1.0, avg_cancel_rate))
    return int(round(capacity * avg_cancel_rate))


def policy_model(
    p_cancels,
    capacity: int,
    cost_empty: float,
    cost_overbook: float,
    o_max: int | None = None,
    n_sims: int = 2000,
    seed: int = 42,
) -> dict:
    """
    นโยบายตามโมเดล: เรียก Monte Carlo optimizer เต็มรูปแบบ
    คืนค่า dict ทั้งก้อน (มี o_star, cost_curve ฯลฯ) เพื่อให้ backtest/report ใช้ debug ได้
    """
    return find_optimal_overbook(
        p_cancels=p_cancels,
        capacity=capacity,
        cost_empty=cost_empty,
        cost_overbook=cost_overbook,
        o_max=o_max,
        n_sims=n_sims,
        seed=seed,
    )


ALL_POLICY_NAMES = ["no_overbook", "fixed", "avg_rate", "model"]
