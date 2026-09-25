"""
ขั้นที่ 2-3 ของชั้นตัดสินใจ:
  - จำลองจำนวนผู้มาจริงต่อคืน (Monte Carlo) จากความน่าจะเป็นยกเลิกของแต่ละการจอง
  - หาจำนวนห้อง overbook (o) ที่ทำให้ต้นทุนคาดหวังต่ำสุด

สมมติฐานสำคัญ: การรับจองเกิน (overbook = o) หมายถึง "รับจองได้สูงสุด capacity + o รายการ"
สำหรับคืนหนึ่ง ถ้ามีคนมาจองทั้งหมด n_bookings รายการ:
  - ถ้า n_bookings <= capacity + o  -> รับทุกรายการ
  - ถ้า n_bookings >  capacity + o  -> รับแค่ capacity+o รายการแรก (เรียงตามลำดับที่จอง/lead_time)
    ส่วนที่เหลือถือว่าถูกปฏิเสธตั้งแต่ตอนจอง (ไม่อยู่ในต้นทุนของคืนนี้ เพราะไม่เคยถูกรับ)
ดังนั้น "o" ที่มากขึ้น = รับจองมากขึ้น = ผู้มาจริงคาดว่าจะมากขึ้น (ลดห้องว่าง แต่เพิ่มความเสี่ยงจองเกินจริง)
p_cancels ควรเรียงตามลำดับที่ต้องการให้ระบบรับจองก่อน-หลัง (เช่น เรียงตามเวลาที่จองเข้ามา)
"""
from __future__ import annotations

import numpy as np


def simulate_show_matrix(p_cancels, n_sims: int = 2000, seed: int = 42) -> np.ndarray:
    """
    จำลองว่าการจองแต่ละรายการ "มาจริง" หรือไม่ ในแต่ละรอบจำลอง
    คืนค่า boolean matrix shape (n_sims, n_bookings)
    """
    p_cancels = np.asarray(p_cancels, dtype=float)
    rng = np.random.default_rng(seed)
    p_show = 1.0 - p_cancels
    shows = rng.random((n_sims, len(p_cancels))) < p_show
    return shows


def find_optimal_overbook(
    p_cancels,
    capacity: int,
    cost_empty: float,
    cost_overbook: float,
    o_max: int | None = None,
    n_sims: int = 2000,
    seed: int = 42,
) -> dict:
    """
    หา o* ที่ทำให้ต้นทุนคาดหวังต่ำสุด โดยไล่ค่า o = 0..o_max
    ใช้ cumulative sum ของ show-matrix เพื่อไม่ต้องจำลองใหม่ทุกค่า o (เร็วกว่ามาก)

    คืนค่า dict: {"o_star": int, "expected_cost": float, "cost_curve": {o: cost}}
    """
    n_bookings = len(p_cancels)
    if n_bookings == 0:
        return {"o_star": 0, "expected_cost": 0.0, "cost_curve": {0: 0.0}, "n_sims": n_sims}

    if o_max is None:
        o_max = max(1, int(0.30 * capacity))

    shows = simulate_show_matrix(p_cancels, n_sims=n_sims, seed=seed)
    # cum_shows[:, j] = จำนวนคนมาจริงถ้ารับจองแค่ (j+1) รายการแรก
    cum_shows = np.cumsum(shows, axis=1)

    cost_curve = {}
    for o in range(0, o_max + 1):
        accepted_n = min(n_bookings, capacity + o)
        if accepted_n == 0:
            arrivals = np.zeros(shows.shape[0])
        else:
            arrivals = cum_shows[:, accepted_n - 1]
        empty_rooms = np.maximum(0, capacity - arrivals)
        over_actual = np.maximum(0, arrivals - capacity)
        cost = empty_rooms * cost_empty + over_actual * cost_overbook
        cost_curve[o] = float(cost.mean())

    o_star = min(cost_curve, key=cost_curve.get)
    return {
        "o_star": o_star,
        "expected_cost": cost_curve[o_star],
        "cost_curve": cost_curve,
        "n_sims": n_sims,
    }
