"""
Backtest แบบ sequential (carryover-aware) — แก้ข้อจำกัดของ backtest.py เดิม
(backtest.py เดิม มองแต่ละคืนอิสระ ไม่รู้ว่าแขกที่เช็กอินมาก่อนแล้วพักต่อยังกันห้องอยู่ในคืนถัดไป)

วิธีคิด: เดินตามคืนเรียงเวลา (ascending) ทีละโรงแรม ทีละนโยบาย โดยจำ "รายการที่ถูกรับไปแล้วแต่ยังไม่เช็กเอาท์"
ไว้เป็น state (active bookings) แล้วหักจำนวนห้องที่ถูกกันไว้ (carryover) ออกจาก capacity ก่อนตัดสินใจรับจองใหม่
ของคืนนั้น — ทำให้ Monte Carlo ของนโยบาย model เห็น "ห้องที่เหลือจริง" ไม่ใช่ capacity เต็มทุกคืน

สมมติฐานที่ต้องประกาศในรายงาน:
  - carryover คำนวณจาก is_canceled จริง (ground truth) ไม่ใช่การพยากรณ์ — คือเรารู้ผลจริงของการจองที่รับไปแล้ว
    ล่วงหน้า (เหมาะกับ backtest ที่ใช้ label จริงอยู่แล้ว) แต่ในระบบจริงตอนตัดสินใจจะไม่รู้แน่ชัดขนาดนี้
  - ที่ขอบเขตของแต่ละไฟล์ (ต้นของ test.csv/production.csv) carryover จากการจองที่เริ่มพักก่อนหน้าไฟล์นี้
    (อยู่ใน train/validation) จะไม่ถูกนับ เพราะไม่มีข้อมูลนั้นอยู่ในไฟล์ที่ส่งเข้ามา — ทำให้คืนแรกๆ ของแต่ละ
    ชุดข้อมูล underestimate carryover เล็กน้อย ควรเขียนข้อจำกัดนี้ในรายงานด้วย
"""
from __future__ import annotations

import pandas as pd

from policies import policy_avg_rate, policy_fixed, policy_no_overbook
from simulate import find_optimal_overbook


def _o_for_policy(
    policy_name: str,
    night_df: pd.DataFrame,
    effective_capacity: int,
    base_capacity: int,
    cost_empty: float,
    cost_overbook: float,
    avg_cancel_rate_lookup: dict,
    fixed_pct: float,
    n_sims: int,
    seed: int,
    hotel,
    night,
) -> int:
    """
    fixed/avg_rate ใช้ base_capacity (ขนาดโรงแรมเต็ม) ในการคำนวณ % ตามนิยามเดิมของนโยบาย (ไม่ปรับตาม carryover)
    no_overbook/model ใช้ effective_capacity (หลังหัก carryover) เพราะเป็นตัวเลขที่ต้องใช้ตัดสินใจ "รับจองใหม่ได้อีกกี่ห้อง"
    """
    if policy_name == "no_overbook":
        return policy_no_overbook(effective_capacity)
    if policy_name == "fixed":
        return policy_fixed(base_capacity, fixed_pct)
    if policy_name == "avg_rate":
        key = (hotel, pd.Timestamp(night).to_period("M"))
        rate = avg_cancel_rate_lookup.get(key, avg_cancel_rate_lookup.get("__default__", 0.05))
        return policy_avg_rate(base_capacity, rate)
    if policy_name == "model":
        if len(night_df) == 0 or effective_capacity <= 0:
            return 0
        result = find_optimal_overbook(
            p_cancels=night_df["p_cancel"].values,
            capacity=effective_capacity,
            cost_empty=cost_empty,
            cost_overbook=cost_overbook,
            n_sims=n_sims,
            seed=seed,
        )
        return result["o_star"]
    raise ValueError(f"unknown policy: {policy_name}")


def backtest_policies_sequential(
    df: pd.DataFrame,
    cost_config,
    k: float | None = None,
    avg_cancel_rate_lookup: dict | None = None,
    policies: list[str] | None = None,
) -> pd.DataFrame:
    """
    df ต้องมีคอลัมน์: hotel, stay_date, adr, is_canceled, p_cancel, lead_time, length_of_stay
    (build_backtest_dataset() ของ real_data_adapter.py สร้างให้ครบแล้ว)

    คืนค่า DataFrame หนึ่งแถวต่อ (hotel, stay_date, policy) — schema เดียวกับ backtest.py เดิม
    บวกคอลัมน์เสริม effective_capacity, carryover_occupancy เพื่อ debug/ตรวจสอบ
    ใช้กับ aggregate_results() จาก backtest.py ได้เลย (schema เข้ากันได้)
    """
    policies = policies or ["no_overbook", "fixed", "avg_rate", "model"]
    avg_cancel_rate_lookup = avg_cancel_rate_lookup or {"__default__": 0.05}
    k = cost_config.default_k if k is None else k

    all_rows = []

    for hotel, hotel_df in df.groupby("hotel"):
        capacity = cost_config.capacity(hotel)
        # ต้องเดินให้ครบทุกคืนที่ "มีคนพักอยู่" ไม่ใช่แค่คืนที่มีการจองใหม่ — ไม่งั้นคืนที่ถูก
        # carryover คลุมอยู่อย่างเดียว (ไม่มีการจองใหม่เลย) จะถูกข้ามไป ทำให้ carryover ไม่ถูกนับต่อ
        checkout_dates = hotel_df["stay_date"] + pd.to_timedelta(hotel_df["length_of_stay"], unit="D")
        last_night = checkout_dates.max() - pd.Timedelta(days=1)
        nights = pd.date_range(hotel_df["stay_date"].min(), last_night, freq="D")
        candidates_by_night = {
            n: g.sort_values("lead_time", ascending=False).reset_index(drop=True)
            for n, g in hotel_df.groupby("stay_date")
        }
        hotel_avg_adr = float(hotel_df["adr"].mean())

        for pol in policies:
            active = []  # แต่ละ item: {"checkout_date": Timestamp, "is_canceled": 0/1, "adr": float}

            for night in nights:
                night = pd.Timestamp(night)
                # เอารายการที่เช็กเอาท์ไปแล้วออกจาก active list
                active = [b for b in active if b["checkout_date"] > night]

                carryover_occ = sum(1 for b in active if b["is_canceled"] == 0)
                carryover_revenue = sum(b["adr"] for b in active if b["is_canceled"] == 0)
                effective_capacity = max(0, capacity - carryover_occ)

                night_df = candidates_by_night.get(night, hotel_df.iloc[0:0])
                adr_ref = float(night_df["adr"].mean()) if len(night_df) else hotel_avg_adr
                cost_empty = cost_config.cost_empty(adr_ref)
                cost_overbook = cost_config.cost_overbook(adr_ref, k=k)

                o = _o_for_policy(
                    pol, night_df, effective_capacity, capacity, cost_empty, cost_overbook,
                    avg_cancel_rate_lookup, cost_config.fixed_overbook_pct,
                    cost_config.n_sims, cost_config.seed, hotel, night,
                )

                accepted_n = min(len(night_df), max(0, effective_capacity + o))
                accepted = night_df.iloc[:accepted_n]
                showed_new = accepted[accepted["is_canceled"] == 0]

                arrivals_actual = carryover_occ + len(showed_new)
                empty_rooms = max(0, capacity - arrivals_actual)
                moved_customers = max(0, arrivals_actual - capacity)
                revenue = carryover_revenue + float(showed_new["adr"].sum())
                cost_total = empty_rooms * cost_empty + moved_customers * cost_overbook
                profit = revenue - cost_total

                all_rows.append({
                    "hotel": hotel, "stay_date": night, "policy": pol, "k": k,
                    "o": o,
                    "effective_capacity": effective_capacity,
                    "carryover_occupancy": carryover_occ,
                    "accepted_n": accepted_n,
                    "arrivals_actual": arrivals_actual,
                    "empty_rooms": empty_rooms,
                    "moved_customers": moved_customers,
                    "revenue": revenue,
                    "cost": cost_total,
                    "profit": profit,
                })

                # เติม active list ด้วยรายการที่เพิ่งรับใหม่ของคืนนี้
                for _, row in accepted.iterrows():
                    checkout = night + pd.Timedelta(days=int(row["length_of_stay"]))
                    active.append({
                        "checkout_date": checkout,
                        "is_canceled": row["is_canceled"],
                        "adr": row["adr"],
                    })

    return pd.DataFrame(all_rows)
