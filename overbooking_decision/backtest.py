"""
ขั้นที่ 4 ของชั้นตัดสินใจ: Backtest 4 นโยบายบนคืนจริงในชุด test / production simulation

รูปแบบข้อมูลที่ต้องการ (DataFrame ต่อคืน อย่างน้อยมีคอลัมน์):
  hotel        : ชื่อโรงแรม
  stay_date    : วันที่เข้าพัก (ใช้ group by)
  booking_order: ลำดับการจอง (ใช้เรียง ใครถูกรับก่อน) - เช่น booking_date หรือ lead_time
  adr          : ราคาห้องต่อคืนของการจองนั้น
  is_canceled  : 1 = ยกเลิกจริง (label), 0 = มาจริง  -- ต้องเป็นคืนที่ "ถึงวันเข้าพักแล้ว" เท่านั้น
  p_cancel     : ความน่าจะเป็นยกเลิกที่ calibrate แล้ว จากบทบาท 2 (หรือ baseline ระหว่างรอ)

ผลลัพธ์ของแต่ละคืน x นโยบาย ถูกรวมเป็นกำไรรวม และจำนวนลูกค้าที่ถูกย้าย (ต่อ 1,000 คืน)
"""
from __future__ import annotations

import pandas as pd

from policies import (
    policy_avg_rate,
    policy_fixed,
    policy_model,
    policy_no_overbook,
)


def backtest_night(night_df: pd.DataFrame, capacity: int, cost_empty: float, cost_overbook: float, o: int) -> dict:
    """
    รันคืนหนึ่งด้วยค่า o ที่กำหนด โดยใช้ label จริง (is_canceled) ไม่ใช่การจำลอง
    """
    n_bookings = len(night_df)
    accepted_n = min(n_bookings, capacity + o)
    accepted = night_df.iloc[:accepted_n]

    showed = accepted[accepted["is_canceled"] == 0]
    arrivals_actual = len(showed)

    revenue = float(showed["adr"].sum())
    empty_rooms = max(0, capacity - arrivals_actual)
    over_actual = max(0, arrivals_actual - capacity)  # = จำนวนลูกค้าที่ต้องถูกย้ายโรงแรม

    cost_total = empty_rooms * cost_empty + over_actual * cost_overbook
    profit = revenue - cost_total

    return {
        "o": o,
        "accepted_n": accepted_n,
        "arrivals_actual": arrivals_actual,
        "empty_rooms": empty_rooms,
        "moved_customers": over_actual,
        "revenue": revenue,
        "cost": cost_total,
        "profit": profit,
    }


def compute_o_for_policy(
    policy_name: str,
    night_df: pd.DataFrame,
    capacity: int,
    cost_empty: float,
    cost_overbook: float,
    avg_cancel_rate_lookup: dict,
    fixed_pct: float,
    n_sims: int,
    seed: int,
) -> int:
    if policy_name == "no_overbook":
        return policy_no_overbook(capacity)
    if policy_name == "fixed":
        return policy_fixed(capacity, fixed_pct)
    if policy_name == "avg_rate":
        key = (night_df["hotel"].iloc[0], night_df["stay_date"].iloc[0].to_period("M"))
        rate = avg_cancel_rate_lookup.get(key, avg_cancel_rate_lookup.get("__default__", 0.05))
        return policy_avg_rate(capacity, rate)
    if policy_name == "model":
        result = policy_model(
            p_cancels=night_df["p_cancel"].values,
            capacity=capacity,
            cost_empty=cost_empty,
            cost_overbook=cost_overbook,
            n_sims=n_sims,
            seed=seed,
        )
        return result["o_star"]
    raise ValueError(f"unknown policy: {policy_name}")


def backtest_policies(
    df: pd.DataFrame,
    cost_config,
    k: float | None = None,
    avg_cancel_rate_lookup: dict | None = None,
    policies: list[str] | None = None,
) -> pd.DataFrame:
    """
    รันทั้ง 4 นโยบายบนทุกคืนใน df แล้วคืน DataFrame สรุปต่อ (hotel, stay_date, policy)
    รวมผลระดับคืน + สรุปกำไรรวมและลูกค้าที่ถูกย้ายต่อ 1,000 คืนในตัวเรียกใช้ (ดู aggregate_results)
    """
    policies = policies or ["no_overbook", "fixed", "avg_rate", "model"]
    avg_cancel_rate_lookup = avg_cancel_rate_lookup or {"__default__": 0.05}
    k = cost_config.default_k if k is None else k

    rows = []
    for (hotel, stay_date), night_df in df.groupby(["hotel", "stay_date"]):
        night_df = night_df.reset_index(drop=True)
        capacity = cost_config.capacity(hotel)
        adr_ref = float(night_df["adr"].mean())
        cost_empty = cost_config.cost_empty(adr_ref)
        cost_overbook = cost_config.cost_overbook(adr_ref, k=k)

        for pol in policies:
            o = compute_o_for_policy(
                pol, night_df, capacity, cost_empty, cost_overbook,
                avg_cancel_rate_lookup, cost_config.fixed_overbook_pct,
                cost_config.n_sims, cost_config.seed,
            )
            result = backtest_night(night_df, capacity, cost_empty, cost_overbook, o)
            result.update({"hotel": hotel, "stay_date": stay_date, "policy": pol, "k": k})
            rows.append(result)

    return pd.DataFrame(rows)


def aggregate_results(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """
    สรุปกำไรรวมและจำนวนลูกค้าที่ถูกย้าย "ต่อ 1,000 คืน" ตาม business metric ในสโคป
    """
    n_nights = backtest_df.groupby("policy")["stay_date"].transform("count")
    summary = (
        backtest_df.groupby("policy")
        .agg(total_profit=("profit", "sum"), total_moved=("moved_customers", "sum"), n_nights=("stay_date", "count"))
        .reset_index()
    )
    summary["moved_per_1000_nights"] = summary["total_moved"] / summary["n_nights"] * 1000
    return summary.sort_values("total_profit", ascending=False)
