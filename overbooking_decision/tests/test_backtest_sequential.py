"""
ทดสอบ backtest_sequential.py โดยเน้น carryover logic (จุดที่แก้จาก backtest.py เดิม)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest_sequential import backtest_policies_sequential
from cost_config import load_config


def make_two_night_df():
    """
    คืนที่ 1: จอง 5 รายการ, พัก 3 คืน (length_of_stay=3), ไม่มีใครยกเลิก (is_canceled=0)
    คืนที่ 2: จอง 0 รายการใหม่ (ทุกคนที่อยู่มาจากคืนก่อน)
    คืนที่ 3: จอง 0 รายการใหม่ เหมือนกัน
    ใช้ capacity เยอะพอ (จาก config เริ่มต้น) แต่ประเด็นคือเช็กว่า carryover_occupancy โผล่ถูกคืน
    """
    rows = []
    for i in range(5):
        rows.append({
            "hotel": "City Hotel",
            "stay_date": pd.Timestamp("2023-06-01"),
            "adr": 100.0,
            "is_canceled": 0,
            "p_cancel": 0.1,
            "lead_time": 30 - i,
            "length_of_stay": 3,
        })
    return pd.DataFrame(rows)


def test_carryover_occupancy_appears_on_following_nights():
    df = make_two_night_df()
    cfg = load_config()
    bt = backtest_policies_sequential(df, cfg, policies=["no_overbook"])

    bt = bt.set_index("stay_date")
    # คืนแรก: ยังไม่มี carryover (เป็นคืนแรกของ active list)
    assert bt.loc[pd.Timestamp("2023-06-01"), "carryover_occupancy"] == 0
    # คืนที่ 2 และ 3: ต้องเห็น carryover = 5 (จากการจอง 3 คืนที่ยังไม่เช็กเอาท์)
    assert bt.loc[pd.Timestamp("2023-06-02"), "carryover_occupancy"] == 5
    assert bt.loc[pd.Timestamp("2023-06-03"), "carryover_occupancy"] == 5


def test_carryover_disappears_after_checkout():
    df = make_two_night_df()
    # เพิ่มคืนที่ 4 ที่ไม่มีการจองใหม่ เพื่อดูว่า carryover หายไปหลังเช็กเอาท์ (เช็กเอาท์เช้า 06-04)
    extra_row = df.iloc[[0]].copy()
    extra_row["stay_date"] = pd.Timestamp("2023-06-04")
    extra_row["length_of_stay"] = 1
    extra_row["lead_time"] = 1
    df2 = pd.concat([df, extra_row], ignore_index=True)

    cfg = load_config()
    bt = backtest_policies_sequential(df2, cfg, policies=["no_overbook"]).set_index("stay_date")
    assert bt.loc[pd.Timestamp("2023-06-04"), "carryover_occupancy"] == 0


def test_effective_capacity_reduced_by_carryover():
    df = make_two_night_df()
    cfg = load_config()  # City Hotel capacity = 300 ใน config.yaml เริ่มต้น
    bt = backtest_policies_sequential(df, cfg, policies=["no_overbook"]).set_index("stay_date")
    capacity = cfg.capacity("City Hotel")
    assert bt.loc[pd.Timestamp("2023-06-02"), "effective_capacity"] == capacity - 5


def test_arrivals_actual_includes_carryover_even_with_no_new_bookings():
    df = make_two_night_df()
    cfg = load_config()
    bt = backtest_policies_sequential(df, cfg, policies=["no_overbook"]).set_index("stay_date")
    # คืนที่ 2 ไม่มีการจองใหม่เลย แต่ arrivals_actual ต้อง = 5 (จาก carryover ที่ไม่ยกเลิก)
    assert bt.loc[pd.Timestamp("2023-06-02"), "arrivals_actual"] == 5


def test_canceled_carryover_booking_does_not_block_capacity():
    df = make_two_night_df()
    df.loc[0, "is_canceled"] = 1  # 1 ใน 5 รายการยกเลิก
    cfg = load_config()
    bt = backtest_policies_sequential(df, cfg, policies=["no_overbook"]).set_index("stay_date")
    assert bt.loc[pd.Timestamp("2023-06-02"), "carryover_occupancy"] == 4


def test_output_schema_compatible_with_aggregate_results():
    from backtest import aggregate_results

    df = make_two_night_df()
    cfg = load_config()
    bt = backtest_policies_sequential(df, cfg, policies=["no_overbook", "model"])
    summary = aggregate_results(bt)
    assert set(summary["policy"].unique()) == {"no_overbook", "model"}
    assert "moved_per_1000_nights" in summary.columns


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
