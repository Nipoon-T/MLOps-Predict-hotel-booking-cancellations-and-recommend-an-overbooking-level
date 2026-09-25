"""
ต้นแบบ (prototype) ใช้ข้อมูลสังเคราะห์ + baseline probability แทนโมเดลจริงจากบทบาทที่ 2
v2: ข้อมูลสังเคราะห์มีการพักหลายคืน (length_of_stay > 1) แล้ว เพื่อทดสอบ carryover logic จริง
เทียบให้เห็นด้วยว่า naive backtest (เดิม) vs sequential backtest (ใหม่) ต่างกันแค่ไหน

python demo.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import aggregate_results, backtest_policies
from backtest_sequential import backtest_policies_sequential
from cost_config import load_config
from sensitivity import plot_sensitivity, run_sensitivity


def make_synthetic_data(n_nights: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hotels = ["City Hotel", "Resort Hotel"]
    rows = []
    for stay_date in pd.date_range("2023-06-01", periods=n_nights):
        for hotel in hotels:
            # ต่อคืน — ตั้งให้สูงพอที่ occupancy สะสม (n_bookings * avg length_of_stay ~2.4) ใกล้/เกิน
            # capacity 300/400 ใน config.yaml เพื่อให้เห็น trade-off ของการ overbook จริง
            n_bookings = rng.integers(110, 150)
            lead_time = rng.exponential(30, n_bookings).clip(0, 300)
            deposit_non_refund = rng.binomial(1, 0.15, n_bookings)
            adr = rng.normal(100, 20, n_bookings).clip(30, 400)
            # ส่วนใหญ่พัก 1-3 คืน บางส่วนพักยาว 4-7 คืน (ให้เกิด carryover ชัดเจน)
            length_of_stay = rng.choice([1, 2, 3, 4, 5, 7], n_bookings, p=[0.35, 0.25, 0.2, 0.1, 0.07, 0.03])

            logit = -2.0 + 0.015 * lead_time + 1.5 * deposit_non_refund
            p_cancel_true = (1 / (1 + np.exp(-logit))).clip(0.02, 0.85)
            is_canceled = rng.binomial(1, p_cancel_true)
            p_cancel_baseline = np.clip(p_cancel_true + rng.normal(0, 0.05, n_bookings), 0.02, 0.9)

            for i in range(n_bookings):
                rows.append({
                    "hotel": hotel,
                    "stay_date": stay_date,
                    "lead_time": lead_time[i],
                    "adr": adr[i],
                    "is_canceled": is_canceled[i],
                    "p_cancel": p_cancel_baseline[i],
                    "length_of_stay": int(length_of_stay[i]),
                })
    df = pd.DataFrame(rows)
    df = df.sort_values(["hotel", "stay_date", "lead_time"], ascending=[True, True, False]).reset_index(drop=True)
    return df


def main():
    cfg = load_config()
    df = make_synthetic_data(n_nights=20, seed=0)
    print(f"ข้อมูลสังเคราะห์: {len(df):,} การจอง, {df['stay_date'].nunique()} คืนที่มีการจองใหม่, "
          f"{df['hotel'].nunique()} โรงแรม, length_of_stay เฉลี่ย {df['length_of_stay'].mean():.1f} คืน\n")

    month = df["stay_date"].dt.to_period("M")
    avg_lookup = df.groupby(["hotel", month])["is_canceled"].mean().to_dict()
    avg_lookup["__default__"] = df["is_canceled"].mean()

    print("=== (A) Backtest แบบเก่า — ไม่คิด carryover ข้ามคืน ===")
    bt_naive = backtest_policies(df, cfg, avg_cancel_rate_lookup=avg_lookup)
    print(aggregate_results(bt_naive).to_string(index=False))

    print("\n=== (B) Backtest แบบใหม่ — sequential, คิด carryover ข้ามคืน ===")
    bt_seq = backtest_policies_sequential(df, cfg, avg_cancel_rate_lookup=avg_lookup)
    summary_seq = aggregate_results(bt_seq)
    print(summary_seq.to_string(index=False))

    print("\n=== Sensitivity analysis (แบบ sequential, ทุกค่า k) ===")
    sens = run_sensitivity(df, cfg, avg_cancel_rate_lookup=avg_lookup, backtest_fn=backtest_policies_sequential)
    pivot = sens.pivot(index="policy", columns="k", values="total_profit")
    print(pivot.to_string())

    path = plot_sensitivity(sens, save_path="sensitivity_plot.png")
    print(f"\nบันทึกกราฟ sensitivity (sequential) ไว้ที่ {path}")


if __name__ == "__main__":
    main()
