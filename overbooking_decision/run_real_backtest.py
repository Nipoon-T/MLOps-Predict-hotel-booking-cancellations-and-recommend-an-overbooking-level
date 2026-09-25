"""
รัน backtest บนข้อมูลจริงหลัง split_data.py + โมเดลจริงจากบทบาทที่ 2 (MLflow)
v2: ใช้ backtest_policies_sequential (carryover-aware — คิดการพักต่อเนื่องหลายคืนด้วย)

รันจาก root ของ repo หลัก (เพื่อให้ import src.modeling.features เจอ):
  python overbooking_decision/run_real_backtest.py \
      --test-csv data/processed/test.csv \
      --model-uri "models:/hotel-cancellation-classifier@candidate"

ถ้าอยากเทียบกับ backtest แบบเก่า (ไม่คิด carryover) เพื่อโชว์ว่าการแก้ไขมีผลแค่ไหน ใช้ --naive
"""
from __future__ import annotations

import argparse

from backtest import aggregate_results, backtest_policies
from backtest_sequential import backtest_policies_sequential
from cost_config import load_config
from real_data_adapter import build_backtest_dataset
from sensitivity import plot_sensitivity, run_sensitivity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", default="data/processed/test.csv")
    parser.add_argument("--model-uri", default="models:/hotel-cancellation-classifier@candidate")
    parser.add_argument("--features-module", default="src.modeling.features")
    parser.add_argument("--out-prefix", default="real")
    parser.add_argument("--naive", action="store_true",
                         help="ใช้ backtest แบบเก่า (ไม่คิด carryover ข้ามคืน) แทน sequential — สำหรับเทียบผล")
    args = parser.parse_args()

    cfg = load_config()
    df = build_backtest_dataset(
        args.test_csv, model_uri=args.model_uri, features_module_path=args.features_module
    )
    print(f"โหลดข้อมูล test จริง: {len(df):,} การจอง, {df['stay_date'].nunique()} คืน, "
          f"{df['hotel'].nunique()} โรงแรม\n")

    month = df["stay_date"].dt.to_period("M")
    avg_lookup = df.groupby(["hotel", month])["is_canceled"].mean().to_dict()
    avg_lookup["__default__"] = df["is_canceled"].mean()

    backtest_fn = backtest_policies if args.naive else backtest_policies_sequential
    mode_label = "naive (ไม่คิด carryover)" if args.naive else "sequential (คิด carryover ข้ามคืน)"
    print(f"โหมด backtest: {mode_label}\n")

    bt = backtest_fn(df, cfg, avg_cancel_rate_lookup=avg_lookup)
    summary = aggregate_results(bt)
    print("=== Backtest บนข้อมูลจริง (k default) ===")
    print(summary.to_string(index=False))
    summary.to_csv(f"{args.out_prefix}_backtest_summary.csv", index=False)

    sens = run_sensitivity(df, cfg, avg_cancel_rate_lookup=avg_lookup, backtest_fn=backtest_fn)
    sens.to_csv(f"{args.out_prefix}_sensitivity.csv", index=False)
    plot_path = plot_sensitivity(sens, save_path=f"{args.out_prefix}_sensitivity_plot.png")

    print(f"\nบันทึก {args.out_prefix}_backtest_summary.csv, {args.out_prefix}_sensitivity.csv, "
          f"{plot_path} แล้ว — เอาไฟล์เหล่านี้ไปใส่รายงาน/สไลด์ได้เลย")


if __name__ == "__main__":
    main()
