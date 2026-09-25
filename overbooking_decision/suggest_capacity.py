"""
รันครั้งเดียวเพื่อดูค่า capacity ที่แนะนำ (ประมาณจากข้อมูล train จริง) แล้วเอาไปใส่ config.yaml เอง
(ตั้งใจให้เป็น manual step เพราะ capacity เป็นสมมติฐานที่ต้องประกาศในรายงาน ไม่ใช่ auto-overwrite)

python suggest_capacity.py --train-csv data/processed/train.csv
"""
from __future__ import annotations

import argparse

from real_data_adapter import estimate_capacity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/processed/train.csv")
    parser.add_argument("--quantile", type=float, default=0.95)
    args = parser.parse_args()

    capacity = estimate_capacity(args.train_csv, quantile=args.quantile)
    print(f"Capacity ที่แนะนำ (p{int(args.quantile*100)} ของ occupancy จริงต่อคืน "
          f"รวมแขกที่พักต่อเนื่องหลายคืนด้วย, จากข้อมูล train):\n")
    for hotel, cap in capacity.items():
        print(f"  {hotel}: {cap}")
    print("\nเอาค่านี้ไปใส่ใน config.yaml ใต้ key 'capacity:' แล้วอธิบายที่มาในรายงาน "
          "(ว่าใช้ percentile เท่าไหร่ จากข้อมูลช่วงไหน)")


if __name__ == "__main__":
    main()
