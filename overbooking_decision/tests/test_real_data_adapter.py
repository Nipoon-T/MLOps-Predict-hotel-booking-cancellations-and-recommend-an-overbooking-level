"""
ทดสอบเฉพาะส่วนที่ไม่ต้องพึ่ง mlflow/features จริงของทีม (add_stay_date, estimate_capacity)
ส่วน build_backtest_dataset() ต้องให้ทีมรันทดสอบเองในเครื่องที่มี repo + mlflow.db จริง
เพราะต้อง import src.modeling.features และโหลดโมเดลจาก MLflow registry ซึ่งไม่มีในเครื่องนี้
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from real_data_adapter import (
    add_stay_date,
    compute_length_of_stay,
    compute_nightly_occupancy,
    estimate_capacity,
)


def make_raw_df(n=200, seed=0, with_stay_cols=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "hotel": rng.choice(["City Hotel", "Resort Hotel"], n),
        "arrival_date_year": rng.choice([2016, 2017], n),
        "arrival_date_month": rng.choice(["July", "August", "December"], n),
        "arrival_date_day_of_month": rng.integers(1, 28, n),
        "adr": rng.uniform(50, 200, n),
        "is_canceled": rng.binomial(1, 0.3, n),
        "lead_time": rng.integers(0, 300, n),
    })
    if with_stay_cols:
        df["stays_in_weekend_nights"] = rng.integers(0, 2, n)
        df["stays_in_week_nights"] = rng.integers(0, 4, n)
    return df


def test_add_stay_date_creates_valid_datetime_column():
    df = make_raw_df()
    out = add_stay_date(df)
    assert "stay_date" in out.columns
    assert pd.api.types.is_datetime64_any_dtype(out["stay_date"])


def test_add_stay_date_matches_source_year_and_month():
    df = make_raw_df(50)
    out = add_stay_date(df)
    assert (out["stay_date"].dt.year == out["arrival_date_year"]).all()
    month_names = out["stay_date"].dt.month_name()
    assert (month_names == out["arrival_date_month"]).all()


def test_add_stay_date_does_not_mutate_input():
    df = make_raw_df(10)
    original_cols = list(df.columns)
    add_stay_date(df)
    assert list(df.columns) == original_cols  # ต้นฉบับไม่ถูกแก้ (ใช้ .copy() ข้างใน)


def test_estimate_capacity_returns_positive_int_per_hotel(tmp_path):
    df = make_raw_df(1000, seed=1)
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)
    capacity = estimate_capacity(str(csv_path), quantile=0.95)
    assert set(capacity.keys()) <= {"City Hotel", "Resort Hotel"}
    assert all(isinstance(v, (int, np.integer)) and v > 0 for v in capacity.values())


def test_estimate_capacity_higher_quantile_gives_higher_or_equal_capacity(tmp_path):
    df = make_raw_df(1000, seed=2, with_stay_cols=True)
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)
    cap_p50 = estimate_capacity(str(csv_path), quantile=0.50)
    cap_p95 = estimate_capacity(str(csv_path), quantile=0.95)
    for hotel in cap_p50:
        assert cap_p95[hotel] >= cap_p50[hotel]


def test_compute_length_of_stay_sums_weekend_and_week_nights():
    df = pd.DataFrame({"stays_in_weekend_nights": [2, 0, 1], "stays_in_week_nights": [3, 0, 4]})
    los = compute_length_of_stay(df)
    assert los.tolist() == [5, 1, 5]  # แถวกลาง (0+0) ถูก clip ขึ้นเป็น 1 คืนขั้นต่ำ


def test_compute_length_of_stay_missing_columns_defaults_to_one():
    df = pd.DataFrame({"adr": [100, 120, 90]})  # ไม่มีคอลัมน์ stays_in_*
    los = compute_length_of_stay(df)
    assert los.tolist() == [1, 1, 1]


def test_compute_nightly_occupancy_counts_multi_night_stay_correctly():
    # 1 การจอง เช็กอิน 2023-06-01 พัก 3 คืน (01,02,03) เช็กเอาท์เช้า 06-04
    df = pd.DataFrame({
        "hotel": ["City Hotel"],
        "arrival_date_year": [2023],
        "arrival_date_month": ["June"],
        "arrival_date_day_of_month": [1],
        "stays_in_weekend_nights": [1],
        "stays_in_week_nights": [2],
        "is_canceled": [0],
    })
    occ = compute_nightly_occupancy(df)
    occ = occ.set_index("date")["occupancy"]
    assert occ[pd.Timestamp("2023-06-01")] == 1
    assert occ[pd.Timestamp("2023-06-02")] == 1
    assert occ[pd.Timestamp("2023-06-03")] == 1
    assert pd.Timestamp("2023-06-04") not in occ.index  # เช็กเอาท์แล้ว ไม่นับเป็นคืนพัก


def test_compute_nightly_occupancy_excludes_canceled_bookings():
    df = pd.DataFrame({
        "hotel": ["City Hotel", "City Hotel"],
        "arrival_date_year": [2023, 2023],
        "arrival_date_month": ["June", "June"],
        "arrival_date_day_of_month": [1, 1],
        "stays_in_weekend_nights": [0, 0],
        "stays_in_week_nights": [2, 2],
        "is_canceled": [0, 1],  # แถวที่สองถูกยกเลิก ไม่ควรถูกนับ
    })
    occ = compute_nightly_occupancy(df).set_index("date")["occupancy"]
    assert occ[pd.Timestamp("2023-06-01")] == 1  # นับแค่แถวที่ไม่ยกเลิก
