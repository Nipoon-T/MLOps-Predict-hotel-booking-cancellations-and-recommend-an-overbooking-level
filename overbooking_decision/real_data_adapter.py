"""
เชื่อมกับ pipeline จริงของทีม (repo: MLOps-Predict-hotel-booking-cancellations-and-recommend-an-overbooking-level)
ใช้ output จากบทบาทที่ 1 (data/processed/*.csv หลัง split_data.py)
และโมเดลจริงจากบทบาทที่ 2 (models:/hotel-cancellation-classifier@candidate + features.build_features())

v2 (2026-09-26): แก้ปัญหา capacity/occupancy นับเฉพาะ arrival_date — ตอนนี้นับ occupancy จริงที่รวม
การพักต่อเนื่องหลายคืน (stays_in_weekend_nights + stays_in_week_nights) ด้วย ผ่าน compute_length_of_stay()
และ compute_nightly_occupancy() ซึ่ง estimate_capacity() และ backtest_sequential.py ใช้ต่อ

หมายเหตุสำคัญ: ไฟล์นี้เขียนตามอินเทอร์เฟซที่ทีมอัปเดตมา (ชื่อฟังก์ชัน build_features, model URI)
ส่วนที่ไม่ต้องพึ่ง mlflow/features จริง (add_stay_date, estimate_capacity, compute_length_of_stay,
compute_nightly_occupancy) มี unit test ยืนยันแล้วใน tests/test_real_data_adapter.py
ส่วน build_backtest_dataset() ต้องให้ทีมรันทดสอบเองในเครื่องที่มี repo จริง
"""
from __future__ import annotations

import calendar
import importlib
import sys

import numpy as np
import pandas as pd

MONTH_NUM = {name: i for i, name in enumerate(calendar.month_name) if name}


def add_stay_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    hotel_bookings.csv ไม่มีคอลัมน์วันที่ตรงๆ ต้องประกอบจาก
    arrival_date_year, arrival_date_month (ชื่อเดือนภาษาอังกฤษเต็ม เช่น 'July'), arrival_date_day_of_month
    """
    df = df.copy()
    month_num = df["arrival_date_month"].map(MONTH_NUM)
    df["stay_date"] = pd.to_datetime({
        "year": df["arrival_date_year"],
        "month": month_num,
        "day": df["arrival_date_day_of_month"],
    })
    return df


def compute_length_of_stay(df: pd.DataFrame) -> pd.Series:
    """
    จำนวนคืนที่พักจริง = stays_in_weekend_nights + stays_in_week_nights
    clip ขั้นต่ำที่ 1 คืน (บางแถวเป็น 0 ซึ่งน่าจะเป็นข้อมูลผิดปกติ แต่ยังต้องกันห้อง 1 คืนอย่างน้อย)
    """
    weekend = df["stays_in_weekend_nights"] if "stays_in_weekend_nights" in df.columns else pd.Series(0, index=df.index)
    week = df["stays_in_week_nights"] if "stays_in_week_nights" in df.columns else pd.Series(0, index=df.index)
    los = (weekend.fillna(0) + week.fillna(0)).astype(int)
    return los.clip(lower=1)


def compute_nightly_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    """
    คำนวณ occupancy จริงต่อคืนต่อโรงแรม โดยนับรวมแขกที่เช็กอินมาก่อนแล้วยังพักอยู่ต่อ
    (ไม่ใช่แค่แขกที่ arrival_date ตรงกับคืนนั้น) ใช้เฉพาะการจองที่ไม่ถูกยกเลิก (is_canceled == 0)
    เพราะต้องการ "จำนวนคนที่พักจริง" ไม่ใช่จำนวนที่จองไว้

    ใช้เทคนิค sweep-line (diff array แล้ว cumsum) ต่อโรงแรม เร็วกว่า loop ทีละแถว
    คืนค่า DataFrame คอลัมน์ hotel, date, occupancy (occupancy = จำนวนห้องที่มีคนพักคืนนั้น)
    """
    df = add_stay_date(df)
    df = df.copy()
    df["length_of_stay"] = compute_length_of_stay(df)
    df["checkout_date"] = df["stay_date"] + pd.to_timedelta(df["length_of_stay"], unit="D")

    if "is_canceled" in df.columns:
        df = df[df["is_canceled"] == 0]

    records = []
    for hotel, g in df.groupby("hotel"):
        if g.empty:
            continue
        start = g["stay_date"].min()
        end = g["checkout_date"].max()
        all_days = pd.date_range(start, end, freq="D")
        diff = pd.Series(0, index=all_days, dtype=int)

        arr_counts = g.groupby("stay_date").size()
        dep_counts = g.groupby("checkout_date").size()
        diff.loc[arr_counts.index] += arr_counts.values
        diff.loc[dep_counts.index] -= dep_counts.values

        occ = diff.cumsum()
        nights = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")  # checkout day ไม่นับเป็นคืนพัก
        occ = occ.loc[nights]
        records.append(pd.DataFrame({"hotel": hotel, "date": occ.index, "occupancy": occ.values}))

    if not records:
        return pd.DataFrame(columns=["hotel", "date", "occupancy"])
    return pd.concat(records, ignore_index=True)


def load_calibrated_model(model_uri: str = "models:/hotel-cancellation-classifier@candidate"):
    """
    รองรับ 2 กรณี:
      - โหลดเป็น sklearn object ตรงๆ (มี .predict_proba) -> ปกติของ RandomForest/CalibratedClassifierCV
      - โหลดเป็น mlflow.pyfunc (มีแค่ .predict ซึ่งควรคืนความน่าจะเป็นตรงๆ ตามที่ทีมบอกไว้)
    """
    import mlflow

    try:
        import mlflow.sklearn
        model = mlflow.sklearn.load_model(model_uri)
        return model, "sklearn"
    except Exception:
        model = mlflow.pyfunc.load_model(model_uri)
        return model, "pyfunc"


def predict_p_cancel(model, kind: str, X) -> np.ndarray:
    if kind == "sklearn":
        return model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    return np.asarray(preds).reshape(-1)


def build_backtest_dataset(
    raw_csv_path: str,
    model_uri: str = "models:/hotel-cancellation-classifier@candidate",
    features_module_path: str = "src.modeling.features",
) -> pd.DataFrame:
    """
    ประกอบ DataFrame พร้อมใช้กับ backtest_policies_sequential() จากไฟล์ CSV ดิบ
    (เช่น data/processed/test.csv หรือ production.csv) + โมเดลจริงของบทบาทที่ 2

    v2: เพิ่มคอลัมน์ length_of_stay (ใช้คำนวณ carryover ข้ามคืนใน backtest_sequential.py)

    ถ้า repo จริงไม่ได้ใช้ path src.modeling.features ให้ส่ง features_module_path ใหม่เข้ามา
    ต้องรันจาก working directory ที่ sys.path เห็น src/ ของ repo (เช่นรันจาก root ของ repo)
    """
    df_raw = pd.read_csv(raw_csv_path)
    df = add_stay_date(df_raw)
    df["length_of_stay"] = compute_length_of_stay(df_raw)

    if "src" not in sys.path and "." not in sys.path:
        sys.path.insert(0, ".")
    features_mod = importlib.import_module(features_module_path)
    X = features_mod.build_features(df_raw)  # ใช้ df_raw ดิบตามสัญญาของทีม ไม่ใช่ df ที่เติม stay_date แล้ว

    model, kind = load_calibrated_model(model_uri)
    p_cancel = predict_p_cancel(model, kind, X)

    out = pd.DataFrame({
        "hotel": df["hotel"].values,
        "stay_date": df["stay_date"].values,
        "adr": df["adr"].values,
        "is_canceled": df["is_canceled"].values,
        "p_cancel": p_cancel,
        "lead_time": df["lead_time"].values,
        "length_of_stay": df["length_of_stay"].values,
    })
    # เรียงตามลำดับ FCFS: lead_time มาก (จองล่วงหน้านาน) = จองก่อน = ควรได้สิทธิ์รับก่อน
    # ถ้าทีมมีคอลัมน์วันที่จองจริง (booking_date) ให้เปลี่ยนมาเรียงตามนั้นแทน จะแม่นกว่า
    out = out.sort_values(
        ["hotel", "stay_date", "lead_time"], ascending=[True, True, False]
    ).reset_index(drop=True)
    return out


def estimate_capacity(train_csv_path: str, quantile: float = 0.95) -> dict:
    """
    ประมาณ capacity ต่อโรงแรมจากข้อมูล train จริง (รันก่อนแก้ config.yaml)
    v2: ใช้ occupancy จริงต่อคืน (รวมแขกที่พักต่อเนื่องหลายคืน) แทนการนับแค่ arrival_date
    ใช้ percentile สูงของ occupancy นั้น เพราะ capacity ควรสะท้อนจำนวนห้องที่มีคนพักพร้อมกันจริง
    """
    df_raw = pd.read_csv(train_csv_path)
    occ = compute_nightly_occupancy(df_raw)
    capacity = occ.groupby("hotel")["occupancy"].quantile(quantile).round().astype(int)
    return capacity.to_dict()
