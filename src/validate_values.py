import os
import sys
import pandas as pd

# เพิ่ม project root เข้า Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from schemas.hotel_booking_schema import (
    BINARY_COLUMNS,
    NON_NEGATIVE_COLUMNS,
    VALID_MONTHS,
    VALID_YEARS,
)


DATA_PATH = "data/raw/hotel_bookings.csv"


def validate_values(df):
    errors = []

    # ==========================================
    # 1. ตรวจ Binary Columns
    # ==========================================
    for column in BINARY_COLUMNS:
        invalid_values = sorted(
            df[column].dropna().unique().tolist()
        )

        invalid_values = [
            value
            for value in invalid_values
            if value not in [0, 1]
        ]

        if invalid_values:
            errors.append({
                "type": "invalid_binary_values",
                "column": column,
                "values": invalid_values,
            })

    # ==========================================
    # 2. ตรวจค่าที่ห้ามติดลบ
    # ==========================================
    for column in NON_NEGATIVE_COLUMNS:
        invalid_rows = df[df[column] < 0]

        if len(invalid_rows) > 0:
            errors.append({
                "type": "negative_values",
                "column": column,
                "count": len(invalid_rows),
                "row_indices": invalid_rows.index.tolist(),
            })

    # ==========================================
    # 3. ตรวจเดือน
    # ==========================================
    invalid_months = sorted(
        set(df["arrival_date_month"].dropna())
        - set(VALID_MONTHS)
    )

    if invalid_months:
        errors.append({
            "type": "invalid_month",
            "values": invalid_months,
        })

    # ==========================================
    # 4. ตรวจปี
    # ==========================================
    invalid_years = sorted(
        set(df["arrival_date_year"].dropna())
        - set(VALID_YEARS)
    )

    if invalid_years:
        errors.append({
            "type": "invalid_year",
            "values": invalid_years,
        })

    return errors


def main():
    print("=" * 60)
    print("VALUE VALIDATION")
    print("=" * 60)

    # โหลด Dataset
    df = pd.read_csv(DATA_PATH)

    # ตรวจสอบค่า
    errors = validate_values(df)

    # ==========================================
    # แสดงผล Validation
    # ==========================================
    if errors:
        print("\n❌ Value validation FAILED")

        for error in errors:
            print("\nError:")
            print(error)

    else:
        print("\n✅ Value validation PASSED")
        print("All checked values are valid.")


if __name__ == "__main__":
    main()