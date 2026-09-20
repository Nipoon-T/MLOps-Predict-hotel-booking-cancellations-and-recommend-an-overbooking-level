import os
import sys
import pandas as pd

# เพิ่ม project root เข้า Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from schemas.hotel_booking_schema import (
    EXPECTED_COLUMNS,
    BINARY_COLUMNS,
    NON_NEGATIVE_COLUMNS,
    VALID_MONTHS,
    VALID_YEARS,
)


DATA_PATH = "data/bad/hotel_bookings_bad.csv"


def validate_schema(df):
    errors = []

    actual_columns = list(df.columns)

    # ตรวจ column ที่หาย
    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in actual_columns
    ]

    if missing_columns:
        errors.append({
            "type": "missing_columns",
            "columns": missing_columns,
        })

    # ตรวจ column ที่เกิน
    unexpected_columns = [
        column
        for column in actual_columns
        if column not in EXPECTED_COLUMNS
    ]

    if unexpected_columns:
        errors.append({
            "type": "unexpected_columns",
            "columns": unexpected_columns,
        })

    # ตรวจจำนวน column
    if len(actual_columns) != len(EXPECTED_COLUMNS):
        errors.append({
            "type": "column_count",
            "expected": len(EXPECTED_COLUMNS),
            "actual": len(actual_columns),
        })

    return errors


def validate_values(df):
    errors = []

    # ==========================================
    # 1. Binary values
    # ==========================================
    for column in BINARY_COLUMNS:

        invalid_rows = df[
            ~df[column].isin([0, 1])
        ]

        if len(invalid_rows) > 0:
            errors.append({
                "type": "invalid_binary_values",
                "column": column,
                "count": len(invalid_rows),
                "row_indices": invalid_rows.index.tolist(),
                "values": invalid_rows[column].tolist(),
            })

    # ==========================================
    # 2. Non-negative values
    # ==========================================
    for column in NON_NEGATIVE_COLUMNS:

        invalid_rows = df[
            df[column] < 0
        ]

        if len(invalid_rows) > 0:
            errors.append({
                "type": "negative_values",
                "column": column,
                "count": len(invalid_rows),
                "row_indices": invalid_rows.index.tolist(),
                "values": invalid_rows[column].tolist(),
            })

    # ==========================================
    # 3. Month
    # ==========================================
    invalid_rows = df[
        ~df["arrival_date_month"].isin(VALID_MONTHS)
    ]

    if len(invalid_rows) > 0:
        errors.append({
            "type": "invalid_month",
            "count": len(invalid_rows),
            "row_indices": invalid_rows.index.tolist(),
            "values": invalid_rows["arrival_date_month"].tolist(),
        })

    # ==========================================
    # 4. Year
    # ==========================================
    invalid_rows = df[
        ~df["arrival_date_year"].isin(VALID_YEARS)
    ]

    if len(invalid_rows) > 0:
        errors.append({
            "type": "invalid_year",
            "count": len(invalid_rows),
            "row_indices": invalid_rows.index.tolist(),
            "values": invalid_rows["arrival_date_year"].tolist(),
        })

    return errors


def main():

    print("=" * 60)
    print("BAD DATA VALIDATION DEMONSTRATION")
    print("=" * 60)

    # โหลด Bad Dataset
    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset: {DATA_PATH}")
    print(f"Rows   : {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # ==========================================
    # Schema Validation
    # ==========================================
    schema_errors = validate_schema(df)

    print("\n" + "-" * 60)
    print("SCHEMA VALIDATION")
    print("-" * 60)

    if schema_errors:
        print("❌ Schema validation FAILED")

        for error in schema_errors:
            print(error)

    else:
        print("✅ Schema validation PASSED")

    # ==========================================
    # Value Validation
    # ==========================================
    value_errors = validate_values(df)

    print("\n" + "-" * 60)
    print("VALUE VALIDATION")
    print("-" * 60)

    if value_errors:

        print("❌ Value validation FAILED")

        for error in value_errors:
            print("\nError:")
            print(error)

    else:
        print("✅ Value validation PASSED")

    # ==========================================
    # Final Result
    # ==========================================
    print("\n" + "=" * 60)

    if schema_errors or value_errors:
        print("❌ BAD DATA DETECTED")
        print("Validation successfully detected invalid data.")

    else:
        print("✅ NO INVALID DATA DETECTED")

    print("=" * 60)


if __name__ == "__main__":
    main()