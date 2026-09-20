import os
import sys
import pandas as pd

# เพิ่ม project root เข้า Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from schemas.hotel_booking_schema import EXPECTED_COLUMNS


DATA_PATH = "data/raw/hotel_bookings.csv"


def validate_schema(df):
    errors = []

    actual_columns = list(df.columns)

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in actual_columns
    ]

    if missing_columns:
        errors.append({
            "type": "missing_columns",
            "columns": missing_columns
        })

    unexpected_columns = [
        column
        for column in actual_columns
        if column not in EXPECTED_COLUMNS
    ]

    if unexpected_columns:
        errors.append({
            "type": "unexpected_columns",
            "columns": unexpected_columns
        })

    if len(actual_columns) != len(EXPECTED_COLUMNS):
        errors.append({
            "type": "column_count",
            "expected": len(EXPECTED_COLUMNS),
            "actual": len(actual_columns)
        })

    return errors


def main():
    print("=" * 60)
    print("SCHEMA VALIDATION")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    errors = validate_schema(df)

    if errors:
        print("\n❌ Schema validation FAILED")

        for error in errors:
            print(f"\nError type: {error['type']}")
            print(error)

    else:
        print("\n✅ Schema validation PASSED")
        print(f"Columns detected: {len(df.columns)}")


if __name__ == "__main__":
    main()