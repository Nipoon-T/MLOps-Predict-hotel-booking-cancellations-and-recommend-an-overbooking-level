import os
import sys
import pandas as pd

# เพิ่ม project root เข้า Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


DATA_PATH = "data/raw/hotel_bookings.csv"


def check_missing_values(df):
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100

    result = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_percent
    })

    # แสดงเฉพาะคอลัมน์ที่มี Missing
    result = result[result["missing_count"] > 0]

    # เรียงจาก Missing มากไปน้อย
    result = result.sort_values(
        by="missing_count",
        ascending=False
    )

    return result


def main():
    print("=" * 60)
    print("MISSING VALUE CHECK")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    result = check_missing_values(df)

    if result.empty:
        print("\n✅ No missing values found.")

    else:
        print("\nMissing values found:")
        print(result.to_string())


if __name__ == "__main__":
    main()