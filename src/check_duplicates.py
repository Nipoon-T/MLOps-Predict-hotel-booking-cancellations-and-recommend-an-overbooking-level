import os
import sys
import pandas as pd

# เพิ่ม project root เข้า Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


DATA_PATH = "data/raw/hotel_bookings.csv"


def check_duplicates(df):

    # แถวที่อยู่ในกลุ่ม duplicate
    duplicate_mask = df.duplicated(keep=False)

    # จำนวนแถวทั้งหมดที่เป็นสมาชิกของ duplicate groups
    duplicate_rows = df[duplicate_mask]

    # จำนวนแถวซ้ำที่เกิดขึ้นหลังจากเก็บแถวแรกของแต่ละกลุ่ม
    duplicate_occurrences = df.duplicated(keep="first").sum()

    # จำนวนกลุ่ม duplicate จริง ๆ
    duplicate_groups = (
        df[df.duplicated(keep=False)]
        .groupby(list(df.columns), dropna=False)
        .size()
    )

    unique_duplicate_groups = len(duplicate_groups)

    # จำนวนแถวหลังลบ exact duplicates
    unique_rows = len(df.drop_duplicates())

    print(f"Total rows                  : {len(df):,}")
    print(f"Rows in duplicate groups   : {duplicate_rows.shape[0]:,}")
    print(f"Duplicate occurrences      : {duplicate_occurrences:,}")
    print(f"Unique duplicate groups    : {unique_duplicate_groups:,}")
    print(f"Unique rows                 : {unique_rows:,}")

    return duplicate_rows


def main():
    print("=" * 60)
    print("DUPLICATE CHECK")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    duplicate_rows = check_duplicates(df)

    if duplicate_rows.empty:
        print("\n✅ No duplicate rows found.")

    else:
        print("\n⚠️ Exact duplicate rows found.")

        print("\nFirst 10 duplicate rows:")
        print(
            duplicate_rows
            .head(10)
            .to_string(index=True)
        )


if __name__ == "__main__":
    main()