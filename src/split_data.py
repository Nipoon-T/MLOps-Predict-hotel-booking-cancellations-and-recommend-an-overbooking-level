import os
import sys
import pandas as pd

# เพิ่ม project root เข้า Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


INPUT_PATH = "data/processed/hotel_bookings_with_date.csv"

TRAIN_PATH = "data/processed/train.csv"
VALIDATION_PATH = "data/processed/validation.csv"
TEST_PATH = "data/processed/test.csv"


def temporal_split(df):

    # ==========================================
    # 1. เรียงข้อมูลตาม Booking Date
    # ==========================================
    df = df.sort_values(
        by="booking_date"
    ).reset_index(drop=True)

    # ==========================================
    # 2. คำนวณจุดแบ่ง
    # ==========================================
    n = len(df)

    train_end = int(n * 0.60)
    validation_end = int(n * 0.80)

    # ==========================================
    # 3. แบ่งข้อมูล
    # ==========================================
    train_df = df.iloc[:train_end].copy()

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[
        validation_end:
    ].copy()

    return train_df, validation_df, test_df


def print_split_info(name, df):

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print(f"Rows       : {len(df):,}")
    print(
        f"Percentage : "
        f"{len(df) / 119390 * 100:.2f}%"
    )

    print(
        f"Date range : "
        f"{df['booking_date'].min().date()} "
        f"→ "
        f"{df['booking_date'].max().date()}"
    )

    print(
        f"Cancellation rate : "
        f"{df['is_canceled'].mean():.4f}"
    )


def main():

    print("=" * 60)
    print("TEMPORAL DATA SPLITTING")
    print("=" * 60)

    # ==========================================
    # Load data
    # ==========================================
    df = pd.read_csv(
        INPUT_PATH,
        parse_dates=[
            "arrival_date",
            "booking_date"
        ]
    )

    print(f"\nTotal rows: {len(df):,}")

    # ==========================================
    # Split
    # ==========================================
    train_df, validation_df, test_df = temporal_split(df)

    # ==========================================
    # แสดงข้อมูลแต่ละชุด
    # ==========================================
    print_split_info(
        "TRAIN",
        train_df
    )

    print_split_info(
        "VALIDATION",
        validation_df
    )

    print_split_info(
        "TEST",
        test_df
    )

    # ==========================================
    # ตรวจสอบว่าไม่มีข้อมูลซ้อนกัน
    # ==========================================
    train_indices = set(train_df.index)
    validation_indices = set(validation_df.index)
    test_indices = set(test_df.index)

    print("\n" + "-" * 60)
    print("OVERLAP CHECK")
    print("-" * 60)

    print(
        f"Train ∩ Validation : "
        f"{len(train_indices & validation_indices)}"
    )

    print(
        f"Train ∩ Test       : "
        f"{len(train_indices & test_indices)}"
    )

    print(
        f"Validation ∩ Test  : "
        f"{len(validation_indices & test_indices)}"
    )

    # ==========================================
    # บันทึกข้อมูล
    # ==========================================
    train_df.to_csv(
        TRAIN_PATH,
        index=False
    )

    validation_df.to_csv(
        VALIDATION_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_PATH,
        index=False
    )

    # ==========================================
    # สรุป
    # ==========================================
    print("\n" + "=" * 60)
    print("FILES CREATED")
    print("=" * 60)

    print(TRAIN_PATH)
    print(VALIDATION_PATH)
    print(TEST_PATH)

    print("\n✅ Temporal split completed successfully.")


if __name__ == "__main__":
    main()