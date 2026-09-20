import os
import sys
import pandas as pd

# เพิ่ม project root เข้า Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


RAW_DATA_PATH = "data/raw/hotel_bookings.csv"
OUTPUT_PATH = "data/processed/hotel_bookings_with_date.csv"


MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}


def create_booking_date(df):

    # แปลงชื่อเดือนเป็นตัวเลข
    arrival_month = df["arrival_date_month"].map(MONTH_MAP)

    # สร้าง Arrival Date
    arrival_date = pd.to_datetime(
        dict(
            year=df["arrival_date_year"],
            month=arrival_month,
            day=df["arrival_date_day_of_month"],
        ),
        errors="coerce",
    )

    # Booking Date = Arrival Date - Lead Time
    booking_date = (
        arrival_date
        - pd.to_timedelta(df["lead_time"], unit="D")
    )

    return arrival_date, booking_date


def main():

    print("=" * 60)
    print("CREATE BOOKING DATE")
    print("=" * 60)

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"\nOriginal rows: {len(df):,}")

    arrival_date, booking_date = create_booking_date(df)

    df["arrival_date"] = arrival_date
    df["booking_date"] = booking_date

    # ตรวจสอบวันที่
    print("\nDate range:")

    print(
        f"Arrival date : "
        f"{df['arrival_date'].min().date()} "
        f"to "
        f"{df['arrival_date'].max().date()}"
    )

    print(
        f"Booking date : "
        f"{df['booking_date'].min().date()} "
        f"to "
        f"{df['booking_date'].max().date()}"
    )

    # สร้างโฟลเดอร์
    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    # บันทึก
    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nSaved to:")
    print(OUTPUT_PATH)

    print("\n✅ Booking date created successfully.")


if __name__ == "__main__":
    main()