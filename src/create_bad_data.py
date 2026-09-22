import os
import sys
import pandas as pd

# เพิ่ม project root เข้า Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# กำหนด paths จาก project root
RAW_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "hotel_bookings.csv"
)

BAD_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "bad",
    "hotel_bookings_bad.csv"
)


def create_bad_data():
    print("=" * 60)
    print("CREATE BAD DATA FOR VALIDATION DEMONSTRATION")
    print("=" * 60)

    # ==========================================
    # 1. โหลดข้อมูลต้นฉบับ
    # ==========================================
    df = pd.read_csv(RAW_DATA_PATH)

    print(f"\nOriginal rows: {len(df):,}")

    # ==========================================
    # 2. สร้างสำเนา
    # ==========================================
    bad_df = df.copy()

    # ==========================================
    # 3. สร้างข้อมูลเสียแบบตั้งใจ
    # ==========================================

    # Bad data #1:
    # is_canceled ต้องเป็น 0 หรือ 1
    bad_df.loc[0, "is_canceled"] = 2

    # Bad data #2:
    # lead_time ต้องไม่ติดลบ
    bad_df.loc[1, "lead_time"] = -10

    # Bad data #3:
    # arrival_date_month ต้องเป็นเดือนที่ถูกต้อง
    bad_df.loc[2, "arrival_date_month"] = "Januar"

    # Bad data #4:
    # adr ต้องไม่ติดลบ
    bad_df.loc[3, "adr"] = -50.0

    # Bad data #5:
    # is_repeated_guest ต้องเป็น 0 หรือ 1
    bad_df.loc[4, "is_repeated_guest"] = 3

    # ==========================================
    # 4. สร้างโฟลเดอร์ถ้ายังไม่มี
    # ==========================================
    os.makedirs(
        os.path.dirname(BAD_DATA_PATH),
        exist_ok=True
    )

    # ==========================================
    # 5. บันทึก Bad Dataset
    # ==========================================
    bad_df.to_csv(
        BAD_DATA_PATH,
        index=False
    )

    print("\nBad data saved to:")
    print(BAD_DATA_PATH)

    print("\nInjected errors:")

    print("1. Row 0: is_canceled = 2")
    print("2. Row 1: lead_time = -10")
    print("3. Row 2: arrival_date_month = 'Januar'")
    print("4. Row 3: adr = -50.0")
    print("5. Row 4: is_repeated_guest = 3")

    print("\n✅ Bad dataset created successfully.")


if __name__ == "__main__":
    create_bad_data()