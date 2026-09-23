"""
features.py

ฟังก์ชันเตรียมฟีเจอร์สำหรับโมเดลทำนายการยกเลิกการจอง (is_canceled)

หลักการ:
- ฟังก์ชันนี้ "ไม่ fit" อะไรเลย (ไม่มีการเรียนรู้ค่าสถิติจากข้อมูล)
  การ impute/encode/scale ทำใน sklearn Pipeline ที่ train.py เรียกใช้ต่อ
- ใช้ฟังก์ชันเดียวกันนี้ทั้งตอนเทรนและตอน serving เพื่อป้องกัน
  Training-Serving Skew
- อ้างอิงคอลัมน์ทั้งหมดจาก configs/columns.yaml (target, leakage_columns,
  excluded_model_feature, team_decision_columns)
"""

import pandas as pd


# ============================================================
# CONFIG (สะท้อนค่าจาก configs/columns.yaml)
# ============================================================
#
# หมายเหตุ: ตอนนี้ hardcode ไว้ในไฟล์นี้เพื่อความชัดเจน
# ถ้าต้องการ ทำเป็น load จาก configs/columns.yaml ด้วย PyYAML
# ภายหลังได้ (จะได้ไม่มี 2 แหล่งความจริงที่อาจ drift กัน)

TARGET_COLUMN = "is_canceled"

# ต้องตัดทิ้งเสมอ — รู้ค่าได้หลัง booking เกิดขึ้นแล้วเท่านั้น (data leakage)
LEAKAGE_COLUMNS = [
    "reservation_status",
    "reservation_status_date",
    "assigned_room_type",
    "booking_changes",
    "required_car_parking_spaces",
    "total_of_special_requests",
]

# ตัดทิ้งเพราะ split ข้อมูลใช้ arrival_date (ที่รวม year ไว้แล้ว)
# ถ้าปล่อยไว้โมเดลจะ "จำปี" แทนที่จะเรียนรู้ pattern ตามฤดูกาล/แนวโน้ม
EXCLUDED_MODEL_FEATURES = [
    "arrival_date_year",
]

# คอลัมน์ที่ clean_data.py สร้างเพิ่ม ไม่ใช่ฟีเจอร์โมเดล
# (row_id/is_duplicate ใช้เพื่อ trace เฉยๆ, arrival_date/booking_date
#  ใช้แค่ตอน split และคำนวณ lead_time ไปแล้ว)
DERIVED_NON_FEATURE_COLUMNS = [
    "row_id",
    "is_duplicate",
    "arrival_date",
    "booking_date",
]

# ทีมยังไม่ฟันธง — ตอนนี้ "รวม" ไว้เป็นฟีเจอร์ก่อน (default)
# ถ้าทีมตัดสินใจเปลี่ยน ให้ย้ายชื่อคอลัมน์เข้า LEAKAGE_COLUMNS แทน
# แล้วบันทึกเหตุผลลง data_decisions.md
TEAM_DECISION_COLUMNS = [
    "days_in_waiting_list",
    "adr",
]

# คอลัมน์ที่มี missing values ตามธรรมชาติของ dataset (ไม่ error ตอน validate)
OPTIONAL_MISSING_COLUMNS = [
    "country",
    "agent",
    "company",
    "children",
]


# ============================================================
# STEP 1: ตัดคอลัมน์ที่ห้ามเป็นฟีเจอร์
# ============================================================

def drop_non_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ตัด target, leakage columns, excluded columns, derived columns ทิ้ง

    เหลือเฉพาะคอลัมน์ที่มีสิทธิ์เป็นฟีเจอร์เข้าโมเดล
    (TEAM_DECISION_COLUMNS ยังคงอยู่ — ตัดสินใจแยกในขั้นถัดไป)
    """

    columns_to_drop = (
        [TARGET_COLUMN]
        + LEAKAGE_COLUMNS
        + EXCLUDED_MODEL_FEATURES
        + DERIVED_NON_FEATURE_COLUMNS
    )

    # กันพลาด: เตือนถ้าคอลัมน์ที่ตั้งใจตัดไม่มีอยู่จริงในข้อมูล
    missing = [c for c in columns_to_drop if c not in df.columns]
    if missing:
        raise ValueError(
            f"คอลัมน์ที่คาดว่าต้องมีแต่หาไม่เจอใน DataFrame: {missing}\n"
            "เช็คว่าข้อมูลที่ส่งเข้ามาผ่าน clean_data.py แล้วหรือยัง"
        )

    return df.drop(columns=columns_to_drop)


# ============================================================
# STEP 2: จัดการ missing values แบบ "ไม่ fit" (เติมค่าคงที่)
# ============================================================
#
# หมายเหตุ: การเติมค่าตรงนี้เป็นค่าคงที่ (constant) ไม่ใช่ mean/median
# ที่ต้องเรียนรู้จาก train set เท่านั้น จึงทำที่นี่ได้อย่างปลอดภัย
# โดยไม่ทำให้เกิด data leakage ข้ามชุดข้อมูล
# ส่วนการ impute เชิงสถิติ (median ของ numeric อื่นๆ) ให้ทำใน
# sklearn Pipeline ที่ train.py แทน เพราะต้อง fit บน train เท่านั้น

def handle_optional_missing(df: pd.DataFrame) -> pd.DataFrame:
    """เติมค่า missing ของคอลัมน์ที่รู้อยู่แล้วว่ามีค่าว่างตามธรรมชาติ

    - country / agent / company : เติม "Unknown" (เป็น category จริง
      ไม่ใช่ error) พร้อมสร้าง flag คอลัมน์บอกว่าค่านี้หายไปจริงหรือไม่
    - children : เติม 0 (สมมติฐาน: booking ที่ไม่ระบุ children
      น่าจะไม่มีเด็กมาด้วย) พร้อมสร้าง flag เช่นกัน
    """

    df = df.copy()

    # country เป็น string อยู่แล้ว (เช่น "PRT", "GBR") — เติม "Unknown" ตรงๆ ได้
    df["country_missing"] = df["country"].isna().astype(int)
    df["country"] = df["country"].fillna("Unknown").astype(str)

    # agent/company เป็นรหัสตัวเลข แต่ pandas เก็บเป็น float เพราะมี NaN ปน
    # (เช่น 1.0, 240.0, NaN) ถ้า fillna ด้วย string "Unknown" ตรงๆ จะได้
    # คอลัมน์ที่มีทั้ง float และ str ปนกัน ทำให้ OneHotEncoder error
    # ("input argument must be uniformly strings or numbers")
    # จึงต้องแปลงเป็น int ก่อน (ตัด .0 ทิ้ง) แล้วค่อยแปลงเป็น str ทั้งคอลัมน์
    for column in ["agent", "company"]:
        df[f"{column}_missing"] = df[column].isna().astype(int)
        is_missing = df[column].isna()
        df[column] = df[column].fillna(-1).astype(int).astype(str)
        df.loc[is_missing, column] = "Unknown"

    df["children_missing"] = df["children"].isna().astype(int)
    df["children"] = df["children"].fillna(0)

    return df


# ============================================================
# STEP 3: ฟังก์ชันหลัก — เรียกจาก train.py และตอน serving
# ============================================================

def build_features(
    df: pd.DataFrame,
    include_team_decision_columns: bool = True,
) -> pd.DataFrame:
    """แปลง DataFrame ดิบ (จาก train/validation/test/production.csv
    หรือ 1 booking ตอน serving) ให้เป็น DataFrame ฟีเจอร์พร้อมเข้า
    sklearn Pipeline

    Parameters
    ----------
    df : DataFrame ที่มีคอลัมน์ตรงกับ data/processed/*.csv
    include_team_decision_columns : ถ้า False จะตัด `adr` และ
        `days_in_waiting_list` ออกด้วย ใช้ตอนทดลองเปรียบเทียบว่า
        สองคอลัมน์นี้ช่วยโมเดลจริงไหม (ต้องตัดสินใจให้จบก่อนขึ้น
        production - ดู data_decisions.md)

    Returns
    -------
    DataFrame ฟีเจอร์ล้วน (ไม่มี target, ไม่มี leakage columns)
    """

    features = drop_non_feature_columns(df)
    features = handle_optional_missing(features)

    if not include_team_decision_columns:
        features = features.drop(columns=TEAM_DECISION_COLUMNS)

    return features


def get_target(df: pd.DataFrame) -> pd.Series:
    """ดึงคอลัมน์ target ออกมาแยกต่างหาก"""
    return df[TARGET_COLUMN].copy()


# ============================================================
# STEP 4: ตัวช่วยสำหรับทดลองผลของ duplicate rows (ตามที่
# data_decisions.md ยกให้ modeling เป็นคนตัดสินใจ)
# ============================================================

def drop_duplicates_from_train(df: pd.DataFrame) -> pd.DataFrame:
    """ตัดแถวที่ is_duplicate == True ออก ใช้กับ TRAIN เท่านั้น

    ห้ามใช้กับ validation/test/production เพราะชุดนั้นต้องสะท้อน
    การกระจายของข้อมูลจริงตอน inference (ที่อาจมี duplicate ปนอยู่)
    """
    if "is_duplicate" not in df.columns:
        raise ValueError("ไม่พบคอลัมน์ is_duplicate — ใช้ข้อมูลที่ผ่าน clean_data.py แล้วหรือยัง?")

    return df[~df["is_duplicate"]].copy()


# ============================================================
# CLI สำหรับทดสอบเร็วๆ ว่าฟังก์ชันทำงานถูกต้อง
# ============================================================

def _self_check():
    """รันตรงๆ ด้วย `python src/modeling/features.py` เพื่อ sanity check
    ว่าฟังก์ชันใช้ได้กับ train.csv จริง"""

    train_path = "data/processed/train.csv"
    df = pd.read_csv(train_path)

    print(f"Raw shape        : {df.shape}")

    y = get_target(df)
    print(f"Target dist      : {y.value_counts(normalize=True).to_dict()}")

    X = build_features(df)
    print(f"Feature shape    : {X.shape}")
    print(f"Feature columns  : {list(X.columns)}")

    X_no_dup = build_features(drop_duplicates_from_train(df))
    print(f"\nAfter dropping is_duplicate rows: {X_no_dup.shape}")

    X_no_team_decision = build_features(df, include_team_decision_columns=False)
    print(f"\nWithout adr/days_in_waiting_list: {X_no_team_decision.shape}")
    print(f"Missing-flag columns present     : "
          f"{[c for c in X.columns if c.endswith('_missing')]}")


if __name__ == "__main__":
    _self_check()
