import os
import json
import hashlib
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

INPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "interim",
    "hotel_bookings_clean.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

TRAIN_PATH = os.path.join(
    OUTPUT_DIR,
    "train.csv"
)

VALIDATION_PATH = os.path.join(
    OUTPUT_DIR,
    "validation.csv"
)

TEST_PATH = os.path.join(
    OUTPUT_DIR,
    "test.csv"
)

PRODUCTION_PATH = os.path.join(
    OUTPUT_DIR,
    "production.csv"
)

MANIFEST_PATH = os.path.join(
    OUTPUT_DIR,
    "split_manifest.json"
)


# ============================================================
# Helper functions
# ============================================================

def sha256_file(path):
    """คำนวณ SHA-256 ของไฟล์"""
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def month_start(date):
    """เปลี่ยนวันที่ให้เป็นวันแรกของเดือน"""
    return pd.Timestamp(date).replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )


# ============================================================
# Split data by arrival date
# ============================================================

def temporal_split(df):
    """
    แบ่งข้อมูลตาม arrival_date

    Train      : 12 เดือน
    Validation : 6 เดือน
    Test       : 3 เดือน
    Production : ช่วงเวลาที่เหลือ
    """

    if "arrival_date" not in df.columns:
        raise ValueError(
            "Column 'arrival_date' not found in dataset."
        )

    df = df.copy()

    df["arrival_date"] = pd.to_datetime(
        df["arrival_date"],
        errors="raise"
    )

    # เรียงตามวันเข้าพัก
    df = df.sort_values(
        by="arrival_date"
    ).reset_index(drop=True)

    min_date = df["arrival_date"].min()

    # จุดเริ่มต้นของเดือนแรก
    start_month = month_start(min_date)

    # --------------------------------------------------------
    # 12 เดือน Train
    # --------------------------------------------------------

    validation_start = (
        start_month
        + pd.DateOffset(months=12)
    )

    # --------------------------------------------------------
    # 6 เดือน Validation
    # --------------------------------------------------------

    test_start = (
        validation_start
        + pd.DateOffset(months=6)
    )

    # --------------------------------------------------------
    # 3 เดือน Test
    # --------------------------------------------------------

    production_start = (
        test_start
        + pd.DateOffset(months=3)
    )

    # --------------------------------------------------------
    # แบ่งข้อมูล
    # --------------------------------------------------------

    train_df = df[
        df["arrival_date"] < validation_start
    ].copy()

    validation_df = df[
        (df["arrival_date"] >= validation_start)
        & (df["arrival_date"] < test_start)
    ].copy()

    test_df = df[
        (df["arrival_date"] >= test_start)
        & (df["arrival_date"] < production_start)
    ].copy()

    production_df = df[
        df["arrival_date"] >= production_start
    ].copy()

    return (
        train_df,
        validation_df,
        test_df,
        production_df,
    )


# ============================================================
# Validation checks
# ============================================================

def _row_keys(df):
    """
    สร้าง key สำหรับระบุแต่ละแถว

    ใช้ข้อมูลทั้งแถวเพื่อให้สามารถตรวจสอบว่า
    แถวเดียวกันปรากฏอยู่ในหลาย split หรือไม่
    """
    return set(
        pd.util.hash_pandas_object(
            df,
            index=False
        ).tolist()
    )


def check_split_integrity(
    original_df,
    train_df,
    validation_df,
    test_df,
    production_df
):
    """ตรวจว่าการแบ่งข้อมูลครบและไม่มี overlap"""

    total_original = len(original_df)

    total_split = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
        + len(production_df)
    )

    print("\n" + "=" * 60)
    print("SPLIT INTEGRITY CHECK")
    print("=" * 60)

    print(f"Original rows : {total_original:,}")
    print(f"Split rows    : {total_split:,}")

    # --------------------------------------------------------
    # ตรวจว่าจำนวนแถวครบ
    # --------------------------------------------------------

    if total_original != total_split:
        raise ValueError(
            "Split does not contain all original rows."
        )

    # --------------------------------------------------------
    # ตรวจ duplicate / overlap ด้วย row content
    # --------------------------------------------------------

    train_keys = _row_keys(train_df)
    validation_keys = _row_keys(validation_df)
    test_keys = _row_keys(test_df)
    production_keys = _row_keys(production_df)

    overlaps = {
        "Train ∩ Validation":
            train_keys & validation_keys,

        "Train ∩ Test":
            train_keys & test_keys,

        "Train ∩ Production":
            train_keys & production_keys,

        "Validation ∩ Test":
            validation_keys & test_keys,

        "Validation ∩ Production":
            validation_keys & production_keys,

        "Test ∩ Production":
            test_keys & production_keys,
    }

    for name, overlap in overlaps.items():
        print(
            f"{name:<24}: {len(overlap)}"
        )

        if overlap:
            raise ValueError(
                f"Data overlap detected: {name}"
            )

    print(
        "\nTotal rows are complete and non-overlapping."
    )


# ============================================================
# Print split information
# ============================================================

def print_split_info(name, df, total_rows):

    print("\n" + "-" * 60)
    print(name)
    print("-" * 60)

    print(f"Rows       : {len(df):,}")

    if len(df) > 0:

        print(
            f"Percentage : "
            f"{len(df) / total_rows * 100:.2f}%"
        )

        print(
            f"Date range : "
            f"{df['arrival_date'].min().date()} "
            f"→ "
            f"{df['arrival_date'].max().date()}"
        )

        print(
            f"Cancellation rate : "
            f"{df['is_canceled'].mean():.4f}"
        )


# ============================================================
# Save manifest
# ============================================================

def create_manifest(
    train_df,
    validation_df,
    test_df,
    production_df
):

    split_data = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
        "production": production_df,
    }

    output_paths = {
        "train": TRAIN_PATH,
        "validation": VALIDATION_PATH,
        "test": TEST_PATH,
        "production": PRODUCTION_PATH,
    }

    manifest = {
        "split_method": "arrival_date",
        "train_months": 12,
        "validation_months": 6,
        "test_months": 3,
        "production": "remaining_period",
        "splits": {}
    }

    for name, df in split_data.items():

        path = output_paths[name]

        manifest["splits"][name] = {
            "rows": len(df),

            "start_date": (
                str(df["arrival_date"].min().date())
                if len(df) > 0
                else None
            ),

            "end_date": (
                str(df["arrival_date"].max().date())
                if len(df) > 0
                else None
            ),

            "cancellation_rate": (
                float(df["is_canceled"].mean())
                if len(df) > 0
                else None
            ),

            "sha256": sha256_file(path),
        }

    with open(
        MANIFEST_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("TEMPORAL DATA SPLITTING")
    print("=" * 60)

    # --------------------------------------------------------
    # ตรวจว่า input มีอยู่
    # --------------------------------------------------------

    if not os.path.exists(INPUT_PATH):

        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_PATH}\n\n"
            "หมายเหตุ: split_data.py เวอร์ชันใหม่ "
            "ต้องใช้ข้อมูลที่ผ่าน clean_data.py แล้ว"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df_all = pd.read_csv(
        INPUT_PATH,
        parse_dates=["arrival_date"]
    )

    print(
        f"\nTotal rows: {len(df_all):,}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
        production_df
    ) = temporal_split(df_all)

    # --------------------------------------------------------
    # แสดงข้อมูล
    # --------------------------------------------------------

    print_split_info(
        "TRAIN",
        train_df,
        len(df_all)
    )

    print_split_info(
        "VALIDATION",
        validation_df,
        len(df_all)
    )

    print_split_info(
        "TEST",
        test_df,
        len(df_all)
    )

    print_split_info(
        "PRODUCTION",
        production_df,
        len(df_all)
    )

    # --------------------------------------------------------
    # ตรวจ chronological order
    # --------------------------------------------------------

    if not (
        train_df["arrival_date"].max()
        < validation_df["arrival_date"].min()
        and
        validation_df["arrival_date"].max()
        < test_df["arrival_date"].min()
        and
        test_df["arrival_date"].max()
        < production_df["arrival_date"].min()
    ):
        raise ValueError(
            "Temporal split order is invalid."
        )

    print(
        "\nTemporal order check: PASSED"
    )

    # --------------------------------------------------------
    # Integrity check
    # --------------------------------------------------------

    check_split_integrity(
        df_all,
        train_df,
        validation_df,
        test_df,
        production_df
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    train_df.to_csv(
        TRAIN_PATH,
        index=False,
        lineterminator="\n"
    )

    validation_df.to_csv(
        VALIDATION_PATH,
        index=False,
        lineterminator="\n"
    )

    test_df.to_csv(
        TEST_PATH,
        index=False,
        lineterminator="\n"
    )

    production_df.to_csv(
        PRODUCTION_PATH,
        index=False,
        lineterminator="\n"
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    create_manifest(
        train_df,
        validation_df,
        test_df,
        production_df
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FILES CREATED")
    print("=" * 60)

    print(TRAIN_PATH)
    print(VALIDATION_PATH)
    print(TEST_PATH)
    print(PRODUCTION_PATH)
    print(MANIFEST_PATH)

    print(
        "\n✅ Temporal split completed successfully."
    )


if __name__ == "__main__":
    main()