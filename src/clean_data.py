import os
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RAW_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "hotel_bookings.csv"
)

INTERIM_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "interim"
)

OUTPUT_PATH = os.path.join(
    INTERIM_DIR,
    "hotel_bookings_clean.csv"
)

MAX_REMOVAL_PERCENT = 1.0


# ============================================================
# Create derived columns
# ============================================================

def create_derived_columns(df):

    df = df.copy()

    # --------------------------------------------------------
    # row_id
    # --------------------------------------------------------

    df.insert(
        0,
        "row_id",
        range(len(df))
    )

    # --------------------------------------------------------
    # arrival_date
    # --------------------------------------------------------

    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str)
        + "-"
        + df["arrival_date_month"]
        + "-"
        + df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
        errors="raise"
    )

    # --------------------------------------------------------
    # booking_date
    # --------------------------------------------------------

    df["booking_date"] = (
        df["arrival_date"]
        - pd.to_timedelta(
            df["lead_time"],
            unit="D"
        )
    )

    return df


# ============================================================
# Clean data
# ============================================================

def clean_data(df):

    df = df.copy()

    original_rows = len(df)

    # --------------------------------------------------------
    # Duplicate flag
    # --------------------------------------------------------

    df["is_duplicate"] = df.duplicated(
        keep=False
    )

    # --------------------------------------------------------
    # Invalid ADR
    # --------------------------------------------------------

    invalid_adr = df["adr"] < 0

    # --------------------------------------------------------
    # Zero guests
    # --------------------------------------------------------

    total_guests = (
        df["adults"].fillna(0)
        + df["children"].fillna(0)
        + df["babies"].fillna(0)
    )

    zero_guests = total_guests == 0

    # --------------------------------------------------------
    # Combine rows to remove
    # --------------------------------------------------------

    remove_mask = (
        invalid_adr
        | zero_guests
    )

    rows_to_remove = int(
        remove_mask.sum()
    )

    removal_percent = (
        rows_to_remove
        / original_rows
        * 100
    )

    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)

    print(
        f"\nOriginal rows : "
        f"{original_rows:,}"
    )

    print(
        f"Invalid ADR   : "
        f"{invalid_adr.sum():,}"
    )

    print(
        f"Zero guests   : "
        f"{zero_guests.sum():,}"
    )

    print(
        f"Rows removed  : "
        f"{rows_to_remove:,}"
    )

    print(
        f"Removal rate  : "
        f"{removal_percent:.4f}%"
    )

    # --------------------------------------------------------
    # Safety threshold
    # --------------------------------------------------------

    if removal_percent > MAX_REMOVAL_PERCENT:

        raise ValueError(
            f"Cleaning would remove "
            f"{removal_percent:.2f}% of rows, "
            f"which exceeds the {MAX_REMOVAL_PERCENT}% limit."
        )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    clean_df = df[
        ~remove_mask
    ].copy()

    print(
        f"\nClean rows    : "
        f"{len(clean_df):,}"
    )

    # --------------------------------------------------------
    # Derived columns
    # --------------------------------------------------------

    clean_df = create_derived_columns(
        clean_df
    )

    return clean_df


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("CLEAN HOTEL BOOKING DATA")
    print("=" * 60)

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    if not os.path.exists(
        RAW_DATA_PATH
    ):

        raise FileNotFoundError(
            f"Raw dataset not found:\n"
            f"{RAW_DATA_PATH}"
        )

    df = pd.read_csv(
        RAW_DATA_PATH
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    clean_df = clean_data(
        df
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        INTERIM_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    clean_df.to_csv(
        OUTPUT_PATH,
        index=False,
        lineterminator="\n"
    )

    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)

    print(
        f"\nSaved to:\n"
        f"{OUTPUT_PATH}"
    )

    print(
        f"\nFinal rows: "
        f"{len(clean_df):,}"
    )

    print(
        "\nColumns added:"
    )

    print(
        " - row_id"
    )
    print(
        " - is_duplicate"
    )
    print(
        " - arrival_date"
    )
    print(
        " - booking_date"
    )

    print(
        "\n✅ Cleaning completed successfully."
    )


if __name__ == "__main__":
    main()