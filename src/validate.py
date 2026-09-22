import os
import sys
import argparse
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DEFAULT_INPUT = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "hotel_bookings.csv"
)


# ============================================================
# SCHEMA
# ============================================================

EXPECTED_COLUMNS = [
    "hotel",
    "is_canceled",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "meal",
    "country",
    "market_segment",
    "distribution_channel",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "reserved_room_type",
    "assigned_room_type",
    "booking_changes",
    "deposit_type",
    "agent",
    "company",
    "days_in_waiting_list",
    "customer_type",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
    "reservation_status",
    "reservation_status_date",
]


# ============================================================
# BINARY COLUMNS
# ============================================================

BINARY_COLUMNS = [
    "is_canceled",
    "is_repeated_guest",
]


# ============================================================
# NON-NEGATIVE COLUMNS
# ============================================================

NON_NEGATIVE_COLUMNS = [
    "lead_time",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]


# ============================================================
# OPTIONAL MISSING COLUMNS
# ============================================================

# Columns where missing values are known to occur in the
# original hotel booking dataset.
#
# These are reported by validate_missingness() but do not
# automatically make the dataset fail validation.

OPTIONAL_MISSING_COLUMNS = [
    "children",
    "country",
    "agent",
    "company",
]


# ============================================================
# CATEGORICAL VALUES
# ============================================================

VALID_MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

VALID_YEARS = [
    2015,
    2016,
    2017,
]

VALID_HOTELS = [
    "Resort Hotel",
    "City Hotel",
]

VALID_MEALS = [
    "BB",
    "FB",
    "HB",
    "SC",
    "Undefined",
]

VALID_MARKET_SEGMENTS = [
    "Direct",
    "Corporate",
    "Online TA",
    "Offline TA/TO",
    "Complementary",
    "Groups",
    "Undefined",
    "Aviation",
]

VALID_DISTRIBUTION_CHANNELS = [
    "Direct",
    "Corporate",
    "TA/TO",
    "Undefined",
    "GDS",
]

VALID_DEPOSIT_TYPES = [
    "No Deposit",
    "Refundable",
    "Non Refund",
]

VALID_CUSTOMER_TYPES = [
    "Transient",
    "Contract",
    "Transient-Party",
    "Group",
]

VALID_RESERVATION_STATUS = [
    "Check-Out",
    "Canceled",
    "No-Show",
]


# ============================================================
# NUMERIC BOUNDS
# ============================================================

# These are structural/domain bounds.
#
# ADR uses a loose upper bound of 6000 as a safety check.
# The actual dataset maximum is below this value.

UPPER_BOUNDS = {
    "arrival_date_week_number": 53,
    "arrival_date_day_of_month": 31,
    "adr": 6000,
}


LOWER_BOUNDS = {
    "arrival_date_week_number": 1,
    "arrival_date_day_of_month": 1,
}


# ============================================================
# DERIVED COLUMNS
# ============================================================

DERIVED_COLUMNS = [
    "row_id",
    "is_duplicate",
    "arrival_date",
    "booking_date",
]


# ============================================================
# VALIDATION: COLUMNS
# ============================================================

def validate_columns(df, allow_derived=False):
    """
    Validate dataset columns.
    """

    errors = []

    actual_columns = list(df.columns)

    if allow_derived:
        allowed_columns = EXPECTED_COLUMNS + DERIVED_COLUMNS

        unexpected_columns = [
            col
            for col in actual_columns
            if col not in allowed_columns
        ]

        missing_columns = [
            col
            for col in EXPECTED_COLUMNS
            if col not in actual_columns
        ]

    else:
        unexpected_columns = [
            col
            for col in actual_columns
            if col not in EXPECTED_COLUMNS
        ]

        missing_columns = [
            col
            for col in EXPECTED_COLUMNS
            if col not in actual_columns
        ]

    if missing_columns:
        errors.append(
            f"Missing columns: {missing_columns}"
        )

    if unexpected_columns:
        errors.append(
            f"Unexpected columns: {unexpected_columns}"
        )

    return errors


# ============================================================
# VALIDATION: MISSING VALUES
# ============================================================

def validate_missingness(df):
    """
    Validate missing values.

    Known optional-missing columns such as company, agent,
    country, and children are reported but do not fail
    validation automatically.

    Critical columns with missing values are treated as errors.
    """

    errors = []

    # Columns that should not contain missing values.
    critical_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in OPTIONAL_MISSING_COLUMNS
    ]

    for column in critical_columns:

        if column not in df.columns:
            continue

        count = int(df[column].isna().sum())

        if count > 0:
            errors.append(
                f"{column}: {count} missing values"
            )

    return errors


def get_missingness_report(df):
    """
    Return missing-value counts for all columns.

    This function is informational and does not determine
    validation pass/fail status.
    """

    report = {}

    for column in df.columns:
        count = int(df[column].isna().sum())

        if count > 0:
            report[column] = count

    return report


# ============================================================
# VALIDATION: DATA TYPES
# ============================================================

def validate_data_types(df):
    """
    Validate that columns have compatible data types.

    Numeric columns may be integer or floating-point because
    pandas can represent columns containing missing values
    as float.
    """

    errors = []

    numeric_columns = [
        "is_canceled",
        "lead_time",
        "arrival_date_year",
        "arrival_date_week_number",
        "arrival_date_day_of_month",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "babies",
        "is_repeated_guest",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "booking_changes",
        "days_in_waiting_list",
        "adr",
        "required_car_parking_spaces",
        "total_of_special_requests",
    ]

    for column in numeric_columns:

        if column not in df.columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[column]):
            errors.append(
                f"{column}: expected numeric type, "
                f"got {df[column].dtype}"
            )

    return errors


# ============================================================
# VALIDATION: BINARY VALUES
# ============================================================

def validate_binary_values(df):
    """
    Validate binary columns.
    """

    errors = []

    for column in BINARY_COLUMNS:

        if column not in df.columns:
            continue

        invalid = (
            ~df[column].isin([0, 1])
            & df[column].notna()
        )

        count = int(invalid.sum())

        if count > 0:
            errors.append(
                f"{column}: {count} invalid binary values"
            )

    return errors


# ============================================================
# VALIDATION: NON-NEGATIVE VALUES
# ============================================================

def validate_non_negative_values(df):
    """
    Validate columns that must be >= 0.
    """

    errors = []

    for column in NON_NEGATIVE_COLUMNS:

        if column not in df.columns:
            continue

        invalid = df[column] < 0

        count = int(invalid.sum())

        if count > 0:
            errors.append(
                f"{column}: {count} negative values"
            )

    return errors


# ============================================================
# VALIDATION: UPPER BOUNDS
# ============================================================

def validate_upper_bounds(df):
    """
    Validate numeric upper bounds.
    """

    errors = []

    for column, maximum in UPPER_BOUNDS.items():

        if column not in df.columns:
            continue

        invalid = (
            df[column].notna()
            & (df[column] > maximum)
        )

        count = int(invalid.sum())

        if count > 0:
            errors.append(
                f"{column}: {count} values above maximum "
                f"allowed value {maximum}"
            )

    return errors


# ============================================================
# VALIDATION: LOWER BOUNDS
# ============================================================

def validate_lower_bounds(df):
    """
    Validate numeric lower bounds.
    """

    errors = []

    for column, minimum in LOWER_BOUNDS.items():

        if column not in df.columns:
            continue

        invalid = (
            df[column].notna()
            & (df[column] < minimum)
        )

        count = int(invalid.sum())

        if count > 0:
            errors.append(
                f"{column}: {count} values below minimum "
                f"allowed value {minimum}"
            )

    return errors


# ============================================================
# VALIDATION: MONTHS
# ============================================================

def validate_months(df):
    """
    Validate arrival_date_month.
    """

    errors = []

    column = "arrival_date_month"

    if column not in df.columns:
        return errors

    invalid = (
        ~df[column].isin(VALID_MONTHS)
        & df[column].notna()
    )

    count = int(invalid.sum())

    if count > 0:
        errors.append(
            f"{column}: {count} invalid month values"
        )

    return errors


# ============================================================
# VALIDATION: YEARS
# ============================================================

def validate_years(df):
    """
    Validate arrival_date_year.
    """

    errors = []

    column = "arrival_date_year"

    if column not in df.columns:
        return errors

    invalid = (
        ~df[column].isin(VALID_YEARS)
        & df[column].notna()
    )

    count = int(invalid.sum())

    if count > 0:
        errors.append(
            f"{column}: {count} invalid year values"
        )

    return errors


# ============================================================
# VALIDATION: CATEGORICAL COLUMNS
# ============================================================

def validate_categories(df):
    """
    Validate categorical values.
    """

    errors = []

    category_rules = {
        "hotel": VALID_HOTELS,
        "meal": VALID_MEALS,
        "market_segment": VALID_MARKET_SEGMENTS,
        "distribution_channel": VALID_DISTRIBUTION_CHANNELS,
        "deposit_type": VALID_DEPOSIT_TYPES,
        "customer_type": VALID_CUSTOMER_TYPES,
        "reservation_status": VALID_RESERVATION_STATUS,
    }

    for column, valid_values in category_rules.items():

        if column not in df.columns:
            continue

        invalid = (
            ~df[column].isin(valid_values)
            & df[column].notna()
        )

        count = int(invalid.sum())

        if count > 0:
            errors.append(
                f"{column}: {count} invalid category values"
            )

    return errors


# ============================================================
# VALIDATION: DATE VALUES
# ============================================================

def validate_dates(df):
    """
    Validate arrival date constructed from:

        arrival_date_year
        arrival_date_month
        arrival_date_day_of_month

    Also validates reservation_status_date when present.
    """

    errors = []

    required_columns = [
        "arrival_date_year",
        "arrival_date_month",
        "arrival_date_day_of_month",
    ]

    if all(
        column in df.columns
        for column in required_columns
    ):

        month_numbers = {
            month: index
            for index, month in enumerate(
                VALID_MONTHS,
                start=1
            )
        }

        month_number = df[
            "arrival_date_month"
        ].map(month_numbers)

        constructed_dates = pd.to_datetime(
            pd.DataFrame({
                "year": df["arrival_date_year"],
                "month": month_number,
                "day": df["arrival_date_day_of_month"],
            }),
            errors="coerce",
        )

        invalid = (
            constructed_dates.isna()
            & df[
                [
                    "arrival_date_year",
                    "arrival_date_month",
                    "arrival_date_day_of_month",
                ]
            ].notna().all(axis=1)
        )

        count = int(invalid.sum())

        if count > 0:
            errors.append(
                f"arrival_date: {count} invalid dates"
            )

    # --------------------------------------------------------
    # reservation_status_date
    # --------------------------------------------------------

    if "reservation_status_date" in df.columns:

        status_dates = pd.to_datetime(
            df["reservation_status_date"],
            errors="coerce"
        )

        invalid = (
            status_dates.isna()
            & df["reservation_status_date"].notna()
        )

        count = int(invalid.sum())

        if count > 0:
            errors.append(
                "reservation_status_date: "
                f"{count} invalid date values"
            )

    return errors


# ============================================================
# VALIDATION: GUEST COUNT
# ============================================================

def validate_guest_count(df):
    """
    Validate that each booking has at least one guest.

    A booking is invalid when:

        adults + children + babies == 0
    """

    errors = []

    required_columns = [
        "adults",
        "children",
        "babies",
    ]

    if not all(
        column in df.columns
        for column in required_columns
    ):
        return errors

    guest_count = (
        df["adults"].fillna(0)
        + df["children"].fillna(0)
        + df["babies"].fillna(0)
    )

    invalid = guest_count == 0

    count = int(invalid.sum())

    if count > 0:
        errors.append(
            f"guest_count: {count} rows with zero guests"
        )

    return errors


# ============================================================
# VALIDATION: DERIVED DATE COLUMNS
# ============================================================

def validate_derived_dates(df):
    """
    Validate derived date columns when they are present.
    """

    errors = []

    for column in [
        "arrival_date",
        "booking_date",
    ]:

        if column not in df.columns:
            continue

        parsed = pd.to_datetime(
            df[column],
            errors="coerce"
        )

        invalid = (
            parsed.isna()
            & df[column].notna()
        )

        count = int(invalid.sum())

        if count > 0:
            errors.append(
                f"{column}: {count} invalid date values"
            )

    return errors


# ============================================================
# MAIN VALIDATION
# ============================================================

def validate_dataframe(df, allow_derived=False):

    errors = []

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    errors.extend(
        validate_columns(
            df,
            allow_derived=allow_derived
        )
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    errors.extend(
        validate_missingness(df)
    )

    # --------------------------------------------------------
    # Data types
    # --------------------------------------------------------

    errors.extend(
        validate_data_types(df)
    )

    # --------------------------------------------------------
    # Binary
    # --------------------------------------------------------

    errors.extend(
        validate_binary_values(df)
    )

    # --------------------------------------------------------
    # Non-negative
    # --------------------------------------------------------

    errors.extend(
        validate_non_negative_values(df)
    )

    # --------------------------------------------------------
    # Bounds
    # --------------------------------------------------------

    errors.extend(
        validate_upper_bounds(df)
    )

    errors.extend(
        validate_lower_bounds(df)
    )

    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    errors.extend(
        validate_months(df)
    )

    errors.extend(
        validate_years(df)
    )

    errors.extend(
        validate_categories(df)
    )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    errors.extend(
        validate_dates(df)
    )

    errors.extend(
        validate_derived_dates(df)
    )

    # --------------------------------------------------------
    # Guest count
    # --------------------------------------------------------

    errors.extend(
        validate_guest_count(df)
    )

    return errors


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Validate hotel booking dataset"
    )

    parser.add_argument(
        "input",
        nargs="?",
        default=DEFAULT_INPUT,
        help="Path to CSV file"
    )

    parser.add_argument(
        "--allow-derived",
        action="store_true",
        help="Allow derived columns created during cleaning"
    )

    return parser.parse_args()


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    args = parse_args()

    print("=" * 60)
    print("HOTEL BOOKING DATA VALIDATION")
    print("=" * 60)

    print()
    print(f"Input : {args.input}")
    print(
        f"Allow derived columns : "
        f"{args.allow_derived}"
    )
    print()

    if not os.path.exists(args.input):
        print(
            f"❌ File not found: {args.input}"
        )
        return 1

    try:

        df = pd.read_csv(
            args.input
        )

    except Exception as exc:

        print(
            f"❌ Failed to read CSV: {exc}"
        )
        return 1

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    print()
    print("-" * 60)
    print("VALIDATION")
    print("-" * 60)

    errors = validate_dataframe(
        df,
        allow_derived=args.allow_derived
    )

    if errors:

        print()
        print("❌ VALIDATION FAILED")
        print()

        for error in errors:
            print(
                f" - {error}"
            )

        print()
        print("=" * 60)

        return 1

    print()
    print("✅ VALIDATION PASSED")
    print()
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())