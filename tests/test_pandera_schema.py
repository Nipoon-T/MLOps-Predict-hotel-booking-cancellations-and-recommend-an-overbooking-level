import pandas as pd
import pandera.pandas as pa

from schemas.hotel_booking_schema import (
    VALID_MONTHS,
    VALID_YEARS,
)


def build_sample_dataframe():
    return pd.DataFrame(
        {
            "hotel": ["Resort Hotel", "City Hotel"],
            "is_canceled": [0, 1],
            "lead_time": [10, 30],
            "arrival_date_year": [2016, 2017],
            "arrival_date_month": ["July", "August"],
            "arrival_date_week_number": [27, 32],
            "arrival_date_day_of_month": [5, 10],
            "stays_in_weekend_nights": [1, 2],
            "stays_in_week_nights": [2, 3],
            "adults": [2, 1],
            "children": [0, 1],
            "babies": [0, 0],
            "booking_changes": [0, 1],
            "days_in_waiting_list": [0, 0],
            "adr": [100.0, 150.0],
            "required_car_parking_spaces": [0, 1],
            "total_of_special_requests": [1, 2],
            "is_repeated_guest": [0, 1],
        }
    )


schema = pa.DataFrameSchema(
    {
        "hotel": pa.Column(
            str,
            checks=pa.Check.isin(["Resort Hotel", "City Hotel"]),
        ),
        "is_canceled": pa.Column(
            int,
            checks=pa.Check.isin([0, 1]),
        ),
        "arrival_date_year": pa.Column(
            int,
            checks=pa.Check.isin(VALID_YEARS),
        ),
        "arrival_date_month": pa.Column(
            str,
            checks=pa.Check.isin(VALID_MONTHS),
        ),
        "lead_time": pa.Column(
            int,
            checks=pa.Check.ge(0),
        ),
        "adr": pa.Column(
            float,
            checks=pa.Check.ge(0),
        ),
        "is_repeated_guest": pa.Column(
            int,
            checks=pa.Check.isin([0, 1]),
        ),
    },
    strict=False,
)


def test_sample_data_passes_pandera():
    df = build_sample_dataframe()

    validated = schema.validate(df)

    assert len(validated) == 2


def test_negative_lead_time_fails():
    df = build_sample_dataframe()
    df.loc[0, "lead_time"] = -1

    try:
        schema.validate(df)
    except pa.errors.SchemaError:
        return

    raise AssertionError("Pandera should reject negative lead_time")


def test_invalid_cancellation_value_fails():
    df = build_sample_dataframe()
    df.loc[0, "is_canceled"] = 2

    try:
        schema.validate(df)
    except pa.errors.SchemaError:
        return

    raise AssertionError("Pandera should reject invalid is_canceled")