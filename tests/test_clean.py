import pandas as pd
import pytest

from src.clean_data import clean_data


def make_sample_data():
    return pd.DataFrame({
        "hotel": ["Resort Hotel", "City Hotel", "City Hotel"],
        "is_canceled": [0, 1, 0],
        "arrival_date_year": [2016, 2016, 2017],
        "arrival_date_month": ["July", "August", "January"],
        "arrival_date_week_number": [27, 31, 1],
        "arrival_date_day_of_month": [4, 15, 10],
        "stays_in_weekend_nights": [1, 0, 2],
        "stays_in_week_nights": [2, 3, 1],
        "adults": [2, 2, 1],
        "children": [0, 0, 0],
        "babies": [0, 0, 0],
        "meal": ["BB", "HB", "BB"],
        "country": ["PRT", "GBR", "FRA"],
        "market_segment": ["Online TA", "Direct", "Offline TA/TO"],
        "distribution_channel": ["TA/TO", "Direct", "TA/TO"],
        "is_repeated_guest": [0, 0, 0],
        "previous_cancellations": [0, 0, 0],
        "previous_bookings_not_canceled": [0, 0, 0],
        "reserved_room_type": ["A", "B", "A"],
        "assigned_room_type": ["A", "B", "A"],
        "booking_changes": [0, 1, 0],
        "deposit_type": ["No Deposit", "No Deposit", "No Deposit"],
        "agent": [9, 7, 1],
        "company": [None, None, None],
        "days_in_waiting_list": [0, 0, 0],
        "customer_type": ["Transient", "Transient", "Transient"],
        "adr": [100.0, 150.0, 200.0],
        "required_car_parking_spaces": [0, 0, 0],
        "total_of_special_requests": [0, 1, 0],
        "reservation_status": ["Check-Out", "Canceled", "Check-Out"],
        "reservation_status_date": [
            "2016-07-06",
            "2016-08-15",
            "2017-01-12",
        ],
        "lead_time": [10, 20, 30],
    })


def make_large_sample_data():
    return pd.concat(
        [make_sample_data()] * 100,
        ignore_index=True
    )


# ============================================================
# BASIC CLEANING TESTS
# ============================================================

def test_clean_data_returns_dataframe():
    df = make_sample_data()

    result = clean_data(df)

    assert isinstance(result, pd.DataFrame)


def test_negative_adr_is_removed():
    df = make_large_sample_data()

    df.loc[0, "adr"] = -10

    result = clean_data(df)

    assert len(result) == len(df) - 1
    assert (result["adr"] < 0).sum() == 0


def test_zero_guest_row_is_removed():
    df = make_large_sample_data()

    df.loc[0, "adults"] = 0
    df.loc[0, "children"] = 0
    df.loc[0, "babies"] = 0

    result = clean_data(df)

    assert len(result) == len(df) - 1


# ============================================================
# DUPLICATE TESTS
# ============================================================

def test_duplicates_are_flagged_without_being_removed():
    df = make_sample_data()

    duplicate_row = df.iloc[[0]].copy()

    df = pd.concat(
        [df, duplicate_row],
        ignore_index=True
    )

    result = clean_data(df)

    assert len(result) == len(df)
    assert "is_duplicate" in result.columns
    assert result["is_duplicate"].sum() == 2


def test_unique_rows_are_not_flagged_as_duplicates():
    df = make_sample_data()

    result = clean_data(df)

    assert result["is_duplicate"].sum() == 0


# ============================================================
# DERIVED COLUMN TESTS
# ============================================================

def test_row_id_is_created():
    df = make_sample_data()

    result = clean_data(df)

    assert "row_id" in result.columns
    assert result["row_id"].tolist() == [0, 1, 2]


def test_arrival_date_is_created_correctly():
    df = make_sample_data()

    result = clean_data(df)

    assert "arrival_date" in result.columns

    assert pd.api.types.is_datetime64_any_dtype(
        result["arrival_date"]
    )

    assert result.loc[0, "arrival_date"] == pd.Timestamp(
        "2016-07-04"
    )


def test_booking_date_is_created_from_lead_time():
    df = make_sample_data()

    result = clean_data(df)

    assert "booking_date" in result.columns

    assert pd.api.types.is_datetime64_any_dtype(
        result["booking_date"]
    )

    assert result.loc[0, "booking_date"] == pd.Timestamp(
        "2016-06-24"
    )


# ============================================================
# SAFETY THRESHOLD TEST
# ============================================================

def test_cleaning_stops_when_removal_exceeds_one_percent():
    df = make_sample_data()

    df.loc[0, "adr"] = -10

    with pytest.raises(ValueError):
        clean_data(df)


# ============================================================
# VALID DATA TEST
# ============================================================

def test_clean_data_keeps_valid_rows():
    df = make_sample_data()

    result = clean_data(df)

    assert len(result) == len(df)
    assert (result["adr"] >= 0).all()