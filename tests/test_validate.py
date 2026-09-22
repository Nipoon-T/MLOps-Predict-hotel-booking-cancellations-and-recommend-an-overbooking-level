import pandas as pd
import subprocess
import sys

from src.validate import (
    validate_columns,
    validate_binary_values,
    validate_non_negative_values,
    validate_months,
    validate_years,
    validate_guest_count,
)

from src.split_data import temporal_split, sha256_file


RAW_DATA_PATH = "data/raw/hotel_bookings.csv"
CLEAN_DATA_PATH = "data/interim/hotel_bookings_clean.csv"


def load_raw_data():
    return pd.read_csv(RAW_DATA_PATH)


def load_clean_data():
    return pd.read_csv(
        CLEAN_DATA_PATH,
        parse_dates=["arrival_date"],
    )


# ============================================================
# VALID DATA TESTS
# ============================================================

def test_schema_columns_are_valid():
    df = load_raw_data()

    errors = validate_columns(df)

    assert errors == []


def test_binary_columns_are_valid():
    df = load_raw_data()

    errors = validate_binary_values(df)

    assert errors == []


def test_raw_adr_negative_value_is_detected():
    df = load_raw_data()

    errors = validate_non_negative_values(df)

    assert any("adr" in error for error in errors)


def test_month_values_are_valid():
    df = load_raw_data()

    errors = validate_months(df)

    assert errors == []


def test_year_values_are_valid():
    df = load_raw_data()

    errors = validate_years(df)

    assert errors == []


# ============================================================
# INVALID DATA TESTS
# ============================================================

def test_invalid_binary_value_is_detected():
    df = load_raw_data()

    df.loc[0, "is_canceled"] = 2

    errors = validate_binary_values(df)

    assert any("is_canceled" in error for error in errors)


def test_negative_lead_time_is_detected():
    df = load_raw_data()

    df.loc[0, "lead_time"] = -10

    errors = validate_non_negative_values(df)

    assert any("lead_time" in error for error in errors)


def test_invalid_month_is_detected():
    df = load_raw_data()

    df.loc[0, "arrival_date_month"] = "Januar"

    errors = validate_months(df)

    assert any("arrival_date_month" in error for error in errors)


def test_invalid_year_is_detected():
    df = load_raw_data()

    df.loc[0, "arrival_date_year"] = 2020

    errors = validate_years(df)

    assert any("arrival_date_year" in error for error in errors)


def test_zero_guest_count_is_detected():
    df = load_raw_data()

    df.loc[0, "adults"] = 0
    df.loc[0, "children"] = 0
    df.loc[0, "babies"] = 0

    errors = validate_guest_count(df)

    assert any("guest_count" in error for error in errors)


# ============================================================
# CLI TESTS
# ============================================================

def test_validation_cli_returns_exit_code_1_for_bad_data():
    result = subprocess.run(
        [
            sys.executable,
            "src/validate.py",
            "data/bad/hotel_bookings_bad.csv",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "VALIDATION FAILED" in result.stdout


# ============================================================
# TEMPORAL SPLIT TESTS
# ============================================================

def test_temporal_split_is_complete():
    df = load_clean_data()

    (
        train_df,
        validation_df,
        test_df,
        production_df,
    ) = temporal_split(df)

    total_split = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
        + len(production_df)
    )

    assert total_split == len(df)


def test_temporal_split_is_chronological():
    df = load_clean_data()

    (
        train_df,
        validation_df,
        test_df,
        production_df,
    ) = temporal_split(df)

    assert (
        train_df["arrival_date"].max()
        < validation_df["arrival_date"].min()
    )

    assert (
        validation_df["arrival_date"].max()
        < test_df["arrival_date"].min()
    )

    assert (
        test_df["arrival_date"].max()
        < production_df["arrival_date"].min()
    )


# ============================================================
# SPLIT INTEGRITY TESTS
# ============================================================

def test_temporal_split_has_no_overlap():
    df = load_clean_data()

    (
        train_df,
        validation_df,
        test_df,
        production_df,
    ) = temporal_split(df)

    train_ids = set(train_df["row_id"])
    validation_ids = set(validation_df["row_id"])
    test_ids = set(test_df["row_id"])
    production_ids = set(production_df["row_id"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert train_ids.isdisjoint(production_ids)

    assert validation_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(production_ids)

    assert test_ids.isdisjoint(production_ids)


def test_split_hash_is_stable():
    train_path = "data/processed/train.csv"
    validation_path = "data/processed/validation.csv"
    test_path = "data/processed/test.csv"
    production_path = "data/processed/production.csv"

    train_hash_1 = sha256_file(train_path)
    validation_hash_1 = sha256_file(validation_path)
    test_hash_1 = sha256_file(test_path)
    production_hash_1 = sha256_file(production_path)

    train_hash_2 = sha256_file(train_path)
    validation_hash_2 = sha256_file(validation_path)
    test_hash_2 = sha256_file(test_path)
    production_hash_2 = sha256_file(production_path)

    assert train_hash_1 == train_hash_2
    assert validation_hash_1 == validation_hash_2
    assert test_hash_1 == test_hash_2
    assert production_hash_1 == production_hash_2