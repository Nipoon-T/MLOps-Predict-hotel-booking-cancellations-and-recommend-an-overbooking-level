import hashlib

import pandas as pd
import pytest

from src.split_data import (
    temporal_split,
    check_split_integrity,
    month_start,
    sha256_file,
)


def make_split_sample():
    """
    สร้างข้อมูลสำหรับทดสอบ temporal split

    ครอบคลุมมากกว่า 21 เดือน
    เพื่อให้มี Train / Validation / Test / Production
    """
    dates = pd.date_range(
        start="2020-01-01",
        end="2022-06-01",
        freq="MS",
    )

    return pd.DataFrame({
        "arrival_date": dates,
        "is_canceled": [0, 1] * (len(dates) // 2)
        + [0] * (len(dates) % 2),
    })


# ============================================================
# TEST 1: SPLIT COMPLETENESS
# ============================================================

def test_temporal_split_contains_all_rows():
    df = make_split_sample()

    train, validation, test, production = temporal_split(df)

    total_split = (
        len(train)
        + len(validation)
        + len(test)
        + len(production)
    )

    assert total_split == len(df)


# ============================================================
# TEST 2: CHRONOLOGICAL ORDER
# ============================================================

def test_temporal_split_is_chronological():
    df = make_split_sample()

    train, validation, test, production = temporal_split(df)

    assert (
        train["arrival_date"].max()
        < validation["arrival_date"].min()
    )

    assert (
        validation["arrival_date"].max()
        < test["arrival_date"].min()
    )

    assert (
        test["arrival_date"].max()
        < production["arrival_date"].min()
    )


# ============================================================
# TEST 3: NO OVERLAP
# ============================================================

def test_temporal_split_has_no_overlap():
    df = make_split_sample()

    train, validation, test, production = temporal_split(df)

    train_dates = set(train["arrival_date"])
    validation_dates = set(validation["arrival_date"])
    test_dates = set(test["arrival_date"])
    production_dates = set(production["arrival_date"])

    assert train_dates.isdisjoint(validation_dates)
    assert train_dates.isdisjoint(test_dates)
    assert train_dates.isdisjoint(production_dates)

    assert validation_dates.isdisjoint(test_dates)
    assert validation_dates.isdisjoint(production_dates)

    assert test_dates.isdisjoint(production_dates)


# ============================================================
# TEST 4: MISSING ARRIVAL DATE
# ============================================================

def test_temporal_split_requires_arrival_date():
    df = pd.DataFrame({
        "is_canceled": [0, 1, 0]
    })

    with pytest.raises(
        ValueError,
        match="arrival_date"
    ):
        temporal_split(df)


# ============================================================
# TEST 5: SHA-256
# ============================================================

def test_sha256_file_returns_correct_hash(tmp_path):
    test_file = tmp_path / "test.txt"

    content = b"hello world"

    test_file.write_bytes(content)

    expected_hash = hashlib.sha256(
        content
    ).hexdigest()

    actual_hash = sha256_file(
        test_file
    )

    assert actual_hash == expected_hash


# ============================================================
# TEST 6: MONTH START
# ============================================================

def test_month_start_returns_first_day_of_month():
    result = month_start(
        pd.Timestamp("2020-05-17 14:30:45")
    )

    assert result == pd.Timestamp(
        "2020-05-01 00:00:00"
    )


# ============================================================
# TEST 7: SPLIT INTEGRITY DETECTS MISSING ROWS
# ============================================================

def test_check_split_integrity_detects_missing_rows():
    df = make_split_sample()

    train, validation, test, production = temporal_split(df)

    # ลบ 1 แถวออกจาก production
    production = production.iloc[:-1].copy()

    with pytest.raises(
        ValueError,
        match="Split does not contain all original rows"
    ):
        check_split_integrity(
            df,
            train,
            validation,
            test,
            production,
        )