"""
pytest -q  (รันจาก overbooking_decision/)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest import aggregate_results, backtest_night, backtest_policies
from cost_config import load_config
from policies import policy_avg_rate, policy_fixed, policy_no_overbook
from simulate import find_optimal_overbook, simulate_show_matrix


def test_simulate_show_matrix_shape():
    p = [0.1, 0.2, 0.3]
    shows = simulate_show_matrix(p, n_sims=500, seed=1)
    assert shows.shape == (500, 3)
    assert shows.dtype == bool


def test_simulate_arrivals_mean_close_to_expected():
    # ทำนายว่าค่าเฉลี่ยของผู้มาจริงต้องใกล้ sum(1-p) เมื่อ n_sims มาก และไม่ถูกตัดด้วย capacity/o
    p = np.array([0.1] * 100)
    shows = simulate_show_matrix(p, n_sims=5000, seed=1)
    mean_arrivals = shows.sum(axis=1).mean()
    expected = (1 - p).sum()
    assert abs(mean_arrivals - expected) < 3.0  # ยอมรับ noise เล็กน้อย


def test_find_optimal_overbook_returns_int_within_range():
    p = np.array([0.15] * 200)
    result = find_optimal_overbook(
        p_cancels=p, capacity=180, cost_empty=100.0, cost_overbook=250.0,
        o_max=50, n_sims=1000, seed=7,
    )
    assert isinstance(result["o_star"], (int, np.integer))
    assert 0 <= result["o_star"] <= 50
    assert result["expected_cost"] >= 0


def test_high_cancel_rate_favors_more_overbooking():
    # อัตรายกเลิกสูง -> ควร overbook มากกว่าอัตรายกเลิกต่ำ (เทียบ o_star)
    low_cancel = np.array([0.02] * 200)
    high_cancel = np.array([0.30] * 200)
    r_low = find_optimal_overbook(low_cancel, capacity=180, cost_empty=100.0, cost_overbook=250.0, o_max=60, n_sims=1500, seed=3)
    r_high = find_optimal_overbook(high_cancel, capacity=180, cost_empty=100.0, cost_overbook=250.0, o_max=60, n_sims=1500, seed=3)
    assert r_high["o_star"] >= r_low["o_star"]


def test_policy_no_overbook_is_zero():
    assert policy_no_overbook(300) == 0


def test_policy_fixed_pct():
    assert policy_fixed(300, fixed_pct=0.05) == 15


def test_policy_avg_rate_clips_between_0_and_1():
    assert policy_avg_rate(300, avg_cancel_rate=1.5) == 300
    assert policy_avg_rate(300, avg_cancel_rate=-0.2) == 0


def test_backtest_night_no_overbook_never_moves_customers_beyond_accepted():
    # ถ้า o=0 accepted_n = capacity เสมอ (หรือน้อยกว่าถ้า n_bookings ไม่พอ) -> moved_customers ต้องเป็น 0 เสมอ
    # เพราะ arrivals_actual <= accepted_n <= capacity
    rng = np.random.default_rng(0)
    n = 250
    night_df = pd.DataFrame({
        "hotel": ["City Hotel"] * n,
        "stay_date": [pd.Timestamp("2023-06-01")] * n,
        "adr": rng.uniform(80, 120, n),
        "is_canceled": rng.binomial(1, 0.15, n),
        "p_cancel": rng.uniform(0.05, 0.3, n),
    })
    result = backtest_night(night_df, capacity=200, cost_empty=100.0, cost_overbook=250.0, o=0)
    assert result["moved_customers"] == 0


def test_backtest_policies_and_aggregate_run_end_to_end():
    rng = np.random.default_rng(1)
    rows = []
    for stay_date in pd.date_range("2023-06-01", periods=5):
        n = rng.integers(180, 260)
        for hotel in ["City Hotel", "Resort Hotel"]:
            for _ in range(n):
                rows.append({
                    "hotel": hotel,
                    "stay_date": stay_date,
                    "adr": rng.uniform(70, 150),
                    "is_canceled": rng.binomial(1, 0.18),
                    "p_cancel": rng.uniform(0.05, 0.35),
                })
    df = pd.DataFrame(rows)
    cfg = load_config()
    bt = backtest_policies(df, cfg, k=2.0, policies=["no_overbook", "fixed", "model"], avg_cancel_rate_lookup={"__default__": 0.15})
    assert set(bt["policy"].unique()) == {"no_overbook", "fixed", "model"}
    summary = aggregate_results(bt)
    assert "moved_per_1000_nights" in summary.columns
    assert len(summary) == 3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
