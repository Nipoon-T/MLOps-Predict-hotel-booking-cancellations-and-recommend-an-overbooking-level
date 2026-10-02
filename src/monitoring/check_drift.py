from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import nannyml as nml
from evidently import Report
from evidently.presets import DataDriftPreset


REFERENCE_PATH = Path("data/processed/train.csv")
REPORT_DIR = Path("reports/monitoring/drift")

PRIOR_SHIFT_THRESHOLD = 0.05

FEATURES = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "required_car_parking_spaces",
    "total_of_special_requests",
]


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)


def get_week_number(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except (ValueError, IndexError):
        return 0


def run_evidently(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    week: int,
) -> Path:

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = Report(
        [
            DataDriftPreset(),
        ]
    )

    result = report.run(
        reference_data=reference[FEATURES],
        current_data=current[FEATURES],
    )

    report_path = REPORT_DIR / f"evidently_week_{week:02d}.html"

    result.save_html(str(report_path))

    return report_path


def run_nannyml(
    reference: pd.DataFrame,
    current: pd.DataFrame,
):

    usable_features = [
        column
        for column in FEATURES
        if column in reference.columns
        and column in current.columns
    ]

    calculator = nml.UnivariateDriftCalculator(
        column_names=usable_features,
        chunk_size=max(100, min(len(current), 500)),
    )

    calculator.fit(reference[usable_features])

    results = calculator.calculate(
        current[usable_features]
    )

    return results, usable_features


def count_nanny_alerts(results) -> int:

    try:
        table = results.to_df()

        alert_columns = [
            column
            for column in table.columns
            if "alert" in str(column).lower()
        ]

        if not alert_columns:
            return 0

        count = 0

        for column in alert_columns:
            values = table[column].astype(str).str.lower()
            count += int(
                values.isin(["true", "alert"]).sum()
            )

        return count

    except Exception as exc:
        print(
            f"Warning: unable to count NannyML alerts: {exc}"
        )
        return 0


def calculate_prior_shift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
):

    reference_rate = reference["is_canceled"].mean()
    current_rate = current["is_canceled"].mean()

    gap = abs(current_rate - reference_rate)

    triggered = gap > PRIOR_SHIFT_THRESHOLD

    return {
        "reference_cancellation_rate": float(
            reference_rate
        ),
        "current_cancellation_rate": float(
            current_rate
        ),
        "absolute_gap": float(gap),
        "threshold": PRIOR_SHIFT_THRESHOLD,
        "triggered": bool(triggered),
    }


def main():

    if len(sys.argv) != 3:
        print(
            "Usage: "
            "python src/monitoring/check_drift.py "
            "<week_csv> <out_json>"
        )
        return 2

    week_csv = Path(sys.argv[1])
    output_json = Path(sys.argv[2])

    if not week_csv.exists():
        raise FileNotFoundError(
            f"Week CSV not found: {week_csv}"
        )

    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference CSV not found: {REFERENCE_PATH}"
        )

    week = get_week_number(week_csv)

    reference = load_data(REFERENCE_PATH)
    current = load_data(week_csv)

    required_columns = FEATURES + ["is_canceled"]

    missing = [
        column
        for column in required_columns
        if column not in reference.columns
        or column not in current.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(set(missing)))
        )

    print(f"Checking drift for week {week:02d}")
    print(f"Reference rows: {len(reference)}")
    print(f"Current rows: {len(current)}")

    # -------------------------
    # Evidently
    # -------------------------

    evidently_report = run_evidently(
        reference,
        current,
        week,
    )

    print(
        f"Evidently report: {evidently_report}"
    )

    # -------------------------
    # NannyML
    # -------------------------

    nanny_results, checked_features = run_nannyml(
        reference,
        current,
    )

    nanny_alerts = count_nanny_alerts(
        nanny_results
    )

    print(
        f"NannyML alerts: {nanny_alerts}"
    )

    # -------------------------
    # Prior shift
    # -------------------------

    prior_shift = calculate_prior_shift(
        reference,
        current,
    )

    print(
        "Reference cancellation rate: "
        f"{prior_shift['reference_cancellation_rate']:.4f}"
    )

    print(
        "Current cancellation rate: "
        f"{prior_shift['current_cancellation_rate']:.4f}"
    )

    print(
        "Cancellation-rate gap: "
        f"{prior_shift['absolute_gap']:.4f}"
    )

    # -------------------------
    # Retraining decision
    # -------------------------

    reasons = []

    if nanny_alerts > 0:
        reasons.append(
            f"NannyML detected "
            f"{nanny_alerts} drift alert(s)"
        )

    if prior_shift["triggered"]:
        reasons.append(
            "Cancellation-rate gap exceeded "
            f"{PRIOR_SHIFT_THRESHOLD:.0%}"
        )

    retrain = len(reasons) > 0

    result = {
        "week": week,
        "retrain": retrain,
        "reasons": reasons,
        "evidence": {
            "evidently_report": str(
                evidently_report
            ),
            "nannyml_alerts": nanny_alerts,
            "nannyml_features_checked": checked_features,
            "prior_shift": prior_shift,
        },
    }

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_json.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())