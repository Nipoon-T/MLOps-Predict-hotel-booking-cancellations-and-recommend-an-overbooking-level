from __future__ import annotations

import json
import sys
from pathlib import Path

import mlflow
import nannyml as nml
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Allow importing the project's existing feature/metric logic.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.features import build_features, get_target  # noqa: E402
from src.modeling.train import compute_metrics  # noqa: E402


REFERENCE_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
CONCEPT_BASELINE_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
REPORT_DIR = PROJECT_ROOT / "reports" / "monitoring" / "drift"

MODEL_NAME = "hotel-cancellation-classifier"
MODEL_ALIAS = "champion"

PRIOR_SHIFT_THRESHOLD = 0.05
CONCEPT_DRIFT_PR_AUC_DROP = 0.03
CONCEPT_DRIFT_MAX_ECE = 0.05

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
    parts = path.stem.split("_")
    try:
        return int(parts[1])
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

    except Exception as exc:  # noqa: BLE001
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


def load_champion():
    """Load the current production/champion model from MLflow."""
    mlflow.set_tracking_uri(
        "sqlite:///" + (PROJECT_ROOT / "mlflow.db").as_posix()
    )

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as exc:  # noqa: BLE001
        print(
            f"Warning: unable to load champion model "
            f"({model_uri}): {exc}"
        )
        return None

    return model


def run_concept_drift(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
):
    """
    Monitor model-performance drift after labels are available.

    Baseline:
        Champion performance on the original held-out test set.

    Current:
        The same champion evaluated on the completed production week.

    Alert rules:
        - PR-AUC drops by more than 0.03 from the baseline.
        - ECE exceeds 0.05.

    This is a label-based performance-drift monitor. It is used as the
    project's operational proxy for concept drift because labels are only
    available after the arrival date has passed.
    """
    model = load_champion()

    if model is None:
        return {
            "available": False,
            "triggered": False,
            "reason": "champion model unavailable; concept drift skipped",
        }

    baseline_x = build_features(baseline)
    baseline_y = get_target(baseline)

    current_x = build_features(current)
    current_y = get_target(current)

    baseline_prob = model.predict_proba(baseline_x)[:, 1]
    current_prob = model.predict_proba(current_x)[:, 1]

    baseline_metrics = compute_metrics(
        baseline_y,
        baseline_prob,
    )
    current_metrics = compute_metrics(
        current_y,
        current_prob,
    )

    pr_auc_drop = baseline_metrics["pr_auc"] - current_metrics["pr_auc"]

    pr_auc_triggered = (
        pr_auc_drop > CONCEPT_DRIFT_PR_AUC_DROP
    )
    ece_triggered = (
        current_metrics["ece"] > CONCEPT_DRIFT_MAX_ECE
    )

    reasons = []

    if pr_auc_triggered:
        reasons.append(
            "Concept/performance drift: "
            f"PR-AUC dropped by {pr_auc_drop:.4f} "
            f"(>{CONCEPT_DRIFT_PR_AUC_DROP:.2f})"
        )

    if ece_triggered:
        reasons.append(
            "Concept/performance drift: "
            f"ECE {current_metrics['ece']:.4f} "
            f">{CONCEPT_DRIFT_MAX_ECE:.2f}"
        )

    return {
        "available": True,
        "triggered": bool(pr_auc_triggered or ece_triggered),
        "model": {
            "name": MODEL_NAME,
            "alias": MODEL_ALIAS,
        },
        "baseline": {
            "dataset": str(CONCEPT_BASELINE_PATH),
            "rows": len(baseline),
            "pr_auc": float(baseline_metrics["pr_auc"]),
            "ece": float(baseline_metrics["ece"]),
        },
        "current": {
            "rows": len(current),
            "pr_auc": float(current_metrics["pr_auc"]),
            "ece": float(current_metrics["ece"]),
        },
        "thresholds": {
            "max_pr_auc_drop": CONCEPT_DRIFT_PR_AUC_DROP,
            "max_ece": CONCEPT_DRIFT_MAX_ECE,
        },
        "pr_auc_drop": float(pr_auc_drop),
        "pr_auc_triggered": bool(pr_auc_triggered),
        "ece_triggered": bool(ece_triggered),
        "reasons": reasons,
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

    if not CONCEPT_BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Concept-drift baseline CSV not found: "
            f"{CONCEPT_BASELINE_PATH}"
        )

    week = get_week_number(week_csv)

    reference = load_data(REFERENCE_PATH)
    baseline = load_data(CONCEPT_BASELINE_PATH)
    current = load_data(week_csv)

    required_columns = FEATURES + ["is_canceled"]

    missing = [
        column
        for column in required_columns
        if column not in reference.columns
        or column not in current.columns
        or column not in baseline.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(set(missing)))
        )

    print(f"Checking drift for week {week:02d}")
    print(f"Reference rows: {len(reference)}")
    print(f"Current rows: {len(current)}")
    print(f"Concept baseline rows: {len(baseline)}")

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
    # Concept / performance drift
    # -------------------------

    concept_drift = run_concept_drift(
        baseline,
        current,
    )

    if concept_drift["available"]:
        print(
            "Concept baseline PR-AUC: "
            f"{concept_drift['baseline']['pr_auc']:.4f}"
        )
        print(
            "Current week PR-AUC: "
            f"{concept_drift['current']['pr_auc']:.4f}"
        )
        print(
            "PR-AUC drop: "
            f"{concept_drift['pr_auc_drop']:.4f}"
        )
        print(
            "Current week ECE: "
            f"{concept_drift['current']['ece']:.4f}"
        )
        print(
            "Concept drift triggered: "
            f"{concept_drift['triggered']}"
        )
    else:
        print(
            "Concept drift: SKIPPED - "
            f"{concept_drift['reason']}"
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

    if concept_drift["triggered"]:
        reasons.extend(concept_drift["reasons"])

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
            "concept_drift": concept_drift,
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
