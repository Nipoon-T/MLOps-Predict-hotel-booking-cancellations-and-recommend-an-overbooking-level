from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_FILE = PROCESSED_DIR / "train.csv"
VALIDATION_FILE = PROCESSED_DIR / "validation.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"
PRODUCTION_FILE = PROCESSED_DIR / "production.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VALIDATION_FILE)
    test = pd.read_csv(TEST_FILE)
    production = pd.read_csv(PRODUCTION_FILE)

    return {
        "train": train,
        "validation": validation,
        "test": test,
        "production": production,
    }


# ============================================================
# 1. CANCELLATION RATE BY HOTEL / MONTH
#    ใช้ TRAIN เท่านั้น
# ============================================================

def cancellation_by_hotel_month(train):
    df = train.copy()

    df["arrival_date"] = pd.to_datetime(df["arrival_date"])

    result = (
        df.groupby(
            [
                df["arrival_date"].dt.to_period("M"),
                "hotel",
            ]
        )
        .agg(
            bookings=("is_canceled", "size"),
            cancellations=("is_canceled", "sum"),
            cancellation_rate=("is_canceled", "mean"),
        )
        .reset_index()
    )

    result["arrival_month"] = result["arrival_date"].astype(str)

    result = result[
        [
            "arrival_month",
            "hotel",
            "bookings",
            "cancellations",
            "cancellation_rate",
        ]
    ]

    result = result.sort_values(
        ["arrival_month", "hotel"]
    )

    return result


# ============================================================
# 2. COMPARISON ACROSS SPLITS
# ============================================================

def split_comparison(datasets):
    rows = []

    for split_name, df in datasets.items():

        hotel_distribution = (
            df["hotel"]
            .value_counts(normalize=True)
            .to_dict()
        )

        deposit_distribution = (
            df["deposit_type"]
            .value_counts(normalize=True)
            .to_dict()
        )

        rows.append(
            {
                "split": split_name,
                "rows": len(df),
                "hotel_City_Hotel_pct": (
                    hotel_distribution.get("City Hotel", 0) * 100
                ),
                "hotel_Resort_Hotel_pct": (
                    hotel_distribution.get("Resort Hotel", 0) * 100
                ),
                "deposit_No_Deposit_pct": (
                    deposit_distribution.get("No Deposit", 0) * 100
                ),
                "deposit_Non_Refund_pct": (
                    deposit_distribution.get("Non Refund", 0) * 100
                ),
                "deposit_Refundable_pct": (
                    deposit_distribution.get("Refundable", 0) * 100
                ),
                "lead_time_mean": df["lead_time"].mean(),
                "lead_time_median": df["lead_time"].median(),
                "cancellation_rate": df["is_canceled"].mean(),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# 3. WRITE MARKDOWN REPORT
# ============================================================

def write_report(
    datasets,
    cancellation_monthly,
    split_summary,
):
    report_file = REPORTS_DIR / "eda_report.md"

    lines = []

    lines.append("# Exploratory Data Analysis Report")
    lines.append("")
    lines.append(
        "This report is generated from the processed "
        "chronological data splits."
    )
    lines.append("")

    # --------------------------------------------------------
    # Dataset sizes
    # --------------------------------------------------------

    lines.append("## Dataset sizes")
    lines.append("")

    size_table = pd.DataFrame(
        [
            {
                "split": name,
                "rows": len(df),
                "cancellation_rate": df["is_canceled"].mean(),
            }
            for name, df in datasets.items()
        ]
    )

    lines.append(
        size_table.to_markdown(
            index=False,
            floatfmt=".4f",
        )
    )

    lines.append("")

    # --------------------------------------------------------
    # Cancellation rate by hotel / month
    # --------------------------------------------------------

    lines.append(
        "## Cancellation rate by hotel and month"
    )
    lines.append("")

    lines.append(
        "This table is calculated from the **train set only**."
    )
    lines.append("")

    monthly_display = cancellation_monthly.copy()

    monthly_display["cancellation_rate"] = (
        monthly_display["cancellation_rate"] * 100
    )

    lines.append(
        monthly_display.to_markdown(
            index=False,
            floatfmt=".2f",
        )
    )

    lines.append("")

    # --------------------------------------------------------
    # Split comparison
    # --------------------------------------------------------

    lines.append(
        "## Comparison across data splits"
    )
    lines.append("")

    lines.append(
        split_summary.to_markdown(
            index=False,
            floatfmt=".2f",
        )
    )

    lines.append("")

    # --------------------------------------------------------
    # Interpretation note
    # --------------------------------------------------------

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Hotel and deposit type are compared using "
        "their percentage distribution within each split."
    )
    lines.append(
        "- Lead time is reported using mean and median."
    )
    lines.append(
        "- Cancellation rate is the mean of `is_canceled`."
    )
    lines.append(
        "- The hotel/month cancellation analysis uses "
        "the train set only."
    )
    lines.append("")

    report_file.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return report_file


# ============================================================
# MAIN
# ============================================================

def main():

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    datasets = load_data()

    train = datasets["train"]

    cancellation_monthly = (
        cancellation_by_hotel_month(train)
    )

    split_summary = split_comparison(
        datasets
    )

    # Save CSV evidence
    cancellation_monthly.to_csv(
        REPORTS_DIR / "cancellation_by_hotel_month.csv",
        index=False,
    )

    split_summary.to_csv(
        REPORTS_DIR / "split_comparison.csv",
        index=False,
    )

    # Save Markdown report
    report_file = write_report(
        datasets,
        cancellation_monthly,
        split_summary,
    )

    print("EDA report generated successfully.")
    print(f"Report: {report_file}")
    print(
        "CSV: reports/cancellation_by_hotel_month.csv"
    )
    print(
        "CSV: reports/split_comparison.csv"
    )


if __name__ == "__main__":
    main()