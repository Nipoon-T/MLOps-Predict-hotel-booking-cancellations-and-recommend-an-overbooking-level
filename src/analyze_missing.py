import os
import sys
import pandas as pd

# เพิ่ม project root เข้า Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


DATA_PATH = "data/raw/hotel_bookings.csv"


def analyze_missing(df):
    columns = [
        "company",
        "agent",
        "country",
        "children",
    ]

    for column in columns:
        missing = df[column].isna()

        print("\n" + "=" * 60)
        print(f"COLUMN: {column}")
        print("=" * 60)

        print(f"Missing rows : {missing.sum():,}")
        print(f"Non-missing  : {(~missing).sum():,}")

        print("\nCancellation rate:")
        print(
            df.groupby(missing)["is_canceled"]
            .mean()
            .rename({
                False: "Not Missing",
                True: "Missing"
            })
        )


def main():
    print("=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    analyze_missing(df)


if __name__ == "__main__":
    main()