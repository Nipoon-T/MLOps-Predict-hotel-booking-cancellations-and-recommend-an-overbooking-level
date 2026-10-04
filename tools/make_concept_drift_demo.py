from pathlib import Path

import pandas as pd

INPUT = Path("data/production_weeks/week_01.csv")
OUTPUT = Path(
    "data/production_weeks/week_01_concept_drift.csv"
)


def main():
    df = pd.read_csv(INPUT)

    # สำเนาข้อมูลเดิม ห้ามแก้ week_01.csv
    drift = df.copy()

    # จำลอง concept drift โดยกลับ label
    # ของส่วนหนึ่งของข้อมูล
    n_flip = int(len(drift) * 0.40)

    drift.loc[: n_flip - 1, "is_canceled"] = (
        1 - drift.loc[: n_flip - 1, "is_canceled"]
    )

    drift.to_csv(
        OUTPUT,
        index=False,
    )

    print(f"Original rows: {len(df)}")
    print(f"Flipped labels: {n_flip}")
    print(f"Saved: {OUTPUT}")

    print()
    print("Original cancellation rate:")
    print(df["is_canceled"].mean())

    print("Concept-drift cancellation rate:")
    print(drift["is_canceled"].mean())


if __name__ == "__main__":
    main()