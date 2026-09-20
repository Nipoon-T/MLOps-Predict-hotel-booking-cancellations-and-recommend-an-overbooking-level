import pandas as pd

DATA_PATH = "data/raw/hotel_bookings.csv"


print("=" * 60)
print("HOTEL BOOKING DATASET")
print("=" * 60)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Basic information
print(f"\nRows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

print("\nColumn names:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i:02d}. {column}")

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())