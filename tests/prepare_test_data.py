import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

fixture = ROOT / "tests" / "fixtures" / "hotel_bookings_sample.csv"
raw = ROOT / "data" / "raw" / "hotel_bookings.csv"

raw.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(fixture, raw)

subprocess.run(
    [sys.executable, "src/clean_data.py"],
    cwd=ROOT,
    check=True,
)

subprocess.run(
    [sys.executable, "src/split_data.py"],
    cwd=ROOT,
    check=True,
)

subprocess.run(
    [sys.executable, "src/create_bad_data.py"],
    cwd=ROOT,
    check=True,
)
