from pathlib import Path
import hashlib


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
INPUT_FILE = RAW_DIR / "hotel_bookings.csv"

# SHA-256 ของ hotel_bookings.csv ที่ทีมตรวจสอบแล้ว
EXPECTED_SHA256 = (
    "7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06"
)


# ============================================================
# FUNCTIONS
# ============================================================

def calculate_sha256(file_path: Path) -> str:
    """คำนวณค่า SHA-256 ของไฟล์"""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def download_data():
    """
    ตรวจสอบ dataset ที่ดาวน์โหลดจาก Google Drive ของทีม

    Dataset ต้องถูกวางไว้ที่:
        data/raw/hotel_bookings.csv

    ฟังก์ชันนี้จะไม่ดาวน์โหลดจาก Kaggle และไม่ต้องใช้ Kaggle API
    """

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Dataset setup")
    print("=" * 60)

    print(f"Expected file: {INPUT_FILE}")

    # ========================================================
    # Check dataset exists
    # ========================================================

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "\nไม่พบ dataset\n\n"
            f"กรุณาดาวน์โหลด hotel_bookings.csv จาก Google Drive "
            "ของทีม และวางไว้ที่:\n"
            f"  {INPUT_FILE}\n"
        )

    print("Dataset found.")

    # ========================================================
    # SHA-256 verification
    # ========================================================

    print("\nChecking SHA-256...")

    actual_sha256 = calculate_sha256(INPUT_FILE)

    print(f"Expected SHA-256: {EXPECTED_SHA256}")
    print(f"Actual SHA-256:   {actual_sha256}")

    if actual_sha256.lower() != EXPECTED_SHA256.lower():
        raise ValueError(
            "\nSHA-256 ไม่ตรงกัน\n"
            "ไฟล์ dataset อาจไม่ใช่ไฟล์เดียวกับที่ทีมกำหนด "
            "หรือไฟล์อาจเสียหาย\n"
        )

    print("SHA-256 verification passed.")

    # ========================================================
    # Final result
    # ========================================================

    print("\nDataset is ready.")
    print(f"Location: {INPUT_FILE}")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    download_data()