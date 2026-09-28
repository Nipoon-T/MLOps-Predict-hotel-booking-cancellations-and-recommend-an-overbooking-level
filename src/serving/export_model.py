"""
export_model.py — คัดลอกโมเดลจาก MLflow Registry ออกมาเป็นโฟลเดอร์ธรรมดา สำหรับใช้ใน Docker

ทำไมต้องมีไฟล์นี้:
    mlflow.db เก็บที่อยู่ของไฟล์โมเดลเป็น path เต็มของเครื่องที่เทรน (เช่น C:\\Users\\...\\mlruns)
    พอเอาไปเปิดใน container (Linux) path นั้นไม่มีอยู่จริง เลยโหลดโมเดลไม่ได้
    จึงคัดลอกโมเดลของ alias ที่ต้องการออกมาไว้ที่ serving_model/ ก่อน
    แล้วให้ container โหลดจากโฟลเดอร์นี้แทน

วิธีรัน (จาก root ของ repo):
    python -m src.serving.export_model
    python -m src.serving.export_model champion    (ถ้าจะใช้ alias อื่น)
"""

import shutil
import sys
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "hotel-cancellation-classifier"
EXPORT_DIR = ROOT / "serving_model"


def main():
    # ถ้าพิมพ์ชื่อ alias มาต่อท้ายคำสั่ง ใช้ alias นั้น ไม่งั้นใช้ candidate
    if len(sys.argv) > 1:
        alias = sys.argv[1]
    else:
        alias = "candidate"

    mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, alias)
    print(f"โมเดลที่จะ export: {MODEL_NAME} alias={alias} -> version {version.version}")

    # ลบของเก่าทิ้งก่อน จะได้ไม่มีไฟล์ของโมเดลเวอร์ชันเก่าปนอยู่
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)

    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{MODEL_NAME}@{alias}",
        dst_path=str(EXPORT_DIR),
    )

    # จดไว้ว่าโฟลเดอร์นี้คือโมเดลเวอร์ชันไหน (ดูย้อนหลังได้ว่า container ใช้โมเดลตัวไหน)
    info_file = EXPORT_DIR / "EXPORTED_FROM.txt"
    info_file.write_text(
        f"model={MODEL_NAME}\nalias={alias}\nversion={version.version}\n",
        encoding="utf-8",
    )

    print(f"[OK] export เสร็จ: {EXPORT_DIR}")


if __name__ == "__main__":
    main()