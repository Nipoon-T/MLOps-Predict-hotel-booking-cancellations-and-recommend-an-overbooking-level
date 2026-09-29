"""จัดการ alias ของโมเดลใน MLflow Registry (บทบาทที่ 5)

alias ที่ใช้:
  candidate          โมเดลใหม่ล่าสุดจาก calibrate.py (ยังไม่ได้ขึ้นใช้งาน)
  champion           โมเดลที่ API ใช้อยู่
  previous_champion  champion ตัวก่อนหน้า เก็บไว้สำหรับ rollback

วิธีใช้ (จาก root ของ repo):
  python -m pipeline.registry status             ดูว่าแต่ละ version มี alias อะไร
  python -m pipeline.registry promote            candidate -> champion
  python -m pipeline.registry promote <alias>    alias อื่น -> champion
  python -m pipeline.registry rollback           previous_champion -> champion

คำสั่งนี้แก้แค่ alias ใน registry ไม่ได้เปลี่ยนโมเดลที่ API ใช้
ถ้าจะให้ API เปลี่ยนด้วย ใช้ python -m pipeline.run --promote / --rollback
"""

import sys
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "hotel-cancellation-classifier"
CHAMPION = "champion"
PREVIOUS = "previous_champion"


class RegistryError(RuntimeError):
    """ทำงานกับ registry ไม่ได้ เช่น ไม่มี alias ที่ต้องการ"""


def _client() -> mlflow.MlflowClient:
    # ใช้ mlflow.db ตัวเดียวกับ train.py / calibrate.py / export_model.py
    mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
    return mlflow.MlflowClient()


def _version_of(client: mlflow.MlflowClient, alias: str) -> str | None:
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias).version
    except MlflowException:
        return None


def status() -> dict[str, str | None]:
    client = _client()
    result = {a: _version_of(client, a) for a in ("candidate", CHAMPION, PREVIOUS)}
    for alias, version in result.items():
        print(f"  {alias:<18} -> version {version if version else '-'}")
    return result


def promote(source_alias: str = "candidate") -> str:
    """ย้าย champion ไปที่ version ของ source_alias และเก็บ champion เดิมไว้ย้อนกลับ"""
    client = _client()
    new_version = _version_of(client, source_alias)
    if new_version is None:
        raise RegistryError(f"ไม่พบ alias '{source_alias}' ใน {MODEL_NAME}")

    old_version = _version_of(client, CHAMPION)
    if old_version == new_version:
        print(f"version {new_version} เป็น champion อยู่แล้ว ไม่ต้องเปลี่ยน")
        return new_version

    if old_version is not None:
        client.set_registered_model_alias(MODEL_NAME, PREVIOUS, old_version)
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, new_version)
    print(f"promote: {source_alias} (version {new_version}) -> champion "
          f"(champion เดิม: {old_version if old_version else '-'})")
    return new_version


def rollback() -> str:
    """สลับ champion กับ previous_champion (รันซ้ำอีกครั้งจะย้อนกลับได้)"""
    client = _client()
    prev_version = _version_of(client, PREVIOUS)
    if prev_version is None:
        raise RegistryError("ไม่มี previous_champion ให้ย้อนกลับ (ยังไม่เคย promote ทับ)")

    cur_version = _version_of(client, CHAMPION)
    client.set_registered_model_alias(MODEL_NAME, CHAMPION, prev_version)
    if cur_version is not None:
        client.set_registered_model_alias(MODEL_NAME, PREVIOUS, cur_version)
    print(f"rollback: champion version {cur_version} -> version {prev_version}")
    return prev_version


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    try:
        if cmd == "status":
            status()
        elif cmd == "promote":
            promote(sys.argv[2] if len(sys.argv) > 2 else "candidate")
        elif cmd == "rollback":
            rollback()
        else:
            print(__doc__)
            return 2
    except RegistryError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
