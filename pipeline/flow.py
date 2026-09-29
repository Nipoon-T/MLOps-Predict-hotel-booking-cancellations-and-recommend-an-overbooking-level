"""Prefect flow ของทั้งระบบ (บทบาทที่ 5: Pipeline + Docker)

ครอบสคริปต์เดิมของทุกคนโดยไม่แก้โค้ดเพื่อน แต่ละขั้นรันเป็น process แยกด้วย
Python ตัวเดียวกับที่รัน flow นี้ ถ้าขั้นไหนจบด้วย exit code ไม่เป็น 0 ให้หยุดทั้ง flow
ยกเว้น validate ข้อมูลดิบ ซึ่งตั้งใจให้ fail อยู่แล้ว (adr ติดลบ 1 แถว, ผู้เข้าพักเป็นศูนย์ 180 แถว)

ลำดับ:
  check raw file → download_data (SHA-256) → validate raw (บันทึกผลอย่างเดียว)
  → clean_data → validate clean (GATE) → split_data → create_bad_data → pytest
  → train → calibrate → export_model → docker compose up → รอ /health = ok
"""

import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from prefect import flow, get_run_logger, task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "hotel_bookings.csv"
CLEAN_FILE = PROJECT_ROOT / "data" / "interim" / "hotel_bookings_clean.csv"
BAD_FILE = PROJECT_ROOT / "data" / "bad" / "hotel_bookings_bad.csv"
HEALTH_URL = "http://127.0.0.1:8000/health"

# สคริปต์ของทีมพิมพ์ภาษาไทยและ emoji บน Windows ต้องบังคับ UTF-8 ไม่งั้นพังตอนพิมพ์
CHILD_ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


class StepFailed(RuntimeError):
    """ขั้นใดขั้นหนึ่งจบด้วย exit code ไม่เป็น 0"""


def _run(cmd: list[str], name: str, allow_fail: bool = False) -> int:
    logger = get_run_logger()
    logger.info("▶ %s: %s", name, " ".join(cmd))
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=CHILD_ENV, check=False)
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        if allow_fail:
            logger.warning("%s จบด้วย exit code %s (ยอมให้ผ่าน) ใช้เวลา %.1f s",
                           name, result.returncode, elapsed)
        else:
            raise StepFailed(f"{name} ล้มเหลว (exit code {result.returncode}) หยุด pipeline")
    else:
        logger.info("✔ %s เสร็จใน %.1f s", name, elapsed)
    return result.returncode


def _py(*args: str) -> list[str]:
    return [sys.executable, *args]


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

@task(name="check-raw-file")
def check_raw_file() -> None:
    # ข้อมูลต้องโหลดเองจาก Google Drive ของทีม (ตัดสินใจร่วมกับบทบาทที่ 1)
    if not RAW_FILE.exists():
        raise StepFailed(
            f"ไม่พบ {RAW_FILE}\n"
            "ดาวน์โหลด hotel_bookings.csv จาก Google Drive ของทีม (ลิงก์ใน README) "
            "แล้ววางไว้ที่ data/raw/hotel_bookings.csv ก่อนรันใหม่"
        )


@task(name="verify-checksum")
def verify_checksum() -> None:
    _run(_py("src/download_data.py"), "download_data (SHA-256)")


@task(name="validate-raw")
def validate_raw() -> int:
    # ข้อมูลดิบคาดว่าจะไม่ผ่าน (ดู README) บันทึกผลอย่างเดียว ไม่หยุด flow
    return _run(_py("src/validate.py", str(RAW_FILE)), "validate raw", allow_fail=True)


@task(name="clean-data")
def clean_data() -> None:
    # clean_data.py หยุดเองถ้าลบข้อมูลเกิน 1%
    _run(_py("src/clean_data.py"), "clean_data")


@task(name="validate-clean-gate")
def validate_clean_gate(path: Path = CLEAN_FILE) -> None:
    # GATE จริง: ไม่ผ่านต้องหยุดทั้ง pipeline
    _run(_py("src/validate.py", str(path), "--allow-derived"), "validate clean (gate)")


@task(name="split-data")
def split_data() -> None:
    _run(_py("src/split_data.py"), "split_data")


@task(name="create-bad-data")
def create_bad_data() -> None:
    # test_validate.py ต้องใช้ data/bad/hotel_bookings_bad.csv
    _run(_py("src/create_bad_data.py"), "create_bad_data")


@task(name="pytest")
def run_tests() -> None:
    _run(_py("-m", "pytest", "-q"), "pytest")


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

@task(name="train")
def train() -> None:
    _run(_py("src/modeling/train.py"), "train (5 experiments)")


@task(name="calibrate")
def calibrate() -> None:
    # calibrate + register เป็น alias candidate
    _run(_py("src/modeling/calibrate.py"), "calibrate + register")


@task(name="export-model")
def export_model(alias: str) -> None:
    # container อ่าน mlflow.db ตรงๆ ไม่ได้ (path เป็นของ Windows) ต้อง export ก่อนเสมอ
    _run(_py("-m", "src.serving.export_model", alias), f"export_model ({alias})")


# ------------------------------------------------------------
# REGISTRY
# ------------------------------------------------------------

@task(name="promote-to-champion")
def promote_task(source_alias: str) -> str:
    from pipeline import registry
    try:
        return registry.promote(source_alias)
    except registry.RegistryError as exc:
        raise StepFailed(str(exc)) from exc


@task(name="rollback-champion")
def rollback_task() -> str:
    from pipeline import registry
    try:
        return registry.rollback()
    except registry.RegistryError as exc:
        raise StepFailed(str(exc)) from exc


# ------------------------------------------------------------
# SERVING
# ------------------------------------------------------------

@task(name="compose-up")
def compose_up(restart: bool = False) -> None:
    # up -d: ถ้า container ยังไม่เปิดจะเปิดให้ ถ้าเปิดอยู่แล้วจะไม่ทำอะไร
    _run(["docker", "compose", "up", "-d", "--build"], "docker compose up")
    if restart:
        # API โหลดโมเดลครั้งเดียวตอนเริ่ม เปลี่ยนโมเดลแล้วต้องรีสตาร์ท
        _run(["docker", "compose", "restart", "api"], "docker compose restart api")


@task(name="wait-health")
def wait_health(timeout_s: int = 120) -> None:
    logger = get_run_logger()
    deadline = time.time() + timeout_s
    last = "ยังเชื่อมต่อไม่ได้"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=5) as resp:
                body = resp.read().decode("utf-8")
            if '"ok"' in body:
                logger.info("✔ API พร้อม: %s", body)
                return
            last = body
        except OSError as exc:
            last = str(exc)
        time.sleep(5)
    raise StepFailed(f"API ไม่พร้อมภายใน {timeout_s} s (ล่าสุด: {last}) ลอง docker compose logs api")


@task(name="validate-bad-data-gate")
def validate_bad_gate(path: Path = BAD_FILE) -> None:
    # gate เดียวกับข้อมูลจริง แต่ป้อนข้อมูลเสีย ต้อง fail แล้วหยุด flow
    _run(_py("src/validate.py", str(path)), "validate bad data (gate)")


# ------------------------------------------------------------
# FLOWS
# ------------------------------------------------------------

@flow(name="data-pipeline")
def data_pipeline(run_tests_step: bool = True) -> None:
    check_raw_file()
    verify_checksum()
    validate_raw()
    clean_data()
    validate_clean_gate()
    split_data()
    create_bad_data()
    if run_tests_step:
        run_tests()


@flow(name="demo-bad-data")
def demo_bad_data() -> None:
    """สาธิต: ข้อมูลเสียต้องถูกหยุดที่ validation gate ก่อนถึง clean/train/deploy

    ไม่แตะข้อมูลจริง, โมเดล หรือ API ที่เปิดอยู่
    """
    logger = get_run_logger()
    if not BAD_FILE.exists():
        create_bad_data()
    validate_bad_gate()
    # ถ้ามาถึงบรรทัดนี้แปลว่า gate ปล่อยข้อมูลเสียผ่าน ซึ่งผิด
    logger.error("gate ปล่อยข้อมูลเสียผ่าน ต้องตรวจ validate.py")
    raise StepFailed("ข้อมูลเสียผ่าน gate ได้ ซึ่งไม่ควรเกิดขึ้น")


@flow(name="model-pipeline")
def model_pipeline() -> None:
    train()
    calibrate()


@flow(name="deploy-model")
def deploy_model(alias: str = "candidate", restart: bool = False) -> None:
    """export โมเดลของ alias ที่ระบุ แล้วเปิด/รีสตาร์ท API

    ใช้ซ้ำได้ทั้งตอนรันครั้งแรก, promote และ rollback (บทบาทที่ 6 เรียกต่อได้)
    """
    export_model(alias)
    compose_up(restart=restart)
    wait_health()


@flow(name="promote-and-deploy")
def promote_and_deploy(source_alias: str = "candidate") -> None:
    """promote alias ที่ระบุเป็น champion แล้วให้ API ใช้ champion ทันที"""
    promote_task(source_alias)
    deploy_model(alias="champion", restart=True)


@flow(name="rollback-and-deploy")
def rollback_and_deploy() -> None:
    """ย้อน champion กลับเป็นตัวก่อนหน้า แล้วให้ API ใช้ทันที"""
    rollback_task()
    deploy_model(alias="champion", restart=True)


@flow(name="hotel-full-pipeline")
def full_pipeline(
    alias: str = "candidate",
    skip_data: bool = False,
    skip_train: bool = False,
    skip_tests: bool = False,
    deploy: bool = True,
) -> None:
    if not skip_data:
        data_pipeline(run_tests_step=not skip_tests)
    if not skip_train:
        model_pipeline()
    if deploy:
        deploy_model(alias=alias)
