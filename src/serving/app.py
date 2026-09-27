import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.modeling.features import build_features  
from src.serving.schemas import Booking  

# ---------- ตั้งค่าโมเดล ----------
MODEL_NAME = "hotel-cancellation-classifier"
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "candidate")  # ภายหลังเปลี่ยนเป็น champion ได้
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
THRESHOLD = 0.5  # ใช้แค่ตัดสิน will_cancel ส่วน overbooking ใช้ความน่าจะเป็นตรงๆ

mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())


def load_model():
    """โหลดโมเดลครั้งเดียวตอนเปิด API (ไม่โหลดใหม่ทุก request เพราะจะช้า)"""
    try:
        loaded = mlflow.sklearn.load_model(MODEL_URI)
        print(f"[OK] โหลดโมเดลสำเร็จ: {MODEL_URI}")
        return loaded
    except Exception as error:
        print(f"[ERROR] โหลดโมเดลไม่ได้: {error}")
        return None


def load_expected_columns():
    path = ROOT / "data" / "processed" / "test.csv"
    columns = list(pd.read_csv(path, nrows=0).columns)

    # พิมพ์ให้ดูว่าคอลัมน์ไหนที่ API ไม่ได้รับมา (จะถูกเติมเป็นค่าว่าง)
    api_fields = set(Booking.model_fields.keys())
    not_from_api = [c for c in columns if c not in api_fields]
    print("[INFO] คอลัมน์ที่ API ไม่รับมา (เติมเป็นค่าว่าง):", not_from_api)
    return columns


model = load_model()
EXPECTED_COLUMNS = load_expected_columns()


def booking_to_dataframe(booking):
    """แปลง 1 booking (จาก JSON) เป็น DataFrame 1 แถว ที่มีคอลัมน์ครบ"""
    data = booking.model_dump()
    row = {}
    for column in EXPECTED_COLUMNS:
        # คอลัมน์ที่ไม่มีใน request เช่น is_canceled, reservation_status
        # เติม None ไว้ก่อน แล้ว build_features() จะตัดทิ้งเอง
        row[column] = data.get(column, None)
    return pd.DataFrame([row])


# ---------- สร้าง API ----------
app = FastAPI(title="Hotel Cancellation API", version="0.1.0")


@app.exception_handler(RequestValidationError)
def handle_bad_input(request: Request, error: RequestValidationError):
    """ทำข้อความ 422 ให้อ่านง่าย: บอกว่าช่องไหนผิด และผิดเพราะอะไร"""
    problems = []
    for item in error.errors():
        field_name = ".".join(str(part) for part in item["loc"] if part != "body")
        problems.append({"field": field_name or "(ทั้งรายการ)", "message": item["msg"]})
    return JSONResponse(
        status_code=422,
        content={"error": "ข้อมูลการจองไม่ถูกต้อง", "problems": problems},
    )


@app.get("/health")
def health():
    """เช็กว่า API ยังทำงาน และโหลดโมเดลได้หรือยัง"""
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_uri": MODEL_URI,
    }


@app.post("/predict")
def predict(booking: Booking):
    """รับ 1 รายการจอง แล้วคืนความน่าจะเป็นที่จะถูกยกเลิก (calibrate แล้ว)"""
    if model is None:
        raise HTTPException(status_code=503, detail="ยังโหลดโมเดลไม่ได้")

    df = booking_to_dataframe(booking)
    X = build_features(df)
    p_cancel = float(model.predict_proba(X)[0, 1])

    return {
        "p_cancel": round(p_cancel, 4),
        "will_cancel": p_cancel >= THRESHOLD,
        "model_uri": MODEL_URI,
    }