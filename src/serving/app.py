import os
import sys
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
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
        # ตั้งใจจับทุก error: ถ้าโหลดโมเดลไม่ได้ API ยังเปิดได้ แล้ว /health จะบอกว่า model_not_loaded
    except Exception as error:  # noqa: BLE001
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
# ---------- เก็บสถิติสำหรับ /metrics ----------
# ไม่นับ path พวกนี้ เพราะไม่ใช่งานจริงของ API
SKIP_PATHS = ["/metrics", "/docs", "/openapi.json", "/favicon.ico"]
MAX_LATENCIES = 1000  # เก็บเวลาตอบล่าสุดไม่เกิน 1000 ค่าต่อ path

counters = {
    "total_requests": 0,
    "client_errors_4xx": 0,  # ผู้ใช้ส่งข้อมูลผิด เช่น 422 (ไม่ใช่ความผิดของระบบ)
    "server_errors_5xx": 0,  # ระบบพัง (ใช้คิด error rate ตาม SLO)
}
requests_by_path = {}
latencies_by_path = {}


def record_request(path, status_code, latency_ms):
    """บันทึก 1 request ลงสถิติ"""
    counters["total_requests"] = counters["total_requests"] + 1
    if 400 <= status_code < 500:
        counters["client_errors_4xx"] = counters["client_errors_4xx"] + 1
    if status_code >= 500:
        counters["server_errors_5xx"] = counters["server_errors_5xx"] + 1

    if path not in requests_by_path:
        requests_by_path[path] = 0
        latencies_by_path[path] = []
    requests_by_path[path] = requests_by_path[path] + 1

    latencies_by_path[path].append(latency_ms)
    # ถ้าเก็บเกินจำนวนที่กำหนด ให้ทิ้งค่าเก่าสุด
    if len(latencies_by_path[path]) > MAX_LATENCIES:
        latencies_by_path[path].pop(0)


@app.middleware("http")
async def measure_request(request: Request, call_next):
    """จับเวลาทุก request (ทำงานอัตโนมัติก่อน/หลังทุก endpoint)"""
    path = request.url.path
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        # ระบบพัง: นับเป็น error 500 แล้วส่ง error ต่อไปตามปกติ
        latency_ms = (time.perf_counter() - start) * 1000
        if path not in SKIP_PATHS:
            record_request(path, 500, latency_ms)
        raise

    latency_ms = (time.perf_counter() - start) * 1000
    if path not in SKIP_PATHS:
        record_request(path, response.status_code, latency_ms)

    # แนบเวลาตอบไว้ใน header ด้วย ดูได้ในหน้า /docs
    response.headers["X-Latency-ms"] = f"{latency_ms:.2f}"
    return response


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


SLO_P95_MS = 100  # ตามสโคป: p95 ≤ 100 ms
SLO_ERROR_RATE = 0.01  # ตามสโคป: error rate ≤ 1%


def summarize_latency(values):
    """สรุปเวลาตอบเป็น p50, p95, max (หน่วย ms)"""
    if len(values) == 0:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    return {
        "count": len(values),
        "p50_ms": round(float(np.percentile(values, 50)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
        "max_ms": round(max(values), 2),
    }


@app.get("/metrics")
def metrics():
    """สถิติการทำงานของ API เทียบกับ SLO"""
    total = counters["total_requests"]
    if total > 0:
        error_rate = counters["server_errors_5xx"] / total
    else:
        error_rate = 0.0

    latency = {}
    for path, values in latencies_by_path.items():
        latency[path] = summarize_latency(values)

    # เช็ก SLO ด้วย p95 ของ /predict (endpoint หลักที่ใช้ตอนจอง)
    predict_p95 = None
    if "/predict" in latency:
        predict_p95 = latency["/predict"]["p95_ms"]

    return {
        "model_uri": MODEL_URI,
        "total_requests": total,
        "client_errors_4xx": counters["client_errors_4xx"],
        "server_errors_5xx": counters["server_errors_5xx"],
        "error_rate": round(error_rate, 4),
        "requests_by_path": requests_by_path,
        "latency": latency,
        "slo": {
            "predict_p95_target_ms": SLO_P95_MS,
            "predict_p95_ok": predict_p95 is not None and predict_p95 <= SLO_P95_MS,
            "error_rate_target": SLO_ERROR_RATE,
            "error_rate_ok": error_rate <= SLO_ERROR_RATE,
        },
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


MAX_BATCH_SIZE = 1000  # กันไม่ให้ส่งมาทีเดียวเยอะเกินจน API ค้าง


@app.post("/predict-batch")
def predict_batch(bookings: list[Booking]):
    """รับหลายรายการจอง แล้วคืนความน่าจะเป็นยกเลิกของแต่ละรายการ"""
    if model is None:
        raise HTTPException(status_code=503, detail="ยังโหลดโมเดลไม่ได้")

    # เช็กจำนวนรายการ
    if len(bookings) == 0:
        raise HTTPException(status_code=422, detail="ต้องส่งมาอย่างน้อย 1 รายการ")
    if len(bookings) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"ส่งได้ไม่เกิน {MAX_BATCH_SIZE} รายการต่อครั้ง (ส่งมา {len(bookings)} รายการ)",
        )

    # แปลงทุกรายการเป็น DataFrame เดียว (1 รายการ = 1 แถว)
    rows = []
    for booking in bookings:
        one_row_df = booking_to_dataframe(booking)
        rows.append(one_row_df)
    df = pd.concat(rows, ignore_index=True)

    # ทำนายทั้งก้อนในครั้งเดียว
    X = build_features(df)
    probabilities = model.predict_proba(X)[:, 1]

    # จัดผลลัพธ์ทีละรายการ
    results = []
    for index in range(len(probabilities)):
        p_cancel = float(probabilities[index])
        results.append(
            {
                "index": index,
                "p_cancel": round(p_cancel, 4),
                "will_cancel": p_cancel >= THRESHOLD,
            }
        )

    return {
        "count": len(results),
        "results": results,
        "model_uri": MODEL_URI,
    }
