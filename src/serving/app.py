"""
app.py — FastAPI สำหรับทำนายการยกเลิกการจอง (คนที่ 4: Serving)

Endpoints:
    GET  /health                 เช็กว่า API และโมเดลพร้อม
    GET  /metrics                สถิติ request, error, latency เทียบ SLO
    POST /predict                ทำนาย 1 การจอง (real-time ตอนจอง)
    POST /predict-batch          ทำนายหลายการจองพร้อมกัน
    POST /recommend-overbooking  แนะนำจำนวนห้องที่ควรรับจองเกินของ 1 คืน

วิธีรัน (จากโฟลเดอร์ root ของ repo):
    uvicorn src.serving.app:app --reload --port 8000
แล้วเปิด http://127.0.0.1:8000/docs
"""

import asyncio.base_events
import os
import socket
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
from joblib import parallel_config

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from overbooking_decision.cost_config import load_config
from overbooking_decision.recommend import recommend_overbooking
from src.modeling.features import build_features
from src.serving.schemas import Booking, OverbookingRequest

# ---------- ตั้งค่าโมเดล ----------
MODEL_NAME = "hotel-cancellation-classifier"
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "candidate")  # ภายหลังเปลี่ยนเป็น champion ได้
# ถ้าตั้ง MODEL_DIR ไว้ (ใช้ใน Docker) จะโหลดโมเดลจากโฟลเดอร์ที่ export ไว้แทน Registry
MODEL_DIR = os.getenv("MODEL_DIR", "")
if MODEL_DIR != "":
    MODEL_URI = MODEL_DIR
else:
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
    """
    อ่านชื่อคอลัมน์จาก header ของ test.csv (อ่านแค่หัวตาราง ไม่อ่านข้อมูล)
    เพื่อให้ DataFrame ที่ส่งเข้า build_features() มีคอลัมน์ครบเหมือนตอนเทรน
    """
    path = ROOT / "data" / "processed" / "test.csv"
    if not path.exists():
        print(f"[WARN] ไม่พบ {path} จะทำนายไม่ได้ แต่ API ยังเปิดได้ (เช่นตอนรันเทสใน CI)")
        return []
    columns = list(pd.read_csv(path, nrows=0).columns)

    # พิมพ์ให้ดูว่าคอลัมน์ไหนที่ API ไม่ได้รับมา (จะถูกเติมเป็นค่าว่าง)
    api_fields = set(Booking.model_fields.keys())
    not_from_api = [c for c in columns if c not in api_fields]
    print("[INFO] คอลัมน์ที่ API ไม่รับมา (เติมเป็นค่าว่าง):", not_from_api)
    return columns


model = load_model()
EXPECTED_COLUMNS = load_expected_columns()

# สมมติฐานต้นทุนของบทบาทที่ 3 (capacity, k, n_sims) โหลดครั้งเดียวตอนเปิด API
COST_CONFIG = load_config()


def booking_to_dataframe(booking):
    """แปลง 1 booking (จาก JSON) เป็น DataFrame 1 แถว ที่มีคอลัมน์ครบ"""
    data = booking.model_dump()
    row = {}
    for column in EXPECTED_COLUMNS:
        # คอลัมน์ที่ไม่มีใน request เช่น is_canceled, reservation_status
        # เติม None ไว้ก่อน แล้ว build_features() จะตัดทิ้งเอง
        row[column] = data.get(column, None)
    return pd.DataFrame([row])


def bookings_to_dataframe(bookings):
    """แปลงหลาย booking เป็น DataFrame เดียว (1 รายการ = 1 แถว)"""
    rows = []
    for booking in bookings:
        one_row_df = booking_to_dataframe(booking)
        rows.append(one_row_df)
    return pd.concat(rows, ignore_index=True)


def predict_cancel_probabilities(X):
    """
    ทำนายความน่าจะเป็นยกเลิกของทุกแถวใน X

    โมเดลถูกเทรนด้วย n_jobs=-1 (ใช้ทุก core) ซึ่งดีตอนเทรนข้อมูลเยอะ
    แต่ตอน serving ที่ทำนายทีละไม่กี่แถว การแบ่งงานให้หลาย thread เสียเวลามากกว่าตัวงานจริง
    จึงบังคับให้ทำงานแบบ thread เดียว (ผลทำนายเหมือนเดิมทุกหลัก)
    """
    with parallel_config(backend="sequential"):
        probabilities = model.predict_proba(X)[:, 1]
    return probabilities


# ---------- แก้ความช้า 40 ms เมื่อรันหลาย worker บน Linux ----------
# Python จะเปิด TCP_NODELAY (ส่งข้อมูลทันที ไม่รอรวมก้อน) ให้เฉพาะ socket ที่ระบุ proto=TCP
# แต่ uvicorn --workers สร้าง socket กลางโดยไม่ระบุ proto ทำให้ worker ทุกตัวไม่ได้เปิด TCP_NODELAY
# ผลคือทุก request ช้าขึ้น ~40 ms บน Linux (เช่นใน Docker) วัดได้จริงด้วย Locust
# จึงเปลี่ยนให้เปิด TCP_NODELAY กับ TCP socket ทุกตัว
def set_tcp_nodelay_always(sock):
    if (
        sock.family in (socket.AF_INET, socket.AF_INET6)
        and sock.type == socket.SOCK_STREAM
    ):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


asyncio.base_events._set_nodelay = set_tcp_nodelay_always

# ---------- สร้าง API ----------
app = FastAPI(title="Hotel Cancellation API", version="0.2.0")

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


# ---------- SLO (ตกลงกับทีมแล้ว หลังวัดจริงด้วย Locust) ----------
SLO_P95_MS = 350  # p95 ของ /predict ≤ 350 ms ... (วัดใน Docker ได้ 330 ms ที่ 120 req/s)
SLO_P95_AT_LOAD_RPS = 120  # ... เมื่อโหลด 120 req/s
SLO_THROUGHPUT_RPS = 150  # รับได้อย่างน้อย 150 req/s ต่อ container
SLO_ERROR_RATE = 0.01  # error rate (5xx) ≤ 1%


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
    """
    สถิติการทำงานของ API เทียบกับ SLO

    หมายเหตุ: ถ้ารันหลาย worker ตัวเลขนี้เป็นของ worker ที่ตอบ request นี้เท่านั้น
    ส่วน throughput วัดด้วย Locust (ดู reports/load_test/)
    """
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
            "predict_p95_target_at_load_rps": SLO_P95_AT_LOAD_RPS,
            "predict_p95_ok": predict_p95 is not None and predict_p95 <= SLO_P95_MS,
            "throughput_target_rps": SLO_THROUGHPUT_RPS,
            "error_rate_target": SLO_ERROR_RATE,
            "error_rate_ok": error_rate <= SLO_ERROR_RATE,
        },
    }


@app.post("/predict")
async def predict(booking: Booking):
    """รับ 1 รายการจอง แล้วคืนความน่าจะเป็นที่จะถูกยกเลิก (calibrate แล้ว)"""
    if model is None:
        raise HTTPException(status_code=503, detail="ยังโหลดโมเดลไม่ได้")

    df = booking_to_dataframe(booking)
    X = build_features(df)
    p_cancel = float(predict_cancel_probabilities(X)[0])

    return {
        "p_cancel": round(p_cancel, 4),
        "will_cancel": p_cancel >= THRESHOLD,
        "model_uri": MODEL_URI,
    }


MAX_BATCH_SIZE = 1000  # กันไม่ให้ส่งมาทีเดียวเยอะเกินจน API ค้าง


@app.post("/predict-batch")
async def predict_batch(bookings: list[Booking]):
    """รับหลายรายการจอง แล้วคืนความน่าจะเป็นยกเลิกของแต่ละรายการ"""
    # เช็กข้อมูลที่ส่งมาก่อนเสมอ (ข้อมูลผิดต้องได้ 422 แม้ไม่มีโมเดล)
    if len(bookings) == 0:
        raise HTTPException(status_code=422, detail="ต้องส่งมาอย่างน้อย 1 รายการ")
    if len(bookings) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"ส่งได้ไม่เกิน {MAX_BATCH_SIZE} รายการต่อครั้ง (ส่งมา {len(bookings)} รายการ)",
        )

    if model is None:
        raise HTTPException(status_code=503, detail="ยังโหลดโมเดลไม่ได้")

    # ทำนายทั้งก้อนในครั้งเดียว
    df = bookings_to_dataframe(bookings)
    X = build_features(df)
    probabilities = predict_cancel_probabilities(X)

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


def get_lead_time(booking):
    """ใช้เป็นเกณฑ์เรียงการจอง"""
    return booking.lead_time


@app.post("/recommend-overbooking")
async def recommend_overbooking_for_night(request: OverbookingRequest):
    """
    แนะนำจำนวนห้องที่ควรรับจองเกิน (o_star) ของ 1 คืน

    ขั้นตอน:
      1. ทำนายความน่าจะเป็นยกเลิกของการจองใหม่ทุกรายการ (โมเดลตัวเดียวกับ /predict)
      2. ส่งเข้าชั้นตัดสินใจของบทบาทที่ 3 (Monte Carlo หาจุดที่ต้นทุนคาดหวังต่ำสุด)
    """
    # already_occupied ต้องไม่เกินจำนวนห้องทั้งหมดของโรงแรม
    capacity = COST_CONFIG.capacity(request.hotel)
    if request.already_occupied > capacity:
        raise HTTPException(
            status_code=422,
            detail=(
                f"already_occupied={request.already_occupied} "
                f"มากกว่าจำนวนห้องของ {request.hotel} ({capacity} ห้อง)"
            ),
        )

    if model is None:
        raise HTTPException(status_code=503, detail="ยังโหลดโมเดลไม่ได้")

    # เรียงการจองจาก lead_time มาก -> น้อย (จองก่อนได้สิทธิ์ก่อน ตามคำแนะนำของบทบาทที่ 3)
    sorted_bookings = sorted(request.bookings, key=get_lead_time, reverse=True)

    df = bookings_to_dataframe(sorted_bookings)
    X = build_features(df)
    probabilities = predict_cancel_probabilities(X)

    p_cancels = []
    for value in probabilities:
        p_cancels.append(float(value))

    # ถ้าไม่ได้ส่งราคาอ้างอิงมา ใช้ adr เฉลี่ยของการจองใหม่
    if request.adr_ref is not None:
        adr_ref = request.adr_ref
    else:
        adr_values = []
        for booking in sorted_bookings:
            adr_values.append(booking.adr)
        adr_ref = float(np.mean(adr_values))

    result = recommend_overbooking(
        p_cancels=p_cancels,
        hotel=request.hotel,
        adr_ref=adr_ref,
        already_occupied=request.already_occupied,
        cost_config=COST_CONFIG,
        k=request.k,
    )

    # จำนวนคนที่คาดว่าจะมาจริง = ผลรวมของ (1 - p_cancel)
    expected_show_ups = 0.0
    for p_cancel in p_cancels:
        expected_show_ups = expected_show_ups + (1 - p_cancel)

    return {
        "hotel": request.hotel,
        "o_star": int(result["o_star"]),
        "capacity": int(result["capacity"]),
        "already_occupied": int(result["already_occupied"]),
        "effective_capacity": int(result["effective_capacity"]),
        "expected_cost": round(float(result["expected_cost"]), 2),
        "adr_ref": round(adr_ref, 2),
        "n_new_bookings": len(p_cancels),
        "mean_p_cancel": round(float(np.mean(p_cancels)), 4),
        "expected_show_ups": round(expected_show_ups, 1),
        "model_uri": MODEL_URI,
    }