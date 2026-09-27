import pytest
from fastapi.testclient import TestClient
 
from src.serving import app as app_module
 
client = TestClient(app_module.app)
 
# ข้ามเทสที่ต้องใช้โมเดล ถ้าโหลดโมเดลไม่ได้
needs_model = pytest.mark.skipif(
    app_module.model is None or len(app_module.EXPECTED_COLUMNS) == 0,
    reason="ไม่มีโมเดลหรือ test.csv (ต้องรัน calibrate.py และ split_data.py ก่อน)",
)
 
 
def make_booking():
    """การจองที่ถูกต้อง 1 รายการ (คืนเป็น dict ใหม่ทุกครั้ง แก้ได้โดยไม่กระทบเทสอื่น)"""
    return {
        "hotel": "City Hotel",
        "lead_time": 120,
        "arrival_date_year": 2017,
        "arrival_date_month": "July",
        "arrival_date_week_number": 28,
        "arrival_date_day_of_month": 12,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "country": "PRT",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "reserved_room_type": "A",
        "deposit_type": "No Deposit",
        "agent": 9,
        "customer_type": "Transient",
        "adr": 110.5,
    }
 
 
def get_problem_fields(response):
    """ดึงชื่อช่องที่ผิดออกมาจาก response 422"""
    fields = []
    for problem in response.json()["problems"]:
        fields.append(problem["field"])
    return fields
 
 
# ============================================================
# /health
# ============================================================
 
 
def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["ok", "model_not_loaded"]
 
 
# ============================================================
# /predict — ข้อมูลผิดต้องได้ 422 (ไม่ต้องใช้โมเดล)
# ============================================================
 
 
def test_predict_rejects_zero_guests():
    booking = make_booking()
    booking["adults"] = 0
    booking["children"] = 0
    booking["babies"] = 0
 
    response = client.post("/predict", json=booking)
 
    assert response.status_code == 422
    assert "ผู้เข้าพักเป็นศูนย์" in response.json()["problems"][0]["message"]
 
 
def test_predict_rejects_negative_lead_time():
    booking = make_booking()
    booking["lead_time"] = -5
 
    response = client.post("/predict", json=booking)
 
    assert response.status_code == 422
    assert "lead_time" in get_problem_fields(response)
 
 
def test_predict_rejects_negative_adr():
    booking = make_booking()
    booking["adr"] = -10
 
    response = client.post("/predict", json=booking)
 
    assert response.status_code == 422
    assert "adr" in get_problem_fields(response)
 
 
def test_predict_rejects_wrong_month():
    booking = make_booking()
    booking["arrival_date_month"] = "Julyy"
 
    response = client.post("/predict", json=booking)
 
    assert response.status_code == 422
    assert "arrival_date_month" in get_problem_fields(response)
 
 
def test_predict_rejects_unknown_hotel():
    booking = make_booking()
    booking["hotel"] = "Beach Hotel"
 
    response = client.post("/predict", json=booking)
 
    assert response.status_code == 422
    assert "hotel" in get_problem_fields(response)
 
 
def test_predict_rejects_missing_field():
    booking = make_booking()
    del booking["lead_time"]
 
    response = client.post("/predict", json=booking)
 
    assert response.status_code == 422
    assert "lead_time" in get_problem_fields(response)
 
 
# ============================================================
# /predict-batch — ข้อมูลผิดต้องได้ 422 (ไม่ต้องใช้โมเดล)
# ============================================================
 
 
def test_batch_rejects_empty_list():
    response = client.post("/predict-batch", json=[])
    assert response.status_code == 422
 
 
def test_batch_rejects_when_one_booking_is_bad():
    good = make_booking()
    bad = make_booking()
    bad["adults"] = 0
    bad["children"] = 0
    bad["babies"] = 0
 
    response = client.post("/predict-batch", json=[good, bad])
 
    assert response.status_code == 422
    # รายการที่ผิดคือรายการที่ 2 (index 1)
    assert "1" in get_problem_fields(response)
 
 
# ============================================================
# ทำนายจริง (ต้องมีโมเดล)
# ============================================================
 
 
@needs_model
def test_predict_returns_probability():
    response = client.post("/predict", json=make_booking())
 
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["p_cancel"] <= 1.0
    assert body["will_cancel"] == (body["p_cancel"] >= 0.5)
 
 
@needs_model
def test_batch_matches_single_predict():
    """ทำนายแบบก้อนกับแบบทีละรายการต้องได้ค่าเท่ากัน (กัน skew ระหว่าง 2 endpoint)"""
    booking = make_booking()
 
    single = client.post("/predict", json=booking).json()
    batch = client.post("/predict-batch", json=[booking, booking]).json()
 
    assert batch["count"] == 2
    assert batch["results"][0]["p_cancel"] == single["p_cancel"]
    assert batch["results"][1]["p_cancel"] == single["p_cancel"]
 
 
# ============================================================
# /metrics
# ============================================================
 
 
def test_metrics_counts_requests():
    before = client.get("/metrics").json()
 
    booking = make_booking()
    booking["lead_time"] = -1
    client.post("/predict", json=booking)  # ได้ 422
 
    after = client.get("/metrics").json()
 
    assert after["total_requests"] == before["total_requests"] + 1
    assert after["client_errors_4xx"] == before["client_errors_4xx"] + 1
    assert after["server_errors_5xx"] == before["server_errors_5xx"]
 
 
def test_metrics_has_slo_section():
    body = client.get("/metrics").json()
 
    assert body["slo"]["predict_p95_target_ms"] == 100
    assert body["slo"]["error_rate_target"] == 0.01
 
