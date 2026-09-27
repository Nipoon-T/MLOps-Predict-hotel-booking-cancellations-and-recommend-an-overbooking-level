"""
locustfile.py — Load test ของ API (คนที่ 4: Serving + Load test)

จำลองผู้ใช้หลายคนยิง request พร้อมกัน เพื่อวัดว่า API ผ่าน SLO ไหม
    - p95 ของ /predict ≤ 100 ms
    - throughput ≥ 200 req/s ต่อ container

วิธีรัน (เปิด API แบบไม่มี --reload ไว้ก่อนในอีก terminal):
    locust -f load_test/locustfile.py --host http://127.0.0.1:8000 --headless -u 20 -r 5 -t 60s --csv reports/load_test/baseline
"""

from locust import HttpUser, constant, task

# การจองที่ถูกต้อง (ชุดเดียวกับตัวอย่างใน /docs)
GOOD_BOOKING = {
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

# การจองที่ผิด (ผู้เข้าพักเป็นศูนย์) ต้องได้ 422
BAD_BOOKING = dict(GOOD_BOOKING)
BAD_BOOKING["adults"] = 0
BAD_BOOKING["children"] = 0
BAD_BOOKING["babies"] = 0


class HotelApiUser(HttpUser):
    # ไม่รอระหว่าง request เลย เพื่อดันให้ถึงขีดสุดของ API
    wait_time = constant(0)

    # ตัวเลขใน @task คือน้ำหนัก: ยิงข้อมูลถูก 10 ครั้ง ต่อข้อมูลผิด 1 ครั้ง
    @task(10)
    def predict_good_booking(self):
        self.client.post("/predict", json=GOOD_BOOKING)

    @task(1)
    def predict_bad_booking(self):
        # ข้อมูลผิดต้องได้ 422 ถึงจะนับว่า "สำเร็จ"
        # ตั้งชื่อแยกไว้ ตัวเลขจะได้ไม่ปนกับ /predict ปกติ
        with self.client.post(
            "/predict",
            json=BAD_BOOKING,
            name="/predict [bad data -> 422]",
            catch_response=True,
        ) as response:
            if response.status_code == 422:
                response.success()
            else:
                response.failure(f"ควรได้ 422 แต่ได้ {response.status_code}")
