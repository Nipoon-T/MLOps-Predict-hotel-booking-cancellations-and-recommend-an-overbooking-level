"""
locustfile_slo.py — ทดสอบ SLO ที่ทีมตกลงกัน: p95 ของ /predict ≤ 100 ms ที่โหลด 120 req/s

ต่างจาก locustfile.py (ยิงเต็มกำลังเพื่อหาความจุสูงสุด):
ไฟล์นี้ยิงที่อัตราคงที่ แต่ละผู้ใช้ยิง 2 ครั้งต่อวินาที
    60 ผู้ใช้ x 2 ครั้ง/วินาที = 120 req/s

วิธีรัน (ยิงจากในเครือข่ายเดียวกับ container hotel-api):
    docker run --rm --network container:hotel-api -v "${PWD}\\load_test:/mnt/load_test" -v "${PWD}\\reports\\load_test:/mnt/reports" locustio/locust:2.46.6 -f /mnt/load_test/locustfile_slo.py --host http://127.0.0.1:8000 --headless -u 60 -r 10 -t 90s --csv /mnt/reports/docker_slo_120rps
"""

from locust import FastHttpUser, constant_throughput, task

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

# แต่ละการจองมาจากลูกค้า/ระบบคนละตัว จึงเปิดการเชื่อมต่อใหม่ทุกครั้ง แล้วปิดทันที
# (ถ้าใช้การเชื่อมต่อค้างไว้ ผู้ใช้จำลองแต่ละคนจะติดอยู่กับ worker ตัวเดิมตลอด
#  ทำให้บาง worker งานล้นขณะที่ตัวอื่นว่าง ซึ่งไม่ตรงกับการใช้งานจริงของระบบจอง)
CLOSE_CONNECTION = {"Connection": "close"}


class SloUser(FastHttpUser):
    # ผู้ใช้แต่ละคนยิง 2 ครั้งต่อวินาทีเท่านั้น (ไม่เร่งเต็มกำลัง)
    wait_time = constant_throughput(2)

    @task
    def predict_booking(self):
            self.client.post("/predict", json=GOOD_BOOKING, headers=CLOSE_CONNECTION)