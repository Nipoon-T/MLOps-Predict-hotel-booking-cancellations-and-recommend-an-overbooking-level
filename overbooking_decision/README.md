# Decision Layer / Backtest — บทบาทที่ 3 (CP413008)

โมดูลนี้แปลงความน่าจะเป็นยกเลิกการจอง (จากบทบาท 2) เป็นจำนวนห้องที่ควรรับจองเกิน (overbook)
และเปรียบเทียบผลกับ 3 นโยบายอื่นด้วยการ backtest บนข้อมูลจริง

## วิธีรัน

```bash
pip install numpy pandas pyyaml matplotlib pytest
python -m pytest tests/ -q      # รัน unit test (9 ข้อ)
python demo.py                  # รันต้นแบบเต็มด้วยข้อมูลสังเคราะห์ + baseline probability
```

`demo.py` สร้างข้อมูลสังเคราะห์เอง (เพราะยังไม่มีข้อมูลจริง/โมเดลจริงจากบทบาท 2) แล้วรัน:
1. Backtest 4 นโยบายที่ k default → พิมพ์ตารางกำไร + จำนวนลูกค้าที่ถูกย้าย/1,000 คืน
2. Sensitivity ข้าม k ทุกค่าใน config.yaml → พิมพ์ตาราง + บันทึก `sensitivity_plot.png`

ผลจากข้อมูลสังเคราะห์ตัวอย่าง: นโยบาย **model** ให้กำไรสูงสุดในทุกค่า k (1.5, 2.0, 3.0) —
นี่คือรูปแบบผลลัพธ์ที่ต้องไปโชว์ในรายงาน/นำเสนอ (ถ้าเจอผลตรงข้าม ต้องตรวจ features/labels ของบทบาท 2)

## ไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `config.yaml` | สมมติฐานต้นทุนทั้งหมด (capacity, k, fixed %) — แก้ที่นี่ที่เดียว |
| `cost_config.py` | โหลด config.yaml เป็น object ใช้งานง่าย |
| `simulate.py` | Monte Carlo: จำลองผู้มาจริง + หา o* ที่ต้นทุนต่ำสุด |
| `policies.py` | 4 นโยบาย (no_overbook, fixed, avg_rate, model) |
| `backtest.py` | รันนโยบายบนคืนจริง (ใช้ label จริง) + สรุปผล |
| `sensitivity.py` | วนค่า k ทั้งหมด + plot |
| `recommend.py` | **จุดเชื่อมกับบทบาทที่ 4** — `recommend_overbooking(...)` เรียกจาก FastAPI ได้ตรงๆ |
| `demo.py` | ข้อมูลสังเคราะห์ + รันทุกอย่างแบบ end-to-end |
| `tests/test_decision.py` | unit tests (pytest) |

## เชื่อมกับ pipeline จริงของทีม (โมเดลจากบทบาทที่ 2 พร้อมแล้ว)

บทบาทที่ 2 ส่งมอบ `models:/hotel-cancellation-classifier@candidate` (Random Forest tuned + isotonic
calibration, PR-AUC 0.752, ECE 0.0379) พร้อม `features.build_features()` ตัวเดียวกับตอนเทรน

ไฟล์ใหม่ 3 ไฟล์ที่เพิ่มเข้ามาเชื่อมกับของจริง:

| ไฟล์ | หน้าที่ |
|---|---|
| `real_data_adapter.py` | ประกอบ `stay_date` จาก `arrival_date_*`, โหลดโมเดลจาก MLflow, เรียก `build_features()` แล้วคืน DataFrame พร้อมใช้กับ `backtest_policies()` |
| `run_real_backtest.py` | CLI รัน backtest+sensitivity เต็มรูปแบบบนข้อมูลจริง บันทึกผลเป็น CSV/PNG |
| `suggest_capacity.py` | คำนวณ capacity ที่แนะนำจากข้อมูล train จริง (p95 ของจำนวนจองพร้อมกันต่อคืน) ให้เอาไปใส่ `config.yaml` เอง |

**วิธีใช้** (รันจาก root ของ repo หลัก เพื่อให้ import `src.modeling.features` เจอ):

```bash
# 1) ดู capacity ที่แนะนำ แล้วเอาไปใส่ config.yaml เอง (เป็น manual step ตั้งใจ เพราะต้องประกาศในรายงาน)
python overbooking_decision/suggest_capacity.py --train-csv data/processed/train.csv

# 2) รัน backtest จริงบน test set (ปรับชื่อไฟล์ตามที่ split_data.py สร้างจริง)
python overbooking_decision/run_real_backtest.py \
    --test-csv data/processed/test.csv \
    --model-uri "models:/hotel-cancellation-classifier@candidate"

# 3) ทำแบบเดียวกันกับ production_simulation.csv (ถ้ามีไฟล์แยก) เพื่อดู robustness นอก test set
```

**สำคัญ — ยังไม่เคยรันกับของจริง:** `real_data_adapter.py`/`run_real_backtest.py` เขียนตามอินเทอร์เฟซ
ที่ทีมอัปเดตมา (ชื่อฟังก์ชัน `build_features`, model URI) แต่เครื่องนี้ไม่มี repo/mlflow.db จริงให้รันทดสอบ
ส่วนที่ทดสอบแล้วคือ `add_stay_date()` และ `estimate_capacity()` (ดู `tests/test_real_data_adapter.py`,
ผ่านครบ 14 ข้อรวมของเดิม) — ส่วน `build_backtest_dataset()` ที่พึ่ง mlflow+features จริง ต้องรันทดสอบเองในเครื่องที่มี repo ก่อนเชื่อผลลัพธ์
ถ้า import error หรือ path ไม่ตรง ปรับ `--features-module` กับชื่อไฟล์ CSV ให้ตรงกับ repo จริงได้เลย

## ข้อสมมติที่ต้องอธิบายได้ตอนนำเสนอ/ในรายงาน

- **การเรียงลำดับการจอง**: โค้ดสมมติว่าการจอง "ถูกรับก่อน-หลัง" ตามลำดับที่ส่งเข้ามาใน DataFrame
  (ใน `demo.py` เรียงตาม lead_time — ควรเปลี่ยนเป็น booking_date จริงตอนใช้ข้อมูลจริง)
- **arrivals ถูกจำกัดด้วย capacity+o เสมอ**: ถ้าคืนไหนมีคนจองมากกว่า capacity+o ส่วนเกินถือว่า
  "ไม่เคยถูกรับ" ไม่นับต้นทุน — สะท้อนว่าระบบต้อง "ปฏิเสธการจอง" ตั้งแต่ตอนนั้น ไม่ใช่ walk-in
- **cost_empty ใช้ adr เฉลี่ยของคืนนั้น**: เป็นค่าประมาณ ราคาจริงต่างกันตาม room type ได้ ถ้าต้องการ
  ละเอียดกว่านี้ให้แยกคำนวณต่อ room type
