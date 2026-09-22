# MLOps-Predict-hotel-booking-cancellations-and-recommend-an-overbooking-level

โครงงานสำหรับพัฒนาระบบ MLOps เพื่อพยากรณ์การยกเลิกการจองโรงแรม และนำผลการพยากรณ์ไปใช้สนับสนุนการตัดสินใจเกี่ยวกับระดับการจองเกิน (Overbooking Level)

## ภาพรวมโครงงาน

กระบวนการเตรียมข้อมูลในปัจจุบันครอบคลุม:

* การทำความสะอาดข้อมูล (Data Cleaning)
* การตรวจสอบคุณภาพข้อมูล (Data Validation)
* การจัดทำเอกสารเกี่ยวกับข้อมูลที่อาจทำให้เกิด Data Leakage
* การแบ่งข้อมูลตามลำดับเวลา (Chronological Data Splitting)
* การสร้างไฟล์ Manifest สำหรับตรวจสอบความถูกต้องและความสามารถในการทำซ้ำ
* การทดสอบระบบอัตโนมัติ (Automated Testing)
* การจัดทำรายงาน EDA และการเปรียบเทียบข้อมูลระหว่างชุดข้อมูล

---

## การติดตั้ง

ติดตั้ง dependencies ที่ใช้ในโครงงานด้วยคำสั่ง:

```powershell
pip install -r requirements.txt
```

แนะนำให้ใช้ virtual environment หรือ Conda environment เพื่อแยก dependencies ของโครงงานออกจากระบบหลัก

---

## การดาวน์โหลดข้อมูล

Dataset ที่ใช้ในโครงงานคือ **Hotel Booking Demand** ซึ่งมีแหล่งที่มาจาก Kaggle โดยใช้ข้อมูลจาก Jesse Mostipak

* Dataset source: [Hotel Booking Demand — Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
* Dataset identifier: `jessemostipak/hotel-booking-demand`

เพื่อให้สมาชิกในทีมสามารถใช้งาน dataset ได้โดยไม่จำเป็นต้องเข้าสู่ระบบ Kaggle หรือกำหนดค่า Kaggle API Key ภายใน environment ของโครงงาน ทีมจึงจัดเก็บไฟล์ dataset สำหรับใช้งานไว้บน Google Drive

### Google Drive Dataset

[ดาวน์โหลด `hotel_bookings.csv` จาก Google Drive](https://drive.google.com/file/d/1wseDlys4qW9KP0IS6x4oECj-KSCjIEiz/view?usp=sharing)

หลังจากดาวน์โหลดไฟล์แล้ว ให้วางไฟล์โดยตรงไว้ที่:

```text
data/raw/hotel_bookings.csv
```

โครงสร้างที่ถูกต้อง:

```text
projectML/
└── data/
    └── raw/
        └── hotel_bookings.csv
```

จากนั้นรัน:

```powershell
python src\download_data.py
```

> หมายเหตุ: ชื่อ `download_data.py` ยังคงใช้เป็นชื่อสคริปต์สำหรับขั้นตอนเตรียม dataset แต่สคริปต์ไม่ได้ดาวน์โหลดไฟล์จาก Kaggle หรือ Google Drive โดยอัตโนมัติ สคริปต์จะตรวจสอบว่าไฟล์ dataset ถูกวางไว้ในตำแหน่งที่กำหนดแล้ว และตรวจสอบความถูกต้องด้วย SHA-256

สคริปต์จะดำเนินการดังนี้:

1. ตรวจสอบว่ามี `data/raw/hotel_bookings.csv`
2. คำนวณ SHA-256 ของไฟล์ dataset
3. ตรวจสอบ SHA-256 กับค่าที่กำหนดไว้
4. หยุดการทำงานหาก checksum ไม่ตรงกัน

ค่า SHA-256 ที่ใช้ตรวจสอบคือ:

```text
7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06
```

หาก SHA-256 ไม่ตรงกับค่าที่กำหนด สคริปต์จะหยุดและแจ้งข้อผิดพลาดเพื่อป้องกันการนำ dataset ที่ไม่ตรงกับข้อมูลที่กำหนดไว้เข้าสู่ pipeline

---

## การเตรียมข้อมูล

### 1. ตรวจสอบข้อมูลดิบ

ตรวจสอบข้อมูลดิบก่อนการทำความสะอาด:

```powershell
python src\validate.py data\raw\hotel_bookings.csv
```

ข้อมูลดิบของโครงงาน **คาดว่าจะไม่ผ่าน validation** เนื่องจากพบข้อมูลที่ต้องจัดการก่อนเข้าสู่ขั้นตอน Data Cleaning

จากข้อมูลปัจจุบันพบ:

* ADR ติดลบ 1 แถว
* จำนวนผู้เข้าพักรวมเป็นศูนย์ 180 แถว

เมื่อ validation พบข้อผิดพลาด โปรแกรมจะแสดงรายละเอียดของปัญหาและคืนค่า exit code ที่ไม่เป็นศูนย์ เพื่อหยุด workflow ก่อนเข้าสู่ขั้นตอนถัดไป

จากนั้นจึงดำเนินการ Data Cleaning

### 2. ทำความสะอาดข้อมูล

```powershell
python src\clean_data.py
```

ผลลัพธ์จะถูกสร้างที่:

```text
data/interim/hotel_bookings_clean.csv
```

กระบวนการ cleaning ปัจจุบัน:

* ลบแถวที่มี `adr < 0`
* ลบแถวที่มีจำนวนผู้เข้าพักรวมเป็นศูนย์
* ไม่ลบ duplicate rows
* สร้าง `is_duplicate` flag
* สร้าง `row_id`
* สร้าง `arrival_date`
* สร้าง `booking_date`
* หยุด pipeline หากสัดส่วนข้อมูลที่ถูกลบมากกว่า safety threshold ที่กำหนดไว้ 1%

จากข้อมูลปัจจุบัน:

```text
Original rows : 119,390
Removed rows  : 181
Clean rows    : 119,209
Removal rate  : 0.1516%
```

อัตราการลบข้อมูลต่ำกว่า safety threshold 1% จึงสามารถดำเนินการต่อได้

### 3. ตรวจสอบข้อมูลหลังทำความสะอาด

เนื่องจากกระบวนการทำความสะอาดสร้าง derived columns ได้แก่ `row_id`, `is_duplicate`, `arrival_date` และ `booking_date` จึงต้องใช้ `--allow-derived`:

```powershell
python src\validate.py data\interim\hotel_bookings_clean.csv --allow-derived
```

ผลลัพธ์ที่คาดหวังคือ:

```text
VALIDATION PASSED
```

พร้อมจำนวนข้อมูล:

```text
Rows    : 119,209
Columns : 36
```

### 4. แบ่งข้อมูลตามลำดับเวลา

```powershell
python src\split_data.py
```

การแบ่งข้อมูลใช้ `arrival_date` เป็นตัวกำหนดลำดับเวลา โดยแบ่งเป็น:

| Split      |   Rows | Date range              |
| ---------- | -----: | ----------------------- |
| Train      | 49,108 | 2015-07-01 → 2016-06-30 |
| Validation | 29,482 | 2016-07-01 → 2016-12-31 |
| Test       | 12,789 | 2017-01-01 → 2017-03-31 |
| Production | 27,830 | 2017-04-01 → 2017-08-31 |

รวมทั้งหมด:

```text
119,209 rows
```

ระบบจะตรวจสอบ:

* จำนวนแถวก่อนและหลังการ split
* Temporal order
* การ overlap ระหว่าง train, validation, test และ production
* ความครบถ้วนของข้อมูล

ผลลัพธ์จะถูกสร้างใน:

```text
data/processed/

├── train.csv
├── validation.csv
├── test.csv
├── production.csv
└── split_manifest.json
```

`split_manifest.json` ใช้เก็บข้อมูลเกี่ยวกับการแบ่ง dataset และ SHA-256 ของไฟล์ split เพื่อช่วยตรวจสอบความถูกต้องและความสามารถในการทำซ้ำของ pipeline

---

## การทดสอบ Validation ด้วย Bad Data

สามารถสร้าง dataset ที่มีข้อมูลผิดโดยตั้งใจเพื่อทดสอบระบบ validation ได้ด้วย:

```powershell
python src\create_bad_data.py
```

ผลลัพธ์:

```text
data/bad/hotel_bookings_bad.csv
```

จากนั้นรัน:

```powershell
python src\validate.py data\bad\hotel_bookings_bad.csv
```

Validation ควรตรวจพบข้อผิดพลาด เช่น:

* invalid binary values
* negative values
* invalid month
* invalid dates
* zero guests

และคืนค่า exit code ที่ไม่เป็นศูนย์

การทดสอบนี้ใช้เพื่อยืนยันว่า validation สามารถหยุด workflow เมื่อพบข้อมูลที่ผิดเงื่อนไขได้จริง

---

## การทดสอบ

รัน automated tests ด้วย:

```powershell
python -m pytest -v
```

ชุดทดสอบประกอบด้วย:

```text
tests/

├── test_clean.py
├── test_split.py
└── test_validate.py
```

ปัจจุบันมี automated tests ทั้งหมด:

```text
34 passed
```

การทดสอบครอบคลุม:

* Data Cleaning
* Data Validation
* Bad Data Validation
* Temporal Data Splitting
* Split Integrity
* Data Quality Rules

---

## Reports

รายงาน EDA และการเปรียบเทียบข้อมูลระหว่างชุดข้อมูลอยู่ใน:

```text
reports/

├── eda.py
├── eda_report.md
├── cancellation_by_hotel_month.csv
└── split_comparison.csv
```

สร้างรายงานด้วย:

```powershell
python reports\eda.py
```

รายงานประกอบด้วย:

* Cancellation rate ตามโรงแรมและเดือน โดยใช้ข้อมูลจาก train
* การเปรียบเทียบสัดส่วนโรงแรมระหว่าง train, validation, test และ production
* การเปรียบเทียบ `deposit_type`
* การเปรียบเทียบ `lead_time`
* Cancellation rate ของแต่ละ split

---

## Data Decisions และ Data Leakage

รายละเอียดการตัดสินใจเกี่ยวกับการทำความสะอาดข้อมูล การจัดการ duplicate/missing values และข้อมูลที่อาจทำให้เกิด Data Leakage อยู่ใน:

```text
data_decisions.md
```

รายการ target, leakage columns และข้อมูลที่ต้องพิจารณาในระดับทีมอยู่ใน:

```text
configs/columns.yaml
```

### Duplicate Data

ระบบจะไม่ลบ duplicate rows ในขั้นตอน cleaning แต่จะสร้าง `is_duplicate` เพื่อให้สามารถวิเคราะห์ผลกระทบในขั้นตอน modeling และ decision layer ได้

จากข้อมูลปัจจุบัน:

```text
40,141 rows
```

เป็นจำนวนทุกแถวที่อยู่ในกลุ่ม duplicate เมื่อใช้ `duplicated(keep=False)`

ส่วน:

```text
31,980 rows
```

เป็นจำนวน duplicate ที่ไม่นับ occurrence แรก เมื่อใช้ `duplicated(keep='first')`

---

## Dataset Source and License

Dataset:

* Kaggle: [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
* Dataset identifier: `jessemostipak/hotel-booking-demand`
* License: Attribution 4.0 International (CC BY 4.0)
* [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/)

> หมายเหตุ: Google Drive ใช้เป็นช่องทางสำหรับแจกไฟล์ dataset ให้สมาชิกทีมดาวน์โหลดเท่านั้น ส่วนแหล่งที่มาของ dataset ยังคงอ้างอิงตาม dataset ต้นทางที่ระบุไว้ข้างต้น

---

## โครงสร้างโครงงาน

```text
projectML/

├── configs/
│   └── columns.yaml
│
├── data/
│   ├── raw/
│   ├── bad/
│   ├── interim/
│   └── processed/
│
├── reports/
│   ├── eda.py
│   ├── eda_report.md
│   ├── cancellation_by_hotel_month.csv
│   └── split_comparison.csv
│
├── schemas/
│   └── hotel_booking_schema.py
│
├── src/
│   ├── clean_data.py
│   ├── split_data.py
│   ├── validate.py
│   ├── create_bad_data.py
│   └── download_data.py
│
├── tests/
│   ├── test_clean.py
│   ├── test_split.py
│   └── test_validate.py
│
├── data_decisions.md
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
```

> หมายเหตุ: `data/raw/`, `data/bad/`, `data/interim/` และ `data/processed/` ไม่ถูกเก็บไว้ใน Git repository โดยข้อมูลต้องดาวน์โหลดจาก Google Drive และสร้างใหม่จากกระบวนการเตรียมข้อมูล

---

## ลำดับการรันหลัก

สำหรับการเตรียมข้อมูลตั้งแต่ต้น สามารถรันตามลำดับดังนี้:

### 1. เตรียม Dataset

ดาวน์โหลด `hotel_bookings.csv` จาก [Google Drive ของทีม](https://drive.google.com/file/d/1wseDlys4qW9KP0IS6x4oECj-KSCjIEiz/view?usp=sharing)

จากนั้นวางไฟล์ไว้ที่:

```text
data/raw/hotel_bookings.csv
```

แล้วตรวจสอบ dataset:

```powershell
python src\download_data.py
```

### 2. Validate ข้อมูลดิบ

```powershell
python src\validate.py data\raw\hotel_bookings.csv
```

**คาดว่าจะ FAIL** หากพบข้อมูลที่ไม่ผ่าน validation

### 3. Clean ข้อมูล

```powershell
python src\clean_data.py
```

### 4. Validate ข้อมูลหลัง Cleaning

```powershell
python src\validate.py data\interim\hotel_bookings_clean.csv --allow-derived
```

**คาดว่าจะ PASS**

### 5. Split ข้อมูลตามเวลา

```powershell
python src\split_data.py
```

### 6. รัน Automated Tests

```powershell
python -m pytest -v
```

### 7. สร้าง Reports

```powershell
python reports\eda.py
```

---

## Expected Pipeline

```text
Google Drive Dataset
        │
        ▼
hotel_bookings.csv
        │
        ▼
download_data.py
        │
        ▼
data/raw/hotel_bookings.csv
        │
        ▼
validate.py
        │
        ├── FAIL → หยุด workflow หากพบข้อมูลผิดเงื่อนไข
        │
        ▼
clean_data.py
        │
        ▼
data/interim/hotel_bookings_clean.csv
        │
        ▼
validate.py --allow-derived
        │
        ├── FAIL → หยุด workflow
        │
        ▼
split_data.py
        │
        ├── train.csv
        ├── validation.csv
        ├── test.csv
        ├── production.csv
        └── split_manifest.json
        │
        ▼
pytest
        │
        ▼
EDA / Reports
```
