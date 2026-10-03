# Handoff → บทบาทที่ 5 (Pipeline + Docker)

จากบทบาทที่ 4 (Serving + Load Test) สรุปสิ่งที่ทำไปแล้ว และสิ่งที่ต้องรู้เพื่อเอา API ไปรวมใน Docker Compose / DAG

## 1. สิ่งที่บทบาทที่ 4 ทำไว้แล้ว

| ไฟล์ | หน้าที่ |
| --- | --- |
| `src/serving/app.py` | FastAPI: `/health`, `/metrics`, `/predict`, `/predict-batch`, `/recommend-overbooking` |
| `src/serving/schemas.py` | รูปแบบข้อมูลที่รับเข้า ข้อมูลผิดตอบ 422 |
| `src/serving/export_model.py` | คัดลอกโมเดลจาก MLflow Registry ออกมาเป็นโฟลเดอร์ `serving_model/` |
| `Dockerfile`, `.dockerignore` | image ของ API (Python 3.13-slim, uvicorn 8 workers) |
| `tests/test_api.py` | เทส API 18 ข้อ |
| `load_test/` | Locust (ความจุสูงสุด + SLO ที่อัตราคงที่) และสคริปต์ profile |
| `reports/load_test/` | ผลวัดทุกรอบ (CSV) |

API ทดสอบใน container แล้ว: throughput 158 req/s, p95 330 ms ที่ 120 req/s, error 0% ผ่าน SLO ที่ทีมตกลง

## 2. สิ่งที่ต้องรู้ก่อนเอาไปใช้

### 2.1 container โหลดโมเดลจาก `mlflow.db` ตรงๆ ไม่ได้

`mlflow.db` จดที่อยู่ไฟล์โมเดลเป็น path เต็มของเครื่องที่เทรน (เช่น `C:\Users\...\mlruns`) พอเปิดใน container (Linux) path นั้นไม่มีอยู่จริง จะขึ้น error `No such artifact`

วิธีที่ใช้: **export โมเดลออกมาเป็นโฟลเดอร์ก่อน** แล้วให้ container อ่านจากโฟลเดอร์นั้น

```powershell
python -m src.serving.export_model             # ใช้ alias candidate (ค่าเริ่มต้น)
python -m src.serving.export_model champion    # หรือระบุ alias อื่น
```

ผลลัพธ์อยู่ที่ `serving_model/` (ถูก `.gitignore` กันไว้แล้ว) และมีไฟล์ `serving_model/EXPORTED_FROM.txt` บอกว่าเป็นโมเดล version ไหน

**ใน DAG ให้ใส่ขั้นนี้ต่อจาก `calibrate.py` (หรือหลังขั้น promote) และก่อนเปิด/รีสตาร์ท API เสมอ**

ข้อควรระวังเพิ่ม: ถ้าจะรันการเทรนใน container ด้วย `mlflow.db` จะจด path ของ container แทน (เช่น `/app/mlruns`) ทำให้เปิดจากเครื่อง Windows ไม่ได้ ควรเลือกให้เทรนและ export ในสภาพแวดล้อมเดียวกัน

### 2.2 API ต้องการ 2 โฟลเดอร์จากภายนอก (mount เข้าไป)

| บนเครื่อง | ใน container | ใช้ทำอะไร |
| --- | --- | --- |
| `./serving_model` | `/app/serving_model` | ตัวโมเดลที่ export แล้ว |
| `./data/processed` | `/app/data/processed` | อ่านแค่หัวตารางของ `test.csv` เพื่อให้คอลัมน์ตรงกับตอนเทรน |

mount แบบ read-only (`:ro`) ได้ API ไม่เขียนไฟล์

### 2.3 ค่าที่ตั้งผ่าน environment variable

| ตัวแปร | ค่าเริ่มต้นใน image | ความหมาย |
| --- | --- | --- |
| `MODEL_DIR` | `/app/serving_model` | โหลดโมเดลจากโฟลเดอร์นี้ (ถ้าไม่ตั้ง จะโหลดจาก registry ซึ่งใช้ได้เฉพาะนอก container) |
| `WORKERS` | `8` | จำนวน process ของ uvicorn ตัวเลข SLO วัดที่ 8 |
| `MODEL_ALIAS` | `candidate` | ใช้เฉพาะตอนไม่ได้ตั้ง `MODEL_DIR` |

## 3. ตัวอย่าง service ใน `docker-compose.yml`

```yaml
services:
  api:
    build: .
    image: hotel-api
    ports:
      - "8000:8000"
    environment:
      WORKERS: "8"
    volumes:
      - ./serving_model:/app/serving_model:ro
      - ./data/processed:/app/data/processed:ro
    healthcheck:
      # image เป็น python:3.13-slim ไม่มี curl จึงใช้ python เช็กแทน
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 40s
```

`start_period: 40s` เพราะ worker 8 ตัวใช้เวลาโหลดโมเดลราว 20–30 วินาที ช่วงนั้นยังตอบไม่ได้

เช็กว่าพร้อม: `/health` ต้องตอบ `"status": "ok"` ถ้าได้ `"model_not_loaded"` แปลว่าไม่เจอ `serving_model/` (ลืม export หรือ mount ผิด)

## 4. ลำดับที่ต้องรันก่อน `docker compose up`

```text
download_data.py → validate.py → clean_data.py → validate.py --allow-derived → split_data.py
      → train.py → calibrate.py → export_model.py → docker compose up
```

ถ้าขาด `data/processed/test.csv` หรือ `serving_model/` API ยังเปิดได้ แต่ทำนายไม่ได้ (ตอบ 503)

## 5. การเปลี่ยนโมเดล (promote / rollback)

API อ่านโมเดลครั้งเดียวตอนเริ่ม ถ้าย้าย alias ใน registry แล้ว ต้อง export ใหม่ แล้วรีสตาร์ท container

```powershell
python -m src.serving.export_model champion
docker compose restart api
```

เช็กได้จาก `serving_model/EXPORTED_FROM.txt` ว่าตอนนี้ API ใช้โมเดล version ไหน

## 6. เรื่องที่บทบาทที่ 6 (CI/CD + Monitoring) ควรรู้

* `python -m pytest tests/test_api.py` รันใน CI ได้แม้ไม่มีโมเดล (เทสที่ต้องใช้โมเดล 3 ข้อจะ skip เอง เหลือผ่าน 15 ข้อ)
* `ruff check src/serving load_test tests/test_api.py` ผ่านทั้งหมด (ruff 0.16.9)
* `/metrics` ตอบเป็น JSON และนับแยกต่อ worker ถ้ารัน 8 worker จะเห็นแค่ของ worker ที่ตอบ request นั้น ถ้าต้องการภาพรวมต้องใช้ Prometheus
* SLO สำหรับตั้ง alert: throughput ≥ 150 req/s, p95 ≤ 350 ms ที่ 120 req/s, error rate (5xx) ≤ 1%

## 7. ข้อควรรู้เกี่ยวกับ Windows

* ถ้ารัน `uvicorn --workers` บน Windows นอก Docker อาจมี worker พังตอนเริ่ม (`WinError 10022`) uvicorn จะเปิดตัวใหม่แทนให้เอง ใน Docker ไม่เจอปัญหานี้
* ตอนรัน `docker run` แบบไม่ใส่ `-d` แล้วกด Ctrl + C บางครั้ง container ยังไม่ปิด ให้ใช้ `docker stop <ชื่อ>`
