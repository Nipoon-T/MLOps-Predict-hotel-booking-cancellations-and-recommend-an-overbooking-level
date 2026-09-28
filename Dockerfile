# Dockerfile — API ทำนายการยกเลิกการจอง (คนที่ 4: Serving)
#
# build:  docker build -t hotel-api .
# run:    ดูคำสั่งใน README ส่วน Serving (ต้อง mount serving_model/ และ data/processed/ เข้าไป)

# ใช้ Python เวอร์ชันเดียวกับที่ทีมใช้พัฒนา
FROM python:3.13-slim

WORKDIR /app

# ติดตั้งไลบรารีก่อนคัดลอกโค้ด: ถ้าแก้แค่โค้ด Docker จะใช้ layer นี้ซ้ำ ไม่ต้องติดตั้งใหม่ทุกครั้ง
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกเฉพาะโค้ดที่ API ต้องใช้ (API + ชั้นตัดสินใจ overbooking ของบทบาทที่ 3)
COPY src/ src/
COPY overbooking_decision/ overbooking_decision/

# บอก app.py ให้โหลดโมเดลจากโฟลเดอร์ (ไม่ใช่จาก mlflow.db)
ENV MODEL_DIR=/app/serving_model
# จำนวน worker (process) ของ uvicorn เปลี่ยนได้ตอน run ด้วย -e WORKERS=...
ENV WORKERS=8

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --workers ${WORKERS}"]