import os

# ปิด telemetry ของ Prefect: ตอนเปิด server ชั่วคราวมันเขียน database ชนกันจนขึ้น
# "database is locked" แม้ไม่กระทบ flow แต่ทำให้ log รก
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")
os.environ.setdefault("DO_NOT_TRACK", "1")