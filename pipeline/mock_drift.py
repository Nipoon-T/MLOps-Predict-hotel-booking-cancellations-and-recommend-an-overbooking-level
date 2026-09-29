"""ตัวตรวจ drift ปลอม (ใช้ระหว่างรอสคริปต์จริงของบทบาทที่ 6)

ทำตาม "ข้อตกลง" เดียวกับที่สคริปต์จริงต้องทำ เพื่อให้เปลี่ยนตัวได้โดยไม่ต้องแก้ flow:
  รับ:   <week_csv> <out_json>
  เขียน: out_json = {"week": int, "retrain": bool, "reasons": [str, ...]}
  exit:  0 เสมอเมื่อตรวจสำเร็จ (ผลอยู่ใน JSON ไม่ใช่ exit code)

ตัวปลอมนี้ไม่ได้คำนวณ drift จริง แค่สั่ง retrain ในสัปดาห์ที่กำหนดผ่าน
environment variable MOCK_DRIFT_WEEKS (ค่าเริ่มต้น "6,14") เพื่อทดสอบ flow
"""

import json
import os
import re
import sys
from pathlib import Path


def main() -> int:
    week_csv, out_json = Path(sys.argv[1]), Path(sys.argv[2])
    week = int(re.search(r"week_(\d+)", week_csv.stem).group(1))
    drift_weeks = {
        int(w) for w in os.environ.get("MOCK_DRIFT_WEEKS", "6,14").split(",") if w.strip()
    }
    retrain = week in drift_weeks
    result = {
        "week": week,
        "retrain": retrain,
        "reasons": [f"MOCK: สัปดาห์ {week} อยู่ใน MOCK_DRIFT_WEEKS"] if retrain else [],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
