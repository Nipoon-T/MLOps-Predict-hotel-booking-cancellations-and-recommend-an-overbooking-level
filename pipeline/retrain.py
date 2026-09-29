"""retrain บนข้อมูลช่วงล่าสุด + gate ก่อน promote (บทบาทที่ 5)

ไม่แก้โค้ดของบทบาทที่ 2: เรียก calibrate.main() เดิม แต่ชี้ path ข้อมูลไปที่หน้าต่างเวลาใหม่

หน้าต่างเวลา (สัดส่วนเดียวกับ split_data.py: 12 / 6 / 3 เดือน) นับถอยจาก cutoff
  train       [cutoff - 21 เดือน, cutoff - 9 เดือน)
  validation  [cutoff - 9 เดือน,  cutoff - 3 เดือน)
  test        [cutoff - 3 เดือน,  cutoff)
cutoff = วันหลังสุดที่รู้ผลจริงแล้ว (label มาช้า: รู้ตอนถึงวันเข้าพัก)

gate (เกณฑ์พื้นฐาน บทบาทที่ 6 เพิ่มเกณฑ์กำไรเทียบ champion ได้ใน check_gate)
  1. candidate ECE <= 0.05
  2. candidate PR-AUC ตกจาก champion ไม่เกิน 0.03 (วัดบน test ของหน้าต่างใหม่ทั้งคู่)

วิธีใช้ (จาก root ของ repo):
  python -m pipeline.retrain window 2017-05-14    สร้างหน้าต่างข้อมูลที่ data/retrain/2017-05-14/
  python -m pipeline.retrain calibrate 2017-05-14 เทรน + calibrate + register เป็น candidate
  python -m pipeline.retrain gate 2017-05-14      exit 0 = ผ่าน, 1 = ไม่ผ่าน
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_FILE = ROOT / "data" / "interim" / "hotel_bookings_clean.csv"
RETRAIN_DIR = ROOT / "data" / "retrain"
REPORT_DIR = ROOT / "reports" / "monitoring"
MODEL_NAME = "hotel-cancellation-classifier"

MAX_ECE = 0.05
MAX_PR_AUC_DROP = 0.03


def window_dir(cutoff: str) -> Path:
    return RETRAIN_DIR / cutoff


def build_window(cutoff: str) -> Path:
    df = pd.read_csv(CLEAN_FILE)
    arrival = pd.to_datetime(df["arrival_date"])
    end = pd.Timestamp(cutoff)
    bounds = {
        "train": (end - pd.DateOffset(months=21), end - pd.DateOffset(months=9)),
        "validation": (end - pd.DateOffset(months=9), end - pd.DateOffset(months=3)),
        "test": (end - pd.DateOffset(months=3), end),
    }
    out = window_dir(cutoff)
    out.mkdir(parents=True, exist_ok=True)
    for name, (start, stop) in bounds.items():
        part = df[(arrival >= start) & (arrival < stop)]
        if part.empty:
            raise ValueError(f"{name} ว่าง ({start.date()} ถึง {stop.date()}) cutoff เร็วเกินไป")
        part.to_csv(out / f"{name}.csv", index=False)
        print(f"{name:<10} {start.date()} → {stop.date()}  {len(part):>6,} แถว"
              f"  cancel {part['is_canceled'].mean():.3f}")
    return out


def _modeling_path() -> None:
    sys.path.insert(0, str(ROOT / "src" / "modeling"))


def run_calibrate(cutoff: str) -> None:
    """calibrate.py เดิมของบทบาทที่ 2 แต่อ่านข้อมูลจากหน้าต่างใหม่"""
    _modeling_path()
    import calibrate  # path ของ src/modeling ถูกเพิ่มใน _modeling_path()

    folder = window_dir(cutoff)
    calibrate.TRAIN_PATH = str(folder / "train.csv")
    calibrate.VALIDATION_PATH = str(folder / "validation.csv")
    calibrate.TEST_PATH = str(folder / "test.csv")
    calibrate.main()


def check_gate(cutoff: str) -> dict:
    _modeling_path()
    import mlflow
    from features import build_features, get_target
    from mlflow.exceptions import MlflowException
    from train import compute_metrics

    mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlflow.db").as_posix())
    test_df = pd.read_csv(window_dir(cutoff) / "test.csv")
    X, y = build_features(test_df), get_target(test_df)

    def evaluate(alias: str) -> dict | None:
        try:
            model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{alias}")
        except MlflowException:
            return None
        metrics = compute_metrics(y, model.predict_proba(X)[:, 1])
        return {k: round(float(v), 4) for k, v in metrics.items()}

    cand, champ = evaluate("candidate"), evaluate("champion")
    if cand is None:
        raise RuntimeError("ไม่พบ alias candidate")

    reasons = []
    if cand["ece"] > MAX_ECE:
        reasons.append(f"ECE {cand['ece']} > {MAX_ECE}")
    if champ is not None and cand["pr_auc"] < champ["pr_auc"] - MAX_PR_AUC_DROP:
        reasons.append(f"PR-AUC {cand['pr_auc']} ตกจาก champion {champ['pr_auc']} เกิน {MAX_PR_AUC_DROP}")
    # TODO(บทบาทที่ 6): เกณฑ์กำไรจำลองไม่ต่ำกว่า champion (ใช้ overbooking_decision)

    report = {"cutoff": cutoff, "passed": not reasons, "reasons": reasons,
              "candidate": cand, "champion": champ,
              "rules": {"max_ece": MAX_ECE, "max_pr_auc_drop": MAX_PR_AUC_DROP}}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"gate_{cutoff}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"candidate: {cand}")
    print(f"champion : {champ if champ else '- (ยังไม่มี)'}")
    print("✅ GATE PASSED" if report["passed"] else f"❌ GATE FAILED: {'; '.join(reasons)}")
    return report


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in {"window", "calibrate", "gate"}:
        print(__doc__)
        return 2
    cmd, cutoff = sys.argv[1], sys.argv[2]
    if cmd == "window":
        build_window(cutoff)
    elif cmd == "calibrate":
        run_calibrate(cutoff)
    else:
        return 0 if check_gate(cutoff)["passed"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
