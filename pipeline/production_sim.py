"""จำลอง production รายสัปดาห์ (บทบาทที่ 5)

ป้อน data/processed/production.csv เข้าระบบทีละสัปดาห์ตามวันเข้าพัก
แต่ละสัปดาห์: ตรวจ drift → ถ้าตัวตรวจบอกให้ retrain และพ้น cooldown
→ retrain บนข้อมูลช่วงล่าสุด → gate → (ผ่าน) promote → redeploy / (ไม่ผ่าน) ใช้ champion เดิม

ตัวตรวจ drift เสียบเปลี่ยนได้ผ่าน drift_cmd (ค่าเริ่มต้นคือตัวปลอม pipeline/mock_drift.py)
สคริปต์จริงของบทบาทที่ 6 ต้องรับ <week_csv> <out_json> และเขียน JSON:
  {"week": int, "retrain": bool, "reasons": [str, ...]}

label มาช้า: รู้ว่ายกเลิกจริงหรือไม่ก็ต่อเมื่อถึงวันเข้าพัก จึงตรวจสัปดาห์ใดได้
หลังสัปดาห์นั้นผ่านไปแล้วเท่านั้น (ข้อมูลแต่ละไฟล์คือการจองที่วันเข้าพักอยู่ในสัปดาห์นั้น)
"""

import json
import shlex
import sys
from pathlib import Path

import pandas as pd
from prefect import flow, get_run_logger, task

from pipeline.flow import (
    PROJECT_ROOT,
    StepFailed,
    _py,
    _run,
    promote_and_deploy,
)

PRODUCTION_FILE = PROJECT_ROOT / "data" / "processed" / "production.csv"
WEEKS_DIR = PROJECT_ROOT / "data" / "production_weeks"
MONITOR_DIR = PROJECT_ROOT / "reports" / "monitoring"
DEFAULT_DRIFT_CMD = f'"{sys.executable}" pipeline/mock_drift.py {{week_csv}} {{out_json}}'


@task(name="split-production-weeks")
def split_production_weeks() -> list[Path]:
    """แบ่ง production.csv เป็น week_01.csv, week_02.csv, ... ตามวันเข้าพัก"""
    logger = get_run_logger()
    if not PRODUCTION_FILE.exists():
        raise StepFailed(f"ไม่พบ {PRODUCTION_FILE} ต้องรัน data pipeline ก่อน")

    df = pd.read_csv(PRODUCTION_FILE)
    arrival = pd.to_datetime(df["arrival_date"])
    week_no = (arrival - arrival.min()).dt.days // 7 + 1

    WEEKS_DIR.mkdir(parents=True, exist_ok=True)
    for old in WEEKS_DIR.glob("week_*.csv"):
        old.unlink()

    paths = []
    for week, part in df.groupby(week_no):
        path = WEEKS_DIR / f"week_{int(week):02d}.csv"
        part.to_csv(path, index=False)
        paths.append(path)
    logger.info("แบ่ง production %s แถว เป็น %s สัปดาห์ (%s ถึง %s)",
                f"{len(df):,}", len(paths), arrival.min().date(), arrival.max().date())
    return paths


@task(name="check-drift")
def check_drift(week_csv: Path, drift_cmd: str) -> dict:
    out_json = MONITOR_DIR / f"{week_csv.stem}.json"
    cmd = shlex.split(
        drift_cmd.format(week_csv=f'"{week_csv}"', out_json=f'"{out_json}"'),
        posix=False,
    )
    cmd = [c.strip('"') for c in cmd]
    _run(cmd, f"check drift {week_csv.stem}")
    if not out_json.exists():
        raise StepFailed(f"ตัวตรวจ drift ไม่ได้เขียน {out_json}")
    result = json.loads(out_json.read_text(encoding="utf-8"))
    missing = {"week", "retrain", "reasons"} - result.keys()
    if missing:
        raise StepFailed(f"ผล drift ขาด key {sorted(missing)} (ดูข้อตกลงใน production_sim.py)")
    return result


@task(name="week-stats")
def week_stats(week_csv: Path) -> dict:
    df = pd.read_csv(week_csv)
    # cutoff = วันหลังสัปดาห์นี้ (label ของทั้งสัปดาห์รู้ครบแล้ว)
    cutoff = (pd.to_datetime(df["arrival_date"]).max() + pd.Timedelta(days=1)).date()
    return {"rows": len(df), "cancel_rate": round(float(df["is_canceled"].mean()), 4),
            "cutoff": str(cutoff)}


@task(name="retrain-window")
def retrain_window(cutoff: str) -> None:
    _run(_py("-m", "pipeline.retrain", "window", cutoff), f"สร้างหน้าต่างข้อมูล ถึง {cutoff}")


@task(name="retrain-calibrate")
def retrain_calibrate(cutoff: str) -> None:
    _run(_py("-m", "pipeline.retrain", "calibrate", cutoff), "retrain + calibrate + register")


@task(name="gate")
def gate(cutoff: str) -> bool:
    # exit 1 = ไม่ผ่าน gate ซึ่งเป็นผลปกติ ไม่ใช่ pipeline พัง
    return _run(_py("-m", "pipeline.retrain", "gate", cutoff), "gate", allow_fail=True) == 0


@flow(name="production-simulation")
def production_simulation(
    max_weeks: int | None = None,
    cooldown_weeks: int = 4,
    retrain: bool = False,
    drift_cmd: str = DEFAULT_DRIFT_CMD,
) -> list[dict]:
    """วนทีละสัปดาห์

    retrain=False (ค่าเริ่มต้น): บันทึกอย่างเดียวว่าสัปดาห์ไหน "จะ" retrain ไม่แตะโมเดล/API
    retrain=True: retrain → promote → redeploy จริง (ใช้ตอนสาธิตวงจรเต็ม)
    """
    logger = get_run_logger()
    weeks = split_production_weeks()
    if max_weeks:
        weeks = weeks[:max_weeks]

    summary = []
    last_retrain_week = None
    for week_csv in weeks:
        stats = week_stats(week_csv)
        result = check_drift(week_csv, drift_cmd)
        week = result["week"]
        action = "-"

        if result["retrain"]:
            in_cooldown = (
                last_retrain_week is not None
                and week - last_retrain_week < cooldown_weeks
            )
            if in_cooldown:
                action = f"ข้าม (cooldown ถึงสัปดาห์ {last_retrain_week + cooldown_weeks})"
            elif not retrain:
                action = "จะ retrain (โหมดทดลอง)"
                last_retrain_week = week
            else:
                logger.info("สัปดาห์ %s: retrain เพราะ %s", week, result["reasons"])
                retrain_window(stats["cutoff"])
                retrain_calibrate(stats["cutoff"])
                if gate(stats["cutoff"]):
                    promote_and_deploy(source_alias="candidate")
                    action = "retrain → gate ผ่าน → promote → redeploy"
                else:
                    action = "retrain → gate ไม่ผ่าน → ใช้ champion เดิม"
                last_retrain_week = week

        row = {"week": week, "rows": stats["rows"], "cancel_rate": stats["cancel_rate"], "retrain_signal": result["retrain"],
               "reasons": "; ".join(result["reasons"]), "action": action}
        summary.append(row)
        logger.info("สัปดาห์ %02d | %5s แถว | cancel %.3f | %s",
                    week, stats["rows"], stats["cancel_rate"], action)

    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    out = MONITOR_DIR / "simulation_summary.csv"
    pd.DataFrame(summary).to_csv(out, index=False, encoding="utf-8-sig")
    logger.info("สรุปรายสัปดาห์: %s", out)
    return summary
