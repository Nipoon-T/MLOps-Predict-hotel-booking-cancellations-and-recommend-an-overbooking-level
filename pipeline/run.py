"""คำสั่งเดียวรันทั้งระบบ

ตัวอย่าง (รันจากโฟลเดอร์ repo, activate .venv แล้ว):
  python -m pipeline.run                      # ทั้งหมด: ข้อมูล → เทรน → export → เปิด API
  python -m pipeline.run --skip-train         # ใช้โมเดลที่เทรนไว้แล้ว
  python -m pipeline.run --no-deploy          # ไม่เปิด Docker
  python -m pipeline.run --redeploy champion  # promote/rollback: export alias นี้แล้วรีสตาร์ท API
  python -m pipeline.run --demo-bad-data      # สาธิต: ข้อมูลเสียถูกหยุดที่ gate
  python -m pipeline.run --promote            # candidate -> champion แล้วให้ API ใช้ทันที
  python -m pipeline.run --promote challenger # alias อื่น -> champion
  python -m pipeline.run --rollback           # champion กลับเป็นตัวก่อนหน้า แล้วให้ API ใช้ทันที
  python -m pipeline.run --simulate           # จำลอง production รายสัปดาห์ (โหมดทดลอง ไม่แตะโมเดล)
  python -m pipeline.run --simulate --weeks 8 --retrain-on-drift   # retrain + promote จริงเมื่อเจอ drift
"""

import argparse
import sys

from pipeline.flow import (
    StepFailed,
    demo_bad_data,
    deploy_model,
    full_pipeline,
    promote_and_deploy,
    rollback_and_deploy,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hotel booking MLOps pipeline")
    parser.add_argument("--alias", default="candidate", help="alias ที่จะ export ไปให้ API")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--no-deploy", action="store_true")
    parser.add_argument("--redeploy", metavar="ALIAS",
                        help="export alias นี้แล้วรีสตาร์ท API อย่างเดียว")
    parser.add_argument("--promote", nargs="?", const="candidate", metavar="ALIAS",
                        help="promote alias (ค่าเริ่มต้น candidate) เป็น champion แล้ว redeploy")
    parser.add_argument("--rollback", action="store_true",
                        help="ย้อน champion เป็นตัวก่อนหน้าแล้ว redeploy")
    parser.add_argument("--simulate", action="store_true",
                        help="จำลอง production รายสัปดาห์")
    parser.add_argument("--weeks", type=int, default=None,
                        help="จำนวนสัปดาห์ที่จะจำลอง (ค่าเริ่มต้น: ทั้งหมด)")
    parser.add_argument("--cooldown", type=int, default=4,
                        help="retrain ห่างกันอย่างน้อยกี่สัปดาห์")
    parser.add_argument("--retrain-on-drift", action="store_true",
                        help="retrain + promote + redeploy จริงเมื่อเจอ drift")
    parser.add_argument("--drift-cmd", default=None,
                        help="คำสั่งตรวจ drift ใช้ {week_csv} และ {out_json}")
    parser.add_argument("--demo-bad-data", action="store_true",
                        help="สาธิตว่าข้อมูลเสียถูกหยุดที่ validation gate")
    args = parser.parse_args()

    if args.demo_bad_data:
        try:
            demo_bad_data()
        except StepFailed as exc:
            if "ผ่าน gate ได้" in str(exc):
                print(f"\n❌ DEMO FAILED: {exc}", file=sys.stderr)
                return 1
            print(
                "\n🛑 PIPELINE STOPPED AT VALIDATION GATE (ผลที่ถูกต้อง)\n"
                "   ไฟล์: data/bad/hotel_bookings_bad.csv\n"
                "   ข้อมูลเสียไม่ถูกส่งต่อไป clean / train / deploy\n"
                "   API และโมเดลที่ใช้อยู่ไม่ถูกแตะต้อง"
            )
            return 1
        return 1

    try:
        if args.simulate:
            from pipeline.production_sim import DEFAULT_DRIFT_CMD, production_simulation
            production_simulation(
                max_weeks=args.weeks,
                cooldown_weeks=args.cooldown,
                retrain=args.retrain_on_drift,
                drift_cmd=args.drift_cmd or DEFAULT_DRIFT_CMD,
            )
        elif args.promote:
            promote_and_deploy(source_alias=args.promote)
        elif args.rollback:
            rollback_and_deploy()
        elif args.redeploy:
            deploy_model(alias=args.redeploy, restart=True)
        else:
            full_pipeline(
                alias=args.alias,
                skip_data=args.skip_data,
                skip_train=args.skip_train,
                skip_tests=args.skip_tests,
                deploy=not args.no_deploy,
            )
    except StepFailed as exc:
        print(f"\n❌ PIPELINE STOPPED: {exc}", file=sys.stderr)
        return 1
    print("\n✅ PIPELINE FINISHED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
