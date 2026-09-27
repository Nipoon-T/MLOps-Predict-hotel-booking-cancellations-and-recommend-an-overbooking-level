import time

from joblib import parallel_config

from src.modeling.features import build_features
from src.serving import app as app_module
from src.serving.schemas import Booking

N_ROUNDS = 200  # ทำซ้ำหลายรอบแล้วเฉลี่ย เพื่อให้ตัวเลขนิ่ง

EXAMPLE = Booking.model_config["json_schema_extra"]["example"]


def average_ms(function):
    """เรียก function ซ้ำ N_ROUNDS ครั้ง แล้วคืนเวลาเฉลี่ยต่อครั้ง (ms)"""
    function()  # รอบแรกไม่นับ (อุ่นเครื่อง)
    start = time.perf_counter()
    for _ in range(N_ROUNDS):
        function()
    total_seconds = time.perf_counter() - start
    return total_seconds / N_ROUNDS * 1000


def main():
    model = app_module.model
    if model is None:
        print("โหลดโมเดลไม่ได้ ต้องรัน calibrate.py ก่อน")
        return

    # แกะโมเดลออกเป็นชิ้นๆ: CalibratedClassifierCV -> FrozenEstimator -> Pipeline
    pipeline = model.calibrated_classifiers_[0].estimator.estimator
    preprocess = pipeline.named_steps["preprocess"]
    forest = pipeline.named_steps["model"]
    print(f"จำนวนต้นไม้ใน Random Forest: {forest.n_estimators}, n_jobs: {forest.n_jobs}")

    # เตรียมข้อมูลของแต่ละขั้นไว้ก่อน จะได้วัดแยกทีละขั้นได้
    booking = Booking(**EXAMPLE)
    df = app_module.booking_to_dataframe(booking)
    X = build_features(df)
    X_encoded = preprocess.transform(X)

    def step_validate():
        Booking(**EXAMPLE)

    def step_to_dataframe():
        app_module.booking_to_dataframe(booking)

    def step_build_features():
        build_features(df)

    def step_preprocess():
        preprocess.transform(X)

    def step_forest():
        with parallel_config(backend="sequential"):
            forest.predict_proba(X_encoded)

    def step_whole_model():
        with parallel_config(backend="sequential"):
            model.predict_proba(X)

    def step_whole_model_multithread():
        model.predict_proba(X)

    results = [
        ("1. ตรวจข้อมูล (pydantic)", average_ms(step_validate)),
        ("2. แปลงเป็น DataFrame", average_ms(step_to_dataframe)),
        ("3. build_features()", average_ms(step_build_features)),
        ("4. preprocess (impute/scale/one-hot)", average_ms(step_preprocess)),
        ("5. Random Forest อย่างเดียว", average_ms(step_forest)),
        ("6. โมเดลทั้งก้อน (4+5+calibrate) thread เดียว", average_ms(step_whole_model)),
        ("7. โมเดลทั้งก้อน แบบ n_jobs=-1 (เดิม)", average_ms(step_whole_model_multithread)),
    ]

    print()
    print(f"เวลาเฉลี่ยต่อ 1 การจอง (เฉลี่ยจาก {N_ROUNDS} รอบ)")
    print("-" * 60)
    for name, ms in results:
        print(f"{name:<48} {ms:8.2f} ms")


if __name__ == "__main__":
    main()
