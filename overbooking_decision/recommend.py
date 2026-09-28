"""
โมดูลนี้คือจุดเชื่อมกับบทบาทที่ 4 (Serving)
เรียกใช้ผ่าน endpoint /recommend-overbooking ได้โดยตรง ไม่ต้องรู้รายละเอียด Monte Carlo ด้านใน

v2 (2026-09-26): เพิ่มพารามิเตอร์ already_occupied เพื่อให้สอดคล้องกับ backtest_sequential.py —
capacity เต็มของโรงแรมไม่ใช่ตัวเลขที่ควรส่งเข้า Monte Carlo ตรงๆ ถ้าคืนนั้นมีแขกที่เช็กอินมาก่อนแล้ว
ยังพักอยู่ (จองหลายคืน) ต้องหักจำนวนนั้นออกก่อน ไม่งั้นระบบจะแนะนำ overbook เกินจริง
"""
from __future__ import annotations

try:  # import แบบแพ็กเกจ: from overbooking_decision.recommend import recommend_overbooking
    from .cost_config import CostConfig, load_config
    from .simulate import find_optimal_overbook
except ImportError:  # รันตรงๆ ในโฟลเดอร์: python recommend.py
    from cost_config import CostConfig, load_config
    from simulate import find_optimal_overbook


def recommend_overbooking(
    p_cancels: list[float],
    hotel: str,
    adr_ref: float,
    already_occupied: int = 0,
    cost_config: CostConfig | None = None,
    k: float | None = None,
) -> dict:
    """
    Args:
      p_cancels : ความน่าจะเป็นยกเลิก (calibrate แล้ว) ของการจอง "ใหม่" ที่กำลังรอตัดสินใจของคืนนั้น
                  เรียงตามลำดับที่ควรรับก่อน-หลัง (เช่น lead_time มาก = จองก่อน = ควรได้สิทธิ์ก่อน)
      hotel     : ชื่อโรงแรม (ใช้หา capacity เต็มจาก config)
      adr_ref   : ราคาห้องอ้างอิงของคืนนั้น (ใช้คำนวณ cost_empty / cost_overbook)
      already_occupied : จำนวนห้องที่ถูก "กันไว้แล้ว" จากการจองเดิมที่ยังไม่เช็กเอาท์ในคืนนั้น
                  (แขกที่เช็กอินมาก่อนแล้วพักต่อเนื่องหลายคืน) — ดึงจากระบบจองของโรงแรม (PMS) ตอนเรียก endpoint
                  **สำคัญ: ถ้าไม่ส่งค่านี้ (ปล่อย default 0) ระบบจะคิดว่าห้องว่างทั้งหมด แล้วแนะนำ overbook
                  เกินจริงมาก** โดยเฉพาะคืนกลางๆ ของช่วงที่มีแขกพักยาวเยอะ — เป็นสิ่งที่ role 3 เจอบั๊กจริงตอนทำ
                  backtest ต้องแก้เป็น sequential/carryover-aware ถึงจะได้ผลลัพธ์ที่เชื่อถือได้
      cost_config: ถ้าไม่ส่งมา จะโหลดจาก config.yaml อัตโนมัติ
      k         : ตัวคูณค่าชดเชย ถ้าไม่ส่งมาใช้ default_k จาก config

    Returns:
      dict พร้อม o_star (จำนวนห้องที่แนะนำให้ overbook สำหรับการจอง "ใหม่" ที่ส่งเข้ามา),
      effective_capacity, expected_cost, และ cost_curve (สำหรับ debug/log)
    """
    cost_config = cost_config or load_config()
    capacity = cost_config.capacity(hotel)
    effective_capacity = max(0, capacity - already_occupied)
    cost_empty = cost_config.cost_empty(adr_ref)
    cost_overbook = cost_config.cost_overbook(adr_ref, k=k)

    result = find_optimal_overbook(
        p_cancels=p_cancels,
        capacity=effective_capacity,
        cost_empty=cost_empty,
        cost_overbook=cost_overbook,
        n_sims=cost_config.n_sims,
        seed=cost_config.seed,
    )
    result["hotel"] = hotel
    result["capacity"] = capacity
    result["already_occupied"] = already_occupied
    result["effective_capacity"] = effective_capacity
    return result


if __name__ == "__main__":
    # ตัวอย่างเรียกใช้แบบง่าย: คืนนี้มีแขกเดิมพักอยู่แล้ว 120 ห้อง (จากการจองหลายคืนก่อนหน้า)
    example_p_cancels = [0.1, 0.2, 0.05, 0.3, 0.15] * 12  # 60 การจองใหม่ที่รอตัดสินใจคืนนี้
    out = recommend_overbooking(
        example_p_cancels, hotel="City Hotel", adr_ref=95.0, already_occupied=120,
    )
    print(f"แนะนำ overbook {out['o_star']} ห้อง สำหรับการจองใหม่ "
          f"(effective_capacity={out['effective_capacity']}, ต้นทุนคาดหวัง {out['expected_cost']:.2f})")
