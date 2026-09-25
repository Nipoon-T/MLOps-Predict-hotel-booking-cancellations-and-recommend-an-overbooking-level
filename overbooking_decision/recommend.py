"""
โมดูลนี้คือจุดเชื่อมกับบทบาทที่ 4 (Serving)
เรียกใช้ผ่าน endpoint /recommend-overbooking ได้โดยตรง ไม่ต้องรู้รายละเอียด Monte Carlo ด้านใน
"""
from __future__ import annotations

from cost_config import CostConfig, load_config
from simulate import find_optimal_overbook


def recommend_overbooking(
    p_cancels: list[float],
    hotel: str,
    adr_ref: float,
    cost_config: CostConfig | None = None,
    k: float | None = None,
) -> dict:
    """
    Args:
      p_cancels : ความน่าจะเป็นยกเลิก (calibrate แล้ว) ของทุกการจองในคืนนั้น เรียงตามลำดับที่ควรรับก่อน-หลัง
      hotel     : ชื่อโรงแรม (ใช้หา capacity จาก config)
      adr_ref   : ราคาห้องอ้างอิงของคืนนั้น (ใช้คำนวณ cost_empty / cost_overbook)
      cost_config: ถ้าไม่ส่งมา จะโหลดจาก config.yaml อัตโนมัติ
      k         : ตัวคูณค่าชดเชย ถ้าไม่ส่งมาใช้ default_k จาก config

    Returns:
      dict พร้อม o_star (จำนวนห้องที่แนะนำให้ overbook), expected_cost, และ cost_curve (สำหรับ debug/log)
    """
    cost_config = cost_config or load_config()
    capacity = cost_config.capacity(hotel)
    cost_empty = cost_config.cost_empty(adr_ref)
    cost_overbook = cost_config.cost_overbook(adr_ref, k=k)

    result = find_optimal_overbook(
        p_cancels=p_cancels,
        capacity=capacity,
        cost_empty=cost_empty,
        cost_overbook=cost_overbook,
        n_sims=cost_config.n_sims,
        seed=cost_config.seed,
    )
    result["hotel"] = hotel
    result["capacity"] = capacity
    return result


if __name__ == "__main__":
    # ตัวอย่างเรียกใช้แบบง่าย
    example_p_cancels = [0.1, 0.2, 0.05, 0.3, 0.15] * 60  # 300 การจองตัวอย่าง
    out = recommend_overbooking(example_p_cancels, hotel="City Hotel", adr_ref=95.0)
    print(f"แนะนำ overbook {out['o_star']} ห้อง (ต้นทุนคาดหวัง {out['expected_cost']:.2f})")
