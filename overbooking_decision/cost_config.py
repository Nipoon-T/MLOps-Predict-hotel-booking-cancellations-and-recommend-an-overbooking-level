"""
โหลดสมมติฐานต้นทุนจาก config.yaml
ทุกฟังก์ชันในโมดูลอื่นควรรับค่าพวกนี้เป็นพารามิเตอร์ ไม่ hardcode เอง
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass
class CostConfig:
    capacity_by_hotel: dict
    default_capacity: int
    k_values: list
    default_k: float
    fixed_overbook_pct: float
    n_sims: int
    seed: int
    o_max_pct: float

    def capacity(self, hotel: str) -> int:
        return self.capacity_by_hotel.get(hotel, self.default_capacity)

    def cost_empty(self, adr: float) -> float:
        """ต้นทุนห้องว่าง = adr ของคืนนั้น (รายได้ที่เสียไป)"""
        return float(adr)

    def cost_overbook(self, adr: float, k: float | None = None) -> float:
        """ต้นทุนจองเกิน = k * adr"""
        k = self.default_k if k is None else k
        return float(k) * float(adr)


def load_config(path: str | Path = CONFIG_PATH) -> CostConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    mc = raw["monte_carlo"]
    return CostConfig(
        capacity_by_hotel=raw["capacity"],
        default_capacity=raw["default_capacity"],
        k_values=raw["k_values"],
        default_k=raw["default_k"],
        fixed_overbook_pct=raw["fixed_overbook_pct"],
        n_sims=mc["n_sims"],
        seed=mc["seed"],
        o_max_pct=mc["o_max_pct"],
    )
