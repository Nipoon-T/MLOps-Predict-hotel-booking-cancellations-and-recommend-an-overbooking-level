"""
schemas.py — กำหนดหน้าตาข้อมูลที่ API รับเข้า

ถ้าข้อมูลผิด เช่น lead_time ติดลบ, ผู้เข้าพักเป็นศูนย์ทั้งหมด, adr ติดลบ, เดือนไม่ถูกต้อง
FastAPI จะตอบ 422 ให้อัตโนมัติ โดยไม่ส่งข้อมูลเข้าโมเดลเลย
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# เดือนที่ยอมรับ (เขียนแบบเดียวกับในชุดข้อมูล)
Month = Literal[
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

HotelName = Literal["City Hotel", "Resort Hotel"]


class Booking(BaseModel):
    # ----- ข้อมูลโรงแรมและวันเข้าพัก -----
    hotel: HotelName
    lead_time: int = Field(ge=0, description="จำนวนวันระหว่างวันจองกับวันเข้าพัก ห้ามติดลบ")
    arrival_date_year: int = Field(ge=2000, le=2100)
    arrival_date_month: Month
    arrival_date_week_number: int = Field(ge=1, le=53)
    arrival_date_day_of_month: int = Field(ge=1, le=31)
    stays_in_weekend_nights: int = Field(ge=0)
    stays_in_week_nights: int = Field(ge=0)

    # ----- ผู้เข้าพัก -----
    adults: int = Field(ge=0)
    children: int | None = Field(default=0, ge=0)
    babies: int = Field(default=0, ge=0)

    # ----- รายละเอียดการจอง -----
    meal: str
    country: str | None = None
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int = Field(ge=0, le=1)
    previous_cancellations: int = Field(default=0, ge=0)
    previous_bookings_not_canceled: int = Field(default=0, ge=0)
    reserved_room_type: str
    booking_changes: int = Field(default=0, ge=0)
    deposit_type: Literal["No Deposit", "Non Refund", "Refundable"]
    agent: int | None = None
    company: int | None = None
    days_in_waiting_list: int = Field(default=0, ge=0)
    customer_type: str
    adr: float = Field(ge=0, description="ราคาห้องเฉลี่ยต่อคืน ห้ามติดลบ")
    required_car_parking_spaces: int = Field(default=0, ge=0)
    total_of_special_requests: int = Field(default=0, ge=0)

    # เช็กกฎที่ต้องดูหลายช่องพร้อมกัน
    @model_validator(mode="after")
    def check_total_guests(self):
        children = self.children if self.children is not None else 0
        total_guests = self.adults + children + self.babies
        if total_guests == 0:
            raise ValueError("ผู้เข้าพักเป็นศูนย์ทั้งหมด (adults + children + babies = 0)")
        return self

    # ตัวอย่างที่จะโชว์ในหน้า /docs (ใช้ตอนสาธิตได้เลย)
    model_config = {
        "json_schema_extra": {
            "example": {
                "hotel": "City Hotel",
                "lead_time": 120,
                "arrival_date_year": 2017,
                "arrival_date_month": "July",
                "arrival_date_week_number": 28,
                "arrival_date_day_of_month": 12,
                "stays_in_weekend_nights": 1,
                "stays_in_week_nights": 2,
                "adults": 2,
                "children": 0,
                "babies": 0,
                "meal": "BB",
                "country": "PRT",
                "market_segment": "Online TA",
                "distribution_channel": "TA/TO",
                "is_repeated_guest": 0,
                "previous_cancellations": 0,
                "previous_bookings_not_canceled": 0,
                "reserved_room_type": "A",
                "booking_changes": 0,
                "deposit_type": "No Deposit",
                "agent": 9,
                "company": None,
                "days_in_waiting_list": 0,
                "customer_type": "Transient",
                "adr": 110.5,
                "required_car_parking_spaces": 0,
                "total_of_special_requests": 1,
            }
        }
    }


# ตัวอย่างการจองใหม่ 2 รายการ สำหรับหน้า /docs ของ /recommend-overbooking
EXAMPLE_BOOKING_1 = dict(Booking.model_config["json_schema_extra"]["example"])
EXAMPLE_BOOKING_2 = dict(EXAMPLE_BOOKING_1)
EXAMPLE_BOOKING_2["lead_time"] = 5
EXAMPLE_BOOKING_2["market_segment"] = "Direct"
EXAMPLE_BOOKING_2["distribution_channel"] = "Direct"
EXAMPLE_BOOKING_2["is_repeated_guest"] = 1
EXAMPLE_BOOKING_2["agent"] = None


class OverbookingRequest(BaseModel):
    """การจองใหม่ของ 1 คืนที่รอตัดสินใจ + สภาพของโรงแรมคืนนั้น"""

    hotel: HotelName
    already_occupied: int = Field(
        ge=0,
        description="จำนวนห้องที่ถูกกันไว้แล้วจากแขกที่เช็กอินก่อนหน้าและยังพักอยู่คืนนี้ (บังคับส่ง)",
    )
    adr_ref: float | None = Field(
        default=None,
        gt=0,
        description="ราคาห้องอ้างอิงของคืนนั้น ถ้าไม่ส่งจะใช้ adr เฉลี่ยของการจองใหม่",
    )
    k: float | None = Field(
        default=None,
        gt=0,
        description="ตัวคูณค่าชดเชยเมื่อจองเกิน ถ้าไม่ส่งใช้ default_k จาก config.yaml",
    )
    bookings: list[Booking] = Field(min_length=1, max_length=1000)

    # การจองทุกรายการต้องเป็นของโรงแรมเดียวกับที่ขอคำแนะนำ
    @model_validator(mode="after")
    def check_same_hotel(self):
        for index in range(len(self.bookings)):
            booking_hotel = self.bookings[index].hotel
            if booking_hotel != self.hotel:
                raise ValueError(
                    f"การจองรายการที่ {index} เป็นของ {booking_hotel} ไม่ตรงกับ hotel={self.hotel}"
                )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "hotel": "City Hotel",
                "already_occupied": 189,
                "bookings": [EXAMPLE_BOOKING_1, EXAMPLE_BOOKING_2],
            }
        }
    }