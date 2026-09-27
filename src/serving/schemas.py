from typing import Literal

from pydantic import BaseModel, Field, model_validator

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


class Booking(BaseModel):
    # ----- ข้อมูลโรงแรมและวันเข้าพัก -----
    hotel: Literal["City Hotel", "Resort Hotel"]
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
