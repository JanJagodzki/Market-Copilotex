from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel


class MarketPriceDailyRead(BaseModel):
    id: int
    asset_id: int
    date: date
    open: Decimal | None
    high: Decimal | None
    low: Decimal | None
    close: Decimal | None
    adjusted_close: Decimal | None
    volume: int | None
    created_at: datetime

    model_config = {
        "from_attributes": True
    }