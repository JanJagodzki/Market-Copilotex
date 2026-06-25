from datetime import datetime

from pydantic import BaseModel


class AssetRead(BaseModel):
    id: int
    symbol: str
    name: str | None
    exchange: str | None
    currency: str | None
    sector: str | None
    industry: str | None
    country: str | None
    market_cap: int | None
    universe_name: str | None
    universe_rank: int | None
    is_active: bool
    data_source: str | None
    last_universe_update: datetime | None

    model_config = {
        "from_attributes": True
    }