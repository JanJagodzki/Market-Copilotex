from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class JournalEntryCreate(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    horizon_days: int | None = None

    decision: str = Field(default="watch", max_length=30)
    status: str = Field(default="open", max_length=30)

    title: str = Field(..., min_length=1, max_length=255)
    thesis: str | None = None
    plan: str | None = None
    notes: str | None = None

    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    position_size: float | None = None

    emotion: str | None = Field(default=None, max_length=50)
    confidence: int | None = Field(default=None, ge=1, le=10)

    tags: list[str] | None = None


class JournalEntryUpdate(BaseModel):
    horizon_days: int | None = None

    decision: str | None = Field(default=None, max_length=30)
    status: str | None = Field(default=None, max_length=30)

    title: str | None = Field(default=None, min_length=1, max_length=255)
    thesis: str | None = None
    plan: str | None = None
    notes: str | None = None

    entry_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    position_size: float | None = None

    emotion: str | None = Field(default=None, max_length=50)
    confidence: int | None = Field(default=None, ge=1, le=10)

    tags: list[str] | None = None


class JournalEntryRead(BaseModel):
    id: int

    asset_id: int
    symbol: str
    name: str | None

    horizon_days: int | None

    decision: str
    status: str

    title: str
    thesis: str | None
    plan: str | None
    notes: str | None

    entry_price: float | None
    stop_loss: float | None
    take_profit: float | None
    position_size: float | None

    emotion: str | None
    confidence: int | None

    tags: Any | None

    created_at: datetime
    updated_at: datetime
