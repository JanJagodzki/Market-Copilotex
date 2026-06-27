from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class TargetDaily(Base):
    __tablename__ = "targets_daily"

    __table_args__ = (
        UniqueConstraint("asset_id", "date", name="uq_target_asset_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    date: Mapped[date] = mapped_column(Date, index=True, nullable=False)

    future_return_1d: Mapped[float | None] = mapped_column(Float, nullable=True)
    future_return_7d: Mapped[float | None] = mapped_column(Float, nullable=True)
    future_return_30d: Mapped[float | None] = mapped_column(Float, nullable=True)
    future_return_90d: Mapped[float | None] = mapped_column(Float, nullable=True)
    future_return_180d: Mapped[float | None] = mapped_column(Float, nullable=True)
    future_return_252d: Mapped[float | None] = mapped_column(Float, nullable=True)

    future_direction_1d: Mapped[int | None] = mapped_column(nullable=True)
    future_direction_7d: Mapped[int | None] = mapped_column(nullable=True)
    future_direction_30d: Mapped[int | None] = mapped_column(nullable=True)
    future_direction_90d: Mapped[int | None] = mapped_column(nullable=True)
    future_direction_180d: Mapped[int | None] = mapped_column(nullable=True)
    future_direction_252d: Mapped[int | None] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    asset = relationship("Asset")