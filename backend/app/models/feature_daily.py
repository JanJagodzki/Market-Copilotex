from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class FeatureDaily(Base):
    __tablename__ = "features_daily"

    __table_args__ = (
        UniqueConstraint("asset_id", "date", name="uq_feature_asset_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    date: Mapped[date] = mapped_column(Date, index=True, nullable=False)

    daily_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    log_return: Mapped[float | None] = mapped_column(Float, nullable=True)

    volume_change: Mapped[float | None] = mapped_column(Float, nullable=True)

    volatility_7d: Mapped[float | None] = mapped_column(Float, nullable=True)
    volatility_30d: Mapped[float | None] = mapped_column(Float, nullable=True)

    sma_20: Mapped[float | None] = mapped_column(Float, nullable=True)
    sma_50: Mapped[float | None] = mapped_column(Float, nullable=True)
    sma_200: Mapped[float | None] = mapped_column(Float, nullable=True)

    distance_from_sma_20: Mapped[float | None] = mapped_column(Float, nullable=True)
    distance_from_sma_200: Mapped[float | None] = mapped_column(Float, nullable=True)

    momentum_7d: Mapped[float | None] = mapped_column(Float, nullable=True)
    momentum_30d: Mapped[float | None] = mapped_column(Float, nullable=True)
    momentum_90d: Mapped[float | None] = mapped_column(Float, nullable=True)

    drawdown_252d: Mapped[float | None] = mapped_column(Float, nullable=True)

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