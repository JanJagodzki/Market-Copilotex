from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    UniqueConstraint,
    func,
)

from app.db.database import Base


class TargetHorizonDaily(Base):
    __tablename__ = "targets_horizon_daily"

    id = Column(BigInteger, primary_key=True, index=True)

    asset_id = Column(
        Integer,
        ForeignKey("assets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    date = Column(Date, nullable=False, index=True)

    horizon_days = Column(Integer, nullable=False, index=True)

    future_return = Column(Float, nullable=True)
    future_direction = Column(Boolean, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "asset_id",
            "date",
            "horizon_days",
            name="uq_targets_horizon_daily_asset_date_horizon",
        ),
        Index(
            "ix_targets_horizon_daily_horizon_date",
            "horizon_days",
            "date",
        ),
    )