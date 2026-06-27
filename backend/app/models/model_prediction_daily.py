from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class ModelPredictionDaily(Base):
    __tablename__ = "model_predictions_daily"

    __table_args__ = (
        UniqueConstraint(
            "asset_id",
            "date",
            "horizon_days",
            "model_name",
            name="uq_model_prediction_asset_date_horizon_model",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    asset_id: Mapped[int] = mapped_column(
        ForeignKey("assets.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    date: Mapped[date] = mapped_column(Date, index=True, nullable=False)

    horizon_days: Mapped[int] = mapped_column(nullable=False, default=30)

    model_name: Mapped[str] = mapped_column(String(100), nullable=False)

    predicted_return: Mapped[float] = mapped_column(Float, nullable=False)

    prediction_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    final_score: Mapped[float] = mapped_column(Float, nullable=False)

    prediction_rank: Mapped[int] = mapped_column(nullable=False)

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