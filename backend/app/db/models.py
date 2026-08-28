from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)

from backend.app.db.database import Base


class Symbol(Base):
    __tablename__ = "symbols"

    id = Column(Integer, primary_key=True)

    ticker = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(255))
    primary_exchange = Column(String(20), index=True)

    active = Column(Boolean, default=True)

    list_date = Column(Date, nullable=True)
    delisted_date = Column(Date, nullable=True)


class DailyPrice(Base):
    __tablename__ = "daily_prices"

    id = Column(BigInteger, primary_key=True)

    symbol_id = Column(
        Integer,
        ForeignKey("symbols.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    date = Column(Date, nullable=False, index=True)

    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)

    adjusted_close = Column(Float)

    volume = Column(BigInteger)
    vwap = Column(Float)
    transactions = Column(BigInteger)

    source = Column(String(30), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "symbol_id",
            "date",
            name="uq_daily_prices_symbol_date",
        ),
    )


class DailyFeature(Base):
    __tablename__ = "daily_features"

    id = Column(BigInteger, primary_key=True)

    symbol_id = Column(
        Integer,
        ForeignKey("symbols.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    date = Column(Date, nullable=False, index=True)

    return_1d = Column(Float)
    return_5d = Column(Float)
    return_20d = Column(Float)

    volatility_5d = Column(Float)
    volatility_20d = Column(Float)

    sma_5_ratio = Column(Float)
    sma_20_ratio = Column(Float)
    sma_50_ratio = Column(Float)

    ema_12_ratio = Column(Float)
    ema_26_ratio = Column(Float)

    rsi_14 = Column(Float)
    macd_ratio = Column(Float)

    high_low_range = Column(Float)
    open_close_return = Column(Float)

    volume_change_1d = Column(Float)

    target_return_1d = Column(Float)
    target_direction_1d = Column(Integer)

    target_return_5d = Column(Float)
    target_direction_5d = Column(Integer)
    target_return_20d = Column(Float)
    target_direction_20d = Column(Integer)

    target_return_60d = Column(Float)
    target_direction_60d = Column(Integer)

    target_return_120d = Column(Float)
    target_direction_120d = Column(Integer)

    target_return_252d = Column(Float)
    target_direction_252d = Column(Integer)
    __table_args__ = (
        UniqueConstraint(
            "symbol_id",
            "date",
            name="uq_daily_features_symbol_date",
        ),
    )


class IntradayPrice(Base):
    __tablename__ = "intraday_prices"

    id = Column(BigInteger, primary_key=True)

    symbol_id = Column(
        Integer,
        ForeignKey("symbols.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    interval = Column(
        String(10),
        nullable=False,
        default="15m",
    )

    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger)

    source = Column(
        String(30),
        nullable=False,
        default="yfinance",
    )

    __table_args__ = (
        UniqueConstraint(
            "symbol_id",
            "timestamp",
            "interval",
            name="uq_intraday_prices_symbol_time_interval",
        ),
    )


class WatchlistItem(Base):
    __tablename__ = "watchlist_items"

    id = Column(BigInteger, primary_key=True)

    symbol_id = Column(
        Integer,
        ForeignKey("symbols.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(BigInteger, primary_key=True)

    model_family = Column(String(30), nullable=False)
    model_name = Column(String(80), nullable=False)
    horizon = Column(Integer, nullable=False)
    interval = Column(String(10), nullable=False, default="1d")
    version = Column(String(40), nullable=False)
    file_path = Column(String(500), nullable=False)

    trained_until = Column(Date, nullable=False)
    validation_auc = Column(Float)
    active = Column(Boolean, nullable=False, default=False, index=True)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "model_family",
            "model_name",
            "horizon",
            "interval",
            "version",
            name="uq_model_versions_identity",
        ),
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(BigInteger, primary_key=True)

    symbol_id = Column(
        Integer,
        ForeignKey("symbols.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    model_version_id = Column(
        BigInteger,
        ForeignKey("model_versions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    prediction_time = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    data_time = Column(
        DateTime(timezone=True),
        nullable=False,
    )

    interval = Column(String(10), nullable=False)
    horizon = Column(Integer, nullable=False)

    probability_up = Column(Float, nullable=False)
    predicted_direction = Column(Integer, nullable=False)
    reference_price = Column(Float, nullable=False)

    actual_return = Column(Float)
    correct = Column(Boolean)
    evaluated_at = Column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint(
            "symbol_id",
            "model_version_id",
            "data_time",
            "interval",
            "horizon",
            name="uq_predictions_model_data_horizon",
        ),
    )


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(BigInteger, primary_key=True)
    pipeline_name = Column(String(80), nullable=False, index=True)
    status = Column(String(20), nullable=False, default="running")

    started_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    finished_at = Column(DateTime(timezone=True))
    processed_rows = Column(BigInteger, nullable=False, default=0)
    message = Column(Text)
