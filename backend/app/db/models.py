from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
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
