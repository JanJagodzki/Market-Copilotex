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
