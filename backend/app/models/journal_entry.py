from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB

from app.db.database import Base


class JournalEntry(Base):
    __tablename__ = "journal_entries"

    id = Column(BigInteger, primary_key=True, index=True)

    asset_id = Column(
        Integer,
        ForeignKey("assets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    horizon_days = Column(Integer, nullable=True, index=True)

    decision = Column(String(30), nullable=False, default="watch")
    status = Column(String(30), nullable=False, default="open")

    title = Column(String(255), nullable=False)
    thesis = Column(Text, nullable=True)
    plan = Column(Text, nullable=True)
    notes = Column(Text, nullable=True)

    entry_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    position_size = Column(Float, nullable=True)

    emotion = Column(String(50), nullable=True)
    confidence = Column(Integer, nullable=True)

    tags = Column(JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
