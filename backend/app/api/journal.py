from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.asset import Asset
from app.models.journal_entry import JournalEntry
from app.schemas.journal import JournalEntryCreate, JournalEntryUpdate


router = APIRouter(prefix="/journal", tags=["journal"])


def get_asset_by_symbol(db: Session, symbol: str) -> Asset:
    asset = db.execute(
        select(Asset).where(func.upper(Asset.symbol) == symbol.upper())
    ).scalar_one_or_none()

    if asset is None:
        raise HTTPException(status_code=404, detail=f"Company {symbol.upper()} not found")

    return asset


def serialize_journal_entry(entry: JournalEntry, asset: Asset) -> dict:
    return {
        "id": entry.id,
        "asset_id": entry.asset_id,
        "symbol": asset.symbol,
        "name": asset.name,
        "horizon_days": entry.horizon_days,
        "decision": entry.decision,
        "status": entry.status,
        "title": entry.title,
        "thesis": entry.thesis,
        "plan": entry.plan,
        "notes": entry.notes,
        "entry_price": entry.entry_price,
        "stop_loss": entry.stop_loss,
        "take_profit": entry.take_profit,
        "position_size": entry.position_size,
        "emotion": entry.emotion,
        "confidence": entry.confidence,
        "tags": entry.tags,
        "created_at": entry.created_at,
        "updated_at": entry.updated_at,
    }


@router.get("")
def list_journal_entries(
    symbol: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[dict]:
    query = (
        select(JournalEntry, Asset)
        .join(Asset, Asset.id == JournalEntry.asset_id)
        .order_by(JournalEntry.created_at.desc())
        .limit(limit)
    )

    if symbol is not None:
        query = query.where(func.upper(Asset.symbol) == symbol.upper())

    if status is not None:
        query = query.where(JournalEntry.status == status)

    rows = db.execute(query).all()

    return [
        serialize_journal_entry(entry=entry, asset=asset)
        for entry, asset in rows
    ]


@router.post("")
def create_journal_entry(
    payload: JournalEntryCreate,
    db: Session = Depends(get_db),
) -> dict:
    asset = get_asset_by_symbol(db, payload.symbol)

    entry = JournalEntry(
        asset_id=asset.id,
        horizon_days=payload.horizon_days,
        decision=payload.decision.lower(),
        status=payload.status.lower(),
        title=payload.title,
        thesis=payload.thesis,
        plan=payload.plan,
        notes=payload.notes,
        entry_price=payload.entry_price,
        stop_loss=payload.stop_loss,
        take_profit=payload.take_profit,
        position_size=payload.position_size,
        emotion=payload.emotion,
        confidence=payload.confidence,
        tags=payload.tags,
    )

    db.add(entry)
    db.commit()
    db.refresh(entry)

    return serialize_journal_entry(entry=entry, asset=asset)


@router.get("/{entry_id}")
def get_journal_entry(
    entry_id: int,
    db: Session = Depends(get_db),
) -> dict:
    row = db.execute(
        select(JournalEntry, Asset)
        .join(Asset, Asset.id == JournalEntry.asset_id)
        .where(JournalEntry.id == entry_id)
    ).one_or_none()

    if row is None:
        raise HTTPException(status_code=404, detail="Journal entry not found")

    entry, asset = row

    return serialize_journal_entry(entry=entry, asset=asset)


@router.put("/{entry_id}")
def update_journal_entry(
    entry_id: int,
    payload: JournalEntryUpdate,
    db: Session = Depends(get_db),
) -> dict:
    row = db.execute(
        select(JournalEntry, Asset)
        .join(Asset, Asset.id == JournalEntry.asset_id)
        .where(JournalEntry.id == entry_id)
    ).one_or_none()

    if row is None:
        raise HTTPException(status_code=404, detail="Journal entry not found")

    entry, asset = row

    update_data = payload.model_dump(exclude_unset=True)

    for key, value in update_data.items():
        if key in {"decision", "status"} and value is not None:
            value = value.lower()

        setattr(entry, key, value)

    db.commit()
    db.refresh(entry)

    return serialize_journal_entry(entry=entry, asset=asset)


@router.delete("/{entry_id}")
def delete_journal_entry(
    entry_id: int,
    db: Session = Depends(get_db),
) -> dict:
    entry = db.get(JournalEntry, entry_id)

    if entry is None:
        raise HTTPException(status_code=404, detail="Journal entry not found")

    db.delete(entry)
    db.commit()

    return {
        "deleted": True,
        "entry_id": entry_id,
    }
