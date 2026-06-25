from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.asset import Asset
from app.schemas.asset import AssetRead


router = APIRouter(prefix="/assets", tags=["assets"])


@router.get("", response_model=list[AssetRead])
def get_assets(
    db: Session = Depends(get_db),
    universe_name: str = Query(default="USA_TOP_100"),
    only_active: bool = Query(default=True),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[Asset]:
    query = select(Asset).where(Asset.universe_name == universe_name)

    if only_active:
        query = query.where(Asset.is_active.is_(True))

    query = query.order_by(Asset.universe_rank.asc()).limit(limit)

    return list(db.execute(query).scalars().all())


@router.get("/{symbol}", response_model=AssetRead)
def get_asset_by_symbol(
    symbol: str,
    db: Session = Depends(get_db),
) -> Asset:
    normalized_symbol = symbol.upper().replace(".", "-").replace("/", "-")

    asset = db.execute(
        select(Asset).where(Asset.symbol == normalized_symbol)
    ).scalar_one_or_none()

    if asset is None:
        raise HTTPException(
            status_code=404,
            detail=f"Asset {normalized_symbol} not found",
        )

    return asset