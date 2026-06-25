from datetime import datetime, timezone
from typing import Any

import requests
from sqlalchemy import select, update

from app.db.database import SessionLocal
from app.models.asset import Asset


NASDAQ_STOCK_SCREENER_URL = (
    "https://api.nasdaq.com/api/screener/stocks"
    "?tableonly=true&limit=10000&offset=0&download=true"
)

UNIVERSE_NAME = "USA_TOP_100"
DATA_SOURCE = "nasdaq_screener"


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/market-activity/stocks/screener",
}


EXCLUDED_NAME_PARTS = [
    "warrant",
    "warrants",
    "unit",
    "units",
    "right",
    "rights",
    "preferred",
    "preference",
    "depositary",
    "note",
    "notes",
    "bond",
    "etf",
    "etn",
    "fund",
]


def clean_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return None

    return text


def parse_market_cap(value: Any) -> int | None:
    text = clean_text(value)

    if text is None:
        return None

    text = text.replace("$", "").replace(",", "").strip()

    if not text:
        return None

    try:
        market_cap = int(float(text))
    except ValueError:
        return None

    if market_cap <= 0:
        return None

    return market_cap


def normalize_symbol(symbol: str) -> str:
    """
    Normalize symbols for later usage with yfinance.
    Example: BRK/B or BRK.B -> BRK-B
    """
    return symbol.strip().upper().replace("/", "-").replace(".", "-")


def is_common_equity(symbol: str, name: str | None) -> bool:
    raw_symbol = symbol.strip().upper()

    if "^" in raw_symbol or "$" in raw_symbol:
        return False

    if name is None:
        return False

    lowered_name = name.lower()

    return not any(part in lowered_name for part in EXCLUDED_NAME_PARTS)


def fetch_nasdaq_rows() -> list[dict[str, Any]]:
    response = requests.get(
        NASDAQ_STOCK_SCREENER_URL,
        headers=HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()

    rows = payload.get("data", {}).get("rows", [])

    if not rows:
        raise RuntimeError("Nasdaq screener returned no rows.")

    return rows


def fetch_usa_top_100_assets() -> list[dict[str, Any]]:
    rows = fetch_nasdaq_rows()

    candidates: list[dict[str, Any]] = []

    for row in rows:
        raw_symbol = clean_text(row.get("symbol"))
        name = clean_text(row.get("name"))
        country = clean_text(row.get("country"))
        sector = clean_text(row.get("sector"))
        industry = clean_text(row.get("industry"))
        market_cap = parse_market_cap(row.get("marketCap"))

        if raw_symbol is None:
            continue

        if country != "United States":
            continue

        if market_cap is None:
            continue

        if not is_common_equity(raw_symbol, name):
            continue

        candidates.append(
            {
                "symbol": normalize_symbol(raw_symbol),
                "name": name,
                "exchange": None,
                "currency": "USD",
                "sector": sector,
                "industry": industry,
                "country": country,
                "market_cap": market_cap,
                "universe_name": UNIVERSE_NAME,
                "data_source": DATA_SOURCE,
            }
        )

    candidates.sort(key=lambda item: item["market_cap"], reverse=True)

    top_assets = candidates[:100]

    for rank, asset in enumerate(top_assets, start=1):
        asset["universe_rank"] = rank

    return top_assets


def update_usa_top_100_assets() -> dict[str, int]:
    top_assets = fetch_usa_top_100_assets()
    now = datetime.now(timezone.utc)

    created_count = 0
    updated_count = 0

    with SessionLocal() as session:
        session.execute(
            update(Asset)
            .where(Asset.universe_name == UNIVERSE_NAME)
            .values(
                is_active=False,
                universe_rank=None,
                last_universe_update=now,
            )
        )

        for asset_data in top_assets:
            symbol = asset_data["symbol"]

            existing_asset = session.execute(
                select(Asset).where(Asset.symbol == symbol)
            ).scalar_one_or_none()

            if existing_asset:
                existing_asset.name = asset_data["name"]
                existing_asset.exchange = asset_data["exchange"]
                existing_asset.currency = asset_data["currency"]
                existing_asset.sector = asset_data["sector"]
                existing_asset.industry = asset_data["industry"]
                existing_asset.country = asset_data["country"]
                existing_asset.market_cap = asset_data["market_cap"]
                existing_asset.universe_name = asset_data["universe_name"]
                existing_asset.universe_rank = asset_data["universe_rank"]
                existing_asset.is_active = True
                existing_asset.data_source = asset_data["data_source"]
                existing_asset.last_universe_update = now
                updated_count += 1
            else:
                asset = Asset(
                    symbol=asset_data["symbol"],
                    name=asset_data["name"],
                    exchange=asset_data["exchange"],
                    currency=asset_data["currency"],
                    sector=asset_data["sector"],
                    industry=asset_data["industry"],
                    country=asset_data["country"],
                    market_cap=asset_data["market_cap"],
                    universe_name=asset_data["universe_name"],
                    universe_rank=asset_data["universe_rank"],
                    is_active=True,
                    data_source=asset_data["data_source"],
                    last_universe_update=now,
                )
                session.add(asset)
                created_count += 1

        session.commit()

    return {
        "fetched": len(top_assets),
        "created": created_count,
        "updated": updated_count,
    }