from backend.app.data.nasdaq_symbols import get_nasdaq_symbols
from backend.app.db.database import SessionLocal
from backend.app.db.models import Symbol


def sync_symbols():
    stocks = get_nasdaq_symbols()
    db = SessionLocal()

    try:
        existing = {
            symbol.ticker: symbol
            for symbol in db.query(Symbol).all()
        }

        for symbol in existing.values():
            symbol.active = False

        added = 0

        for row in stocks.itertuples(index=False):
            ticker = row.ticker.strip()

            if ticker in existing:
                symbol = existing[ticker]

                symbol.name = row.name
                symbol.primary_exchange = "NASDAQ"
                symbol.active = True

            else:
                db.add(
                    Symbol(
                        ticker=ticker,
                        name=row.name,
                        primary_exchange="NASDAQ",
                        active=True,
                    )
                )

                added += 1

        db.commit()

        print(f"Active symbols: {len(stocks)}")
        print(f"New symbols: {added}")

    finally:
        db.close()


if __name__ == "__main__":
    sync_symbols()
