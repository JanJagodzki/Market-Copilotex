import argparse

from backend.app.data.yahoo_intraday import sync_intraday_prices
from backend.app.db.database import Base, engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="+")
    parser.add_argument("--days", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)

    print("Starting 15-minute price update")
    result = sync_intraday_prices(
        tickers=args.tickers,
        days=args.days,
        batch_size=args.batch_size,
    )

    print()
    print("Intraday update finished")
    print(f"Requested symbols: {result['requested_symbols']}")
    print(f"Updated symbols: {result['updated_symbols']}")
    print(f"Saved rows: {result['rows']}")
    print(f"Missing symbols: {result['missing_symbols']}")
    print(f"Failed symbols: {result['failed_symbols']}")


if __name__ == "__main__":
    main()
