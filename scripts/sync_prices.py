import argparse

from backend.app.data.yahoo_prices import sync_all_symbols


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    sync_all_symbols(
        period="max",
        only_without_prices=True,
        batch_size=10,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
