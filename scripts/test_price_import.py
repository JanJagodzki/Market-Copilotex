from backend.app.data.yahoo_prices import sync_selected_symbols


def main():
    sync_selected_symbols(
        tickers=[
            "AAPL",
            "MSFT",
            "NVDA",
        ],
        period="max",
    )


if __name__ == "__main__":
    main()
