
from io import StringIO

import pandas as pd
import requests


NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"


def get_nasdaq_symbols():
    response = requests.get(
        NASDAQ_URL,
        timeout=30,
    )
    response.raise_for_status()

    data = pd.read_csv(
        StringIO(response.text),
        sep="|",
    )

    data = data.dropna(
        subset=["Symbol", "Security Name"]
    )

    data["Symbol"] = (
        data["Symbol"]
        .astype(str)
        .str.strip()
    )

    data = data[data["Symbol"] != ""]

    data = data[data["Test Issue"] == "N"]
    data = data[data["ETF"] == "N"]

    excluded = (
        "Warrant|Warrants|Unit|Units|Right|Rights|"
        "Preferred|Notes"
    )

    data = data[
        ~data["Security Name"].str.contains(
            excluded,
            case=False,
            na=False,
        )
    ]

    data = data[
        ["Symbol", "Security Name"]
    ].copy()

    data.columns = ["ticker", "name"]

    return data.reset_index(drop=True)
