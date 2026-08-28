# MarketCopilotex

MarketCopilotex is a machine learning project for stock analysys.

The project collects daily market data, stores it in PostgreSQL and uses
different machine learning models to analyse historical data and create
predictions.

## Active neural models

After a neural benchmark, select the best model for each horizon with:

```bash
python -m scripts.promote_neural_models
```

The script uses validation AUC, saves the selected models in
`backend/app/ml/active_models.json` and checks if their local model files exist.
Model weights stay in the ignored `models/` directory because they can be large.

Start the API and request current predictions for a symbol:

```bash
uvicorn backend.app.main:app --reload
curl http://127.0.0.1:8000/api/symbols/AAPL/ai-predictions
```

The endpoint returns probabilities for 1, 5, 20, 60, 120 and 252 trading days.
These are experimental results and should not be treated as buy or sell advice.

## Watchlist and automatic updates

The watchlist is stored in PostgreSQL. While the FastAPI application is
running, completed 15-minute candles for watched symbols are downloaded every
20 minutes.

```bash
curl http://127.0.0.1:8000/api/watchlist
curl -X POST http://127.0.0.1:8000/api/watchlist/AAPL
curl -X DELETE http://127.0.0.1:8000/api/watchlist/AAPL
```

The web dashboard can add and remove companies, refresh the selected company
manually and display all active neural model predictions.
