from fastapi import FastAPI

from backend.app.api.symbols import router as symbols_router


app = FastAPI(
    title="MarketCopilotex API",
    version="0.1.0",
)

app.include_router(symbols_router)


@app.get("/health")
def health():
    return {"status": "ok"}
