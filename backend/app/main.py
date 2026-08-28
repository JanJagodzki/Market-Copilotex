import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from backend.app.api.symbols import router as symbols_router
from backend.app.api.watchlist import router as watchlist_router
from backend.app.db.database import Base, engine
from backend.app.services.watchlist_sync import (
    watchlist_sync_loop,
)


@asynccontextmanager
async def lifespan(app):
    Base.metadata.create_all(bind=engine)

    sync_task = asyncio.create_task(
        watchlist_sync_loop()
    )

    yield

    sync_task.cancel()

    try:
        await sync_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="MarketCopilotex API",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(symbols_router)
app.include_router(watchlist_router)

PROJECT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_DIR / "frontend"

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static",
)


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}
