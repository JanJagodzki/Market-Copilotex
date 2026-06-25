from fastapi import FastAPI

from app.api.assets import router as assets_router
from app.db.database import check_database_connection

app = FastAPI(
    title="MarketCopilotex API",
    description="AI-assisted market briefing and investment decision journal.",
    version="0.1.0",
)

app.include_router(assets_router)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "MarketCopilotex API",
    }


@app.get("/db-health")
def db_health_check():
    check_database_connection()
    return {
        "status": "ok",
        "database": "connected",
    }