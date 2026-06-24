from fastapi import FastAPI

app = FastAPI(
    title="MarketCopilotex API",
    description="AI-assisted market briefing and investment decision journal.",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "MarketCopilotex API"
    }
