from fastapi import FastAPI

from src.api.routes import router


app = FastAPI(
    title="INSIGHT",
    description="AI-Powered Decision Intelligence Platform",
    version="0.1.0",
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "application": "INSIGHT",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }