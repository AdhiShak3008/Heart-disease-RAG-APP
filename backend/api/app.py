from fastapi import FastAPI

from backend.api.predict import router

app = FastAPI(
    title="Heart Disease AI",
    version="1.0.0",
)

app.include_router(router)
