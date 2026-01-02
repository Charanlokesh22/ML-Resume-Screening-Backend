from fastapi import FastAPI
from app.api.predict import router as predict_router

app = FastAPI(title="ML Resume Screening Backend")

app.include_router(predict_router)
