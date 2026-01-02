from fastapi import APIRouter
from pydantic import BaseModel
from app.ml.model import predict_resume

router = APIRouter()

class ResumeRequest(BaseModel):
    resume_text: str

@router.post("/predict")
def predict(request: ResumeRequest):
    score = predict_resume(request.resume_text)
    return {
        "shortlist_score": score,
        "decision": "Shortlisted" if score > 0.6 else "Rejected"
    }
