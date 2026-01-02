import joblib
from app.ml.preprocess import clean_text

model = joblib.load("models/resume_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_resume(text: str) -> float:
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    score = model.predict_proba(vector)[0][1]
    return round(score, 2)
