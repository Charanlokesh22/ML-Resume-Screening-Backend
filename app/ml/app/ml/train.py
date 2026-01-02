import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from app.ml.preprocess import clean_text

# Sample dataset
data = pd.DataFrame({
    "resume": [
        "python developer with django experience",
        "machine learning engineer with pytorch",
        "sales executive with marketing background",
        "java backend developer spring boot"
    ],
    "label": [1, 1, 0, 1]
})

data["resume"] = data["resume"].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data["resume"])
y = data["label"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "models/resume_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
