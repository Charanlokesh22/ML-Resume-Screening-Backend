 ML Resume Screening Backend

A machine learning–powered backend service that evaluates resumes and predicts shortlisting probability based on textual features.

 Features
- Resume text preprocessing
- TF-IDF feature extraction
- Supervised ML model for screening
- REST API for predictions
- Dockerized deployment

 Tech Stack
- Python
- FastAPI
- Scikit-learn
- Pandas, NumPy
- PostgreSQL
- Docker

 ML Workflow
1. Resume text cleaning
2. Feature extraction using TF-IDF
3. Model inference using trained classifier
4. Score-based shortlisting decision

 Setup
```bash
git clone https://github.com/yourname/ml-resume-screener
cd ml-resume-screener
docker build -t resume-screener .
docker run -p 8000:8000 resume-screener
