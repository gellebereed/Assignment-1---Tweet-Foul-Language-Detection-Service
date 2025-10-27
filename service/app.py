from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel, Field
from joblib import load
import os

APP_VERSION = "1.1.0"
ARTIFACT_PATH = os.environ.get("MODEL_ARTIFACT", "artifacts/model.joblib")

app = FastAPI(title="Tweet Foul-Language Detection Service", version=APP_VERSION)

# Load the trained pipeline + threshold once when the server starts
artifact = None

@app.on_event("startup")
def load_artifact():
    global artifact
    if not os.path.exists(ARTIFACT_PATH):
        raise RuntimeError(f"Model artifact not found at {ARTIFACT_PATH}. Run the training notebook first.")
    artifact = load(ARTIFACT_PATH)

class PredictIn(BaseModel):
    text: str = Field(..., description="Tweet text to classify")

class PredictOut(BaseModel):
    label: int
    label_name: str
    score: float
    threshold: float
    version: str = APP_VERSION

@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}

@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn = Body(...)):
    if artifact is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    text = payload.text
    if not isinstance(text, str) or len(text.strip()) == 0:
        raise HTTPException(status_code=422, detail="`text` must be a non-empty string")

    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]
    label_map = artifact.get("label_map", {0: "proper", 1: "foul"})

    # Probability for class 1 (foul). If model lacks predict_proba, fall back to a sigmoid on decision_function.
    try:
        p1 = float(pipeline.predict_proba([text])[0, 1])
    except Exception:
        import numpy as np
        z = float(pipeline.decision_function([text])[0])
        p1 = 1.0 / (1.0 + np.exp(-z))

    label = int(p1 >= threshold)
    return PredictOut(label=label, label_name=label_map[label], score=p1, threshold=threshold)
