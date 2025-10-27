# Tweet Foul-Language Detection Service

Production-style API that classifies tweets as **foul/offensive (1)** or **proper (0)**.

## Tech Stack
- Python 3.11, FastAPI, Uvicorn
- scikit-learn (TF‑IDF + **three models**: Logistic Regression, Linear SVM (calibrated), Complement NB)
- Dataset: Kaggle **labeled_data.csv** at `data/labeled_data.csv` 
- Tests: `pytest`
- Docker for containerized run
- Artifacts saved via `joblib`

## How to train (notebook)
```
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/train_foul_language.ipynb
```
This produces:
```
artifacts/model.joblib
artifacts/metrics.json
artifacts/fig_cm.png
artifacts/fig_roc.png
artifacts/fig_pr.png
artifacts/test_metrics_winner.csv
artifacts/validation_model_compare.csv
```

## How model selection works
- Shared TF‑IDF (unigrams+bigrams) for **all three** models.
- For each model, tune a decision threshold on the **validation** set to ensure **Recall ≥ 0.80**.
- Among thresholds that satisfy this, choose the one with **highest Precision** (tie-break by F1).
- Pick the **validation winner**, evaluate on **test**, and save the artifact.

## Run the API
```
export MODEL_ARTIFACT=artifacts/model.joblib   # Windows: set MODEL_ARTIFACT=artifacts\model.joblib
uvicorn service.app:app --reload --port 8000
```
- `GET /health`
- `POST /predict` with JSON: `{ "text": "your tweet here" }`

## Run tests
```
pytest -q
```

## Docker
```
docker build -t foul-svc .
docker run -p 8000:8000 -e MODEL_ARTIFACT=/app/artifacts/model.joblib foul-svc
```
