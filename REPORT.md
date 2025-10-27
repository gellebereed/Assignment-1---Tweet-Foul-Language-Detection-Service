# Assignment 1 — Tweet Foul-Language Detection
**Team:** Solo • **Date:** 2025-10-27

## Problem
Build a small service that predicts whether a tweet is **foul (1)** or **proper (0)**.

## Data
Primary dataset: **Kaggle “Hate Speech & Offensive Language” (Davidson et al.)** at `data/labeled_data.csv`.  
Label mapping to binary:
- **foul (1):** `class ∈ {0, 1}` (hate, offensive)
- **proper (0):** `class = 2` (neither)

Light hygiene: remove “RT ” marker, normalize URLs to `<URL>` and mentions to `<USER>`, and drop duplicates.  
Splits: **train/validation/test** with stratification (train→val = 80/20; test = 20% of full data).

*(Optional fallback: a local CSV at `data/offensive.csv` with columns `text,label` where label ∈ {0,1}.)*

## Method
**Features:** TF-IDF with unigrams + bigrams (`ngram_range=(1,2)`), `strip_accents='unicode'`, `min_df=2`, `max_df=0.95`.  
**Models:** three classic linear baselines — Logistic Regression, Linear SVM (probability-calibrated), Complement Naive Bayes.  
**Thresholding:** tune a decision threshold on the **validation** set to enforce **Recall ≥ 0.80**; among thresholds that satisfy this, pick the one with **highest Precision** (tie-break by F1).  
**Metrics:** Accuracy; Precision/Recall/F1 (binary, macro, weighted); ROC-AUC; PR-AUC; Confusion Matrix.

## Results

**Validation model comparison (TF-IDF 1–2g, shared):**
- **Winner:** `linear_svm` (with probability calibration)
- **Chosen threshold (from validation):** **0.9478**
- Validation summary (top 3):
  - linear_svm — Acc **0.8374**, Prec (bin) **0.9970**, Rec (bin) **0.8071**, F1 (bin) **0.8921**, ROC-AUC **0.9798**, PR-AUC **0.9952**
  - logreg — Acc 0.8328, Prec (bin) 0.9970, Rec (bin) 0.8016, F1 (bin) 0.8887, ROC-AUC 0.9769, PR-AUC 0.9948
  - comp_nb — Acc 0.8275, Prec (bin) 0.9887, Rec (bin) 0.8020, F1 (bin) 0.8856, ROC-AUC 0.9592, PR-AUC 0.9909

**Test set (using the validation-tuned threshold above):**
- **Accuracy:** **0.8358**
- **Precision / Recall / F1 (binary, positive=foul):** **0.9970 / 0.8051 / 0.8908**
- **Macro:** Precision **0.7516**, Recall **0.8965**, F1 **0.7801**
- **Weighted:** Precision **0.9144**, Recall **0.8358**, F1 **0.8536**
- **ROC-AUC:** **0.9812**
- **PR-AUC:** **0.9960**
- **Artifacts:** `fig_cm.png` (Confusion Matrix), `fig_roc.png`, `fig_pr.png`, `artifacts/validation_model_compare.csv`, `artifacts/test_metrics_winner.csv`

**Interpretation (brief):**
- The **Recall ≥ 0.80** target is satisfied while keeping **very high Precision (~0.997)**; this is ideal when we prefer to rarely accuse a proper tweet of being foul.
- PR-AUC ~**0.996** and ROC-AUC ~**0.981** show a strong separability under TF-IDF+linear assumptions.
- The confusion matrix shows very few false positives (proper → foul), consistent with the high precision.


## Observations
- Thresholding met the **Recall ≥ 0.80 target** while keeping **very high Precision (~0.997)**; this matches the PR curve ~1.0 and few false positives in the confusion matrix.  
- **Failure cases:** sarcasm; obfuscated terms (e.g., leetspeak); quoted/reported speech.  
- **Fairness:** identity terms can correlate with offensive labels; audit subsets and consider human review for edge cases.

## Engineering
- **Service:** FastAPI loads `artifacts/model.joblib` at startup and exposes `/predict`.  
- **Tests:** happy path, foul example, invalid input.  
- **Docker:** reproducible image; `uvicorn` entrypoint.

## Diagram
TF-IDF → (LogReg | Linear SVM + calibration | Complement NB) → probability p(foul) → threshold → label {0,1}
