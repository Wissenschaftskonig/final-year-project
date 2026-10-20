# Alternative Credit Scoring for Unbanked SMEs

Code for the final year project "Design and Implementation of an Alternative Credit Scoring System Using Transaction
History Analysis for Unbanked SMEs" (B.Sc. Computer Science, National Open University of Nigeria).

The system turns twelve-month transaction histories into eight features, scores each SME with seven interpretable rules
and an XGBoost classifier, combines the two into a hybrid score, explains each prediction with SHAP, and presents the
result in a lender-facing dashboard.

**All data are synthetic.** No real SME or customer records are used. Results describe how the pipeline behaves on the
generated data. They do not show predictive performance on live Nigerian lending data.

## Repository layout

| Path | Purpose |
|---|---|
| `pipeline_src/data_generator.py` | Generates 5,000 synthetic SME transaction logs (12 months each) with latent low, moderate and high risk groups in 60:30:10 proportions. |
| `pipeline_src/feature_engineering.py` | Converts each log into eight features: transaction frequency, total volume, average transaction value, volatility, seasonality, net cash flow, cash-flow stability, growth trend. Target is 1 for the latent high-risk group (elevated risk), 0 otherwise. |
| `pipeline_src/rule_scoring.py` | Applies seven fixed rules (maximum 13 points) and normalises to a score between 0 and 1. |
| `pipeline_src/train_evaluate.py` | 80:20 stratified split, 5-fold cross-validation (seeds 1, 2, 3), threshold and alpha selection on the training partition, test-set comparison of the rule-based, XGBoost and hybrid models, feature importance and SHAP. Saves the model to `models/`. |
| `pipeline_src/additional_analyses.py` | Bootstrap confidence intervals, McNemar and DeLong tests, cross-validated AUC for all three models, feature ablation, component-disagreement analysis, dashboard examples. |
| `scoring_api.py` | FastAPI service exposing `POST /score/features`. |
| `dashboard.html` | Single-page lender dashboard. It displays three precomputed SME profiles embedded in the page, with the same fields as the API response. It is not connected to the live API. |
| `models/` | Trained XGBoost model (`xgboost_model.json`). |
| `outputs/` | Result tables, thresholds and figures produced by the pipeline. |
| `run_all.sh` | Runs the full pipeline in order. |
| `verify_reproduction.py` | Checks the generated results against the values reported in the thesis. |

## Setup

Tested environment: Python 3.12.15, macOS 15.5 (Darwin 25.5.0), Apple M4, 16 GB RAM.

Exact dependency versions (pinned in `requirements.txt`):
- `numpy==2.4.4`
- `pandas==3.0.2`
- `scikit-learn==1.8.0`
- `xgboost==3.4.1`
- `shap==0.54.1`
- `scipy==1.17.1`
- `matplotlib==3.11.2`
- `fastapi==0.152.0`
- `uvicorn==0.54.0`
- `httpx==0.28.1`
- `pydantic==2.14.1`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce the results

```bash
bash run_all.sh
python verify_reproduction.py
```

`run_all.sh` runs, in order: data generation, feature engineering, rule scoring, training and evaluation, additional
analyses. The `data/` folder is created and filled on first run (about 2.3 million simulated transactions). It is not
committed because the fixed seeds regenerate it exactly.

Seeds: 42 for data generation, the train/test split and the final model; 1, 2 and 3 for cross-validation; 2026 for the
bootstrap. XGBoost settings are fixed in `train_evaluate.py` (`XGB_PARAMS`, `POS_WEIGHT = 9.14`). Small differences in
the last decimal place can occur on other library versions or hardware.

## Main results (untouched test set, 1,000 SMEs, 99 elevated risk)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Rule-based | 0.879 | 0.408 | 0.495 | 0.447 | 0.783 |
| XGBoost | 0.884 | 0.432 | 0.545 | 0.482 | 0.858 |
| Hybrid (alpha 0.50) | 0.886 | 0.440 | 0.556 | 0.491 | 0.841 |

The hybrid model's F1 advantage over XGBoost is not statistically distinguishable from zero on this test set
(95% bootstrap interval for the difference: -0.036 to +0.056).

## Scoring API

```bash
uvicorn scoring_api:app --port 8000
```

`GET /health` returns `{"status": "ok"}`. `POST /score/features` accepts the eight features and returns the rule score
and triggered rules, the XGBoost probability, the hybrid score, the risk category and the SHAP contributions.

```bash
curl -X POST http://localhost:8000/score/features -H "Content-Type: application/json" -d '{
  "sme_name": "Example SME",
  "transaction_frequency": 59.5,
  "total_volume": 8666481.06,
  "avg_transaction_value": 12137.93,
  "volatility": 0.4246,
  "seasonality": 2.1949,
  "cashflow_net": -1923348.95,
  "cashflow_stability": 0.0715,
  "growth_trend": -0.0329
}'
```

Expected: `rule_score` 0.3077 (4 of 13 points), `xgboost_probability` 0.0025, `hybrid_score` 0.1551,
`decision_threshold` 0.579, `risk_category` "Acceptable-risk".

## Dashboard

Open `dashboard.html` in a browser. It needs no server and no build step.

## Limitations

The data are synthetic and the risk labels come from the generator, so the results show technical feasibility under
stated assumptions. The system is a decision-support prototype. It has no production security, no integration with
banking systems, and no regulatory approval for real lending decisions.