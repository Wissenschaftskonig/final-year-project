"""
FastAPI scoring service.

Exposes the trained hybrid credit-scoring pipeline (rule-based score +
XGBoost classifier + hybrid combination) over a REST endpoint, per
Chapter 3, Section 3.4 (Instrument for Data Collection and Development
Tools).

POST /score/features accepts an SME's eight engineered features
and returns:
  - the rule-based score and which rules fired
  - the XGBoost probability of the elevated-risk class
  - the hybrid score (using the alpha selected during evaluation)
  - the resulting risk category (acceptable-risk / elevated-risk)
  - the top SHAP feature contributions for that specific prediction

This lets the dashboard (or any client) request a full credit
profile for a single SME without re-running the training pipeline.
"""

import json
import os
from pathlib import Path

import numpy as np
import shap
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "xgboost_model.json"
THRESHOLDS_PATH = BASE_DIR / "outputs" / "selected_thresholds.json"

FEATURE_COLS = [
    "transaction_frequency",
    "total_volume",
    "avg_transaction_value",
    "volatility",
    "seasonality",
    "cashflow_net",
    "cashflow_stability",
    "growth_trend",
]

RULES = [
    ("low_frequency", lambda f: f["transaction_frequency"] < 22, 2,
     "Fewer than 22 transactions per month on average"),
    ("low_volume", lambda f: f["total_volume"] < 3_000_000, 2,
     "Total annual transaction volume below NGN 3,000,000"),
    ("high_volatility", lambda f: f["volatility"] > 0.42, 2,
     "Coefficient of variation of monthly totals above 0.42"),
    ("unstable_cashflow", lambda f: f["cashflow_stability"] > 0.30, 3,
     "Net cash-flow instability above 0.30"),
    ("declining_trend", lambda f: f["growth_trend"] < -0.01, 2,
     "Negative growth trend across the observation window"),
    ("negative_net_flow", lambda f: f["cashflow_net"] < -2_000_000, 1,
     "Cumulative net cash flow below NGN -2,000,000"),
    ("high_seasonality", lambda f: f["seasonality"] > 2.2, 1,
     "High seasonal concentration of transaction activity"),
]
MAX_POINTS = sum(weight for _, _, weight, _ in RULES)

app = FastAPI(title="Alternative Credit Scoring API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = xgb.XGBClassifier()
_model.load_model(MODEL_PATH)
_explainer = shap.TreeExplainer(_model)

with open(THRESHOLDS_PATH) as f:
    _thresholds = json.load(f)

T_RULE = _thresholds["t_rule"]
T_XGB = _thresholds["t_xgb"]
BEST_ALPHA = _thresholds["best_alpha"]
T_HYBRID = _thresholds["t_hybrid"]


class SMEFeatures(BaseModel):
    sme_name: str = Field(default="Unnamed SME")
    transaction_frequency: float
    total_volume: float
    avg_transaction_value: float
    volatility: float
    seasonality: float
    cashflow_net: float
    cashflow_stability: float
    growth_trend: float


def rule_score(features: dict):
    points, triggered = 0, []
    for name, cond, weight, description in RULES:
        if cond(features):
            points += weight
            triggered.append({"rule": name, "description": description, "points": weight})
    return points / MAX_POINTS, triggered


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score/features")
def score_features(payload: SMEFeatures):
    features = payload.model_dump(exclude={"sme_name"})
    x = np.array([[features[c] for c in FEATURE_COLS]])

    r_score, triggered_rules = rule_score(features)
    xgb_prob = float(_model.predict_proba(x)[0, 1])
    hybrid_score = BEST_ALPHA * r_score + (1 - BEST_ALPHA) * xgb_prob
    risk_category = "Elevated-risk" if hybrid_score >= T_HYBRID else "Acceptable-risk"

    shap_values = _explainer.shap_values(x)[0]
    contributions = sorted(
        [
            {"feature": name, "shap_value": float(val), "feature_value": features[name]}
            for name, val in zip(FEATURE_COLS, shap_values)
        ],
        key=lambda d: abs(d["shap_value"]),
        reverse=True,
    )

    return {
        "sme_name": payload.sme_name,
        "rule_score": round(r_score, 4),
        "rule_points": round(r_score * MAX_POINTS),
        "rule_max_points": MAX_POINTS,
        "triggered_rules": triggered_rules,
        "xgboost_probability": round(xgb_prob, 4),
        "hybrid_score": round(hybrid_score, 4),
        "alpha_used": BEST_ALPHA,
        "decision_threshold": T_HYBRID,
        "risk_category": risk_category,
        "shap_contributions": contributions,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
