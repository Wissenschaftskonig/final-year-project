"""
Rule-based scoring module.

Applies fixed, pre-defined thresholds to the engineered features and
produces an interpretable risk-point score, normalised to [0, 1]. The
thresholds are set once, from domain reasoning about what "risky"
transaction behaviour looks like, and are not re-tuned after inspecting
test-set performance (Section 3.7, content validity).

Each triggered condition adds a fixed number of risk points. The maximum
attainable score is used to normalise the output so it can be combined
with the XGBoost probability in Section 3.10.
"""

import pandas as pd

from paths import DATA_DIR

RULES = [
    ("low_frequency", lambda r: r["transaction_frequency"] < 22, 2),
    ("low_volume", lambda r: r["total_volume"] < 3_000_000, 2),
    ("high_volatility", lambda r: r["volatility"] > 0.42, 2),
    ("unstable_cashflow", lambda r: r["cashflow_stability"] > 0.30, 3),
    ("declining_trend", lambda r: r["growth_trend"] < -0.01, 2),
    ("negative_net_flow", lambda r: r["cashflow_net"] < -2_000_000, 1),
    ("high_seasonality", lambda r: r["seasonality"] > 2.2, 1),
]

MAX_POINTS = sum(points for _, _, points in RULES)


def score_row(row):
    points = 0
    triggered = []
    for name, condition, weight in RULES:
        if condition(row):
            points += weight
            triggered.append(name)
    return points, triggered


def apply_rule_scoring(df):
    df = df.copy()
    scores, triggers = [], []
    for _, row in df.iterrows():
        pts, trig = score_row(row)
        scores.append(pts)
        triggers.append(";".join(trig))
    df["rule_points"] = scores
    df["rule_score"] = df["rule_points"] / MAX_POINTS
    df["rule_triggers"] = triggers
    return df


if __name__ == "__main__":
    dataset = pd.read_csv(DATA_DIR / "feature_dataset.csv")
    scored = apply_rule_scoring(dataset)
    scored.to_csv(DATA_DIR / "dataset_with_rule_score.csv", index=False)

    print("Max attainable points:", MAX_POINTS)
    print(scored.groupby("target")["rule_score"].describe())

    # Quick standalone rule-based classification check at rule_score >= 0.5
    pred = (scored["rule_score"] >= 0.5).astype(int)
    tp = ((pred == 1) & (scored["target"] == 1)).sum()
    fp = ((pred == 1) & (scored["target"] == 0)).sum()
    fn = ((pred == 0) & (scored["target"] == 1)).sum()
    tn = ((pred == 0) & (scored["target"] == 0)).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    print(f"Rule-only @0.5 -> TP{tp} FP{fp} FN{fn} TN{tn} P{precision:.3f} R{recall:.3f}")
