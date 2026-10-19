"""
End-to-end model training and evaluation.

Follows the procedure in Chapter 3, Sections 3.9-3.11:
  1. Split the dataset into training and untouched final-test partitions.
  2. Run 5-fold cross-validation on the training partition to assess the
     XGBoost configuration (repeated across three random seeds).
  3. Fit the rule-based score (already computed) and the XGBoost classifier
     on the training partition only.
  4. Evaluate the rule-based, XGBoost, and hybrid (alpha in
     {0.25, 0.50, 0.75}) configurations on the untouched test partition
     using accuracy, precision, recall, F1, ROC-AUC and a confusion matrix.
  5. Compute XGBoost feature importance and SHAP values on the test set.

All numeric results are written to /outputs as CSV/JSON, and figures are
written as PNG, so they can be dropped directly into Chapter 4 tables and
figures.
"""

import json
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")

DATA_PATH = "/home/claude/credit-scoring/data/dataset_with_rule_score.csv"
OUT_DIR = "/home/claude/credit-scoring/outputs"

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

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.5,
    min_child_weight=3,
    eval_metric="logloss",
)

# The elevated-risk class is roughly 10% of observations (Section 3.3).
# scale_pos_weight rebalances the gradient contribution of the minority
# class during training; the decision threshold below is then selected on
# the training partition, before the test partition is touched.
POS_WEIGHT = 9.14
DECISION_THRESHOLD = None  # resolved by cross-validation in main()


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def best_f1_threshold(y_true, scores):
    """Sweep candidate thresholds and return the one maximising F1.
    Selection happens only on data the caller has designated as training
    data (out-of-fold predictions or the training partition itself)."""
    candidates = np.unique(np.round(scores, 3))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        pred = (scores >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def run_cross_validation(X_train, y_train, seeds=(1, 2, 3)):
    rows = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
            model = xgb.XGBClassifier(
                random_state=seed, scale_pos_weight=POS_WEIGHT, **XGB_PARAMS
            )
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            val_prob = model.predict_proba(X_train.iloc[val_idx])[:, 1]
            val_pred = (val_prob >= 0.5).astype(int)
            m = compute_metrics(y_train.iloc[val_idx], val_pred, val_prob)
            m.update({"seed": seed, "fold": fold})
            rows.append(m)
    return pd.DataFrame(rows)


def get_oof_predictions(X_train, y_train, seed=42):
    """Single 5-fold out-of-fold probability vector on the training
    partition, used only for threshold and alpha selection (never touches
    the test partition)."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y_train))
    for tr_idx, val_idx in skf.split(X_train, y_train):
        model = xgb.XGBClassifier(
            random_state=seed, scale_pos_weight=POS_WEIGHT, **XGB_PARAMS
        )
        model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        oof[val_idx] = model.predict_proba(X_train.iloc[val_idx])[:, 1]
    return oof


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS]
    y = df["target"]
    rule_score_full = df["rule_score"]

    X_train, X_test, y_train, y_test, rule_train, rule_test = train_test_split(
        X, y, rule_score_full, test_size=0.20, stratify=y, random_state=42
    )

    # --- 1. Cross-validation on the training partition -------------------
    cv_results = run_cross_validation(X_train, y_train)
    cv_summary = cv_results.drop(columns=["seed", "fold"]).agg(["mean", "std"]).T
    cv_summary.to_csv(f"{OUT_DIR}/cv_summary.csv")
    cv_results.to_csv(f"{OUT_DIR}/cv_fold_results.csv", index=False)

    # --- 2. Threshold and alpha selection on the TRAINING partition only --
    # Out-of-fold XGBoost probabilities avoid selecting a threshold that
    # merely reflects in-sample fit.
    oof_prob = get_oof_predictions(X_train, y_train)

    t_rule, _ = best_f1_threshold(y_train.values, rule_train.values)
    t_xgb, _ = best_f1_threshold(y_train.values, oof_prob)

    alphas = [0.25, 0.50, 0.75]
    alpha_thresholds = {}
    for alpha in alphas:
        hybrid_train_score = alpha * rule_train.values + (1 - alpha) * oof_prob
        t_h, f1_h = best_f1_threshold(y_train.values, hybrid_train_score)
        alpha_thresholds[alpha] = (t_h, f1_h)

    best_alpha = max(alpha_thresholds, key=lambda a: alpha_thresholds[a][1])
    t_hybrid = alpha_thresholds[best_alpha][0]

    with open(f"{OUT_DIR}/selected_thresholds.json", "w") as f:
        json.dump(
            {
                "t_rule": t_rule,
                "t_xgb": t_xgb,
                "best_alpha": best_alpha,
                "t_hybrid": t_hybrid,
                "alpha_sweep_train_f1": {str(a): v[1] for a, v in alpha_thresholds.items()},
            },
            f,
            indent=2,
        )

    # --- 3. Fit final XGBoost model on the full training partition -------
    final_model = xgb.XGBClassifier(
        random_state=42, scale_pos_weight=POS_WEIGHT, **XGB_PARAMS
    )
    final_model.fit(X_train, y_train)
    final_model.save_model(f"{OUT_DIR}/../models/xgboost_model.json")

    xgb_test_prob = final_model.predict_proba(X_test)[:, 1]
    xgb_test_pred = (xgb_test_prob >= t_xgb).astype(int)

    # --- 4. Rule-based model on the test partition ------------------------
    rule_test_pred = (rule_test >= t_rule).astype(int)
    rule_test_prob = rule_test  # already normalised to [0,1]

    # --- 5. Hybrid model across predefined alpha values --------------------
    results = {}
    results["rule_based"] = compute_metrics(y_test, rule_test_pred, rule_test_prob)
    results["xgboost"] = compute_metrics(y_test, xgb_test_pred, xgb_test_prob)

    hybrid_probs = {}
    for alpha in alphas:
        hybrid_score = alpha * rule_test.values + (1 - alpha) * xgb_test_prob
        t_h = alpha_thresholds[alpha][0]
        hybrid_pred = (hybrid_score >= t_h).astype(int)
        key = f"hybrid_alpha_{alpha}"
        results[key] = compute_metrics(y_test, hybrid_pred, hybrid_score)
        hybrid_probs[alpha] = hybrid_score

    results_df = pd.DataFrame(results).T
    results_df.to_csv(f"{OUT_DIR}/test_set_comparison.csv")

    hybrid_rows = {k: v for k, v in results.items() if k.startswith("hybrid_alpha")}
    best_alpha_key = f"hybrid_alpha_{best_alpha}"

    with open(f"{OUT_DIR}/best_alpha.json", "w") as f:
        json.dump({"best_alpha": best_alpha, "metrics": hybrid_rows[best_alpha_key]}, f, indent=2)

    # --- 5. Confusion matrices --------------------------------------------
    cms = {}
    for name, pred in [
        ("rule_based", rule_test_pred),
        ("xgboost", xgb_test_pred),
        (
            f"hybrid_alpha_{best_alpha}",
            (
                best_alpha * rule_test.values + (1 - best_alpha) * xgb_test_prob
                >= t_hybrid
            ).astype(int),
        ),
    ]:
        cm = confusion_matrix(y_test, pred)
        cms[name] = cm.tolist()

    with open(f"{OUT_DIR}/confusion_matrices.json", "w") as f:
        json.dump(cms, f, indent=2)

    # --- 6. Feature importance and SHAP -----------------------------------
    importance = pd.Series(
        final_model.feature_importances_, index=FEATURE_COLS
    ).sort_values(ascending=False)
    importance.to_csv(f"{OUT_DIR}/xgb_feature_importance.csv")

    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test)
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=FEATURE_COLS
    ).sort_values(ascending=False)
    mean_abs_shap.to_csv(f"{OUT_DIR}/shap_mean_abs.csv")

    # --- Figures ------------------------------------------------------------
    # ROC curves
    plt.figure(figsize=(6, 5))
    for name, prob in [
        ("Rule-based", rule_test_prob),
        ("XGBoost", xgb_test_prob),
        (f"Hybrid (alpha={best_alpha})", best_alpha * rule_test.values + (1 - best_alpha) * xgb_test_prob),
    ]:
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc_val = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves on Untouched Test Partition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/roc_curves.png", dpi=150)
    plt.close()

    # Confusion matrix heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, cm) in zip(axes, cms.items()):
        cm_arr = np.array(cm)
        ax.imshow(cm_arr, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Acceptable", "Elevated"])
        ax.set_yticklabels(["Acceptable", "Elevated"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrices.png", dpi=150)
    plt.close()

    # Feature importance bar chart
    plt.figure(figsize=(6, 4))
    importance.sort_values().plot(kind="barh")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
    plt.close()

    # SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLS, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_summary.png", dpi=150)
    plt.close()

    print("=== CV summary (mean, std across 3 seeds x 5 folds) ===")
    print(cv_summary)
    print("\n=== Test-set comparison ===")
    print(results_df)
    print("\nBest hybrid alpha by F1:", best_alpha)
    print(hybrid_rows[best_alpha_key])
    print("\n=== XGBoost feature importance ===")
    print(importance)
    print("\n=== Mean |SHAP| ===")
    print(mean_abs_shap)


if __name__ == "__main__":
    main()
