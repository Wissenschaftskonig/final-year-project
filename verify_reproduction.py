#!/usr/bin/env python3
"""
Verification script for reproducing the thesis results.

Checks all expected values from Section 3 of the agent instructions.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

TOL = 0.001


def check_equal(name, actual, expected, tol=TOL):
    if isinstance(expected, (list, tuple)):
        actual = np.array(actual)
        expected = np.array(expected)
        if actual.shape != expected.shape:
            print(f"FAIL {name}: shape mismatch {actual.shape} vs {expected.shape}")
            return False
        diff = np.abs(actual - expected).max()
        if diff <= tol:
            print(f"PASS {name}")
            return True
        else:
            print(f"FAIL {name}: max diff {diff:.6f} > {tol}")
            print(f"  actual: {actual}")
            print(f"  expected: {expected}")
            return False
    else:
        diff = abs(actual - expected)
        if diff <= tol:
            print(f"PASS {name}")
            return True
        else:
            print(f"FAIL {name}: diff {diff:.6f} > {tol}")
            print(f"  actual: {actual}")
            print(f"  expected: {expected}")
            return False


def check_exact(name, actual, expected):
    if actual == expected:
        print(f"PASS {name}")
        return True
    else:
        print(f"FAIL {name}: {actual} != {expected}")
        return False


def main():
    all_pass = True

    # 3.1 Data generation and features
    print("\n=== 3.1 Data generation ===")
    profiles = pd.read_csv(OUTPUT_DIR.parent / "data" / "sme_profiles.csv")
    tx = pd.read_csv(OUTPUT_DIR.parent / "data" / "transactions_raw.csv")
    all_pass &= check_exact("SME count", len(profiles), 5000)
    all_pass &= check_exact("Total transactions", len(tx), 2320143)

    risk_counts = profiles["latent_risk"].value_counts()
    all_pass &= check_exact("Low risk count", risk_counts.get("low", 0), 3049)
    all_pass &= check_exact("Moderate risk count", risk_counts.get("moderate", 0), 1458)
    all_pass &= check_exact("High risk count", risk_counts.get("high", 0), 493)

    # 3.2 Thresholds
    print("\n=== 3.2 Thresholds ===")
    with open(OUTPUT_DIR / "selected_thresholds.json") as f:
        th = json.load(f)
    all_pass &= check_equal("t_rule", th["t_rule"], 0.615)
    all_pass &= check_equal("t_xgb", th["t_xgb"], 0.613)
    all_pass &= check_equal("best_alpha", th["best_alpha"], 0.5)
    all_pass &= check_equal("t_hybrid", th["t_hybrid"], 0.579)
    all_pass &= check_equal("alpha 0.25 F1", th["alpha_sweep_train_f1"]["0.25"], 0.4557)
    all_pass &= check_equal("alpha 0.50 F1", th["alpha_sweep_train_f1"]["0.5"], 0.4641)
    all_pass &= check_equal("alpha 0.75 F1", th["alpha_sweep_train_f1"]["0.75"], 0.4564)

    # 3.3 Cross-validation
    print("\n=== 3.3 Cross-validation ===")
    cv = pd.read_csv(OUTPUT_DIR / "cv_summary.csv", index_col=0)
    all_pass &= check_equal("CV accuracy mean", cv.loc["accuracy", "mean"], 0.846)
    all_pass &= check_equal("CV accuracy std", cv.loc["accuracy", "std"], 0.009)
    all_pass &= check_equal("CV precision mean", cv.loc["precision", "mean"], 0.343)
    all_pass &= check_equal("CV precision std", cv.loc["precision", "std"], 0.022)
    all_pass &= check_equal("CV recall mean", cv.loc["recall", "mean"], 0.608)
    all_pass &= check_equal("CV recall std", cv.loc["recall", "std"], 0.050)
    all_pass &= check_equal("CV F1 mean", cv.loc["f1", "mean"], 0.438)
    all_pass &= check_equal("CV F1 std", cv.loc["f1", "std"], 0.027)
    all_pass &= check_equal("CV ROC-AUC mean", cv.loc["roc_auc", "mean"], 0.835)
    all_pass &= check_equal("CV ROC-AUC std", cv.loc["roc_auc", "std"], 0.023)

    # 3.4 Test set
    print("\n=== 3.4 Test set ===")
    test_comp = pd.read_csv(OUTPUT_DIR / "test_set_comparison.csv", index_col=0)

    # rule_based
    all_pass &= check_equal("rule accuracy", test_comp.loc["rule_based", "accuracy"], 0.879)
    all_pass &= check_equal("rule precision", test_comp.loc["rule_based", "precision"], 0.408)
    all_pass &= check_equal("rule recall", test_comp.loc["rule_based", "recall"], 0.495)
    all_pass &= check_equal("rule F1", test_comp.loc["rule_based", "f1"], 0.447)
    all_pass &= check_equal("rule ROC-AUC", test_comp.loc["rule_based", "roc_auc"], 0.783)

    # xgboost
    all_pass &= check_equal("xgb accuracy", test_comp.loc["xgboost", "accuracy"], 0.884)
    all_pass &= check_equal("xgb precision", test_comp.loc["xgboost", "precision"], 0.432)
    all_pass &= check_equal("xgb recall", test_comp.loc["xgboost", "recall"], 0.545)
    all_pass &= check_equal("xgb F1", test_comp.loc["xgboost", "f1"], 0.482)
    all_pass &= check_equal("xgb ROC-AUC", test_comp.loc["xgboost", "roc_auc"], 0.858)

    # hybrid alpha 0.25
    all_pass &= check_equal("hyb0.25 accuracy", test_comp.loc["hybrid_alpha_0.25", "accuracy"], 0.884)
    all_pass &= check_equal("hyb0.25 precision", test_comp.loc["hybrid_alpha_0.25", "precision"], 0.432)
    all_pass &= check_equal("hyb0.25 recall", test_comp.loc["hybrid_alpha_0.25", "recall"], 0.545)
    all_pass &= check_equal("hyb0.25 F1", test_comp.loc["hybrid_alpha_0.25", "f1"], 0.482)
    all_pass &= check_equal("hyb0.25 ROC-AUC", test_comp.loc["hybrid_alpha_0.25", "roc_auc"], 0.853)

    # hybrid alpha 0.5
    all_pass &= check_equal("hyb0.5 accuracy", test_comp.loc["hybrid_alpha_0.5", "accuracy"], 0.886)
    all_pass &= check_equal("hyb0.5 precision", test_comp.loc["hybrid_alpha_0.5", "precision"], 0.440)
    all_pass &= check_equal("hyb0.5 recall", test_comp.loc["hybrid_alpha_0.5", "recall"], 0.556)
    all_pass &= check_equal("hyb0.5 F1", test_comp.loc["hybrid_alpha_0.5", "f1"], 0.491)
    all_pass &= check_equal("hyb0.5 ROC-AUC", test_comp.loc["hybrid_alpha_0.5", "roc_auc"], 0.841)

    # hybrid alpha 0.75
    all_pass &= check_equal("hyb0.75 accuracy", test_comp.loc["hybrid_alpha_0.75", "accuracy"], 0.882)
    all_pass &= check_equal("hyb0.75 precision", test_comp.loc["hybrid_alpha_0.75", "precision"], 0.423)
    all_pass &= check_equal("hyb0.75 recall", test_comp.loc["hybrid_alpha_0.75", "recall"], 0.525)
    all_pass &= check_equal("hyb0.75 F1", test_comp.loc["hybrid_alpha_0.75", "f1"], 0.468)
    all_pass &= check_equal("hyb0.75 ROC-AUC", test_comp.loc["hybrid_alpha_0.75", "roc_auc"], 0.818)

    # Confusion matrices
    print("\n=== Confusion matrices ===")
    with open(OUTPUT_DIR / "confusion_matrices.json") as f:
        cms = json.load(f)
    all_pass &= check_equal("rule_based CM", cms["rule_based"], [[830, 71], [50, 49]])
    all_pass &= check_equal("xgboost CM", cms["xgboost"], [[830, 71], [45, 54]])
    all_pass &= check_equal("hybrid CM", cms["hybrid_alpha_0.5"], [[831, 70], [44, 55]])

    # SHAP
    print("\n=== SHAP mean abs ===")
    shap_ma = pd.read_csv(OUTPUT_DIR / "shap_mean_abs.csv", index_col=0, header=None, names=["val"])
    expected_shap = {
        "cashflow_stability": 0.961,
        "cashflow_net": 0.693,
        "total_volume": 0.638,
        "transaction_frequency": 0.492,
        "volatility": 0.281,
        "avg_transaction_value": 0.275,
        "growth_trend": 0.262,
        "seasonality": 0.171,
    }
    for feat, val in expected_shap.items():
        all_pass &= check_equal(f"SHAP {feat}", shap_ma.loc[feat, "val"], val)

    # XGBoost feature importance
    print("\n=== XGBoost feature importance ===")
    xgb_fi = pd.read_csv(OUTPUT_DIR / "xgb_feature_importance.csv", index_col=0, header=None, names=["val"])
    expected_fi = {
        "cashflow_stability": 0.280,
        "total_volume": 0.192,
        "transaction_frequency": 0.112,
        "cashflow_net": 0.109,
        "volatility": 0.085,
        "growth_trend": 0.077,
        "avg_transaction_value": 0.077,
        "seasonality": 0.069,
    }
    for feat, val in expected_fi.items():
        all_pass &= check_equal(f"FI {feat}", xgb_fi.loc[feat, "val"], val)

    print("\n=== Summary ===")
    if all_pass:
        print("ALL CHECKS PASSED")
        return 0
    else:
        print("SOME CHECKS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())