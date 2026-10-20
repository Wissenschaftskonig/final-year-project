"""
Additional analyses reported in Chapter 4:
- Bootstrap confidence intervals (Table 4.6)
- McNemar and DeLong tests
- Cross-validated AUC for all three configurations
- Feature ablation (Table 4.8)
- Component-disagreement analysis
- Table 4.9 examples
Run after data_generator.py, feature_engineering.py, rule_scoring.py and train_evaluate.py.
"""

import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from paths import DATA_DIR, OUTPUT_DIR
import train_evaluate as te

warnings.filterwarnings("ignore")

df = pd.read_csv(te.DATA_PATH)
F = te.FEATURE_COLS
X = df[F]
y = df["target"]
rs = df["rule_score"]

Xtr, Xte, ytr, yte, rtr, rte = train_test_split(
    X, y, rs, test_size=0.2, stratify=y, random_state=42
)

th = json.load(open(OUTPUT_DIR / "selected_thresholds.json"))
tr_ = th["t_rule"]
tx_ = th["t_xgb"]
th_ = th["t_hybrid"]

m = xgb.XGBClassifier(
    random_state=42, scale_pos_weight=te.POS_WEIGHT, **te.XGB_PARAMS
).fit(Xtr, ytr)
p = m.predict_proba(Xte)[:, 1]
r = rte.values
yt = yte.values

hs = 0.5 * r + 0.5 * p
S = {"rule": r, "xgb": p, "hyb": hs}
T = {"rule": tr_, "xgb": tx_, "hyb": th_}
P = {k: (S[k] >= T[k]).astype(int) for k in S}

print("majority baseline acc", 1 - yt.mean(), "n elevated", yt.sum())

# bootstrap
rng = np.random.default_rng(2026)
n = len(yt)
B = 5000


def mets(idx, k):
    yy = yt[idx]
    pp = P[k][idx]
    return (
        precision_score(yy, pp, zero_division=0),
        recall_score(yy, pp),
        f1_score(yy, pp),
        roc_auc_score(yy, S[k][idx]),
    )


res = {k: [] for k in S}
for b in range(B):
    idx = rng.integers(0, n, n)
    if yt[idx].sum() == 0:
        continue
    for k in S:
        res[k].append(mets(idx, k))

print("--- Table 4.6 (bootstrap 95% CI)")
for k in S:
    a = np.array(res[k])
    lo = np.percentile(a, 2.5, axis=0)
    hi = np.percentile(a, 97.5, axis=0)
    pt = mets(np.arange(n), k)
    print(k, " ".join(f"{pt[i]:.3f}({lo[i]:.3f}-{hi[i]:.3f})" for i in range(4)))

d = np.array(res["hyb"]) - np.array(res["xgb"])
print(
    "hyb-xgb dF1 %.3f (%.3f,%.3f)  dAUC %.3f (%.3f,%.3f)"
    % (
        d[:, 2].mean(),
        *np.percentile(d[:, 2], [2.5, 97.5]),
        d[:, 3].mean(),
        *np.percentile(d[:, 3], [2.5, 97.5]),
    )
)

# McNemar exact
def mcn(a, b):
    ca = P[a] == yt
    cb = P[b] == yt
    x = int((ca & ~cb).sum())
    z = int((~ca & cb).sum())
    return x, z, stats.binomtest(x, x + z, 0.5).pvalue


for a, b in [("hyb", "xgb"), ("xgb", "rule"), ("hyb", "rule")]:
    print("McNemar", a, "vs", b, mcn(a, b))

# DeLong
def delong(y, s1, s2):
    pos = y == 1
    neg = ~pos
    m_ = pos.sum()
    n_ = neg.sum()

    def comps(s):
        sp = s[pos]
        sn = s[neg]
        v10 = np.array(
            [(x > sn).mean() + 0.5 * (x == sn).mean() for x in sp]
        )
        v01 = np.array(
            [(x < sp).mean() + 0.5 * (x == sp).mean() for x in sn]
        )
        return v10, v01

    a10, a01 = comps(s1)
    b10, b01 = comps(s2)
    auc1, auc2 = a10.mean(), b10.mean()
    s10 = np.cov(np.vstack([a10, b10]))
    s01 = np.cov(np.vstack([a01, b01]))
    S_ = s10 / m_ + s01 / n_
    var = S_[0, 0] + S_[1, 1] - 2 * S_[0, 1]
    z = (auc1 - auc2) / np.sqrt(var)
    return auc1 - auc2, 2 * stats.norm.sf(abs(z))


for a, b in [("hyb", "xgb"), ("xgb", "rule"), ("hyb", "rule")]:
    print("DeLong", a, "-", b, delong(yt, S[a], S[b]))

# disagreement
rv = P["rule"]
xv = P["xgb"]
hv = P["hyb"]
dis = rv != xv
bothlow = (rv == 0) & (xv == 0)
print(
    "disagree",
    dis.sum(),
    "elev among",
    yt[dis].sum(),
    "%.3f" % yt[dis].mean(),
    "| bothlow",
    bothlow.sum(),
    "elev rate %.3f" % yt[bothlow].mean(),
    "base %.3f" % yt.mean(),
)
print(
    "hybrid FN",
    ((hv == 0) & (yt == 1)).sum(),
    "of which disagree",
    ((hv == 0) & (yt == 1) & dis).sum(),
)
rev = (hv == 1) | dis
print(
    "review rule flagged",
    rev.sum(),
    "TP",
    (rev & (yt == 1)).sum(),
    "recall %.3f prec %.3f"
    % ((rev & (yt == 1)).sum() / yt.sum(), (rev & (yt == 1)).sum() / rev.sum()),
    "hyb flagged",
    hv.sum(),
)
print(
    "corr freq-volume (full data)",
    np.corrcoef(df.transaction_frequency, df.total_volume)[0, 1],
    " train:",
    np.corrcoef(Xtr.transaction_frequency, Xtr.total_volume)[0, 1],
)

# CV AUC for 3 configs + ablation
def cvrun(cols, collect=False):
    out = []
    rl = []
    hl = []
    bel = []
    for seed in (1, 2, 3):
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        for a, b in skf.split(Xtr, ytr):
            mm = xgb.XGBClassifier(
                random_state=seed, scale_pos_weight=te.POS_WEIGHT, **te.XGB_PARAMS
            ).fit(Xtr.iloc[a][cols], ytr.iloc[a])
            pp = mm.predict_proba(Xtr.iloc[b][cols])[:, 1]
            yy = ytr.iloc[b].values
            out.append(roc_auc_score(yy, pp))
            if collect:
                rr = rtr.iloc[b].values
                rl.append(roc_auc_score(yy, rr))
                hl.append(roc_auc_score(yy, 0.5 * rr + 0.5 * pp))
    return np.array(out), np.array(rl), np.array(hl)


x, rl, hl = cvrun(F, True)
print(
    "--- CV AUC: rule %.3f(%.3f) xgb %.3f(%.3f) hyb %.3f(%.3f); hyb<xgb in %d folds"
    % (rl.mean(), rl.std(ddof=1), x.mean(), x.std(ddof=1), hl.mean(), hl.std(ddof=1), (hl < x).sum())
)

print("--- Table 4.8 ablation")
groups = {
    "Transaction frequency": ["transaction_frequency"],
    "Total transaction volume": ["total_volume"],
    "Average transaction value": ["avg_transaction_value"],
    "Volatility": ["volatility"],
    "Seasonality": ["seasonality"],
    "Net cash flow": ["cashflow_net"],
    "Cash-flow stability": ["cashflow_stability"],
    "Growth trend": ["growth_trend"],
    "Volume pair": ["total_volume", "avg_transaction_value"],
    "Cash-flow pair": ["cashflow_net", "cashflow_stability"],
}
for g, rm in groups.items():
    cols = [c for c in F if c not in rm]
    a, _, _ = cvrun(cols)
    print(f"{g}: {a.mean():.3f} ({a.std(ddof=1):.3f}) change {a.mean() - x.mean():+.3f}")

# Table 4.9 examples
ex = shap.TreeExplainer(m)
for sid in (2877, 4757, 136):
    row = df.loc[[sid]]
    Xr = row[F]
    pr = m.predict_proba(Xr)[0, 1]
    rr = row["rule_score"].values[0]
    h = 0.5 * rr + 0.5 * pr
    sv = pd.Series(ex.shap_values(Xr)[0], index=F)
    top = sv.reindex(sv.abs().sort_values(ascending=False).index)[:5]
    print(
        sid,
        "label",
        row.target.values[0],
        "rule %.3f xgb %.3f hyb %.3f" % (rr, pr, h),
        "pred",
        int(h >= th_),
        "trig",
        row.rule_triggers.values[0],
    )
    print(
        "  freq %.1f vol %.0f net %.0f stab %.3f vola %.3f seas %.3f growth %.3f"
        % tuple(row[c].values[0] for c in [
            "transaction_frequency",
            "total_volume",
            "cashflow_net",
            "cashflow_stability",
            "volatility",
            "seasonality",
            "growth_trend",
        ])
    )
    print("  SHAP", ", ".join(f"{k} {v:+.2f}" for k, v in top.items()))
    print("  in train?", sid in Xtr.index)