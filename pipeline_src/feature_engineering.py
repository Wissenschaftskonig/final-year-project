"""
Feature-engineering module.

Converts the raw transaction log (one row per transaction) into the six
study features defined in Chapter 3, Table 3.1:
  1. transaction_frequency
  2. transaction_volume        (total_volume, avg_transaction_value)
  3. volatility
  4. seasonality
  5. cashflow_net, cashflow_stability
  6. growth_trend

One row is produced per SME. Operational definitions match Table 3.1 and
are fixed here before any model is trained (Section 3.5 / 3.7).
"""

import numpy as np
import pandas as pd

from paths import DATA_DIR

N_MONTHS = 12


def _seasonality_score(monthly_totals):
    """Amplitude of the dominant periodic component relative to the mean
    level, computed from the FFT of the de-trended monthly series."""
    values = monthly_totals.values.astype(float)
    if len(values) < 2 or values.mean() == 0:
        return 0.0
    # Remove linear trend before measuring periodic variation.
    x = np.arange(len(values))
    slope, intercept = np.polyfit(x, values, 1)
    detrended = values - (slope * x + intercept)
    spectrum = np.abs(np.fft.rfft(detrended))
    if len(spectrum) <= 1:
        return 0.0
    dominant = spectrum[1:].max()  # exclude the DC component
    return float(dominant / (values.mean() + 1e-9))


def engineer_features(transactions_df, n_smes, n_months=N_MONTHS):
    records = []

    # Pre-aggregate to monthly totals per SME per direction for speed.
    grp = transactions_df.groupby(["sme_id", "month", "direction"])["amount"].sum()
    counts = transactions_df.groupby("sme_id").size()

    for sme_id in range(n_smes):
        # Build a full 12-month series (missing months = 0 activity).
        monthly_gross = pd.Series(0.0, index=range(n_months))
        monthly_inflow = pd.Series(0.0, index=range(n_months))
        monthly_outflow = pd.Series(0.0, index=range(n_months))

        if sme_id in grp.index.get_level_values(0):
            sub = grp.loc[sme_id]
            for (month, direction), val in sub.items():
                if direction == "inflow":
                    monthly_inflow[month] += val
                else:
                    monthly_outflow[month] += val
        monthly_gross = monthly_inflow + monthly_outflow

        n_tx_total = int(counts.get(sme_id, 0))
        total_volume = float(monthly_gross.sum())
        avg_tx_value = float(total_volume / n_tx_total) if n_tx_total > 0 else 0.0

        # 1. Frequency
        transaction_frequency = n_tx_total / n_months

        # 3. Volatility: coefficient of variation of monthly gross totals.
        mean_m = monthly_gross.mean()
        std_m = monthly_gross.std(ddof=0)
        volatility = float(std_m / mean_m) if mean_m > 0 else 0.0

        # 4. Seasonality
        seasonality = _seasonality_score(monthly_gross)

        # 5. Cash-flow pattern
        net_flow = monthly_inflow - monthly_outflow
        cashflow_net = float(net_flow.sum())
        cashflow_stability = (
            float(net_flow.std(ddof=0) / (monthly_gross.mean() + 1e-9))
            if mean_m > 0
            else 0.0
        )

        # 6. Growth trend: slope of monthly gross totals, normalised by mean.
        x = np.arange(n_months)
        if mean_m > 0:
            slope, _ = np.polyfit(x, monthly_gross.values, 1)
            growth_trend = float(slope / mean_m)
        else:
            growth_trend = 0.0

        records.append(
            {
                "sme_id": sme_id,
                "transaction_frequency": transaction_frequency,
                "total_volume": total_volume,
                "avg_transaction_value": avg_tx_value,
                "volatility": volatility,
                "seasonality": seasonality,
                "cashflow_net": cashflow_net,
                "cashflow_stability": cashflow_stability,
                "growth_trend": growth_trend,
            }
        )

    return pd.DataFrame(records)


if __name__ == "__main__":
    profiles = pd.read_csv(DATA_DIR / "sme_profiles.csv")
    tx = pd.read_csv(DATA_DIR / "transactions_raw.csv")

    features = engineer_features(tx, n_smes=len(profiles))
    dataset = features.merge(profiles, on="sme_id")
    dataset["target"] = (dataset["latent_risk"] == "high").astype(int)

    dataset.to_csv(DATA_DIR / "feature_dataset.csv", index=False)
    print(dataset.shape)
    print(dataset["target"].value_counts(normalize=True))
    print(dataset.describe().T)
