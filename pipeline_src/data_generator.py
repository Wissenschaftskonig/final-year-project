"""
Synthetic SME transaction-history generator.

Produces raw transaction logs for N simulated SMEs, each covering a
12-month observation window, with a latent risk profile (low / moderate /
high) drawn in proportions 60 / 30 / 10, matching Chapter 3, Section 3.3
of the thesis.

The latent risk profile drives the *parameters* used to simulate monthly
transaction behaviour (count, amount, volatility, cash-flow split, growth
trend). It is not copied directly into the engineered feature vector, and
substantial per-SME and per-month noise is injected so that the
classification problem is not trivially separable. This mirrors Section
3.6 (Construction of Risk Labels) which requires separation between the
latent generation parameters and the observed feature vector, to reduce
direct target leakage.
"""

import numpy as np
import pandas as pd

N_SMES = 5000
N_MONTHS = 12
RISK_PROPORTIONS = {"low": 0.60, "moderate": 0.30, "high": 0.10}
RANDOM_SEED = 42


def _monthly_lambda(risk, base_noise):
    """Base expected transaction count per month, before seasonality/growth."""
    base = {"low": 36.0, "moderate": 30.0, "high": 23.0}[risk]
    return max(3.0, base * base_noise)


def _amount_params(risk, amount_noise):
    """(mu, sigma) of the lognormal distribution used for transaction sizes."""
    mu = {"low": 9.35, "moderate": 9.20, "high": 9.00}[risk] + np.log(amount_noise)
    sigma = {"low": 0.65, "moderate": 0.75, "high": 0.90}[risk]
    return mu, sigma


def _growth_rate(risk, rng):
    """Per-month multiplicative drift applied to the base transaction level."""
    center = {"low": 0.007, "moderate": 0.001, "high": -0.009}[risk]
    return rng.normal(center, 0.020)


def _outflow_ratio(risk, rng):
    """Fraction of gross transaction value that represents outflow (expenses,
    withdrawals, chargebacks) rather than inflow (sales receipts)."""
    center = {"low": 0.62, "moderate": 0.68, "high": 0.76}[risk]
    ratio = rng.normal(center, 0.12)
    return float(np.clip(ratio, 0.30, 0.98))


def generate_transaction_log(n_smes=N_SMES, n_months=N_MONTHS, seed=RANDOM_SEED):
    """Return (sme_profile_df, transactions_df).

    sme_profile_df: one row per SME with the latent risk group (generation
    control only â€” not a model input).

    transactions_df: one row per individual simulated transaction, with
    columns [sme_id, month, day, amount, direction], where direction is
    'inflow' or 'outflow'. This is the raw log that the feature-engineering
    module consumes.
    """
    rng = np.random.default_rng(seed)

    risk_labels = rng.choice(
        list(RISK_PROPORTIONS.keys()),
        size=n_smes,
        p=list(RISK_PROPORTIONS.values()),
    )

    sme_rows = []
    tx_rows = []

    for sme_id in range(n_smes):
        risk = risk_labels[sme_id]

        # Per-SME heterogeneity: two similar businesses with the same latent
        # risk group still behave differently.
        base_noise = rng.lognormal(mean=0.0, sigma=0.40)
        amount_noise = rng.lognormal(mean=0.0, sigma=0.38)
        lam0 = _monthly_lambda(risk, base_noise)
        mu0, sigma_amt = _amount_params(risk, amount_noise)
        growth = _growth_rate(risk, rng)
        outflow_ratio = _outflow_ratio(risk, rng)

        # Seasonal pattern: random phase and amplitude, deliberately only
        # weakly related to risk group, so seasonality behaves as a
        # genuine nuisance / signal component rather than a proxy label.
        seasonal_amp = rng.uniform(0.05, 0.35)
        seasonal_phase = rng.uniform(0, 2 * np.pi)

        for month in range(n_months):
            growth_factor = (1 + growth) ** month
            seasonal_factor = 1 + seasonal_amp * np.sin(
                2 * np.pi * month / 12 + seasonal_phase
            )
            volatility_shock = rng.lognormal(mean=0.0, sigma=0.30)

            lam = max(2.0, lam0 * growth_factor * seasonal_factor * volatility_shock)
            n_tx = rng.poisson(lam)
            if n_tx == 0:
                continue

            days = rng.integers(1, 29, size=n_tx)
            amounts = rng.lognormal(mean=mu0, sigma=sigma_amt, size=n_tx)
            amounts *= growth_factor

            # Assign each transaction a direction; outflow-heavy months are
            # noisier for higher-risk profiles (irregular expense timing).
            outflow_draw = rng.random(n_tx)
            directions = np.where(outflow_draw < outflow_ratio, "outflow", "inflow")

            for d, amt, direction in zip(days, amounts, directions):
                tx_rows.append((sme_id, month, int(d), float(amt), direction))

        sme_rows.append((sme_id, risk))

    sme_profile_df = pd.DataFrame(sme_rows, columns=["sme_id", "latent_risk"])
    transactions_df = pd.DataFrame(
        tx_rows, columns=["sme_id", "month", "day", "amount", "direction"]
    )
    return sme_profile_df, transactions_df


if __name__ == "__main__":
    profiles, tx = generate_transaction_log()
    profiles.to_csv("/home/claude/credit-scoring/data/sme_profiles.csv", index=False)
    tx.to_csv("/home/claude/credit-scoring/data/transactions_raw.csv", index=False)
    print("SMEs:", len(profiles))
    print(profiles["latent_risk"].value_counts(normalize=True))
    print("Total transactions:", len(tx))
    print(tx.groupby("sme_id").size().describe())
