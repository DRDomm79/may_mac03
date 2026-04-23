"""
Train frequency-severity models for IPD hospital reimbursement pricing.
Generates synthetic claims data, trains:
  - Poisson regression for claim frequency (how often)
  - Gamma regression for claim severity (how much per claim)
  - Separate models for each optional rider: OPD, Dental, Maternity

Usage: python scripts/train_model.py
Output: models/ipd_freq.pkl, models/ipd_sev.pkl, models/opd_freq.pkl, etc.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor, GammaRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import joblib

SEED = 42
N_SAMPLES = 10000
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ─── Encoding maps (must match backend) ─────────────────────────────────────
GENDER_MAP = {"Male": 0, "Female": 1, "Other": 2}
SMOKING_MAP = {"Never": 0, "Former": 1, "Current": 2}
EXERCISE_MAP = {"Sedentary": 0, "Light": 1, "Moderate": 2, "Active": 3}
OCCUPATION_MAP = {"Office/Desk": 0, "Retail/Service": 1, "Healthcare": 2, "Manual Labor": 3, "Industrial/High-Risk": 4}
REGION_FACTORS = {"Phnom Penh": 1.20, "Siem Reap": 1.05, "Battambang": 0.90,
                  "Sihanoukville": 1.10, "Kampong Cham": 0.85, "Rural Areas": 0.75,
                  "Ho Chi Minh City": 1.25, "Hanoi": 1.20, "Da Nang": 1.05,
                  "Can Tho": 0.90, "Hai Phong": 0.95}


def generate_synthetic_claims(n, seed, coverage_type="ipd"):
    """Generate synthetic insurance claims data with realistic distributions."""
    rng = np.random.RandomState(seed)

    ages = rng.randint(18, 76, size=n)
    genders = rng.choice([0, 1, 2], size=n, p=[0.48, 0.48, 0.04])
    smoking = rng.choice([0, 1, 2], size=n, p=[0.55, 0.25, 0.20])
    exercise = rng.choice([0, 1, 2, 3], size=n, p=[0.30, 0.35, 0.25, 0.10])
    occupation = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.35, 0.25, 0.15, 0.15, 0.10])
    regions = rng.choice(list(range(6)), size=n)  # 0-5 region indices
    preexist = rng.choice([0, 1, 2, 3], size=n, p=[0.50, 0.25, 0.15, 0.10])

    # ─── Frequency model (claims per year) ───────────────────────────────
    # Base claim rate depends on coverage type
    base_rates = {"ipd": 0.12, "opd": 2.5, "dental": 0.8, "maternity": 0.15}
    base = base_rates.get(coverage_type, 0.12)

    # Frequency influenced by risk factors
    freq_lambda = np.zeros(n)
    for i in range(n):
        age_f = 1.0 + max(0, (ages[i] - 35)) * 0.008
        if ages[i] < 5: age_f = 1.3  # children
        smoking_f = [1.0, 1.15, 1.40][smoking[i]]
        exercise_f = [1.20, 1.05, 0.90, 0.80][exercise[i]]
        occupation_f = [0.85, 1.0, 1.05, 1.15, 1.30][occupation[i]]
        preexist_f = 1.0 + preexist[i] * 0.20
        gender_f = [1.0, 1.02, 1.0][genders[i]]
        if coverage_type == "maternity":
            gender_f = [0.01, 1.0, 0.5][genders[i]]  # mostly female
            age_f = 1.0 if 20 <= ages[i] <= 45 else 0.1

        freq_lambda[i] = base * age_f * smoking_f * exercise_f * occupation_f * preexist_f * gender_f

    # Generate actual claim counts (Poisson distributed)
    claim_counts = rng.poisson(freq_lambda)

    # ─── Severity model (cost per claim) ─────────────────────────────────
    base_severity = {"ipd": 2500, "opd": 60, "dental": 120, "maternity": 3500}
    sev_base = base_severity.get(coverage_type, 2500)

    severity_mean = np.zeros(n)
    for i in range(n):
        age_s = 1.0 + max(0, (ages[i] - 30)) * 0.006
        preexist_s = 1.0 + preexist[i] * 0.15
        region_s = [1.20, 1.05, 0.90, 1.10, 0.85, 0.75][regions[i]]
        smoking_s = [1.0, 1.10, 1.25][smoking[i]]
        occupation_s = [0.90, 1.0, 1.05, 1.10, 1.20][occupation[i]]

        severity_mean[i] = sev_base * age_s * preexist_s * region_s * smoking_s * occupation_s

    # Generate actual claim amounts (Gamma distributed)
    claim_amounts = np.zeros(n)
    for i in range(n):
        if claim_counts[i] > 0:
            claims = rng.gamma(shape=4.0, scale=severity_mean[i] / 4.0, size=claim_counts[i])
            claim_amounts[i] = np.sum(claims)

    df = pd.DataFrame({
        "age": ages, "gender": genders, "smoking": smoking, "exercise": exercise,
        "occupation": occupation, "region": regions, "preexist_conditions": preexist,
        "claim_count": claim_counts, "total_claim_amount": claim_amounts.round(2),
        "avg_claim_amount": np.where(claim_counts > 0, (claim_amounts / claim_counts).round(2), 0),
        "has_claim": (claim_counts > 0).astype(int),
    })
    return df


def train_freq_sev_model(coverage_type, seed):
    """Train frequency and severity models for a given coverage type."""
    print(f"\n{'='*60}")
    print(f"Training {coverage_type.upper()} frequency-severity model")
    print(f"{'='*60}")

    df = generate_synthetic_claims(N_SAMPLES, seed, coverage_type)

    features = ["age", "gender", "smoking", "exercise", "occupation", "region", "preexist_conditions"]
    X = df[features].values

    # ─── Frequency Model (Poisson) ───────────────────────────────────────
    y_freq = df["claim_count"].values
    print(f"\nFrequency model (Poisson regression):")
    print(f"  Mean claims/year: {y_freq.mean():.3f}")
    print(f"  Claim rate: {(y_freq > 0).mean():.1%}")

    freq_model = PoissonRegressor(alpha=0.01, max_iter=500)
    freq_model.fit(X, y_freq)

    freq_scores = cross_val_score(freq_model, X, y_freq, cv=5, scoring="neg_mean_poisson_deviance")
    print(f"  5-fold CV deviance: {-freq_scores.mean():.4f}")

    # ─── Severity Model (GBR on claims > 0) ─────────────────────────────
    mask = df["claim_count"] > 0
    X_sev = X[mask]
    y_sev = df.loc[mask, "avg_claim_amount"].values
    print(f"\nSeverity model (GradientBoosting on {mask.sum()} claimants):")
    print(f"  Mean severity: ${y_sev.mean():,.0f}")
    print(f"  Median severity: ${np.median(y_sev):,.0f}")

    sev_model = GradientBoostingRegressor(
        n_estimators=150, max_depth=4, learning_rate=0.07,
        subsample=0.85, random_state=seed,
    )
    sev_model.fit(X_sev, y_sev)

    sev_scores = cross_val_score(sev_model, X_sev, y_sev, cv=5, scoring="neg_root_mean_squared_error")
    print(f"  5-fold CV RMSE: ${-sev_scores.mean():,.0f}")
    print(f"  Training R²: {sev_model.score(X_sev, y_sev):.4f}")

    # Feature importance
    print(f"\n  Feature importance (severity):")
    for name, imp in sorted(zip(features, sev_model.feature_importances_), key=lambda x: -x[1]):
        print(f"    {name:20s} {imp:.3f} {'█' * int(imp * 40)}")

    # Stamp versions
    freq_model._model_version = f"v1.0.0"
    freq_model._coverage_type = coverage_type
    sev_model._model_version = f"v1.0.0"
    sev_model._coverage_type = coverage_type

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    freq_path = os.path.join(OUTPUT_DIR, f"{coverage_type}_freq.pkl")
    sev_path = os.path.join(OUTPUT_DIR, f"{coverage_type}_sev.pkl")
    joblib.dump(freq_model, freq_path)
    joblib.dump(sev_model, sev_path)
    print(f"\n  Saved: {freq_path} ({os.path.getsize(freq_path)/1024:.0f} KB)")
    print(f"  Saved: {sev_path} ({os.path.getsize(sev_path)/1024:.0f} KB)")

    # Sanity check
    test = np.array([[35, 0, 2, 0, 0, 0, 0]])  # 35M, current smoker, sedentary, office, phnom penh, no preexist
    pred_freq = freq_model.predict(test)[0]
    pred_sev = sev_model.predict(test)[0]
    print(f"\n  Sanity: 35M smoker sedentary → {pred_freq:.3f} claims/yr × ${pred_sev:,.0f}/claim = ${pred_freq * pred_sev:,.0f} expected")

    return freq_model, sev_model


def train():
    print("=" * 60)
    print("DAC HealthPrice — Frequency-Severity Model Training")
    print("=" * 60)

    # IPD models are pre-trained from real Kaggle data (notebook pipeline)
    # Only train OPD, Dental, Maternity from synthetic data
    ipd_freq_path = os.path.join(OUTPUT_DIR, "ipd_freq.pkl")
    ipd_sev_path = os.path.join(OUTPUT_DIR, "ipd_sev.pkl")
    if os.path.exists(ipd_freq_path) and os.path.exists(ipd_sev_path):
        print("\n  IPD models already exist (from notebook pipeline) — skipping")
    else:
        train_freq_sev_model("ipd", SEED)

    for i, cov in enumerate(["opd", "dental", "maternity"]):
        train_freq_sev_model(cov, SEED + i + 1)

    # Create a combined metadata file
    meta = {
        "version": "v2.0.0",
        "coverage_types": ["ipd", "opd", "dental", "maternity"],
        "features": ["age", "gender", "smoking", "exercise", "occupation", "region", "preexist_conditions"],
        "ipd_source": "Kaggle real data (insurance + cross-sell datasets)",
        "other_source": "Synthetic data (OPD, Dental, Maternity)",
        "frequency_model": "PoissonRegressor",
        "severity_model": "GradientBoostingRegressor",
    }
    joblib.dump(meta, os.path.join(OUTPUT_DIR, "model_meta.pkl"))
    print(f"\nAll models trained and saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    train()
