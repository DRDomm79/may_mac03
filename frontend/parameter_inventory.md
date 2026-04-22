# DAC HealthPrice Parameter Inventory

All parameters extracted from `train_model.py` (lines 26–85).

---

## 1. Base Claim Rates (Claims Per Year, Line 51)

| Coverage Type | Base Frequency |
|---|---|
| IPD (Hospital) | 0.12 |
| OPD (Outpatient) | 2.5 |
| Dental | 0.8 |
| Maternity | 0.15 |

---

## 2. Base Severity Costs (Cost Per Claim, Line 74)

| Coverage Type | Base Severity |
|---|---|
| IPD (Hospital) | $2,500 |
| OPD (Outpatient) | $60 |
| Dental | $120 |
| Maternity | $3,500 |

---

## 3. Frequency Multipliers (Lines 57–68)

Applied to base frequency as: `freq_lambda = base × age_f × smoking_f × exercise_f × occupation_f × preexist_f × gender_f`

### 3.1 Age Multiplier (Line 57)
```
age_f = 1.0 + max(0, (age - 35)) × 0.008
```
**Formula interpretation**:
- At age 35: 1.0× (baseline)
- At age 45: 1.0 + (10 × 0.008) = 1.08×
- At age 65: 1.0 + (30 × 0.008) = 1.24×

**Special case**: Children under 5 → 1.3×

### 3.2 Smoking Multiplier (Line 59)
| Status | Multiplier |
|---|---|
| Never (code 0) | 1.0× |
| Former (code 1) | 1.15× |
| Current (code 2) | 1.40× |

### 3.3 Exercise Multiplier (Line 60)
| Level | Multiplier |
|---|---|
| Sedentary (code 0) | 1.20× |
| Light (code 1) | 1.05× |
| Moderate (code 2) | 0.90× |
| Active (code 3) | 0.80× |

### 3.4 Occupation Multiplier (Line 61)
| Type | Multiplier |
|---|---|
| Office/Desk (code 0) | 0.85× |
| Retail/Service (code 1) | 1.0× |
| Healthcare (code 2) | 1.05× |
| Manual Labor (code 3) | 1.15× |
| Industrial/High-Risk (code 4) | 1.30× |

### 3.5 Pre-Existing Conditions Multiplier (Line 62)
```
preexist_f = 1.0 + preexist_conditions × 0.20
```
**Codes**: 0, 1, 2, 3 (number of conditions)

| # Conditions | Multiplier |
|---|---|
| 0 | 1.0× |
| 1 | 1.2× |
| 2 | 1.4× |
| 3 | 1.6× |

### 3.6 Gender Multiplier — General (Line 63)
| Gender | Multiplier |
|---|---|
| Male (code 0) | 1.0× |
| Female (code 1) | 1.02× |
| Other (code 2) | 1.0× |

### 3.7 Gender Multiplier — Maternity Only (Line 65)
| Gender | Multiplier |
|---|---|
| Male (code 0) | 0.01× |
| Female (code 1) | 1.0× |
| Other (code 2) | 0.5× |

### 3.8 Age Multiplier — Maternity Only (Line 66)
```
age_f = 1.0 if 20 ≤ age ≤ 45 else 0.1
```

---

## 4. Severity Multipliers (Lines 78–85)

Applied to base severity as: `severity_mean = sev_base × age_s × preexist_s × region_s × smoking_s × occupation_s`

### 4.1 Age Multiplier (Line 79)
```
age_s = 1.0 + max(0, (age - 30)) × 0.006
```
**Formula interpretation**:
- At age 30: 1.0× (baseline)
- At age 50: 1.0 + (20 × 0.006) = 1.12×
- At age 70: 1.0 + (40 × 0.006) = 1.24×

### 4.2 Pre-Existing Conditions Multiplier (Line 80)
```
preexist_s = 1.0 + preexist_conditions × 0.15
```
| # Conditions | Multiplier |
|---|---|
| 0 | 1.0× |
| 1 | 1.15× |
| 2 | 1.30× |
| 3 | 1.45× |

### 4.3 Region Multiplier (Line 81)
| Region (Code Index) | Location | Multiplier |
|---|---|---|
| 0 | Phnom Penh | 1.20× |
| 1 | Siem Reap | 1.05× |
| 2 | Battambang | 0.90× |
| 3 | Sihanoukville | 1.10× |
| 4 | Kampong Cham | 0.85× |
| 5 | Rural Areas | 0.75× |
| (Vietnam) | Ho Chi Minh City | 1.25× |
| (Vietnam) | Hanoi | 1.20× |
| (Vietnam) | Da Nang | 1.05× |
| (Vietnam) | Can Tho | 0.90× |
| (Vietnam) | Hai Phong | 0.95× |

### 4.4 Smoking Multiplier (Line 82)
| Status | Multiplier |
|---|---|
| Never (code 0) | 1.0× |
| Former (code 1) | 1.10× |
| Current (code 2) | 1.25× |

### 4.5 Occupation Multiplier (Line 83)
| Type | Multiplier |
|---|---|
| Office/Desk (code 0) | 0.90× |
| Retail/Service (code 1) | 1.0× |
| Healthcare (code 2) | 1.05× |
| Manual Labor (code 3) | 1.10× |
| Industrial/High-Risk (code 4) | 1.20× |

---

## 5. Population Distribution (Lines 42–47)

Default synthetic population composition (when generating 10,000 samples):

### 5.1 Gender Distribution
| Gender | Probability |
|---|---|
| Male | 48% |
| Female | 48% |
| Other | 4% |

### 5.2 Smoking Distribution
| Status | Probability |
|---|---|
| Never | 55% |
| Former | 25% |
| Current | 20% |

### 5.3 Exercise Distribution
| Level | Probability |
|---|---|
| Sedentary | 30% |
| Light | 35% |
| Moderate | 25% |
| Active | 10% |

### 5.4 Occupation Distribution
| Type | Probability |
|---|---|
| Office/Desk | 35% |
| Retail/Service | 25% |
| Healthcare | 15% |
| Manual Labor | 15% |
| Industrial/High-Risk | 10% |

### 5.5 Pre-Existing Conditions Distribution
| # Conditions | Probability |
|---|---|
| 0 | 50% |
| 1 | 25% |
| 2 | 15% |
| 3 | 10% |

### 5.6 Age Distribution
| Range | Notes |
|---|---|
| 18–75 | Uniform random |

### 5.7 Region Distribution
| Code | Count | Notes |
|---|---|---|
| 0–5 | Equal probability | Cambodia regions (6 options) |
| Vietnam regions | Hardcoded in `REGION_FACTORS` dict | 11 regions total (6 Cambodia + 5 Vietnam) |

---

## 6. Gamma Shape Parameter (Line 91)

**Severity distribution shape**: 4.0 (fixed)

```python
claims = rng.gamma(shape=4.0, scale=severity_mean[i] / 4.0, size=claim_counts[i])
```

This defines the distribution *shape* of individual claim amounts around the mean severity.

---

## 7. Model Hyperparameters (Lines 121, 137)

### 7.1 Poisson Regressor (Frequency Model)
```python
PoissonRegressor(alpha=0.01, max_iter=500)
```
- `alpha`: L2 regularization strength
- `max_iter`: Maximum iterations for solver

### 7.2 Gamma Regressor (Severity Model)
```python
GammaRegressor(alpha=0.01, max_iter=500)
```
- `alpha`: L2 regularization strength
- `max_iter`: Maximum iterations for solver

---

## 8. Training Configuration (Lines 21–22)

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Training samples per coverage type | 10,000 |
| Cross-validation folds | 5 |

---

## Summary for Thesis

**Total parameters to source**:
- 4 base claim rates
- 4 base severity costs
- 18 multipliers (frequency & severity across all factors)
- 11 regional factors
- 5+ population distributions

**Easiest to source** (likely published):
- Base claim rates (WHO, World Bank)
- Age multipliers (SOA actuarial tables)
- Regional variation (insurance regulator, reinsurance)

**Hardest to source** (may not exist in literature):
- Exercise level multipliers
- Exact occupation multipliers for health insurance
- Cambodia-specific combinations

