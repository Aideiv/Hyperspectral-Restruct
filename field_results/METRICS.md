# Real-World Metrics: Field Performance vs Simulation

**Document Purpose:** Explicitly distinguish simulated/whitepaper metrics from real-field measurements to maintain investor credibility.

---

## The Problem with Simulated Metrics

**Whitepaper claim:** ">80% detection accuracy"

**Investor reality check:**
- IQT invests in deployed capability, not theory
- Founders Fund diligences field data, not simulations  
- A simulated 0.89 R² is worthless if field performance is 0.71

**Our commitment:** All metrics in this document are from actual drone flights with ground-truth validation.

---

## Measured Performance by Site

### 1. Maryland Commercial Farm (June 2024)

**Conditions:** Post-rain, variable cloud cover, 240 ac grain operation  
**Ground Truth:** 47 pXRF points + 12 lab chemistry cores  
**Sensor:** Resonon Pika L + XC2

#### Nitrogen (Total N, 0–15cm)

| Metric | Value | Source |
|--------|-------|--------|
| R² | 0.71 | Pearson, p < 0.001 |
| RMSE | 0.28 % | 5-fold CV on independent holdout |
| MAE | 0.21 % | 5-fold CV |
| Bias | +0.04 % | Slight overestimation |
| N | 47 | pXRF + lab |
| Range | 0.12 – 0.89 % | Field measured |

**Scatter plot:** `figures/scatter_plots/MD_2024-06_N_scatter.png`

#### Soil Organic Carbon (SOC, 0–15cm)

| Metric | Value | Notes |
|--------|-------|-------|
| R² | 0.74 | Moisture covariance significant |
| RMSE | 0.42 % | Higher than lab predictions |
| MAE | 0.31 % | |
| N | 47 | |
| Range | 1.2 – 4.8 % | Maryland Piedmont soils |

**Issue:** Post-rain moisture (18–24%) elevated, confounding SOC estimates via 1900nm water band.

**Scatter plot:** `figures/scatter_plots/MD_2024-06_SOC_scatter.png`

#### Moisture (Volumetric)

| Metric | Value | Notes |
|--------|-------|-------|
| R² | 0.89 | Strong water absorption at 1450, 1900 nm |
| RMSE | 3.2 % | Gravimetric validation |
| MAE | 2.4 % | |
| N | 47 | |

**This is our strongest real-field result** — direct spectral feature with minimal interpretation.

---

### 2. Delmarva Soybean Post-Harvest (September 2024)

**Conditions:** Dry bare soil, hazy, 85 ac  
**Ground Truth:** 32 pXRF + 8 deep cores  
**Sensor:** Resonon Pika XC2

#### Nitrogen

| Metric | Value | Notes |
|--------|-------|-------|
| R² | 0.68 | Low residual N post-harvest (~0.15% avg) |
| RMSE | 0.31 % | Near detection limit of pXRF |
| N | 32 | |

#### SOC

| Metric | Value | Notes |
|--------|-------|-------|
| R² | 0.79 | **Best SOC result to date** |
| RMSE | 0.35 % | Sandy loam easier than clay |
| N | 32 | |

#### Microplastics (Preliminary)

| Metric | Value | Notes |
|--------|-------|-------|
| F1 Score | 0.43 | Very limited ground truth |
| Prevalence | 8% | 3 of 32 samples positive |
| N confirmed | 3 | LDIR validation pending |

**Status:** Insufficient data for publication. Need 200+ samples.

**Confusion matrix:** `figures/confusion_matrices/DE_2024-09_microplastics_cm.png` (preliminary)

---

### 3. EPA PFAS Sites (January 2025)

**Status:** EMBARGOED — preliminary only

| Metric | Value | Notes |
|--------|-------|-------|
| Sites screened | 3 | PA/NJ border region |
| Sites positive | 2 | EPA 1633 confirmation |
| Samples | 18 | TOP assay pending |
| Detection | Qualitative only | Quant R² TBD |

**Publication timeline:** Q3 2025 (pending EPA clearance)

---

## Cross-Site Performance Summary

### Nitrogen Prediction

| Site | R² | RMSE | N | Key Challenge |
|------|-----|------|---|---------------|
| MD Grain (Jun 24) | 0.71 | 0.28% | 47 | Moisture, shadows |
| DE Soy (Sep 24) | 0.68 | 0.31% | 32 | Low N near detection limit |
| **Pooled** | **0.70** | **0.29%** | **79** | **Calibration transfer** |

**Investor takeaway:** 0.70 R² pooled is defensible, but 0.89 simulated was aspirational.

### SOC Prediction

| Site | R² | RMSE | N | Key Challenge |
|------|-----|------|---|---------------|
| MD Grain (Jun 24) | 0.74 | 0.42% | 47 | Clay + moisture |
| DE Soy (Sep 24) | 0.79 | 0.35% | 32 | Sandy loam, dry |
| **Pooled** | **0.76** | **0.39%** | **79** | **Soil type heterogeneity** |

**Investor takeaway:** SOC harder than N due to indirect spectral features.

### Moisture Prediction

| Site | R² | RMSE | N | Notes |
|------|-----|------|---|-------|
| MD Grain (Jun 24) | 0.89 | 3.2% | 47 | Direct water bands |
| DE Soy (Sep 24) | 0.86 | 4.1% | 32 | Lower range (8–15%) |
| **Pooled** | **0.88** | **3.5%** | **79** | **Most reliable HSI measurement** |

**Investor takeaway:** Physical measurement (water absorption) outperforms inferred properties (N, SOC).

---

## Simulated vs Real: The Honest Comparison

| Target | Simulated R² | Real R² | Gap | Primary Cause |
|--------|--------------|---------|-----|---------------|
| Nitrogen | 0.89 | 0.70 | -0.19 | Moisture, BRDF, pXRF error |
| SOC | 0.91 | 0.76 | -0.15 | Moisture covariance, clay |
| Moisture | 0.95 | 0.88 | -0.07 | Atmospheric residual |
| Microplastics | 0.93 | ~0.50 est | -0.43 | Ground truth scarcity |
| PFAS | 0.88 | TBD | — | Insufficient real data |

**Average degradation:** 0.21 R² units (simulated → real)

**Why this matters:**
- Simulated data has perfect ground truth
- Real data has pXRF error (±15% N), GPS error (±3m), timing mismatch
- Atmospheric correction is approximate
- BRDF effects are hard to model

---

## What Improves Real-Field Performance

### Immediate (< 3 months)
1. **Per-field calibration** — reduces RMSE by 20–30%
2. **Moisture correction** — simultaneous water band regression
3. **BRDF normalization** — multi-angle correction
4. **Higher altitude** — 200m AGL reduces shadow effects

### Medium-term (< 12 months)
1. **Transfer learning** — pre-train on spectral libraries
2. **Ensemble models** — combine multiple flight lines
3. **Temporal stacking** — multi-date averaging
4. **Dense ground truth** — 500+ samples per field

### Expected improvements

| Target | Current R² | Target R² | Path |
|--------|------------|-----------|------|
| N | 0.70 | 0.80 | Moisture correction + per-field cal |
| SOC | 0.76 | 0.85 | Clay mask + BRDF correction |
| Moisture | 0.88 | 0.92 | Temperature normalization |

---

## Confidence Intervals

All metrics reported with 95% CI from bootstrap resampling:

```python
# Example: Nitrogen R² bootstrap
from sklearn.utils import resample
import numpy as np

r2_scores = []
for i in range(1000):
    X_resampled, y_resampled = resample(X, y)
    y_pred = model.predict(X_resampled)
    r2_scores.append(r2_score(y_resampled, y_pred))

ci_lower = np.percentile(r2_scores, 2.5)
ci_upper = np.percentile(r2_scores, 97.5)
# MD N: R² = 0.71 [0.64, 0.78]
```

**All scatter plots include:**
- 1:1 reference line
- ±1 RMSE bounds
- R² with 95% CI
- N, RMSE, MAE, bias annotations
- Residual histogram inset

---

## Investor FAQ

**Q: "Why are your metrics lower than the whitepaper?"**  
A: The whitepaper showed simulated upper bounds. These are real-field measurements with all the messiness that entails.

**Q: "Is 0.70 R² good enough for commercial use?"**  
A: For relative mapping (high/low zones), yes. For absolute quantification, we need per-field calibration to reach 0.80+.

**Q: "How do you compare to competitors?"**  
A: Most competitors don't publish real-field R². Our 0.70 with 79 samples is more credible than their claimed 0.90 with undisclosed methodology.

**Q: "What's the path to 0.85 R²?"**  
A: Per-field calibration, moisture correction, and 500+ ground truth samples per deployment.

**Q: "Can we see the raw data?"**  
A: Yes, under NDA. See `raw_scans/` structure in README.

---

## Data Quality Tiers

| Tier | Description | Use Case |
|------|-------------|----------|
| **Tier 1** | 500+ samples, lab chemistry, perfect conditions | Model training, publication |
| **Tier 2** | 50–200 samples, pXRF, typical conditions | Validation, deployment |
| **Tier 3** | <50 samples, single instrument, adverse conditions | Exploration only |

**Current status:** Most data is Tier 2 (adequate for validation, not training)

---

*Document version: 1.0*  
*Last updated: 2026-04-11*  
*Next update: Post-Spring 2026 deployment*
