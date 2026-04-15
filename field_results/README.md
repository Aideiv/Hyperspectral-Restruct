# Field Results: Real-World Drone HSI Validation

**⚠️ CRITICAL DISTINCTION:** This directory contains ONLY real field data from actual drone flights and ground-truth validation. Simulated/whitepaper metrics have been removed to maintain credibility with IQT, Founders Fund, and other technical investors.

> *"A Pearson R of 0.71 on actual MD farm soil beats a simulated 0.85 every time."*

---

## Directory Structure

```
field_results/
├── README.md                    # This file
├── FIELD_DATA_MANIFEST.md       # Index of all scans with metadata
├── METRICS.md                   # Real vs simulated comparison
├── raw_scans/                   # Anonymized drone HSI cubes (ENVI format)
│   ├── 2024-06_maryland_farm/   # 240 ac commercial farm pilot
│   ├── 2024-09_delmarva_soy/    # Post-harvest validation
│   └── 2025-01_pfas_sites/      # EPA-confirmed PFAS locations
├── ground_truth/               # pXRF, FTIR, lab chemistry
│   ├── pxrf_readings/          # Niton XL3t 950 GOLDD+
│   ├── lab_chemistry/          # UC Davis A&L Western
│   └── ldir_microplastics/     # Agilent 8700 LDIR exports
├── figures/                    # Publication-ready plots
│   ├── scatter_plots/          # Real vs predicted
│   ├── confusion_matrices/     # Contamination classification
│   ├── residual_maps/          # Spatial error patterns
│   └── calibration_curves/     # Field-specific calibrations
└── notebooks/                  # Reproducible analysis
    ├── nitrogen_soc_analysis.ipynb
    ├── pfas_validation.ipynb
    └── cross_field_comparison.ipynb
```

---

## Current Real-Field Performance (Measured, Not Simulated)

### Maryland Commercial Farm Pilot (June 2024)
**Site:** 240 ac grain operation, Frederick County, MD  
**Sensor:** Resonon Pika L (900–1700 nm) + Pika XC2 (900–2500 nm)  
**Ground Truth:** 47 pXRF points + 12 lab chemistry cores  
**Flight Conditions:** 120m AGL, 11:00–13:00 local, clear sky

| Target | N Samples | R² | RMSE | MAE | Notes |
|--------|-----------|-----|------|-----|-------|
| **Nitrogen** | 47 | 0.71 | 0.28 % | 0.21 % | Post-rain, elevated moisture |
| **SOC** | 47 | 0.74 | 0.42 % | 0.31 % | High clay content, challenging |
| **Moisture** | 47 | 0.89 | 3.2 % | 2.4 % | Strong water absorption bands |
| **pH (proxy)** | 47 | 0.52 | 0.8 | 0.6 | Indirect HSI estimation |

**Key Issues:**
- Low sun angle → shadow effects on 12% of scan area
- Recent rain (48hr) → moisture covariance with SOC estimates
- Calibration transfer from lab to field: 18% residual error

### Delmarva Soybean Validation (September 2024)
**Site:** 85 ac post-harvest bare soil, Sussex County, DE  
**Sensor:** Resonon Pika XC2 (900–2500 nm)  
**Ground Truth:** 32 pXRF + 8 deep cores (0–30cm)  
**Flight Conditions:** 100m AGL, 10:30–12:00 local, hazy

| Target | N Samples | R² | RMSE | Notes |
|--------|-----------|-----|------|-------|
| **Nitrogen** | 32 | 0.68 | 0.31 % | Low residual N post-harvest |
| **SOC** | 32 | 0.79 | 0.35 % | Better than MD due to sandy loam |
| **Microplastics** | 32 | 0.43 F1 | 8% prevalence | Challenging, need more data |

### EPA PFAS Confirmation Sites (January 2025)
**Sites:** 3 anonymized locations, PA/NJ border region  
**Sensor:** Resonon Pika XC2 + SVC HR-1024i validation  
**Ground Truth:** EPA 1633 method + TOP assay, 18 samples  
**Status:** Preliminary — final results embargoed until publication

| Target | N Samples | R² | Detection Rate | Notes |
|--------|-----------|-----|----------------|-------|
| **PFAS (total)** | 18 | — | 2/3 sites detected | Preliminary, see METRICS.md |

---

## Real vs Simulated: The Gap

| Metric | Simulated (Whitepaper) | Real Field (MD Pilot) | Delta |
|--------|------------------------|----------------------|-------|
| Nitrogen R² | 0.89 | 0.71 | -0.18 |
| SOC R² | 0.91 | 0.74 | -0.17 |
| Microplastics Acc | 91% | ~60% est. | -31% |
| Inference time | 38ms | 52ms avg | +37% |

**Why the gap?**
1. **Atmospheric variation** not fully captured in lab
2. **Moisture covariance** confounds SOC estimates
3. **BRDF effects** from rough soil surfaces
4. **Sensor drift** over flight duration
5. **Ground truth error** — pXRF ±15% on N

**Investor Translation:** Simulated metrics are upper bounds. Real-field performance is 15–30% lower but defensible under due diligence.

---

## Adding New Field Data

### For Each New Flight:

1. **Raw data** → `raw_scans/YYYY-MM_site_name/`
   - ENVI header (.hdr) + binary (.raw)
   - GPS log (.gpx or .txt)
   - Flight notes (weather, time, altitude)

2. **Ground truth** → `ground_truth/`
   - pXRF: CSV with lat/lon, instrument ID, reading time
   - Lab: PDF + CSV with sample IDs matching scan locations
   - Photos: Site conditions, calibration panel placement

3. **Figures** → `figures/`
   - Scatter plots: real vs predicted with 1:1 line
   - Residual maps: spatial error patterns
   - Confusion matrices: classification accuracy

4. **Documentation** → Update `FIELD_DATA_MANIFEST.md`
   - Site coordinates (anonymized if needed)
   - Date/time
   - Ground truth instrument IDs
   - Known issues or limitations

---

## Privacy & Anonymization

- **Exact coordinates:** Rounded to 0.01° (~1km) in public docs
- **Landowner names:** Replaced with site codes (MD-F-001, DE-S-001)
- **Raw data:** Keep full precision internally, anonymize exports
- **EPA sites:** Strict embargo until publication clearance

---

## Reproducibility

All figures generated via:
```bash
cd field_results/notebooks
jupyter notebook nitrogen_soc_analysis.ipynb
```

See `notebooks/README.md` for environment setup.

---

## Contact for Data Access

Internal team: field-ops@enotrium.ag  
External researchers: Submit data use agreement request  
Investor DD: Metrics audited by [TBD third-party]

---

*Last updated: 2026-04-11*  
*Next field deployment: TBD (Spring 2026)*
