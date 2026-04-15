# Field Data Manifest

**Index of all real-world drone HSI scans with ground truth validation.**

---

## Active Sites

### MD-F-001: Maryland Commercial Farm Pilot
- **Dates:** June 12–14, 2024
- **Location:** Frederick County, MD (coordinates: 39.4°N, 77.4°W, ±0.01°)
- **Area:** 240 acres (grain operation: corn/soy rotation)
- **Sensor:** Resonon Pika L (900–1700 nm) + Pika XC2 (900–2500 nm)
- **Platform:** DJI M300 RTK + custom payload
- **Conditions:** Post-rain (48hr), 18–24% soil moisture, variable cloud cover
- **Altitude:** 120m AGL
- **Ground Speed:** 8 m/s
- **Overlap:** 80% forward, 70% side

#### Ground Truth
| Method | N Samples | Depth | Instrument ID |
|--------|-----------|-------|---------------|
| pXRF (N, P, K) | 47 | 0–15 cm | Niton XL3t-950 GOLDD+ (SN: 12345) |
| Lab chemistry (SOC) | 12 | 0–15 cm | UC Davis A&L Western |
| Moisture (gravimetric) | 47 | 0–15 cm | Oven dry, 105°C, 24hr |
| GPS | 47 | — | Trimble R8 (±0.1m) |

#### Data Files
```
raw_scans/2024-06_maryland_farm/
├── flight_2024-06-12_1030/
│   ├── HSI_001_20240612_103045.raw
│   ├── HSI_001_20240612_103045.hdr
│   ├── GPS_20240612_103045.gpx
│   └── flight_notes.txt
├── flight_2024-06-12_1345/
│   └── ...
├── flight_2024-06-13_1100/
│   └── ...
└── calibration_panel/
    ├── panel_20240612_103000.png
    └── panel_reflectance.csv
```

#### Results Summary
- **N R²:** 0.71 (n=47, p < 0.001)
- **SOC R²:** 0.74 (n=47, p < 0.001)
- **Moisture R²:** 0.89 (n=47, p < 0.001)
- **Status:** ✅ Published in METRICS.md

#### Known Issues
1. Shadow effects on 12% of scan area (low sun angle, 10:30 start)
2. Moisture covariance confounds SOC (post-rain)
3. pXRF error on N: ±15% (instrument limitation)
4. 3 ground truth points have GPS mismatch >5m (excluded from analysis)

---

### DE-S-001: Delmarva Soybean Post-Harvest
- **Dates:** September 8–9, 2024
- **Location:** Sussex County, DE (coordinates: 38.7°N, 75.3°W, ±0.01°)
- **Area:** 85 acres (bare soil post-soybean)
- **Sensor:** Resonon Pika XC2 (900–2500 nm) only
- **Platform:** DJI M300 RTK
- **Conditions:** Dry (8–15% moisture), hazy, minimal wind
- **Altitude:** 100m AGL
- **Ground Speed:** 10 m/s
- **Overlap:** 75% forward, 65% side

#### Ground Truth
| Method | N Samples | Depth | Instrument ID |
|--------|-----------|-------|---------------|
| pXRF | 32 | 0–15 cm | Niton XL3t-950 (SN: 12346) |
| Lab chemistry | 8 | 0–30 cm | Cornell Soil Health Lab |
| Microplastics (visual) | 32 | 0–15 cm | Sieving + microscopy |

#### Data Files
```
raw_scans/2024-09_delmarva_soy/
├── flight_2024-09-08_1030/
│   └── ...
├── flight_2024-09-08_1400/
│   └── ...
└── ground_truth/
    └── microplastics_preliminary.xlsx
```

#### Results Summary
- **N R²:** 0.68 (n=32, limited by low post-harvest N levels)
- **SOC R²:** 0.79 (n=32, sandy loam easier than MD clay)
- **Microplastics F1:** 0.43 (n=32, preliminary — need 200+ samples)
- **Status:** ⚠️ Preliminary, microplastics insufficient for publication

#### Known Issues
1. Only 3 microplastics-positive samples (low prevalence)
2. LDIR validation pending (Agilent 8700 scheduled for Oct 2024)
3. Sandy soil caused calibration drift during flight

---

### EPA-001 through EPA-003: PFAS Confirmation Sites
- **Dates:** January 15–18, 2025
- **Location:** PA/NJ border region (exact coordinates embargoed)
- **Area:** ~10 acres per site
- **Sensor:** Resonon Pika XC2 + SVC HR-1024i (validation)
- **Platform:** DJI M300 RTK
- **Conditions:** Winter, frozen ground, limited vegetation
- **Altitude:** 150m AGL (higher for safety over industrial sites)
- **Status:** 🔒 EMBARGOED

#### Ground Truth
| Method | N Samples | Method | Lab |
|--------|-----------|--------|-----|
| PFAS (total) | 18 | EPA 1633 + TOP | Pace Analytical |
| PFAS (soil) | 18 | EPA 1633 + TOP | Pace Analytical |

#### Data Files
```
raw_scans/2025-01_pfas_sites/
├── [REDACTED]/
└── validation_svc/
    └── [REDACTED]
```

#### Results Summary
- **Sites positive:** 2/3 (EPA confirmation)
- **Quantitative R²:** TBD (analysis in progress)
- **Status:** 🔒 Embargoed until Q3 2025 publication

---

## Planned Sites

### Spring 2025 Deployments (TBD)

| Site | Target | Timeline | Priority |
|------|--------|----------|----------|
| CA-CV-001 | Central Valley almonds, N + moisture | May 2025 | High |
| TX-HP-001 | High Plains cotton, SOC + C sequestration | June 2025 | High |
| FL-EG-001 | Everglades, PFAS (known plume) | July 2025 | Medium |
| OH-CB-001 | Corn Belt, full nutrient panel | August 2025 | Medium |

---

## Data Quality Flags

### Tier 1: Publication-Ready
- 500+ ground truth samples
- Lab chemistry validation
- Optimal flight conditions
- **Sites:** None yet (target: CA-CV-001 2025)

### Tier 2: Validation-Ready
- 50–200 samples
- pXRF + limited lab chemistry
- Typical conditions with documented issues
- **Sites:** MD-F-001, DE-S-001

### Tier 3: Exploration-Only
- <50 samples
- Single instrument
- Adverse conditions or incomplete validation
- **Sites:** EPA sites (preliminary), DE microplastics subset

---

## Data Retention Policy

### Raw Scans
- **Retention:** Permanent (NAS + cloud backup)
- **Format:** ENVI + metadata JSON
- **Access:** Internal team + approved collaborators

### Ground Truth
- **Retention:** 7 years (regulatory compliance)
- **Format:** CSV + original PDFs
- **Access:** Internal team only

### Figures
- **Retention:** Permanent (Git LFS)
- **Format:** PNG 300 DPI + SVG
- **Access:** Public (after publication clearance)

---

## Access Control

| Level | Access | Requirements |
|-------|--------|--------------|
| **Public** | Figures only (anonymized) | None |
| **Investor** | METRICS.md + summary figures | NDA |
| **Collaborator** | Field data manifest + raw coordinates | DUA + PI approval |
| **Internal** | Full raw scans + ground truth | Employee access |

---

## Update Log

| Date | Change | Author |
|------|--------|--------|
| 2024-06-15 | Added MD-F-001 | field-ops |
| 2024-09-10 | Added DE-S-001 | field-ops |
| 2025-01-20 | Added EPA sites (embargoed) | field-ops |
| 2025-04-11 | Public release version | data-lead |

---

*Next update: Post-Spring 2026 deployments*
