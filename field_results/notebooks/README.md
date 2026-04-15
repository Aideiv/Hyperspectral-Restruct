# Field Analysis Notebooks

Reproducible analysis notebooks for real-world drone HSI validation data.

---

## Quick Start

```bash
cd field_results/notebooks

# Setup environment (one-time)
conda env create -f environment.yml
conda activate field-analysis

# Launch Jupyter
jupyter lab
```

---

## Notebooks

### 1. `nitrogen_soc_analysis.ipynb`
**Purpose:** Primary analysis of N and SOC predictions from real field data  
**Inputs:** `../raw_scans/`, `../ground_truth/pxrf_readings/`  
**Outputs:**
- Real vs predicted scatter plots (with 95% CI)
- Residual maps
- Per-field calibration curves
- Statistical summary tables

**Key sections:**
1. Load and merge HSI + ground truth
2. Coordinate matching (GPS ±3m tolerance)
3. Model inference on flight cubes
4. Scatter plot generation (R², RMSE, MAE, bias)
5. Bootstrap confidence intervals
6. Cross-site comparison

---

### 2. `pfas_validation.ipynb`
**Purpose:** EPA PFAS site analysis (embargoed sections marked)  
**Inputs:** `../raw_scans/2025-01_pfas_sites/` (restricted)  
**Outputs:**
- Detection maps
- Confusion matrices
- SVC validation comparison

**Status:** 🔒 Most sections embargoed until Q3 2025

---

### 3. `cross_field_comparison.ipynb`
**Purpose:** Compare model performance across multiple field sites  
**Inputs:** All site data in `../raw_scans/`  
**Outputs:**
- Pooled metrics across sites
- Site-to-site calibration transfer analysis
- Cross-field false positive rates

**Key sections:**
1. Per-site performance summary
2. Calibration degradation analysis
3. Soil-type effect on accuracy
4. Moisture covariance analysis

---

## Data Loading Template

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spectral import envi
from sklearn.metrics import r2_score, mean_squared_error
import geopandas as gpd

# Load HSI cube
cube = envi.open('path/to/HSI.hdr', 'path/to/HSI.raw')

# Load ground truth
pxrf = pd.read_csv('../ground_truth/pxrf_readings/MD_2024-06.csv')

# Match by GPS (±3m tolerance)
def match_points(hsi_coords, ground_truth_df, tolerance=3.0):
    matches = []
    for i, (x, y) in enumerate(hsi_coords):
        distances = np.sqrt((ground_truth_df.x - x)**2 + (ground_truth_df.y - y)**2)
        if distances.min() < tolerance:
            matches.append((i, distances.idxmin()))
    return matches

# Extract and analyze
# ... (full template in notebooks)
```

---

## Environment

```yaml
# environment.yml
name: field-analysis
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - scikit-image
  - spectral  # for ENVI files
  - geopandas
  - rasterio
  - shapely
  - jupyterlab
  - pip
  - pip:
    - pyproj
    - contextily  # for basemaps
```

---

## Output Organization

Each notebook saves outputs to:
- `../figures/scatter_plots/`
- `../figures/confusion_matrices/`
- `../figures/residual_maps/`
- `../figures/calibration_curves/`

With naming convention: `{SITE}_{YYYY-MM}_{TARGET}_{type}.png`

Examples:
- `MD_2024-06_N_scatter.png`
- `DE_2024-09_microplastics_cm.png`
- `pooled_2024_N_calibration.png`

---

## Reproducibility Checklist

Before committing figures:
- [ ] Random seed set for bootstrap
- [ ] Git commit hash noted in figure caption
- [ ] Model version recorded
- [ ] Ground truth instrument IDs logged
- [ ] Weather conditions documented
- [ ] Known issues flagged

---

## Collaboration

**For external collaborators:**
1. Sign Data Use Agreement (DUA)
2. Get added to `collaborators/` group
3. Access only sites specified in DUA
4. Acknowledge in publications: "Data provided by Enotrium Agricultural Intelligence"

**For publication:**
All notebooks must pass:
- Internal peer review
- Data provenance audit
- PI sign-off on conclusions

---

*Last updated: 2026-04-11*
