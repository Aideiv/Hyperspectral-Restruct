*Hyperspectral Soil Spectrum*
**Enotrium // Agri-Intelligence**
 
# Hyperspectral CNN for soil chemistry prediction.

### Use hyperspectral drones to predict soil chemistry instantly. This data determines what gets grown, contracted, and turned into industrial materials. The Icarus Model processes drone-captured spectral data to estimate soil composition and is deployed in Enotrium’s pipeline to contract farms and optimize cash crop economics.

Drone + Hyperspectral Imaging (HSI) - Remote sensing &amp; verification layer

**Eyes Down. Soil Analytes.**  
Autonomous hyperspectral remote sensing & AI verification layer for regenerative agriculture and national-security-grade feedstock assurance.

## Core Capabilities
- UAV/drone hyperspectral imaging (400–2500 nm, SWIR focus)
- 3D DNN models for contaminant detection (microplastics, heavy metals, PFAS, glyphosate, etc.)
- Phytoremediation mapping & soil health scoring
- Real-time verification for traceable, contaminant-free bio-materials
- Edge/offline deployment for austere environments
- Direct integration with AIP0 (zk-secured provenance) and Bioverge (feedstock qualification)

📂 **Field Data:** See `field_results/` for actual drone flight results with ground truth validation (79 samples, MD + DE pilots). Includes real vs predicted scatter plots for N, SOC, moisture, and confusion matrices.

## Abstract

This pipeline converts raw spectral data into actionable soil intelligence.

Instead of sampling soil manually, we:
- scan entire fields using UAV hyperspectral imaging
- reconstruct chemical and biological properties from spectral signatures
- generate maps that directly inform agricultural and industrial decisions
- track seed residuals for phenotyping genomic data

The output is a layer over land use and input economics.

Soil chemistry determines crop yield and input costs.
By predicting soil conditions at scale, we can:
- contract farms based on expected output
- optimize crop selection and downstream material propertie
- control downstream material tensile stregnth

The model output directly informs:
- land valuation
- crop selection
- contamination detection
- input optimization

## Deployment
- Integrated with drone-based hyperspectral imaging system  
- Used in MD commercial farm contracting workflows  
- Connected to hemp processing and logistics pipeline  


### Introduction
Current approaches rely on RGB or multispectral indices, discard most spectral information.  Artificial intelligence approaches to environmental monitoring rely heavily on low-dimensional RGB or multispectral indices such as NDVI. Human perception of soil health operates as a fundamentally subjective and limited experience — we cannot directly observe the hundreds of spectral reflectance bands that encode contamination profiles, microbial activity, nutrient density, and CO₂ sequestration capacity. We’ve utilized publicly available UAV hyperspectral imaging (HSI) datasets and simulated drone-captured reflectance cubes to demonstrate that spectral activation patterns contain rich diagnostic information that can inform precise phytoremediation, land valuation, and regenerative supply-chain transparency.

This work implements a **Convolutional Neural Network (CNN)** architecture optimized for soil health classification and contaminant mapping from UAV HSI data. While HSI presents known limitations in atmospheric interference and the need for ground-truth calibration, deep learning architectures may be able to extract subtle spectral-spatial patterns that traditional analysis methods miss.

### System Architecture Flow 

        ┌──────────────────────────────┐
        │   UAV Drone (HSI Sensor)     │
        │ 400–2500 nm Spectral Capture │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │ Hyperspectral Data Cube      │
        │ (x, y, λ) ~ 200+ bands       │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │ Preprocessing Pipeline       │
        │ - Smoothing                  │
        │ - Continuum Removal          │
        │ - Normalization              │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │ 3D CNN Model                 │
        │ Spectral-Spatial Learning    │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │ Soil Prediction Maps         │
        │ - DNN:Nutrients (N, C)       │
        │ - Contaminants (PFAS, MPs)   │
        │ - Moisture / Structure       │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │ Field-Level Decisions        │
        │ - Land valuation             │
        │ - Crop selection             │
        │ - Remediation targeting      │
        └──────────────────────────────┘

### Theoretical Framework
#### Spectral Signatures of Soil Health Stages
Soil systems progress through distinct health stages characterized by shifting reflectance patterns across visible, near-infrared (NIR), and short-wave infrared (SWIR) bands. These transitions are particularly evident in regions affected by legacy agrochemicals, heavy metals, PFAs, glyphosate residues, and microbial imbalances.

HSI captures these stages through high-dimensional reflectance vectors, providing an indirect but reliable measure of chemical and biological activity. Research has demonstrated strong correlations between specific spectral features and soil contaminants, nutrient profiles, and ecological restoration potential.
### Microplastics Detection

Microplastics (MPs, particles <5 mm) are among the fastest-growing and most insidious soil contaminants worldwide. They originate from plastic mulch degradation, biosolids application, tire wear, atmospheric deposition, and irrigation water. Once in soil they alter microbial communities, reduce water retention, impair root growth, and enter the food chain, directly threatening regenerative agriculture and long-term soil health.

Our 400–2500 nm hyperspectral pipeline is exceptionally well-suited for non-invasive MP detection from UAV platforms. While visible and near-infrared (VNIR, 400–1000 nm) can only provide limited discrimination, the **short-wave infrared (SWIR, 1000–2500 nm)** region captures the diagnostic overtone and combination absorption bands of synthetic polymers that are absent in natural soil matrices.

#### LDIR Ground-Truth Verification
Use `verification/ldir_verification.py` to cross-validate 3D CNN predictions against Agilent 8700 LDIR lab results (Jia et al. 2022 and similar studies). Supports CSV exports from LDIR software.

#### Key Spectral Signatures
Common polymers exhibit distinct C–H, C–O, and C=O vibrational features:

- **Polyethylene (PE)**: strong first-overtone C–H bands at ~1720–1760 nm and second-overtone features near 1210 nm  
- **Polypropylene (PP)**: characteristic absorption around 1150–1200 nm and 1700 nm  
- **Polyamide (PA), Polystyrene (PS), PET**: additional aromatic C–H and carbonyl features between 1600–1900 nm and 2100–2300 nm  

These narrow, polymer-specific absorption valleys contrast sharply with the broad, relatively featureless spectra of mineral-dominated soils and organic matter.

Recent laboratory SWIR-HSI studies (using MCT sensors covering 1000–2500 nm) have demonstrated detection of PE, PP, and PA microplastics at concentrations as low as **0.01 wt%**, with overall classification accuracies >93 % when coupled with modern machine-learning models. Our 3D CNN architecture naturally exploits both the **full spectral depth** (capturing these subtle absorption patterns across all ~200 bands) and the **spatial context** (fragment size, shape, and clustering) within a single forward pass. The 3×3×B kernels learn exactly the spectral-spatial signatures that distinguish MP hotspots from background soil variability, even under realistic UAV conditions such as varying illumination and partial pixel mixing.

By integrating microplastic mapping into the same end-to-end pipeline used for heavy-metal, PFAS, and glyphosate detection, the model delivers unified “soil health + contamination” cubes. These outputs directly support:
- Targeted phytoremediation planning
- Regenerative supply-chain verification
- Tokenized land-valuation models that reward MP-free, high-integrity soil

Future multi-modal fusion (HSI + XRF + microbiome data) will further refine sub-pixel quantification and polymer typing, enabling the first scalable, drone-based microplastic monitoring system for regenerative agriculture.

#### Diagnostic Absorption Bands

The model is trained to detect the following characteristic SWIR absorption features of common microplastic polymers:

**Polyethylene (PE)**  
\[\lambda_{\text{PE}} \approx 1210\,\text{nm} \quad (\text{C-H 2nd overtone}), \quad 1725{-}1760\,\text{nm} \quad (\text{C-H 1st overtone})\]

**Polypropylene (PP)**  
\[\lambda_{\text{PP}} \approx 1155{-}1200\,\text{nm}, \quad 1700{-}1735\,\text{nm}\]

**Polystyrene (PS) and Polyethylene Terephthalate (PET)**  
\[\lambda \approx 1600{-}1750\,\text{nm}, \quad 2100{-}2350\,\text{nm} \quad (\text{aromatic C-H + C=O combinations})\]

### Methods
Our implementation addresses two core challenges: extracting meaningful patterns from complex high-dimensional HSI cubes and developing architectures capable of learning from these patterns. This section outlines our approach in three parts: data preprocessing implementation, HSI-specific augmentation strategies, and spectral-aware CNN architecture design.

### Data Collection and Processing
The implementation utilizes hyperspectral reflectance cubes collected via UAV drones (400–2500 nm, ~5–10 nm band resolution) supplemented with satellite data and portable XRF ground-truth samples, following the protocols in the Enotrium whitepaper. Each dataset provides specific insights into contamination and remediation dynamics.

**Preprocessing Pipeline**  
Our implementation uses a three-stage preprocessing approach optimized for spectral pattern preservation:

\[\text{Preprocess}(I) = N\left(R\left(S(I)\right)\right)\]

where \( S \) performs dimension validation, \( R \) applies spatial resizing, and \( N \) implements spectral normalization.

## Usage
### Broadband Reflectance Features

In addition to the full hyperspectral cube, the following broadband statistics are computed per pixel (or per region of interest) to capture overall soil brightness, texture variation, and moisture-related properties:

- **Mean reflectance and standard deviation** in the visible spectrum (VIS; 400–700 nm), near-infrared (NIR; 700–1300 nm), and short-wave infrared (SWIR; 1300–2500 nm)
- **NIR/SWIR ratio**: \( R_{860\,\text{nm}} / R_{1610\,\text{nm}} \)

Mean reflectance captures overall soil brightness, while the standard deviation reflects variation due to crop residue, bare soil patches, or surface heterogeneity. The NIR/SWIR ratio is a strong proxy for surface moisture and texture.

### HSI Indices Used

The pipeline also computes the following widely adopted hyperspectral indices. These serve both as interpretable features for downstream analysis and as strong baselines against which the 3D CNN’s learned spectral-spatial patterns are evaluated:

**NDVI** (Normalized Difference Vegetation Index)  

Indexes density and health of vegetation/foliage. Healthy foliage strongly absorbs red light.  
\[\text{NDVI} = \frac{R_{\text{NIR}} - R_{\text{Red}}}{R_{\text{NIR}} + R_{\text{Red}}}\]

**Red-edge slope and position**  
The “red edge” marks the sharp transition from strong absorption in the red to high reflectance in the NIR. Both the slope and the wavelength of the inflection point are sensitive indicators of foliage vitality and stress.

**NDWI** (Normalized Difference Water Index)  

\[\text{NDWI} = \frac{R_{860} - R_{1240}}{R_{860} + R_{1240}}\]  

Higher NDWI values indicate greater surface water content. It also serves as a proxy for electrical conductivity, contaminant transport potential, and redox-sensitive metal mobility.

**NIR/SWIR ratio** (860 nm / 1610 nm)  
Directly related to water absorption. A lower ratio signals higher water absorption and therefore wetter surface conditions, which correlate with increased microbial activity and higher mobility of many soil contaminants.

from hyperspectral_soil.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline(
    smoothing=True,
    normalization="snv",
    continuum=True,
    wavelength_range=(400, 2500)
)

X_processed, wavelengths = pipeline.transform(X, wavelengths)

### Preprocessing hyperspectral data
Raw spectra → Smoothing → Continuum Removal → Normalization → Filtered spectra



#### Dimension Validation
HSI acquisitions vary in spatial and spectral dimensionality. Our validation ensures consistent cube shape while preserving full spectral information:

\[
I_{\text{valid}} = \begin{cases}
I & \text{if shape matches target} \\
\text{pad/trim}(I) & \text{otherwise}
\end{cases}
\]

This validation maintains spatial integrity while ensuring proper band alignment across flights and sensors.

#### Spatial Resizing
The implementation standardizes spatial dimensions while maintaining geospatial proportions through bilinear interpolation:

\[
I_{\text{resized}}[i,j,b] = \sum_{m,n} I[m,n,b] \cdot w(i-m,j-n)
\]

Target dimensions balance spatial resolution and computational efficiency for UAV-scale data.

#### Spectral Normalization
Following HSI preprocessing protocols, we implement band-wise normalization accounting for reflectance dynamics:

\[
I_{\text{norm}}(b) = \frac{I(b) - \mu_b}{\sigma_b + \epsilon}
\]

where \( \mu_b \) and \( \sigma_b \) are the mean and standard deviation at band \( b \), and \( \epsilon \) prevents division by zero.

This normalization preserves relative spectral signatures while standardizing intensity across flights, sensors, and atmospheric conditions.

### Data Augmentation Strategies
Our implementation includes a comprehensive suite of domain-specific augmentation techniques designed to enhance model robustness while respecting the unique characteristics of UAV HSI data:

**Spectral Masking**  
Adaptive random-length masks (1–20 contiguous bands) simulate sensor dropouts or atmospheric absorption while maintaining spectral coherence.

**Spatial Masking**  
Structured dropout in the spatial domain with contiguous region masking to handle regional variations in drone imagery.

**Elastic Deformation**  
Geometrically-constrained deformations that preserve geospatial plausibility and introduce realistic variations from flight altitude and terrain.

### Model Architecture
Our architecture is a CNN tailored for hyperspectral cubes as detailed in the Enotrium whitepaper. The implementation consists of three primary components optimized for the unique high-dimensional nature of soil reflectance data:

**Convolution Operation**  
Each pixel in the HSI cube is treated as a high-dimensional reflectance vector \( \mathbf{r} \in \mathbb{R}^{B} \) (typically \( B \approx 200 \) bands). A 3D kernel \( K \) of size \( 3 \times 3 \times B \) is convolved across the cube. The output feature value at spatial location \( (i,j) \) for a given kernel is computed via dot product:

\[f(i,j) = \sum_{m=-1}^{1} \sum_{n=-1}^{1} \sum_{b=1}^{B} I(i+m, j+n, b) \cdot K(m, n, b)\]

Multiple kernels produce a stack of feature maps \( F \in \mathbb{R}^{h \times w \times C} \), where \( C \) is the number of output channels. Subsequent layers apply further convolutions, batch normalization, and GELU activations.

**Regularization – Dropout**  
We apply spatial dropout for hyperspectral feature maps to combat overfitting in high-dimensional spectral space. For a feature map tensor \( F \in \mathbb{R}^{h \times w \times C} \):

\[F_{\text{drop}}(i,j,c) = F(i,j,c) \cdot M_c \quad \text{where} \quad M_c \sim \text{Bernoulli}(p)\]

(with \( p \) increasing with network depth). This forces the model to learn distributed spectral-spatial patterns rather than relying on any single band or pixel region.

**Channel Reduction & Progressive Processing**  
Initial dimensionality reduction (full 200+ bands → 64 channels) followed by deeper convolutional blocks with increasing dropout probability, mirroring the whitepaper’s emphasis on preventing overfitting while preserving low-level spectral features.

### Hyperdimensional Computing (HDC) — Model Gluing

Based on **"Gluing Neural Networks Symbolically Through Hyperdimensional Computing"** (arXiv:2205.15534, Sutor et al.), we implement a novel ensemble technique that fuses multiple neural networks at the symbolic level using high-dimensional binary vectors.

**Core Concept**
Traditional ensembles average predictions from multiple models. HDC "glues" models by encoding their pre-classification outputs as binary hypervectors and combining them through **consensus bundling**—element-wise summation followed by binarization. This enables:

- **Online Learning**: Add/remove models without retraining existing ones
- **Life-Long Learning**: Bundle hypervectors over time as new data arrives
- **Minimal Overhead**: Hypervector operations are 1000× faster than neural forward passes
- **Symbolic Fusion**: Models fuse at the representation level, not just output level

**Architecture Variants**

| Variant | Description | Use Case |
|---------|-------------|----------|
| `hdc` | Single model with HDC classification | Online adaptation, fast retraining |
| `ensemble_hdc` | Multiple models fused via HV consensus | Maximum accuracy, model diversity |

**How It Works**

```
Model A Output → Encode → Hypervector A ─┐
                                           ├→ Bundle → Consensus HV → Classify
Model B Output → Encode → Hypervector B ─┘
```

1. **Encoding**: Neural outputs (logits) are projected to high-dimensional binary space via random/learned projection
2. **Bundling (⊕)**: Hypervectors are summed and binarized to form consensus
3. **Classification**: Query hypervector similarity to class hypervectors (cosine similarity)

**Usage Example**

```python
from model import create_model, create_ensemble_hdc

# Single HDC model
model = create_model("hdc", hv_dim=10000, num_classes=5)

# Ensemble with HDC fusion
base_a = create_model("base")
base_b = create_model("se")
ensemble = create_ensemble_hdc([base_a, base_b], hv_dim=10000)
```

**Key Hyperparameters**

- `hv_dim`: Hypervector dimension (1000–10000, default 10000). Higher = more capacity but more memory
- `learnable_weights`: Learn ensemble weights via backprop (default True) vs. fixed equal weighting

**Performance Notes**

- HDC ensemble adds <5% parameter overhead vs. base models
- Inference overhead is negligible (~1ms) compared to CNN forward pass
- Supports both differentiable training and non-differentiable online updates

See `train_hdc_example.py` for complete training examples in three modes: single HDC, ensemble HDC, and online learning.

### Training Protocol
**Mixed Precision Training**  
Dynamic loss scaling for numerical stability with high-dimensional cubes.

**Optimization Strategy**  
AdamW optimizer with parameters validated for spectral data: learning rate \( 3 \times 10^{-4} \), weight decay 0.05.

**Learning Rate Schedule**  
Custom warmup-decay schedule optimized for HSI convergence.

**Regularization and Early Stopping**  
Label smoothing, gradient clipping (norm 5.0), and early stopping based on validation loss plateau.

### Results: Real Field vs Simulated

⚠️ **Critical distinction for investors:** We report both simulated upper bounds and real-field measurements. Simulated metrics are theoretical; real-field metrics are what withstand due diligence.

#### Real-Field Performance (Measured)
From actual drone flights with pXRF + lab chemistry ground truth:

| Target | R² | RMSE | N | Sites | Status |
|--------|-----|------|---|-------|--------|
| **Nitrogen** | **0.70** | 0.29 % | 79 | MD + DE | ✅ Published |
| **SOC** | **0.76** | 0.39 % | 79 | MD + DE | ✅ Published |
| **Moisture** | **0.88** | 3.5 % | 79 | MD + DE | ✅ Strongest result |
| **Microplastics** | **~0.50 F1** | — | 32 | DE | ⚠️ Preliminary |

**Field sites:** Maryland commercial farm (240 ac, June 2024) + Delaware soybean (85 ac, September 2024)

**Figures:** Real vs predicted scatter plots, confusion matrices in `field_results/figures/`

#### Simulated Upper Bounds (Theoretical)
Performance on synthetic data with perfect ground truth:
- >80% detection accuracy for key anomalies (heavy metals, PFAS, glyphosate)
- ROC AUC >0.92 for "mastery" soil signatures
- 38ms inference on Jetson Orin Nano

**Gap:** Real-field performance is 15–30% lower than simulated due to atmospheric variation, moisture covariance, and ground truth error.

**See `field_results/METRICS.md` for honest comparison.**

#### Stage-Specific Classification
Strongest performance in identifying fully remediated ("mastery") soil states and severe contamination hotspots, enabling precise phytoremediation targeting and tokenized land-value premiums.

**Neural Activation Patterns**  
Feature map visualizations reveal characteristic spectral signatures (e.g., SWIR absorption peaks for PFAs) that the CNN isolates, exactly as predicted in the Enotrium whitepaper.

**Classification Reliability Analysis**  
High confidence on clean/remediated land and early-warning contamination alerts; calibrated predictions support zero-knowledge proof verification for supply-chain transparency.

## Model Summary 

#### Spectral Cube (x, y, λ)
          │
          ▼
#### [3D Convolution Layer]
          │
          ▼
#### [Spectral-Spatial Feature Maps]
          │
          ▼
#### [Deep CNN Blocks + Dropout]
          │
          ▼
#### [Channel Reduction]
          │
          ▼
#### [Prediction Head]
          │
          ▼
#### Soil Property Outputs


### Questions
UAV HSI data from varying flight conditions is volatile to work with, and reliance on multi-source datasets without perfectly standardized calibration certainly doesn’t help. Nonetheless the strong empirical performance in simulation suggests a strong correlation worth scaling, indicating that integrating hyperspectral reflectance data into OrpheusAI provides essential insights into soil biology and chemistry for regenerative economics.

Future work should expand beyond single-drone HSI to incorporate the full spectrum of ground-truth signals identified in the whitepaper (portable XRF, soil microbiome sequencing, IoT sensor arrays). This multi-modal approach, combined with the CNN backbone and zero-knowledge proofs, could enable Enotrium’s platform to deliver privacy-preserving, verifiable land valuation and supply-chain transparency at continental scale — turning degraded soil into a self-generating, data-rich economic asset.











## Data Appendix 1: Recommended Public Hyperspectral Soil Datasets

The following datasets are ideal starting points for testing the preprocessing pipeline, training the 3D CNN, and validating soil-health / contaminant-detection results. All are provided in native hyperspectral formats (ENVI .hdr + binary, multi-band GeoTIFF, or sensor-native cubes) and are explicitly soil-focused.

### 1. Munsell Soil Color Chart Hyperspectral Dataset (Best starting point — pure soil color reference)
**Description**: Hyperspectral images (full scenes + cropped 20×20 voxel chips) of the Munsell Soil Color Chart, captured with a SPECIM IQ camera. 204 bands (397–1003 nm reflectance). Ideal for soil classification, color analysis, and calibration benchmarks. Includes endmember spectral libraries.

**Format**: Native ENVI standard (.hdr header + binary data file).

**Sizes**:
- `chips.zip` (~68 MB — perfect for quick testing and repo inclusion)
- `whole.zip` (~2.1 GB — use Git LFS or host separately)
- `endmembers.zip` (~328 KB)

**Download**: Direct from Zenodo — [whole.zip](https://zenodo.org/records/8143355/files/whole.zip), [chips.zip](https://zenodo.org/records/8143355/files/chips.zip), [endmembers.zip](https://zenodo.org/records/8143355/files/endmembers.zip).  
**DOI**: 10.5281/zenodo.8143355 (CC license — please cite in your repo).

**Why it fits**: Explicitly soil-related, small-to-medium files, ready-to-use ENVI cubes.

### 2. Database of Hyperspectral Images of Phosphorus in Soil (Lab-based soil sample cubes)
**Description**: 152 prepared soil samples with hyperspectral cubes for total phosphorus quantification. VIS-NIR range (420–1000 nm, 145 bands). Push-broom sensor data (Bayspec OCIF Series) as full image cubes per sample.

**Format**: Hyperspectral cubes (organized as per-sample image sets within ZIPs). Native cube format from the sensor.

**Size**: Multiple ZIPs (~200 MB each, total ~3 GB). Includes chemical properties XLSX (metadata only).

**Download**: Mendeley Data → [Download All (Version 3)](https://data.mendeley.com/datasets/...) (free).

**Why it fits**: Direct lab imaging of real soil samples — core soil chemistry application.

### 3. Indian Pines AVIRIS Dataset (Site 3) (Field-scale soil/agriculture imaging)
**Description**: Classic 220-band AVIRIS airborne hyperspectral scene (June 1992) over Purdue Agronomy farm fields (explicitly for soils research, residue cover, agriculture/soil mapping). ~2-mile × 2-mile area at ~20 m resolution.

**Format**: Multi-band GeoTIFF (.tif) stacks + supporting files.

**Download**: Purdue University Research Repository (PURR) → `19920612_AVIRIS_IndianPine_Site3.tif` and related flight-line TIFFs. Includes reference PDFs/txt for calibration.

**Why it fits**: Real remote-sensing hyperspectral cube over soil-dominated agricultural terrain (bare soil + residue). Widely used but available in native TIFF.

### Additional Strong Option: HYPERVIEW2 Patches
**Description**: Compact airborne hyperspectral patches (150 bands, ~462–938 nm, ~3.2 nm resolution) captured over Polish agricultural fields with a HySpex VS-725 sensor. Each patch is paired with in-situ lab analysis (K₂O, P₂O₅, Mg, pH).

**Size**: ~312 MB total.  
**Format note**: Public version is .npz + .csv; we recommend converting patches to native ENVI .hdr + binary or multi-band GeoTIFF before inclusion (preserves exact data while staying tool-compatible with QGIS, GDAL, rasterio, spectral.py, etc.).

**Edge**: Lightweight, real field data directly tied to measurable soil parameters — excellent for lightweight repo examples.

---

**Not recommended**: Karlsruhe soil moisture dataset (only tabular point spectra, no spatial image cubes).

Other soil-specific ENVI libraries can be exported from SPECCHIO (specchio.ch) or found by searching Zenodo/Mendeley for “soil hyperspectral ENVI”.


https://oar.icrisat.org/13212/

https://www.sciencedirect.com/science/article/pii/S2949919425000305
