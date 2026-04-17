import os
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom, map_coordinates, gaussian_filter
from pathlib import Path
from typing import List, Tuple, Optional, Union, Any, Dict
import warnings

# Import calibration module
try:
    from calibration import CalibrationPipeline, CalibrationConfig
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    warnings.warn("Calibration module not available. Install calibration dependencies for radiometric correction.")


# ── Augmentation pipeline ─────────────────────────────────────────────────────

class HyperspectralTransform:
    """
    Advanced augmentation pipeline for hyperspectral cubes shaped (Bands, H, W).

    Design decisions vs. original
    ──────────────────────────────
    FIX 1 — masking order:
        Spectral and spatial masking are now applied BEFORE band-wise z-score
        normalization (see HyperspectralSoilDataset.__getitem__).  This class
        therefore receives a raw (un-normalised) cube so that zeroing masked
        regions genuinely means "no reflectance signal", not an arbitrary
        post-normalisation artefact.  The fill value used is 0 (raw DN / raw
        reflectance ~0), which is semantically correct at this stage.

    FIX 2 — no albumentations:
        albumentations was designed for uint8 RGB images.  With 200-band
        float32 cubes several internals break silently (dtype coercion,
        channel-count assumptions, ElasticTransform's cv2 backend).  All
        spatial augmentations are re-implemented here using only numpy /
        scipy so behaviour is explicit and band-count agnostic.

    ENHANCEMENT 3 — advanced augmentations:
        Added spectral augmentations and noise injection for improved
        robustness and generalization.

        Implemented transforms:
            • Random horizontal / vertical flip
            • Random crop + resize  (replaces RandomResizedCrop)
            • Elastic deformation   (replaces A.ElasticTransform)
              — uses scipy.ndimage.map_coordinates directly, exactly as
                suggested in the original code review.
            • Spectral augmentation (band-wise noise, wavelength shift)
            • Gaussian noise and Poisson noise injection
            • Random brightness and contrast adjustment
            
    NOTE: Mixup/CutMix are not implemented at the transform level because
    they require a second sample. Use a custom collate_fn in DataLoader
    if these augmentations are needed.
    """

    def __init__(
        self,
        target_size: tuple = (64, 64),
        # spectral masking
        spec_mask_prob: float = 0.6,
        spec_mask_max_width: int = 20,
        # spatial masking
        spat_mask_prob: float = 0.4,
        spat_mask_max_width: int = 15,
        # augmentation probabilities
        flip_prob: float = 0.5,
        elastic_prob: float = 0.3,
        crop_resize_prob: float = 0.8,
        # advanced augmentation parameters
        spectral_noise_prob: float = 0.2,
        spectral_shift_range: float = 10.0,  # nm shift (number of bands)
        brightness_prob: float = 0.3,
        contrast_prob: float = 0.3,
        noise_std: float = 0.02,
        poisson_lambda: float = 0.1,
        crop_scale: tuple = (0.8, 1.0),
        elastic_alpha: float = 1.0,
        elastic_sigma: float = 50.0,
        # clipping control (disable for raw reflectance > 1.0, e.g., SWIR bands)
        clip_range: Optional[Tuple[float, float]] = None,
    ):
        self.target_size = target_size
        self.spec_mask_prob = spec_mask_prob
        self.spec_mask_max_width = spec_mask_max_width
        self.spat_mask_prob = spat_mask_prob
        self.spat_mask_min = 1
        self.spat_mask_max = spat_mask_max_width
        self.flip_prob = flip_prob
        self.crop_prob = crop_resize_prob
        self.crop_scale = crop_scale
        self.elastic_prob = elastic_prob
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.spectral_noise_prob = spectral_noise_prob
        self.spectral_shift_range = spectral_shift_range
        self.brightness_prob = brightness_prob
        self.contrast_prob = contrast_prob
        self.noise_std = noise_std
        self.poisson_lambda = poisson_lambda
        self.clip_range = clip_range

    # ── masking (called on RAW cube, before normalisation) ────────────────────

    def spectral_mask(self, cube: np.ndarray) -> np.ndarray:
        """Zero out 1–spec_mask_max_width contiguous bands."""
        if np.random.rand() < self.spec_mask_prob:
            n_bands = cube.shape[0]
            width = np.random.randint(1, self.spec_mask_max_width + 1)
            start = np.random.randint(0, max(1, n_bands - width))
            cube[start: start + width] = 0.0
        return cube

    def spatial_mask(self, cube: np.ndarray) -> np.ndarray:
        """Zero out a random rectangular patch across all bands."""
        if np.random.rand() < self.spat_mask_prob:
            H, W = cube.shape[1], cube.shape[2]
            mask_h = np.random.randint(self.spat_mask_min, self.spat_mask_max + 1)
            mask_w = np.random.randint(self.spat_mask_min, self.spat_mask_max + 1)
            x1 = np.random.randint(0, max(1, H - mask_h))
            y1 = np.random.randint(0, max(1, W - mask_w))
            cube[:, x1: x1 + mask_h, y1: y1 + mask_w] = 0.0
        return cube

    # ── spatial augmentations (band-agnostic numpy/scipy) ─────────────────────

    def random_flip(self, cube: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.flip_prob:
            cube = cube[:, ::-1, :].copy()   # vertical
        if np.random.rand() < self.flip_prob:
            cube = cube[:, :, ::-1].copy()   # horizontal
        return cube

    def random_crop_resize(self, cube: np.ndarray) -> np.ndarray:
        """
        Randomly crop a sub-window then zoom back to target_size.
        Equivalent to albumentations RandomResizedCrop but works for any
        number of bands via scipy.ndimage.zoom.
        """
        if np.random.rand() >= self.crop_prob:
            # Just resize to target if no crop this iteration
            H, W = cube.shape[1], cube.shape[2]
            tH, tW = self.target_size
            if (H, W) != (tH, tW):
                cube = zoom(cube, (1, tH / H, tW / W), order=1)
            return cube

        H, W = cube.shape[1], cube.shape[2]
        tH, tW = self.target_size

        scale = np.random.uniform(*self.crop_scale)
        crop_h = max(1, int(H * scale))
        crop_w = max(1, int(W * scale))

        x0 = np.random.randint(0, max(1, H - crop_h + 1))
        y0 = np.random.randint(0, max(1, W - crop_w + 1))

        cropped = cube[:, x0: x0 + crop_h, y0: y0 + crop_w]
        # Resize back to target
        cube = zoom(cropped, (1, tH / crop_h, tW / crop_w), order=1)
        return cube

    def elastic_deform(self, cube: np.ndarray) -> np.ndarray:
        """
        FIX 2 — elastic deformation implemented directly with
        scipy.ndimage.map_coordinates.

        A single displacement field is computed for the H×W grid and then
        applied identically to every spectral band so spatial coherence
        across bands is preserved.
        """
        if np.random.rand() >= self.elastic_prob:
            return cube

        B, H, W = cube.shape

        # Random displacement fields, smoothed to look locally coherent
        dx = gaussian_filter((np.random.rand(H, W) - 0.5) * 2 * self.elastic_alpha,
                              sigma=self.elastic_sigma)
        dy = gaussian_filter((np.random.rand(H, W) - 0.5) * 2 * self.elastic_alpha,
                              sigma=self.elastic_sigma)

        # Build coordinate grids
        grid_y, grid_x = np.mgrid[0:H, 0:W]
        coords_y = grid_y + dy
        coords_x = grid_x + dx

        # Apply to each band separately
        deformed = np.empty_like(cube)
        for b in range(B):
            deformed[b] = map_coordinates(
                cube[b], [coords_y, coords_x], order=1, mode='nearest'
            )

        return deformed
    
    def spectral_augmentation(self, cube: np.ndarray) -> np.ndarray:
        """Advanced spectral augmentation techniques."""
        augmented = cube.copy()
        C, H, W = cube.shape

        # Band-wise noise injection
        if np.random.rand() < self.spectral_noise_prob:
            noise = np.random.normal(0, self.noise_std, (C, 1, 1))
            augmented += noise

        # Wavelength shift simulation (sensor calibration drift)
        # Rolls along band axis by swapping bands, not spatial rolling
        if np.random.rand() < self.spectral_noise_prob:
            shift = int(np.random.uniform(-self.spectral_shift_range, self.spectral_shift_range))
            if shift != 0:
                augmented = np.roll(augmented, shift, axis=0)

        return augmented
    
    def mixup_augmentation(self, cube1: np.ndarray, cube2: np.ndarray, alpha: float = None) -> np.ndarray:
        """Mixup augmentation for regularization."""
        if alpha is None:
            alpha = np.random.uniform(0.2, 0.8)
        
        return alpha * cube1 + (1 - alpha) * cube2
    
    def cutmix_augmentation(self, cube1: np.ndarray, cube2: np.ndarray, lambda_val: float = None) -> np.ndarray:
        """CutMix augmentation for improved robustness."""
        if lambda_val is None:
            lambda_val = np.random.uniform(0.3, 0.7)

        C, H, W = cube1.shape
        # Random bounding box center and size
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        cut_w = int(W * np.sqrt(1 - lambda_val))
        cut_h = int(H * np.sqrt(1 - lambda_val))

        # Bounding box coordinates
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply CutMix: paste region from cube2 into cube1
        mixed = cube1.copy()
        mixed[:, y1:y2, x1:x2] = cube2[:, y1:y2, x1:x2]

        return mixed
    
    def brightness_contrast_augmentation(self, cube: np.ndarray) -> np.ndarray:
        """Random brightness and contrast adjustment."""
        if np.random.rand() < self.brightness_prob:
            # Brightness adjustment
            brightness_factor = np.random.uniform(0.7, 1.3)
            cube = cube * brightness_factor
        
        if np.random.rand() < self.contrast_prob:
            # Contrast adjustment
            contrast_factor = np.random.uniform(0.8, 1.5)
            mean_val = np.mean(cube, axis=(1, 2), keepdims=True)
            cube = (cube - mean_val) * contrast_factor + mean_val
        
        if self.clip_range is not None:
            cube = np.clip(cube, self.clip_range[0], self.clip_range[1])
        return cube
    
    def noise_injection(self, cube: np.ndarray) -> np.ndarray:
        """Advanced noise injection (Gaussian + Poisson)."""
        noisy = cube.copy()
        
        # Gaussian noise
        if np.random.rand() < 0.5:  # 50% chance
            gaussian_noise = np.random.normal(0, self.noise_std, cube.shape)
            noisy += gaussian_noise
        
        # Poisson noise (sensor shot noise)
        if np.random.rand() < 0.3:  # 30% chance
            # Ensure positive values for Poisson
            positive_cube = np.maximum(cube, 1e-8)
            poisson_noise = np.random.poisson(positive_cube * self.poisson_lambda) / self.poisson_lambda - positive_cube
            noisy += poisson_noise
        
        if self.clip_range is not None:
            noisy = np.clip(noisy, self.clip_range[0], self.clip_range[1])
        return noisy

    # ── full pipeline (operates on RAW cube) ──────────────────────────────────

    def __call__(self, cube: np.ndarray) -> np.ndarray:
        """
        Apply comprehensive augmentation pipeline with advanced techniques.
        Normalization is intentionally NOT performed here — it happens after
        this call in __getitem__ so masked zeros stay semantically meaningful.
        """
        # FIX 1 — mask first, normalise later
        cube = self.spectral_mask(cube)
        cube = self.spatial_mask(cube)
        
        # Spatial augmentations (band-count agnostic, no albumentations)
        cube = self.random_flip(cube)
        cube = self.random_crop_resize(cube)   # also handles final resize
        cube = self.elastic_deform(cube)
        
        # Advanced augmentations
        cube = self.spectral_augmentation(cube)

        # Brightness/contrast adjustment
        cube = self.brightness_contrast_augmentation(cube)
        
        # Noise injection
        cube = self.noise_injection(cube)
        
        return cube


# ── Dataset ───────────────────────────────────────────────────────────────────

class HyperspectralSoilDataset(Dataset):
    """
    PyTorch Dataset for hyperspectral soil cubes stored as .npy files.

    Args:
        data_paths      : list of paths to (Bands, H, W) float32 .npy cubes.
        labels          : list of (health_class: int, contam_vector: list[float]).
        num_bands       : target number of spectral bands (resampled if needed).
        target_size     : (H, W) spatial size every cube is resized to.
        train           : if True, applies HyperspectralTransform augmentation.

    Normalisation note (FIX 1 / FIX 3)
    ────────────────────────────────────
    Band-wise z-score normalization is computed per-sample on the POST-MASKED
    cube.  Per-sample normalisation is correct for HSI because illumination
    and sensor gain vary per acquisition; dataset-level statistics would leak
    train-set information into the validation set and are inappropriate here.

    Order of operations in __getitem__:
        1. Load raw cube
        2. Resample bands / spatial size if needed
        3. Apply augmentation (masking + spatial transforms) — RAW values
        4. Band-wise z-score normalisation
        5. Convert to tensor
    """

    def __init__(
        self,
        data_paths: list,
        labels: list,
        num_bands: int = 200,
        target_size: tuple = (64, 64),
        train: bool = True,
        calibrate: bool = False,
        calib_config: Optional[str] = None,
        wavelengths: Optional[np.ndarray] = None,
        validate_data: bool = True,
        cache_data: bool = False,
        quality_threshold: float = 0.1,
    ):
        # Comprehensive input validation
        self._validate_inputs(data_paths, labels, num_bands, target_size)
        
        self.data_paths = data_paths
        self.labels = labels
        self.num_bands = num_bands
        self.target_size = target_size
        self.transform = HyperspectralTransform(target_size) if train else None
        self.validate_data = validate_data
        self.cache_data = cache_data
        self.quality_threshold = quality_threshold
        
        # Data quality tracking
        # WARNING: These are mutated in __getitem__ which runs in DataLoader workers.
        # With num_workers > 1, worker processes have separate memory - mutations
        # never propagate back to the main process. For accurate quality reports,
        # run validation in a separate pre-pass before training instead.
        self.data_stats = {}
        self.corrupted_files = []
        self.processed_files = 0
        
        # Initialize calibration pipeline
        self.calibrate = calibrate and CALIBRATION_AVAILABLE
        self.calibration_pipeline = None
        
        if self.calibrate:
            if calib_config is None:
                # Use default calibration config
                calib_config = "calibration/configs/default.yaml"
            
            if wavelengths is None:
                # Default wavelength range (400-1000nm, 3nm steps for 200 bands)
                wavelengths = np.linspace(400, 1000, num_bands)
            
            try:
                config = CalibrationConfig.from_yaml(calib_config)
                self.calibration_pipeline = CalibrationPipeline(config, wavelengths)
                self.wavelengths = wavelengths
                print(f"✅ Calibration pipeline initialized with {len(wavelengths)} wavelengths")
            except Exception as e:
                print(f"❌ Failed to initialize calibration pipeline: {e}")
                self.calibrate = False
        
        # Initialize data cache if enabled
        self.data_cache = {} if cache_data else None
        
        print(f"📊 Dataset initialized: {len(data_paths)} samples, {num_bands} bands, target_size={target_size}")
        if self.calibrate:
            print("🔧 Radiometric calibration enabled")
        if cache_data:
            print("💾 Data caching enabled")

    def _validate_inputs(self, data_paths: list, labels: list, num_bands: int, target_size: tuple) -> None:
        """Comprehensive input validation for dataset initialization."""
        # Validate data paths
        if not data_paths or len(data_paths) == 0:
            raise ValueError("data_paths cannot be empty")
        
        # Validate labels
        if not labels or len(labels) == 0:
            raise ValueError("labels cannot be empty")
        
        # Validate length match
        if len(data_paths) != len(labels):
            raise ValueError(f"data_paths and labels length mismatch: {len(data_paths)} vs {len(labels)}")
        
        # Validate num_bands
        if num_bands <= 0:
            raise ValueError(f"num_bands must be positive, got {num_bands}")
        
        # Validate target_size
        if (not isinstance(target_size, tuple) or len(target_size) != 2 or 
            any(t <= 0 for t in target_size)):
            raise ValueError(f"target_size must be tuple of 2 positive integers, got {target_size}")
        
        # Validate file existence (first few files)
        missing_files = []
        for i, path in enumerate(data_paths[:5]):  # Check first 5 files
            if not os.path.exists(path):
                missing_files.append(path)
        
        if missing_files:
            print(f"⚠️  Warning: {len(missing_files)} data files not found:")
            for file in missing_files:
                print(f"  - {file}")
    
    def _validate_data_quality(self, cube: np.ndarray, idx: int) -> bool:
        """Validate data quality and track statistics."""
        # Check for NaN or infinite values
        if np.any(np.isnan(cube)) or np.any(np.isinf(cube)):
            self.corrupted_files.append(idx)
            print(f"❌ Sample {idx}: Contains NaN or infinite values")
            return False
        
        # Check data range
        data_min, data_max = np.min(cube), np.max(cube)
        if data_max == data_min:  # Flat spectrum - unusable for training
            print(f"❌ Sample {idx}: Flat spectrum (min={data_min:.3f}, max={data_max:.3f}) - rejecting")
            self.corrupted_files.append(idx)
            return False
        
        # Check signal-to-noise ratio (simple estimate)
        signal_power = np.mean(cube ** 2)
        noise_estimate = np.var(cube, axis=(1, 2)).mean()
        if noise_estimate > 0:
            snr_db = 10 * np.log10(signal_power / noise_estimate)
            if snr_db < self.quality_threshold * 100:  # Convert threshold to dB scale
                print(f"⚠️  Sample {idx}: Low SNR ({snr_db:.1f} dB)")
        
        # Update statistics
        self.data_stats[idx] = {
            'min': float(data_min),
            'max': float(data_max),
            'mean': float(np.mean(cube)),
            'std': float(np.std(cube)),
            'shape': cube.shape
        }
        
        return True
    
    def _load_cached_data(self, idx: int) -> Optional[np.ndarray]:
        """Load data from cache if available."""
        if self.data_cache is not None and idx in self.data_cache:
            return self.data_cache[idx]
        return None
    
    def _cache_data(self, idx: int, data: np.ndarray, is_valid: bool = True) -> None:
        """Cache data if caching is enabled.
        
        Args:
            idx: sample index
            data: numpy array to cache
            is_valid: if False, don't cache (prevents caching zero fallbacks)
        """
        if self.data_cache is not None and is_valid:
            self.data_cache[idx] = data.copy()
    
    def get_data_quality_report(self) -> dict:
        """Generate comprehensive data quality report."""
        if not self.data_stats:
            return {"message": "No data processed yet"}
        
        # Calculate statistics
        all_means = [stats['mean'] for stats in self.data_stats.values()]
        all_stds = [stats['std'] for stats in self.data_stats.values()]
        
        report = {
            'total_samples': len(self.data_stats),
            'corrupted_files': len(self.corrupted_files),
            'processed_files': self.processed_files,
            'data_quality': {
                'mean_signal': float(np.mean(all_means)),
                'signal_variance': float(np.var(all_means)),
                'mean_noise': float(np.mean(all_stds)),
                'noise_variance': float(np.var(all_stds))
            },
            'quality_issues': []
        }
        
        # Identify quality issues
        if len(self.corrupted_files) > 0:
            report['quality_issues'].append(f"{len(self.corrupted_files)} corrupted files")
        
        if report['data_quality']['mean_noise'] > 0.5:
            report['quality_issues'].append("High noise levels detected")
        
        return report

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int):
        # Step 1: Check cache first
        cached_data = self._load_cached_data(idx)
        if cached_data is not None:
            cube = cached_data
        else:
            # Step 2: Load raw cube with error handling
            try:
                cube = np.load(self.data_paths[idx]).astype(np.float32)  # (B, H, W)
            except Exception as e:
                raise RuntimeError(f"Failed to load data file {self.data_paths[idx]}: {e}")

            # Step 3: Validate data quality if enabled
            is_valid = True
            if self.validate_data:
                is_valid = self._validate_data_quality(cube, idx)
                if not is_valid:
                    # Use fallback for corrupted files (consistent shape)
                    if idx in self.corrupted_files:
                        # Return zeros for corrupted files (consistent shape)
                        cube = np.zeros((self.num_bands, *self.target_size), dtype=np.float32)
                    else:
                        print(f"⚠️  Using sample {idx} despite quality issues")

            # Step 4: Cache data if enabled (only cache valid data, not zero fallbacks)
            self._cache_data(idx, cube, is_valid=is_valid)
        
        # Step 5: Apply calibration if enabled (before other processing)
        if self.calibrate and self.calibration_pipeline is not None:
            try:
                # Transpose to (H, W, B) for calibration pipeline
                cube_hwb = cube.transpose(1, 2, 0)
                calibrated_cube, _ = self.calibration_pipeline.calibrate_single_cube(cube_hwb)
                # Transpose back to (B, H, W)
                cube = calibrated_cube.transpose(2, 0, 1)
            except Exception as e:
                print(f"❌ Calibration failed for sample {idx}: {e}")
                # Continue with uncalibrated data

        # Step 6: Resample to canonical shape
        # Band dimension
        if cube.shape[0] != self.num_bands:
            cube = zoom(cube, (self.num_bands / cube.shape[0], 1, 1), order=1)
        # Spatial dimensions (only if not training, where crop_resize handles it)
        if not self.transform:
            H, W = cube.shape[1], cube.shape[2]
            tH, tW = self.target_size
            if (H, W) != (tH, tW):
                cube = zoom(cube, (1, tH / H, tW / W), order=1)

        # Step 7: Augmentation on RAW cube (masking + spatial)
        if self.transform is not None:
            cube = self.transform(cube)

        # Step 8: Band-wise z-score normalisation
        mean = cube.mean(axis=(1, 2), keepdims=True)
        std  = cube.std(axis=(1, 2), keepdims=True) + 1e-8
        cube = (cube - mean) / std

        # Step 9: To tensor (1, Bands, H, W)
        cube_t = torch.from_numpy(cube).unsqueeze(0)  # add channel dim

        # Update processing statistics
        self.processed_files += 1

        health_label, contam_label = self.labels[idx]
        
        # Log data processing for monitoring
        if hasattr(self, '_log_processing'):
            self._log_processing(idx, cube.shape, health_label, contam_label)
        
        return (
            cube_t,
            torch.tensor(health_label, dtype=torch.long),
            torch.tensor(contam_label, dtype=torch.float32),
        )
    
    def _log_processing(self, idx: int, data_shape: tuple, health_label: Any, contam_label: Any):
        """Log data processing for monitoring and debugging."""
        if not hasattr(self, '_processing_log'):
            self._processing_log = []
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'sample_idx': idx,
            'data_shape': data_shape,
            'health_label': health_label,
            'contam_label': contam_label,
            'calibration_enabled': self.calibrate,
            'cache_enabled': self.cache_data is not None,
            'validation_enabled': self.validate_data
        }
        
        self._processing_log.append(log_entry)
        
        # Keep only last 1000 entries to prevent memory issues
        if len(self._processing_log) > 1000:
            self._processing_log = self._processing_log[-1000:]
    
    def get_processing_log(self) -> List[Dict]:
        """Get the processing log for debugging."""
        return getattr(self, '_processing_log', [])
    
    def clear_processing_log(self):
        """Clear the processing log."""
        if hasattr(self, '_processing_log'):
            self._processing_log = []
    
    def get_dataset_summary(self) -> Dict:
        """Get comprehensive dataset summary."""
        if not self.data_stats:
            return {"message": "No data processed yet"}
        
        # Calculate statistics from processed data
        all_means = [stats['mean'] for stats in self.data_stats.values()]
        all_stds = [stats['std'] for stats in self.data_stats.values()]
        
        return {
            'total_samples': len(self.data_stats),
            'corrupted_files': len(self.corrupted_files),
            'processed_files': self.processed_files,
            'data_quality': {
                'mean_signal': float(np.mean(all_means)),
                'signal_variance': float(np.var(all_means)),
                'mean_noise': float(np.mean(all_stds)),
                'noise_variance': float(np.var(all_stds))
            },
            'processing_options': {
                'calibration_enabled': self.calibrate,
                'cache_enabled': self.cache_data is not None,
                'validation_enabled': self.validate_data,
                'target_size': self.target_size,
                'num_bands': self.num_bands
            }
        }


# ── HYPERVIEW2 Dataset ────────────────────────────────────────────────────────

class Hyperview2Dataset(Dataset):
    """
    Dataset for HYPERVIEW2 Challenge patches.

    Expected layout:
        data_dir/
            <patch_id>.npz   — one file per patch, array shape (H, W, C) or (C, H, W)
        labels_csv           — columns: patch_id, K, Mg, P2O5, pH

    Label mapping:
        Health class (0–4) is derived from a composite soil health score built
        by scoring each parameter against agronomic optimal ranges, averaging
        across all four parameters, then percentile-binning into five classes.
        Contaminant labels are all zeros (HYPERVIEW2 has no contaminant GT).
    """

    # Agronomic optimal ranges used for per-parameter scoring
    _OPTIMA = {
        "K":    (150.0, 250.0),
        "Mg":   (100.0, 200.0),
        "P2O5": (40.0,  80.0),
        "pH":   (6.0,   7.0),
    }

    def __init__(
        self,
        data_dir: str,
        labels_csv: str,
        num_bands: int = 150,
        target_size: tuple = (64, 64),
        train: bool = True,
    ):
        import pandas as pd

        self.data_dir = Path(data_dir)
        self.num_bands = num_bands
        self.target_size = target_size
        self.transform = HyperspectralTransform(target_size) if train else None

        df = pd.read_csv(labels_csv)

        scores = self._compute_health_scores(df)
        health_classes = self._bin_scores(scores)

        self.patch_ids = df["patch_id"].tolist()
        self.health_classes = health_classes
        self.contam_labels = [[0.0, 0.0, 0.0, 0.0]] * len(df)

        print(f"📊 Hyperview2Dataset: {len(self.patch_ids)} patches, "
              f"health class distribution: {np.bincount(health_classes).tolist()}")

    def _compute_health_scores(self, df) -> np.ndarray:
        """
        Score 0–1 per parameter (1.0 = inside optimal range, decreasing
        linearly outside), then average across all four parameters.
        """
        scores = np.zeros(len(df), dtype=np.float32)
        n_params = 0
        for col, (lo, hi) in self._OPTIMA.items():
            if col not in df.columns:
                continue
            vals = df[col].to_numpy(dtype=np.float32)
            param_score = np.ones_like(vals)
            below = vals < lo
            above = vals > hi
            param_score[below] = vals[below] / lo
            param_score[above] = hi / np.maximum(vals[above], 1e-8)
            scores += param_score
            n_params += 1
        return scores / max(n_params, 1)

    def _bin_scores(self, scores: np.ndarray) -> List[int]:
        """Bin composite scores into 5 health classes via percentile cuts."""
        thresholds = np.percentile(scores, [20, 40, 60, 80])
        classes = np.zeros(len(scores), dtype=int)
        for i, t in enumerate(thresholds):
            classes[scores > t] = i + 1
        return classes.tolist()

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int):
        patch_path = self.data_dir / f"{self.patch_ids[idx]}.npz"
        data = np.load(patch_path)
        key = list(data.keys())[0]
        cube = data[key].astype(np.float32)

        # Ensure (C, H, W) — HYPERVIEW2 patches are typically (H, W, C)
        if cube.ndim == 3 and cube.shape[2] < cube.shape[0]:
            cube = cube.transpose(2, 0, 1)

        # Resample to target band count if needed
        if cube.shape[0] != self.num_bands:
            cube = zoom(cube, (self.num_bands / cube.shape[0], 1, 1), order=1)

        # Spatial resize (transform handles it when training)
        if self.transform is None:
            H, W = cube.shape[1], cube.shape[2]
            tH, tW = self.target_size
            if (H, W) != (tH, tW):
                cube = zoom(cube, (1, tH / H, tW / W), order=1)

        if self.transform is not None:
            cube = self.transform(cube)

        # Band-wise z-score normalisation
        mean = cube.mean(axis=(1, 2), keepdims=True)
        std  = cube.std(axis=(1, 2), keepdims=True) + 1e-8
        cube = (cube - mean) / std

        cube_t = torch.from_numpy(cube).unsqueeze(0)  # (1, C, H, W)

        return (
            cube_t,
            torch.tensor(self.health_classes[idx], dtype=torch.long),
            torch.tensor(self.contam_labels[idx], dtype=torch.float32),
        )


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, os

    # Create a tiny fake dataset
    tmp = tempfile.mkdtemp()
    paths, lbls = [], []
    for i in range(6):
        cube = np.random.rand(210, 70, 70).astype(np.float32)  # intentionally off-size
        p = os.path.join(tmp, f"cube_{i}.npy")
        np.save(p, cube)
        paths.append(p)
        lbls.append((np.random.randint(0, 5), np.random.rand(4).tolist()))

    ds_train = HyperspectralSoilDataset(paths[:4], lbls[:4], train=True)
    ds_val   = HyperspectralSoilDataset(paths[4:], lbls[4:], train=False)

    cube_t, health, contam = ds_train[0]
    print("train cube shape :", cube_t.shape)   # (1, 200, 64, 64)
    print("health label     :", health)
    print("contam label     :", contam)
    print("cube dtype       :", cube_t.dtype)
    print("cube mean ~0     :", cube_t.mean().item())

    cube_v, _, _ = ds_val[0]
    print("val cube shape   :", cube_v.shape)   # (1, 200, 64, 64)
