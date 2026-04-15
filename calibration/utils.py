"""
Utility functions for hyperspectral calibration pipeline.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import warnings


def validate_calibration_data(data: np.ndarray, wavelengths: np.ndarray) -> Dict:
    """
    Validate calibrated hyperspectral data.
    
    Args:
        data: Calibrated hyperspectral cube (H x W x C)
        wavelengths: Wavelength array
        
    Returns:
        Validation results dictionary
    """
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'statistics': {}
    }
    
    # Check data range
    data_min, data_max = np.min(data), np.max(data)
    validation_results['statistics']['data_range'] = [float(data_min), float(data_max)]
    
    if data_min < 0:
        validation_results['warnings'].append(f"Negative values found: min = {data_min}")
        validation_results['is_valid'] = False
    
    if data_max > 1.1:  # Allow small margin above 1.0
        validation_results['warnings'].append(f"Values > 1.0 found: max = {data_max}")
    
    # Check for NaN or infinite values
    nan_count = np.sum(np.isnan(data))
    inf_count = np.sum(np.isinf(data))
    
    if nan_count > 0:
        validation_results['errors'].append(f"NaN values found: {nan_count}")
        validation_results['is_valid'] = False
    
    if inf_count > 0:
        validation_results['errors'].append(f"Infinite values found: {inf_count}")
        validation_results['is_valid'] = False
    
    # Check spectral consistency
    if len(data.shape) == 3:
        mean_spectrum = np.mean(data, axis=(0, 1))
        validation_results['statistics']['mean_spectrum'] = mean_spectrum.tolist()
        
        # Check for spectral anomalies
        spectral_gradient = np.gradient(mean_spectrum)
        high_gradient_count = np.sum(np.abs(spectral_gradient) > 0.1)
        
        if high_gradient_count > len(wavelengths) * 0.1:  # More than 10% of bands
            validation_results['warnings'].append(f"High spectral gradients detected: {high_gradient_count} bands")
    
    # Check spatial consistency
    if len(data.shape) == 3:
        spatial_variance = np.var(data, axis=2)
        validation_results['statistics']['spatial_variance'] = {
            'mean': float(np.mean(spatial_variance)),
            'std': float(np.std(spatial_variance))
        }
    
    return validation_results


def calculate_calibration_metrics(original_data: np.ndarray, 
                                 calibrated_data: np.ndarray,
                                 wavelengths: np.ndarray) -> Dict:
    """
    Calculate quality metrics for calibration assessment.
    
    Args:
        original_data: Original uncalibrated data
        calibrated_data: Calibrated data
        wavelengths: Wavelength array
        
    Returns:
        Quality metrics dictionary
    """
    metrics = {
        'spectral_metrics': {},
        'spatial_metrics': {},
        'statistical_metrics': {}
    }
    
    # Spectral metrics
    if len(original_data.shape) == 3 and len(calibrated_data.shape) == 3:
        original_mean = np.mean(original_data, axis=(0, 1))
        calibrated_mean = np.mean(calibrated_data, axis=(0, 1))
        
        # Spectral correlation
        spectral_correlation = np.corrcoef(original_mean, calibrated_mean)[0, 1]
        metrics['spectral_metrics']['spectral_correlation'] = float(spectral_correlation)
        
        # Spectral angle mapper
        spectral_angle = np.arccos(np.clip(
            np.sum(original_mean * calibrated_mean) / 
            (np.linalg.norm(original_mean) * np.linalg.norm(calibrated_mean) + 1e-10),
            -1, 1
        ))
        metrics['spectral_metrics']['spectral_angle_degrees'] = float(np.degrees(spectral_angle))
        
        # Band-wise statistics
        metrics['spectral_metrics']['band_statistics'] = {}
        for i, wavelength in enumerate(wavelengths):
            orig_band = original_data[:, :, i].flatten()
            calib_band = calibrated_data[:, :, i].flatten()
            
            metrics['spectral_metrics']['band_statistics'][f'band_{wavelength}_nm'] = {
                'original_mean': float(np.mean(orig_band)),
                'calibrated_mean': float(np.mean(calib_band)),
                'original_std': float(np.std(orig_band)),
                'calibrated_std': float(np.std(calib_band)),
                'correlation': float(np.corrcoef(orig_band, calib_band)[0, 1])
            }
    
    # Spatial metrics
    if len(calibrated_data.shape) == 3:
        # Calculate spatial uniformity metrics
        for band_idx in range(min(5, calibrated_data.shape[2])):  # Check first 5 bands
            band_data = calibrated_data[:, :, band_idx]
            
            # Spatial entropy (measure of information content)
            hist, _ = np.histogram(band_data.flatten(), bins=50, range=(0, 1))
            hist = hist / np.sum(hist)  # Normalize
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            
            # Spatial contrast (standard deviation)
            contrast = np.std(band_data)
            
            wavelength = wavelengths[band_idx]
            metrics['spatial_metrics'][f'band_{wavelength}_nm'] = {
                'entropy': float(entropy),
                'contrast': float(contrast)
            }
    
    # Statistical metrics
    metrics['statistical_metrics'] = {
        'original_data': {
            'mean': float(np.mean(original_data)),
            'std': float(np.std(original_data)),
            'min': float(np.min(original_data)),
            'max': float(np.max(original_data))
        },
        'calibrated_data': {
            'mean': float(np.mean(calibrated_data)),
            'std': float(np.std(calibrated_data)),
            'min': float(np.min(calibrated_data)),
            'max': float(np.max(calibrated_data))
        }
    }
    
    # Signal-to-noise ratio estimation
    if len(calibrated_data.shape) == 3:
        # Simple SNR estimation using spatial variance
        signal = np.mean(calibrated_data, axis=(0, 1))
        noise = np.std(calibrated_data, axis=(0, 1))
        snr = signal / (noise + 1e-10)
        
        metrics['statistical_metrics']['snr'] = {
            'mean_snr': float(np.mean(snr)),
            'min_snr': float(np.min(snr)),
            'max_snr': float(np.max(snr)),
            'snr_per_band': snr.tolist()
        }
    
    return metrics


def estimate_solar_irradiance(wavelengths: np.ndarray, 
                            atmospheric_model: str = "clear_sky") -> np.ndarray:
    """
    Estimate solar irradiance spectrum for given wavelengths.
    
    Args:
        wavelengths: Wavelength array in nm
        atmospheric_model: Atmospheric model type
        
    Returns:
        Solar irradiance spectrum
    """
    # Simplified solar irradiance model
    # In practice, this should use measured data or radiative transfer models
    
    if atmospheric_model == "clear_sky":
        # Simple clear sky model
        # Peak around 550nm (green light)
        peak_wavelength = 550
        peak_irradiance = 1.8  # W/m²/nm at Earth's surface
        
        # Gaussian-like spectrum
        irradiance = peak_irradiance * np.exp(-((wavelengths - peak_wavelength) / 200) ** 2)
        
        # Add atmospheric absorption features
        # Water vapor absorption bands
        water_bands = [940, 1140, 1380, 1875]
        for band_center in water_bands:
            band_idx = np.argmin(np.abs(wavelengths - band_center))
            if np.abs(wavelengths[band_idx] - band_center) < 20:
                irradiance[band_idx] *= 0.3  # Reduce irradiance at absorption bands
        
    else:
        # Default simple model
        irradiance = np.ones_like(wavelengths) * 1.5
    
    return irradiance


def create_synthetic_reference_targets(wavelengths: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Create synthetic reference target spectra for testing.
    
    Args:
        wavelengths: Wavelength array in nm
        
    Returns:
        Dictionary of synthetic reference spectra
    """
    targets = {}
    
    # White reference (high, flat reflectance)
    targets['white_reference'] = np.ones_like(wavelengths) * 0.95
    
    # Gray reference (medium, flat reflectance)
    targets['gray_reference'] = np.ones_like(wavelengths) * 0.5
    
    # Black reference (low, flat reflectance)
    targets['black_reference'] = np.ones_like(wavelengths) * 0.05
    
    # Vegetation spectrum (green peak, red edge)
    vegetation = np.zeros_like(wavelengths)
    green_mask = (wavelengths >= 500) & (wavelengths <= 600)
    red_edge_mask = (wavelengths >= 700) & (wavelengths <= 750)
    vegetation[green_mask] = 0.3 + 0.4 * np.exp(-((wavelengths[green_mask] - 550) / 30) ** 2)
    vegetation[red_edge_mask] = 0.1 + 0.3 * (wavelengths[red_edge_mask] - 700) / 50
    vegetation[~(green_mask | red_edge_mask)] = 0.1
    targets['vegetation'] = vegetation
    
    # Soil spectrum (increasing with wavelength)
    soil = 0.1 + 0.3 * (wavelengths - wavelengths[0]) / (wavelengths[-1] - wavelengths[0])
    targets['soil'] = soil
    
    # Water spectrum (low in visible, higher in NIR)
    water = np.zeros_like(wavelengths)
    visible_mask = wavelengths < 700
    nir_mask = wavelengths >= 700
    water[visible_mask] = 0.05
    water[nir_mask] = 0.02 + 0.08 * (wavelengths[nir_mask] - 700) / 300
    targets['water'] = water
    
    return targets


def interpolate_spectrum(spectrum: np.ndarray, 
                         original_wavelengths: np.ndarray,
                         target_wavelengths: np.ndarray) -> np.ndarray:
    """
    Interpolate spectrum to target wavelengths.
    
    Args:
        spectrum: Original spectrum
        original_wavelengths: Original wavelength array
        target_wavelengths: Target wavelength array
        
    Returns:
        Interpolated spectrum
    """
    # Use linear interpolation for simplicity
    # Could use cubic spline for smoother results
    return np.interp(target_wavelengths, original_wavelengths, spectrum)


def detect_spectral_anomalies(data: np.ndarray, 
                            wavelengths: np.ndarray,
                            threshold: float = 3.0) -> Dict:
    """
    Detect spectral anomalies in hyperspectral data.
    
    Args:
        data: Hyperspectral cube (H x W x C)
        wavelengths: Wavelength array
        threshold: Threshold for anomaly detection (standard deviations)
        
    Returns:
        Anomaly detection results
    """
    if len(data.shape) != 3:
        raise ValueError("Expected 3D data (H x W x C)")
    
    # Calculate mean spectrum and deviations
    mean_spectrum = np.mean(data, axis=(0, 1))
    std_spectrum = np.std(data, axis=(0, 1))
    
    # Detect anomalous pixels
    anomaly_mask = np.zeros((data.shape[0], data.shape[1]), dtype=bool)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            pixel_spectrum = data[i, j, :]
            z_scores = np.abs((pixel_spectrum - mean_spectrum) / (std_spectrum + 1e-10))
            
            # Flag pixel if any band exceeds threshold
            if np.any(z_scores > threshold):
                anomaly_mask[i, j] = True
    
    # Calculate anomaly statistics
    anomaly_count = np.sum(anomaly_mask)
    total_pixels = data.shape[0] * data.shape[1]
    anomaly_percentage = (anomaly_count / total_pixels) * 100
    
    return {
        'anomaly_mask': anomaly_mask,
        'anomaly_count': int(anomaly_count),
        'anomaly_percentage': float(anomaly_percentage),
        'mean_spectrum': mean_spectrum,
        'std_spectrum': std_spectrum
    }


def calculate_spectral_indices(data: np.ndarray, 
                             wavelengths: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate common spectral indices for vegetation analysis.
    
    Args:
        data: Hyperspectral cube (H x W x C)
        wavelengths: Wavelength array in nm
        
    Returns:
        Dictionary of spectral indices
    """
    indices = {}
    
    # Find band indices for common wavelengths
    def find_band(wavelength):
        return np.argmin(np.abs(wavelengths - wavelength))
    
    # NDVI (Normalized Difference Vegetation Index)
    try:
        red_band = find_band(670)  # Red
        nir_band = find_band(800)  # Near-infrared
        
        if red_band < data.shape[2] and nir_band < data.shape[2]:
            red = data[:, :, red_band]
            nir = data[:, :, nir_band]
            ndvi = (nir - red) / (nir + red + 1e-10)
            indices['NDVI'] = ndvi
    except Exception:
        pass
    
    # NDRE (Normalized Difference Red Edge Index)
    try:
        red_edge_band = find_band(720)  # Red edge
        nir_band = find_band(800)  # Near-infrared
        
        if red_edge_band < data.shape[2] and nir_band < data.shape[2]:
            red_edge = data[:, :, red_edge_band]
            nir = data[:, :, nir_band]
            ndre = (nir - red_edge) / (nir + red_edge + 1e-10)
            indices['NDRE'] = ndre
    except Exception:
        pass
    
    # WDRVI (Wide Dynamic Range Vegetation Index)
    try:
        red_band = find_band(670)
        nir_band = find_band(800)
        
        if red_band < data.shape[2] and nir_band < data.shape[2]:
            red = data[:, :, red_band]
            nir = data[:, :, nir_band]
            alpha = 0.2  # Expansion factor
            wdrvi = (alpha * nir - red) / (alpha * nir + red + 1e-10)
            indices['WDRVI'] = wdrvi
    except Exception:
        pass
    
    return indices
