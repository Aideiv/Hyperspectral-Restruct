"""
Centralized constants for the Hyperspectral-Restruct project.

This module contains all shared configuration values, label definitions,
and hyperparameters to ensure consistency across the codebase.
"""

from typing import Dict, List, Tuple

# =============================================================================
# Label Definitions
# =============================================================================

HEALTH_LABELS: Dict[int, str] = {
    0: "Severely Degraded",
    1: "Degraded",
    2: "Moderate",
    3: "Recovering",
    4: "Healthy / Remediated",
}

CONTAMINANT_NAMES: List[str] = ["metal", "pfas", "glyphosate", "microplastics"]

CONTAM_LABELS: List[str] = [
    "Polyethylene (PE)",
    "Polypropylene (PP)",
    "PS / PET",
    "Other Contaminant",
]

CONTAM_THRESHOLD: float = 0.5  # sigmoid threshold for binary contaminant decision

# =============================================================================
# Model Architecture Defaults
# =============================================================================

# Available model variants
MODEL_VARIANTS: List[str] = ["base", "se", "spectral", "deep", "hybrid"]

MODEL_DEFAULTS = {
    "num_bands": 200,
    "num_classes": 5,
    "num_contaminants": 4,
    "bottleneck_dim": 512,
    "dropout_p": 0.5,
    "target_size": (64, 64),
    "model_name": "base",  # Default model variant
}

# Variant-specific hyperparameter recommendations
# These adjust based on model capacity to prevent overfitting/underfitting
MODEL_VARIANT_CONFIGS = {
    "base": {
        "dropout_p": 0.5,
        "lr": 3e-4,
        "weight_decay": 0.05,
        "description": "Standard 3D CNN - balanced for most datasets",
    },
    "se": {
        "dropout_p": 0.5,
        "lr": 3e-4,
        "weight_decay": 0.05,
        "se_reduction": 16,
        "description": "SE-ResNet with channel attention - +3-5% expected accuracy",
    },
    "spectral": {
        "dropout_p": 0.5,
        "lr": 3e-4,
        "weight_decay": 0.05,
        "description": "Spectral attention - best for SWIR-dominant features",
    },
    "deep": {
        "dropout_p": 0.6,  # Higher dropout for larger model
        "lr": 2e-4,  # Lower LR for stability
        "weight_decay": 0.1,  # Stronger regularization
        "blocks_per_layer": 3,
        "description": "4-layer deep network - for large datasets (>1000 samples)",
    },
    "hybrid": {
        "dropout_p": 0.55,
        "lr": 2.5e-4,
        "weight_decay": 0.08,
        "se_reduction": 16,
        "description": "Premium architecture - all improvements combined",
    },
}

# =============================================================================
# Training Configuration Defaults
# =============================================================================

TRAINING_DEFAULTS = {
    "batch_size": 16,
    "epochs": 100,
    "lr": 3e-4,
    "weight_decay": 0.05,
    "contam_loss_weight": 0.5,
    "label_smoothing": 0.1,
    "cosine_T0": 10,
    "cosine_T_mult": 2,
    "grad_clip": 1.0,
    "num_workers": 8,
    "patience": 10,  # Early stopping patience
}

# =============================================================================
# Data Configuration
# =============================================================================

DATA_DEFAULTS = {
    "wavelength_range": (400, 2500),  # nm
    "train_val_split": 0.2,
    "random_seed": 42,
}

# =============================================================================
# Paths
# =============================================================================

DEFAULT_PATHS = {
    "save_path": "soil_3dcnn_enotrium.pth",
    "config_path": "configs/default.yaml",
    "checkpoint_dir": "checkpoints/",
    "model_dir": "models/",
}

# =============================================================================
# API Configuration
# =============================================================================

API_DEFAULTS = {
    "host": "0.0.0.0",
    "port": 8000,
    "model_path": "models/nitrogen_model.onnx",
    "title": "Hyperspectral-Restruct Nitrogen API",
    "version": "1.0",
}

# =============================================================================
# Nitrogen Prediction Bands (SWIR)
# =============================================================================

NITROGEN_SWIR_BANDS: List[int] = [1478, 1697, 2050, 2104, 2410]

# =============================================================================
# Augmentation Defaults
# =============================================================================

AUGMENTATION_DEFAULTS = {
    "target_size": (64, 64),
    "spec_mask_prob": 0.6,
    "spec_mask_max_width": 20,
    "spat_mask_prob": 0.4,
    "spat_mask_max_width": 15,
    "flip_prob": 0.5,
    "elastic_prob": 0.3,
    "crop_resize_prob": 0.8,
    "spectral_noise_prob": 0.2,
    "spectral_shift_range": 10.0,
    "brightness_prob": 0.3,
    "contrast_prob": 0.3,
    "noise_std": 0.02,
    "poisson_lambda": 0.1,
    "crop_scale": (0.8, 1.0),
    "elastic_alpha": 1.0,
    "elastic_sigma": 50.0,
}


def get_full_config() -> Dict:
    """Returns a complete configuration dictionary with all defaults."""
    return {
        **MODEL_DEFAULTS,
        **TRAINING_DEFAULTS,
        **DATA_DEFAULTS,
        **DEFAULT_PATHS,
        **API_DEFAULTS,
    }


def get_model_config(model_name: str) -> Dict:
    """
    Returns recommended configuration for a specific model variant.
    
    Args:
        model_name: One of MODEL_VARIANTS
    
    Returns:
        Dictionary with recommended hyperparameters
    """
    model_name = model_name.lower()
    if model_name not in MODEL_VARIANTS:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {MODEL_VARIANTS}")
    
    base_config = {**MODEL_DEFAULTS, **TRAINING_DEFAULTS}
    variant_config = MODEL_VARIANT_CONFIGS.get(model_name, {})
    
    # Merge variant-specific settings
    result = {**base_config, **variant_config}
    result["model_name"] = model_name
    
    return result
