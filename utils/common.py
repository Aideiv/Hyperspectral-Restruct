"""
Common utilities shared across the Hyperspectral-Restruct codebase.

This module provides centralized functions for:
- Configuration loading and validation
- Model loading from checkpoints
- Common I/O operations
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import yaml

from configs.constants import HEALTH_LABELS, CONTAMINANT_NAMES, MODEL_DEFAULTS

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/default.yaml") -> Dict:
    """
    Load configuration from YAML file with error handling and validation.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid or cannot be parsed
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config file {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {config_path}: {e}")
    
    if config is None:
        raise ValueError(f"Config file {config_path} is empty")
    
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a dictionary, got {type(config).__name__}")
    
    return config


def validate_config(config: Dict, required_sections: Optional[list] = None) -> None:
    """
    Validate configuration dictionary structure.
    
    Args:
        config: Configuration dictionary to validate
        required_sections: List of required top-level sections
        
    Raises:
        ValueError: If required sections are missing or config is invalid
    """
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a dictionary, got {type(config).__name__}")
    
    required_sections = required_sections or ["model"]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: torch.device,
    num_bands: Optional[int] = None,
    num_classes: Optional[int] = None,
    num_contaminants: Optional[int] = None,
    model_name: Optional[str] = None,
) -> torch.nn.Module:
    """
    Load model from a checkpoint file using the model factory.
    
    Supports multiple checkpoint formats and all model variants (base, se, spectral, deep, hybrid).
    Automatically detects model variant from checkpoint config if available.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        device: Device to load the model onto
        num_bands: Number of spectral bands (uses default if None)
        num_classes: Number of health classes (uses default if None)
        num_contaminants: Number of contaminant types (uses default if None)
        model_name: Model variant to use (auto-detected from checkpoint if None)
        
    Returns:
        Loaded model in eval mode on the specified device
        
    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If checkpoint format is invalid
        RuntimeError: If model loading fails
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Set defaults from constants
    num_bands = num_bands or MODEL_DEFAULTS["num_bands"]
    num_classes = num_classes or len(HEALTH_LABELS)
    num_contaminants = num_contaminants or len(CONTAMINANT_NAMES)
    
    try:
        from model import create_model, MODEL_VARIANTS
    except ImportError as e:
        raise RuntimeError(f"Cannot import model factory: {e}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {e}")
    
    # Extract state dict and config from various checkpoint formats
    state_dict = None
    checkpoint_cfg = None
    
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            checkpoint_cfg = checkpoint.get("cfg", {})
            # Log checkpoint metadata if available
            if "best_auc" in checkpoint:
                logger.info(f"Checkpoint best AUC: {checkpoint['best_auc']:.4f}")
            if "epoch" in checkpoint:
                logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            checkpoint_cfg = checkpoint.get("cfg", {})
        else:
            # Assume dict itself is the state dict
            state_dict = checkpoint
    else:
        raise ValueError(f"Invalid checkpoint format: expected dict, got {type(checkpoint).__name__}")
    
    if state_dict is None:
        raise ValueError("No state dict found in checkpoint")
    
    # Determine model variant
    if model_name is None:
        # Try to auto-detect from checkpoint config
        if checkpoint_cfg:
            model_name = checkpoint_cfg.get("model_name", "base")
        else:
            model_name = "base"  # Default fallback
    
    if model_name not in MODEL_VARIANTS:
        logger.warning(f"Unknown model variant '{model_name}' in checkpoint, falling back to 'base'")
        model_name = "base"
    
    # Extract model-specific kwargs from checkpoint config
    model_kwargs = {
        "num_bands": num_bands,
        "num_classes": num_classes,
        "num_contaminants": num_contaminants,
        "bottleneck_dim": MODEL_DEFAULTS.get("bottleneck_dim", 512),
        "dropout_p": MODEL_DEFAULTS.get("dropout_p", 0.5),
    }
    
    if checkpoint_cfg:
        model_kwargs["bottleneck_dim"] = checkpoint_cfg.get("bottleneck_dim", model_kwargs["bottleneck_dim"])
        model_kwargs["dropout_p"] = checkpoint_cfg.get("dropout_p", model_kwargs["dropout_p"])
        
        # Add variant-specific kwargs
        if model_name in ["se", "hybrid"]:
            model_kwargs["se_reduction"] = checkpoint_cfg.get("se_reduction", 16)
        if model_name == "deep":
            model_kwargs["blocks_per_layer"] = checkpoint_cfg.get("blocks_per_layer", 3)
    
    # Create model using factory
    model = create_model(model_name, **model_kwargs)
    logger.info(f"Created {model_name} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load state dict with validation
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        # Try loading with strict=False to see if it's just key mismatch
        try:
            model.load_state_dict(state_dict, strict=False)
            logger.warning(f"Loaded checkpoint with strict=False - some keys may be missing or unexpected")
        except RuntimeError:
            raise ValueError(f"Failed to load state dict: {e}")
    
    model.to(device)
    model.eval()
    
    return model


def get_device(prefer_cpu: bool = False) -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        prefer_cpu: If True, return CPU even if CUDA is available
        
    Returns:
        torch.device object
    """
    if prefer_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to the directory
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_checkpoint_info(checkpoint_path: Union[str, Path]) -> Dict:
    """
    Get metadata from a checkpoint file without loading the full model.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        return {"error": "Checkpoint not found"}
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"error": f"Failed to load checkpoint: {e}"}
    
    info = {
        "path": str(checkpoint_path),
        "size_mb": round(checkpoint_path.stat().st_size / (1024 * 1024), 2),
    }
    
    if isinstance(checkpoint, dict):
        info["has_model_state"] = "model_state_dict" in checkpoint
        info["has_optimizer_state"] = "optimizer_state_dict" in checkpoint
        info["has_scheduler_state"] = "scheduler_state_dict" in checkpoint
        
        if "epoch" in checkpoint:
            info["epoch"] = checkpoint["epoch"]
        if "best_auc" in checkpoint:
            info["best_auc"] = checkpoint["best_auc"]
        if "cfg" in checkpoint:
            info["has_config"] = True
    
    return info
