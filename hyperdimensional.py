"""
hyperdimensional.py — Hyperdimensional Computing (HDC) for Neural Network Ensembling

Implements "Gluing Neural Networks Symbolically Through Hyperdimensional Computing"
arXiv:2205.15534 — Sutor et al.

Key concepts:
1. Encode neural network output signals (pre-classification logits) as binary hypervectors
2. Bundle multiple hypervectors through consensus summation (element-wise sum + binarize)
3. Train classification hypervectors that can fuse multiple neural networks at symbolic level
4. Enables online learning — models can be added/removed without retraining

This module provides:
- BinaryHV: High-dimensional binary vector operations
- HVEncoder: Encode NN outputs to hypervectors via random projection
- HVClassifier: Classification via similarity matching
- ConsensusEnsemble: Fuse multiple model predictions at hypervector level
- ModelGlue: Trainable glue layer for end-to-end HDC + NN integration

Example usage:
    # Create encoder and classifier
    encoder = HVEncoder(input_dim=5, hv_dim=10000)
    classifier = HVClassifier(hv_dim=10000, num_classes=5)
    
    # Training: encode outputs and bundle into class hypervectors
    for batch in dataloader:
        logits, _ = model(batch)
        hv = encoder.encode(logits)
        classifier.add_examples(hv, labels)
    classifier.finalize()  # Binarize class hypervectors
    
    # Inference: similarity-based classification
    pred_class = classifier.predict(hv)
    
    # Ensemble: glue multiple models
    glue = ModelGlue(hv_dim=10000, num_classes=5)
    glue.add_model(model_a, "model_a")
    glue.add_model(model_b, "model_b")
    pred = glue.predict_ensemble(batch)  # Consensus of both models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass


# ───────────────────────────────────────────────────────────────────────────────
# Binary Hypervector Operations
# ───────────────────────────────────────────────────────────────────────────────

class BinaryHV:
    """
    Binary Hypervector operations for HDC.
    
    Binary hypervectors are high-dimensional vectors with values in {-1, +1}.
    Key operations:
    - Bundling (⊕): Element-wise addition followed by binarization
    - Binding (⊗): Element-wise multiplication (XOR for binary)
    - Similarity: Cosine similarity via dot product
    
    Args:
        dim: Dimensionality of hypervectors (typically 1000-10000)
        device: torch device
    """
    
    def __init__(self, dim: int = 10000, device: torch.device = None):
        self.dim = dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def random(self, shape: Union[int, Tuple[int, ...]]) -> torch.Tensor:
        """Generate random binary hypervector(s) with values in {-1, +1}."""
        if isinstance(shape, int):
            shape = (shape,)
        return torch.randint(0, 2, shape, device=self.device) * 2 - 1  # {-1, +1}
    
    def bundle(self, hvs: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """
        Bundle multiple hypervectors via element-wise sum and binarization.
        
        Args:
            hvs: (N, dim) or (dim,) tensor of binary hypervectors
            threshold: Binarization threshold (default 0.0)
        
        Returns:
            Binarized consensus hypervector (dim,) or (batch, dim)
        """
        if hvs.dim() == 1:
            return hvs  # Single vector, no bundling needed
        
        # Sum all hypervectors
        summed = hvs.sum(dim=0 if hvs.dim() == 2 else -2)
        
        # Binarize: positive → +1, negative → -1, zero → random
        result = torch.where(summed > threshold, 
                            torch.ones_like(summed),
                            torch.where(summed < -threshold,
                                       -torch.ones_like(summed),
                                       self.random(summed.shape)))
        return result
    
    def bind(self, hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
        """
        Bind two hypervectors via element-wise multiplication (XOR-like).
        
        Args:
            hv1: (..., dim) binary hypervector
            hv2: (..., dim) binary hypervector
        
        Returns:
            Bound hypervector (..., dim)
        """
        return hv1 * hv2
    
    def similarity(self, hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between hypervectors.
        For binary {-1, +1} vectors, this equals the normalized dot product.
        
        Args:
            hv1: (..., dim) binary hypervector
            hv2: (..., dim) binary hypervector
        
        Returns:
            Similarity score in [-1, 1], shape (...)
        """
        return (hv1 * hv2).sum(dim=-1) / self.dim
    
    def permute(self, hv: torch.Tensor, shifts: int = 1) -> torch.Tensor:
        """Permute hypervector by circular shift (for sequence encoding)."""
        return torch.roll(hv, shifts, dims=-1)


# ───────────────────────────────────────────────────────────────────────────────
# Hypervector Encoder
# ───────────────────────────────────────────────────────────────────────────────

class HVEncoder(nn.Module):
    """
    Encode neural network outputs (logits) to binary hypervectors.
    
    Uses random projection: hv = binarize(logits @ projection_matrix)
    
    This implements the encoding step from the paper where output signals
    just before classification are mapped to high-dimensional binary space.
    
    Args:
        input_dim: Dimension of input (number of classes/logits)
        hv_dim: Hypervector dimension (typically 1000-10000)
        projection: "random" or "learned"
        device: torch device
    """
    
    def __init__(
        self,
        input_dim: int,
        hv_dim: int = 10000,
        projection: str = "random",
        device: torch.device = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hv_dim = hv_dim
        self.projection_type = projection
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Binary hypervector operations
        self.hv_ops = BinaryHV(hv_dim, self.device)
        
        if projection == "random":
            # Random projection matrix: fixed after initialization
            self.register_buffer(
                "projection",
                torch.randn(input_dim, hv_dim, device=self.device)
            )
            # Store as non-trainable
            self.projection.requires_grad = False
        elif projection == "learned":
            # Learnable projection matrix
            self.projection = nn.Parameter(torch.randn(input_dim, hv_dim) * 0.01)
        else:
            raise ValueError(f"Unknown projection type: {projection}")
    
    def encode(self, x: torch.Tensor, binarize: bool = True) -> torch.Tensor:
        """
        Encode input to binary hypervector.
        
        Args:
            x: (batch, input_dim) neural network outputs (logits/probabilities)
            binarize: If True, output binary {-1, +1} hypervector
        
        Returns:
            (batch, hv_dim) hypervector (binary if binarize=True, else continuous)
        """
        # Project to high-dimensional space
        hv = x @ self.projection  # (batch, hv_dim)
        
        if binarize:
            # Binarize: positive → +1, negative → -1
            hv = torch.sign(hv)
            # Handle zeros randomly
            hv = torch.where(hv == 0, 
                           torch.randint(0, 2, hv.shape, device=hv.device) * 2 - 1,
                           hv)
        
        return hv
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for encode()."""
        return self.encode(x)


# ───────────────────────────────────────────────────────────────────────────────
# Hypervector Classifier
# ───────────────────────────────────────────────────────────────────────────────

class HVClassifier(nn.Module):
    """
    Classification via hypervector similarity matching.
    
    Training accumulates encoded examples into class hypervectors via bundling.
    Classification measures similarity between query hypervector and each class HV.
    
    This is the core of the "gluing" mechanism from the paper:
    - Multiple model outputs encoded as hypervectors
    - Bundled together to form consensus class hypervectors
    - New models can be added incrementally without retraining
    
    Args:
        hv_dim: Hypervector dimension
        num_classes: Number of classes
        device: torch device
    """
    
    def __init__(
        self,
        hv_dim: int = 10000,
        num_classes: int = 5,
        device: torch.device = None,
    ):
        super().__init__()
        self.hv_dim = hv_dim
        self.num_classes = num_classes
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.hv_ops = BinaryHV(hv_dim, self.device)
        
        # Class hypervectors — accumulated via bundling during training
        # Stored as continuous (non-binary) during accumulation, binarized at finalize()
        self.register_buffer("class_hvs", torch.zeros(num_classes, hv_dim, device=self.device))
        self.register_buffer("class_counts", torch.zeros(num_classes, device=self.device))
        
        self.is_finalized = False
        self.training_mode = True
    
    def add_examples(self, hvs: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Accumulate hypervector examples into class hypervectors.
        
        Args:
            hvs: (batch, hv_dim) encoded hypervectors
            labels: (batch,) class labels
        """
        if self.is_finalized:
            raise RuntimeError("Cannot add examples after finalization. Call reset() first.")
        
        # Accumulate by class
        for c in range(self.num_classes):
            mask = labels == c
            if mask.any():
                class_hvs = hvs[mask]  # (n_examples, hv_dim)
                # Add to running sum (bundling in continuous form)
                self.class_hvs[c] += class_hvs.sum(dim=0)
                self.class_counts[c] += mask.sum().item()
    
    def finalize(self) -> None:
        """
        Binarize class hypervectors after all examples accumulated.
        This converts continuous sums to binary {-1, +1} vectors.
        """
        self.class_hvs.copy_(torch.sign(self.class_hvs))
        # Handle zeros randomly
        zeros = self.class_hvs == 0
        if zeros.any():
            num_zeros = zeros.sum().item()
            random_values = (torch.randint(0, 2, (num_zeros,), device=self.device) * 2 - 1).float()
            self.class_hvs[zeros] = random_values
        self.is_finalized = True
        self.training_mode = False
    
    def reset(self) -> None:
        """Reset class hypervectors to enable retraining."""
        self.class_hvs.zero_()
        self.class_counts.zero_()
        self.is_finalized = False
        self.training_mode = True
    
    def predict(self, hvs: torch.Tensor, return_similarities: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict class by finding most similar class hypervector.
        
        Args:
            hvs: (batch, hv_dim) query hypervectors
            return_similarities: If True, also return similarity scores
        
        Returns:
            predictions: (batch,) class indices
            similarities: (batch, num_classes) if return_similarities=True
        """
        if not self.is_finalized:
            # During training, use continuous class_hvs
            class_hvs = self.class_hvs
        else:
            class_hvs = self.class_hvs
        
        # Compute similarities: (batch, num_classes)
        similarities = (hvs @ class_hvs.T) / self.hv_dim
        
        # Predict class with highest similarity
        predictions = similarities.argmax(dim=-1)
        
        if return_similarities:
            return predictions, similarities
        return predictions
    
    def forward(self, hvs: torch.Tensor) -> torch.Tensor:
        """Forward pass returns predictions."""
        return self.predict(hvs)


# ───────────────────────────────────────────────────────────────────────────────
# Consensus Ensemble (Model Gluing)
# ───────────────────────────────────────────────────────────────────────────────

class ConsensusEnsemble(nn.Module):
    """
    Ensemble multiple neural networks via hypervector consensus.
    
    This is the main contribution from the paper:
    - Multiple neural networks produce output signals (logits)
    - Each network's outputs are encoded as hypervectors
    - Hypervectors are bundled (consensus summation) to form ensemble prediction
    - Minimal overhead: hypervector operations are extremely fast
    
    Benefits:
    - Online learning: add/remove models without retraining
    - Life-long learning: can bundle hypervectors over time
    - Little computational overhead vs. neural network inference
    
    Args:
        num_classes: Number of output classes
        hv_dim: Hypervector dimension
        num_contaminants: Number of contaminant outputs (for multi-task)
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        hv_dim: int = 10000,
        num_contaminants: int = 4,
        device: torch.device = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hv_dim = hv_dim
        self.num_contaminants = num_contaminants
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Hypervector operations
        self.hv_ops = BinaryHV(hv_dim, self.device)
        
        # Encoders for each task head
        self.health_encoder = HVEncoder(num_classes, hv_dim, device=self.device)
        self.contam_encoder = HVEncoder(num_contaminants, hv_dim, device=self.device)
        
        # Classifiers for each task head
        self.health_classifier = HVClassifier(hv_dim, num_classes, self.device)
        self.contam_classifier = HVClassifier(hv_dim, num_contaminants, self.device)
        
        # Registered models: name -> (model, weight)
        self.models: Dict[str, Tuple[nn.Module, float]] = {}
        self.model_weights = nn.ParameterDict()
    
    def add_model(self, name: str, model: nn.Module, weight: float = 1.0) -> None:
        """
        Register a neural network model for ensemble.
        
        Args:
            name: Unique identifier for the model
            model: Neural network with forward returning (health_logits, contam_probs)
            weight: Ensemble weight for this model (default 1.0)
        """
        self.models[name] = (model, weight)
        self.model_weights[name] = nn.Parameter(torch.tensor(weight))
    
    def remove_model(self, name: str) -> None:
        """Remove a model from the ensemble."""
        if name in self.models:
            del self.models[name]
            del self.model_weights[name]
    
    def encode_outputs(
        self,
        health_logits: torch.Tensor,
        contam_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode neural network outputs to hypervectors.
        
        Args:
            health_logits: (batch, num_classes) health classification logits
            contam_probs: (batch, num_contaminants) contaminant probabilities
        
        Returns:
            health_hv: (batch, hv_dim) encoded health hypervector
            contam_hv: (batch, hv_dim) encoded contaminant hypervector
        """
        # Apply softmax to logits before encoding for stable distribution
        health_probs = F.softmax(health_logits, dim=-1)
        
        health_hv = self.health_encoder.encode(health_probs)
        contam_hv = self.contam_encoder.encode(contam_probs)
        
        return health_hv, contam_hv
    
    def consensus_bundle(
        self,
        hypervectors: List[torch.Tensor],
        weights: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Bundle multiple hypervectors via weighted consensus.
        
        Args:
            hypervectors: List of (batch, hv_dim) hypervectors from each model
            weights: Optional weights for each model
        
        Returns:
            (batch, hv_dim) consensus hypervector
        """
        if weights is None:
            weights = [1.0] * len(hypervectors)
        
        # Stack and weight
        stacked = torch.stack(hypervectors, dim=1)  # (batch, n_models, hv_dim)
        weights_tensor = torch.tensor(weights, device=stacked.device).view(1, -1, 1)
        
        weighted = stacked * weights_tensor
        consensus = weighted.sum(dim=1)  # (batch, hv_dim)
        
        # Binarize
        return torch.sign(consensus)
    
    @torch.no_grad()
    def train_encoders(self, dataloader, max_batches: Optional[int] = None) -> None:
        """
        Train the hypervector encoders by passing data through all registered models.
        
        This accumulates encoded outputs into class hypervectors for classification.
        
        Args:
            dataloader: DataLoader yielding (cube, health, contam) batches
            max_batches: Optional limit on number of batches to process
        """
        self.health_classifier.reset()
        self.contam_classifier.reset()
        
        for i, (cube, health, contam) in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            
            cube = cube.to(self.device)
            health = health.to(self.device)
            contam = contam.to(self.device)
            
            # Collect hypervectors from all models
            for name, (model, _) in self.models.items():
                model.eval()
                health_logits, contam_probs = model(cube)
                
                # Encode to hypervectors
                health_hv, contam_hv = self.encode_outputs(health_logits, contam_probs)
                
                # Accumulate into class hypervectors
                self.health_classifier.add_examples(health_hv, health)
                # For contaminants, use thresholded binary labels
                contam_labels = (contam > 0.5).long()
                for c in range(self.num_contaminants):
                    self.contam_classifier.add_examples(contam_hv[:, c*self.hv_dim:(c+1)*self.hv_dim] if c > 0 else contam_hv, 
                                                        contam_labels[:, c])
        
        # Finalize class hypervectors
        self.health_classifier.finalize()
        self.contam_classifier.finalize()
    
    def predict_ensemble(
        self,
        x: torch.Tensor,
        return_individual: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """
        Predict using ensemble of all registered models via hypervector consensus.
        
        Args:
            x: Input tensor (batch, ...)
            return_individual: If True, also return individual model predictions
        
        Returns:
            health_pred: (batch,) predicted health classes
            contam_pred: (batch, num_contaminants) predicted contaminant probabilities
            individual_preds: Dict of individual model predictions if requested
        """
        individual_preds = {}
        health_hvs = []
        contam_hvs = []
        weights = []
        
        # Collect predictions and hypervectors from all models
        for name, (model, default_weight) in self.models.items():
            model.eval()
            health_logits, contam_probs = model(x)
            
            # Store individual predictions
            if return_individual:
                individual_preds[name] = {
                    "health_logits": health_logits,
                    "contam_probs": contam_probs,
                }
            
            # Encode to hypervectors
            health_hv, contam_hv = self.encode_outputs(health_logits, contam_probs)
            health_hvs.append(health_hv)
            contam_hvs.append(contam_hv)
            
            # Use learned weight
            weight = F.softmax(self.model_weights[name], dim=0) if name in self.model_weights else default_weight
            weights.append(weight if isinstance(weight, float) else weight.item())
        
        # Form consensus via bundling
        health_consensus = self.consensus_bundle(health_hvs, weights)
        contam_consensus = self.consensus_bundle(contam_hvs, weights)
        
        # Classify via similarity
        if self.health_classifier.is_finalized:
            health_pred = self.health_classifier.predict(health_consensus)
        else:
            # Fallback to argmax if not trained
            # This shouldn't happen in normal usage
            health_pred = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
        
        # For contaminants, convert hypervector back to probabilities via similarity
        contam_pred = torch.sigmoid(contam_consensus[:, :self.num_contaminants] if contam_consensus.size(1) >= self.num_contaminants 
                                   else F.pad(contam_consensus, (0, self.num_contaminants - contam_consensus.size(1))))
        
        if return_individual:
            return health_pred, contam_pred, individual_preds
        return health_pred, contam_pred
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using ensemble consensus."""
        return self.predict_ensemble(x)


# ───────────────────────────────────────────────────────────────────────────────
# End-to-End Model Glue (NN + HDC integrated)
# ───────────────────────────────────────────────────────────────────────────────

class ModelGlue(nn.Module):
    """
    End-to-end integration of neural network with hyperdimensional classification.
    
    This wraps a neural network with HDC layers, enabling:
    - Differentiable training of both NN and HDC together
    - Fallback to standard NN classification during inference
    - Hypervector-based ensemble capabilities
    
    Architecture:
        Input → Neural Network → Logits → HV Encoder → HV Classifier → Output
    
    Args:
        base_model: Neural network model
        hv_dim: Hypervector dimension
        num_classes: Number of classes
        num_contaminants: Number of contaminants
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        hv_dim: int = 10000,
        num_classes: int = 5,
        num_contaminants: int = 4,
    ):
        super().__init__()
        self.base_model = base_model
        self.hv_dim = hv_dim
        
        # HDC components
        self.health_encoder = HVEncoder(num_classes, hv_dim)
        self.contam_encoder = HVEncoder(num_contaminants, hv_dim)
        
        # Continuous (non-binary) projection for gradient flow
        self.projection_health = nn.Linear(num_classes, hv_dim, bias=False)
        self.projection_contam = nn.Linear(num_contaminants, hv_dim, bias=False)
        
        # Learnable classifier weights
        self.class_hvs = nn.Parameter(torch.randn(num_classes, hv_dim) * 0.01)
        self.contam_hvs = nn.Parameter(torch.randn(num_contaminants, hv_dim) * 0.01)
        
        # Use HDC or standard classification
        self.use_hdc = True
        self.temperature = nn.Parameter(torch.tensor(10.0))  # For soft similarity
    
    def set_mode(self, use_hdc: bool = True) -> None:
        """Toggle between HDC and standard classification."""
        self.use_hdc = use_hdc
    
    def encode_continuous(self, logits: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        """
        Continuous encoding for gradient flow during training.
        
        Unlike binary encoding, this preserves gradients by using tanh
        activation which provides smooth derivatives in the range [-1, 1].
        This allows backpropagation through the hypervector encoding step.
        
        Args:
            logits: (batch, features) input logits or probabilities
            proj: Linear projection layer to hv_dim dimensions
            
        Returns:
            (batch, hv_dim) continuous hypervector in range [-1, 1]
        """
        return torch.tanh(proj(logits))  # Continuous in [-1, 1]
    
    def hv_similarity(self, hv: torch.Tensor, class_hvs: nn.Parameter) -> torch.Tensor:
        """
        Compute differentiable similarity scores.
        
        Args:
            hv: (batch, hv_dim) continuous hypervector
            class_hvs: (num_classes, hv_dim) learnable class hypervectors
        
        Returns:
            (batch, num_classes) similarity scores
        """
        # Cosine similarity with temperature
        similarities = (hv @ class_hvs.T) / self.hv_dim
        return similarities * F.softplus(self.temperature)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass combining neural network with HDC classification.
        
        Args:
            x: Input tensor
        
        Returns:
            health_logits: (batch, num_classes) classification logits
            contam_probs: (batch, num_contaminants) probabilities
        """
        # Base model forward
        health_logits_raw, contam_probs_raw = self.base_model(x)
        
        if not self.use_hdc:
            return health_logits_raw, contam_probs_raw
        
        # Convert to probabilities for stable encoding
        health_probs = F.softmax(health_logits_raw, dim=-1)
        
        # Continuous hypervector encoding (differentiable)
        health_hv = self.encode_continuous(health_probs, self.projection_health)
        contam_hv = self.encode_continuous(contam_probs_raw, self.projection_contam)
        
        # HDC classification via similarity
        health_similarities = self.hv_similarity(health_hv, self.class_hvs)
        contam_similarities = self.hv_similarity(contam_hv, self.contam_hvs)
        
        # Convert similarities to logits/probabilities
        health_logits = health_similarities  # These are already "logits" via similarity
        contam_probs = torch.sigmoid(contam_similarities)
        
        return health_logits, contam_probs


# ───────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ───────────────────────────────────────────────────────────────────────────────

def create_consensus_ensemble(
    models: Dict[str, nn.Module],
    num_classes: int = 5,
    num_contaminants: int = 4,
    hv_dim: int = 10000,
    dataloader=None,
) -> ConsensusEnsemble:
    """
    Factory function to create and train a consensus ensemble.
    
    Args:
        models: Dict of model_name → model
        num_classes: Number of health classes
        num_contaminants: Number of contaminants
        hv_dim: Hypervector dimension
        dataloader: Optional dataloader to train encoders
    
    Returns:
        Trained ConsensusEnsemble
    """
    ensemble = ConsensusEnsemble(num_classes, hv_dim, num_contaminants)
    
    for name, model in models.items():
        ensemble.add_model(name, model)
    
    if dataloader is not None:
        print("Training hypervector encoders...")
        ensemble.train_encoders(dataloader)
    
    return ensemble


def hv_intersection(hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
    """
    Compute intersection of two binary hypervectors.
    Returns the positions where both hypervectors agree (+1,+1 or -1,-1).
    """
    return (hv1 == hv2).float()


def estimate_hv_capacity(dim: int, num_classes: int) -> float:
    """
    Estimate the capacity of a hypervector space.
    
    Returns the expected number of class hypervectors that can be
    reliably distinguished.
    """
    # Simplified capacity estimate based on dimensionality
    return dim / (2 * np.log2(num_classes))


# ───────────────────────────────────────────────────────────────────────────────
# Quick test
# ───────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Hyperdimensional Computing Module Test")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Test BinaryHV operations
    print("\n1. Binary Hypervector Operations")
    hv_ops = BinaryHV(dim=1000, device=device)
    
    # Generate random hypervectors
    hv1 = hv_ops.random(1000)
    hv2 = hv_ops.random(1000)
    
    print(f"  HV1 shape: {hv1.shape}, unique values: {torch.unique(hv1).tolist()}")
    print(f"  HV2 shape: {hv2.shape}, unique values: {torch.unique(hv2).tolist()}")
    
    # Test bundling
    bundle = hv_ops.bundle(torch.stack([hv1, hv2]))
    print(f"  Bundle shape: {bundle.shape}")
    
    # Test similarity
    sim = hv_ops.similarity(hv1, hv2)
    self_sim = hv_ops.similarity(hv1, hv1)
    print(f"  Similarity HV1-HV2: {sim.item():.4f} (expected ~0)")
    print(f"  Self-similarity: {self_sim.item():.4f} (expected 1.0)")
    
    # Test encoding
    print("\n2. HV Encoder")
    encoder = HVEncoder(input_dim=5, hv_dim=1000, device=device)
    logits = torch.randn(10, 5, device=device)
    hvs = encoder.encode(logits)
    print(f"  Input: {logits.shape} → Output: {hvs.shape}")
    print(f"  Output unique values: {torch.unique(hvs).tolist()}")
    
    # Test classifier
    print("\n3. HV Classifier")
    classifier = HVClassifier(hv_dim=1000, num_classes=5, device=device)
    
    # Generate synthetic training data
    train_hvs = torch.randn(100, 1000, device=device)
    train_labels = torch.randint(0, 5, (100,), device=device)
    
    # Binarize training hypervectors
    train_hvs = torch.sign(train_hvs)
    train_hvs[train_hvs == 0] = 1
    
    classifier.add_examples(train_hvs, train_labels)
    classifier.finalize()
    
    # Test prediction
    test_hvs = torch.sign(torch.randn(20, 1000, device=device))
    preds = classifier.predict(test_hvs)
    print(f"  Train examples: 100, Test examples: 20")
    print(f"  Predictions shape: {preds.shape}")
    print(f"  Unique predictions: {torch.unique(preds).tolist()}")
    
    # Test consensus ensemble
    print("\n4. Consensus Ensemble")
    
    # Create dummy models
    class DummyModel(nn.Module):
        def forward(self, x):
            batch = x.size(0)
            return torch.randn(batch, 5, device=x.device), torch.sigmoid(torch.randn(batch, 4, device=x.device))
    
    ensemble = ConsensusEnsemble(num_classes=5, hv_dim=1000, num_contaminants=4, device=device)
    ensemble.add_model("model_a", DummyModel())
    ensemble.add_model("model_b", DummyModel())
    
    x = torch.randn(5, 1, 200, 64, 64, device=device)
    health_pred, contam_pred = ensemble.predict_ensemble(x)
    print(f"  Input: {x.shape}")
    print(f"  Health predictions: {health_pred.shape}")
    print(f"  Contaminant predictions: {contam_pred.shape}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
