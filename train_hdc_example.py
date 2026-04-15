"""
train_hdc_example.py — Example training script for Hyperdimensional Computing models.

Demonstrates how to train and use HDC-enabled models based on:
"Gluing Neural Networks Symbolically Through Hyperdimensional Computing"
arXiv:2205.15534 — Sutor et al.

Three training modes demonstrated:
1. Standard HDC model (SoilHSI3DCNN_HDC) — single model with HDC classification
2. Ensemble HDC (SoilHSI3DCNN_EnsembleHDC) — fuse multiple models via consensus
3. Online learning with ConsensusEnsemble — add/remove models dynamically

Usage:
    # Train standard HDC model
    python train_hdc_example.py --mode hdc --model hdc --epochs 50
    
    # Train ensemble of multiple models with HDC fusion
    python train_hdc_example.py --mode ensemble --models base,se --epochs 50
    
    # Train with online learning (add models incrementally)
    python train_hdc_example.py --mode online --epochs 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

from model import create_model, create_ensemble_hdc, SoilHSI3DCNN_HDC
from hyperdimensional import (
    BinaryHV, HVEncoder, HVClassifier, 
    ConsensusEnsemble, create_consensus_ensemble
)
from configs.constants import CONTAMINANT_NAMES, MODEL_DEFAULTS, DATA_DEFAULTS


def train_hdc_model(
    model: SoilHSI3DCNN_HDC,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = None,
    save_path: str = "runs/hdc_model.pth",
) -> Dict:
    """
    Train a single HDC-enabled model.
    
    The model can toggle between HDC and standard classification modes.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    ce_criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCELoss()
    
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    print(f"\n{'='*60}")
    print(f"Training HDC Model")
    print(f"Device: {device}")
    print(f"HDC mode: {model.use_hdc}")
    print(f"HV dim: {model.hv_dim}")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for cube, health, contam in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            cube, health, contam = cube.to(device), health.to(device), contam.to(device)
            
            optimizer.zero_grad()
            health_logits, contam_probs = model(cube)
            
            loss = ce_criterion(health_logits, health) + \
                   0.5 * bce_criterion(contam_probs, contam)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for cube, health, contam in val_loader:
                cube, health, contam = cube.to(device), health.to(device), contam.to(device)
                
                health_logits, contam_probs = model(cube)
                
                loss = ce_criterion(health_logits, health)
                val_loss += loss.item()
                
                preds = health_logits.argmax(dim=1)
                correct += (preds == health).sum().item()
                total += health.size(0)
        
        val_acc = correct / total
        
        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch+1}: train_loss={history['train_loss'][-1]:.4f}, "
              f"val_loss={history['val_loss'][-1]:.4f}, val_acc={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, save_path)
    
    return history


def train_ensemble_hdc(
    base_model_names: List[str],
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    hv_dim: int = 10000,
    device: torch.device = None,
    save_path: str = "runs/ensemble_hdc.pth",
) -> Dict:
    """
    Train an ensemble of models fused via Hyperdimensional Computing.
    
    This implements the "gluing" technique from the paper where multiple
    neural networks are fused at the hypervector level via consensus.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create base models
    print(f"\n{'='*60}")
    print(f"Creating Ensemble HDC with models: {base_model_names}")
    print(f"{'='*60}\n")
    
    base_models = []
    for name in base_model_names:
        model = create_model(name, num_bands=200, num_classes=5, num_contaminants=4)
        base_models.append(model)
        print(f"  Added {name}: {sum(p.numel() for p in model.parameters()):,} params")
    
    # Create ensemble with HDC fusion
    ensemble = create_ensemble_hdc(
        base_models=base_models,
        num_classes=5,
        num_contaminants=4,
        hv_dim=hv_dim,
        learnable_weights=True
    ).to(device)
    
    print(f"\n  Ensemble HV dim: {hv_dim}")
    print(f"  Total params: {sum(p.numel() for p in ensemble.parameters()):,}")
    print(f"  Learnable ensemble weights: True")
    
    # Optimizer — note we train both base models and HDC components
    optimizer = AdamW(ensemble.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    ce_criterion = nn.CrossEntropyLoss()
    bce_criterion = nn.BCELoss()
    
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "ensemble_weights": []}
    
    print(f"\n{'='*60}")
    print(f"Training Ensemble HDC")
    print(f"{'='*60}\n")
    
    for epoch in range(epochs):
        # Training phase
        ensemble.train()
        train_loss = 0.0
        
        for cube, health, contam in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            cube, health, contam = cube.to(device), health.to(device), contam.to(device)
            
            optimizer.zero_grad()
            health_logits, contam_probs = ensemble(cube)
            
            loss = ce_criterion(health_logits, health) + \
                   0.5 * bce_criterion(contam_probs, contam)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # Get ensemble weights (softmax normalized)
        with torch.no_grad():
            weights = F.softmax(ensemble.weights, dim=0).cpu().numpy()
        history["ensemble_weights"].append(weights.tolist())
        
        # Validation phase
        ensemble.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for cube, health, contam in val_loader:
                cube, health, contam = cube.to(device), health.to(device), contam.to(device)
                
                health_logits, contam_probs = ensemble(cube)
                
                loss = ce_criterion(health_logits, health)
                val_loss += loss.item()
                
                preds = health_logits.argmax(dim=1)
                correct += (preds == health).sum().item()
                total += health.size(0)
        
        val_acc = correct / total
        
        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(val_acc)
        
        weight_str = ", ".join([f"{name}={w:.3f}" for name, w in zip(base_model_names, weights)])
        print(f"Epoch {epoch+1}: train_loss={history['train_loss'][-1]:.4f}, "
              f"val_loss={history['val_loss'][-1]:.4f}, val_acc={val_acc:.4f}")
        print(f"  Ensemble weights: {weight_str}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": ensemble.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "ensemble_weights": weights.tolist(),
            }, save_path)
    
    return history


def train_online_hdc(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs_per_model: int = 20,
    hv_dim: int = 10000,
    device: torch.device = None,
    save_dir: str = "runs/online_hdc",
) -> Dict:
    """
    Demonstrate online learning with ConsensusEnsemble.
    
    Models are added incrementally without retraining previous models,
    demonstrating the life-long learning capability from the paper.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Online Learning with HDC Consensus Ensemble")
    print(f"{'='*60}\n")
    print(f"This demonstrates adding models incrementally without")
    print(f"retraining — the core 'life-long learning' capability.")
    print(f"\nHV dim: {hv_dim}")
    
    # Create online ensemble
    ensemble = ConsensusEnsemble(
        num_classes=5,
        hv_dim=hv_dim,
        num_contaminants=4,
        device=device
    )
    
    history = {"models_added": [], "val_accuracies": []}
    
    # Phase 1: Train and add first model
    print(f"\n{'='*60}")
    print(f"Phase 1: Train and add Model A (base)")
    print(f"{'='*60}\n")
    
    model_a = create_model("base", num_bands=200, num_classes=5, num_contaminants=4).to(device)
    optimizer_a = AdamW(model_a.parameters(), lr=1e-3)
    ce_criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs_per_model):
        model_a.train()
        for cube, health, contam in train_loader:
            cube, health = cube.to(device), health.to(device)
            optimizer_a.zero_grad()
            health_logits, _ = model_a(cube)
            loss = ce_criterion(health_logits, health)
            loss.backward()
            optimizer_a.step()
    
    # Add model A to ensemble
    ensemble.add_model("model_a", model_a, weight=1.0)
    print(f"✓ Added Model A to ensemble")
    
    # Train encoders with model A's outputs
    print(f"\nTraining hypervector encoders...")
    ensemble.train_encoders(train_loader, max_batches=50)
    print(f"✓ Encoders trained")
    
    # Evaluate
    model_a.eval()
    correct_a = 0
    total = 0
    with torch.no_grad():
        for cube, health, contam in val_loader:
            cube, health = cube.to(device), health.to(device)
            health_logits, _ = model_a(cube)
            preds = health_logits.argmax(dim=1)
            correct_a += (preds == health).sum().item()
            total += health.size(0)
    
    acc_a = correct_a / total
    print(f"Model A standalone accuracy: {acc_a:.4f}")
    
    # Phase 2: Train and add second model
    print(f"\n{'='*60}")
    print(f"Phase 2: Train and add Model B (se)")
    print(f"{'='*60}\n")
    
    model_b = create_model("se", num_bands=200, num_classes=5, num_contaminants=4, 
                          se_reduction=16).to(device)
    optimizer_b = AdamW(model_b.parameters(), lr=1e-3)
    
    for epoch in range(epochs_per_model):
        model_b.train()
        for cube, health, contam in train_loader:
            cube, health = cube.to(device), health.to(device)
            optimizer_b.zero_grad()
            health_logits, _ = model_b(cube)
            loss = ce_criterion(health_logits, health)
            loss.backward()
            optimizer_b.step()
    
    # Add model B to ensemble (without retraining model A!)
    ensemble.add_model("model_b", model_b, weight=1.0)
    print(f"✓ Added Model B to ensemble (Model A unchanged!)")
    
    # Retrain encoders with both models' outputs
    print(f"\nRetraining encoders with both models...")
    ensemble.train_encoders(train_loader, max_batches=50)
    print(f"✓ Encoders retrained (2 models)")
    
    # Evaluate ensemble
    correct_ensemble = 0
    with torch.no_grad():
        for cube, health, contam in val_loader:
            cube, health = cube.to(device), health.to(device)
            health_pred, _ = ensemble.predict_ensemble(cube)
            correct_ensemble += (health_pred == health).sum().item()
    
    acc_ensemble = correct_ensemble / total
    
    print(f"\n{'='*60}")
    print(f"Online Learning Results")
    print(f"{'='*60}")
    print(f"Model A (base) accuracy: {acc_a:.4f}")
    print(f"Ensemble (A+B) accuracy: {acc_ensemble:.4f}")
    print(f"Improvement: {(acc_ensemble - acc_a):.4f}")
    print(f"\nKey: Model A was NOT retrained when adding Model B!")
    print(f"This is the 'gluing' capability from the paper.")
    
    history["models_added"].extend(["model_a", "model_b"])
    history["val_accuracies"].extend([acc_a, acc_ensemble])
    
    return history


def create_dummy_loaders(batch_size: int = 8, num_samples: int = 100):
    """Create dummy data loaders for testing."""
    from torch.utils.data import TensorDataset
    
    # Generate dummy data
    cubes = torch.randn(num_samples, 1, 200, 64, 64)
    health_labels = torch.randint(0, 5, (num_samples,))
    contam_labels = torch.randint(0, 2, (num_samples, 4)).float()
    
    # Split into train/val
    split = int(0.8 * num_samples)
    
    train_dataset = TensorDataset(cubes[:split], health_labels[:split], contam_labels[:split])
    val_dataset = TensorDataset(cubes[split:], health_labels[split:], contam_labels[split:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train HDC models")
    parser.add_argument("--mode", type=str, default="hdc", 
                       choices=["hdc", "ensemble", "online"],
                       help="Training mode: hdc, ensemble, or online")
    parser.add_argument("--model", type=str, default="hdc",
                       help="Model name for single model training")
    parser.add_argument("--models", type=str, default="base,se",
                       help="Comma-separated model names for ensemble")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--hv-dim", type=int, default=5000,
                       help="Hypervector dimension")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--save-dir", type=str, default="runs/hdc",
                       help="Directory to save models")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dummy data loaders
    train_loader, val_loader = create_dummy_loaders(args.batch_size)
    print(f"Created dummy data: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "hdc":
        # Train single HDC model
        model = create_model(args.model, num_bands=200, num_classes=5, 
                           num_contaminants=4, hv_dim=args.hv_dim)
        history = train_hdc_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            save_path=str(save_dir / "hdc_model.pth"),
        )
        
    elif args.mode == "ensemble":
        # Train ensemble with HDC fusion
        model_names = args.models.split(",")
        history = train_ensemble_hdc(
            base_model_names=model_names,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            hv_dim=args.hv_dim,
            device=device,
            save_path=str(save_dir / "ensemble_hdc.pth"),
        )
        
    elif args.mode == "online":
        # Demonstrate online learning
        history = train_online_hdc(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs_per_model=args.epochs,
            hv_dim=args.hv_dim,
            device=device,
            save_dir=str(save_dir),
        )
    
    # Save history
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training complete. Results saved to {save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
