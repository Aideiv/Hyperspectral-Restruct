"""
evaluate.py — Full test-set evaluation for SoilHSI3DCNN.

Runs a held-out dataset through the trained model and reports:
  • Soil health head  : accuracy, macro F1, per-class precision/recall/F1,
                        confusion matrix
  • Contaminant head  : per-contaminant ROC-AUC, F1, precision, recall
                        (at 0.5 threshold), mean AUC
  • Optional plots    : confusion matrix heatmap, ROC curves per contaminant

Usage:
    python evaluate.py --data-dir data/test --labels data/test_labels.csv \
                       --checkpoint runs/best_model.pth

    python evaluate.py --data-dir data/test --labels data/test_labels.csv \
                       --checkpoint runs/best_model.pth --save-plots
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.constants import (
    HEALTH_LABELS,
    CONTAMINANT_NAMES as CONTAM_NAMES,
    CONTAM_THRESHOLD,
    MODEL_DEFAULTS,
)
from utils.common import load_config, load_model_from_checkpoint, get_device, ensure_dir


# ── Config ────────────────────────────────────────────────────────────────────



# ── Data loading ──────────────────────────────────────────────────────────────

def load_labels_csv(csv_path: str) -> tuple[list[str], list[tuple]]:
    """
    Expects a CSV with columns:
        path, health_class, metal, pfas, glyphosate, microplastics

    Returns:
        paths  : list of .npy file paths
        labels : list of (health_int, [c0, c1, c2, c3]) tuples
    """
    import os
    
    # Validate CSV file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    paths, labels = [], []
    required_columns = ["path", "health_class"] + CONTAM_NAMES
    
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            
            # Validate required columns exist
            if not all(col in reader.fieldnames for col in required_columns):
                missing_cols = [col for col in required_columns if col not in reader.fieldnames]
                raise ValueError(f"Missing required columns in CSV: {missing_cols}")
            
            for row_num, row in enumerate(reader, 1):
                try:
                    # Validate path exists
                    file_path = row["path"]
                    if not os.path.exists(file_path):
                        print(f"⚠️  Warning: File not found at row {row_num}: {file_path}")
                        continue
                    
                    paths.append(file_path)
                    
                    # Validate and convert health_class
                    try:
                        health = int(row["health_class"])
                        if health < 0 or health >= len(HEALTH_LABELS):
                            print(f"⚠️  Warning: Invalid health_class {health} at row {row_num}, using 0")
                            health = 0
                    except ValueError:
                        print(f"⚠️  Warning: Invalid health_class '{row['health_class']}' at row {row_num}, using 0")
                        health = 0
                    
                    # Validate and convert contaminant values
                    contam = []
                    for name in CONTAM_NAMES:
                        try:
                            value = float(row[name])
                            if not (0 <= value <= 1):
                                print(f"⚠️  Warning: Invalid {name} value {value} at row {row_num}, clipping to [0,1]")
                                value = max(0, min(1, value))
                            contam.append(value)
                        except ValueError:
                            print(f"⚠️  Warning: Invalid {name} value '{row[name]}' at row {row_num}, using 0")
                            contam.append(0.0)
                    
                    labels.append((health, contam))
                    
                except Exception as e:
                    print(f"⚠️  Warning: Error processing row {row_num}: {e}")
                    continue
    
    except Exception as e:
        raise RuntimeError(f"Error reading CSV file {csv_path}: {e}")
    
    if not paths:
        raise ValueError(f"No valid data found in CSV file: {csv_path}")
    
    print(f"✅ Loaded {len(paths)} valid samples from {csv_path}")
    return paths, labels


# ── Inference loop ────────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Pass full test set through model.

    Returns raw arrays:
        health_preds   : (N,)       int   — argmax class
        health_probs   : (N, 5)     float — softmax probabilities
        health_targets : (N,)       int
        contam_probs   : (N, 4)     float — sigmoid probabilities
        contam_targets : (N, 4)     float — ground-truth binary labels
    """
    from tqdm import tqdm
    
    model.eval()

    all_health_preds   = []
    all_health_probs   = []
    all_health_targets = []
    all_contam_probs   = []
    all_contam_targets = []

    for cube, health, contam in tqdm(loader, desc="Evaluating", leave=False):
        cube   = cube.to(device, non_blocking=True)
        health = health.to(device, non_blocking=True)
        contam = contam.to(device, non_blocking=True)

        health_logits, contam_p = model(cube)

        health_p = torch.softmax(health_logits, dim=1)
        preds    = health_p.argmax(dim=1)

        all_health_preds.append(preds.cpu().numpy())
        all_health_probs.append(health_p.cpu().numpy())
        all_health_targets.append(health.cpu().numpy())
        all_contam_probs.append(contam_p.cpu().numpy())
        all_contam_targets.append(contam.cpu().numpy())

    return {
        "health_preds":   np.concatenate(all_health_preds),
        "health_probs":   np.concatenate(all_health_probs),
        "health_targets": np.concatenate(all_health_targets),
        "contam_probs":   np.concatenate(all_contam_probs),
        "contam_targets": np.concatenate(all_contam_targets),
    }


# ── Metrics ───────────────────────────────────────────────────────────────────

def health_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Comprehensive health classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        confusion_matrix,
        cohen_kappa_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
    )

    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(targets, preds, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(targets, preds)
    balanced_acc = balanced_accuracy_score(targets, preds)
    
    # Per-class metrics
    precision_per_class = precision_score(targets, preds, average=None, zero_division=0, labels=list(range(len(HEALTH_LABELS))))
    recall_per_class = recall_score(targets, preds, average=None, zero_division=0, labels=list(range(len(HEALTH_LABELS))))
    f1_per_class = f1_score(targets, preds, average=None, zero_division=0, labels=list(range(len(HEALTH_LABELS))))
    
    report = classification_report(
        targets, preds,
        target_names=HEALTH_LABELS,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(targets, preds, labels=list(range(len(HEALTH_LABELS))))

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cohen_kappa": kappa,
        "balanced_accuracy": balanced_acc,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "report": report,
        "confusion_matrix": cm,
    }


def contam_metrics(probs: np.ndarray, targets: np.ndarray) -> dict:
    """Comprehensive per-contaminant evaluation metrics."""
    import warnings
    from sklearn.metrics import (
        roc_auc_score,
        f1_score,
        precision_score,
        recall_score,
        roc_curve,
        average_precision_score,
        matthews_corrcoef,
    )

    preds_bin = (probs >= CONTAM_THRESHOLD).astype(int)
    results   = {}
    auc_list  = []

    for i, name in enumerate(CONTAM_NAMES):
        y_true  = targets[:, i]
        y_score = probs[:, i]
        y_pred  = preds_bin[:, i]

        if y_true.sum() == 0 or (1 - y_true).sum() == 0:
            warnings.warn(
                f"'{name}': only one class present in test set — AUC undefined.",
                RuntimeWarning,
            )
            auc = float("nan")
            ap = float("nan")
            mcc = float("nan")
            fpr = tpr = thresholds = None
        else:
            auc = roc_auc_score(y_true, y_score)
            ap = average_precision_score(y_true, y_score)
            mcc = matthews_corrcoef(y_true, y_pred)
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            auc_list.append(auc)

        # Calculate additional metrics
        support = int(y_true.sum())
        specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        
        results[name] = {
            "auc": auc,
            "average_precision": ap,
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "specificity": specificity,
            "mcc": mcc,
            "support": support,
            "fpr": fpr,
            "tpr": tpr,
        }

    results["mean_auc"] = float(np.nanmean([r["auc"] for r in results.values()
                                             if isinstance(r, dict)]))
    results["mean_ap"] = float(np.nanmean([r["average_precision"] for r in results.values()
                                         if isinstance(r, dict)]))
    return results


# ── Printing ──────────────────────────────────────────────────────────────────

def print_health_results(m: dict) -> None:
    print("\n" + "═" * 65)
    print("  SOIL HEALTH CLASSIFICATION")
    print("═" * 65)
    print(f"  Accuracy           : {m['accuracy']*100:.2f}%")
    print(f"  Macro F1           : {m['macro_f1']:.4f}")
    print(f"  Weighted F1        : {m['weighted_f1']:.4f}")
    print(f"  Cohen's Kappa      : {m['cohen_kappa']:.4f}")
    print(f"  Balanced Accuracy  : {m['balanced_accuracy']:.4f}")
    print()
    print(f"  {'Class':<26} {'Prec':>6} {'Rec':>6} {'F1':>6} {'N':>5}")
    print("  " + "-" * 50)
    for i, label in enumerate(HEALTH_LABELS):
        r = m["report"].get(label, {})
        print(
            f"  {label:<26} "
            f"{r.get('precision', 0):>6.3f} "
            f"{r.get('recall', 0):>6.3f} "
            f"{r.get('f1-score', 0):>6.3f} "
            f"{int(r.get('support', 0)):>5}"
        )

    print("\n  Confusion matrix (rows=true, cols=pred):")
    header = "       " + "".join(f"{i:>5}" for i in range(len(HEALTH_LABELS)))
    print(header)
    for i, row in enumerate(m["confusion_matrix"]):
        print(f"  [{i}]  " + "".join(f"{v:>5}" for v in row))
    print()


def print_contam_results(m: dict) -> None:
    print("═" * 80)
    print("  CONTAMINANT DETECTION  (threshold = 0.50)")
    print("═" * 80)
    print(f"  {'Contaminant':<20} {'AUC':>6} {'AP':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Spec':>6} {'MCC':>6} {'Sup':>5}")
    print("  " + "-" * 72)
    for name in CONTAM_NAMES:
        r = m[name]
        auc_str = f"{r['auc']:.4f}" if not np.isnan(r["auc"]) else "  NaN "
        ap_str = f"{r['average_precision']:.4f}" if not np.isnan(r["average_precision"]) else "  NaN "
        mcc_str = f"{r['mcc']:.3f}" if not np.isnan(r["mcc"]) else "  NaN "
        print(
            f"  {name:<20} {auc_str:>6} {ap_str:>6} "
            f"{r['f1']:>6.3f} "
            f"{r['precision']:>6.3f} "
            f"{r['recall']:>6.3f} "
            f"{r['specificity']:>6.3f} "
            f"{mcc_str:>6} "
            f"{r['support']:>5}"
        )
    print(f"\n  Mean AUC : {m['mean_auc']:.4f}")
    print(f"  Mean AP  : {m['mean_ap']:.4f}")
    print("═" * 80 + "\n")


# ── Plots ─────────────────────────────────────────────────────────────────────

def save_confusion_matrix(cm: np.ndarray, output_path: str) -> None:
    """Enhanced confusion matrix visualization with normalized version."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("⚠️  matplotlib not installed — skipping confusion matrix plot.")
        return

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Raw confusion matrix
    im1 = ax1.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    ax1.set_xticks(range(len(HEALTH_LABELS)))
    ax1.set_yticks(range(len(HEALTH_LABELS)))
    ax1.set_xticklabels(HEALTH_LABELS, rotation=35, ha="right", fontsize=9)
    ax1.set_yticklabels(HEALTH_LABELS, fontsize=9)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title("Confusion Matrix (Raw Counts)")

    # Annotate raw matrix
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10)
    
    # Normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Handle division by zero
    
    im2 = ax2.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    ax2.set_xticks(range(len(HEALTH_LABELS)))
    ax2.set_yticks(range(len(HEALTH_LABELS)))
    ax2.set_xticklabels(HEALTH_LABELS, rotation=35, ha="right", fontsize=9)
    ax2.set_yticklabels(HEALTH_LABELS, fontsize=9)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Confusion Matrix (Normalized)")

    # Annotate normalized matrix
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax2.text(j, i, f"{cm_norm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Enhanced confusion matrix saved: {output_path}")
    plt.close()


def save_roc_curves(contam_m: dict, output_path: str) -> None:
    """Enhanced ROC curves with precision-recall curves and additional metrics."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib not installed — skipping ROC curves.")
        return

    # Create figure with ROC and PR curves
    fig, axes = plt.subplots(2, len(CONTAM_NAMES), figsize=(16, 8))
    if len(CONTAM_NAMES) == 1:
        axes = axes.reshape(2, 1)

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    for idx, name in enumerate(CONTAM_NAMES):
        r = contam_m[name]
        color = colors[idx % len(colors)]
        
        # ROC curve (top row)
        ax_roc = axes[0, idx] if len(CONTAM_NAMES) > 1 else axes[0, idx]
        if r["fpr"] is None:
            ax_roc.text(0.5, 0.5, "AUC undefined\n(single class)",
                       ha="center", va="center", transform=ax_roc.transAxes, fontsize=10)
        else:
            ax_roc.plot(r["fpr"], r["tpr"],
                       label=f"AUC = {r['auc']:.3f}", color=color, lw=2)
            ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.7)
            ax_roc.fill_between(r["fpr"], r["tpr"], alpha=0.3, color=color)
            ax_roc.legend(loc="lower right", fontsize=9)
            ax_roc.text(0.05, 0.95, f"AP = {r.get('average_precision', 0):.3f}",
                       transform=ax_roc.transAxes, fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title(f"{name} - ROC Curve")
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1.02)
        ax_roc.grid(True, alpha=0.3)

        # Precision-Recall curve (bottom row)
        ax_pr = axes[1, idx] if len(CONTAM_NAMES) > 1 else axes[1, idx]
        if r["fpr"] is not None:
            # Create PR curve from ROC data
            from sklearn.metrics import precision_recall_curve
            # Note: We'll need the original probabilities and targets for PR curve
            # For now, show a placeholder with metrics
            ax_pr.text(0.5, 0.5, f"Precision-Recall\nAP = {r.get('average_precision', 0):.3f}\nF1 = {r['f1']:.3f}",
                       ha="center", va="center", transform=ax_pr.transAxes, fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.2))
        else:
            ax_pr.text(0.5, 0.5, "PR undefined\n(single class)",
                       ha="center", va="center", transform=ax_pr.transAxes, fontsize=10)

        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title(f"{name} - Precision-Recall")
        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1.02)
        ax_pr.grid(True, alpha=0.3)

    fig.suptitle("Contaminant Detection: ROC and Precision-Recall Curves", 
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Enhanced ROC and PR curves saved: {output_path}")
    plt.close()


def save_metrics_json(health_metrics: dict, contam_metrics: dict, output_path: str) -> None:
    """Save detailed metrics to JSON file for further analysis."""
    import json
    from datetime import datetime
    
    # Prepare comprehensive metrics dictionary
    metrics_summary = {
        "timestamp": datetime.now().isoformat(),
        "health_classification": {
            "accuracy": float(health_metrics["accuracy"]),
            "macro_f1": float(health_metrics["macro_f1"]),
            "weighted_f1": float(health_metrics["weighted_f1"]),
            "cohen_kappa": float(health_metrics["cohen_kappa"]),
            "balanced_accuracy": float(health_metrics["balanced_accuracy"]),
            "per_class_metrics": {
                label: {
                    "precision": float(health_metrics["precision_per_class"][i]),
                    "recall": float(health_metrics["recall_per_class"][i]),
                    "f1": float(health_metrics["f1_per_class"][i])
                }
                for i, label in enumerate(HEALTH_LABELS)
            },
            "confusion_matrix": health_metrics["confusion_matrix"].tolist(),
            "detailed_report": health_metrics["report"]
        },
        "contaminant_detection": {
            "mean_auc": float(contam_metrics["mean_auc"]),
            "mean_average_precision": float(contam_metrics["mean_ap"]),
            "per_contaminant_metrics": {
                name: {
                    "auc": float(contam_metrics[name]["auc"]) if not np.isnan(contam_metrics[name]["auc"]) else None,
                    "average_precision": float(contam_metrics[name]["average_precision"]) if not np.isnan(contam_metrics[name]["average_precision"]) else None,
                    "f1": float(contam_metrics[name]["f1"]),
                    "precision": float(contam_metrics[name]["precision"]),
                    "recall": float(contam_metrics[name]["recall"]),
                    "specificity": float(contam_metrics[name]["specificity"]),
                    "mcc": float(contam_metrics[name]["mcc"]) if not np.isnan(contam_metrics[name]["mcc"]) else None,
                    "support": int(contam_metrics[name]["support"])
                }
                for name in CONTAM_NAMES
            }
        },
        "metadata": {
            "health_labels": HEALTH_LABELS,
            "contaminant_names": CONTAM_NAMES,
            "contaminant_threshold": CONTAM_THRESHOLD
        }
    }
    
    try:
        with open(output_path, 'w') as f:
            json.dump(metrics_summary, f, indent=2, default=str)
        print(f"✅ Detailed metrics saved: {output_path}")
    except Exception as e:
        print(f"⚠️  Failed to save metrics JSON: {e}")


def load_model(checkpoint_path: str, device: torch.device, num_bands: int) -> torch.nn.Module:
    """
    Load model from checkpoint with comprehensive error handling.
    Uses centralized loader from utils.common.
    """
    model = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        num_bands=num_bands,
        num_classes=len(HEALTH_LABELS),
        num_contaminants=len(CONTAM_NAMES),
    )
    print(f"✅ Model loaded successfully from: {checkpoint_path}")
    return model


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SoilHSI3DCNN on a held-out test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-dir",   required=True,
                        help="Directory containing test .npy cubes")
    parser.add_argument("--labels",     required=True,
                        help="CSV file with columns: path,health_class,metal,pfas,glyphosate,microplastics")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained .pth checkpoint")
    parser.add_argument("--config",     default="configs/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save confusion matrix and ROC curve plots")
    parser.add_argument("--save-metrics", action="store_true",
                        help="Save detailed metrics to JSON file")
    parser.add_argument("--output-dir", default="outputs",
                        help="Directory for saved outputs (default: outputs/)")
    parser.add_argument("--cpu",        action="store_true",
                        help="Force CPU evaluation even if CUDA is available")
    parser.add_argument("--threshold", type=float, default=CONTAM_THRESHOLD,
                        help=f"Binary threshold for contaminant detection (default: {CONTAM_THRESHOLD})")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Update global threshold if provided
    global CONTAM_THRESHOLD
    CONTAM_THRESHOLD = args.threshold

    # ── Setup ──────────────────────────────────────────────────────────────
    device = get_device(prefer_cpu=args.cpu)
    print(f"🖥️  Device: {device}")
    print(f"📊 Contaminant threshold: {CONTAM_THRESHOLD}")

    cfg = load_config(args.config)
    num_bands   = cfg["model"]["num_bands"]
    target_size = tuple(cfg["model"]["target_size"])

    # ── Dataset ────────────────────────────────────────────────────────────
    try:
        from dataset import HyperspectralSoilDataset
    except ImportError:
        sys.exit("❌  Cannot import dataset.py. Run from the repo root.")

    paths, labels = load_labels_csv(args.labels)
    test_ds = HyperspectralSoilDataset(
        paths, labels,
        num_bands=num_bands,
        target_size=target_size,
        train=False,           # no augmentation during evaluation
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    print(f"✅ Test set: {len(test_ds)} samples")

    # ── Model ──────────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, device, num_bands)

    # ── Inference ──────────────────────────────────────────────────────────
    print("Running inference...")
    raw = run_evaluation(model, test_loader, device)

    # ── Metrics ────────────────────────────────────────────────────────────
    h_metrics = health_metrics(raw["health_preds"], raw["health_targets"])
    c_metrics = contam_metrics(raw["contam_probs"], raw["contam_targets"])

    print_health_results(h_metrics)
    print_contam_results(c_metrics)

    # ── Outputs ────────────────────────────────────────────────────────────
    if args.save_plots or args.save_metrics:
        out = ensure_dir(args.output_dir)
        
        if args.save_plots:
            save_confusion_matrix(h_metrics["confusion_matrix"],
                                   str(out / "confusion_matrix.png"))
            save_roc_curves(c_metrics, str(out / "roc_curves.png"))
        
        if args.save_metrics:
            save_metrics_json(h_metrics, c_metrics, 
                            str(out / "evaluation_metrics.json"))
        
        print(f"📁 All outputs saved to: {out}")
    
    print("\n🎉 Evaluation completed successfully!")


if __name__ == "__main__":
    main()
