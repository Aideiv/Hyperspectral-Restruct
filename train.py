"""
train.py — Icarus: Hyperspectral Soil CNN
Trains a 3D CNN on HYPERVIEW2 patches loaded via its STAC catalog.

Usage:
    # Step 1 — download patches from catalog (only needed once)
    python train.py --prepare --catalog /root/.cache/eotdl/datasets/HYPERVIEW2/catalog.v2.parquet --data_dir ./data/hyperview2

    # Step 2 — train
    python train.py --data_dir ./data/hyperview2

    # Quick smoke test (no data needed)
    python train.py --dummy
"""

import argparse
import os
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from configs.constants import (
    CONTAMINANT_NAMES,
    DEFAULT_PATHS,
    HEALTH_LABELS,
    MODEL_DEFAULTS,
    MODEL_VARIANTS,
    TRAINING_DEFAULTS,
    get_model_config,
)
from model import create_model

# ── Config ────────────────────────────────────────────────────────────────────

CFG = {
    **MODEL_DEFAULTS,
    **TRAINING_DEFAULTS,
    "save_path": DEFAULT_PATHS["save_path"],
    "num_bands": 150,       # HYPERVIEW2: ~150 VNIR bands
    "num_workers": 4,
}

# ── Data preparation ──────────────────────────────────────────────────────────

def parse_catalog(catalog_path: str) -> pd.DataFrame:
    """
    Read the STAC catalog parquet and return a DataFrame with columns:
        id, href
    one row per file in the dataset.
    """
    df = pd.read_parquet(catalog_path)
    records = []
    for _, row in df.iterrows():
        assets = row.get("assets", {})
        if isinstance(assets, dict):
            asset = assets.get("asset", {})
            href = asset.get("href", "")
            records.append({"id": row["id"], "href": href})
    return pd.DataFrame(records)


def download_file(url: str, dest: Path, chunk_size: int = 8192) -> bool:
    """Download a single file with a progress bar."""
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=dest.name, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size):
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  Failed {dest.name}: {e}")
        return False


def prepare_data(catalog_path: str, data_dir: str) -> None:
    """
    Parse the STAC catalog and download all patch files + train_gt.csv
    into data_dir. Skips files that already exist.
    """
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)

    catalog = parse_catalog(catalog_path)
    print(f"Catalog has {len(catalog)} entries.")

    for _, row in tqdm(catalog.iterrows(), total=len(catalog), desc="Downloading"):
        file_id = row["id"]
        href    = row["href"]
        dest    = out / file_id

        if dest.exists():
            continue
        if not href:
            print(f"  No href for {file_id} — skipping.")
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        download_file(href, dest)

    print(f"\nData ready at: {data_dir}")


# ── Dataset ───────────────────────────────────────────────────────────────────

# Agronomic optimal ranges for composite soil health scoring
_OPTIMA = {
    "K":    (150.0, 250.0),
    "Mg":   (100.0, 200.0),
    "P2O5": (40.0,  80.0),
    "pH":   (6.0,   7.0),
}


def chemistry_to_health_class(df: pd.DataFrame) -> List[int]:
    """
    Map K/Mg/P2O5/pH values to health classes 0–4.
    Each parameter is scored 0–1 (1 = inside optimal range),
    scores are averaged, then percentile-binned into 5 classes.
    """
    scores = np.zeros(len(df), dtype=np.float32)
    n = 0
    for col, (lo, hi) in _OPTIMA.items():
        if col not in df.columns:
            continue
        vals = df[col].to_numpy(dtype=np.float32)
        s = np.ones_like(vals)
        s[vals < lo] = vals[vals < lo] / lo
        above = vals > hi
        s[above] = hi / np.maximum(vals[above], 1e-8)
        scores += s
        n += 1

    if n == 0:
        raise ValueError("train_gt.csv has none of the expected columns: K, Mg, P2O5, pH")

    scores /= n
    thresholds = np.percentile(scores, [20, 40, 60, 80])
    classes = np.zeros(len(scores), dtype=int)
    for i, t in enumerate(thresholds):
        classes[scores > t] = i + 1
    return classes.tolist()


class Hyperview2Dataset(Dataset):
    """
    Loads HYPERVIEW2 patches from data_dir.

    Expects:
        data_dir/train_gt.csv       — patch_id, K, Mg, P2O5, pH
        data_dir/<patch_id>.*       — patch file (.npz or .npy), shape (H, W, C) or (C, H, W)
    """

    def __init__(
        self,
        data_dir: str,
        patch_ids: List[str],
        health_classes: List[int],
        num_bands: int = 150,
        target_size: Tuple[int, int] = (64, 64),
        train: bool = True,
    ):
        self.data_dir     = Path(data_dir)
        self.patch_ids    = patch_ids
        self.health_classes = health_classes
        self.num_bands    = num_bands
        self.target_size  = target_size
        self.train        = train

        dist = np.bincount(health_classes)
        print(f"  {'Train' if train else 'Val'} — {len(patch_ids)} patches, "
              f"class dist: {dist.tolist()}")

    def _find_patch_file(self, patch_id: str) -> Path:
        for ext in (".npz", ".npy", ".tif"):
            p = self.data_dir / f"{patch_id}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(
            f"No patch file found for '{patch_id}' in {self.data_dir}. "
            f"Run `python train.py --prepare` first."
        )

    def _load_patch(self, path: Path) -> np.ndarray:
        """Load patch as float32 (C, H, W)."""
        if path.suffix == ".npz":
            data = np.load(path)
            cube = data[list(data.keys())[0]].astype(np.float32)
        elif path.suffix == ".npy":
            cube = np.load(path).astype(np.float32)
        else:
            raise ValueError(f"Unsupported patch format: {path.suffix}")

        # Ensure (C, H, W)
        if cube.ndim == 2:
            cube = cube[np.newaxis, :, :]           # (1, H, W) — single band
        elif cube.ndim == 3 and cube.shape[2] < cube.shape[0]:
            cube = cube.transpose(2, 0, 1)          # (H, W, C) → (C, H, W)

        return cube

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int):
        path = self._find_patch_file(self.patch_ids[idx])
        cube = self._load_patch(path)

        # Resample bands
        if cube.shape[0] != self.num_bands:
            cube = zoom(cube, (self.num_bands / cube.shape[0], 1, 1), order=1)

        # Spatial resize
        tH, tW = self.target_size
        H, W = cube.shape[1], cube.shape[2]
        if (H, W) != (tH, tW):
            cube = zoom(cube, (1, tH / H, tW / W), order=1)

        # Random flips during training
        if self.train:
            if np.random.rand() > 0.5:
                cube = cube[:, ::-1, :].copy()
            if np.random.rand() > 0.5:
                cube = cube[:, :, ::-1].copy()

        # Band-wise z-score normalisation
        mean = cube.mean(axis=(1, 2), keepdims=True)
        std  = cube.std(axis=(1, 2),  keepdims=True) + 1e-8
        cube = (cube - mean) / std

        cube_t = torch.from_numpy(cube).unsqueeze(0)  # (1, C, H, W)

        return (
            cube_t,
            torch.tensor(self.health_classes[idx], dtype=torch.long),
            torch.zeros(len(CONTAMINANT_NAMES), dtype=torch.float32),
        )


def make_dataloaders(data_dir: str, cfg: dict) -> Tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders from HYPERVIEW2 data directory."""
    gt_path = Path(data_dir) / "train_gt.csv"
    if not gt_path.exists():
        raise FileNotFoundError(
            f"train_gt.csv not found at {gt_path}. Run --prepare first."
        )

    gt = pd.read_csv(gt_path)
    print(f"Loaded train_gt.csv: {len(gt)} patches, columns: {gt.columns.tolist()}")

    health_classes = chemistry_to_health_class(gt)

    # Identify the patch ID column
    id_col = next((c for c in gt.columns if "id" in c.lower() or "patch" in c.lower()), gt.columns[0])
    patch_ids = gt[id_col].astype(str).tolist()

    # Train / val split
    np.random.seed(42)
    idx = np.random.permutation(len(patch_ids))
    split = int(len(idx) * 0.8)
    train_idx, val_idx = idx[:split], idx[split:]

    num_bands   = cfg.get("num_bands", 150)
    target_size = tuple(cfg.get("target_size", MODEL_DEFAULTS["target_size"]))

    train_ds = Hyperview2Dataset(
        data_dir, [patch_ids[i] for i in train_idx],
        [health_classes[i] for i in train_idx],
        num_bands=num_bands, target_size=target_size, train=True,
    )
    val_ds = Hyperview2Dataset(
        data_dir, [patch_ids[i] for i in val_idx],
        [health_classes[i] for i in val_idx],
        num_bands=num_bands, target_size=target_size, train=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available(), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


# ── Dummy data (smoke test) ───────────────────────────────────────────────────

class DummyDataset(Dataset):
    def __init__(self, n: int, cfg: dict):
        sz = cfg.get("target_size", (64, 64))
        self.x = torch.randn(n, 1, cfg["num_bands"], sz[0], sz[1])
        self.y = torch.randint(0, cfg["num_classes"], (n,))
        self.c = torch.zeros(n, cfg["num_contaminants"])

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i], self.c[i]


def make_dummy_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    print("Using dummy data — replace with real HYPERVIEW2 data for production.")
    train_ds = DummyDataset(80, cfg)
    val_ds   = DummyDataset(20, cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"])
    return train_loader, val_loader


# ── Loss & validation ─────────────────────────────────────────────────────────

def compute_loss(pred_health, pred_contam, health, contam, ce, contam_w):
    ce_loss  = ce(pred_health, health)
    bce_loss = F.binary_cross_entropy(pred_contam, contam)
    return ce_loss + contam_w * bce_loss, ce_loss, bce_loss


@torch.no_grad()
def validate(model, loader, ce, contam_w, device):
    model.eval()
    total_loss = total_ce = total_bce = 0.0
    correct = total = n_batches = 0
    all_probs, all_labels = [], []

    for cube, health, contam in loader:
        cube, health, contam = cube.to(device), health.to(device), contam.to(device)
        ph, pc = model(cube)
        loss, ce_loss, bce_loss = compute_loss(ph, pc, health, contam, ce, contam_w)
        total_loss += loss.item(); total_ce += ce_loss.item(); total_bce += bce_loss.item()
        correct += (ph.argmax(1) == health).sum().item()
        total += health.size(0)
        n_batches += 1
        all_probs.append(pc.cpu().numpy())
        all_labels.append(contam.cpu().numpy())

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    auc_scores = []
    for c, name in enumerate(CONTAMINANT_NAMES):
        y_true, y_score = labels[:, c], probs[:, c]
        if y_true.sum() > 0 and (1 - y_true).sum() > 0:
            auc_scores.append(roc_auc_score(y_true, y_score))

    return {
        "val_loss":     total_loss / n_batches,
        "val_ce":       total_ce   / n_batches,
        "val_bce":      total_bce  / n_batches,
        "val_acc":      correct / total,
        "val_auc_mean": float(np.mean(auc_scores)) if auc_scores else float("nan"),
    }


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: dict, data_dir: str = None, dummy: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if dummy:
        train_loader, val_loader = make_dummy_dataloaders(cfg)
    else:
        train_loader, val_loader = make_dataloaders(data_dir, cfg)

    model_name = cfg.get("model_name", "base")
    model = create_model(
        model_name,
        num_bands=cfg["num_bands"],
        num_classes=cfg["num_classes"],
        num_contaminants=cfg["num_contaminants"],
        bottleneck_dim=cfg.get("bottleneck_dim", 512),
        dropout_p=cfg.get("dropout_p", 0.5),
    ).to(device)
    print(f"Model: {model_name}  params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg["cosine_T0"], T_mult=cfg["cosine_T_mult"])
    scaler    = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")
    ce        = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    best_auc      = -1.0
    patience_left = cfg.get("patience", 10)

    for epoch in range(cfg["epochs"]):
        model.train()
        running_loss = running_ce = running_bce = 0.0
        n = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch:03d}", leave=False)

        for i, (cube, health, contam) in pbar:
            cube, health, contam = cube.to(device), health.to(device), contam.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                ph, pc = model(cube)
                loss, ce_loss, bce_loss = compute_loss(
                    ph, pc, health, contam, ce, cfg["contam_loss_weight"]
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch + i / len(train_loader))

            running_loss += loss.item(); running_ce += ce_loss.item()
            running_bce += bce_loss.item(); n += 1

            pbar.set_postfix(loss=f"{running_loss/n:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        metrics = validate(model, val_loader, ce, cfg["contam_loss_weight"], device)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {running_loss/n:.4f} | "
            f"val loss {metrics['val_loss']:.4f}  acc {metrics['val_acc']:.3f}  "
            f"AUC {metrics['val_auc_mean']:.3f}"
        )

        if metrics["val_auc_mean"] > best_auc:
            best_auc = metrics["val_auc_mean"]
            patience_left = cfg.get("patience", 10)
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_auc": best_auc, "cfg": cfg}, cfg["save_path"])
            print(f"  Saved best model (AUC {best_auc:.4f})")
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"\nDone. Best val AUC: {best_auc:.4f}  →  {cfg['save_path']}")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Icarus Hyperspectral CNN on HYPERVIEW2")
    p.add_argument("--prepare",     action="store_true",
                   help="Download patches from STAC catalog instead of training")
    p.add_argument("--catalog",     type=str, default="/root/.cache/eotdl/datasets/HYPERVIEW2/catalog.v2.parquet",
                   help="Path to catalog.v2.parquet")
    p.add_argument("--data_dir",    type=str, default="./data/hyperview2",
                   help="Directory where patches are stored (or will be downloaded to)")
    p.add_argument("--dummy",       action="store_true",
                   help="Train on synthetic data (no download needed)")
    p.add_argument("--model",       type=str, default="base", choices=MODEL_VARIANTS)
    p.add_argument("--epochs",      type=int)
    p.add_argument("--lr",          type=float)
    p.add_argument("--batch_size",  type=int)
    p.add_argument("--save_path",   type=str)
    p.add_argument("--resume",      type=str, help="Path to checkpoint to resume from")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.prepare:
        prepare_data(args.catalog, args.data_dir)
    else:
        cfg = CFG.copy()

        # Apply model-variant defaults
        cfg.update(get_model_config(args.model))

        # CLI overrides
        if args.epochs:     cfg["epochs"]     = args.epochs
        if args.lr:         cfg["lr"]         = args.lr
        if args.batch_size: cfg["batch_size"] = args.batch_size
        if args.save_path:  cfg["save_path"]  = args.save_path

        print("Config:", {k: cfg[k] for k in
              ["model_name", "num_bands", "num_classes", "lr", "batch_size", "epochs", "save_path"]})

        train(cfg, data_dir=args.data_dir, dummy=args.dummy)
