#!/usr/bin/env python3
"""
download_datasets.py - Automated downloader for hyperspectral soil datasets.

Downloads and sets up the HYPERVIEW2 hyperspectral soil dataset.

Usage:
    python download_datasets.py --dataset hyperview2
    python download_datasets.py --list
"""

import argparse
import os
import sys
import zipfile
import requests
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, List, Optional

# Dataset configurations
DATASETS = {
    "hyperview2": {
        "name": "HYPERVIEW2 (AI4EO Challenge)",
        "urls": [
            "https://www.eotdl.com/datasets/HYPERVIEW2",
        ],
        "target_dir": "data/hyperview2",
        "description": "~150 bands VNIR-SWIR, airborne HSI patches over Polish agricultural fields with soil parameter ground truth (K, P₂O₅, Mg, pH)",
        "size_info": "~312 MB"
    },
}

def create_directories() -> None:
    """Create necessary directories for dataset storage."""
    dirs = ["data", "data/raw", "data/processed", "logs"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✅ Created necessary directories")

def download_file(url: str, destination: str, chunk_size: int = 8192) -> bool:
    """Download a file with progress indication."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r📥 Downloading: {percent:.1f}%", end='', flush=True)
        
        print(f"\n✅ Downloaded: {destination}")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download {url}: {e}")
        return False

def extract_zip(zip_path: str, extract_to: str) -> bool:
    """Extract a ZIP file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Extracted: {zip_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to extract {zip_path}: {e}")
        return False

def download_dataset(dataset_name: str) -> bool:
    """Download and set up a specific dataset."""
    if dataset_name not in DATASETS:
        print(f"❌ Unknown dataset: {dataset_name}")
        return False
    
    dataset = DATASETS[dataset_name]
    print(f"\n🔄 Setting up: {dataset['name']}")
    print(f"📋 Description: {dataset['description']}")
    print(f"💾 Size: {dataset['size_info']}")
    
    # Create target directory
    target_dir = Path(dataset['target_dir'])
    target_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    
    for i, url in enumerate(dataset['urls']):
        print(f"\n📥 Downloading file {i+1}/{len(dataset['urls'])}")
        
        # Parse filename from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path).split('?')[0]
        if not filename:
            filename = f"download_{i+1}.zip"
        
        filepath = target_dir / filename
        
        # Download if file doesn't exist
        if not filepath.exists():
            if download_file(url, str(filepath)):
                success_count += 1
                
                # Extract if it's a zip file
                if filename.endswith('.zip'):
                    extract_dir = target_dir / f"extracted_{i+1}"
                    extract_dir.mkdir(exist_ok=True)
                    if extract_zip(str(filepath), str(extract_dir)):
                        print(f"📁 Extracted to: {extract_dir}")
            else:
                print(f"⚠️  Skipping extraction due to download failure")
        else:
            print(f"✅ File already exists: {filepath}")
            success_count += 1
    
    print(f"\n📊 Dataset setup complete: {success_count}/{len(dataset['urls'])} files processed")
    return success_count > 0

def list_datasets() -> None:
    """List all available datasets."""
    print("📚 Available Datasets:")
    print("=" * 60)
    
    for key, dataset in DATASETS.items():
        print(f"\n🔹 {key}: {dataset['name']}")
        print(f"   📋 {dataset['description']}")
        print(f"   💾 {dataset['size_info']}")
        print(f"   📁 Target: {dataset['target_dir']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download hyperspectral soil datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_datasets.py --list
  python download_datasets.py --dataset hyperview2
        """
    )
    
    parser.add_argument("--dataset", choices=list(DATASETS.keys()),
                       help="Dataset to download")
    parser.add_argument("--list", action="store_true",
                       help="List all available datasets")
    args = parser.parse_args()

    # Create necessary directories
    create_directories()

    if args.list:
        list_datasets()
        return

    if not args.dataset:
        parser.print_help()
        return
    
    download_dataset(args.dataset)

if __name__ == "__main__":
    main()
