#!/usr/bin/env python3
"""
Download the best.pth model from Google Drive or external storage
This script runs during deployment to fetch the trained model
"""

import os
import urllib.request
import sys
from pathlib import Path

# Model information
MODEL_URL = "https://drive.google.com/uc?export=download&id=1rRLJe9J0XCwHgq8SR91fMNzTqjxiacHA"  # Google Drive direct download
MODEL_PATH = "results/checkpoints_unetpp/best.pth"
EXPECTED_SIZE_MB = 104

def download_model():
    """Download the trained model checkpoint"""
    
    print("=" * 60)
    print("MODEL DOWNLOAD SCRIPT")
    print("=" * 60)
    
    # Create directory if it doesn't exist
    model_path = Path(MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if model already exists
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Model already exists: {model_path}")
        print(f"✓ Size: {size_mb:.2f} MB")
        
        if size_mb > EXPECTED_SIZE_MB * 0.9:  # Allow 10% variance
            print("✓ Model size looks correct!")
            return True
        else:
            print(f"⚠ Model file is too small ({size_mb:.2f} MB vs expected {EXPECTED_SIZE_MB} MB)")
            print("⚠ Downloading fresh copy...")
    
    # Download model
    if MODEL_URL == "YOUR_GOOGLE_DRIVE_DIRECT_LINK_HERE":
        print("=" * 60)
        print("✗ ERROR: MODEL_URL not configured!")
        print("=" * 60)
        print("Please upload best.pth to Google Drive or other cloud storage")
        print("and update MODEL_URL in download_model.py")
        print("=" * 60)
        print("Using random weights for now...")
        return False
    
    try:
        print(f"Downloading model from: {MODEL_URL}")
        print(f"Destination: {model_path}")
        
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = (downloaded / total_size) * 100 if total_size > 0 else 0
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\rDownloading: {downloaded_mb:.1f}/{total_mb:.1f} MB ({percent:.1f}%)", end='')
        
        urllib.request.urlretrieve(MODEL_URL, model_path, reporthook=report_progress)
        print("\n✓ Download complete!")
        
        # Verify download
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✓ Downloaded size: {size_mb:.2f} MB")
        
        if size_mb < EXPECTED_SIZE_MB * 0.9:
            print(f"✗ ERROR: Downloaded file is too small!")
            return False
        
        print("✓ Model download successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR downloading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
