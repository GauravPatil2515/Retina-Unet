"""
Download additional retina vessel segmentation datasets
Automatically downloads DRIVE, STARE, and CHASE_DB1 datasets
"""

import os
import requests
import zipfile
from pathlib import Path
import shutil

def download_file(url, destination):
    """Download file with progress"""
    print(f"📥 Downloading from {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        if total_size == 0:
            f.write(response.content)
        else:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                downloaded += len(chunk)
                f.write(chunk)
                done = int(50 * downloaded / total_size)
                print(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded}/{total_size} bytes", end='')
    print()

def download_drive_dataset(output_dir="datasets/DRIVE"):
    """
    Download DRIVE dataset (40 images)
    Direct link: https://www.isi.uu.nl/Research/Databases/DRIVE/
    
    Note: DRIVE requires manual download due to terms of use
    """
    print("\n" + "="*70)
    print("📚 DRIVE DATASET")
    print("="*70)
    print("\n⚠️  DRIVE dataset requires manual download:")
    print("\n1. Visit: https://www.isi.uu.nl/Research/Databases/DRIVE/download.php")
    print("2. Accept the terms of use")
    print("3. Download 'training.zip' and 'test.zip'")
    print("4. Extract them to: datasets/DRIVE/")
    print("\nFolder structure should be:")
    print("  datasets/DRIVE/")
    print("    ├── training/")
    print("    │   ├── images/")
    print("    │   └── 1st_manual/")
    print("    └── test/")
    print("        ├── images/")
    print("        └── 1st_manual/")
    print("\n" + "="*70)
    
def download_stare_dataset(output_dir="datasets/STARE"):
    """
    Download STARE dataset (20 images)
    Website: http://cecas.clemson.edu/~ahoover/stare/
    
    Note: STARE also requires manual steps
    """
    print("\n" + "="*70)
    print("📚 STARE DATASET")
    print("="*70)
    print("\n⚠️  STARE dataset requires manual download:")
    print("\n1. Visit: http://cecas.clemson.edu/~ahoover/stare/probing/index.html")
    print("2. Download 'all-images.zip' (images)")
    print("3. Download 'labels-ah.zip' (vessel labels)")
    print("4. Extract them to: datasets/STARE/")
    print("\nFolder structure should be:")
    print("  datasets/STARE/")
    print("    ├── images/")
    print("    └── labels/")
    print("\n" + "="*70)

def download_chasedb1_dataset(output_dir="datasets/CHASE_DB1"):
    """
    Download CHASE_DB1 dataset (28 images)
    """
    print("\n" + "="*70)
    print("📚 CHASE_DB1 DATASET")
    print("="*70)
    print("\n⚠️  CHASE_DB1 dataset requires manual download:")
    print("\n1. Visit: https://blogs.kingston.ac.uk/retinal/chasedb1/")
    print("2. Download the dataset")
    print("3. Extract to: datasets/CHASE_DB1/")
    print("\nFolder structure should be:")
    print("  datasets/CHASE_DB1/")
    print("    ├── images/")
    print("    └── labels/")
    print("\n" + "="*70)

def kaggle_datasets():
    """Show Kaggle retinal datasets"""
    print("\n" + "="*70)
    print("🏆 KAGGLE DATASETS (Easy to download)")
    print("="*70)
    print("\nInstall Kaggle CLI:")
    print("  pip install kaggle")
    print("\nDownload popular datasets:")
    print("\n1. Retinal Blood Vessel Segmentation (Large):")
    print("   kaggle datasets download -d abdallahalidev/retina-blood-vessel-segmentation")
    print("\n2. DRIVE Dataset (Kaggle Mirror):")
    print("   kaggle datasets download -d andrewmvd/drive-digital-retinal-images-for-vessel-extraction")
    print("\n3. Retinal OCT Images:")
    print("   kaggle datasets download -d paultimothymooney/kermany2018")
    print("\n" + "="*70)

def create_combined_dataset():
    """Script to combine all datasets into one"""
    print("\n" + "="*70)
    print("🔄 COMBINE DATASETS")
    print("="*70)
    print("\nAfter downloading all datasets, run:")
    print("  python prepare_combined_dataset.py")
    print("\nThis will:")
    print("  • Resize all images to 512x512")
    print("  • Normalize naming conventions")
    print("  • Split into train/val/test (70/15/15)")
    print("  • Create augmented versions")
    print("\n" + "="*70)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 RETINA VESSEL SEGMENTATION - DATASET DOWNLOADER")
    print("="*70)
    
    print("\n📊 CURRENT STATUS:")
    print("  ✅ You have: 80 training images + 20 test images = 100 total")
    print("  🎯 Goal: Get 300-500+ images for better performance")
    
    print("\n💡 RECOMMENDED DATASETS TO ADD:")
    print("  1. DRIVE: 40 images (high quality)")
    print("  2. STARE: 20 images (diverse pathologies)")
    print("  3. CHASE_DB1: 28 images (child retinas)")
    print("  4. Kaggle datasets: 100+ images")
    print("\n  Total potential: 100 (current) + 188+ (new) = 288+ images")
    
    # Show download instructions
    download_drive_dataset()
    download_stare_dataset()
    download_chasedb1_dataset()
    kaggle_datasets()
    create_combined_dataset()
    
    print("\n" + "="*70)
    print("✅ NEXT STEPS:")
    print("="*70)
    print("\n1. Download datasets using links above")
    print("2. Install Kaggle CLI for easy downloads: pip install kaggle")
    print("3. Run: python prepare_combined_dataset.py")
    print("4. Train with more data: python train_improved.py")
    print("\n" + "="*70)
    print()
