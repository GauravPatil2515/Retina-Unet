"""
Project Cleanup Script
Organizes the project by archiving old files and creating proper structure
"""

import os
import shutil
from pathlib import Path

def get_folder_size(folder):
    """Calculate folder size in MB"""
    total = 0
    try:
        for entry in os.scandir(folder):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_folder_size(entry.path)
    except:
        pass
    return total / (1024 * 1024)  # Convert to MB

def safe_move(src, dst):
    """Safely move file/folder"""
    try:
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            return True
    except Exception as e:
        print(f"   ⚠️  Could not move {src}: {e}")
    return False

def safe_delete(path):
    """Safely delete file/folder"""
    try:
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
            return True
    except Exception as e:
        print(f"   ⚠️  Could not delete {path}: {e}")
    return False

def main():
    print("\n" + "="*70)
    print("🧹 RETINA U-NET PROJECT CLEANUP")
    print("="*70 + "\n")
    
    # Calculate current size
    print("📊 Calculating current project size...\n")
    folders_to_check = ['results', 'logs', 'checkpoints', 'predictions', 'models']
    for folder in folders_to_check:
        if os.path.exists(folder):
            size = get_folder_size(folder)
            print(f"   {folder:20s}: {size:>6.1f} MB")
    
    print("\n" + "="*70)
    print("🗂️  CLEANUP PLAN")
    print("="*70 + "\n")
    
    print("This script will:")
    print("  1. Move old/redundant files to 'archive/' folder (safe backup)")
    print("  2. Delete large temporary files (results/, logs/)")
    print("  3. Keep all essential files and trained models")
    print("  4. Create organized folder structure")
    print("\n" + "-"*70 + "\n")
    
    # Ask for confirmation
    response = input("Proceed with cleanup? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n❌ Cleanup cancelled. No changes made.\n")
        return
    
    print("\n" + "="*70)
    print("🚀 STARTING CLEANUP...")
    print("="*70 + "\n")
    
    # Create folders
    print("📁 Creating organized folder structure...")
    folders_to_create = [
        'archive',
        'archive/old_scripts',
        'archive/old_docs',
        'archive/old_examples',
        'datasets',
        'experiments',
        'final_models'
    ]
    
    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)
        print(f"   ✅ Created: {folder}")
    
    # Archive old scripts
    print("\n📦 Archiving old scripts...")
    old_scripts = [
        ('config.py', 'archive/old_scripts/config.py'),
        ('train.py', 'archive/old_scripts/train.py'),
        ('train_improved.py', 'archive/old_scripts/train_improved.py'),
        ('test.py', 'archive/old_scripts/test.py'),
        ('show_results.py', 'archive/old_scripts/show_results.py'),
        ('visualize.py', 'archive/old_scripts/visualize.py'),
    ]
    
    for src, dst in old_scripts:
        if safe_move(src, dst):
            print(f"   ✅ Archived: {src}")
    
    # Archive old documentation
    print("\n📚 Archiving redundant documentation...")
    old_docs = [
        ('COMPLETE_PROJECT_GUIDE.md', 'archive/old_docs/COMPLETE_PROJECT_GUIDE.md'),
        ('PROJECT_SUMMARY.md', 'archive/old_docs/PROJECT_SUMMARY.md'),
        ('COMMAND_REFERENCE.md', 'archive/old_docs/COMMAND_REFERENCE.md'),
        ('train-log.txt', 'archive/old_docs/train-log.txt'),
    ]
    
    for src, dst in old_docs:
        if safe_move(src, dst):
            print(f"   ✅ Archived: {src}")
    
    # Archive example files
    print("\n🖼️  Archiving example files...")
    if os.path.exists('src'):
        if safe_move('src', 'archive/old_examples/src'):
            print(f"   ✅ Archived: src/ folder")
    
    if os.path.exists('example.png'):
        if safe_move('example.png', 'archive/old_examples/example.png'):
            print(f"   ✅ Archived: example.png")
    
    # Delete large temporary files
    print("\n🗑️  Deleting temporary files (freeing disk space)...")
    
    # Ask about results folder
    if os.path.exists('results'):
        size = get_folder_size('results')
        print(f"\n   results/ folder: {size:.1f} MB (training visualizations)")
        delete_results = input("   Delete results/ folder? (yes/no): ").strip().lower()
        if delete_results in ['yes', 'y']:
            if safe_delete('results'):
                print(f"   ✅ Deleted: results/ ({size:.1f} MB freed)")
        else:
            if safe_move('results', 'archive/results'):
                print(f"   ✅ Moved to archive: results/")
    
    # Ask about logs folder
    if os.path.exists('logs'):
        size = get_folder_size('logs')
        print(f"\n   logs/ folder: {size:.1f} MB (old tensorboard logs)")
        delete_logs = input("   Delete logs/ folder? (yes/no): ").strip().lower()
        if delete_logs in ['yes', 'y']:
            if safe_delete('logs'):
                print(f"   ✅ Deleted: logs/ ({size:.1f} MB freed)")
        else:
            if safe_move('logs', 'archive/logs'):
                print(f"   ✅ Moved to archive: logs/")
    
    # Copy best model to final_models
    print("\n💎 Preserving best model...")
    if os.path.exists('models/best_model.pth'):
        shutil.copy('models/best_model.pth', 'final_models/best_model_68_dice.pth')
        print("   ✅ Copied best_model.pth → final_models/best_model_68_dice.pth")
        print("   💡 Your trained model is safely backed up!")
    
    # Summary
    print("\n" + "="*70)
    print("✅ CLEANUP COMPLETE!")
    print("="*70 + "\n")
    
    print("📂 NEW PROJECT STRUCTURE:")
    print("""
    retina-unet-segmentation/
    ├── 🎯 MAIN CODE (USE THESE)
    │   ├── config_optimized.py      ← Best configuration
    │   ├── train_optimized.py       ← Best training script
    │   ├── inference.py             ← Make predictions
    │   ├── evaluate_results.py      ← Evaluate performance
    │   ├── download_datasets.py     ← Get more data
    │   ├── unet.py                  ← Model architecture
    │   ├── dataloader.py            ← Data loading
    │   └── utils.py                 ← Helper functions
    │
    ├── 📊 DATA & MODELS
    │   ├── Retina/                  ← Your dataset (100 images)
    │   ├── models/                  ← Trained models
    │   ├── final_models/            ← Production models (backed up)
    │   ├── predictions/             ← Latest predictions
    │   ├── datasets/                ← For new datasets (DRIVE, etc.)
    │   └── checkpoints/             ← Training checkpoints
    │
    ├── 📚 DOCUMENTATION
    │   ├── README.md                ← Project overview
    │   ├── QUICKSTART.md            ← Quick setup guide
    │   ├── IMPROVEMENT_PLAN.txt     ← This improvement plan
    │   └── RESULTS_SUMMARY.md       ← Current results
    │
    └── 📦 ARCHIVED (Old files - safe to delete later)
        └── archive/
            ├── old_scripts/         ← Old training scripts
            ├── old_docs/            ← Old documentation
            └── old_examples/        ← Example files
    """)
    
    print("\n" + "="*70)
    print("🚀 NEXT STEPS")
    print("="*70 + "\n")
    
    print("1️⃣  View the improvement plan:")
    print("   » notepad IMPROVEMENT_PLAN.txt")
    print("   (or just open it in VS Code)")
    print()
    print("2️⃣  Check optimized configuration:")
    print("   » python config_optimized.py")
    print()
    print("3️⃣  Start optimized training:")
    print("   » python train_optimized.py")
    print("   (This will take 60-90 minutes and improve to 75-82% Dice)")
    print()
    print("4️⃣  Evaluate new results:")
    print("   » python evaluate_results.py")
    print()
    print("\n💡 TIP: Read IMPROVEMENT_PLAN.txt for complete step-by-step guide!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
