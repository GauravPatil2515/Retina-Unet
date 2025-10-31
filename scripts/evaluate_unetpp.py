"""
Evaluation script for U-Net++ model
Load trained model and evaluate on test set with comprehensive metrics
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_plus_plus import UNetPlusPlus
from scripts.dataloader_unetpp import FullImageDataset
from models.losses_unetpp import calculate_metrics


def calculate_metrics_from_probs(pred, target, threshold=0.5):
    """
    Calculate metrics from probabilities (not logits)
    
    Args:
        pred: Predicted probabilities [B, 1, H, W] (already sigmoid applied)
        target: Ground truth binary mask [B, 1, H, W]
    """
    # Convert to numpy
    pred_np = pred.cpu().detach().numpy().flatten()
    target_np = target.cpu().detach().numpy().flatten()
    
    # Binarize predictions
    pred_binary = (pred_np > threshold).astype(np.float32)
    
    # Calculate confusion matrix elements
    TP = np.sum((pred_binary == 1) & (target_np == 1))
    TN = np.sum((pred_binary == 0) & (target_np == 0))
    FP = np.sum((pred_binary == 1) & (target_np == 0))
    FN = np.sum((pred_binary == 0) & (target_np == 1))
    
    # Calculate metrics
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    sensitivity = TP / (TP + FN + 1e-6)
    specificity = TN / (TN + FP + 1e-6)
    
    # Calculate AUC
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(target_np, pred_np)
    except:
        auc = 0.0
    
    return {
        'dice': dice,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc': auc,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }


def reconstruct_from_patches(image, model, device, patch_size=128, stride=64):
    """
    Reconstruct full-size prediction from patches
    Uses weighted averaging for overlapping regions
    
    Args:
        image: Full-size image tensor [C, H, W]
        model: Trained model
        device: torch device
        patch_size: Size of patches
        stride: Stride for patch extraction
    
    Returns:
        Reconstructed prediction [1, H, W]
    """
    C, H, W = image.shape
    
    # Create output tensors
    reconstructed = torch.zeros(1, H, W, dtype=torch.float32)
    count = torch.zeros(1, H, W, dtype=torch.float32)
    
    # Extract and predict patches
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            # Extract patch
            patch = image[:, y:y+patch_size, x:x+patch_size].unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(patch)
                if isinstance(outputs, tuple):
                    pred = outputs[-1]  # Use final output (logits)
                else:
                    pred = outputs  # logits
                
                # Apply sigmoid to get probabilities for reconstruction
                pred = torch.sigmoid(pred)
            
            # Add to reconstruction
            pred = pred.squeeze(0).cpu()
            reconstructed[:, y:y+patch_size, x:x+patch_size] += pred
            count[:, y:y+patch_size, x:x+patch_size] += 1
    
    # Average overlapping regions
    reconstructed = reconstructed / (count + 1e-6)
    
    return reconstructed


def evaluate_model(model_path, test_img_dir, test_mask_dir, device, save_dir='evaluation_results'):
    """
    Evaluate trained model on test set
    
    Args:
        model_path: Path to trained model checkpoint
        test_img_dir: Test images directory
        test_mask_dir: Test masks directory
        device: torch device
        save_dir: Directory to save evaluation results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    print("\n[LOAD] Loading trained model...")
    model = UNetPlusPlus(in_channels=3, out_channels=1, deep_supervision=True).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"[OK] Model loaded from: {model_path}")
    print(f"[OK] Trained for {checkpoint['epoch']} epochs")
    print(f"[OK] Best Val Dice: {checkpoint['metrics'].get('dice', 'N/A')}")
    
    # Load test dataset
    print("\n[LOAD] Loading test dataset...")
    test_dataset = FullImageDataset(test_img_dir, test_mask_dir)
    print(f"[OK] Test images: {len(test_dataset)}")
    
    # Evaluate each image
    all_metrics = []
    predictions = []
    
    print("\n[EVAL] Evaluating model...")
    for idx in tqdm(range(len(test_dataset))):
        image, mask = test_dataset[idx]
        
        # Reconstruct prediction from patches
        pred = reconstruct_from_patches(image, model, device, patch_size=128, stride=64)
        
        # Calculate metrics - move tensors to device
        pred_tensor = pred.unsqueeze(0).to(device)
        mask_tensor = mask.unsqueeze(0).to(device)
        
        # Debug: Check prediction range
        if idx == 0:
            print(f"\nDebug Info (First Image):")
            print(f"  Prediction range: [{pred.min():.4f}, {pred.max():.4f}]")
            print(f"  Mask range: [{mask.min():.4f}, {mask.max():.4f}]")
            print(f"  Mean prediction: {pred.mean():.4f}")
            print(f"  Mean mask: {mask.mean():.4f}")
        
        metrics = calculate_metrics_from_probs(pred_tensor, mask_tensor)
        
        all_metrics.append(metrics)
        predictions.append({
            'image': image.cpu().numpy(),
            'mask': mask.cpu().numpy(),
            'pred': pred.cpu().numpy()
        })
    
    # Calculate average metrics
    avg_metrics = {
        'dice': np.mean([m['dice'] for m in all_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
        'sensitivity': np.mean([m['sensitivity'] for m in all_metrics]),
        'specificity': np.mean([m['specificity'] for m in all_metrics]),
        'auc': np.mean([m['auc'] for m in all_metrics])
    }
    
    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\nAverage Metrics on {len(test_dataset)} test images:")
    print(f"  Dice Coefficient: {avg_metrics['dice']:.4f} ({avg_metrics['dice']*100:.2f}%)")
    print(f"  Accuracy:         {avg_metrics['accuracy']:.4f} ({avg_metrics['accuracy']*100:.2f}%)")
    print(f"  Sensitivity:      {avg_metrics['sensitivity']:.4f} ({avg_metrics['sensitivity']*100:.2f}%)")
    print(f"  Specificity:      {avg_metrics['specificity']:.4f} ({avg_metrics['specificity']*100:.2f}%)")
    print(f"  AUC:              {avg_metrics['auc']:.4f}")
    print("="*80 + "\n")
    
    # Save visualizations
    print("[SAVE] Saving visualizations...")
    visualize_predictions(predictions[:5], save_dir)  # Save first 5 examples
    
    # Save metrics
    import json
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    metrics_to_save = {
        'average': avg_metrics,
        'per_image': all_metrics
    }
    metrics_to_save = convert_to_python_types(metrics_to_save)
    
    with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    
    print(f"[OK] Results saved to: {save_dir}")
    
    return avg_metrics


def visualize_predictions(predictions, save_dir):
    """Visualize predictions vs ground truth"""
    
    for idx, pred_dict in enumerate(predictions):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        img = pred_dict['image'].transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth
        mask = pred_dict['mask'][0]  # [1, H, W] -> [H, W]
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Prediction
        pred = pred_dict['pred'][0]  # [1, H, W] -> [H, W]
        axes[2].imshow(pred, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'prediction_{idx+1}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"[OK] Saved {len(predictions)} prediction visualizations")


if __name__ == "__main__":
    # Configuration
    CHECKPOINT_PATH = "checkpoints_unetpp/best.pth"
    DATA_ROOT = os.path.join(os.getcwd(), "Retina")
    TEST_IMG_DIR = os.path.join(DATA_ROOT, "test", "image")
    TEST_MASK_DIR = os.path.join(DATA_ROOT, "test", "mask")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDevice: {DEVICE}")
    
    # Evaluate
    if os.path.exists(CHECKPOINT_PATH):
        metrics = evaluate_model(
            CHECKPOINT_PATH,
            TEST_IMG_DIR,
            TEST_MASK_DIR,
            DEVICE,
            save_dir='evaluation_results_unetpp'
        )
    else:
        print(f"\n[ERROR] Checkpoint not found: {CHECKPOINT_PATH}")
        print("[INFO] Please train the model first using: python train_unetpp.py")
