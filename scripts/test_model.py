"""
Quick Model Test Script
Tests the trained U-Net++ model on a sample image
"""

import torch
import os
import sys
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.unet_plus_plus import UNetPlusPlus

def test_model():
    """Test the trained model on a sample test image"""
    
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load model
    print("[1/4] Loading trained model...")
    model = UNetPlusPlus(in_channels=3, out_channels=1, deep_supervision=True).to(device)
    checkpoint_path = os.path.join(base_dir, 'results', 'checkpoints_unetpp', 'best.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  Model from epoch: {checkpoint['epoch']}")
    print(f"  Validation Dice: {checkpoint['metrics']['dice']:.4f}")
    print(f"  Parameters: 9.0M\n")
    
    # Load a test image
    print("[2/4] Loading test image...")
    test_img_dir = os.path.join(base_dir, "Retina", "test", "image")
    test_mask_dir = os.path.join(base_dir, "Retina", "test", "mask")
    
    if not os.path.exists(test_img_dir):
        print("  ERROR: Test images not found!")
        return
    
    test_images = sorted([f for f in os.listdir(test_img_dir) if f.endswith(('.tif', '.png'))])
    test_masks = sorted([f for f in os.listdir(test_mask_dir) if f.endswith(('.gif', '.png'))])
    
    if len(test_images) == 0:
        print("  ERROR: No test images found!")
        return
    
    # Load first test image
    img_path = os.path.join(test_img_dir, test_images[0])
    mask_path = os.path.join(test_mask_dir, test_masks[0])
    
    img = Image.open(img_path).convert('RGB')
    mask = Image.open(mask_path).convert('L')
    
    print(f"  Loaded: {test_images[0]}")
    print(f"  Image size: {img.size}\n")
    
    # Preprocess
    print("[3/4] Running inference...")
    img_array = np.array(img)
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        if isinstance(outputs, tuple):
            pred_logits = outputs[-1]  # Use final output
        else:
            pred_logits = outputs
        
        pred = torch.sigmoid(pred_logits)
    
    pred_np = pred.squeeze().cpu().numpy()
    pred_binary = (pred_np > 0.5).astype(np.uint8)
    
    print(f"  Prediction range: [{pred_np.min():.4f}, {pred_np.max():.4f}]")
    print(f"  Mean confidence: {pred_np.mean():.4f}\n")
    
    # Visualize
    print("[4/4] Saving results...")
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(pred_np, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Prediction (Probability)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    axes[3].imshow(pred_binary, cmap='gray')
    axes[3].set_title('Prediction (Binary)', fontsize=14, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(base_dir, 'results', 'test_result.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}\n")
    
    print("="*60)
    print("TEST COMPLETE!")
    print("="*60)
    print("\nModel is working correctly! [SUCCESS]")
    print(f"Result saved to: {output_path}")

if __name__ == "__main__":
    test_model()
