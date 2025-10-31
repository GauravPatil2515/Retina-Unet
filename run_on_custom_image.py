"""
Run U-Net++ model on a custom retina image
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from models.unet_plus_plus import UNetPlusPlus
import sys
import os

def predict_on_image(image_path, output_path='custom_prediction.png'):
    """Run inference on a custom image"""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Input image: {image_path}\n")
    
    # Load model
    print("[1/4] Loading trained U-Net++ model...")
    model = UNetPlusPlus(in_channels=3, out_channels=1, deep_supervision=True).to(device)
    
    checkpoint_path = 'results/checkpoints_unetpp/best.pth'
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Model checkpoint not found at {checkpoint_path}")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  ✓ Model loaded from epoch {checkpoint['epoch']}")
    print(f"  ✓ Validation Dice: {checkpoint['metrics']['dice']:.4f}\n")
    
    # Load and preprocess image
    print("[2/4] Loading and preprocessing image...")
    img = Image.open(image_path).convert('RGB')
    original_size = img.size
    print(f"  ✓ Image size: {original_size}")
    
    # Convert to array and normalize
    img_array = np.array(img)
    img_normalized = img_array.astype(np.float32) / 255.0
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    print(f"  ✓ Tensor shape: {img_tensor.shape}\n")
    
    # Run inference
    print("[3/4] Running inference...")
    with torch.no_grad():
        outputs = model(img_tensor)
        if isinstance(outputs, tuple):
            pred_logits = outputs[-1]  # Use final output from deep supervision
        else:
            pred_logits = outputs
        
        # Apply sigmoid to get probabilities
        pred_prob = torch.sigmoid(pred_logits)
    
    # Convert to numpy
    pred_prob_np = pred_prob.squeeze().cpu().numpy()
    pred_binary = (pred_prob_np > 0.5).astype(np.uint8)
    
    print(f"  ✓ Prediction range: [{pred_prob_np.min():.4f}, {pred_prob_np.max():.4f}]")
    print(f"  ✓ Mean probability: {pred_prob_np.mean():.4f}")
    print(f"  ✓ Vessel pixels detected: {pred_binary.sum():,} ({100*pred_binary.sum()/pred_binary.size:.2f}%)\n")
    
    # Visualize results
    print("[4/4] Saving results...")
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original Retina Image', fontsize=16, fontweight='bold')
    axes[0].axis('off')
    
    # Probability map
    im1 = axes[1].imshow(pred_prob_np, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Vessel Probability Map', fontsize=16, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Binary segmentation
    axes[2].imshow(pred_binary, cmap='gray')
    axes[2].set_title('Binary Vessel Segmentation', fontsize=16, fontweight='bold')
    axes[2].axis('off')
    
    # Overlay
    overlay = img_array.copy()
    # Create colored mask (red for vessels)
    vessel_mask = pred_prob_np > 0.5
    overlay[vessel_mask] = overlay[vessel_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
    axes[3].imshow(overlay.astype(np.uint8))
    axes[3].set_title('Overlay (Red = Vessels)', fontsize=16, fontweight='bold')
    axes[3].axis('off')
    
    plt.suptitle(f'U-Net++ Blood Vessel Segmentation\n'
                 f'Model: Best checkpoint (Epoch {checkpoint["epoch"]}, Dice: {checkpoint["metrics"]["dice"]:.4f})',
                 fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Results saved to: {output_path}\n")
    
    # Also save individual outputs
    prob_output = output_path.replace('.png', '_probability.png')
    binary_output = output_path.replace('.png', '_binary.png')
    overlay_output = output_path.replace('.png', '_overlay.png')
    
    plt.imsave(prob_output, pred_prob_np, cmap='hot', vmin=0, vmax=1)
    plt.imsave(binary_output, pred_binary, cmap='gray')
    plt.imsave(overlay_output, overlay.astype(np.uint8))
    
    print(f"  ✓ Probability map: {prob_output}")
    print(f"  ✓ Binary mask: {binary_output}")
    print(f"  ✓ Overlay: {overlay_output}")
    
    print("\n" + "="*70)
    print("SEGMENTATION COMPLETE! [SUCCESS]")
    print("="*70)
    print(f"\nVessel Coverage: {100*pred_binary.sum()/pred_binary.size:.2f}%")
    print(f"Confidence: {pred_prob_np.mean():.4f}")
    
    return pred_prob_np, pred_binary

if __name__ == "__main__":
    # Check if image path is provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else 'custom_prediction.png'
    else:
        # Look for uploaded image in current directory
        possible_names = ['image.jpg', 'image.png', 'retina.jpg', 'retina.png', 
                         'input.jpg', 'input.png', 'test_image.jpg', 'test_image.png']
        
        image_path = None
        for name in possible_names:
            if os.path.exists(name):
                image_path = name
                break
        
        if image_path is None:
            print("ERROR: No image found!")
            print("Usage: python run_on_custom_image.py <image_path> [output_path]")
            sys.exit(1)
        
        output_path = 'custom_prediction.png'
    
    predict_on_image(image_path, output_path)
