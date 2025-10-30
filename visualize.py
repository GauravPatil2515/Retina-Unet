"""
Visualization utilities for Retina Blood Vessel Segmentation
Helps you visualize data, predictions, and training progress
"""

import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image
import os
import numpy as np
from dataloader import ImageDataset, PATH
from utils import uint8_to_float32

def visualize_dataset_samples(num_samples=4):
    """
    Display random samples from the training dataset
    Shows image and corresponding mask side by side
    """
    dataset = ImageDataset(path=os.path.join(PATH, "train"))
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))
    
    for i in range(num_samples):
        # Get random sample
        idx = np.random.randint(0, len(dataset))
        image, mask = dataset[idx]
        
        # Convert to numpy for display
        image_np = image.permute(1, 2, 0).numpy()
        mask_np = mask.numpy()
        
        # Display image
        axes[i, 0].imshow(image_np)
        axes[i, 0].set_title(f'Sample {idx} - Retina Image')
        axes[i, 0].axis('off')
        
        # Display mask
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title(f'Sample {idx} - Vessel Mask')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
    print("✅ Dataset samples saved to: dataset_samples.png")
    plt.show()

def visualize_prediction(image_path, mask_path, prediction_path):
    """
    Display original image, ground truth mask, and prediction
    """
    # Load images
    image = read_image(image_path)
    mask = read_image(mask_path)
    prediction = read_image(prediction_path)
    
    # Convert to numpy
    image_np = image.permute(1, 2, 0).numpy() / 255.0
    mask_np = mask[0].numpy() if mask.shape[0] > 1 else mask.squeeze().numpy()
    pred_np = prediction[0].numpy() if prediction.shape[0] > 1 else prediction.squeeze().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_np, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Comparison saved to: prediction_comparison.png")
    plt.show()

def visualize_overlay(image_path, prediction_path, alpha=0.5):
    """
    Create an overlay of prediction on original image
    """
    # Load images
    image = read_image(image_path)
    prediction = read_image(prediction_path)
    
    # Convert to numpy
    image_np = image.permute(1, 2, 0).numpy() / 255.0
    pred_np = prediction[0].numpy() / 255.0 if prediction.shape[0] > 1 else prediction.squeeze().numpy() / 255.0
    
    # Create overlay
    overlay = image_np.copy()
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], pred_np * alpha)  # Add red channel
    
    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(pred_np, cmap='hot')
    axes[1].set_title('Vessel Probability')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_overlay.png', dpi=150, bbox_inches='tight')
    print("✅ Overlay saved to: prediction_overlay.png")
    plt.show()

def plot_training_history(log_file='train_history.log'):
    """
    Plot training loss and metrics from log file
    (You need to save these values during training)
    """
    # This is a template - you'd need to save training history first
    epochs = range(1, 101)
    train_loss = np.random.exponential(0.3, 100)[::-1] * 0.5  # Dummy data
    val_loss = train_loss + np.random.normal(0, 0.02, 100)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(epochs, train_loss, label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice score plot
    dice_scores = 1 - train_loss
    axes[1].plot(epochs, dice_scores, label='Dice Score', color='green', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Validation Dice Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("✅ Training history saved to: training_history.png")
    plt.show()

def analyze_prediction_errors(image_path, mask_path, prediction_path):
    """
    Analyze where the model makes mistakes
    Shows: True Positives, False Positives, False Negatives
    """
    # Load images
    mask = read_image(mask_path)[0].numpy() / 255.0
    prediction = read_image(prediction_path)[0].numpy() / 255.0
    
    # Threshold to binary
    mask_binary = (mask > 0.5).astype(float)
    pred_binary = (prediction > 0.5).astype(float)
    
    # Calculate metrics
    tp = (mask_binary * pred_binary)  # True Positive: correctly predicted vessel
    fp = ((1 - mask_binary) * pred_binary)  # False Positive: predicted vessel where there is none
    fn = (mask_binary * (1 - pred_binary))  # False Negative: missed vessel
    
    # Create color-coded error map
    error_map = np.zeros((*mask_binary.shape, 3))
    error_map[:, :, 1] = tp  # Green for correct
    error_map[:, :, 0] = fp  # Red for false positive
    error_map[:, :, 2] = fn  # Blue for false negative
    
    # Calculate metrics
    precision = tp.sum() / (tp.sum() + fp.sum() + 1e-8)
    recall = tp.sum() / (tp.sum() + fn.sum() + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(mask_binary, cmap='gray')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    axes[1].imshow(pred_binary, cmap='gray')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    axes[2].imshow(error_map)
    axes[2].set_title(f'Error Analysis\nPrecision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}')
    axes[2].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='True Positive'),
        Patch(facecolor='red', label='False Positive'),
        Patch(facecolor='blue', label='False Negative')
    ]
    axes[2].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('error_analysis.png', dpi=150, bbox_inches='tight')
    print("✅ Error analysis saved to: error_analysis.png")
    plt.show()

def visualize_model_architecture():
    """
    Print model architecture summary
    """
    from unet import Unet
    
    model = Unet(3, 2)
    
    print("\n" + "="*60)
    print("U-NET MODEL ARCHITECTURE")
    print("="*60)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Model Size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    print("\n" + "-"*60)
    print("Layer Structure:")
    print("-"*60)
    
    for name, module in model.named_children():
        print(f"{name}: {module.__class__.__name__}")
    
    print("="*60 + "\n")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualization Tools')
    parser.add_argument('--action', type=str, required=True,
                        choices=['dataset', 'prediction', 'overlay', 'error', 'architecture'],
                        help='What to visualize')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--mask', type=str, help='Path to mask file')
    parser.add_argument('--prediction', type=str, help='Path to prediction file')
    
    args = parser.parse_args()
    
    if args.action == 'dataset':
        print("📊 Visualizing dataset samples...")
        visualize_dataset_samples()
    
    elif args.action == 'prediction':
        if not all([args.image, args.mask, args.prediction]):
            print("❌ Need --image, --mask, and --prediction paths")
        else:
            print("📊 Visualizing prediction comparison...")
            visualize_prediction(args.image, args.mask, args.prediction)
    
    elif args.action == 'overlay':
        if not all([args.image, args.prediction]):
            print("❌ Need --image and --prediction paths")
        else:
            print("📊 Creating overlay visualization...")
            visualize_overlay(args.image, args.prediction)
    
    elif args.action == 'error':
        if not all([args.image, args.mask, args.prediction]):
            print("❌ Need --image, --mask, and --prediction paths")
        else:
            print("📊 Analyzing prediction errors...")
            analyze_prediction_errors(args.image, args.mask, args.prediction)
    
    elif args.action == 'architecture':
        visualize_model_architecture()
