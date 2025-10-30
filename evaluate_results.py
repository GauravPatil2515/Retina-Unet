"""
Evaluate model predictions on test set
Calculate metrics: Dice, IoU, Accuracy, Precision, Recall, F1
"""

import os
import cv2
import numpy as np
from pathlib import Path

def load_mask(path):
    """Load mask as binary"""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {path}")
    return (mask > 128).astype(np.uint8)

def calculate_metrics(pred_mask, gt_mask):
    """Calculate segmentation metrics"""
    pred = pred_mask.flatten()
    gt = gt_mask.flatten()
    
    # True Positives, False Positives, True Negatives, False Negatives
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    tn = np.sum((pred == 0) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    
    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Dice coefficient
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    # IoU (Jaccard Index)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'dice': dice,
        'iou': iou
    }

def evaluate_predictions(gt_dir, pred_dir):
    """Evaluate all predictions against ground truth"""
    
    print("\n" + "="*70)
    print("🔬 RETINA VESSEL SEGMENTATION - TEST SET EVALUATION")
    print("="*70 + "\n")
    
    # Get all prediction files
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.startswith('pred_') and f.endswith('.png')])
    
    if not pred_files:
        print("❌ No prediction files found!")
        return
    
    print(f"📊 Found {len(pred_files)} predictions to evaluate\n")
    
    all_metrics = []
    
    for idx, pred_file in enumerate(pred_files, 1):
        # Get corresponding ground truth file
        img_name = pred_file.replace('pred_', '')
        gt_path = os.path.join(gt_dir, img_name)
        pred_path = os.path.join(pred_dir, pred_file)
        
        if not os.path.exists(gt_path):
            print(f"⚠️  [{idx}/{len(pred_files)}] Skipping {img_name} - no ground truth found")
            continue
        
        # Load masks
        gt_mask = load_mask(gt_path)
        pred_mask = load_mask(pred_path)
        
        # Calculate metrics
        metrics = calculate_metrics(pred_mask, gt_mask)
        all_metrics.append(metrics)
        
        # Print individual results
        print(f"[{idx}/{len(pred_files)}] {img_name:15s} | "
              f"Dice: {metrics['dice']:.4f} | "
              f"IoU: {metrics['iou']:.4f} | "
              f"Acc: {metrics['accuracy']:.4f} | "
              f"F1: {metrics['f1']:.4f}")
    
    # Calculate average metrics
    if all_metrics:
        print("\n" + "="*70)
        print("📈 AVERAGE METRICS ON TEST SET")
        print("="*70 + "\n")
        
        avg_metrics = {
            key: np.mean([m[key] for m in all_metrics])
            for key in all_metrics[0].keys()
        }
        
        print(f"🎯 Dice Coefficient:  {avg_metrics['dice']:.4f}  (higher is better)")
        print(f"🎯 IoU (Jaccard):     {avg_metrics['iou']:.4f}  (higher is better)")
        print(f"🎯 Accuracy:          {avg_metrics['accuracy']:.4f}  (higher is better)")
        print(f"🎯 Precision:         {avg_metrics['precision']:.4f}  (higher is better)")
        print(f"🎯 Recall:            {avg_metrics['recall']:.4f}  (higher is better)")
        print(f"🎯 F1 Score:          {avg_metrics['f1']:.4f}  (higher is better)")
        
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETE!")
        print("="*70 + "\n")
        
        # Performance assessment
        print("📊 PERFORMANCE ASSESSMENT:")
        if avg_metrics['dice'] >= 0.80:
            print("   🌟 EXCELLENT - Model is performing very well!")
        elif avg_metrics['dice'] >= 0.70:
            print("   ✅ GOOD - Model shows strong performance")
        elif avg_metrics['dice'] >= 0.60:
            print("   👍 FAIR - Model is learning but has room for improvement")
        elif avg_metrics['dice'] >= 0.50:
            print("   ⚠️  MODERATE - Consider training longer or adjusting parameters")
        else:
            print("   ❌ NEEDS IMPROVEMENT - Try different loss function or architecture tweaks")
        
        print("\n💡 RECOMMENDATIONS:")
        if avg_metrics['dice'] < 0.70:
            print("   • Train for more epochs (200-300)")
            print("   • Try combined loss (Dice + CrossEntropy)")
            print("   • Lower learning rate to 0.0001")
            print("   • Add more training data if available")
        else:
            print("   • Model is performing well!")
            print("   • Consider fine-tuning for specific use cases")
            print("   • Test on external datasets for generalization")
        
        print()

if __name__ == "__main__":
    GT_DIR = "Retina/test/mask"
    PRED_DIR = "predictions"
    
    evaluate_predictions(GT_DIR, PRED_DIR)
