"""
Quick test script to verify YOLO inference is working locally.
Run this to debug if the model is properly loaded and generating masks.

Usage:
    python test_inference_local.py <path_to_image>

Example:
    python test_inference_local.py complete\ dataset/Offroad_Segmentation_testImages/ADE_test_00026646.jpg
"""

import sys
import os
from pathlib import Path
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from predict_yolo import YOLOSegmentationModel

def test_inference():
    """Test YOLO inference on a test image."""
    
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Try to find a test image
        test_dirs = [
            "complete dataset/Offroad_Segmentation_testImages",
            "yolo_dataset/val/images",
        ]
        image_path = None
        for test_dir in test_dirs:
            test_dir_full = Path(__file__).parent / test_dir
            if test_dir_full.exists():
                images = list(test_dir_full.glob("*.jpg")) + list(test_dir_full.glob("*.png"))
                if images:
                    image_path = str(images[0])
                    break
    
    if not image_path or not Path(image_path).exists():
        print("❌ ERROR: No test image found!")
        print("\nUsage: python test_inference_local.py <path_to_image>")
        print("\nExample images:")
        print("  python test_inference_local.py 'complete dataset/Offroad_Segmentation_testImages/ADE_test_00026646.jpg'")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"🧪 YOLO Inference Test")
    print(f"{'='*70}")
    
    # Load image
    print(f"\n📁 Loading image: {image_path}")
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"   ✓ Image size: {image.size[0]}x{image.size[1]}")
    except Exception as e:
        print(f"   ❌ Failed to load image: {e}")
        sys.exit(1)
    
    # Load model
    print(f"\n🤖 Loading YOLO model...")
    try:
        model = YOLOSegmentationModel(
            model_path="yolov8m-seg.pt",
            device=None,  # Auto-detect
            conf_threshold=0.5,
        )
        print(f"   ✓ Model loaded on device: {model.device}")
        print(f"   ✓ Number of classes: {model.num_classes}")
        print(f"   ✓ Classes: {model.class_names}")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Run inference
    print(f"\n⚡ Running inference...")
    try:
        result = model.predict(image)
        print(f"   ✓ Inference complete!")
    except Exception as e:
        print(f"   ❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Analyze results
    print(f"\n📊 Results Analysis:")
    print(f"   • Total pixels: {result['total_pixels']:,}")
    print(f"   • Detections: {result['detections']}")
    print(f"   • Overall confidence: {result['overall_confidence']:.2%}")
    print(f"   • Present classes: {result['present_classes']}")
    
    # Show class statistics
    class_stats = result.get('class_stats', [])
    print(f"\n📋 Per-Class Statistics:")
    print(f"   {' '*8}{'Class':<20} {'Coverage':<10} {'Pixels':<12} {'Confidence':<12}")
    print(f"   {'-'*60}")
    
    detected_count = 0
    for stat in class_stats:
        if stat['pixel_count'] > 0:
            detected_count += 1
            print(
                f"   {stat['class_id']:2d}. {stat['class_name']:<18} "
                f"{stat['coverage_pct']:>6.2f}% {stat['pixel_count']:>10,} "
                f"{stat['mean_confidence']*100:>10.1f}%"
            )
    
    print(f"   {'-'*60}")
    print(f"   {'Total Detected:':>36} {detected_count} classes")
    
    # Check for issues
    print(f"\n✅ Diagnostics:")
    if detected_count == 0:
        print(f"   ⚠️  WARNING: No classes detected! Possible issues:")
        print(f"      - Confidence threshold too high (current: {model.conf_threshold})")
        print(f"      - Model not trained on this dataset")
        print(f"      - Image doesn't contain relevant objects")
    else:
        print(f"   ✓ Model is detecting {detected_count} classes")
        print(f"   ✓ Average confidence: {result['overall_confidence']*100:.1f}%")
        print(f"   ✓ Coverage: {sum(s['coverage_pct'] for s in class_stats if s['pixel_count'] > 0):.1f}%")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    test_inference()
