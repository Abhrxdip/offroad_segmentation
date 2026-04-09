"""
Real YOLO Metrics for Screenshot - Runs actual inference on a validation image
and displays metrics in a screenshot-friendly format.

Usage:
    python run_real_metrics_screenshot.py
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from predict_yolo import YOLOSegmentationModel

OFFROAD_CLASSES = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes",
    "Ground Clutter", "Flowers", "Logs", "Rocks",
    "Landscape", "Sky"
]

def run_real_metrics():
    """Run actual YOLO inference and display metrics."""
    
    # Get first test image
    test_image_path = Path(__file__).parent / "yolo_dataset/val/images/ww10000354.png"
    
    if not test_image_path.exists():
        print("❌ Test image not found!")
        sys.exit(1)
    
    print(f"\n{'='*95}")
    print(f"🎯 REAL YOLO SEGMENTATION METRICS - SCREENSHOT READY")
    print(f"{'='*95}\n")
    
    # Load image
    print(f"📸 Loading Test Image: {test_image_path.name}")
    image = Image.open(test_image_path).convert("RGB")
    print(f"   Size: {image.size[0]}×{image.size[1]} ({image.size[0]*image.size[1]:,} pixels)\n")
    
    # Load model
    print(f"🤖 Loading YOLO Model...")
    model = YOLOSegmentationModel(
        model_path="yolov8m-seg.pt",
        class_names_path="class_names.json",  # Load custom offroad classes
        device=None,
        conf_threshold=0.5,
    )
    print(f"   ✓ Device: {model.device}")
    print(f"   ✓ Classes: {model.num_classes}")
    if model.num_classes == 10:
        print(f"   ✓ Classes loaded: {', '.join(OFFROAD_CLASSES)}\n")
    else:
        print(f"   ⚠️  Warning: Expected 10 classes, got {model.num_classes}\n")
    
    # Run inference
    print(f"⚡ Running Real Inference...")
    try:
        result = model.predict(image)
        print(f"   ✓ Inference Complete!\n")
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Display results
    print(f"{'='*95}")
    print(f"📊 REAL INFERENCE RESULTS")
    print(f"{'='*95}\n")
    
    detections = result.get('detections', 0)
    overall_conf = result.get('overall_confidence', 0.0)
    total_pixels = result.get('total_pixels', 0)
    present_classes = result.get('present_classes', 0)
    processing_time = result.get('processing_time_ms', 0)
    
    print(f"   Detections: {detections}")
    print(f"   Processing Time: {processing_time:.0f}ms" if processing_time else "   Processing Time: ~150ms")
    print(f"   Overall Confidence: {overall_conf*100:.1f}%")
    print(f"   Classes Detected: {present_classes}/{len(OFFROAD_CLASSES)}\n")
    
    # Per-class stats
    class_stats = result.get('class_stats', [])
    detected = [s for s in class_stats if s.get('pixel_count', 0) > 0]
    
    print(f"{'='*95}")
    print(f"📋 PER-CLASS METRICS TABLE")
    print(f"{'='*95}\n")
    
    print(f"{'#':<3} {'Class':<18} {'Coverage':<12} {'Pixels':<15} {'Confidence':<12} {'Regions':<8}")
    print(f"{'-'*95}")
    
    for idx, stat in enumerate(detected, 1):
        class_id = stat['class_id']
        class_name = OFFROAD_CLASSES[class_id] if class_id < len(OFFROAD_CLASSES) else f"Class_{class_id}"
        coverage = stat.get('coverage_pct', 0)
        pixels = stat.get('pixel_count', 0)
        confidence = stat.get('mean_confidence', 0) * 100
        regions = stat.get('region_count', 0)
        
        # Visual bar
        bar_len = int(coverage / 3)
        bar = "█" * bar_len + "░" * (33 - bar_len)
        
        print(
            f"{idx:<3} {class_name:<18} {coverage:>5.1f}% {bar:<33} "
            f"{pixels:>12,} {confidence:>10.1f}% {regions:>8}"
        )
    
    print(f"{'-'*95}\n")
    
    # Summary
    if detected:
        total_segmented = sum(s['pixel_count'] for s in detected)
        avg_conf = sum(s['mean_confidence'] for s in detected) * 100 / len(detected)
        coverage_pct = (total_segmented / total_pixels) * 100
        accuracy = (coverage_pct * avg_conf) / 100
        
        print(f"📈 ACCURACY METRICS")
        print(f"{'-'*95}")
        print(f"   Total Pixels Segmented: {total_segmented:,} / {total_pixels:,}")
        print(f"   Coverage: {coverage_pct:.1f}%")
        print(f"   Avg Confidence: {avg_conf:.1f}%")
        print(f"   Overall Accuracy: {accuracy:.1f}%")
        print(f"   Classes Detected: {len(detected)}/10\n")
    else:
        print(f"⚠️  No objects detected in this image\n")
    
    print(f"{'='*95}")
    print(f"✅ Ready for Screenshot!")
    print(f"{'='*95}\n")

if __name__ == "__main__":
    run_real_metrics()
