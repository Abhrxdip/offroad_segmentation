"""
YOLO Segmentation Model - Offroad Semantic Segmentation
Uses YOLOv8 segmentation model for end-to-end instance+semantic segmentation.

Author: Duality AI
Purpose: Fast YOLO-based segmentation for offroad environments
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics package required. Install with: pip install ultralytics")


# ============================================================================
# CONFIG & CONSTANTS
# ============================================================================

DEFAULT_CLASS_NAMES = [
    "Trees",
    "Lush Bushes",
    "Dry Grass",
    "Dry Bushes",
    "Ground Clutter",
    "Flowers",
    "Logs",
    "Rocks",
    "Landscape",
    "Sky",
]

DEFAULT_COLOR_PALETTE = np.array([
    [34, 139, 34],      # Trees - Dark Green
    [0, 200, 70],       # Lush Bushes - Bright Green
    [210, 180, 140],    # Dry Grass - Tan
    [139, 90, 43],      # Dry Bushes - Brown
    [120, 120, 20],     # Ground Clutter - Olive
    [255, 120, 180],    # Flowers - Pink
    [139, 69, 19],      # Logs - Saddle Brown
    [128, 128, 128],    # Rocks - Gray
    [160, 82, 45],      # Landscape - Sienna
    [135, 206, 235],    # Sky - Sky Blue
], dtype=np.uint8)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def build_color_palette(num_classes: int) -> np.ndarray:
    """Build color palette for visualization."""
    if num_classes <= len(DEFAULT_COLOR_PALETTE):
        return DEFAULT_COLOR_PALETTE[:num_classes]

    rng = np.random.default_rng(42)
    extra = rng.integers(0, 255, size=(num_classes - len(DEFAULT_COLOR_PALETTE), 3), dtype=np.uint8)
    return np.vstack([DEFAULT_COLOR_PALETTE, extra])


def resolve_class_names(class_names_path: Optional[str], num_classes: int) -> List[str]:
    """Load class names from JSON file if provided, else use defaults."""
    if class_names_path and os.path.exists(class_names_path):
        try:
            with open(class_names_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                loaded = [str(x) for x in data]
            elif isinstance(data, dict):
                loaded = [str(data.get(str(i), f"Class_{i}")) for i in range(num_classes)]
            else:
                loaded = []

            if len(loaded) == num_classes:
                return loaded
        except Exception as e:
            print(f"Warning: Failed to load class names from {class_names_path}: {e}")

    if num_classes == len(DEFAULT_CLASS_NAMES):
        return DEFAULT_CLASS_NAMES
    return [f"Class_{i}" for i in range(num_classes)]


def mask_to_color(mask: np.ndarray, color_palette: np.ndarray) -> np.ndarray:
    """Convert class mask to RGB color visualization."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(len(color_palette)):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask


def overlay_mask(rgb: np.ndarray, color_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend color mask with original RGB image."""
    return cv2.addWeighted(rgb, 1.0 - alpha, color_mask, alpha, 0.0)


# ============================================================================
# MODEL LOADER
# ============================================================================

class YOLOSegmentationModel:
    """
    Wrapper for YOLOv8 segmentation model.
    End-to-end instance and semantic segmentation.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8m-seg.pt",
        class_names_path: Optional[str] = None,
        device: Optional[str] = None,
        conf_threshold: float = 0.5,
    ):
        """
        Initialize YOLO model.
        
        Args:
            model_path: Path to YOLOv8 segmentation model or model name
            class_names_path: Optional path to JSON file with class names
            device: Device to load model on ('cuda', 'cpu'). Auto-detects if None.
            conf_threshold: Confidence threshold for predictions
        
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        
        print(f"Loading YOLO model: {model_path} on device: {self.device}")
        self.model = YOLO(model_path).to(self.device)
        
        # Get number of classes from model
        self.num_classes = len(self.model.names)
        
        # Load class names and color palette
        self.class_names = resolve_class_names(class_names_path, self.num_classes)
        self.color_palette = build_color_palette(self.num_classes)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference on image using YOLO.
        
        Args:
            image: PIL Image (RGB or RGBA)
        
        Returns:
            {
                "mask": np.ndarray (h, w) - semantic mask with class indices,
                "color_mask": np.ndarray (h, w, 3) - RGB visualization,
                "overlay": np.ndarray (h, w, 3) - blended with original,
                "coverage": List[(class_name, percentage), ...],
                "class_distribution": {class_name: percentage, ...},
                "class_stats": List[dict],
                "class_counts": {class_name: pixel_count, ...},
                "detections": int - number of objects detected
            }
        """
        # Convert to RGB if needed
        image_rgb = image.convert("RGB")
        original = np.array(image_rgb)
        
        # Run YOLO inference with segmentation enabled
        print(f"[PREDICT] Running inference with confidence threshold: {self.conf_threshold}")
        results = self.model(image_rgb, conf=self.conf_threshold, verbose=False, task='segment')
        result = results[0]
        
        print(f"[PREDICT] Got result - Detections: {len(result.boxes)}, Has masks: {result.masks is not None}")
        
        h, w = original.shape[:2]
        semantic_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Build semantic mask from instance masks
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            print(f"[PREDICT] Processing {len(masks)} masks")
            
            for i, (mask, class_id) in enumerate(zip(masks, class_ids)):
                # Resize mask to original image size
                mask_resized = cv2.resize(
                    mask.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_LINEAR
                )
                semantic_mask[mask_resized > 0.5] = class_id
        else:
            print("[PREDICT] WARNING: No masks returned from YOLO!")
        
        # Generate visualizations
        color_mask = mask_to_color(semantic_mask, self.color_palette)
        overlay = overlay_mask(original, color_mask)
        
        # Compute class coverage statistics
        coverage = []
        class_dist = {}
        class_counts = {}
        total_pixels = h * w
        
        for class_id in range(self.num_classes):
            pixel_count = np.sum(semantic_mask == class_id)
            if pixel_count == 0:
                continue
            percentage = (pixel_count / total_pixels) * 100
            class_name = self.class_names[class_id]
            coverage.append((class_name, percentage))
            class_dist[class_name] = percentage
            class_counts[class_name] = int(pixel_count)
        
        # Sort by coverage (descending)
        coverage.sort(key=lambda x: x[1], reverse=True)
        
        # Build stats - include ALL classes for transparency
        class_stats = []
        for class_id in range(self.num_classes):
            class_mask = semantic_mask == class_id
            pixel_count = int(np.sum(class_mask))
            coverage_pct = float((pixel_count / total_pixels) * 100)
            
            # Calculate mean confidence for this class
            class_confidences = []
            if len(result.boxes.conf) > 0:
                for i, pred_class in enumerate(result.boxes.cls):
                    if int(pred_class.item()) == class_id:
                        class_confidences.append(float(result.boxes.conf[i].item()))
            
            mean_confidence = float(np.mean(class_confidences)) if class_confidences else 0.0
            
            # Count separate regions for this class
            if pixel_count > 0:
                labeled_regions, region_count_val = cv2.connectedComponents(class_mask.astype(np.uint8))
                region_count = max(0, int(region_count_val) - 1)  # Subtract 1 for background
            else:
                region_count = 0
            
            class_stats.append({
                "class_id": class_id,
                "class_name": self.class_names[class_id],
                "pixel_count": pixel_count,
                "coverage_pct": coverage_pct,
                "mean_confidence": mean_confidence,
                "region_count": region_count
            })
        
        # Compute overall confidence
        overall_confidence = float(result.boxes.conf.mean().item()) if len(result.boxes.conf) > 0 else 0.0
        
        # Compute present classes
        present_classes = int(np.sum(np.array([len(np.where(semantic_mask == i)[0]) for i in range(self.num_classes)]) > 0))
        
        print(f"[PREDICT] Classes detected: {present_classes}, Overall confidence: {overall_confidence:.2f}")
        
        return {
            "mask": semantic_mask,
            "color_mask": color_mask,
            "overlay": overlay,
            "coverage": coverage,
            "class_distribution": class_dist,
            "class_stats": class_stats,  # Return all stats, not filtered
            "class_counts": class_counts,
            "class_confidence": {self.class_names[i]: float(overall_confidence) for i in range(self.num_classes)},
            "detections": len(result.boxes),
            "overall_confidence": overall_confidence,
            "present_classes": present_classes,
            "total_pixels": int(h * w),
        }
    
    @staticmethod
    def image_to_base64(image: np.ndarray) -> str:
        """Convert numpy image to base64 string for JSON serialization."""
        import base64
        _, buffer = cv2.imencode('.png', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    @staticmethod
    def save_results(output_dir: str, prefix: str, results: Dict[str, Any], original_image: np.ndarray):
        """Save prediction results to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(os.path.join(output_dir, f"{prefix}_original.png"), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f"{prefix}_mask.png"), results["color_mask"])
        cv2.imwrite(os.path.join(output_dir, f"{prefix}_overlay.png"), cv2.cvtColor(results["overlay"], cv2.COLOR_RGB2BGR))
