"""
FastAPI Inference Server - Offroad Semantic Segmentation
Temporary local API that simulates the future civic backend.

This server exposes a /predict endpoint that can be replaced with the civic backend
when it becomes available. The API contract is standardized for easy integration.

Usage:
    uvicorn inference_server:app --host 0.0.0.0 --port 8000

Environment Variables:
    MODEL_CHECKPOINT: Path to segmentation head checkpoint (default: ./segmentation_head.pth)
    CLASS_NAMES_JSON: Path to class names JSON (optional)
    DEVICE: torch device ('cpu' or 'cuda', default: auto)
    UPLOAD_DIR: Directory for uploading images (default: ./uploads)
    API_PORT: Port to run API on (default: 8000)
"""

import base64
import json
import os
import io
import time
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import torch

from predict import SegmentationModel


# ============================================================================
# CONFIG
# ============================================================================

MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", "./segmentation_head.pth")
CLASS_NAMES_JSON = os.getenv("CLASS_NAMES_JSON")
DEVICE = os.getenv("DEVICE")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
API_PORT = int(os.getenv("API_PORT", 8000))
TRAIN_METRICS_DIR = os.getenv("TRAIN_METRICS_DIR", "")

# Create upload directory
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="Offroad Segmentation API",
    description="ML inference API for semantic segmentation. Simulates future civic backend.",
    version="1.0.0",
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# GLOBAL MODEL (Loaded once at startup)
# ============================================================================

model: Optional[SegmentationModel] = None


def _to_float_or_none(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int_or_none(value):
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _pick_best_index(values):
    best_idx = None
    best_value = None

    if not isinstance(values, list):
        return best_idx

    for idx, value in enumerate(values):
        num = _to_float_or_none(value)
        if num is None:
            continue
        if best_value is None or num > best_value:
            best_value = num
            best_idx = idx

    return best_idx


def _safe_index(values, idx):
    if not isinstance(values, list):
        return None
    if idx is None:
        return None
    if idx < 0 or idx >= len(values):
        return None
    return _to_float_or_none(values[idx])


def _candidate_metrics_dirs() -> List[Path]:
    script_dir = Path(__file__).resolve().parent
    candidates: List[Path] = []

    if TRAIN_METRICS_DIR:
        configured = Path(TRAIN_METRICS_DIR)
        candidates.append(configured)
        if not configured.is_absolute():
            candidates.append(script_dir / configured)

    candidates.extend([
        script_dir / "train_stats_epoch2_gpu_eval",
        script_dir / "train_stats_epoch2_gpu",
        script_dir / "train_stats_epoch8_gpu",
        script_dir / "train_stats",
    ])

    for path in script_dir.glob("train_stats*"):
        if path.is_dir():
            candidates.append(path)

    unique_paths = []
    seen = set()
    for path in candidates:
        resolved = str(path.resolve()) if path.exists() else str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists() and path.is_dir():
            unique_paths.append(path)

    unique_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return unique_paths


def _read_json_if_exists(path: Path):
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _extract_training_metrics(summary_payload: dict, history_payload: dict) -> dict:
    history_root = history_payload.get("history", history_payload)
    if not isinstance(history_root, dict):
        history_root = {}

    epochs = history_root.get("epoch", [])
    val_iou = history_root.get("val_iou", [])
    val_dice = history_root.get("val_dice", [])
    val_pixel_acc = history_root.get("val_pixel_acc", [])

    best_idx = _pick_best_index(val_iou)

    metrics = {
        "epochs_ran": _to_int_or_none(summary_payload.get("epochs_ran"))
        if summary_payload.get("epochs_ran") is not None
        else _to_int_or_none(len(epochs) if isinstance(epochs, list) else None),
        "best_epoch": _to_int_or_none(summary_payload.get("best_epoch")),
        "best_val_iou": _to_float_or_none(summary_payload.get("best_val_iou")),
        "best_val_dice": None,
        "best_val_pixel_acc": None,
        "last_val_iou": _safe_index(val_iou, len(val_iou) - 1 if isinstance(val_iou, list) else None),
        "last_val_dice": _safe_index(val_dice, len(val_dice) - 1 if isinstance(val_dice, list) else None),
        "last_val_pixel_acc": _safe_index(
            val_pixel_acc,
            len(val_pixel_acc) - 1 if isinstance(val_pixel_acc, list) else None,
        ),
    }

    if metrics["best_val_iou"] is None and best_idx is not None:
        metrics["best_val_iou"] = _safe_index(val_iou, best_idx)

    if best_idx is not None:
        metrics["best_val_dice"] = _safe_index(val_dice, best_idx)
        metrics["best_val_pixel_acc"] = _safe_index(val_pixel_acc, best_idx)

    if metrics["best_epoch"] is None and best_idx is not None and isinstance(epochs, list):
        epoch_value = epochs[best_idx] if best_idx < len(epochs) else None
        metrics["best_epoch"] = _to_int_or_none(epoch_value)

    return metrics


def load_live_training_metrics() -> dict:
    default_metrics = {
        "epochs_ran": None,
        "best_epoch": None,
        "best_val_iou": None,
        "best_val_dice": None,
        "best_val_pixel_acc": None,
        "last_val_iou": None,
        "last_val_dice": None,
        "last_val_pixel_acc": None,
        "source_dir": "",
    }

    for metrics_dir in _candidate_metrics_dirs():
        summary_path = metrics_dir / "training_summary.json"
        history_path = metrics_dir / "history.json"

        if not summary_path.exists() and not history_path.exists():
            continue

        summary_payload = _read_json_if_exists(summary_path)
        history_payload = _read_json_if_exists(history_path)
        metrics = _extract_training_metrics(summary_payload, history_payload)

        metrics["source_dir"] = str(metrics_dir)
        return {**default_metrics, **metrics}

    return default_metrics


def _safe_stat_row(class_stat: dict) -> dict:
    """Normalize a class-stat row to printable numeric values."""
    return {
        "class_name": str(class_stat.get("class_name", "Unknown")),
        "coverage_pct": _to_float_or_none(class_stat.get("coverage_pct")) or 0.0,
        "mean_confidence": _to_float_or_none(class_stat.get("mean_confidence")) or 0.0,
        "region_count": _to_int_or_none(class_stat.get("region_count")) or 0,
        "pixel_count": _to_int_or_none(class_stat.get("pixel_count")) or 0,
    }


def _print_prediction_report(filename: str, result: dict, processing_time_ms: float) -> None:
    """Print a compact per-image prediction table to backend terminal logs."""
    class_stats = result.get("class_stats", [])
    if not isinstance(class_stats, list):
        class_stats = []

    rows = [_safe_stat_row(item) for item in class_stats if isinstance(item, dict)]
    overall_confidence = _to_float_or_none(result.get("overall_confidence")) or 0.0
    present_classes = _to_int_or_none(result.get("present_classes")) or 0
    total_classes = len(rows)

    print("\n" + "=" * 92)
    print(
        f"[PREDICT] file={filename} | latency={processing_time_ms:.1f}ms "
        f"| overall_confidence={overall_confidence:.2f}% | present_classes={present_classes}/{total_classes}"
    )
    print("-" * 92)
    print(f"{'Class':<22}{'Coverage %':>12}{'Confidence %':>14}{'Regions':>10}{'Pixels':>14}")
    print("-" * 92)

    for row in rows:
        print(
            f"{row['class_name'][:22]:<22}"
            f"{row['coverage_pct']:>12.2f}"
            f"{row['mean_confidence']:>14.2f}"
            f"{row['region_count']:>10d}"
            f"{row['pixel_count']:>14d}"
        )

    # Print explicit highlights often requested in demos.
    rocks = next((r for r in rows if r["class_name"].strip().lower() == "rocks"), None)
    sky = next((r for r in rows if r["class_name"].strip().lower() == "sky"), None)
    if rocks or sky:
        print("-" * 92)
        if rocks:
            print(
                f"Rocks -> coverage={rocks['coverage_pct']:.2f}%, "
                f"confidence={rocks['mean_confidence']:.2f}%, "
                f"regions={rocks['region_count']}, pixels={rocks['pixel_count']}"
            )
        if sky:
            print(
                f"Sky   -> coverage={sky['coverage_pct']:.2f}%, "
                f"confidence={sky['mean_confidence']:.2f}%, "
                f"regions={sky['region_count']}, pixels={sky['pixel_count']}"
            )

    print("=" * 92)


def _ensure_model_loaded():
    """Lazy-load model on first use instead of at startup."""
    global model
    
    if model is not None:
        return
    
    try:
        print(f"[LAZY LOAD] Loading model from: {MODEL_CHECKPOINT}")
        print(f"Device: {DEVICE or 'auto'}")
        
        if not os.path.exists(MODEL_CHECKPOINT):
            raise FileNotFoundError(
                f"Model checkpoint not found: {MODEL_CHECKPOINT}\n"
                "Please set MODEL_CHECKPOINT environment variable or ensure checkpoint exists."
            )
        
        model = SegmentationModel(
            checkpoint_path=MODEL_CHECKPOINT,
            class_names_path=CLASS_NAMES_JSON,
            device=DEVICE,
        )
        print("[OK] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    _ensure_model_loaded()
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": model.device if model else None,
        "num_classes": model.out_channels if model else None,
        "class_names": model.class_names if model else None,
    }


# ============================================================================
# MAIN PREDICTION ENDPOINT
# ============================================================================

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Main prediction endpoint.
    
    Request:
        - file: Image file (PNG, JPG, etc.)
    
    Response:
        {
            "success": bool,
            "mask": str (base64-encoded PNG),
            "overlay": str (base64-encoded PNG),
            "class_distribution": {class_name: percentage, ...},
            "coverage": [[class_name, percentage], ...],
            "class_stats": [
                {
                    "class_id": int,
                    "class_name": str,
                    "pixel_count": int,
                    "coverage_pct": float,
                    "mean_confidence": float,
                    "region_count": int
                }
            ],
            "overall_confidence": float,
            "num_classes": int,
            "image_width": int,
            "image_height": int,
        }
    """
    if model is None:
        _ensure_model_loaded()
    
    try:
        start_time = time.perf_counter()

        # Read uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Store original dimensions
        orig_width, orig_height = image.size
        
        # Run inference
        result = model.predict(image)
        processing_time_ms = (time.perf_counter() - start_time) * 1000.0
        training_metrics = load_live_training_metrics()
        _print_prediction_report(file.filename or "uploaded_image", result, processing_time_ms)
        
        # Prepare response with base64-encoded images
        response = {
            "success": True,
            "mask": SegmentationModel.image_to_base64(result["color_mask"]),
            "overlay": SegmentationModel.image_to_base64(result["overlay"]),
            "class_distribution": result["class_distribution"],
            "coverage": result["coverage"],
            "class_counts": result.get("class_counts", {}),
            "class_confidence": result.get("class_confidence", {}),
            "class_stats": result.get("class_stats", []),
            "overall_confidence": result.get("overall_confidence", 0.0),
            "present_classes": result.get("present_classes", 0),
            "total_pixels": result.get("total_pixels", 0),
            "processing_time_ms": processing_time_ms,
            "training_metrics": training_metrics,
            "num_classes": model.out_channels,
            "image_width": orig_width,
            "image_height": orig_height,
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e),
            }
        )


# ============================================================================
# BATCH PREDICTION ENDPOINT
# ============================================================================

@app.post("/predict-batch")
async def predict_batch_endpoint(files: List[UploadFile] = File(...)):
    """
    Batch prediction endpoint for multiple images.
    
    Request:
        - files: Multiple image files
    
    Response:
        {
            "success": bool,
            "results": [
                {prediction result for each image}
            ],
            "total_images": int,
            "successful": int,
            "failed": int,
            "processing_time_ms": float,
        }
    """
    if model is None:
        _ensure_model_loaded()
    
    import time
    start_time = time.time()
    
    results = []
    successful = 0
    failed = 0
    
    for file in files:
        try:
            per_file_start = time.perf_counter()
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            orig_width, orig_height = image.size
            
            result = model.predict(image)
            per_file_ms = (time.perf_counter() - per_file_start) * 1000.0
            _print_prediction_report(file.filename or "uploaded_image", result, per_file_ms)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "mask": SegmentationModel.image_to_base64(result["color_mask"]),
                "overlay": SegmentationModel.image_to_base64(result["overlay"]),
                "class_distribution": result["class_distribution"],
                "coverage": result["coverage"],
                "class_counts": result.get("class_counts", {}),
                "class_confidence": result.get("class_confidence", {}),
                "class_stats": result.get("class_stats", []),
                "overall_confidence": result.get("overall_confidence", 0.0),
                "present_classes": result.get("present_classes", 0),
                "total_pixels": result.get("total_pixels", 0),
                "image_width": orig_width,
                "image_height": orig_height,
            })
            successful += 1
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e),
            })
            failed += 1
    
    processing_time = (time.time() - start_time) * 1000
    
    return JSONResponse(content={
        "success": failed == 0,
        "results": results,
        "total_images": len(files),
        "successful": successful,
        "failed": failed,
        "processing_time_ms": processing_time,
    })


# ============================================================================
# METADATA ENDPOINT
# ============================================================================

@app.get("/metadata")
async def get_metadata():
    """Get model metadata and API information."""
    if model is None:
        _ensure_model_loaded()

    training_metrics = load_live_training_metrics()
    
    return {
        "api_version": "1.0.0",
        "model_type": "DINOv2 + ConvNeXt Segmentation Head",
        "num_classes": model.out_channels,
        "class_names": model.class_names,
        "device": model.device,
        "checkpoint_path": model.checkpoint_path,
        "input_height": model.h,
        "input_width": model.w,
        "supported_formats": ["PNG", "JPG", "JPEG", "BMP"],
        "api_endpoints": {
            "/health": "GET - Health check",
            "/metadata": "GET - Model metadata",
            "/predict": "POST - Single image prediction",
            "/predict-batch": "POST - Batch prediction",
        },
        "training_metrics": training_metrics,
    }


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "name": "Offroad Segmentation API",
        "version": "1.0.0",
        "description": "ML inference API for offroad semantic segmentation",
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
    }


# ============================================================================
# STARTUP LOGGING
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Offroad Segmentation API - Inference Server")
    print("=" * 60)
    print(f"Model checkpoint: {MODEL_CHECKPOINT}")
    print(f"Class names: {CLASS_NAMES_JSON or 'Using defaults'}")
    print(f"Device: {DEVICE or 'auto'}")
    print(f"Upload directory: {UPLOAD_DIR}")
    print("")
    print("Starting server...")
    print(f"API will be available at: http://localhost:{API_PORT}")
    print("Documentation at: http://localhost:{API_PORT}/docs")
    print("=" * 60)
    
    uvicorn.run(
        "inference_server:app",
        host="0.0.0.0",
        port=API_PORT,
        reload=False,
    )
