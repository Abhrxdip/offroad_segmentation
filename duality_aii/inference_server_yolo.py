"""
FastAPI Inference Server - YOLO Segmentation
FastAPI backend using YOLOv8 for inference.

Usage:
    uvicorn inference_server_yolo:app --host 0.0.0.0 --port 8000

Environment Variables:
    YOLO_MODEL: Path or name of YOLO model (default: yolov8m-seg.pt)
    CLASS_NAMES_JSON: Path to class names JSON (optional)
    DEVICE: torch device ('cpu' or 'cuda', default: auto)
    UPLOAD_DIR: Directory for uploading images (default: ./uploads)
    API_PORT: Port to run API on (default: 8000)
    CONF_THRESHOLD: Confidence threshold for YOLO (default: 0.5)
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

from predict_yolo import YOLOSegmentationModel


# ============================================================================
# CONFIG
# ============================================================================

YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8m-seg.pt")
CLASS_NAMES_JSON = os.getenv("CLASS_NAMES_JSON")
DEVICE = os.getenv("DEVICE")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
API_PORT = int(os.getenv("API_PORT", 8000))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.5))
TRAIN_METRICS_DIR = os.getenv("TRAIN_METRICS_DIR", "")

# Create upload directory
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================================================
# FASTAPI APP SETUP
# ============================================================================

app = FastAPI(
    title="Offroad Segmentation API (YOLO)",
    description="ML inference API for semantic segmentation using YOLOv8.",
    version="2.0.0",
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

model: Optional[YOLOSegmentationModel] = None


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


def _candidate_metrics_dirs() -> List[Path]:
    script_dir = Path(__file__).resolve().parent
    candidates: List[Path] = []

    if TRAIN_METRICS_DIR:
        configured = Path(TRAIN_METRICS_DIR)
        candidates.append(configured)
        if not configured.is_absolute():
            candidates.append(script_dir / configured)

    candidates.extend([
        script_dir / "train_stats_yolo",
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


def load_live_training_metrics() -> dict:
    default_metrics = {
        "epochs_ran": None,
        "best_epoch": None,
        "source_dir": "",
    }

    for metrics_dir in _candidate_metrics_dirs():
        summary_path = metrics_dir / "results.json"
        if not summary_path.exists():
            continue

        summary_payload = _read_json_if_exists(summary_path)
        if summary_payload:
            return {**default_metrics, "source_dir": str(metrics_dir), **summary_payload}

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
    detections = _to_int_or_none(result.get("detections")) or 0

    print("\n" + "=" * 92)
    print(
        f"[YOLO PREDICT] file={filename} | latency={processing_time_ms:.1f}ms "
        f"| confidence={overall_confidence:.2f}% | present_classes={present_classes}/{total_classes} | detections={detections}"
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

    print("=" * 92)


def _ensure_model_loaded():
    """Lazy-load model on first use instead of at startup."""
    global model
    
    if model is not None:
        return
    
    try:
        print(f"[LAZY LOAD] Loading YOLO model: {YOLO_MODEL}")
        print(f"Device: {DEVICE or 'auto'}")
        print(f"Confidence threshold: {CONF_THRESHOLD}")
        
        model = YOLOSegmentationModel(
            model_path=YOLO_MODEL,
            class_names_path=CLASS_NAMES_JSON,
            device=DEVICE,
            conf_threshold=CONF_THRESHOLD,
        )
        print("[OK] YOLO model loaded successfully")
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
        "model_type": "YOLOv8 Segmentation",
        "device": model.device if model else None,
        "num_classes": model.num_classes if model else None,
        "class_names": model.class_names if model else None,
    }


# ============================================================================
# MAIN PREDICTION ENDPOINT
# ============================================================================

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Main prediction endpoint using YOLO.
    
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
            "detections": int,
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
            "mask": YOLOSegmentationModel.image_to_base64(result["color_mask"]),
            "overlay": YOLOSegmentationModel.image_to_base64(result["overlay"]),
            "class_distribution": result["class_distribution"],
            "coverage": result["coverage"],
            "class_counts": result.get("class_counts", {}),
            "class_confidence": result.get("class_confidence", {}),
            "class_stats": result.get("class_stats", []),
            "overall_confidence": result.get("overall_confidence", 0.0),
            "detections": result.get("detections", 0),
            "present_classes": result.get("present_classes", 0),
            "total_pixels": result.get("total_pixels", 0),
            "processing_time_ms": processing_time_ms,
            "training_metrics": training_metrics,
            "num_classes": model.num_classes,
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
    Batch prediction endpoint for multiple images using YOLO.
    
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
                "mask": YOLOSegmentationModel.image_to_base64(result["color_mask"]),
                "overlay": YOLOSegmentationModel.image_to_base64(result["overlay"]),
                "class_distribution": result["class_distribution"],
                "coverage": result["coverage"],
                "class_counts": result.get("class_counts", {}),
                "class_confidence": result.get("class_confidence", {}),
                "class_stats": result.get("class_stats", []),
                "overall_confidence": result.get("overall_confidence", 0.0),
                "detections": result.get("detections", 0),
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
# TRAINING RESULTS ENDPOINT
# ============================================================================

@app.get("/api/training-results")
async def get_training_results():
    """Get YOLO training results and metrics."""
    try:
        # Look for training results JSON in various locations
        script_dir = Path(__file__).resolve().parent
        possible_paths = [
            script_dir / "train_stats_yolo" / "yolo_training_results.json",
            script_dir / "train_stats_yolo" / "yolo_final_run" / "results.json",
            script_dir / "train_stats" / "training_results.json",
        ]
        
        for path in possible_paths:
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                return JSONResponse(content=results)
        
        # If no file found, return a structured response with mock data reference
        return JSONResponse(content={
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "YOLOv8m-seg",
            "status": "No training results file found - returning mock data for demo",
            "training_config": {
                "epochs": 4,
                "batch_size": 8,
                "image_size": 640,
                "device": "CUDA"
            },
            "notes": "To use real training results, ensure train_stats_yolo/yolo_training_results.json exists"
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to load training results: {str(e)}"}
        )


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
        "api_version": "2.0.0",
        "model_type": "YOLOv8 Segmentation",
        "num_classes": model.num_classes,
        "class_names": model.class_names,
        "device": model.device,
        "model_path": model.model_path,
        "conf_threshold": model.conf_threshold,
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
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    print("[STARTUP] Initializing YOLO inference server...")
    _ensure_model_loaded()
    print("[STARTUP] Server ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("[SHUTDOWN] Shutting down YOLO inference server...")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=API_PORT,
    )
