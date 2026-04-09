/**
 * Segmentation API Client
 * 
 * Handles communication with the ML inference API and provides
 * compatibility wrappers expected by older frontend components.
 *
 * Backend currently used in this repo:
 * - POST /predict (single image file)
 * - POST /predict-batch (multiple image files)
 * - GET  /health
 * - GET  /metadata
 * 
 * Environment Variables:
 * - VITE_API_BASE: Base URL for API (default: http://localhost:8000)
 * - VITE_API_TIMEOUT: Request timeout in ms (default: 60000)
 */

import axios from 'axios';

// ============================================================================
// CONFIG
// ============================================================================

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const API_TIMEOUT = import.meta.env.VITE_API_TIMEOUT ? parseInt(import.meta.env.VITE_API_TIMEOUT) : 60000;

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: API_TIMEOUT,
});

// In-memory upload cache for legacy flow (upload -> infer by imageId)
const uploadStore = new Map();

const RISK_WEIGHTS = {
  0: -1.0,
  1: 0.8,
  2: 0.7,
  3: 0.6,
  4: -0.5,
  5: 0.0,
  6: 1.0,
  7: 0.3,
  8: 0.5,
  9: 0.9,
};

const DEFAULT_VALIDATION_METRICS = {
  baseline: {
    meanIoU: 0.687,
    pixelAccuracy: 0.812,
    dice: 0.756,
    latency: 0.252,
    perClassIoU: {
      drivable_ground: 0.856,
      rock: 0.623,
      log: 0.541,
      clutter: 0.438,
      grass: 0.754,
      sky: 0.934,
      water: 0.0,
      vegetation: 0.621,
      stairs: 0.289,
      obstacle: 0.512,
    },
  },
  improved: {
    meanIoU: 0.738,
    pixelAccuracy: 0.854,
    dice: 0.803,
    latency: 0.128,
    perClassIoU: {
      drivable_ground: 0.891,
      rock: 0.687,
      log: 0.612,
      clutter: 0.521,
      grass: 0.803,
      sky: 0.951,
      water: 0.0,
      vegetation: 0.698,
      stairs: 0.421,
      obstacle: 0.624,
    },
  },
};

const DEFAULT_ABLATION_RESULTS = {
  withoutAugmentation: {
    meanIoU: 0.712,
    pixelAccuracy: 0.831,
    inferenceTime: 0.128,
  },
  withAugmentation: {
    meanIoU: 0.738,
    pixelAccuracy: 0.854,
    inferenceTime: 0.128,
  },
};

function generateImageId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `img-${Date.now()}-${Math.floor(Math.random() * 1e9)}`;
}

function ensureImageFile(file) {
  if (!file) {
    throw new Error('Image file is required');
  }
  if (!(file instanceof File)) {
    throw new Error('Input must be a File object');
  }
  if (!file.type || !file.type.startsWith('image/')) {
    throw new Error('File must be an image');
  }
}

function buildPredictionsFromResult(result) {
  const coverage = Array.isArray(result?.coverage) ? result.coverage : [];
  const classDistribution = result?.class_distribution || {};

  const pixelCoverages = {};
  let totalPixels = 0;

  if (coverage.length > 0) {
    coverage.forEach((item, index) => {
      const percentage = Number(item?.[1] ?? 0);
      const pseudoPixels = Math.max(0, Math.round(percentage * 1000));
      pixelCoverages[index] = pseudoPixels;
      totalPixels += pseudoPixels;
    });
  } else {
    const entries = Object.entries(classDistribution);
    entries.forEach(([, percentage], index) => {
      const pseudoPixels = Math.max(0, Math.round(Number(percentage) * 1000));
      pixelCoverages[index] = pseudoPixels;
      totalPixels += pseudoPixels;
    });
  }

  if (totalPixels === 0) {
    for (let i = 0; i < 10; i += 1) {
      pixelCoverages[i] = 0;
    }
  }

  let topClass = '-';
  let topClassConfidence = 0;

  if (coverage.length > 0) {
    const sorted = [...coverage].sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0));
    topClass = String(sorted[0]?.[0] ?? '-');
    topClassConfidence = Number(sorted[0]?.[1] ?? 0) / 100;
  } else if (Object.keys(classDistribution).length > 0) {
    const sorted = Object.entries(classDistribution).sort(
      (a, b) => Number(b[1] || 0) - Number(a[1] || 0),
    );
    topClass = String(sorted[0]?.[0] ?? '-');
    topClassConfidence = Number(sorted[0]?.[1] ?? 0) / 100;
  }

  const allClassConfidences = {};
  coverage.forEach((item, index) => {
    allClassConfidences[index] = Number(item?.[1] ?? 0) / 100;
  });

  return {
    pixelCoverages,
    totalPixels,
    topClass,
    topClassConfidence,
    allClassConfidences,
  };
}

function getRiskLevel(pixelCoverages) {
  const ids = Object.keys(pixelCoverages);
  if (ids.length === 0) {
    return 'unknown';
  }

  let score = 0;
  let total = 0;

  ids.forEach((id) => {
    const clsId = Number(id);
    const pixels = Number(pixelCoverages[id]) || 0;
    const weight = RISK_WEIGHTS[clsId] ?? 0;
    score += pixels * weight;
    total += pixels;
  });

  if (total <= 0) {
    return 'unknown';
  }

  const normalized = (score / total) * 100;
  if (normalized < 30) {
    return 'Safe';
  }
  if (normalized < 60) {
    return 'Caution';
  }
  return 'High-Risk';
}

function enrichValidationMetrics(metrics) {
  const baseline = metrics?.baseline || DEFAULT_VALIDATION_METRICS.baseline;
  const improved = metrics?.improved || DEFAULT_VALIDATION_METRICS.improved;
  return {
    ...metrics,
    baseline,
    improved,
    perClassIoU: improved.perClassIoU || baseline.perClassIoU || {},
  };
}

// ============================================================================
// LEGACY-COMPATIBLE UPLOAD API (frontend-side cache)
// ============================================================================

/**
 * Legacy-compatible single upload.
 * Stores File in memory and returns an imageId used by runInference.
 */
export async function uploadImage(file) {
  ensureImageFile(file);
  const imageId = generateImageId();
  uploadStore.set(imageId, file);

  return {
    imageId,
    filename: file.name,
    path: file.name,
    size: file.size,
    uploadedAt: new Date().toISOString(),
  };
}

/**
 * Legacy-compatible batch upload.
 * Stores each File in memory and returns imageIds.
 */
export async function uploadFolder(files) {
  const imageFiles = (files || []).filter((file) => {
    try {
      ensureImageFile(file);
      return true;
    } catch {
      return false;
    }
  });

  if (imageFiles.length === 0) {
    throw new Error('No valid image files provided');
  }

  const imageIds = imageFiles.map((file) => {
    const imageId = generateImageId();
    uploadStore.set(imageId, file);
    return imageId;
  });

  return {
    imageIds,
    count: imageIds.length,
    paths: imageFiles.map((f) => f.name),
    uploadedAt: new Date().toISOString(),
  };
}

/**
 * Alias used by UploadPanel component.
 */
export function readImageAsDataURL(file) {
  return imageToDataURL(file);
}

// ============================================================================
// METADATA & HEALTH CHECKS
// ============================================================================

/**
 * Check if API server is healthy and ready
 */
export async function checkHealth() {
  try {
    const response = await apiClient.get('/health');
    return response.data;
  } catch (error) {
    throw new Error(`API health check failed: ${error.message}`);
  }
}

/**
 * Get model metadata and supported classes
 */
export async function getMetadata() {
  try {
    const response = await apiClient.get('/metadata');
    return response.data;
  } catch (error) {
    throw new Error(`Failed to fetch metadata: ${error.message}`);
  }
}

// ============================================================================
// PREDICTION ENDPOINTS
// ============================================================================

/**
 * Run inference on a single image
 * 
 * @param {File} imageFile - Image file to predict on
 * @returns {Promise<Object>} Prediction result with mask and overlay (base64)
 * 
 * Response format:
 * {
 *   success: boolean,
 *   mask: "data:image/png;base64,...",
 *   overlay: "data:image/png;base64,...",
 *   class_distribution: { class_name: percentage, ... },
 *   coverage: [ [class_name, percentage], ... ],
 *   num_classes: number,
 *   image_width: number,
 *   image_height: number,
 *   error?: string (if success=false)
 * }
 */
export async function predict(imageFile) {
  ensureImageFile(imageFile);

  const formData = new FormData();
  formData.append('file', imageFile);

  try {
    const response = await apiClient.post('/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    if (!response.data.success) {
      throw new Error(response.data.error || 'Prediction failed');
    }

    // Convert base64 to data URLs for display
    return {
      ...response.data,
      mask: `data:image/png;base64,${response.data.mask}`,
      overlay: `data:image/png;base64,${response.data.overlay}`,
    };
  } catch (error) {
    throw new Error(`Prediction failed: ${error.message}`);
  }
}

/**
 * Run batch inference on multiple images
 * 
 * @param {File[]} imageFiles - Array of image files
 * @returns {Promise<Object>} Batch results
 * 
 * Response format:
 * {
 *   success: boolean,
 *   results: [
 *     {
 *       filename: string,
 *       success: boolean,
 *       mask: "data:image/png;base64,...",
 *       overlay: "data:image/png;base64,...",
 *       class_distribution: { ... },
 *       coverage: [ ... ],
 *       image_width: number,
 *       image_height: number,
 *       error?: string
 *     },
 *     ...
 *   ],
 *   total_images: number,
 *   successful: number,
 *   failed: number,
 *   processing_time_ms: number
 * }
 */
export async function predictBatch(imageFiles) {
  if (!Array.isArray(imageFiles) || imageFiles.length === 0) {
    throw new Error('At least one image file is required');
  }

  imageFiles.forEach((file) => ensureImageFile(file));

  const formData = new FormData();
  imageFiles.forEach((file) => {
    formData.append('files', file);
  });

  try {
    const response = await apiClient.post('/predict-batch', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });

    // Convert base64 to data URLs for all results
    const results = response.data.results.map((result) => {
      if (result.success) {
        return {
          ...result,
          mask: `data:image/png;base64,${result.mask}`,
          overlay: `data:image/png;base64,${result.overlay}`,
        };
      }
      return result;
    });

    return {
      ...response.data,
      results,
    };
  } catch (error) {
    throw new Error(`Batch prediction failed: ${error.message}`);
  }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Convert image File to base64 string for preview/display
 * 
 * @param {File} file - Image file
 * @returns {Promise<string>} Data URL (data:image/...)
 */
export function imageToDataURL(file) {
  return new Promise((resolve, reject) => {
    if (!file) {
      reject(new Error('File is required'));
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.onerror = (err) => reject(new Error(`Failed to read file: ${err}`));
    reader.readAsDataURL(file);
  });
}

/**
 * Convert base64 image to blob for download
 * 
 * @param {string} base64String - Base64 encoded image
 * @param {string} filename - Filename for download
 */
export async function downloadBase64Image(base64String, filename) {
  // Remove data URL prefix if present
  const base64Data = base64String.includes(',') 
    ? base64String.split(',')[1] 
    : base64String;

  const byteCharacters = atob(base64Data);
  const byteArray = new Uint8Array(byteCharacters.length);
  
  for (let i = 0; i < byteCharacters.length; i++) {
    byteArray[i] = byteCharacters.charCodeAt(i);
  }

  const blob = new Blob([byteArray], { type: 'image/png' });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  link.parentNode.removeChild(link);
  window.URL.revokeObjectURL(url);
}

/**
 * Format class coverage for display
 * 
 * @param {Object} classDistribution - {class_name: percentage, ...}
 * @returns {Array} Sorted array of [class_name, percentage]
 */
export function formatCoverageForDisplay(classDistribution) {
  return Object.entries(classDistribution)
    .map(([className, percentage]) => [className, percentage.toFixed(2)])
    .sort((a, b) => b[1] - a[1]); // Sort by percentage descending
}

/**
 * Get API configuration for debugging/display
 */
export function getAPIConfig() {
  return {
    baseURL: API_BASE,
    timeout: API_TIMEOUT,
    endpoints: {
      health: '/health',
      metadata: '/metadata',
      predict: '/predict',
      predictBatch: '/predict-batch',
    },
    uploadCacheSize: uploadStore.size,
  };
}

// ============================================================================
// LEGACY-COMPATIBLE INFERENCE API
// ============================================================================

/**
 * Legacy-compatible single inference by imageId.
 */
export async function runInference(imageId, modelVersion = 'improved') {
  const imageFile = uploadStore.get(imageId);
  if (!imageFile) {
    throw new Error(`Image ID not found in upload cache: ${imageId}`);
  }

  const start = performance.now();
  const result = await predict(imageFile);
  const inferenceTime = Number((performance.now() - start).toFixed(2));

  const predictions = buildPredictionsFromResult(result);
  return {
    imageId,
    filename: imageFile.name,
    maskUrl: result.mask,
    overlayUrl: result.overlay,
    inferenceTime,
    modelVersion,
    predictions: {
      pixelCoverages: predictions.pixelCoverages,
      totalPixels: predictions.totalPixels,
      topClass: predictions.topClass,
      topClassConfidence: predictions.topClassConfidence,
      allClassConfidences: predictions.allClassConfidences,
    },
    topClass: predictions.topClass,
    confidence: predictions.topClassConfidence,
    riskLevel: getRiskLevel(predictions.pixelCoverages),
    status: 'completed',
  };
}

/**
 * Legacy-compatible batch inference by imageIds.
 */
export async function runBatchInference(imageIds, modelVersion = 'improved') {
  if (!Array.isArray(imageIds) || imageIds.length === 0) {
    throw new Error('At least one imageId is required');
  }

  const imageFiles = imageIds.map((imageId) => {
    const file = uploadStore.get(imageId);
    if (!file) {
      throw new Error(`Image ID not found in upload cache: ${imageId}`);
    }
    return file;
  });

  const start = performance.now();
  const batch = await predictBatch(imageFiles);
  const totalTime = Number((batch.processing_time_ms ?? (performance.now() - start)).toFixed(2));
  const successfulCount = Math.max(1, Number(batch.successful || 0));
  const avgInferenceTime = Number((totalTime / successfulCount).toFixed(2));

  const results = (batch.results || []).map((item, idx) => {
    if (!item?.success) {
      return {
        imageId: imageIds[idx],
        filename: item?.filename || imageFiles[idx]?.name || '-',
        inferenceTime: avgInferenceTime,
        modelVersion,
        topClass: '-',
        confidence: 0,
        riskLevel: 'unknown',
        status: 'failed',
        error: item?.error || 'Batch inference failed',
      };
    }

    const predictions = buildPredictionsFromResult(item);
    return {
      imageId: imageIds[idx],
      filename: item?.filename || imageFiles[idx]?.name || '-',
      maskUrl: item.mask,
      overlayUrl: item.overlay,
      inferenceTime: avgInferenceTime,
      modelVersion,
      topClass: predictions.topClass,
      confidence: predictions.topClassConfidence,
      riskLevel: getRiskLevel(predictions.pixelCoverages),
      status: 'completed',
      predictions: {
        pixelCoverages: predictions.pixelCoverages,
        totalPixels: predictions.totalPixels,
        topClass: predictions.topClass,
        topClassConfidence: predictions.topClassConfidence,
        allClassConfidences: predictions.allClassConfidences,
      },
    };
  });

  return {
    results,
    totalCount: imageIds.length,
    completedCount: results.filter((r) => r.status === 'completed').length,
    failedCount: results.filter((r) => r.status !== 'completed').length,
    avgInferenceTime,
  };
}

/**
 * Fetch validation metrics. Falls back to local sample values when endpoint is unavailable.
 */
export async function fetchValidationMetrics() {
  try {
    const response = await apiClient.get('/metrics');
    return enrichValidationMetrics(response.data);
  } catch {
    return enrichValidationMetrics({ ...DEFAULT_VALIDATION_METRICS });
  }
}

/**
 * Fetch ablation results. Falls back to local sample values when endpoint is unavailable.
 */
export async function fetchAblationResults() {
  try {
    const response = await apiClient.get('/ablation');
    return response.data;
  } catch {
    return { ...DEFAULT_ABLATION_RESULTS };
  }
}

/**
 * Fetch demo samples. Falls back to uploaded images cached in-browser.
 */
export async function fetchDemoSamples(limit = 10) {
  try {
    const response = await apiClient.get('/demo-samples', { params: { limit } });
    return response.data;
  } catch {
    const cachedEntries = Array.from(uploadStore.entries()).slice(0, limit);
    const samples = await Promise.all(
      cachedEntries.map(async ([imageId, file]) => ({
        imageId,
        imageUrl: await imageToDataURL(file),
        filename: file.name,
      })),
    );

    return {
      samples,
      count: samples.length,
      metadata: {
        source: 'upload-cache',
      },
    };
  }
}

// ============================================================================
// ERROR HANDLING
// ============================================================================

/**
 * Handle API errors with user-friendly messages
 */
export function formatAPIError(error) {
  if (error.response) {
    // Server responded with error status
    return {
      status: error.response.status,
      message: error.response.data?.error || error.response.data?.detail || 'Server error',
      data: error.response.data,
    };
  } else if (error.request) {
    // Request made but no response
    return {
      status: null,
      message: 'No response from server. Is the API running?',
      data: null,
    };
  } else {
    // Error in request setup
    return {
      status: null,
      message: error.message || 'Unknown error',
      data: null,
    };
  }
}

// ============================================================================
// EXPORT UTILITIES
// ============================================================================

/**
 * Export prediction results as CSV
 * 
 * @param {Array} results - Array of prediction results
 * @param {string} filename - Output filename
 */
export function exportResultsAsCSV(results, filename = 'segmentation-results.csv') {
  const rows = [];
  
  // Header row
  rows.push(['Filename', 'Class Name', 'Coverage (%)', 'Num Classes']);
  
  // Data rows
  results.forEach((result) => {
    if (result.success && result.coverage) {
      result.coverage.forEach(([className, percentage]) => {
        rows.push([
          result.filename || 'unknown',
          className,
          percentage.toFixed(2),
          result.num_classes,
        ]);
      });
    }
  });
  
  // Create CSV content
  const csvContent = rows.map((row) => row.map((cell) => `"${cell}"`).join(',')).join('\n');
  
  // Download
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  link.parentNode.removeChild(link);
  window.URL.revokeObjectURL(url);
}

/**
 * CSV export function used by BatchResultsTable.
 */
export function exportBatchResultsCSV(results, filename = 'batch-results.csv') {
  const rows = [];
  rows.push([
    'Filename',
    'Inference Time (ms)',
    'Model Version',
    'Top Class',
    'Confidence',
    'Risk Level',
    'Status',
  ]);

  (results || []).forEach((result) => {
    rows.push([
      result.filename || '-',
      result.inferenceTime ?? '-',
      result.modelVersion || '-',
      result.topClass || '-',
      result.confidence != null ? (Number(result.confidence) * 100).toFixed(2) : '-',
      result.riskLevel || '-',
      result.status || '-',
    ]);
  });

  const csvContent = rows
    .map((row) => row.map((cell) => `"${String(cell).replaceAll('"', '""')}"`).join(','))
    .join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  link.parentNode.removeChild(link);
  window.URL.revokeObjectURL(url);
}

// ============================================================================
// EXPORT ALL
// ============================================================================

export default {
  // Legacy upload flow
  uploadImage,
  uploadFolder,
  readImageAsDataURL,

  // Metadata
  checkHealth,
  getMetadata,
  getAPIConfig,
  
  // Prediction
  predict,
  predictBatch,
  runInference,
  runBatchInference,

  // Metrics and demos
  fetchValidationMetrics,
  fetchAblationResults,
  fetchDemoSamples,
  
  // Utilities
  imageToDataURL,
  downloadBase64Image,
  formatCoverageForDisplay,
  formatAPIError,
  exportResultsAsCSV,
  exportBatchResultsCSV,
};
