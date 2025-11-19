"""
Pydantic schemas for request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class SegmentationResult(BaseModel):
    """Response schema for image segmentation"""
    success: bool
    original_image: str = Field(..., description="Base64 encoded original image")
    mask: str = Field(..., description="Base64 encoded binary mask")
    overlay: str = Field(..., description="Base64 encoded overlay visualization")
    heatmap: str = Field(..., description="Base64 encoded probability heatmap")
    dice: float = Field(..., ge=0.0, le=1.0, description="Dice coefficient")
    iou: float = Field(..., ge=0.0, le=1.0, description="Intersection over Union")
    pixel_accuracy: float = Field(..., ge=0.0, le=1.0, description="Pixel accuracy")
    vessel_coverage: float = Field(..., ge=0.0, le=100.0, description="Vessel coverage percentage")
    mean_confidence: float = Field(..., ge=0.0, le=1.0, description="Mean confidence score")
    image_size: str = Field(..., description="Original image size (WxH)")
    processing_time: float = Field(..., ge=0.0, description="Processing time in seconds")
    model_info: Dict[str, Any] = Field(..., description="Model information")


class ErrorResponse(BaseModel):
    """Response schema for errors"""
    success: bool = False
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Detailed error information")


class HealthCheckResponse(BaseModel):
    """Response schema for health check endpoint"""
    status: str = Field(..., description="System status")
    timestamp: str = Field(..., description="ISO format timestamp")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Compute device (cuda/cpu)")


class VersionResponse(BaseModel):
    """Response schema for version endpoint"""
    version: str = Field(..., description="API version")
    model: str = Field(..., description="Model name")
    architecture: str = Field(..., description="Model architecture")
    parameters: str = Field(..., description="Number of model parameters")
    device: str = Field(..., description="Compute device")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    last_updated: str = Field(..., description="Last update timestamp")


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    dice: float = Field(..., ge=0.0, le=100.0, description="Dice coefficient percentage")
    accuracy: float = Field(..., ge=0.0, le=100.0, description="Accuracy percentage")
    sensitivity: float = Field(..., ge=0.0, le=100.0, description="Sensitivity percentage")
    specificity: float = Field(..., ge=0.0, le=100.0, description="Specificity percentage")
    auc: float = Field(..., ge=0.0, le=100.0, description="AUC percentage")


class StatsResponse(BaseModel):
    """Response schema for stats endpoint"""
    model: Dict[str, Any] = Field(..., description="Model information")
    performance: ModelMetrics = Field(..., description="Model performance metrics")
    history: List[Dict[str, Any]] = Field(..., description="Inference history")


class HistoryEntry(BaseModel):
    """Single history entry"""
    timestamp: datetime
    stats: Dict[str, Any]


class ValidationError(BaseModel):
    """Validation error details"""
    field: str
    message: str
    error_code: str
