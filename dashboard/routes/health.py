"""Health check and system info endpoints"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from datetime import datetime

from services.model_service import model_service
from services.utils_service import utils_service
from schemas import HealthCheckResponse, VersionResponse, StatsResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint for monitoring
    
    Returns system status, model state, and device information
    """
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.1.0',
        'model_loaded': model_service.checkpoint_info.get('loaded', False),
        'device': str(model_service.device) if model_service.device else 'unknown'
    }


@router.get("/api/version", response_model=VersionResponse)
async def get_version():
    """
    Get API version and system information
    
    Returns version, model details, and device info
    """
    return {
        'version': '1.1.0',
        'model': 'U-Net++',
        'architecture': 'Nested U-Net with Deep Supervision',
        'parameters': '9.0M',
        'device': str(model_service.device),
        'model_loaded': model_service.checkpoint_info.get('loaded', False),
        'last_updated': datetime.now().isoformat()
    }


@router.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get model statistics and performance metrics
    
    Returns model info, performance metrics, and inference history
    """
    metrics_dict = utils_service.load_metrics()
    history = utils_service.load_history()

    return {
        'model': {
            'name': 'U-Net++',
            'parameters': '9.0M',
            'epoch': model_service.checkpoint_info.get('epoch', 'N/A'),
            'val_dice': model_service.checkpoint_info.get('dice', 0.0)
        },
        'performance': metrics_dict,
        'history': history
    }
