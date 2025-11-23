"""Routes package - exports all API routers"""

from .segment import router as segment_router
from .health import router as health_router

# Export routers for easy importing
segment = segment_router
health = health_router

__all__ = ['segment', 'health', 'segment_router', 'health_router']
