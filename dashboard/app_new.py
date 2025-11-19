"""
FastAPI Dashboard for Retina Blood Vessel Segmentation
BACKWARD COMPATIBILITY WRAPPER - Imports from refactored main.py
"""

# This file is kept for backward compatibility
# All functionality has been moved to main.py with modular architecture
# This simply re-exports the app from main.py

from main import app

# Re-export for any external imports
__all__ = ['app']

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
