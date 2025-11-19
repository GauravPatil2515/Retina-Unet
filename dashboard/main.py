"""
FastAPI Dashboard for Retina Blood Vessel Segmentation
Refactored modular architecture - v1.1.0 OPTIMIZED
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
import asyncio

# Import services and routes
from config import api_settings, checkpoint_info
from services.model_service import model_service
from routes import segment, health


# Get base directory
BASE_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    try:
        print("=" * 60)
        print("🚀 Starting RetinaAI Dashboard v1.1.0")
        print("=" * 60)
        
        # Load model in background
        await model_service.load_model_async()
        
        if model_service.checkpoint_info.get('loaded'):
            epoch = model_service.checkpoint_info.get('epoch')
            dice = model_service.checkpoint_info.get('dice')
            print(f"✓ Model loaded! Epoch {epoch}, Dice: {dice:.4f}")
        else:
            error = model_service.checkpoint_info.get('error', 'Unknown error')
            print(f"⚠ Model loading failed: {error}")
        
        print("=" * 60)
        print("🎉 Dashboard ready at http://localhost:8000")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ Error during startup: {e}")
        import traceback
        traceback.print_exc()
    
    yield
    
    # Shutdown
    print("Shutting down RetinaAI Dashboard...")


# Initialize FastAPI with optimizations
app = FastAPI(
    title=api_settings.app_name,
    description=api_settings.app_description,
    version=api_settings.app_version,
    docs_url=api_settings.docs_url,
    redoc_url=api_settings.redoc_url,
    lifespan=lifespan
)

# Add CORS middleware
if api_settings.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_settings.cors_origins,
        allow_credentials=api_settings.cors_credentials,
        allow_methods=api_settings.cors_methods,
        allow_headers=api_settings.cors_headers,
    )

# Mount static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Include routers
app.include_router(segment.router)
app.include_router(health.router)


@app.get("/", response_class=HTMLResponse)
async def home():
    """Render landing page"""
    return templates.TemplateResponse("landing.html", {"request": None})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request):
    """Render main dashboard"""
    return templates.TemplateResponse("index_platform.html", {
        "request": request,
        "model_loaded": model_service.checkpoint_info.get('loaded', False),
        "checkpoint_info": model_service.checkpoint_info
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=api_settings.host,
        port=api_settings.port
    )
