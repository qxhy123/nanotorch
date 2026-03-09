"""Main FastAPI application for Transformer Visualization API."""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
import os

# Add parent directory to path to import nanotorch
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
sys.path.insert(0, _project_root)

from app.api.routes import transformer, tokenizer, layer
from app.models.schemas import HealthResponse

# Create FastAPI app
app = FastAPI(
    title="Transformer Visualization API",
    description="API for visualizing Transformer model computations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check nanotorch availability
try:
    import nanotorch
    from nanotorch.nn.transformer import Transformer
    NANOTORCH_AVAILABLE = True
    NANOTORCH_VERSION = getattr(nanotorch, "__version__", "unknown")
except ImportError:
    NANOTORCH_AVAILABLE = False
    NANOTORCH_VERSION = None


# Include routers
app.include_router(transformer.router)
app.include_router(tokenizer.router)
app.include_router(layer.router)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        nanotorch_available=NANOTORCH_AVAILABLE,
    )


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Transformer Visualization API",
        "version": "1.0.0",
        "description": "API for visualizing Transformer model computations",
        "nanotorch_available": NANOTORCH_AVAILABLE,
        "nanotorch_version": NANOTORCH_VERSION,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": f"Internal server error: {str(exc)}",
        },
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    import logging
    logger = logging.getLogger("uvicorn")

    if NANOTORCH_AVAILABLE:
        logger.info("nanotorch is available")
    else:
        logger.warning("nanotorch is NOT available - some endpoints may not work")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
