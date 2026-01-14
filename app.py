"""
FastAPI application for Graphiti-Lapa agent with memory.

Run with:
    uvicorn app:app --reload --host 0.0.0.0 --port 8080
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.text import router as text_router
from config.logging_config import configure_logging
from config.phoenix_config import setup_phoenix_instrumentation
from clients.graphiti_client import get_graphiti_client
import logging

from utils import setup_langsmith

# Configure logging on startup
configure_logging(level="INFO")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler - runs once on startup/shutdown.
    Builds Graphiti indices on startup to avoid repeated builds per request.
    """
    # Startup: Setup Phoenix observability
    logger.info("Application startup: setting up Phoenix instrumentation...")
    try:
        phoenix_enabled = setup_phoenix_instrumentation()
        if phoenix_enabled:
            logger.info("✓ Phoenix instrumentation enabled")
        else:
            logger.info("ℹ Phoenix instrumentation disabled or unavailable")
        setup_langsmith()
    except Exception as e:
        logger.warning(f"Phoenix setup warning: {e}")
    
    # Startup: build indices once
    logger.info("Initializing Graphiti indices...")
    try:
        graphiti = await get_graphiti_client()
        await graphiti.graphiti.build_indices_and_constraints()
        logger.info("✓ Graphiti indices built successfully")
    except Exception as e:
        logger.warning(f"Index building warning (may already exist): {e}")
    
    yield
    
    # Shutdown: cleanup
    logger.info("Application shutdown: closing Graphiti connection...")
    try:
        await graphiti.close()
        logger.info("✓ Graphiti connection closed")
    except Exception as e:
        logger.error(f"Error closing Graphiti: {e}")

# Create FastAPI application
app = FastAPI(
    title="Graphiti-Lapa Agent API",
    description="AI agent з довготривалою пам'яттю на базі Lapa LLM та Graphiti",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(text_router, tags=["text"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Graphiti-Lapa Agent API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "POST /text": "Обробка текстових повідомлень агентом"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}