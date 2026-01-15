from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.text import router as text_router
from config.logging_config import configure_logging
from config.phoenix_config import setup_phoenix_instrumentation
import logging

from utils import setup_langsmith

# Configure logging on startup
configure_logging(level="INFO")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    yield

# Create FastAPI application
app = FastAPI(
    title="Tabularas Agent API",
    description="AI agent з довготривалою пам'яттю на базі Lapa LLM",
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
        "message": "Tabularas Agent API",
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