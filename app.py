"""
FastAPI application for Graphiti-Lapa agent with memory.

Run with:
    uvicorn app:app --reload --host 0.0.0.0 --port 8080
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.text import router as text_router
from config.logging_config import configure_logging

# Configure logging on startup
configure_logging(level="INFO")

# Create FastAPI application
app = FastAPI(
    title="Graphiti-Lapa Agent API",
    description="AI agent з довготривалою пам'яттю на базі Lapa LLM та Graphiti",
    version="0.1.0",
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