"""Main FastAPI application with MongoDB integration and comprehensive LLM tracking."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from config.settings import get_settings
from services.database_service import initialize_database, close_database
from services.llm_tracking_service import get_tracking_service
from .middleware import (
    RateLimitMiddleware,
    LoggingMiddleware,
    SecurityMiddleware,
    CacheMiddleware
)
from .routes import health, translation, validation, models, analytics, interactions, history, chat, auth, agent_demo, agent_demo_stream

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    
    # Startup
    logger.info("🚀 Starting MPPW MCP API with MongoDB...")
    
    try:
        # Initialize MongoDB connection and Beanie ODM
        await initialize_database()
        logger.info("✅ MongoDB connection initialized")
        
        # Initialize LLM tracking service
        tracking_service = get_tracking_service()
        logger.info("📊 LLM tracking service initialized")
        
        logger.info("🎉 API startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Failed to start API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down MPPW MCP API...")
    
    try:
        # Close MongoDB connection
        await close_database()
        logger.info("🔒 MongoDB connection closed")
        
        logger.info("✅ API shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.api.title,
    description=settings.api.description,
    version=settings.api.version,
    debug=settings.api.debug,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors.origins,
    allow_credentials=settings.cors.allow_credentials,
    allow_methods=settings.cors.allow_methods,
    allow_headers=settings.cors.allow_headers,
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add custom middleware
app.add_middleware(RateLimitMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(SecurityMiddleware)
app.add_middleware(CacheMiddleware)

# Include API routes
app.include_router(health.router)
app.include_router(translation.router)
app.include_router(validation.router)
app.include_router(models.router)
app.include_router(analytics.router)  # New analytics endpoints
app.include_router(interactions.router)  # Live interactions streaming
app.include_router(history.router)  # Query history
app.include_router(chat.router)  # Chat functionality
app.include_router(auth.router)  # Authentication endpoints
app.include_router(agent_demo.router)  # Multi-agent demo endpoint (blocking)
app.include_router(agent_demo_stream.router)  # Multi-agent streaming SSE endpoint

logger.info(f"📍 API routes registered: {len(app.routes)} endpoints")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "type": type(exc).__name__ if settings.is_development else "Error"
        }
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    
    return {
        "name": settings.api.title,
        "version": settings.api.version,
        "description": settings.api.description,
        "status": "healthy",
        "database": "MongoDB",
        "features": [
            "Natural Language to GraphQL Translation",
            "Query Validation and Optimization",
            "AI Model Management",
            "Comprehensive LLM Interaction Tracking",
            "Real-time Analytics and Monitoring",
            "Session Management",
            "Batch Processing"
        ],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "translation": "/translation",
            "validation": "/validation", 
            "models": "/models",
            "analytics": "/analytics"
        }
    }


@app.get("/info")
async def get_api_info():
    """Get detailed API and system information."""
    
    try:
        from services.database_service import get_database_service
        from services.ollama_service import OllamaService
        
        # Get database stats
        db_service = await get_database_service()
        db_stats = await db_service.get_stats()
        
        # Get Ollama health
        ollama_service = OllamaService()
        ollama_health = await ollama_service.health_check()
        
        # Get tracking service stats
        tracking_service = get_tracking_service()
        tracking_analytics = await tracking_service.get_interaction_analytics()
        
        return {
            "api": {
                "title": settings.api.title,
                "version": settings.api.version,
                "environment": settings.env,
                "debug": settings.api.debug
            },
            "database": {
                "type": "MongoDB",
                "status": "connected" if db_stats.get("database") else "disconnected",
                "collections": db_stats.get("database", {}).get("collections", 0),
                "total_documents": sum(
                    collection.get("count", 0) 
                    for collection in db_stats.get("collections", {}).values()
                )
            },
            "ollama": {
                "status": ollama_health.get("status", "unknown"),
                "base_url": ollama_health.get("base_url"),
                "available_models": ollama_health.get("available_models", 0),
                "response_time": ollama_health.get("response_time", 0)
            },
            "tracking": {
                "total_interactions": tracking_analytics.get("summary", {}).get("total_interactions", 0),
                "success_rate": tracking_analytics.get("summary", {}).get("success_rate", 0),
                "recent_24h": tracking_analytics.get("summary", {}).get("recent_24h", 0)
            },
            "features": {
                "llm_tracking": settings.llm_tracking.enabled,
                "prometheus_metrics": settings.monitoring.prometheus_metrics,
                "rate_limiting": True,
                "cors_enabled": True,
                "compression": True
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get API info: {e}")
        return {
            "error": "Failed to retrieve system information",
            "message": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"🌐 Starting API server on {settings.api.host}:{settings.api.port}")
    
    uvicorn.run(
        "api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.is_development,
        workers=settings.api.workers if not settings.is_development else 1,
        log_level="debug" if settings.is_development else "info"
    ) 