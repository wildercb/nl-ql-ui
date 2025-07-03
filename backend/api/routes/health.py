"""Health check API endpoints."""

import time
from typing import Dict, Any
from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel
import structlog

from config import get_settings
from services.ollama_service import OllamaService

logger = structlog.get_logger()
router = APIRouter()


class HealthResponse(BaseModel):
    """Response model for health checks."""
    status: str
    timestamp: float
    version: str
    environment: str
    services: Dict[str, Any]


async def get_ollama_service() -> OllamaService:
    """Dependency to get Ollama service."""
    return OllamaService()


@router.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint for the API.
    Returns a simple status to indicate the API is up and running.
    """
    return {"status": "healthy"}


@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    settings = get_settings()
    
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version=settings.api.version,
        environment=settings.env,
        services={}
    )


@router.get("/detailed")
async def detailed_health_check(
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Detailed health check including all services."""
    settings = get_settings()
    start_time = time.time()
    
    services = {}
    overall_status = "healthy"
    
    # Check Ollama service
    try:
        models = await ollama_service.get_available_models()
        services["ollama"] = {
            "status": "healthy",
            "model_count": len(models),
            "base_url": settings.ollama.base_url,
            "default_model": settings.ollama.default_model
        }
    except Exception as e:
        logger.warning("Ollama health check failed", error=str(e))
        services["ollama"] = {
            "status": "unhealthy",
            "error": str(e),
            "base_url": settings.ollama.base_url
        }
        overall_status = "degraded"
    
    # Check database connection (placeholder)
    try:
        # This would check actual database connection
        services["database"] = {
            "status": "healthy",
            "url": settings.database.url.split("@")[-1] if "@" in settings.database.url else "configured"
        }
    except Exception as e:
        logger.warning("Database health check failed", error=str(e))
        services["database"] = {
            "status": "unhealthy", 
            "error": str(e)
        }
        overall_status = "degraded"
    
    # Check Redis connection (placeholder)
    try:
        # This would check actual Redis connection
        services["redis"] = {
            "status": "healthy",
            "url": settings.redis.url
        }
    except Exception as e:
        logger.warning("Redis health check failed", error=str(e))
        services["redis"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        overall_status = "degraded"
    
    response_time = time.time() - start_time
    
    return {
        "status": overall_status,
        "timestamp": time.time(),
        "version": settings.api.version,
        "environment": settings.env,
        "response_time": response_time,
        "services": services,
        "configuration": {
            "cors_origins": settings.cors.origins,
            "rate_limiting": {
                "requests_per_minute": settings.rate_limit.requests_per_minute,
                "burst": settings.rate_limit.burst
            },
            "query_limits": {
                "max_query_length": settings.query.max_query_length,
                "max_results_per_page": settings.query.max_results_per_page,
                "cache_ttl": settings.query.cache_ttl
            }
        }
    }


@router.get("/ready")
async def readiness_check(
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Readiness check for container orchestration."""
    try:
        # Check critical services
        await ollama_service.get_available_models()
        
        return {"status": "ready", "timestamp": time.time()}
    except Exception as e:
        logger.error("Readiness check failed", error=str(e))
        return {"status": "not_ready", "error": str(e), "timestamp": time.time()}


@router.get("/live")
async def liveness_check():
    """Liveness check for container orchestration."""
    return {"status": "alive", "timestamp": time.time()} 