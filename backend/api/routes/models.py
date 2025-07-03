"""Model management API endpoints."""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import structlog

from services.ollama_service import OllamaService
from config import get_settings

logger = structlog.get_logger()
router = APIRouter(prefix="/models", tags=["Model Management"])


class ModelInfo(BaseModel):
    """Model information from Ollama."""
    name: str
    size: Optional[int] = None
    modified_at: Optional[str] = None
    digest: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class ModelResponse(BaseModel):
    """Response model for model information."""
    name: str
    size: Optional[int] = None
    modified_at: Optional[str] = None
    digest: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_model_info(cls, model_info: Dict[str, Any]) -> "ModelResponse":
        """Create ModelResponse from Ollama model info."""
        return cls(
            name=model_info.get("name", ""),
            size=model_info.get("size"),
            modified_at=model_info.get("modified_at"),
            digest=model_info.get("digest"),
            details=model_info.get("details")
        )


class ModelListResponse(BaseModel):
    """Response model for model list."""
    models: List[ModelResponse]
    total_count: int


async def get_ollama_service() -> OllamaService:
    """Dependency to get Ollama service."""
    return OllamaService()


@router.get("/health/status")
async def model_service_health(
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Check the health of the model service."""
    try:
        health_info = await ollama_service.health_check()
        return {
            "status": "healthy",
            "service": "ollama",
            "details": health_info
        }
        
    except Exception as e:
        logger.error(f"Model service health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "ollama",
            "error": str(e)
        }


@router.get("/", response_model=ModelListResponse)
async def list_models(
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Get list of available models."""
    try:
        models_data = await ollama_service.list_models()
        
        # models_data is already a list of model dictionaries
        models = [
            ModelResponse.from_model_info(model_info)
            for model_info in models_data
        ]
        
        return ModelListResponse(
            models=models,
            total_count=len(models)
        )
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/{model_name}", response_model=ModelResponse)
async def get_model_info(
    model_name: str,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Get detailed information about a specific model."""
    try:
        model_info = await ollama_service.show_model_info(model_name)
        return ModelResponse.from_model_info(model_info)
        
    except Exception as e:
        logger.error(f"Failed to get model info for {model_name}: {e}")
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")


@router.post("/{model_name}/pull")
async def pull_model(
    model_name: str,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Pull a model from Ollama."""
    try:
        success = await ollama_service.pull_model(model_name)
        
        if success:
            return {"message": f"Model {model_name} pulled successfully"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to pull model {model_name}")
            
    except Exception as e:
        logger.error(f"Failed to pull model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to pull model: {str(e)}")


@router.delete("/{model_name}")
async def delete_model(
    model_name: str,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Delete a model from Ollama."""
    try:
        success = await ollama_service.delete_model(model_name)
        
        if success:
            return {"message": f"Model {model_name} deleted successfully"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to delete model {model_name}")
            
    except Exception as e:
        logger.error(f"Failed to delete model {model_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}") 