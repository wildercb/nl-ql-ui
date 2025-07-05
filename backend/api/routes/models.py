"""Model management API endpoints."""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import structlog
import asyncio
import json

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


@router.post("/{model_name}/pull/stream")
async def pull_model_stream(
    model_name: str,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """Pull a model from Ollama with streaming progress updates."""
    
    async def generate_progress():
        try:
            # Start the pull process
            logger.info(f"Starting streaming pull for model: {model_name}")
            
            # Send initial status
            yield f"data: {json.dumps({'status': 'starting', 'model': model_name, 'progress': 0})}\n\n"
            
            # Simulate progress updates (in a real implementation, you'd hook into Ollama's actual progress)
            for i in range(1, 11):
                await asyncio.sleep(0.5)  # Simulate work
                progress = i * 10
                yield f"data: {json.dumps({'status': 'downloading', 'model': model_name, 'progress': progress})}\n\n"
            
            # Actually pull the model
            success = await ollama_service.pull_model(model_name)
            
            if success:
                yield f"data: {json.dumps({'status': 'completed', 'model': model_name, 'progress': 100})}\n\n"
            else:
                yield f"data: {json.dumps({'status': 'failed', 'model': model_name, 'error': 'Pull failed'})}\n\n"
                
        except Exception as e:
            logger.error(f"Streaming pull failed for {model_name}: {e}")
            yield f"data: {json.dumps({'status': 'failed', 'model': model_name, 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


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