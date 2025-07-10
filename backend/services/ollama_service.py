"""
Ollama Service - Compatibility layer for unified architecture.

This service provides backward compatibility with existing Ollama integrations
while using the unified provider system under the hood.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
import httpx
from dataclasses import dataclass

from .unified_providers import get_provider_service
from config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class OllamaResponse:
    """Response from Ollama API."""
    text: str
    model: str
    processing_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class OllamaService:
    """
    Ollama service that wraps the unified provider system.
    
    Provides backward compatibility for existing Ollama integrations.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.provider_service = get_provider_service()
        self.base_url = self.settings.ollama.base_url
        self.timeout = self.settings.ollama.timeout
        logger.info(f"OllamaService initialized with base URL: {self.base_url}")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> OllamaResponse:
        """
        Chat completion using unified provider system.
        
        Args:
            messages: List of chat messages
            model: Model name to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            OllamaResponse with generated text
        """
        try:
            # Use unified provider system
            response = await self.provider_service.chat(
                messages=messages,
                model=f"ollama::{model}",  # Prefix with provider
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return OllamaResponse(
                text=response.text,
                model=model,
                processing_time=response.processing_time,
                metadata=response.metadata
            )
            
        except Exception as e:
            logger.error(f"Ollama chat completion failed: {e}")
            return OllamaResponse(
                text=f"Error: {str(e)}",
                model=model,
                processing_time=0.0,
                metadata={"error": str(e)}
            )
    
    async def generate_text(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> OllamaResponse:
        """
        Generate text using unified provider system.
        
        Args:
            prompt: Input prompt
            model: Model name to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            OllamaResponse with generated text
        """
        try:
            # Use unified provider system
            response = await self.provider_service.generate(
                prompt=prompt,
                model=f"ollama::{model}",  # Prefix with provider
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return OllamaResponse(
                text=response.text,
                model=model,
                processing_time=response.processing_time,
                metadata=response.metadata
            )
            
        except Exception as e:
            logger.error(f"Ollama text generation failed: {e}")
            return OllamaResponse(
                text=f"Error: {str(e)}",
                model=model,
                processing_time=0.0,
                metadata={"error": str(e)}
            )
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get available Ollama models."""
        try:
            # Try to get from unified provider
            all_models = await self.provider_service.list_all_models()
            ollama_models = all_models.get("ollama", [])
            
            if ollama_models:
                return ollama_models
            
            # Fallback: direct API call
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    result = response.json()
                    return result.get("models", [])
                else:
                    logger.warning(f"Failed to get models: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    async def check_connection(self) -> bool:
        """Check if Ollama server is accessible."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/version")
                return response.status_code == 200
        except Exception:
            return False
    
    async def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/show",
                    json={"name": model}
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"error": f"Model not found: {model}"}
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {"error": str(e)} 

    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        """
        Stream chat completion using unified provider system.
        
        Args:
            messages: List of chat messages
            model: Model name to use
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Yields:
            Streaming chunks from the LLM
        """
        try:
            # Use unified provider system for streaming
            async for chunk in self.provider_service.stream_chat(
                messages=messages,
                model=f"ollama::{model}",  # Prefix with provider
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Ollama streaming chat completion failed: {e}")
            # Fallback: yield error message
            yield {"error": str(e)} 