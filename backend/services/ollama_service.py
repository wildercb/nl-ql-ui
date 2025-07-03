"""Enhanced Ollama service with comprehensive LLM interaction tracking."""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any
import httpx
from pydantic import BaseModel

from config.settings import get_settings
from services.llm_tracking_service import get_tracking_service

logger = logging.getLogger(__name__)


class GenerationResult(BaseModel):
    """Result of a text generation request."""
    
    text: str
    model: str
    processing_time: float
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    response_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class OllamaService:
    """Enhanced Ollama service with comprehensive tracking and monitoring."""
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.ollama.base_url
        self.default_model = self.settings.ollama.default_model
        self.timeout = self.settings.ollama.timeout
        self.tracking_service = get_tracking_service()
        
        # HTTP client for Ollama API
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=self.timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        
        logger.info(f"🦙 Ollama service initialized - Base URL: {self.base_url}")
    
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make HTTP request to Ollama API with error handling."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            logger.debug(f"🌐 Making {method} request to {url}")
            
            if method.upper() == "GET":
                response = await self.client.get(url)
            elif method.upper() == "POST":
                response = await self.client.post(url, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Ollama API HTTP error: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Ollama API error: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"❌ Ollama API request error: {e}")
            raise Exception(f"Ollama connection error: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected Ollama API error: {e}")
            raise Exception(f"Ollama service error: {e}")
    
    async def generate_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        interaction_type: str = "generation"
    ) -> GenerationResult:
        """Generate response from model with comprehensive tracking."""
        model = model or self.default_model
        temperature = temperature or self.settings.ollama.temperature
        max_tokens = max_tokens or self.settings.ollama.max_tokens
        session_id = session_id or f"ollama-{int(time.time())}"
        
        # Build payload for Ollama
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        # Track the interaction
        async with self.tracking_service.track_interaction(
            session_id=session_id,
            model=model,
            provider="ollama",
            interaction_type=interaction_type,
            user_id=user_id,
            context_data={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
        ) as tracker:
            
            # Set tracking data
            tracker.set_prompt(prompt)
            tracker.set_system_prompt(system_prompt)
            tracker.set_parameters(temperature=temperature, max_tokens=max_tokens)
            
            start_time = time.time()
            
            try:
                logger.debug(f"🦙 Generating response with model: {model}")
                
                # Make request to Ollama
                response = await self._make_request("POST", "/api/generate", payload)
                
                processing_time = time.time() - start_time
                
                # Extract response text
                response_text = response.get("response", "")
                
                # Track response
                tracker.set_response(response_text)
                tracker.set_performance_metrics(
                    processing_time=processing_time,
                    tokens_used=response.get("eval_count", 0) + response.get("prompt_eval_count", 0),
                    prompt_tokens=response.get("prompt_eval_count", 0),
                    response_tokens=response.get("eval_count", 0)
                )
                
                # Estimate confidence based on response quality
                confidence = self._estimate_confidence(response_text, response)
                tracker.set_performance_metrics(confidence_score=confidence)
                
                logger.info(
                    f"✅ Generated response: {len(response_text)} chars, "
                    f"{processing_time:.2f}s, model: {model}"
                )
                
                return GenerationResult(
                    text=response_text,
                    model=model,
                    processing_time=processing_time,
                    tokens_used=response.get("eval_count", 0) + response.get("prompt_eval_count", 0),
                    prompt_tokens=response.get("prompt_eval_count", 0),
                    response_tokens=response.get("eval_count", 0),
                    finish_reason=response.get("done_reason"),
                    metadata={
                        "eval_duration": response.get("eval_duration"),
                        "load_duration": response.get("load_duration"),
                        "prompt_eval_duration": response.get("prompt_eval_duration"),
                        "total_duration": response.get("total_duration")
                    }
                )
                
            except Exception as e:
                processing_time = time.time() - start_time
                error_message = str(e)
                
                # Track error
                tracker.set_error(error_message)
                tracker.set_performance_metrics(processing_time=processing_time)
                
                logger.error(f"❌ Generation failed: {error_message}")
                raise
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> GenerationResult:
        """Generate chat completion response with tracking."""
        model = model or self.default_model
        temperature = temperature or self.settings.ollama.temperature
        max_tokens = max_tokens or self.settings.ollama.max_tokens
        session_id = session_id or f"chat-{int(time.time())}"
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        # Convert messages to prompt for tracking
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        async with self.tracking_service.track_interaction(
            session_id=session_id,
            model=model,
            provider="ollama",
            interaction_type="chat_completion",
            user_id=user_id,
            context_data={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "message_count": len(messages)
            }
        ) as tracker:
            
            tracker.set_prompt(prompt)
            tracker.set_parameters(temperature=temperature, max_tokens=max_tokens)
            
            start_time = time.time()
            
            try:
                response = await self._make_request("POST", "/api/chat", payload)
                
                processing_time = time.time() - start_time
                response_text = response.get("message", {}).get("content", "")
                
                tracker.set_response(response_text)
                tracker.set_performance_metrics(
                    processing_time=processing_time,
                    tokens_used=response.get("eval_count", 0) + response.get("prompt_eval_count", 0),
                    prompt_tokens=response.get("prompt_eval_count", 0),
                    response_tokens=response.get("eval_count", 0)
                )
                
                confidence = self._estimate_confidence(response_text, response)
                tracker.set_performance_metrics(confidence_score=confidence)
                
                return GenerationResult(
                    text=response_text,
                    model=model,
                    processing_time=processing_time,
                    tokens_used=response.get("eval_count", 0) + response.get("prompt_eval_count", 0),
                    prompt_tokens=response.get("prompt_eval_count", 0),
                    response_tokens=response.get("eval_count", 0),
                    finish_reason=response.get("done_reason"),
                    metadata=response
                )
                
            except Exception as e:
                processing_time = time.time() - start_time
                tracker.set_error(str(e))
                tracker.set_performance_metrics(processing_time=processing_time)
                raise
    
    def _estimate_confidence(self, response_text: str, response_data: Dict) -> float:
        """Estimate confidence score based on response characteristics."""
        try:
            # Base confidence starts at 0.5
            confidence = 0.5
            
            # Adjust based on response length (longer responses often indicate more detail)
            if len(response_text) > 100:
                confidence += 0.1
            if len(response_text) > 500:
                confidence += 0.1
                
            # Adjust based on eval_count (more tokens processed suggests better quality)
            eval_count = response_data.get("eval_count", 0)
            if eval_count > 50:
                confidence += 0.1
            if eval_count > 200:
                confidence += 0.1
                
            # Check for completion indicators
            if response_data.get("done", False):
                confidence += 0.1
            
            # Check for error indicators in response text
            error_indicators = ["error", "sorry", "cannot", "unable", "don't know"]
            if any(indicator in response_text.lower() for indicator in error_indicators):
                confidence -= 0.2
            
            # Ensure confidence is within bounds
            return max(0.0, min(1.0, confidence))
            
        except Exception:
            return 0.5  # Default confidence if estimation fails
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        try:
            response = await self._make_request("GET", "/api/tags")
            models = response.get("models", [])
            
            logger.info(f"📋 Found {len(models)} available models")
            return models
            
        except Exception as e:
            logger.error(f"❌ Failed to list models: {e}")
            return []
    
    async def show_model_info(self, model: str) -> Dict[str, Any]:
        """Get detailed information about a specific model."""
        try:
            payload = {"name": model}
            response = await self._make_request("POST", "/api/show", payload)
            
            logger.debug(f"📊 Retrieved info for model: {model}")
            return response
            
        except Exception as e:
            logger.error(f"❌ Failed to get model info for {model}: {e}")
            return {}
    
    async def pull_model(self, model: str) -> bool:
        """Pull/download a model."""
        try:
            payload = {"name": model}
            response = await self._make_request("POST", "/api/pull", payload)
            
            logger.info(f"⬇️ Successfully pulled model: {model}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to pull model {model}: {e}")
            return False
    
    async def delete_model(self, model: str) -> bool:
        """Delete a model."""
        try:
            payload = {"name": model}
            response = await self._make_request("DELETE", "/api/delete", payload)
            
            logger.info(f"🗑️ Successfully deleted model: {model}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete model {model}: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Ollama service health."""
        try:
            start_time = time.time()
            models = await self.list_models()
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "base_url": self.base_url,
                "default_model": self.default_model,
                "available_models": len(models),
                "response_time": response_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "base_url": self.base_url,
                "timestamp": time.time()
            }
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
        logger.info("🔒 Ollama service client closed") 