"""
Unified Provider System for MPPW-MCP

This module provides a streamlined, extensible LLM provider framework that:
1. Makes adding new providers trivial with consistent interfaces
2. Integrates with the unified configuration system
3. Provides automatic provider discovery and registration
4. Handles authentication, rate limiting, and error handling consistently
5. Supports provider-specific optimizations and features
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union, Type, Protocol, AsyncGenerator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import time
import json
import os
from pathlib import Path
import httpx
from contextlib import asynccontextmanager

from config.unified_config import (
    get_unified_config, ModelProvider, ProviderConfig, ModelConfig,
    ModelSize
)
from services.llm_tracking_service import get_tracking_service

logger = logging.getLogger(__name__)


# =============================================================================
# Core Provider Framework
# =============================================================================

@dataclass
class GenerationRequest:
    """Request for text generation."""
    prompt: str
    model: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResponse:
    """Response from text generation."""
    text: str
    model: str
    provider: str
    processing_time: float
    tokens_used: int = 0
    prompt_tokens: int = 0
    response_tokens: int = 0
    finish_reason: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage:
    """Individual chat message."""
    role: str  # "system", "user", "assistant"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatRequest:
    """Request for chat completion."""
    messages: List[ChatMessage]
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    stream: bool = False
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderCapabilities:
    """Capabilities supported by a provider."""
    supports_chat: bool = True
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_embeddings: bool = False
    max_context_length: int = 4096
    supported_formats: List[str] = field(default_factory=lambda: ["text"])


# =============================================================================
# Base Provider Interface
# =============================================================================

class BaseProvider(ABC):
    """
    Base class for all LLM providers in the unified system.
    
    Provides consistent interface and common functionality:
    - Configuration integration
    - Authentication handling
    - Rate limiting
    - Error handling and retries
    - Performance tracking
    - Automatic registration
    """
    
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.config_manager = get_unified_config()
        self.tracking_service = get_tracking_service()
        
        # Get provider configuration
        self.config = self.config_manager.get_provider(provider_name)
        if not self.config:
            logger.warning(f"No configuration found for provider: {provider_name}")
            self.config = self._create_default_config()
        
        # Initialize HTTP client with common settings
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=self.config.timeout),
            headers=self._get_default_headers(),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        
        # Rate limiting
        self._last_request_time = 0.0
        self._rate_limit_delay = 60.0 / self.config.rate_limit if self.config.rate_limit else 0.0
        
        logger.info(f"Initialized provider: {provider_name}")
    
    @abstractmethod
    def get_provider_type(self) -> ModelProvider:
        """Get the provider type."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities."""
        pass
    
    @abstractmethod
    async def _generate_impl(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _chat_impl(self, request: ChatRequest) -> GenerationResponse:
        """Chat completion. Must be implemented by subclasses."""
        pass
    
    async def stream_chat(self, request: ChatRequest):
        """Stream chat completion. Must be implemented by subclasses."""
        raise NotImplementedError("stream_chat not implemented for this provider")
    
    def _create_default_config(self) -> ProviderConfig:
        """Create a default configuration for this provider."""
        return ProviderConfig(
            name=self.provider_name,
            provider_type=self.get_provider_type(),
            base_url="",
            api_key_env_var=f"{self.provider_name.upper()}_API_KEY",
            default_model=""
        )
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers for requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"MPPW-MCP/{self.provider_name}"
        }
        
        # Add provider-specific headers
        headers.update(self.config.headers)
        
        # Add authentication header if API key is available
        api_key = self._get_api_key()
        if api_key:
            headers.update(self._get_auth_headers(api_key))
        
        return headers
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        if self.config.api_key_env_var:
            return os.environ.get(self.config.api_key_env_var)
        return None
    
    def _get_auth_headers(self, api_key: str) -> Dict[str, str]:
        """Get authentication headers. Override for provider-specific auth."""
        return {"Authorization": f"Bearer {api_key}"}
    
    async def _apply_rate_limiting(self):
        """Apply rate limiting if configured."""
        if self._rate_limit_delay > 0:
            time_since_last = time.time() - self._last_request_time
            if time_since_last < self._rate_limit_delay:
                await asyncio.sleep(self._rate_limit_delay - time_since_last)
        self._last_request_time = time.time()
    
    async def _make_request(
        self,
        method: str,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> httpx.Response:
        """Make HTTP request with error handling and retries."""
        await self._apply_rate_limiting()
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if method.upper() == "POST":
                    if stream:
                        response = await self.client.post(url, json=data, stream=True)
                    else:
                        response = await self.client.post(url, json=data)
                elif method.upper() == "GET":
                    if stream:
                        response = await self.client.get(url, stream=True)
                    else:
                        response = await self.client.get(url)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response
                
            except httpx.HTTPStatusError as e:
                if attempt == self.config.max_retries:
                    logger.error(f"Provider {self.provider_name} HTTP error: {e.response.status_code}")
                    raise Exception(f"HTTP {e.response.status_code}: {e.response.text}")
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
                
            except httpx.RequestError as e:
                if attempt == self.config.max_retries:
                    logger.error(f"Provider {self.provider_name} request error: {e}")
                    raise Exception(f"Request error: {e}")
                
                await asyncio.sleep(2 ** attempt)
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text with tracking and error handling."""
        start_time = time.time()
        
        # Validate request
        if not request.model:
            raise ValueError("Model is required")
        
        if not request.prompt:
            raise ValueError("Prompt is required")
        
        # Check if model is supported
        if not self._is_model_supported(request.model):
            raise ValueError(f"Model {request.model} not supported by {self.provider_name}")
        
        # Track the generation
        async with self.tracking_service.track_interaction(
            session_id=request.session_id or f"{self.provider_name}-{int(time.time())}",
            model=request.model,
            provider=self.provider_name,
            interaction_type="generation",
            user_id=request.user_id,
            context_data=request.metadata
        ) as tracker:
            
            tracker.set_prompt(request.prompt)
            if request.system_prompt:
                tracker.set_system_prompt(request.system_prompt)
            tracker.set_parameters(
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            try:
                response = await self._generate_impl(request)
                
                # Track successful generation
                processing_time = time.time() - start_time
                response.processing_time = processing_time
                response.provider = self.provider_name
                
                tracker.set_response(response.text)
                tracker.set_performance_metrics(
                    processing_time=processing_time,
                    tokens_used=response.tokens_used,
                    prompt_tokens=response.prompt_tokens,
                    response_tokens=response.response_tokens,
                    confidence_score=response.confidence
                )
                
                return response
                
            except Exception as e:
                processing_time = time.time() - start_time
                tracker.set_error(str(e))
                tracker.set_performance_metrics(processing_time=processing_time)
                raise
    
    async def chat(self, request: ChatRequest) -> GenerationResponse:
        """Chat completion with tracking and error handling."""
        start_time = time.time()
        
        # Validate request
        if not request.model:
            raise ValueError("Model is required")
        
        if not request.messages:
            raise ValueError("Messages are required")
        
        # Check capabilities
        if not self.get_capabilities().supports_chat:
            raise ValueError(f"Provider {self.provider_name} does not support chat")
        
        # Track the chat
        prompt_text = "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
        
        async with self.tracking_service.track_interaction(
            session_id=request.session_id or f"{self.provider_name}-chat-{int(time.time())}",
            model=request.model,
            provider=self.provider_name,
            interaction_type="chat",
            user_id=request.user_id,
            context_data=request.metadata
        ) as tracker:
            
            tracker.set_prompt(prompt_text)
            tracker.set_parameters(
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            try:
                response = await self._chat_impl(request)
                
                # Track successful chat
                processing_time = time.time() - start_time
                response.processing_time = processing_time
                response.provider = self.provider_name
                
                tracker.set_response(response.text)
                tracker.set_performance_metrics(
                    processing_time=processing_time,
                    tokens_used=response.tokens_used,
                    prompt_tokens=response.prompt_tokens,
                    response_tokens=response.response_tokens,
                    confidence_score=response.confidence
                )
                
                return response
                
            except Exception as e:
                processing_time = time.time() - start_time
                tracker.set_error(str(e))
                tracker.set_performance_metrics(processing_time=processing_time)
                raise
    
    def _is_model_supported(self, model: str) -> bool:
        """Check if a model is supported by this provider."""
        return model in self.config.supported_models or len(self.config.supported_models) == 0
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models. Override for provider-specific implementation."""
        return [
            {"name": model, "provider": self.provider_name}
            for model in self.config.supported_models
        ]
    
    async def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get model information. Override for provider-specific implementation."""
        return {
            "name": model,
            "provider": self.provider_name,
            "supported": self._is_model_supported(model)
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.client.aclose()


# =============================================================================
# Concrete Provider Implementations
# =============================================================================

class OllamaProvider(BaseProvider):
    """Ollama provider implementation."""
    
    def __init__(self):
        super().__init__("ollama")
    
    def get_provider_type(self) -> ModelProvider:
        return ModelProvider.OLLAMA
    
    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_chat=True,
            supports_streaming=True,
            supports_function_calling=False,
            supports_vision=False,
            max_context_length=8192
        )
    
    def _get_auth_headers(self, api_key: str) -> Dict[str, str]:
        """Ollama doesn't need authentication headers."""
        return {}
    
    async def _generate_impl(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text using Ollama API."""
        payload = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": request.stream,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            }
        }
        
        if request.system_prompt:
            payload["system"] = request.system_prompt
        
        url = f"{self.config.base_url}/api/generate"
        response = await self._make_request("POST", url, payload)
        
        result = response.json()
        
        return GenerationResponse(
            text=result.get("response", ""),
            model=request.model,
            provider=self.provider_name,
            processing_time=0.0,  # Will be set by caller
            tokens_used=result.get("eval_count", 0) + result.get("prompt_eval_count", 0),
            prompt_tokens=result.get("prompt_eval_count", 0),
            response_tokens=result.get("eval_count", 0),
            finish_reason=result.get("done_reason"),
            metadata=result
        )
    
    async def _chat_impl(self, request: ChatRequest) -> GenerationResponse:
        """Chat completion using Ollama API."""
        payload = {
            "model": request.model,
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ],
            "stream": request.stream,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            }
        }
        
        url = f"{self.config.base_url}/api/chat"
        response = await self._make_request("POST", url, payload)
        
        result = response.json()
        message = result.get("message", {})
        
        return GenerationResponse(
            text=message.get("content", ""),
            model=request.model,
            provider=self.provider_name,
            processing_time=0.0,
            tokens_used=result.get("eval_count", 0) + result.get("prompt_eval_count", 0),
            prompt_tokens=result.get("prompt_eval_count", 0),
            response_tokens=result.get("eval_count", 0),
            finish_reason=result.get("done_reason"),
            metadata=result
        )
    
    async def stream_chat(self, request: ChatRequest):
        """Stream chat completion using Ollama API."""
        payload = {
            "model": request.model,
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ],
            "stream": request.stream,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            }
        }
        
        url = f"{self.config.base_url}/api/chat"
        async with self.client.stream("POST", url, json=payload) as response:
            async for chunk in response.aiter_text():
                yield chunk
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Ollama models."""
        try:
            url = f"{self.config.base_url}/api/tags"
            response = await self._make_request("GET", url)
            result = response.json()
            return result.get("models", [])
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []


class OpenAICompatibleProvider(BaseProvider):
    """Provider for OpenAI-compatible APIs (Groq, OpenRouter, etc.)."""
    
    def __init__(self, provider_name: str):
        super().__init__(provider_name)
    
    def get_provider_type(self) -> ModelProvider:
        # Map provider names to types
        provider_map = {
            "groq": ModelProvider.GROQ,
            "openrouter": ModelProvider.OPENROUTER,
            "openai": ModelProvider.OPENAI,
            "anthropic": ModelProvider.ANTHROPIC
        }
        return provider_map.get(self.provider_name, ModelProvider.OPENAI)
    
    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_chat=True,
            supports_streaming=True,
            supports_function_calling=True,
            max_context_length=32768  # Many support larger contexts
        )
    
    async def _generate_impl(self, request: GenerationRequest) -> GenerationResponse:
        """Generate using OpenAI-compatible API (convert to chat format)."""
        messages = []
        if request.system_prompt:
            messages.append(ChatMessage("system", request.system_prompt))
        messages.append(ChatMessage("user", request.prompt))
        
        chat_request = ChatRequest(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream,
            session_id=request.session_id,
            user_id=request.user_id,
            metadata=request.metadata
        )
        
        return await self._chat_impl(chat_request)
    
    async def _chat_impl(self, request: ChatRequest) -> GenerationResponse:
        """Chat completion using OpenAI-compatible API."""
        payload = {
            "model": request.model,
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream
        }
        
        url = f"{self.config.base_url}/chat/completions"
        response = await self._make_request("POST", url, payload)
        
        result = response.json()
        choice = result["choices"][0]
        message = choice["message"]
        usage = result.get("usage", {})
        
        return GenerationResponse(
            text=message["content"],
            model=request.model,
            provider=self.provider_name,
            processing_time=0.0,
            tokens_used=usage.get("total_tokens", 0),
            prompt_tokens=usage.get("prompt_tokens", 0),
            response_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason"),
            metadata=result
        )

    async def stream_chat(self, request: ChatRequest):
        """Stream chat completion using OpenAI-compatible API."""
        payload = {
            "model": request.model,
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream
        }
        
        url = f"{self.config.base_url}/chat/completions"
        async with self.client.stream("POST", url, json=payload) as response:
            async for chunk in response.aiter_text():
                yield chunk


# =============================================================================
# Provider Registry and Factory
# =============================================================================

class ProviderRegistry:
    """Registry for managing and discovering LLM providers."""
    
    def __init__(self):
        self.providers: Dict[str, BaseProvider] = {}
        self.config_manager = get_unified_config()
        
        # Auto-discover and register built-in providers
        self._discover_builtin_providers()
    
    def _discover_builtin_providers(self):
        """Discover and register all built-in providers."""
        # Register Ollama provider
        if self._should_register_provider("ollama"):
            self.register_provider(OllamaProvider())
        
        # Register OpenAI-compatible providers
        openai_compatible = ["groq", "openrouter", "openai", "anthropic"]
        for provider_name in openai_compatible:
            if self._should_register_provider(provider_name):
                self.register_provider(OpenAICompatibleProvider(provider_name))
    
    def _should_register_provider(self, provider_name: str) -> bool:
        """Check if provider should be registered based on configuration."""
        provider_config = self.config_manager.get_provider(provider_name)
        return provider_config is not None and provider_config.enabled
    
    def register_provider(self, provider: BaseProvider) -> None:
        """Register a provider in the registry."""
        self.providers[provider.provider_name] = provider
        logger.info(f"Registered provider: {provider.provider_name}")
    
    def get_provider(self, name: str) -> Optional[BaseProvider]:
        """Get a provider by name."""
        return self.providers.get(name)
    
    def list_providers(self) -> List[str]:
        """List all registered provider names."""
        return list(self.providers.keys())
    
    def get_provider_for_model(self, model: str) -> Optional[BaseProvider]:
        """Get the provider that supports a specific model."""
        # Check for provider prefix (e.g., "groq::llama3-8b")
        if "::" in model:
            provider_name = model.split("::", 1)[0]
            return self.get_provider(provider_name)
        
        # Find provider that supports this model
        for provider in self.providers.values():
            if provider._is_model_supported(model):
                return provider
        
        return None
    
    async def cleanup_all(self):
        """Cleanup all providers."""
        for provider in self.providers.values():
            await provider.cleanup()


# =============================================================================
# Unified Provider Interface
# =============================================================================

class UnifiedProviderService:
    """
    Unified service that provides a single interface to all providers.
    Handles provider selection, model routing, and fallbacks.
    """
    
    def __init__(self):
        self.registry = ProviderRegistry()
        self.config_manager = get_unified_config()
    
    def _parse_model_spec(self, model: str) -> tuple[str, str]:
        """Parse model specification into provider and model name."""
        if "::" in model:
            provider_name, model_name = model.split("::", 1)
            return provider_name, model_name
        
        # No provider specified, find one that supports this model
        provider = self.registry.get_provider_for_model(model)
        if provider:
            return provider.provider_name, model
        
        # Default to first available provider
        providers = self.registry.list_providers()
        if providers:
            return providers[0], model
        
        raise ValueError(f"No provider available for model: {model}")
    
    async def generate(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> GenerationResponse:
        """Generate text using the appropriate provider."""
        provider_name, model_name = self._parse_model_spec(model)
        provider = self.registry.get_provider(provider_name)
        
        if not provider:
            raise ValueError(f"Provider not found: {provider_name}")
        
        request = GenerationRequest(
            prompt=prompt,
            model=model_name,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return await provider.generate(request)
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> GenerationResponse:
        """Chat completion using the appropriate provider."""
        provider_name, model_name = self._parse_model_spec(model)
        provider = self.registry.get_provider(provider_name)
        
        if not provider:
            raise ValueError(f"Provider not found: {provider_name}")
        
        chat_messages = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in messages
        ]
        
        request = ChatRequest(
            messages=chat_messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return await provider.chat(request)

    # ---------------------------------------------------------------------
    # Backward Compatibility Aliases
    # ---------------------------------------------------------------------

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> GenerationResponse:
        """Alias for `chat` to keep legacy code working."""
        return await self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    
    async def list_all_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all models from all providers."""
        all_models = {}
        
        for provider_name, provider in self.registry.providers.items():
            try:
                models = await provider.list_models()
                all_models[provider_name] = models
            except Exception as e:
                logger.error(f"Failed to list models from {provider_name}: {e}")
                all_models[provider_name] = []
        
        return all_models
    
    def get_provider_info(self, provider_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a provider."""
        provider = self.registry.get_provider(provider_name)
        if not provider:
            return None
        
        return {
            "name": provider.provider_name,
            "type": provider.get_provider_type().value,
            "capabilities": provider.get_capabilities().__dict__,
            "config": provider.config.__dict__
        }

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ):
        """Stream chat completion using the appropriate provider."""
        provider_name, model_name = self._parse_model_spec(model)
        provider = self.registry.get_provider(provider_name)
        
        if not provider:
            raise ValueError(f"Provider not found: {provider_name}")
        
        chat_messages = [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in messages
        ]
        
        request = ChatRequest(
            messages=chat_messages,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        
        # Stream the response
        async for chunk in provider.stream_chat(request):
            yield chunk


# =============================================================================
# Global Service Instance
# =============================================================================

# Global unified provider service
provider_service = UnifiedProviderService()


def get_provider_service() -> UnifiedProviderService:
    """Get the global provider service instance."""
    return provider_service


def register_custom_provider(provider: BaseProvider) -> None:
    """Register a custom provider."""
    provider_service.registry.register_provider(provider)


# =============================================================================
# Easy Provider Creation
# =============================================================================

def create_custom_provider(
    name: str,
    base_url: str,
    api_key_env_var: str,
    provider_type: ModelProvider = ModelProvider.OPENAI,
    **kwargs
) -> BaseProvider:
    """Create a custom OpenAI-compatible provider."""
    
    # Register configuration
    config = get_unified_config()
    provider_config = ProviderConfig(
        name=name,
        provider_type=provider_type,
        base_url=base_url,
        api_key_env_var=api_key_env_var,
        default_model=kwargs.get("default_model", ""),
        **kwargs
    )
    config.register_provider(provider_config)
    
    # Create and register provider
    if provider_type == ModelProvider.OLLAMA:
        provider = OllamaProvider()
        provider.provider_name = name
    else:
        provider = OpenAICompatibleProvider(name)
    
    register_custom_provider(provider)
    return provider


# Example usage for testing
if __name__ == "__main__":
    import asyncio
    
    async def test_providers():
        service = get_provider_service()
        
        # Test text generation
        try:
            response = await service.generate(
                prompt="What is the capital of France?",
                model="phi3:mini",
                temperature=0.7
            )
            print(f"Generated: {response.text[:100]}...")
        except Exception as e:
            print(f"Generation failed: {e}")
        
        # Test chat
        try:
            response = await service.chat(
                messages=[
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                model="phi3:mini"
            )
            print(f"Chat response: {response.text[:100]}...")
        except Exception as e:
            print(f"Chat failed: {e}")
        
        # List providers
        providers = service.registry.list_providers()
        print(f"Available providers: {providers}")
    
    asyncio.run(test_providers()) 