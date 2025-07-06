from __future__ import annotations

"""Service wrapper for providers exposing the OpenAI-compatible chat API.

Currently supports:
• Groq – https://console.groq.com/docs
• OpenRouter – https://openrouter.ai/docs

Usage is identical to `OllamaService` but requires a provider identifier in
{groq, openrouter} so that the correct base_url and auth header are applied.
"""

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import os

import httpx

from config.settings import get_settings
from services.llm_tracking_service import get_tracking_service
from services.ollama_service import GenerationResult  # Re-use existing dataclass

logger = logging.getLogger(__name__)


_PROVIDER_ENDPOINTS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
}


class OpenAIProxyService:
    """A minimal async client for OpenAI-spec endpoints (Groq, OpenRouter)."""

    def __init__(self, provider: str):
        provider = provider.lower()
        if provider not in _PROVIDER_ENDPOINTS:
            raise ValueError(f"Unsupported provider: {provider}")

        self.provider = provider
        info = _PROVIDER_ENDPOINTS[provider]
        self.base_url = info["base_url"].rstrip("/")
        self.api_key = os.environ.get(info["api_key_env"])
        if not self.api_key:
            raise RuntimeError(f"{info['api_key_env']} environment variable must be set for provider '{provider}'.")

        self.tracking_service = get_tracking_service()
        self.timeout = 120.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _request(self, payload: Dict[str, Any], stream: bool = False):
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            url = f"{self.base_url}/chat/completions"
            response = await client.post(url, headers=headers, json=payload, stream=stream)
            if response.status_code != 200:
                text = await response.aread() if stream else response.text
                raise RuntimeError(f"{self.provider} API error {response.status_code}: {text[:200]}")
            return response

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        interaction_type: str = "chat_completion",
    ) -> GenerationResult:
        """One-shot completion (non-stream) returning GenerationResult."""
        session_id = session_id or f"{self.provider}-{int(time.time())}"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        async with self.tracking_service.track_interaction(
            session_id=session_id,
            model=model,
            provider=self.provider,
            interaction_type=interaction_type,
            user_id=user_id,
            context_data={"temperature": temperature, "max_tokens": max_tokens},
        ) as tracker:
            tracker.set_prompt("\n".join(f"{m['role']}: {m['content']}" for m in messages))
            tracker.set_parameters(temperature=temperature, max_tokens=max_tokens)
            start = time.time()
            try:
                response = await self._request(payload, stream=False)
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                elapsed = time.time() - start
                usage = data.get("usage", {})
                tracker.set_response(text)
                tracker.set_performance_metrics(
                    processing_time=elapsed,
                    tokens_used=usage.get("total_tokens"),
                    prompt_tokens=usage.get("prompt_tokens"),
                    response_tokens=usage.get("completion_tokens"),
                )
                return GenerationResult(
                    text=text,
                    model=model,
                    processing_time=elapsed,
                    tokens_used=usage.get("total_tokens"),
                    prompt_tokens=usage.get("prompt_tokens"),
                    response_tokens=usage.get("completion_tokens"),
                    finish_reason=data["choices"][0].get("finish_reason"),
                    metadata={},
                )
            except Exception as e:
                tracker.set_error(str(e))
                raise

    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Yield streaming tokens in the same shape as OllamaService."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        response = await self._request(payload, stream=True)
        async for line in response.aiter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                chunk = line[len("data: "):]
                if chunk.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(chunk)
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield {"message": {"content": delta}}
                except Exception:
                    continue 