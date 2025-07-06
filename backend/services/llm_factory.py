from __future__ import annotations

"""Factory and utilities for routing LLM requests to the correct provider service.

This indirection allows the rest of the codebase to simply pass a model string
which *may* include a provider prefix (e.g. "groq::llama3-8b-8192").  The
factory will return:
    1. the appropriate service instance (OllamaService, OpenAIProxyService, …)
    2. the *stripped* model name that the downstream provider expects.
    3. the provider name ("ollama", "groq", …) for convenience.
"""

import functools
from typing import Tuple

from config.settings import get_settings

# Import lazily to avoid heavy dependencies at import-time
from .ollama_service import OllamaService

try:
    # OpenAI-compatible proxy (Groq, OpenRouter, etc.)
    from .openai_proxy_service import OpenAIProxyService  # type: ignore
except ModuleNotFoundError:  # pragma: no cover – created later in this diff
    OpenAIProxyService = None  # type: ignore

PROVIDER_PREFIX_SEPARATOR = "::"

def _parse_provider(model: str) -> Tuple[str, str]:
    """Extract provider + model.

    If no explicit provider prefix is present, assume "ollama" to preserve
    backward compatibility.
    """
    if PROVIDER_PREFIX_SEPARATOR in model:
        provider, model_name = model.split(PROVIDER_PREFIX_SEPARATOR, 1)
        return provider.lower(), model_name
    # Fallback – default provider
    return "ollama", model

@functools.lru_cache(maxsize=None)
def _get_service_for_provider(provider: str):
    """Return (and cache) the service instance for the given provider."""
    if provider == "ollama":
        return OllamaService()
    if provider in ("groq", "openrouter"):
        if OpenAIProxyService is None:
            raise RuntimeError("OpenAIProxyService not available – ensure file is imported")
        return OpenAIProxyService(provider)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def resolve_llm(model: str):
    """Helper returning (service, provider, stripped_model_name)."""
    provider, stripped_model = _parse_provider(model)
    service = _get_service_for_provider(provider)
    return service, provider, stripped_model 