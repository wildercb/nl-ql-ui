"""
LLM Factory - Compatibility layer for unified architecture.

This module provides model resolution and factory functions for 
backward compatibility with existing code.
"""

import logging
from typing import Tuple, Any

logger = logging.getLogger(__name__)


def resolve_llm(model: str) -> Tuple[Any, str, str]:
    """
    Resolve LLM model specification to service, provider, and model name.
    
    This is a compatibility function that delegates to the unified provider system.
    
    Args:
        model: Model specification (e.g., "phi3:mini" or "groq::llama3-8b")
        
    Returns:
        Tuple of (service, provider, model_name)
    """
    try:
        from .unified_providers import get_provider_service
        
        provider_service = get_provider_service()
        
        # Parse model specification
        if "::" in model:
            provider_name, model_name = model.split("::", 1)
        else:
            # Default to ollama for backward compatibility
            provider_name = "ollama"
            model_name = model
        
        # Return the provider service and model info
        return provider_service, provider_name, model_name
        
    except Exception as e:
        logger.error(f"Failed to resolve LLM: {e}")
        # Return a mock service for compatibility
        return None, "unknown", model 