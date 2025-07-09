"""
Modular Prompt System for MPPW-MCP

This package provides a scalable, modular system for managing prompts
across different agents, domains, and strategies. It allows easy editing
and extension of prompts without code changes.

Structure:
- agents/: Agent-specific prompt templates
- domains/: Domain-specific prompts (e-commerce, social, etc.)
- strategies/: Strategy-specific prompts (minimal, detailed, chain-of-thought)
"""

from .manager import PromptManager
from .loader import PromptLoader
from .types import PromptTemplate, PromptStrategy, PromptContext

__all__ = [
    'PromptManager',
    'PromptLoader', 
    'PromptTemplate',
    'PromptStrategy',
    'PromptContext'
]

# Global prompt manager instance
prompt_manager = PromptManager()

def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager instance."""
    return prompt_manager 