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

from .unified_prompts import (
    PromptTemplate,
    PromptStrategy,
    PromptTemplates,
    UnifiedPromptManager,
    get_prompt_manager,
    get_prompt_for_agent,
    load_template_from_file
)

# Compatibility aliases
PromptManager = UnifiedPromptManager
PromptLoader = UnifiedPromptManager
PromptContext = dict  # Simple alias for dict

__all__ = [
    'PromptManager',
    'PromptLoader', 
    'PromptTemplate',
    'PromptStrategy',
    'PromptContext',
    'PromptTemplates',
    'UnifiedPromptManager',
    'get_prompt_manager',
    'get_prompt_for_agent'
]

# Global prompt manager instance
prompt_manager = get_prompt_manager() 