"""FastMCP Prompts for GraphQL Translation."""

from .translation import register_translation_prompts
from .analysis import register_analysis_prompts

__all__ = [
    "register_translation_prompts",
    "register_analysis_prompts"
] 