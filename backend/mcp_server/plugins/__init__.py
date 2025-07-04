"""
MCP Server Plugin Architecture

This module provides a sophisticated plugin system for the MCP server that enables:
- Type-safe plugin interfaces
- Plugin metadata management
- Enhanced testing and debugging capabilities
"""

from typing import Dict, List, Type, Optional, Any
import logging
from pathlib import Path

from .base import BasePlugin, PluginMetadata, PluginCategory

logger = logging.getLogger(__name__)

def get_available_plugins() -> List[str]:
    """Get list of all available plugins."""
    # For now, return empty list until plugins are implemented
    return []

def get_plugin_stats() -> Dict[str, Any]:
    """Get comprehensive plugin system statistics."""
    return {
        'plugin_count': 0,
        'categories': {
            category.value: 0
            for category in PluginCategory
        },
        'status': 'base_architecture_ready'
    }

__all__ = [
    'BasePlugin',
    'PluginMetadata', 
    'PluginCategory',
    'get_available_plugins',
    'get_plugin_stats'
] 