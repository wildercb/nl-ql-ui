"""
Advanced Agent Framework for MPPW-MCP

This package provides a sophisticated agent orchestration system with:
- Automatic agent discovery and registration
- Configurable pipeline execution
- Dynamic context engineering
- Performance monitoring and optimization
"""

from .registry import AgentRegistry, agent_registry
from .base import BaseAgent, AgentContext, AgentMetadata, AgentCapability
from .pipeline import Pipeline, PipelineConfig, ExecutionResult
from .context import ContextEngineering, PromptStrategy, ContextManager

# Global registry instance
registry = agent_registry

__all__ = [
    # Core classes
    "BaseAgent",
    "AgentContext", 
    "AgentMetadata",
    "AgentCapability",
    
    # Registry system
    "AgentRegistry",
    "registry",
    
    # Pipeline system
    "Pipeline",
    "PipelineConfig", 
    "ExecutionResult",
    
    # Context engineering
    "ContextEngineering",
    "PromptStrategy",
    "ContextManager"
]

def discover_agents():
    """Auto-discover and register all agents in the agents directory."""
    registry.discover_agents()

def get_available_agents():
    """Get list of all registered agents."""
    return registry.list_agents()

def create_pipeline(config: dict):
    """Create a new pipeline from configuration."""
    return Pipeline.from_config(config) 