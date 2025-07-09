"""
Advanced Agent Framework for MPPW-MCP

This package provides a sophisticated agent orchestration system with:
- Automatic agent discovery and registration
- Configurable pipeline execution
- Dynamic context engineering
- Performance monitoring and optimization
"""

from .unified_agents import (
    AgentFactory,
    AgentContext,
    AgentResult,
    AgentType,
    AgentCapability,
    BaseAgent,
    PipelineExecutor,
    RewriterAgent,
    TranslatorAgent,
    ReviewerAgent,
    AnalyzerAgent
)

# Compatibility aliases
AgentRegistry = AgentFactory
registry = AgentFactory
agent_registry = AgentFactory

__all__ = [
    # Core classes
    "BaseAgent",
    "AgentContext", 
    "AgentResult",
    "AgentType",
    "AgentCapability",
    
    # Factory and registry
    "AgentFactory",
    "AgentRegistry",
    "registry",
    "agent_registry",
    
    # Pipeline system
    "PipelineExecutor",
    
    # Concrete agents
    "RewriterAgent",
    "TranslatorAgent", 
    "ReviewerAgent",
    "AnalyzerAgent"
]

def discover_agents():
    """Auto-discover and register all agents."""
    # This is handled automatically by the unified system
    pass

def get_available_agents():
    """Get list of all registered agents."""
    return AgentFactory.list_available_agents()

def create_pipeline(config: dict):
    """Create a new pipeline from configuration."""
    executor = PipelineExecutor()
    return executor 