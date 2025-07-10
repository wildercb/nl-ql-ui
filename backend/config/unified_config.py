"""
Unified Configuration System for MPPW-MCP

This module provides a comprehensive, type-safe configuration system that brings together:
- Agent configurations and capabilities
- Model provider settings and routing
- MCP tool definitions and registration
- Pipeline orchestration settings
- Prompt template management
- Performance and monitoring settings

The design focuses on:
1. Easy extensibility - adding new providers, agents, tools, or pipelines
2. Type safety - all configurations are validated
3. Environment flexibility - development, staging, production
4. Modularity - components can be configured independently
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union, Callable, Type
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import logging
from abc import ABC, abstractmethod
import os

logger = logging.getLogger(__name__)


# =============================================================================
# Core Types and Enums
# =============================================================================

class AgentType(Enum):
    """Types of agents available in the system."""
    REWRITER = "rewriter"
    TRANSLATOR = "translator"  
    REVIEWER = "reviewer"
    OPTIMIZER = "optimizer"
    DATA_REVIEWER = "data_reviewer"
    VALIDATOR = "validator"
    ANALYZER = "analyzer"
    PROCESSOR = "processor"


class AgentCapability(Enum):
    """Capabilities that agents can provide."""
    REWRITE = "rewrite"
    TRANSLATE = "translate"
    REVIEW = "review"
    VALIDATE = "validate"
    OPTIMIZE = "optimize"
    ANALYZE = "analyze"
    PROCESS = "process"
    EXTRACT = "extract"
    TRANSFORM = "transform"
    GENERATE = "generate"


class ModelProvider(Enum):
    """Supported model providers."""
    OLLAMA = "ollama"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class ModelSize(Enum):
    """Model size categories for automatic selection."""
    NANO = "nano"        # Ultra-fast, minimal (1B params)
    SMALL = "small"      # Fast, efficient (3-7B params)
    MEDIUM = "medium"    # Balanced (7-13B params)
    LARGE = "large"      # High quality (13-30B params)
    XLARGE = "xlarge"    # Premium quality (30B+ params)


class PipelineStrategy(Enum):
    """Pipeline execution strategies."""
    FAST = "fast"                    # Translation only
    STANDARD = "standard"            # Rewrite → Translate → Review
    COMPREHENSIVE = "comprehensive"  # Full pipeline with optimization
    ADAPTIVE = "adaptive"            # Context-based strategy selection
    CUSTOM = "custom"               # User-defined pipeline


class PromptStrategy(Enum):
    """Prompt engineering strategies."""
    MINIMAL = "minimal"              # Basic prompts
    DETAILED = "detailed"            # Comprehensive prompts
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Step-by-step reasoning
    FEW_SHOT = "few_shot"           # Example-based prompts
    ADAPTIVE = "adaptive"           # Context-dependent prompts


class ToolCategory(Enum):
    """Categories for organizing MCP tools."""
    TRANSLATION = "translation"
    VALIDATION = "validation"
    ANALYSIS = "analysis"
    SCHEMA = "schema"
    MODEL = "model"
    UTILITY = "utility"
    MONITORING = "monitoring"


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    provider: ModelProvider
    size: ModelSize
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 120.0
    cost_per_token: float = 0.0  # For cost tracking
    capabilities: List[AgentCapability] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def provider_model_name(self) -> str:
        """Get the model name as expected by the provider."""
        return self.metadata.get('provider_model_name', self.name)


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""
    name: str
    provider_type: ModelProvider
    base_url: str
    api_key_env_var: str
    default_model: str
    supported_models: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 120.0
    max_retries: int = 3
    rate_limit: Optional[int] = None  # requests per minute
    enabled: bool = True


@dataclass  
class AgentConfig:
    """Configuration for an agent."""
    name: str
    agent_type: AgentType
    capabilities: List[AgentCapability]
    primary_model: str
    fallback_models: List[str] = field(default_factory=list)
    timeout: float = 30.0
    max_retries: int = 2
    prompt_strategy: PromptStrategy = PromptStrategy.DETAILED
    context_window: int = 4096
    temperature: float = 0.7
    dependencies: List[str] = field(default_factory=list)
    required_inputs: List[str] = field(default_factory=list)
    output_schema: Optional[Dict[str, Any]] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class ToolConfig:
    """Configuration for an MCP tool."""
    name: str
    category: ToolCategory
    description: str
    function_name: str
    module_path: str
    input_schema: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    timeout: float = 30.0
    rate_limit: Optional[int] = None
    cache_ttl: Optional[int] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Configuration for a processing pipeline."""
    name: str
    strategy: PipelineStrategy
    description: str
    agents: List[str]  # Agent names in execution order
    execution_mode: str = "sequential"  # sequential, parallel, conditional
    timeout: float = 60.0
    max_parallel: int = 3
    optimization_level: str = "balanced"  # speed, balanced, quality
    context_sharing: bool = True
    error_recovery: Dict[str, str] = field(default_factory=dict)
    conditions: Dict[str, str] = field(default_factory=dict)  # Conditional execution
    metrics_collection: bool = True
    enabled: bool = True


@dataclass
class PromptConfig:
    """Configuration for prompt templates."""
    name: str
    agent_type: Optional[AgentType] = None
    strategy: PromptStrategy = PromptStrategy.DETAILED
    template_path: Optional[str] = None
    content: Optional[str] = None
    variables: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Main Configuration Classes
# =============================================================================

class UnifiedConfig:
    """
    Main configuration manager that brings together all system configurations.
    
    Provides a single point of access for:
    - Model providers and their configurations
    - Agent definitions and capabilities  
    - MCP tool registrations
    - Pipeline orchestration settings
    - Prompt template management
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path(__file__).parent
        
        # Configuration stores
        self.providers: Dict[str, ProviderConfig] = {}
        self.models: Dict[str, ModelConfig] = {}
        self.agents: Dict[str, AgentConfig] = {}
        self.tools: Dict[str, ToolConfig] = {}
        self.pipelines: Dict[str, PipelineConfig] = {}
        self.prompts: Dict[str, PromptConfig] = {}
        
        # Model routing by size category
        self.model_routing: Dict[ModelSize, str] = {}
        
        # Initialize with default configurations
        self._initialize_default_config()
    
    def _initialize_default_config(self):
        """Initialize with sensible default configurations."""
        
        # Default providers
        self.register_provider(ProviderConfig(
            name="ollama",
            provider_type=ModelProvider.OLLAMA,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            api_key_env_var="",  # Ollama doesn't need API key
            default_model="phi3:mini",
            supported_models=["phi3:mini", "gemma2:2b", "qwen2.5:3b", "llama3.2:3b"],
            timeout=300.0  # Increased to 5 minutes to prevent timeouts
        ))
        
        self.register_provider(ProviderConfig(
            name="groq",
            provider_type=ModelProvider.GROQ,
            base_url="https://api.groq.com/openai/v1",
            api_key_env_var="GROQ_API_KEY",
            default_model="llama-3.1-8b-instant",
            supported_models=["llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            timeout=60.0
        ))
        
        # Default models
        self.register_model(ModelConfig(
            name="phi3:mini",
            provider=ModelProvider.OLLAMA,
            size=ModelSize.SMALL,
            max_tokens=4096,
            temperature=0.7,
            capabilities=[AgentCapability.TRANSLATE, AgentCapability.REWRITE, AgentCapability.REVIEW]
        ))
        
        self.register_model(ModelConfig(
            name="gemma2:2b", 
            provider=ModelProvider.OLLAMA,
            size=ModelSize.NANO,
            max_tokens=2048,
            temperature=0.7,
            capabilities=[AgentCapability.REWRITE, AgentCapability.PROCESS]
        ))
        
        # Model routing
        self.model_routing = {
            ModelSize.NANO: "gemma2:2b",
            ModelSize.SMALL: "phi3:mini", 
            ModelSize.MEDIUM: "phi3:mini",  # Will be upgraded as better models available
            ModelSize.LARGE: "phi3:mini",
            ModelSize.XLARGE: "phi3:mini"
        }
        
        # Default agents
        self.register_agent(AgentConfig(
            name="rewriter",
            agent_type=AgentType.REWRITER,
            capabilities=[AgentCapability.REWRITE],
            primary_model="phi3:mini",
            fallback_models=["gemma2:2b"],
            timeout=120.0,  # Increased to 2 minutes
            prompt_strategy=PromptStrategy.CHAIN_OF_THOUGHT,
            required_inputs=["original_query"]
        ))
        
        self.register_agent(AgentConfig(
            name="translator",
            agent_type=AgentType.TRANSLATOR,
            capabilities=[AgentCapability.TRANSLATE],
            primary_model="phi3:mini",
            fallback_models=["gemma2:2b"],
            timeout=120.0,  # Increased to 2 minutes
            prompt_strategy=PromptStrategy.FEW_SHOT,
            required_inputs=["query", "schema_context"]
        ))
        
        self.register_agent(AgentConfig(
            name="reviewer",
            agent_type=AgentType.REVIEWER,
            capabilities=[AgentCapability.REVIEW, AgentCapability.VALIDATE],
            primary_model="phi3:mini",
            fallback_models=["gemma2:2b"],
            timeout=120.0,  # Increased to 2 minutes
            prompt_strategy=PromptStrategy.DETAILED,
            required_inputs=["graphql_query", "original_query"]
        ))
        
        self.register_agent(AgentConfig(
            name="analyzer",
            agent_type=AgentType.ANALYZER,
            capabilities=[AgentCapability.ANALYZE, AgentCapability.EXTRACT],
            primary_model="phi3:mini",
            fallback_models=["gemma2:2b"],
            timeout=120.0,  # Increased to 2 minutes
            prompt_strategy=PromptStrategy.DETAILED,
            required_inputs=["graphql_query", "original_query"]
        ))
        
        # Default pipelines
        self.register_pipeline(PipelineConfig(
            name="fast",
            strategy=PipelineStrategy.FAST,
            description="Speed-optimized pipeline with translation only",
            agents=["translator"],
            timeout=15.0,
            optimization_level="speed"
        ))
        
        self.register_pipeline(PipelineConfig(
            name="standard",
            strategy=PipelineStrategy.STANDARD,
            description="Standard pipeline with rewrite, translate, and review",
            agents=["rewriter", "translator", "reviewer"],
            timeout=300.0,  # Increased to 5 minutes
            optimization_level="balanced"
        ))

        # Comprehensive pipeline - includes analysis for deeper insight
        self.register_pipeline(PipelineConfig(
            name="comprehensive",
            strategy=PipelineStrategy.COMPREHENSIVE,
            description="Comprehensive pipeline with rewrite, translate, review, and analysis",
            agents=["rewriter", "translator", "reviewer", "analyzer"],
            timeout=600.0,  # Increased to 10 minutes
            optimization_level="quality",
            context_sharing=True,
        ))
        
        # Default tools
        self.register_tool(ToolConfig(
            name="translate_query",
            category=ToolCategory.TRANSLATION,
            description="Translate natural language to GraphQL",
            function_name="translate_query",
            module_path="mcp_server.tools.translation",
            input_schema={
                "type": "object",
                "properties": {
                    "natural_query": {"type": "string"},
                    "schema_context": {"type": "string"},
                    "model": {"type": "string"}
                },
                "required": ["natural_query"]
            }
        ))
        
        self.register_tool(ToolConfig(
            name="validate_graphql",
            category=ToolCategory.VALIDATION,
            description="Validate GraphQL query syntax and structure",
            function_name="validate_graphql",
            module_path="mcp_server.tools.validation",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "schema": {"type": "string"},
                    "strict_mode": {"type": "boolean"}
                },
                "required": ["query"]
            }
        ))
    
    # =========================================================================
    # Registration Methods
    # =========================================================================
    
    def register_provider(self, config: ProviderConfig) -> None:
        """Register a new model provider."""
        self.providers[config.name] = config
        logger.info(f"Registered provider: {config.name} ({config.provider_type.value})")
    
    def register_model(self, config: ModelConfig) -> None:
        """Register a new model configuration."""
        self.models[config.name] = config
        logger.info(f"Registered model: {config.name} ({config.provider.value}, {config.size.value})")
    
    def register_agent(self, config: AgentConfig) -> None:
        """Register a new agent configuration."""
        self.agents[config.name] = config
        logger.info(f"Registered agent: {config.name} ({config.agent_type.value})")
    
    def register_tool(self, config: ToolConfig) -> None:
        """Register a new tool configuration."""
        self.tools[config.name] = config
        logger.info(f"Registered tool: {config.name} ({config.category.value})")
    
    def register_pipeline(self, config: PipelineConfig) -> None:
        """Register a new pipeline configuration."""
        self.pipelines[config.name] = config
        logger.info(f"Registered pipeline: {config.name} ({config.strategy.value})")
    
    def register_prompt(self, config: PromptConfig) -> None:
        """Register a new prompt configuration."""
        self.prompts[config.name] = config
        logger.info(f"Registered prompt: {config.name}")
    
    # =========================================================================
    # Access Methods
    # =========================================================================
    
    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """Get provider configuration by name."""
        return self.providers.get(name)
    
    def get_model(self, name: str) -> Optional[ModelConfig]:
        """Get model configuration by name."""
        return self.models.get(name)
    
    def get_agent(self, name: str) -> Optional[AgentConfig]:
        """Get agent configuration by name."""
        return self.agents.get(name)
    
    def get_tool(self, name: str) -> Optional[ToolConfig]:
        """Get tool configuration by name."""
        return self.tools.get(name)
    
    def get_pipeline(self, name: str) -> Optional[PipelineConfig]:
        """Get pipeline configuration by name."""
        return self.pipelines.get(name)
    
    def get_prompt(self, name: str) -> Optional[PromptConfig]:
        """Get prompt configuration by name."""
        return self.prompts.get(name)
    
    # =========================================================================
    # Smart Selection Methods
    # =========================================================================
    
    def get_model_for_size(self, size: ModelSize) -> Optional[ModelConfig]:
        """Get the best model for a given size category."""
        model_name = self.model_routing.get(size)
        if model_name:
            return self.get_model(model_name)
        return None
    
    def find_agents_by_capability(self, capability: AgentCapability) -> List[AgentConfig]:
        """Find all agents with a specific capability."""
        return [agent for agent in self.agents.values() 
                if capability in agent.capabilities and agent.enabled]
    
    def find_tools_by_category(self, category: ToolCategory) -> List[ToolConfig]:
        """Find all tools in a specific category."""
        return [tool for tool in self.tools.values() 
                if tool.category == category and tool.enabled]
    
    def get_pipeline_agents(self, pipeline_name: str) -> List[AgentConfig]:
        """Get all agent configurations for a pipeline."""
        pipeline = self.get_pipeline(pipeline_name)
        if not pipeline:
            return []
        
        return [self.get_agent(agent_name) for agent_name in pipeline.agents
                if agent_name in self.agents]
    
    def recommend_model_for_agent(self, agent_name: str) -> Optional[str]:
        """Recommend the best model for an agent based on its configuration."""
        agent = self.get_agent(agent_name)
        if not agent:
            return None
        
        # Try primary model first
        if agent.primary_model in self.models:
            return agent.primary_model
        
        # Try fallback models
        for fallback in agent.fallback_models:
            if fallback in self.models:
                return fallback
        
        # Find any model with required capabilities
        for model in self.models.values():
            if any(cap in model.capabilities for cap in agent.capabilities):
                return model.name
        
        return None
    
    # =========================================================================
    # Utility Methods  
    # =========================================================================
    
    def validate_configuration(self) -> List[str]:
        """Validate the entire configuration and return any errors."""
        errors = []
        
        # Validate agent model references
        for agent_name, agent in self.agents.items():
            if agent.primary_model not in self.models:
                errors.append(f"Agent {agent_name} references unknown model: {agent.primary_model}")
            
            for fallback in agent.fallback_models:
                if fallback not in self.models:
                    errors.append(f"Agent {agent_name} references unknown fallback model: {fallback}")
        
        # Validate pipeline agent references
        for pipeline_name, pipeline in self.pipelines.items():
            for agent_name in pipeline.agents:
                if agent_name not in self.agents:
                    errors.append(f"Pipeline {pipeline_name} references unknown agent: {agent_name}")
        
        # Validate model provider references
        for model_name, model in self.models.items():
            provider_name = model.provider.value
            if provider_name not in self.providers:
                errors.append(f"Model {model_name} references unknown provider: {provider_name}")
        
        return errors
    
    def get_stats(self) -> Dict[str, Any]:
        """Get configuration statistics."""
        return {
            "providers": len(self.providers),
            "models": len(self.models),
            "agents": len(self.agents),
            "tools": len(self.tools),
            "pipelines": len(self.pipelines),
            "prompts": len(self.prompts),
            "enabled_agents": len([a for a in self.agents.values() if a.enabled]),
            "enabled_tools": len([t for t in self.tools.values() if t.enabled]),
            "enabled_pipelines": len([p for p in self.pipelines.values() if p.enabled]),
            "validation_errors": len(self.validate_configuration())
        }
    
    def export_config(self) -> Dict[str, Any]:
        """Export the entire configuration as a dictionary."""
        return {
            "providers": {name: config.__dict__ for name, config in self.providers.items()},
            "models": {name: config.__dict__ for name, config in self.models.items()},
            "agents": {name: config.__dict__ for name, config in self.agents.items()},
            "tools": {name: config.__dict__ for name, config in self.tools.items()},
            "pipelines": {name: config.__dict__ for name, config in self.pipelines.items()},
            "prompts": {name: config.__dict__ for name, config in self.prompts.items()},
            "model_routing": {size.value: model for size, model in self.model_routing.items()}
        }


# =============================================================================
# Global Configuration Instance
# =============================================================================

# Global configuration manager instance
config = UnifiedConfig()


def get_unified_config() -> UnifiedConfig:
    """Get the global unified configuration instance."""
    return config


def register_config_extension(extension_func: Callable[[UnifiedConfig], None]) -> None:
    """Register a configuration extension function that will be called during setup."""
    extension_func(config)
    logger.info("Registered configuration extension")


# =============================================================================
# Configuration Builders for Easy Setup
# =============================================================================

class ConfigBuilder:
    """Builder pattern for easy configuration setup."""
    
    def __init__(self, config_instance: Optional[UnifiedConfig] = None):
        self.config = config_instance or get_unified_config()
    
    def add_provider(self, name: str, provider_type: ModelProvider, base_url: str, **kwargs) -> 'ConfigBuilder':
        """Add a new provider with fluent interface."""
        provider_config = ProviderConfig(
            name=name,
            provider_type=provider_type,
            base_url=base_url,
            api_key_env_var=kwargs.get('api_key_env_var', f"{name.upper()}_API_KEY"),
            default_model=kwargs.get('default_model', ''),
            **{k: v for k, v in kwargs.items() if k not in ['api_key_env_var', 'default_model']}
        )
        self.config.register_provider(provider_config)
        return self
    
    def add_model(self, name: str, provider: ModelProvider, size: ModelSize, **kwargs) -> 'ConfigBuilder':
        """Add a new model with fluent interface."""
        model_config = ModelConfig(
            name=name,
            provider=provider,
            size=size,
            **kwargs
        )
        self.config.register_model(model_config)
        return self
    
    def add_agent(self, name: str, agent_type: AgentType, capabilities: List[AgentCapability], 
                  primary_model: str, **kwargs) -> 'ConfigBuilder':
        """Add a new agent with fluent interface."""
        agent_config = AgentConfig(
            name=name,
            agent_type=agent_type,
            capabilities=capabilities,
            primary_model=primary_model,
            **kwargs
        )
        self.config.register_agent(agent_config)
        return self
    
    def build(self) -> UnifiedConfig:
        """Build and return the configuration."""
        return self.config


# Example usage for documentation
if __name__ == "__main__":
    # Example of extending configuration
    builder = ConfigBuilder()
    
    # Add a new provider
    builder.add_provider(
        name="anthropic",
        provider_type=ModelProvider.ANTHROPIC,
        base_url="https://api.anthropic.com/v1",
        api_key_env_var="ANTHROPIC_API_KEY",
        default_model="claude-3-haiku-20240307"
    )
    
    # Add models for that provider
    builder.add_model(
        name="claude-3-haiku-20240307",
        provider=ModelProvider.ANTHROPIC,
        size=ModelSize.MEDIUM,
        max_tokens=4096,
        capabilities=[AgentCapability.TRANSLATE, AgentCapability.REVIEW, AgentCapability.ANALYZE]
    )
    
    # Add a custom agent
    builder.add_agent(
        name="advanced_analyzer",
        agent_type=AgentType.ANALYZER,
        capabilities=[AgentCapability.ANALYZE, AgentCapability.EXTRACT],
        primary_model="claude-3-haiku-20240307",
        timeout=45.0,
        prompt_strategy=PromptStrategy.CHAIN_OF_THOUGHT
    )
    
    config = builder.build()
    print(f"Configuration ready with {config.get_stats()}") 