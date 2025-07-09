# Configuration Guide - Unified Architecture

## Overview

The MPPW-MCP system uses a unified configuration system that provides type-safe, centralized management of all system components. This guide covers how to configure agents, providers, models, tools, and pipelines.

## Unified Configuration System

### Core Concepts

The unified configuration system (`backend/config/unified_config.py`) provides:

- **Type Safety**: All configuration uses dataclasses and enums
- **Centralized Management**: Single source of truth for all settings
- **Easy Extension**: Simple API for adding new components
- **Validation**: Comprehensive validation and error checking
- **Environment Integration**: Support for environment variables

### Configuration Builder

Use the `ConfigBuilder` for fluent configuration setup:

```python
from backend.config.unified_config import ConfigBuilder, ModelProvider, ModelSize, AgentType

builder = ConfigBuilder()
```

## Model Configuration

### Adding Models

Models are selected via the UI, but can also be configured programmatically:

```python
# Add a model (typically done via UI)
builder.add_model(
    name="gpt-4",
    provider=ModelProvider.OPENAI,
    size=ModelSize.LARGE,
    capabilities=["translation", "rewriting", "analysis", "review"]
)

builder.add_model(
    name="llama3:8b",
    provider=ModelProvider.OLLAMA, 
    size=ModelSize.MEDIUM,
    capabilities=["translation", "analysis"]
)
```

### Model Enums

```python
class ModelProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"
    GROQ = "groq"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"

class ModelSize(Enum):
    SMALL = "small"      # < 1B parameters
    MEDIUM = "medium"    # 1-10B parameters  
    LARGE = "large"      # 10-100B parameters
    XLARGE = "xlarge"    # > 100B parameters
```

## Provider Configuration

### Adding Providers

```python
# OpenAI Provider
builder.add_provider(
    name="openai",
    provider_type=ModelProvider.OPENAI,
    settings={
        "api_key": "your-openai-api-key",
        "base_url": "https://api.openai.com/v1",  # optional
        "timeout": 30
    }
)

# Ollama Provider (local)
builder.add_provider(
    name="ollama",
    provider_type=ModelProvider.OLLAMA,
    settings={
        "base_url": "http://localhost:11434",
        "timeout": 60
    }
)

# Groq Provider
builder.add_provider(
    name="groq",
    provider_type=ModelProvider.GROQ,
    settings={
        "api_key": "your-groq-api-key",
        "base_url": "https://api.groq.com/openai/v1"
    }
)
```

### Environment Variables

Providers can use environment variables for sensitive data:

```bash
# .env file
OPENAI_API_KEY=your-openai-key
GROQ_API_KEY=your-groq-key
ANTHROPIC_API_KEY=your-anthropic-key
OLLAMA_BASE_URL=http://localhost:11434
```

## Agent Configuration

### Agent Types

```python
class AgentType(Enum):
    TRANSLATOR = "translator"
    REWRITER = "rewriter" 
    REVIEWER = "reviewer"
    ANALYZER = "analyzer"

class AgentCapability(Enum):
    TRANSLATION = "translation"
    REWRITING = "rewriting"
    REVIEW = "review"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
```

### Adding Agents

**Important**: Agents use models selected in the UI, not hardcoded defaults.

```python
# Configure agent behavior (models selected via UI)
builder.add_agent(
    agent_type=AgentType.TRANSLATOR,
    capabilities=[AgentCapability.TRANSLATION, AgentCapability.VALIDATION],
    primary_model=None,  # Selected via UI
    fallback_models=[],  # Configured via UI
    prompt_strategy=PromptStrategy.DETAILED,
    max_retries=3,
    timeout=30,
    custom_settings={
        "temperature": 0.3,
        "max_tokens": 1000
    }
)
```

### Prompt Strategies

```python
class PromptStrategy(Enum):
    DETAILED = "detailed"           # Comprehensive prompts
    MINIMAL = "minimal"             # Concise prompts
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Step-by-step reasoning
    FEW_SHOT = "few_shot"          # Example-based prompts
```

## Tool Configuration

### Tool Categories

```python
class ToolCategory(Enum):
    TRANSLATION = "translation"
    VALIDATION = "validation"
    ANALYSIS = "analysis"
    PROCESSING = "processing"
    UTILITY = "utility"
    MCP = "mcp"
```

### Adding Tools

```python
builder.add_tool(
    name="translate_query",
    category=ToolCategory.TRANSLATION,
    settings={
        "max_length": 1000,
        "timeout": 30
    }
)

builder.add_tool(
    name="validate_graphql",
    category=ToolCategory.VALIDATION,
    settings={
        "strict_mode": True,
        "check_syntax": True,
        "check_semantics": True
    }
)
```

## Pipeline Configuration

### Pipeline Strategies

```python
class PipelineStrategy(Enum):
    SEQUENTIAL = "sequential"      # Execute agents in order
    PARALLEL = "parallel"          # Execute agents concurrently
    CONDITIONAL = "conditional"    # Execute based on conditions
    ITERATIVE = "iterative"        # Execute with feedback loops
```

### Adding Pipelines

```python
builder.add_pipeline(
    name="translation_pipeline",
    agents=[AgentType.TRANSLATOR, AgentType.REVIEWER],
    strategy=PipelineStrategy.SEQUENTIAL,
    settings={
        "max_iterations": 3,
        "quality_threshold": 0.8
    }
)

builder.add_pipeline(
    name="analysis_pipeline", 
    agents=[AgentType.ANALYZER],
    strategy=PipelineStrategy.PARALLEL,
    settings={
        "concurrent_limit": 5
    }
)
```

## Complete Configuration Example

```python
from backend.config.unified_config import (
    ConfigBuilder, set_global_config,
    ModelProvider, ModelSize, AgentType, AgentCapability,
    PromptStrategy, ToolCategory, PipelineStrategy
)

def setup_production_config():
    """Complete production configuration example."""
    builder = ConfigBuilder()
    
    # Providers
    builder.add_provider("openai", ModelProvider.OPENAI, {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "timeout": 30
    })
    
    builder.add_provider("ollama", ModelProvider.OLLAMA, {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "timeout": 60
    })
    
    # Models (typically configured via UI)
    builder.add_model("gpt-4", ModelProvider.OPENAI, ModelSize.LARGE, 
                     ["translation", "rewriting", "review"])
    builder.add_model("llama3:8b", ModelProvider.OLLAMA, ModelSize.MEDIUM,
                     ["translation", "analysis"])
    
    # Agents (models selected via UI)
    builder.add_agent(
        AgentType.TRANSLATOR,
        [AgentCapability.TRANSLATION],
        prompt_strategy=PromptStrategy.DETAILED,
        max_retries=3
    )
    
    builder.add_agent(
        AgentType.REVIEWER,
        [AgentCapability.REVIEW, AgentCapability.VALIDATION],
        prompt_strategy=PromptStrategy.CHAIN_OF_THOUGHT,
        max_retries=2
    )
    
    # Tools
    builder.add_tool("translate_query", ToolCategory.TRANSLATION, 
                    {"max_length": 1000})
    builder.add_tool("validate_graphql", ToolCategory.VALIDATION,
                    {"strict_mode": True})
    
    # Pipelines
    builder.add_pipeline("translation_pipeline", 
                        [AgentType.TRANSLATOR, AgentType.REVIEWER],
                        PipelineStrategy.SEQUENTIAL)
    
    # Build and set global config
    config = builder.build()
    set_global_config(config)
    
    return config
```

## Accessing Configuration

### Get Global Configuration

```python
from backend.config.unified_config import get_config

config = get_config()
```

### Access Specific Components

```python
# Get model configuration
model_config = config.get_model_config("gpt-4")

# Get agent configuration  
agent_config = config.get_agent_config(AgentType.TRANSLATOR)

# Get tool configuration
tool_config = config.get_tool_config("translate_query")

# Get pipeline configuration
pipeline_config = config.get_pipeline_config("translation_pipeline")

# Get provider configuration
provider_config = config.get_provider_config("openai")
```

### Smart Model Selection

```python
# Get best model for capability
best_model = config.get_best_model_for_capability("translation")

# Get model with fallbacks
model_with_fallbacks = config.get_model_with_fallbacks("gpt-4")

# Get models by provider
ollama_models = config.get_models_by_provider(ModelProvider.OLLAMA)
```

## Configuration Validation

### Built-in Validation

```python
# Validate configuration
validation_result = config.validate()

if not validation_result.is_valid:
    print("Configuration errors:")
    for error in validation_result.errors:
        print(f"- {error}")
```

### Custom Validation Rules

```python
class CustomValidator:
    def validate_model_availability(self, config):
        """Custom validation for model availability."""
        errors = []
        
        for model_name, model_config in config.models.items():
            if model_config.provider == ModelProvider.OLLAMA:
                # Check if Ollama model is available
                if not self.check_ollama_model(model_name):
                    errors.append(f"Ollama model {model_name} not available")
        
        return errors
```

## Environment-Specific Configuration

### Development Configuration

```python
def setup_dev_config():
    """Development configuration with local services."""
    builder = ConfigBuilder()
    
    # Local Ollama only
    builder.add_provider("ollama", ModelProvider.OLLAMA, {
        "base_url": "http://localhost:11434"
    })
    
    # Local models
    builder.add_model("llama3:8b", ModelProvider.OLLAMA, ModelSize.MEDIUM,
                     ["translation", "analysis"])
    
    return builder.build()
```

### Production Configuration

```python
def setup_prod_config():
    """Production configuration with cloud services."""
    builder = ConfigBuilder()
    
    # Cloud providers with API keys
    builder.add_provider("openai", ModelProvider.OPENAI, {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "timeout": 30
    })
    
    builder.add_provider("groq", ModelProvider.GROQ, {
        "api_key": os.getenv("GROQ_API_KEY"),
        "timeout": 15
    })
    
    # Production models
    builder.add_model("gpt-4", ModelProvider.OPENAI, ModelSize.LARGE,
                     ["translation", "rewriting", "review"])
    builder.add_model("llama3-70b", ModelProvider.GROQ, ModelSize.LARGE,
                     ["translation", "analysis"])
    
    return builder.build()
```

## Configuration Hot Reloading

The unified configuration supports runtime updates:

```python
# Update model configuration
config.update_model_config("gpt-4", {
    "temperature": 0.2,
    "max_tokens": 1500
})

# Add new provider at runtime
config.add_provider_config("new_provider", ProviderConfig(...))

# Reload configuration from file
config.reload_from_file("config.json")
```

## Best Practices

### Security
- Use environment variables for API keys
- Never commit secrets to version control
- Validate all external inputs
- Use secure defaults

### Performance
- Cache configuration objects
- Minimize configuration lookups in hot paths
- Use connection pooling for providers
- Monitor provider response times

### Maintainability
- Use the ConfigBuilder for setup
- Validate configuration at startup
- Document custom settings
- Use meaningful names for components

### Model Selection
- **Always use UI for model selection** - never hardcode models
- Configure fallback strategies via UI
- Monitor model performance and costs
- Use appropriate model sizes for tasks

## Troubleshooting

### Common Issues

1. **Configuration not found**: Ensure `set_global_config()` called
2. **Model not available**: Check provider connection and model name
3. **API key errors**: Verify environment variables
4. **Validation failures**: Check configuration against schema

### Debug Configuration

```python
# Enable configuration debugging
import logging
logging.getLogger("unified_config").setLevel(logging.DEBUG)

# Print configuration summary
config = get_config()
print(f"Models: {list(config.models.keys())}")
print(f"Providers: {list(config.providers.keys())}")
print(f"Agents: {list(config.agents.keys())}")
print(f"Tools: {list(config.tools.keys())}")
```

## Migration from Legacy Configuration

The unified configuration system replaces the previous scattered configuration files:

- `config/settings.py` → `config/unified_config.py`
- `config/agent_config.py` → Agent configuration in unified system
- Various service configs → Provider configurations in unified system

For migration assistance, see the [Refactoring Summary](REFACTORING_SUMMARY.md).

This unified configuration system provides a solid foundation for managing complex AI agent systems while maintaining type safety and ease of use. 