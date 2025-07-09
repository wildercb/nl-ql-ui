# Unified Architecture Guide

## Overview

The MPPW-MCP system has been completely refactored into a unified, modular architecture that emphasizes type safety, extensibility, and maintainability. This guide covers the new architecture and how to work with it.

## Core Principles

### 1. Unified Configuration
- **Single Source of Truth**: All configuration is managed through `backend/config/unified_config.py`
- **Type Safety**: Comprehensive dataclasses and enums ensure type safety
- **Easy Extension**: Adding new providers, agents, or tools requires minimal configuration

### 2. Standardized Interfaces
- **Consistent Patterns**: All components follow the same interface patterns
- **Predictable Behavior**: Every component implements standard methods and properties
- **Easy Testing**: Standardized interfaces make testing straightforward

### 3. Modular Design
- **Loose Coupling**: Components can work independently
- **High Cohesion**: Related functionality is grouped together
- **Clear Boundaries**: Well-defined interfaces between components

## Architecture Components

### Unified Configuration System (`backend/config/unified_config.py`)

The configuration system is the foundation of the new architecture:

```python
from backend.config.unified_config import get_config

# Get the global configuration
config = get_config()

# Access specific configurations
model_config = config.get_model_config("gpt-4")
agent_config = config.get_agent_config(AgentType.TRANSLATOR)
tool_config = config.get_tool_config("translate_query")
```

#### Key Components:

- **Enums**: AgentType, AgentCapability, ModelProvider, ModelSize, PipelineStrategy, PromptStrategy, ToolCategory
- **Configuration Classes**: ModelConfig, ProviderConfig, AgentConfig, ToolConfig, PipelineConfig, PromptConfig
- **Main Class**: UnifiedConfig with registration and access methods
- **Builder**: ConfigBuilder for fluent configuration setup

### Standardized Prompt System (`backend/prompts/unified_prompts.py`)

The prompt system provides intelligent template management:

```python
from backend.prompts.unified_prompts import get_prompt_manager

prompt_manager = get_prompt_manager()

# Get a prompt for a specific agent and strategy
prompt = prompt_manager.get_prompt(
    agent_type=AgentType.TRANSLATOR,
    strategy=PromptStrategy.DETAILED,
    context={"source_language": "en", "target_language": "es"}
)
```

#### Features:

- **Template Engine**: Jinja2-based templating with validation
- **Strategy Selection**: Different prompt strategies (detailed, minimal, chain-of-thought)
- **Context Integration**: Dynamic context injection
- **Validation**: Template syntax and variable validation

### Unified Agent Framework (`backend/agents/unified_agents.py`)

The agent framework provides consistent agent behavior:

```python
from backend.agents.unified_agents import AgentFactory, PipelineExecutor

# Create an agent
agent = AgentFactory.create_agent(AgentType.TRANSLATOR)

# Execute a pipeline
executor = PipelineExecutor()
result = await executor.execute_pipeline(
    pipeline_name="translation_pipeline",
    context=AgentContext(query="Hello world", session_id="session_123")
)
```

#### Key Components:

- **AgentContext**: Data flow management with session tracking
- **AgentResult**: Standardized execution results
- **BaseAgent**: Abstract base class with consistent interface
- **Concrete Agents**: RewriterAgent, TranslatorAgent, ReviewerAgent, AnalyzerAgent
- **AgentFactory**: Instance creation and management
- **PipelineExecutor**: Pipeline orchestration

### Modernized MCP Tools (`backend/mcp_server/tools/unified_tools.py`)

The tools system provides MCP-compatible functionality:

```python
from backend.mcp_server.tools.unified_tools import get_tool_registry

registry = get_tool_registry()

# Get a tool
tool = registry.get_tool("translate_query")

# Execute the tool
result = await tool.execute({"query": "Hello", "target_lang": "es"})
```

#### Features:

- **ToolRegistry**: Automatic discovery and registration
- **BaseTool**: Consistent tool interface
- **MCP Integration**: Full MCP server compatibility
- **Comprehensive Tools**: Translation, validation, model management, configuration

### Streamlined Provider System (`backend/services/unified_providers.py`)

The provider system offers unified access to different LLM providers:

```python
from backend.services.unified_providers import get_provider_service

provider_service = get_provider_service()

# Generate text
response = await provider_service.generate(
    provider_name="openai",
    model="gpt-4",
    prompt="Translate this text...",
    max_tokens=150
)
```

#### Features:

- **BaseProvider**: Abstract provider interface
- **Multiple Providers**: Ollama, OpenAI-compatible (Groq, OpenRouter, etc.)
- **ProviderRegistry**: Provider management
- **UnifiedProviderService**: Single interface for all providers
- **Easy Extension**: Simple provider addition

## Implementation Patterns

### Adding a New Agent

1. **Define Agent Type** (if new):
```python
# In backend/config/unified_config.py
class AgentType(Enum):
    TRANSLATOR = "translator"
    REWRITER = "rewriter"
    REVIEWER = "reviewer"
    ANALYZER = "analyzer"
    NEW_AGENT = "new_agent"  # Add this
```

2. **Create Agent Class**:
```python
# In backend/agents/unified_agents.py
class NewAgent(BaseAgent):
    def __init__(self, config: UnifiedConfig):
        super().__init__(config, AgentType.NEW_AGENT)
    
    async def execute(self, context: AgentContext) -> AgentResult:
        # Implementation here
        pass
```

3. **Register in Factory**:
```python
# In AgentFactory.create_agent method
if agent_type == AgentType.NEW_AGENT:
    return NewAgent(config)
```

4. **Add Prompts**:
```python
# In backend/prompts/unified_prompts.py
# Add templates for the new agent
```

### Adding a New Provider

1. **Create Provider Class**:
```python
# In backend/services/unified_providers.py
class NewProvider(BaseProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        # Implementation here
        pass
```

2. **Register Provider**:
```python
# In get_provider_service function
registry.register_provider("new_provider", NewProvider(provider_config))
```

### Adding a New Tool

1. **Create Tool Class**:
```python
# In backend/mcp_server/tools/unified_tools.py
class NewTool(BaseTool):
    def __init__(self, config: UnifiedConfig):
        super().__init__("new_tool", "Description", ToolCategory.PROCESSING, config)
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        # Implementation here
        pass
```

2. **Auto-Registration**: The tool will be automatically registered by the ToolRegistry

## Configuration Examples

### Basic Configuration Setup

```python
from backend.config.unified_config import ConfigBuilder, get_config

# Build configuration
builder = ConfigBuilder()

# Add models
builder.add_model("gpt-4", ModelProvider.OPENAI, ModelSize.LARGE, ["translation", "rewriting"])
builder.add_model("llama3:8b", ModelProvider.OLLAMA, ModelSize.MEDIUM, ["analysis"])

# Add providers
builder.add_provider("openai", ModelProvider.OPENAI, {"api_key": "your-key"})
builder.add_provider("ollama", ModelProvider.OLLAMA, {"base_url": "http://localhost:11434"})

# Add agents
builder.add_agent(AgentType.TRANSLATOR, ["translation"], "gpt-4", PromptStrategy.DETAILED)

# Add tools
builder.add_tool("translate_query", ToolCategory.TRANSLATION, {"max_length": 1000})

# Add pipelines
builder.add_pipeline("translation_pipeline", [AgentType.TRANSLATOR], PipelineStrategy.SEQUENTIAL)

# Build and set as global
config = builder.build()
```

### Advanced Configuration

```python
# Custom agent configuration
agent_config = AgentConfig(
    agent_type=AgentType.TRANSLATOR,
    capabilities=[AgentCapability.TRANSLATION, AgentCapability.VALIDATION],
    primary_model="gpt-4",
    fallback_models=["gpt-3.5-turbo", "llama3:8b"],
    prompt_strategy=PromptStrategy.CHAIN_OF_THOUGHT,
    max_retries=3,
    timeout=30,
    custom_settings={"temperature": 0.3, "max_tokens": 1000}
)

config.register_agent_config(agent_config)
```

## Testing

The unified architecture makes testing straightforward:

```python
import pytest
from backend.config.unified_config import ConfigBuilder
from backend.agents.unified_agents import AgentFactory, AgentContext

@pytest.fixture
def test_config():
    builder = ConfigBuilder()
    # Add test configuration
    return builder.build()

@pytest.mark.asyncio
async def test_translator_agent(test_config):
    agent = AgentFactory.create_agent(AgentType.TRANSLATOR, test_config)
    context = AgentContext(query="Hello world", session_id="test")
    
    result = await agent.execute(context)
    
    assert result.success
    assert result.output
```

## Migration from Legacy System

The new system maintains backward compatibility while providing migration paths:

1. **Gradual Migration**: Use new components alongside existing ones
2. **Configuration Bridge**: Legacy configurations can be converted to new format
3. **Interface Compatibility**: Existing interfaces are preserved where possible

## Best Practices

### Configuration Management
- Use the global configuration instance for consistency
- Register all configurations at startup
- Use environment variables for sensitive data
- Validate configurations early

### Agent Development
- Always extend BaseAgent for consistency
- Implement proper error handling
- Use the context system for data flow
- Add comprehensive logging

### Tool Development
- Follow the BaseTool interface
- Ensure MCP compatibility
- Add proper parameter validation
- Document tool capabilities

### Provider Integration
- Implement BaseProvider interface
- Handle provider-specific errors gracefully
- Add retry logic for reliability
- Monitor provider performance

## Performance Considerations

### Caching
- Prompt templates are cached after first load
- Configuration is cached at startup
- Provider instances are reused

### Async Operations
- All operations are async where possible
- Proper connection pooling for providers
- Batch operations when beneficial

### Resource Management
- Proper cleanup of resources
- Connection pooling for databases
- Memory management for large operations

## Troubleshooting

### Common Issues

1. **Configuration Errors**: Check configuration validation
2. **Provider Connection Issues**: Verify provider settings and connectivity
3. **Prompt Template Errors**: Validate template syntax and variables
4. **Agent Execution Failures**: Check logs for detailed error information

### Debugging

- Use the built-in logging system
- Enable debug mode for detailed output
- Use the configuration validation tools
- Monitor performance metrics

## Future Extensions

The architecture is designed for easy extension:

- **New Agent Types**: Add to AgentType enum and implement agent class
- **New Providers**: Implement BaseProvider interface
- **New Tools**: Extend BaseTool for MCP compatibility
- **New Strategies**: Add to PromptStrategy enum and implement templates
- **Custom Pipelines**: Use PipelineConfig for complex workflows

## Architecture Diagrams

### System Architecture Overview

```mermaid
graph TB
    subgraph "Unified Configuration Layer"
        UC[Unified Config]
        CB[Config Builder]
        MC[Model Config]
        PC[Provider Config]
        AC[Agent Config]
        TC[Tool Config]
        PLC[Pipeline Config]
        PRC[Prompt Config]
    end

    subgraph "Prompt Management System"
        UPM[Unified Prompt Manager]
        PT[Prompt Templates]
        PS[Prompt Strategies]
        TE[Template Engine]
    end

    subgraph "Agent Framework"
        AF[Agent Factory]
        PE[Pipeline Executor]
        BA[Base Agent]
        TA[Translator Agent]
        RA[Rewriter Agent]
        REA[Reviewer Agent]
        AA[Analyzer Agent]
        ACTX[Agent Context]
        AR[Agent Result]
    end

    subgraph "MCP Tools System"
        TR[Tool Registry]
        BT[Base Tool]
        TQT[Translate Query Tool]
        VT[Validation Tool]
        MT[Model Tool]
        CT[Config Tool]
        PQT[Pipeline Query Tool]
    end

    subgraph "Provider System"
        UPS[Unified Provider Service]
        PR[Provider Registry]
        BP[Base Provider]
        OP[Ollama Provider]
        OAP[OpenAI Provider]
        GR[Groq Provider]
        OR[OpenRouter Provider]
    end

    subgraph "External Services"
        OLLAMA[Ollama API]
        OPENAI[OpenAI API]
        GROQ[Groq API]
        OPENROUTER[OpenRouter API]
    end

    subgraph "MCP Server"
        MCPS[MCP Server]
        MCPH[MCP Handler]
    end

    subgraph "API Layer"
        FAPI[FastAPI Server]
        ROUTES[API Routes]
        MW[Middleware]
    end

    %% Configuration connections
    UC --> MC
    UC --> PC
    UC --> AC
    UC --> TC
    UC --> PLC
    UC --> PRC
    CB --> UC

    %% Prompt system connections
    UPM --> PT
    UPM --> PS
    UPM --> TE
    UPM --> UC

    %% Agent framework connections
    AF --> BA
    AF --> UC
    BA --> TA
    BA --> RA
    BA --> REA
    BA --> AA
    PE --> AF
    PE --> ACTX
    PE --> AR
    AF --> UPM

    %% Tool system connections
    TR --> BT
    BT --> TQT
    BT --> VT
    BT --> MT
    BT --> CT
    BT --> PQT
    TR --> UC
    TR --> AF
    TR --> UPS

    %% Provider system connections
    UPS --> PR
    PR --> BP
    BP --> OP
    BP --> OAP
    BP --> GR
    BP --> OR
    UPS --> UC

    %% External connections
    OP --> OLLAMA
    OAP --> OPENAI
    GR --> GROQ
    OR --> OPENROUTER

    %% MCP Server connections
    MCPS --> TR
    MCPS --> MCPH
    MCPH --> TR

    %% API connections
    FAPI --> ROUTES
    FAPI --> MW
    ROUTES --> AF
    ROUTES --> UPS
    ROUTES --> TR

    %% Data flow
    FAPI -.->|"User Request"| PE
    PE -.->|"Execute"| TA
    TA -.->|"Get Prompt"| UPM
    TA -.->|"Generate"| UPS
    UPS -.->|"API Call"| OPENAI
    TA -.->|"Result"| AR

    style UC fill:#e1f5fe
    style UPM fill:#f3e5f5
    style AF fill:#e8f5e8
    style TR fill:#fff3e0
    style UPS fill:#fce4ec
```

### Request Processing Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as "FastAPI Server"
    participant PE as "Pipeline Executor"
    participant AF as "Agent Factory"
    participant Agent as "Translator Agent"
    participant UPM as "Unified Prompt Manager"
    participant UPS as "Unified Provider Service"
    participant Provider as "OpenAI Provider"
    participant External as "OpenAI API"

    Client->>API: POST /translate {"query": "Hello", "target_lang": "es"}
    API->>PE: execute_pipeline("translation_pipeline", context)
    
    Note over PE: Load pipeline configuration
    PE->>AF: create_agent(AgentType.TRANSLATOR)
    AF->>Agent: new TranslatorAgent(config)
    Agent-->>AF: agent instance
    AF-->>PE: agent instance
    
    PE->>Agent: execute(context)
    
    Note over Agent: Process request
    Agent->>UPM: get_prompt(TRANSLATOR, DETAILED, context)
    UPM->>UPM: render_template(context)
    UPM-->>Agent: rendered prompt
    
    Agent->>UPS: generate(provider="openai", model="gpt-4", prompt=...)
    UPS->>Provider: generate(request)
    Provider->>External: HTTP POST /v1/chat/completions
    External-->>Provider: response
    Provider-->>UPS: GenerationResponse
    UPS-->>Agent: response
    
    Agent->>Agent: process_response()
    Agent-->>PE: AgentResult(success=True, output="Hola")
    
    PE-->>API: pipeline result
    API-->>Client: {"result": "Hola", "status": "success"}

    Note over Client,External: Request flow with error handling
    
    rect rgb(255, 240, 240)
        Note over Agent,External: Error Handling Flow
        Agent->>UPS: generate() fails
        UPS->>Agent: retry with fallback model
        Agent->>UPS: generate(model="gpt-3.5-turbo")
        UPS-->>Agent: success response
    end
```

### MCP Tool Interactions

```mermaid
graph LR
    subgraph "MCP Client"
        CLIENT[MCP Client<br/>Claude Desktop]
    end

    subgraph "MCP Server Interface"
        MCPS[MCP Server]
        MCPH[MCP Protocol Handler]
        TR[Tool Registry]
    end

    subgraph "Available Tools"
        TQT[Translate Query Tool<br/>translate_query]
        PQT[Process Query Pipeline Tool<br/>process_query_pipeline]
        VT[Validate GraphQL Tool<br/>validate_graphql]
        LMT[List Models Tool<br/>list_models]
        GMT[Get Model Info Tool<br/>get_model_info]
        GCT[Get Config Tool<br/>get_config]
    end

    subgraph "Core Systems"
        UC[Unified Config]
        AF[Agent Factory]
        UPS[Unified Provider Service]
        PE[Pipeline Executor]
    end

    subgraph "External Resources"
        MODELS[Model Providers<br/>OpenAI, Ollama, Groq]
        DB[Database<br/>MongoDB/Neo4j]
    end

    %% MCP Protocol Flow
    CLIENT -->|"tool_call request"| MCPS
    MCPS --> MCPH
    MCPH --> TR
    
    %% Tool Registration
    TR --> TQT
    TR --> PQT
    TR --> VT
    TR --> LMT
    TR --> GMT
    TR --> GCT

    %% Tool Execution Flows
    TQT -->|"uses"| AF
    TQT -->|"uses"| UPS
    
    PQT -->|"uses"| PE
    PQT -->|"uses"| UC
    
    VT -->|"validates"| DB
    
    LMT -->|"queries"| UC
    GMT -->|"queries"| UC
    GCT -->|"reads"| UC

    %% Core System Interactions
    AF -->|"creates agents"| UC
    PE -->|"orchestrates"| AF
    UPS -->|"connects to"| MODELS

    %% Response Flow
    TR -.->|"tool_result"| MCPH
    MCPH -.->|"response"| MCPS
    MCPS -.->|"result"| CLIENT

    %% Styling
    style CLIENT fill:#e3f2fd
    style MCPS fill:#f1f8e9
    style TR fill:#fff8e1
    style TQT fill:#fce4ec
    style PQT fill:#fce4ec
    style VT fill:#fce4ec
    style LMT fill:#fce4ec
    style GMT fill:#fce4ec
    style GCT fill:#fce4ec
    style UC fill:#e8eaf6
    style MODELS fill:#f3e5f5
```

This unified architecture provides a solid foundation for scalable, maintainable, and extensible AI agent systems. 