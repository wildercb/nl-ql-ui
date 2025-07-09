# Quick Start Guide: Unified Architecture

## Overview

This guide will get you up and running with the new unified MPPW-MCP architecture in under 10 minutes.

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Basic understanding of async Python

## 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd mppw-mcp

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Start services
docker-compose up -d
```

## 2. Basic Configuration

Create a simple configuration to get started:

```python
# examples/basic_setup.py
from backend.config.unified_config import ConfigBuilder, set_global_config

async def setup_basic_config():
    """Set up a basic configuration for development."""
    builder = ConfigBuilder()
    
    # Add a local Ollama provider
    builder.add_provider(
        name="ollama",
        provider_type=ModelProvider.OLLAMA,
        settings={"base_url": "http://localhost:11434"}
    )
    
    # Add models
    builder.add_model(
        name="llama3:8b",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.MEDIUM,
        capabilities=["translation", "rewriting", "analysis"]
    )
    
    # Add OpenAI provider (if you have an API key)
    builder.add_provider(
        name="openai",
        provider_type=ModelProvider.OPENAI,
        settings={"api_key": "your-openai-api-key"}
    )
    
    builder.add_model(
        name="gpt-4",
        provider=ModelProvider.OPENAI,
        size=ModelSize.LARGE,
        capabilities=["translation", "rewriting", "analysis", "review"]
    )
    
    # Configure agents
    builder.add_agent(
        agent_type=AgentType.TRANSLATOR,
        capabilities=[AgentCapability.TRANSLATION],
        primary_model="gpt-4",
        fallback_models=["llama3:8b"],
        prompt_strategy=PromptStrategy.DETAILED
    )
    
    builder.add_agent(
        agent_type=AgentType.REWRITER,
        capabilities=[AgentCapability.REWRITING],
        primary_model="gpt-4",
        fallback_models=["llama3:8b"],
        prompt_strategy=PromptStrategy.CHAIN_OF_THOUGHT
    )
    
    # Add tools
    builder.add_tool(
        name="translate_query",
        category=ToolCategory.TRANSLATION,
        settings={"max_length": 1000}
    )
    
    # Add pipelines
    builder.add_pipeline(
        name="translation_pipeline",
        agents=[AgentType.TRANSLATOR],
        strategy=PipelineStrategy.SEQUENTIAL
    )
    
    # Build and set global config
    config = builder.build()
    set_global_config(config)
    
    return config

# Run setup
import asyncio
asyncio.run(setup_basic_config())
```

## 3. Using Agents

### Simple Agent Usage

```python
# examples/agent_usage.py
from backend.agents.unified_agents import AgentFactory, AgentContext

async def translate_text():
    """Simple translation example."""
    # Create a translator agent
    agent = AgentFactory.create_agent(AgentType.TRANSLATOR)
    
    # Create context
    context = AgentContext(
        query="Hello, how are you?",
        session_id="demo-session",
        metadata={
            "source_language": "en",
            "target_language": "es"
        }
    )
    
    # Execute translation
    result = await agent.execute(context)
    
    if result.success:
        print(f"Translation: {result.output}")
        print(f"Confidence: {result.confidence}")
    else:
        print(f"Error: {result.error}")

# Run example
asyncio.run(translate_text())
```

### Pipeline Usage

```python
# examples/pipeline_usage.py
from backend.agents.unified_agents import PipelineExecutor, AgentContext

async def run_translation_pipeline():
    """Run a complete translation pipeline."""
    executor = PipelineExecutor()
    
    context = AgentContext(
        query="Translate this to Spanish: Hello world",
        session_id="pipeline-demo"
    )
    
    result = await executor.execute_pipeline(
        pipeline_name="translation_pipeline",
        context=context
    )
    
    print(f"Pipeline result: {result.output}")
    print(f"Execution time: {result.execution_time}")

asyncio.run(run_translation_pipeline())
```

## 4. Using Providers

### Direct Provider Usage

```python
# examples/provider_usage.py
from backend.services.unified_providers import get_provider_service
from backend.services.unified_providers import GenerationRequest

async def use_provider_directly():
    """Use a provider directly for text generation."""
    provider_service = get_provider_service()
    
    request = GenerationRequest(
        prompt="Translate 'Hello world' to Spanish",
        max_tokens=50,
        temperature=0.3
    )
    
    # Use specific provider
    response = await provider_service.generate(
        provider_name="openai",
        model="gpt-4",
        request=request
    )
    
    print(f"Generated text: {response.text}")
    print(f"Provider used: {response.provider}")
    print(f"Model used: {response.model}")

asyncio.run(use_provider_directly())
```

## 5. Using MCP Tools

### MCP Server Setup

```python
# examples/mcp_server_demo.py
from backend.mcp_server.tools.unified_tools import get_tool_registry
from backend.mcp_server.main import MCPServer

async def demo_mcp_tools():
    """Demonstrate MCP tool usage."""
    registry = get_tool_registry()
    
    # List available tools
    tools = registry.list_tools()
    print("Available tools:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
    
    # Use translation tool
    translate_tool = registry.get_tool("translate_query")
    result = await translate_tool.execute({
        "query": "Hello world",
        "target_language": "es"
    })
    
    print(f"Translation result: {result.data}")

asyncio.run(demo_mcp_tools())
```

## 6. Custom Prompts

### Using Prompt Manager

```python
# examples/prompt_usage.py
from backend.prompts.unified_prompts import get_prompt_manager
from backend.config.unified_config import AgentType, PromptStrategy

async def use_custom_prompts():
    """Demonstrate custom prompt usage."""
    prompt_manager = get_prompt_manager()
    
    # Get a translation prompt
    prompt = prompt_manager.get_prompt(
        agent_type=AgentType.TRANSLATOR,
        strategy=PromptStrategy.DETAILED,
        context={
            "source_language": "English",
            "target_language": "Spanish",
            "query": "Hello, how are you today?",
            "domain": "casual_conversation"
        }
    )
    
    print(f"Generated prompt:\n{prompt}")

asyncio.run(use_custom_prompts())
```

## 7. Testing Your Setup

Create a comprehensive test to verify everything works:

```python
# examples/system_test.py
import pytest
from backend.config.unified_config import get_config
from backend.agents.unified_agents import AgentFactory, AgentContext
from backend.services.unified_providers import get_provider_service

async def test_complete_system():
    """Test the complete system integration."""
    
    # Test configuration
    config = get_config()
    assert config is not None
    print("✓ Configuration loaded")
    
    # Test provider service
    provider_service = get_provider_service()
    providers = provider_service.list_providers()
    assert len(providers) > 0
    print(f"✓ Provider service loaded with {len(providers)} providers")
    
    # Test agent creation
    agent = AgentFactory.create_agent(AgentType.TRANSLATOR)
    assert agent is not None
    print("✓ Agent created successfully")
    
    # Test agent execution
    context = AgentContext(
        query="Test translation",
        session_id="test-session"
    )
    
    result = await agent.execute(context)
    print(f"✓ Agent execution: {'Success' if result.success else 'Failed'}")
    
    # Test MCP tools
    from backend.mcp_server.tools.unified_tools import get_tool_registry
    registry = get_tool_registry()
    tools = registry.list_tools()
    assert len(tools) > 0
    print(f"✓ MCP tools loaded: {len(tools)} tools available")
    
    print("\n🎉 All systems operational!")

# Run the test
asyncio.run(test_complete_system())
```

## 8. Common Use Cases

### Translation with Fallback

```python
async def robust_translation():
    """Translation with automatic fallback."""
    agent = AgentFactory.create_agent(AgentType.TRANSLATOR)
    
    context = AgentContext(
        query="Translate: The weather is beautiful today",
        metadata={
            "target_language": "french",
            "max_retries": 3
        }
    )
    
    result = await agent.execute(context)
    
    print(f"Translation: {result.output}")
    print(f"Model used: {result.metadata.get('model_used')}")
    print(f"Attempts: {result.metadata.get('attempts')}")
```

### Multi-Agent Pipeline

```python
async def multi_agent_workflow():
    """Run multiple agents in sequence."""
    executor = PipelineExecutor()
    
    # First translate
    context = AgentContext(
        query="Hello world",
        metadata={"target_language": "spanish"}
    )
    
    translation_result = await executor.execute_pipeline(
        "translation_pipeline",
        context
    )
    
    # Then review the translation
    review_context = AgentContext(
        query=translation_result.output,
        metadata={"task": "review_translation"}
    )
    
    review_result = await executor.execute_pipeline(
        "review_pipeline",
        review_context
    )
    
    print(f"Original: {context.query}")
    print(f"Translation: {translation_result.output}")
    print(f"Review: {review_result.output}")
```

## 9. Development Workflow

### Adding a Custom Agent

```python
# 1. Define in unified_agents.py
class CustomAgent(BaseAgent):
    def __init__(self, config: UnifiedConfig):
        super().__init__(config, AgentType.CUSTOM)
    
    async def execute(self, context: AgentContext) -> AgentResult:
        # Your custom logic here
        pass

# 2. Register in AgentFactory
def create_agent(agent_type: AgentType, config: UnifiedConfig = None):
    if agent_type == AgentType.CUSTOM:
        return CustomAgent(config)
    # ... existing code
```

### Adding a Custom Provider

```python
# 1. Define provider class
class CustomProvider(BaseProvider):
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        # Your custom provider logic
        pass

# 2. Register in provider service
def get_provider_service():
    # ... existing code
    registry.register_provider("custom", CustomProvider(config))
```

## 10. Production Deployment

### Environment Configuration

```bash
# .env.production
ENVIRONMENT=production
DATABASE_URL=mongodb://production-db:27017/mppw
REDIS_URL=redis://production-redis:6379
OPENAI_API_KEY=your-production-key
LOG_LEVEL=INFO
```

### Docker Production

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  backend:
    build: 
      context: ./backend
      dockerfile: Dockerfile.prod
    environment:
      - ENV=production
    ports:
      - "8000:8000"
  
  frontend:
    build:
      context: ./frontend  
      dockerfile: Dockerfile.prod
    ports:
      - "80:80"
```

## Troubleshooting

### Common Issues

1. **Configuration not found**: Ensure `set_global_config()` is called
2. **Provider connection errors**: Check API keys and network connectivity
3. **Agent execution failures**: Check logs for detailed error information
4. **Tool registration errors**: Verify tool implementations follow BaseTool interface

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use configuration validation
from backend.config.unified_config import get_config
config = get_config()
validation_result = config.validate()
print(f"Configuration valid: {validation_result}")
```

## Next Steps

- Read the [Unified Architecture Guide](UNIFIED_ARCHITECTURE_GUIDE.md) for detailed implementation details
- Check out [Examples](EXAMPLES.md) for more complex use cases
- Review the [Configuration Guide](CONFIGURATION.md) for advanced configuration options
- Explore the [Usage Guide](USAGE_GUIDE.md) for comprehensive feature documentation

## Support

- Check the documentation in the `/docs` folder
- Review the example code in `/examples`
- Check the test files for usage patterns
- Open issues for bugs or feature requests

Happy coding with the unified MPPW-MCP architecture! 🚀 