# Examples - Unified Architecture

This document provides practical examples of using the MPPW-MCP unified architecture for various tasks and scenarios.

## Basic Setup

### 1. Configuration Setup

```python
from backend.config.unified_config import (
    ConfigBuilder, set_global_config,
    ModelProvider, ModelSize, AgentType, AgentCapability,
    PromptStrategy, ToolCategory, PipelineStrategy
)
import os

def setup_basic_config():
    """Set up basic configuration for the unified architecture."""
    builder = ConfigBuilder()
    
    # Add providers (using environment variables for API keys)
    builder.add_provider(
        name="openai",
        provider_type=ModelProvider.OPENAI,
        settings={
            "api_key": os.getenv("OPENAI_API_KEY"),
            "timeout": 30
        }
    )
    
    builder.add_provider(
        name="ollama",
        provider_type=ModelProvider.OLLAMA,
        settings={
            "base_url": "http://localhost:11434",
            "timeout": 60
        }
    )
    
    # Add models (typically done via UI)
    builder.add_model(
        name="gpt-4",
        provider=ModelProvider.OPENAI,
        size=ModelSize.LARGE,
        capabilities=["translation", "rewriting", "review"]
    )
    
    builder.add_model(
        name="llama3:8b",
        provider=ModelProvider.OLLAMA,
        size=ModelSize.MEDIUM,
        capabilities=["translation", "analysis"]
    )
    
    # Configure agents (models selected via UI)
    builder.add_agent(
        agent_type=AgentType.TRANSLATOR,
        capabilities=[AgentCapability.TRANSLATION],
        prompt_strategy=PromptStrategy.DETAILED,
        max_retries=3
    )
    
    # Add tools
    builder.add_tool(
        name="translate_query",
        category=ToolCategory.TRANSLATION,
        settings={"max_length": 1000}
    )
    
    # Build and set global config
    config = builder.build()
    set_global_config(config)
    
    return config

# Initialize configuration
config = setup_basic_config()
```

## Agent Usage Examples

### 2. Simple Translation

```python
from backend.agents.unified_agents import AgentFactory, AgentContext

async def simple_translation():
    """Basic translation example using unified agents."""
    
    # Create a translator agent
    agent = AgentFactory.create_agent(AgentType.TRANSLATOR)
    
    # Create context with translation request
    context = AgentContext(
        query="Hello, how are you today?",
        session_id="example-session",
        metadata={
            "source_language": "en",
            "target_language": "es",
            "domain": "casual"
        }
    )
    
    # Execute translation
    result = await agent.execute(context)
    
    if result.success:
        print(f"Original: {context.query}")
        print(f"Translation: {result.output}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Model used: {result.metadata.get('model_used')}")
    else:
        print(f"Translation failed: {result.error}")
    
    return result

# Run example
import asyncio
result = asyncio.run(simple_translation())
```

### 3. Pipeline Execution

```python
from backend.agents.unified_agents import PipelineExecutor, AgentContext

async def pipeline_translation():
    """Execute a complete translation pipeline."""
    
    executor = PipelineExecutor()
    
    context = AgentContext(
        query="Translate this to French: The weather is beautiful today",
        session_id="pipeline-example",
        metadata={
            "target_language": "fr",
            "quality_level": "high"
        }
    )
    
    # Execute the translation pipeline
    result = await executor.execute_pipeline(
        pipeline_name="translation_pipeline",
        context=context
    )
    
    print(f"Pipeline result: {result.output}")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Pipeline steps: {result.metadata.get('pipeline_steps', [])}")
    
    return result

# Run pipeline example
result = asyncio.run(pipeline_translation())
```

## Provider Usage Examples

### 4. Direct Provider Access

```python
from backend.services.unified_providers import get_provider_service
from backend.services.unified_providers import GenerationRequest

async def direct_provider_usage():
    """Use providers directly for text generation."""
    
    provider_service = get_provider_service()
    
    # Create generation request
    request = GenerationRequest(
        prompt="Translate 'Good morning' to Spanish",
        max_tokens=50,
        temperature=0.3,
        model="gpt-4"
    )
    
    # Generate using specific provider
    response = await provider_service.generate(
        provider_name="openai",
        model="gpt-4",
        request=request
    )
    
    print(f"Generated text: {response.text}")
    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")
    print(f"Tokens used: {response.tokens_used}")
    
    return response

# Run provider example
response = asyncio.run(direct_provider_usage())
```

### 5. Provider Fallback

```python
async def provider_with_fallback():
    """Example showing automatic fallback between providers."""
    
    provider_service = get_provider_service()
    
    request = GenerationRequest(
        prompt="Explain quantum computing in simple terms",
        max_tokens=150,
        temperature=0.7
    )
    
    # Try OpenAI first, fallback to Ollama
    try:
        response = await provider_service.generate(
            provider_name="openai",
            model="gpt-4",
            request=request
        )
        print(f"Used OpenAI: {response.text}")
    except Exception as e:
        print(f"OpenAI failed: {e}")
        # Fallback to local model
        response = await provider_service.generate(
            provider_name="ollama",
            model="llama3:8b",
            request=request
        )
        print(f"Fallback to Ollama: {response.text}")
    
    return response

# Run fallback example
response = asyncio.run(provider_with_fallback())
```

## MCP Tool Examples

### 6. Using MCP Tools

```python
from backend.mcp_server.tools.unified_tools import get_tool_registry

async def mcp_tool_usage():
    """Examples of using MCP tools."""
    
    registry = get_tool_registry()
    
    # List available tools
    tools = registry.list_tools()
    print("Available tools:")
    for tool in tools:
        print(f"- {tool.name}: {tool.description}")
    
    # Use translation tool
    translate_tool = registry.get_tool("translate_query")
    translation_result = await translate_tool.execute({
        "query": "Hello world",
        "target_language": "es",
        "context": "greeting"
    })
    
    print(f"Translation tool result: {translation_result.data}")
    
    # Use validation tool
    validate_tool = registry.get_tool("validate_graphql")
    validation_result = await validate_tool.execute({
        "query": "query { user { name email } }",
        "schema_context": "user management"
    })
    
    print(f"Validation result: {validation_result.data}")
    
    return [translation_result, validation_result]

# Run MCP tools example
results = asyncio.run(mcp_tool_usage())
```

## Advanced Examples

### 7. Custom Agent Implementation

```python
from backend.agents.unified_agents import BaseAgent, AgentContext, AgentResult
from backend.config.unified_config import AgentType

class CustomAnalyzerAgent(BaseAgent):
    """Custom agent for specialized analysis tasks."""
    
    def __init__(self, config):
        super().__init__(config, AgentType.ANALYZER)
    
    async def execute(self, context: AgentContext) -> AgentResult:
        """Execute custom analysis logic."""
        try:
            # Get prompt for analysis task
            prompt = await self.get_prompt(context)
            
            # Generate analysis using provider
            response = await self.generate_response(prompt)
            
            # Process and structure the analysis
            analysis = self.process_analysis_response(response)
            
            return AgentResult(
                success=True,
                output=analysis,
                confidence=0.85,
                metadata={
                    "analysis_type": "custom",
                    "model_used": response.model
                }
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                error=str(e),
                confidence=0.0
            )
    
    def process_analysis_response(self, response):
        """Process the raw response into structured analysis."""
        # Custom processing logic here
        return {
            "summary": response.text[:100],
            "key_points": response.text.split("."),
            "confidence": 0.85
        }

# Register and use custom agent
from backend.agents.unified_agents import AgentFactory

# This would typically be done in the agent factory
async def use_custom_agent():
    """Example of using a custom agent."""
    agent = CustomAnalyzerAgent(config)
    
    context = AgentContext(
        query="Analyze the sentiment of this text: I love this product!",
        session_id="custom-example"
    )
    
    result = await agent.execute(context)
    print(f"Custom analysis: {result.output}")
    
    return result

result = asyncio.run(use_custom_agent())
```

### 8. Batch Processing

```python
async def batch_translation_example():
    """Process multiple translations in batch."""
    
    queries = [
        "Hello world",
        "How are you?",
        "Thank you very much",
        "Good morning",
        "See you later"
    ]
    
    agent = AgentFactory.create_agent(AgentType.TRANSLATOR)
    results = []
    
    # Process in parallel
    tasks = []
    for i, query in enumerate(queries):
        context = AgentContext(
            query=query,
            session_id=f"batch-{i}",
            metadata={"target_language": "es"}
        )
        tasks.append(agent.execute(context))
    
    # Wait for all translations
    batch_results = await asyncio.gather(*tasks)
    
    # Display results
    for i, result in enumerate(batch_results):
        if result.success:
            print(f"{queries[i]} → {result.output}")
        else:
            print(f"{queries[i]} → Error: {result.error}")
    
    return batch_results

# Run batch example
results = asyncio.run(batch_translation_example())
```

### 9. Custom Prompt Strategy

```python
from backend.prompts.unified_prompts import get_prompt_manager
from backend.config.unified_config import AgentType, PromptStrategy

async def custom_prompt_example():
    """Example using different prompt strategies."""
    
    prompt_manager = get_prompt_manager()
    
    # Different prompt strategies for the same task
    strategies = [
        PromptStrategy.DETAILED,
        PromptStrategy.MINIMAL,
        PromptStrategy.CHAIN_OF_THOUGHT
    ]
    
    context = {
        "query": "Translate 'I am learning AI' to French",
        "source_language": "English",
        "target_language": "French",
        "domain": "education"
    }
    
    for strategy in strategies:
        prompt = prompt_manager.get_prompt(
            agent_type=AgentType.TRANSLATOR,
            strategy=strategy,
            context=context
        )
        
        print(f"\n{strategy.value.upper()} STRATEGY:")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)

# Run prompt strategy example
asyncio.run(custom_prompt_example())
```

### 10. Configuration Validation

```python
from backend.config.unified_config import get_config

def configuration_validation_example():
    """Example of validating and debugging configuration."""
    
    config = get_config()
    
    # Validate configuration
    validation_result = config.validate()
    
    if validation_result.is_valid:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration errors:")
        for error in validation_result.errors:
            print(f"  - {error}")
    
    # Display configuration summary
    print(f"\nConfiguration Summary:")
    print(f"Models: {list(config.models.keys())}")
    print(f"Providers: {list(config.providers.keys())}")
    print(f"Agents: {list(config.agents.keys())}")
    print(f"Tools: {list(config.tools.keys())}")
    print(f"Pipelines: {list(config.pipelines.keys())}")
    
    # Get model information
    for model_name, model_config in config.models.items():
        print(f"\nModel: {model_name}")
        print(f"  Provider: {model_config.provider.value}")
        print(f"  Size: {model_config.size.value}")
        print(f"  Capabilities: {model_config.capabilities}")

# Run configuration validation
configuration_validation_example()
```

## Error Handling Examples

### 11. Robust Error Handling

```python
async def robust_translation_with_error_handling():
    """Example showing comprehensive error handling."""
    
    agent = AgentFactory.create_agent(AgentType.TRANSLATOR)
    
    context = AgentContext(
        query="Translate this complex technical text...",
        session_id="error-handling-example",
        metadata={
            "max_retries": 3,
            "timeout": 30
        }
    )
    
    try:
        result = await agent.execute(context)
        
        if result.success:
            print(f"✅ Translation successful: {result.output}")
            
            # Check confidence level
            if result.confidence < 0.7:
                print("⚠️ Low confidence translation")
                
            # Check for warnings
            if result.warnings:
                print("⚠️ Warnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")
        else:
            print(f"❌ Translation failed: {result.error}")
            
            # Handle specific error types
            if "timeout" in result.error.lower():
                print("💡 Suggestion: Try with a faster model")
            elif "api_key" in result.error.lower():
                print("💡 Suggestion: Check your API key configuration")
            
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        
        # Log error for debugging
        import logging
        logging.error(f"Translation error: {e}", exc_info=True)

# Run robust error handling example
asyncio.run(robust_translation_with_error_handling())
```

## Integration Examples

### 12. FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.agents.unified_agents import AgentFactory, AgentContext

app = FastAPI(title="Unified Architecture API")

class TranslationRequest(BaseModel):
    query: str
    target_language: str
    session_id: str = None

class TranslationResponse(BaseModel):
    result: str
    confidence: float
    model_used: str

@app.post("/translate", response_model=TranslationResponse)
async def translate_endpoint(request: TranslationRequest):
    """FastAPI endpoint using unified architecture."""
    
    try:
        agent = AgentFactory.create_agent(AgentType.TRANSLATOR)
        
        context = AgentContext(
            query=request.query,
            session_id=request.session_id or "api-request",
            metadata={"target_language": request.target_language}
        )
        
        result = await agent.execute(context)
        
        if result.success:
            return TranslationResponse(
                result=result.output,
                confidence=result.confidence,
                model_used=result.metadata.get("model_used", "unknown")
            )
        else:
            raise HTTPException(status_code=500, detail=result.error)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# This would be run with: uvicorn main:app --reload
```

## Testing Examples

### 13. Unit Testing

```python
import pytest
from unittest.mock import AsyncMock, patch
from backend.config.unified_config import ConfigBuilder
from backend.agents.unified_agents import AgentFactory, AgentContext

@pytest.fixture
def test_config():
    """Create test configuration."""
    builder = ConfigBuilder()
    # Add minimal test configuration
    return builder.build()

@pytest.mark.asyncio
async def test_translator_agent(test_config):
    """Test translator agent functionality."""
    
    with patch('backend.config.unified_config.get_config', return_value=test_config):
        agent = AgentFactory.create_agent(AgentType.TRANSLATOR)
        
        context = AgentContext(
            query="Hello",
            session_id="test-session",
            metadata={"target_language": "es"}
        )
        
        # Mock the provider response
        with patch.object(agent, 'generate_response') as mock_generate:
            mock_generate.return_value = AsyncMock()
            mock_generate.return_value.text = "Hola"
            mock_generate.return_value.model = "test-model"
            
            result = await agent.execute(context)
            
            assert result.success
            assert "Hola" in result.output
            assert result.confidence > 0

@pytest.mark.asyncio 
async def test_pipeline_execution():
    """Test pipeline execution."""
    # Similar testing pattern for pipelines
    pass

# Run tests with: pytest test_examples.py -v
```

## Performance Examples

### 14. Performance Monitoring

```python
import time
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def performance_monitor(operation_name: str):
    """Context manager for monitoring operation performance."""
    start_time = time.time()
    memory_start = None  # Could add memory monitoring
    
    try:
        print(f"🚀 Starting {operation_name}...")
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        print(f"✅ {operation_name} completed in {duration:.2f}s")

async def performance_example():
    """Example of monitoring performance."""
    
    async with performance_monitor("Batch Translation"):
        agent = AgentFactory.create_agent(AgentType.TRANSLATOR)
        
        # Process multiple translations
        tasks = []
        for i in range(10):
            context = AgentContext(
                query=f"Test query {i}",
                session_id=f"perf-test-{i}"
            )
            tasks.append(agent.execute(context))
        
        results = await asyncio.gather(*tasks)
        
        success_count = sum(1 for r in results if r.success)
        avg_confidence = sum(r.confidence for r in results if r.success) / success_count
        
        print(f"📊 Results: {success_count}/10 successful")
        print(f"📊 Average confidence: {avg_confidence:.2f}")

# Run performance example
asyncio.run(performance_example())
```

These examples demonstrate the power and flexibility of the unified architecture. The system provides:

- **Type Safety**: All operations are type-checked
- **Consistency**: Uniform interfaces across all components
- **Extensibility**: Easy to add new agents, providers, and tools
- **Reliability**: Comprehensive error handling and fallbacks
- **Performance**: Efficient async operations and caching
- **Testing**: Straightforward unit and integration testing

For more detailed information, see:
- [Unified Architecture Guide](UNIFIED_ARCHITECTURE_GUIDE.md)
- [Configuration Guide](CONFIGURATION.md)
- [Quick Start Guide](QUICK_START_UNIFIED.md) 