# Modular MCP Architecture Guide

This guide explains the new modular architecture for MPPW-MCP that makes it easy to extend agents, prompts, tools, and client providers.

## Overview

The modular architecture is organized into these key areas:

1. **Modular Prompt System** - Easily editable prompt templates
2. **Tool Abstraction Layer** - Scalable tool definitions
3. **Agent Framework** - Pluggable agent implementations
4. **Client Abstraction** - Swappable LLM providers
5. **Pipeline System** - Composable processing strategies

## 1. Modular Prompt System

### Structure

```
backend/prompts/
├── __init__.py              # Main prompt system
├── types.py                 # Data types and enums
├── loader.py                # File loading logic
├── manager.py               # Core management interface
├── agents/                  # Agent-specific prompts
│   ├── translator/
│   │   ├── basic_translation.yaml
│   │   └── advanced_translation.yaml
│   ├── reviewer/
│   │   ├── security_review.yaml
│   │   └── performance_review.yaml
│   └── rewriter/
├── domains/                 # Domain-specific prompts
│   ├── ecommerce/
│   │   ├── product_queries.yaml
│   │   └── order_management.yaml
│   ├── social_media/
│   └── analytics/
└── strategies/              # Strategy-specific prompts
    ├── minimal/
    │   └── quick_translation.yaml
    ├── detailed/
    └── chain_of_thought/
```

### Creating New Prompts

Create a YAML file with this structure:

```yaml
name: my_custom_prompt
agent_type: translator  # optional: translator, reviewer, rewriter, etc.
domain: ecommerce      # optional: ecommerce, social_media, etc.
strategy: detailed     # minimal, detailed, chain_of_thought, etc.
description: "Description of what this prompt does"
version: "1.0"
tags:
  - custom
  - experimental
variables:
  - query
  - schema_context
  - examples
content: |
  You are an expert in {{ domain }} GraphQL queries.
  
  Query: {{ query }}
  
  {% if schema_context %}
  Schema: {{ schema_context }}
  {% endif %}
  
  Generate accurate GraphQL following best practices.
  
  Return JSON: {"graphql": "...", "confidence": 0.95}

examples:
  - natural: "Example natural language"
    graphql: "query { example }"
```

### Using Prompts Programmatically

```python
from prompts import get_prompt_manager
from prompts.types import PromptContext, AgentType, DomainType

# Get the prompt manager
pm = get_prompt_manager()

# Create context
context = PromptContext(
    query="Show me products under $50",
    domain=DomainType.ECOMMERCE,
    schema_context="type Product { id: ID! name: String! price: Float! }"
)

# Generate prompt automatically
result = pm.generate_smart_prompt(
    context=context,
    agent_type=AgentType.TRANSLATOR
)

print(result.content)  # The rendered prompt
print(result.template_used)  # Which template was selected
```

## 2. Tool Abstraction Layer

### MCP Tools Structure

```python
# backend/mcp_server/tools/
├── __init__.py
├── base.py                  # Base tool classes
├── translation/
│   ├── __init__.py
│   ├── fast_translation.py
│   ├── standard_translation.py
│   └── comprehensive_translation.py
├── validation/
│   ├── __init__.py
│   ├── syntax_validator.py
│   └── security_validator.py
└── utilities/
    ├── __init__.py
    ├── schema_inspector.py
    └── query_optimizer.py
```

### Creating New Tools

```python
from mcp_server.tools.base import BaseMCPTool
from typing import Dict, Any

class MyCustomTool(BaseMCPTool):
    name = "my_custom_tool"
    description = "Does something useful"
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        # Your tool logic here
        query = kwargs.get('query', '')
        
        # Process the query
        result = await self.process_query(query)
        
        return {
            'success': True,
            'result': result,
            'metadata': {'tool_version': '1.0'}
        }
    
    async def process_query(self, query: str) -> str:
        # Implementation
        return f"Processed: {query}"

# Register the tool
from mcp_server.enhanced_agent import EnhancedMCPServer
server = EnhancedMCPServer()
server.register_tool(MyCustomTool())
```

## 3. Agent Framework

### Agent Structure

```python
# backend/agents/
├── __init__.py
├── base.py                  # Base agent interface
├── registry.py             # Agent registration system
├── context.py              # Context management
├── implementations.py      # Concrete implementations
└── specialized/            # Domain-specific agents
    ├── ecommerce_agent.py
    ├── analytics_agent.py
    └── security_agent.py
```

### Creating Custom Agents

```python
from agents.base import BaseAgent, AgentContext, AgentCapability
from agents.registry import agent
from prompts import get_prompt_manager
from prompts.types import PromptContext, AgentType

@agent(
    name="custom_processor",
    capabilities=[AgentCapability.TRANSLATE, AgentCapability.VALIDATE],
    description="Custom processing agent"
)
class CustomProcessorAgent(BaseAgent):
    
    async def run(self, ctx: AgentContext, config=None, **kwargs):
        # Get appropriate prompt
        pm = get_prompt_manager()
        prompt_context = PromptContext(
            query=ctx.original_query,
            domain=ctx.domain,
            schema_context=ctx.schema_context
        )
        
        prompt_result = pm.generate_smart_prompt(
            context=prompt_context,
            agent_type=AgentType.TRANSLATOR
        )
        
        # Use your LLM service
        result = await self.call_llm(prompt_result.content)
        
        # Store results
        ctx.add_agent_output(self.name, result)
```

## 4. Client Abstraction Layer

### Provider Structure

```python
# backend/services/llm_providers/
├── __init__.py
├── base.py                  # Base provider interface
├── ollama_provider.py       # Ollama implementation
├── openai_provider.py       # OpenAI implementation
├── groq_provider.py         # Groq implementation
└── factory.py               # Provider factory
```

### Base Provider Interface

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncGenerator

class BaseLLMProvider(ABC):
    
    @abstractmethod
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str,
        **kwargs
    ) -> Any:
        """Generate chat completion."""
        pass
    
    @abstractmethod
    async def stream_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        pass
    
    @abstractmethod
    async def list_models(self) -> List[str]:
        """List available models."""
        pass
```

### Adding New Providers

```python
from services.llm_providers.base import BaseLLMProvider

class HuggingFaceProvider(BaseLLMProvider):
    
    def __init__(self, api_key: str, base_url: str = None):
        self.api_key = api_key
        self.base_url = base_url or "https://api-inference.huggingface.co"
    
    async def chat_completion(self, messages, model, **kwargs):
        # Implementation for Hugging Face API
        # ...
        pass
    
    async def stream_completion(self, messages, model, **kwargs):
        # Implementation for streaming
        # ...
        pass
    
    # etc.

# Register provider
from services.llm_providers.factory import LLMProviderFactory
factory = LLMProviderFactory()
factory.register_provider("huggingface", HuggingFaceProvider)
```

### Using Providers

```python
from services.llm_providers.factory import get_llm_provider

# Get provider by configuration
provider = get_llm_provider("openai")  # or "ollama", "groq", etc.

# Use provider
result = await provider.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4",
    temperature=0.7
)
```

## 5. Pipeline System

### Pipeline Strategies

The system supports multiple pipeline strategies:

- **Fast**: Translation only
- **Standard**: Rewrite → Translate → Review
- **Comprehensive**: All agents + optimization + data review
- **Adaptive**: Strategy selected based on query complexity

### Creating Custom Pipelines

```python
from agents.pipeline import Pipeline, PipelineStrategy

class CustomPipeline(Pipeline):
    name = "custom"
    description = "Custom processing pipeline"
    
    async def execute(self, context: AgentContext) -> Dict[str, Any]:
        # Step 1: Custom preprocessing
        await self.run_agent("custom_preprocessor", context)
        
        # Step 2: Translation
        await self.run_agent("translator", context)
        
        # Step 3: Custom validation
        await self.run_agent("custom_validator", context)
        
        return self.collect_results(context)

# Register pipeline
from mcp_server.enhanced_agent import EnhancedMCPServer
server = EnhancedMCPServer()
server.register_pipeline("custom", CustomPipeline())
```

## 6. Configuration Management

### Environment-Based Configuration

```python
# config/providers.yaml
llm_providers:
  default: ollama
  providers:
    ollama:
      base_url: http://localhost:11434
      default_model: gemma3:4b
      timeout: 30
    openai:
      api_key: ${OPENAI_API_KEY}
      default_model: gpt-4
      base_url: https://api.openai.com/v1
    groq:
      api_key: ${GROQ_API_KEY}
      default_model: llama3-8b-8192
      base_url: https://api.groq.com/openai/v1

# config/agents.yaml
agents:
  translator:
    default_prompt: basic_translation
    fallback_prompts:
      - minimal_translation
      - advanced_translation
    models:
      primary: gemma3:4b
      fallback: llama3:8b
  reviewer:
    default_prompt: security_review
    models:
      primary: gemma3:4b
```

## 7. Usage Examples

### Basic Translation

```python
from mcp_server.enhanced_agent import EnhancedMCPServer

server = EnhancedMCPServer()
await server.initialize()

result = await server.call_tool("process_query_fast", {
    "query": "Show me products under $50",
    "translator_model": "gemma3:4b"
})

print(result['translation']['graphql'])
```

### Custom Domain Processing

```python
from prompts import get_prompt_manager
from prompts.types import PromptContext, DomainType

pm = get_prompt_manager()

# Create domain-specific context
context = PromptContext(
    query="Show me trending posts from friends",
    domain=DomainType.SOCIAL_MEDIA,
    examples=[
        {"natural": "recent posts", "graphql": "query { posts(orderBy: {createdAt: DESC}) { id content } }"}
    ]
)

# Generate optimized prompt
result = pm.generate_smart_prompt(context)
```

### Adding Custom Functionality

```python
# 1. Add custom prompt
pm.create_template(
    name="my_custom_prompt",
    content="Custom prompt content with {{ query }}",
    strategy=PromptStrategy.DETAILED,
    domain=DomainType.ANALYTICS
)

# 2. Add custom agent
@agent(name="my_analyzer", capabilities=[AgentCapability.ANALYZE])
class MyAnalyzerAgent(BaseAgent):
    async def run(self, ctx, config=None, **kwargs):
        # Custom analysis logic
        pass

# 3. Add custom tool
class MyAnalysisTool(BaseMCPTool):
    name = "analyze_query"
    
    async def execute(self, **kwargs):
        # Custom tool logic
        return {"analysis": "detailed analysis result"}
```

## 8. Development Workflow

### Testing Changes

```bash
# Test prompt changes
python -m pytest tests/test_prompts.py

# Test new agents
python -m pytest tests/test_agents.py

# Test MCP server integration
python test_enhanced_mcp.py

# Test with specific models
python test_enhanced_mcp.py --model gemma3:4b
```

### Hot Reloading

The system supports hot reloading of prompts without server restart:

```python
# Prompts auto-reload every 30 seconds by default
pm = get_prompt_manager()
pm.auto_reload = True

# Manual reload
reloaded = pm.reload_templates()
print(f"Reloaded {reloaded} templates")
```

### Debugging

Enable debug logging to see the full prompt resolution process:

```python
import logging
logging.getLogger('prompts').setLevel(logging.DEBUG)
logging.getLogger('agents').setLevel(logging.DEBUG)
logging.getLogger('mcp_server').setLevel(logging.DEBUG)
```

## 9. Best Practices

### Prompt Development

1. **Be Specific**: Use clear, specific instructions
2. **Include Examples**: Provide concrete examples in your prompts
3. **Use Variables**: Make prompts reusable with template variables
4. **Version Control**: Track prompt versions and changes
5. **Test Thoroughly**: Test prompts with various query types

### Agent Development

1. **Single Responsibility**: Each agent should have a clear, focused purpose
2. **Error Handling**: Include comprehensive error handling
3. **Logging**: Add detailed logging for debugging
4. **Configuration**: Make agents configurable via external settings
5. **Testing**: Write unit tests for agent logic

### Performance Optimization

1. **Caching**: Cache frequently used prompts and results
2. **Batch Processing**: Use batch operations where possible
3. **Resource Limits**: Set appropriate timeouts and limits
4. **Monitoring**: Monitor performance metrics
5. **Scaling**: Design for horizontal scaling

This modular architecture makes MPPW-MCP highly extensible and maintainable. You can easily add new prompts, agents, tools, and providers without modifying core code. 