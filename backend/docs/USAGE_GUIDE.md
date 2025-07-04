# MPPW MCP Usage & Extension Guide

This guide covers how to use, customize, and extend your Natural Language to GraphQL translation system.

## Table of Contents
1. [Basic Usage](#basic-usage)
2. [Customizing Prompts](#customizing-prompts)
3. [Adding New Tools](#adding-new-tools)
4. [Switching Model Providers](#switching-model-providers)
5. [Advanced Customization](#advanced-customization)
6. [Examples](#examples)

## Basic Usage

### 1. Sending Queries to Ollama

#### Via REST API
```bash
# Translate natural language to GraphQL
curl -X POST "http://localhost:8000/api/v1/translation/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "natural_query": "Get all users with their email addresses and posts",
    "schema_context": "type User { id: ID!, email: String!, posts: [Post!]! }",
    "model": "llama2"
  }'
```

#### Via Python Client
```python
import httpx

async def translate_query():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/translation/translate",
            json={
                "natural_query": "Find posts by author ID with comments",
                "schema_context": "type Post { id: ID!, title: String!, author: User!, comments: [Comment!]! }",
                "model": "llama2"
            }
        )
        return response.json()
```

#### Via MCP Client
```python
from backend.mcp_server import MCPServer

# Initialize MCP server
server = MCPServer()

# Call translation tool
result = await server._handle_translate_query({
    "natural_query": "Get user profile with recent activity",
    "schema_context": "type User { profile: Profile!, activities: [Activity!]! }"
})
```

### 2. Available Endpoints

| Endpoint | Purpose | Method |
|----------|---------|---------|
| `/api/v1/translation/translate` | Single query translation | POST |
| `/api/v1/translation/translate/batch` | Batch translation | POST |
| `/api/v1/validation/validate` | GraphQL validation | POST |
| `/api/v1/models/` | List available models | GET |
| `/api/v1/models/pull` | Pull new model | POST |

## Customizing Prompts

### 1. Modifying Translation Prompts

Edit `backend/services/translation_service.py`:

```python
def _build_system_prompt(self, schema_context: str = "") -> str:
    """Build system prompt for GraphQL translation."""
    base_prompt = """You are an expert GraphQL query translator specialized in [YOUR DOMAIN].

    CUSTOM INSTRUCTIONS:
    1. Always include pagination arguments when dealing with lists
    2. Use fragments for user information across queries
    3. Follow [YOUR COMPANY] naming conventions
    4. Optimize for performance with selective field fetching
    
    DOMAIN-SPECIFIC RULES:
    - For user queries, always include: id, email, profile.displayName
    - For posts, include: id, title, createdAt, author { id, profile { displayName } }
    - Use camelCase for all field names
    
    Return ONLY a JSON object with this structure:
    {
      "graphql": "the GraphQL query string",
      "confidence": 0.0-1.0,
      "explanation": "brief explanation",
      "warnings": ["any warnings"],
      "suggestions": ["improvements"]
    }"""
    
    if schema_context:
        base_prompt += f"\n\nGraphQL Schema:\n{schema_context}"
        
    # Add your custom schema patterns
    base_prompt += f"\n\nCommon Patterns:\n{self._get_domain_patterns()}"
    
    return base_prompt

def _get_domain_patterns(self) -> str:
    """Return domain-specific GraphQL patterns."""
    return """
    USER_QUERY_PATTERN: query GetUser($id: ID!) { user(id: $id) { ...UserFields } }
    POST_LIST_PATTERN: query GetPosts($first: Int, $after: String) { posts(first: $first, after: $after) { edges { node { ...PostFields } } } }
    SEARCH_PATTERN: query Search($term: String!) { search(query: $term) { ... on User { ...UserFields } ... on Post { ...PostFields } } }
    """
```

### 2. Adding Model-Specific Prompts

Create different prompts for different models:

```python
def _build_system_prompt(self, schema_context: str = "", model: str = None) -> str:
    """Build model-specific system prompts."""
    prompts = {
        "llama2": self._get_llama2_prompt(),
        "codellama": self._get_codellama_prompt(),
        "mistral": self._get_mistral_prompt(),
        "custom-model": self._get_custom_prompt()
    }
    
    base_prompt = prompts.get(model, self._get_default_prompt())
    
    if schema_context:
        base_prompt += f"\n\nSchema Context:\n{schema_context}"
    
    return base_prompt

def _get_codellama_prompt(self) -> str:
    """Optimized prompt for CodeLlama models."""
    return """You are CodeLlama, specialized in generating GraphQL queries.
    
    FOCUS ON:
    - Syntactic correctness
    - Optimal field selection
    - Proper variable usage
    - Performance considerations
    
    RESPONSE FORMAT: JSON only, no explanations outside the JSON structure."""
```

### 3. Dynamic Prompt Templates

Create a template system for different query types:

```python
class PromptTemplateManager:
    def __init__(self):
        self.templates = {
            "user_query": "Generate a GraphQL query to fetch user information: {query}",
            "search_query": "Create a search query for: {query}",
            "mutation_query": "Build a mutation to: {query}",
            "analytics_query": "Generate analytics query for: {query}"
        }
    
    def get_prompt(self, query_type: str, natural_query: str, **kwargs) -> str:
        template = self.templates.get(query_type, self.templates["user_query"])
        return template.format(query=natural_query, **kwargs)

# Usage in translation service
def _classify_query_type(self, natural_query: str) -> str:
    """Classify the type of query to use appropriate template."""
    if any(word in natural_query.lower() for word in ["create", "update", "delete", "add"]):
        return "mutation_query"
    elif any(word in natural_query.lower() for word in ["search", "find", "filter"]):
        return "search_query"
    elif any(word in natural_query.lower() for word in ["analytics", "count", "sum", "average"]):
        return "analytics_query"
    else:
        return "user_query"
```

## Adding New Tools

### 1. Create a New MCP Tool

Add to `backend/mcp_server/server.py`:

```python
# In the _setup_tools method, add a new tool
Tool(
    name="optimize_query_performance",
    description="Analyze and optimize GraphQL query performance",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "GraphQL query to optimize"},
            "schema": {"type": "string", "description": "GraphQL schema"},
            "target_performance": {"type": "string", "description": "Performance target (fast/balanced/detailed)"}
        },
        "required": ["query"]
    }
)

# Add the handler method
async def _handle_optimize_query_performance(self, arguments: Dict[str, Any]) -> CallToolResult:
    """Optimize GraphQL query for performance."""
    query = arguments.get("query", "")
    schema = arguments.get("schema", "")
    target = arguments.get("target_performance", "balanced")
    
    try:
        optimizer = QueryOptimizer()
        result = await optimizer.optimize(query, schema, target)
        
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps({
                        "original_query": query,
                        "optimized_query": result.optimized_query,
                        "optimizations_applied": result.optimizations,
                        "performance_improvement": result.estimated_improvement,
                        "recommendations": result.recommendations
                    }, indent=2)
                )
            ]
        )
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Optimization error: {str(e)}")]
        )
```

### 2. Create the Query Optimizer Service

```python
# backend/services/query_optimizer.py
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class OptimizationResult:
    optimized_query: str
    optimizations: List[str]
    estimated_improvement: str
    recommendations: List[str]

class QueryOptimizer:
    """Service for optimizing GraphQL queries."""
    
    async def optimize(self, query: str, schema: str, target: str) -> OptimizationResult:
        """Optimize a GraphQL query."""
        optimizations = []
        optimized_query = query
        
        # Add fragments for repeated field sets
        if self._has_repeated_fields(query):
            optimized_query = self._add_fragments(optimized_query)
            optimizations.append("Added fragments for repeated field sets")
        
        # Optimize field selection
        if self._has_unnecessary_fields(query, schema):
            optimized_query = self._optimize_fields(optimized_query, target)
            optimizations.append("Removed unnecessary field selections")
        
        # Add pagination
        if self._needs_pagination(query):
            optimized_query = self._add_pagination(optimized_query)
            optimizations.append("Added pagination for list fields")
        
        return OptimizationResult(
            optimized_query=optimized_query,
            optimizations=optimizations,
            estimated_improvement="15-30% faster execution",
            recommendations=self._get_recommendations(query, schema)
        )
    
    def _has_repeated_fields(self, query: str) -> bool:
        # Implementation for detecting repeated field patterns
        pass
    
    def _add_fragments(self, query: str) -> str:
        # Implementation for adding GraphQL fragments
        pass
```

### 3. Add Schema Analysis Tool

```python
# New tool for schema analysis
Tool(
    name="analyze_schema",
    description="Analyze GraphQL schema and provide insights",
    inputSchema={
        "type": "object", 
        "properties": {
            "schema": {"type": "string", "description": "GraphQL schema to analyze"},
            "analysis_type": {"type": "string", "enum": ["structure", "performance", "security"], "default": "structure"}
        },
        "required": ["schema"]
    }
)

async def _handle_analyze_schema(self, arguments: Dict[str, Any]) -> CallToolResult:
    """Analyze GraphQL schema."""
    schema = arguments.get("schema", "")
    analysis_type = arguments.get("analysis_type", "structure")
    
    analyzer = SchemaAnalyzer()
    result = await analyzer.analyze(schema, analysis_type)
    
    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=json.dumps(result.to_dict(), indent=2)
            )
        ]
    )
```

## Switching Model Providers

### 1. Create Abstract Model Provider

```python
# backend/services/base_model_provider.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ModelInfo:
    name: str
    description: str
    capabilities: List[str]
    max_tokens: int
    cost_per_token: float

@dataclass
class GenerationRequest:
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    model: Optional[str] = None

@dataclass
class GenerationResponse:
    content: str
    model: str
    tokens_used: int
    cost: float
    metadata: Dict[str, Any]

class BaseModelProvider(ABC):
    """Abstract base class for model providers."""
    
    @abstractmethod
    async def get_available_models(self) -> List[ModelInfo]:
        """Get list of available models."""
        pass
    
    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate response from model."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check provider health."""
        pass
```

### 2. Implement OpenAI Provider

```python
# backend/services/openai_provider.py
import openai
from .base_model_provider import BaseModelProvider, ModelInfo, GenerationRequest, GenerationResponse

class OpenAIProvider(BaseModelProvider):
    """OpenAI model provider implementation."""
    
    def __init__(self, api_key: str):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.models = {
            "gpt-4": ModelInfo(
                name="gpt-4",
                description="Most capable GPT-4 model",
                capabilities=["chat", "code", "analysis"],
                max_tokens=8192,
                cost_per_token=0.00003
            ),
            "gpt-3.5-turbo": ModelInfo(
                name="gpt-3.5-turbo", 
                description="Fast and efficient GPT-3.5",
                capabilities=["chat", "code"],
                max_tokens=4096,
                cost_per_token=0.000002
            )
        }
    
    async def get_available_models(self) -> List[ModelInfo]:
        return list(self.models.values())
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        response = await self.client.chat.completions.create(
            model=request.model or "gpt-3.5-turbo",
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return GenerationResponse(
            content=response.choices[0].message.content,
            model=response.model,
            tokens_used=response.usage.total_tokens,
            cost=response.usage.total_tokens * self.models[response.model].cost_per_token,
            metadata={"finish_reason": response.choices[0].finish_reason}
        )
    
    async def health_check(self) -> Dict[str, Any]:
        try:
            models = await self.client.models.list()
            return {"status": "healthy", "available_models": len(models.data)}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
```

### 3. Implement Anthropic Provider

```python
# backend/services/anthropic_provider.py
import anthropic
from .base_model_provider import BaseModelProvider, ModelInfo, GenerationRequest, GenerationResponse

class AnthropicProvider(BaseModelProvider):
    """Anthropic Claude model provider."""
    
    def __init__(self, api_key: str):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.models = {
            "claude-3-opus": ModelInfo(
                name="claude-3-opus",
                description="Most capable Claude model",
                capabilities=["chat", "code", "analysis", "reasoning"],
                max_tokens=4096,
                cost_per_token=0.000075
            ),
            "claude-3-sonnet": ModelInfo(
                name="claude-3-sonnet",
                description="Balanced Claude model",
                capabilities=["chat", "code", "analysis"],
                max_tokens=4096,
                cost_per_token=0.000015
            )
        }
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        message = await self.client.messages.create(
            model=request.model or "claude-3-sonnet",
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system=request.system_prompt or "You are a helpful assistant.",
            messages=[{"role": "user", "content": request.prompt}]
        )
        
        return GenerationResponse(
            content=message.content[0].text,
            model=request.model or "claude-3-sonnet",
            tokens_used=message.usage.input_tokens + message.usage.output_tokens,
            cost=(message.usage.input_tokens + message.usage.output_tokens) * self.models[message.model].cost_per_token,
            metadata={"stop_reason": message.stop_reason}
        )
```

### 4. Create Provider Factory

```python
# backend/services/model_provider_factory.py
from typing import Dict, Type
from .base_model_provider import BaseModelProvider
from .ollama_provider import OllamaProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from ..config import get_settings

class ModelProviderFactory:
    """Factory for creating model providers."""
    
    _providers: Dict[str, Type[BaseModelProvider]] = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }
    
    @classmethod
    def create_provider(cls, provider_name: str, **kwargs) -> BaseModelProvider:
        """Create a model provider instance."""
        settings = get_settings()
        
        if provider_name not in cls._providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        provider_class = cls._providers[provider_name]
        
        # Provider-specific configuration
        if provider_name == "ollama":
            return provider_class(base_url=settings.ollama.base_url)
        elif provider_name == "openai":
            api_key = kwargs.get("api_key") or settings.openai.api_key
            return provider_class(api_key=api_key)
        elif provider_name == "anthropic":
            api_key = kwargs.get("api_key") or settings.anthropic.api_key
            return provider_class(api_key=api_key)
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseModelProvider]):
        """Register a new provider."""
        cls._providers[name] = provider_class
```

### 5. Update Translation Service

```python
# Update backend/services/translation_service.py
from .model_provider_factory import ModelProviderFactory

class TranslationService:
    def __init__(self, provider_name: str = "ollama"):
        self.settings = get_settings()
        self.provider = ModelProviderFactory.create_provider(provider_name)
    
    async def translate_to_graphql(
        self,
        natural_query: str,
        schema_context: str = "",
        model: Optional[str] = None,
        provider: Optional[str] = None
    ) -> TranslationResult:
        """Translate using specified provider."""
        
        # Switch provider if specified
        if provider and provider != self.provider.__class__.__name__.lower():
            self.provider = ModelProviderFactory.create_provider(provider)
        
        # Build request
        request = GenerationRequest(
            prompt=self._build_user_prompt(natural_query),
            system_prompt=self._build_system_prompt(schema_context),
            model=model,
            temperature=0.3,
            max_tokens=2048
        )
        
        # Generate response
        response = await self.provider.generate(request)
        
        # Process and return result
        return self._process_response(response, natural_query)
```

## Advanced Customization

### 1. Custom Tool Extensions

Create a plugin system for tools:

```python
# backend/mcp_server/tools/plugin_manager.py
from typing import Dict, Callable, Any
import importlib
import pkgutil

class ToolPluginManager:
    """Manages dynamically loaded tool plugins."""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_schemas: Dict[str, Dict] = {}
    
    def load_plugins(self, plugin_directory: str = "tools/plugins"):
        """Load tool plugins from directory."""
        for finder, name, ispkg in pkgutil.iter_modules([plugin_directory]):
            module = importlib.import_module(f"{plugin_directory.replace('/', '.')}.{name}")
            if hasattr(module, 'register_tool'):
                tool_info = module.register_tool()
                self.register_tool(tool_info['name'], tool_info['handler'], tool_info['schema'])
    
    def register_tool(self, name: str, handler: Callable, schema: Dict[str, Any]):
        """Register a new tool."""
        self.tools[name] = handler
        self.tool_schemas[name] = schema
    
    def get_tools(self) -> Dict[str, Dict]:
        """Get all registered tools."""
        return self.tool_schemas
    
    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool by name."""
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found")
        return await self.tools[name](arguments)
```

### 2. Example Custom Tool Plugin

```python
# backend/mcp_server/tools/plugins/database_analyzer.py
from typing import Dict, Any

async def analyze_database_schema(arguments: Dict[str, Any]):
    """Analyze database schema for GraphQL optimization."""
    connection_string = arguments.get("connection_string")
    database_type = arguments.get("database_type", "postgresql")
    
    # Implementation for analyzing database schema
    analyzer = DatabaseSchemaAnalyzer(database_type)
    schema_info = await analyzer.analyze(connection_string)
    
    return {
        "tables": schema_info.tables,
        "relationships": schema_info.relationships,
        "indexes": schema_info.indexes,
        "graphql_suggestions": schema_info.generate_graphql_suggestions()
    }

def register_tool():
    """Register this tool with the plugin manager."""
    return {
        "name": "analyze_database_schema",
        "handler": analyze_database_schema,
        "schema": {
            "type": "object",
            "properties": {
                "connection_string": {"type": "string", "description": "Database connection string"},
                "database_type": {"type": "string", "enum": ["postgresql", "mysql", "sqlite"], "default": "postgresql"}
            },
            "required": ["connection_string"]
        }
    }
```

### 3. Configuration System

```python
# backend/config/custom_settings.py
from pydantic import BaseSettings
from typing import Dict, List, Optional

class CustomModelSettings(BaseSettings):
    """Custom model provider settings."""
    
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    huggingface_token: Optional[str] = None
    
    # Custom provider endpoints
    custom_providers: Dict[str, str] = {}
    
    # Model preferences by use case
    translation_models: Dict[str, str] = {
        "fast": "gpt-3.5-turbo",
        "accurate": "gpt-4",
        "local": "llama2"
    }
    
    # Custom prompt templates
    prompt_templates: Dict[str, str] = {}
    
    class Config:
        env_prefix = "CUSTOM_"
```

## Examples

### 1. Complete Custom Provider Example

```python
# backend/services/huggingface_provider.py
import aiohttp
from .base_model_provider import BaseModelProvider, ModelInfo, GenerationRequest, GenerationResponse

class HuggingFaceProvider(BaseModelProvider):
    """HuggingFace Inference API provider."""
    
    def __init__(self, api_token: str, endpoint_url: str = "https://api-inference.huggingface.co"):
        self.api_token = api_token
        self.endpoint_url = endpoint_url
        self.models = {
            "codellama/CodeLlama-7b-Instruct-hf": ModelInfo(
                name="codellama/CodeLlama-7b-Instruct-hf",
                description="Code Llama 7B Instruct",
                capabilities=["code", "chat"],
                max_tokens=4096,
                cost_per_token=0.0
            )
        }
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        headers = {"Authorization": f"Bearer {self.api_token}"}
        
        payload = {
            "inputs": f"{request.system_prompt}\n\nUser: {request.prompt}\nAssistant:",
            "parameters": {
                "temperature": request.temperature,
                "max_new_tokens": request.max_tokens,
                "return_full_text": False
            }
        }
        
        model_id = request.model or "codellama/CodeLlama-7b-Instruct-hf"
        url = f"{self.endpoint_url}/models/{model_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                result = await response.json()
                
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", "")
                else:
                    generated_text = ""
                
                return GenerationResponse(
                    content=generated_text,
                    model=model_id,
                    tokens_used=len(generated_text.split()),  # Approximation
                    cost=0.0,  # HuggingFace Inference API is free for many models
                    metadata={"provider": "huggingface"}
                )
```

### 2. Using the System

```python
# Example usage script
import asyncio
from backend.services.translation_service import TranslationService

async def main():
    # Initialize with different providers
    ollama_service = TranslationService(provider_name="ollama")
    openai_service = TranslationService(provider_name="openai") 
    
    natural_query = "Get all active users with their recent posts and comment counts"
    schema = """
    type User {
        id: ID!
        email: String!
        isActive: Boolean!
        posts: [Post!]!
    }
    
    type Post {
        id: ID!
        title: String!
        author: User!
        comments: [Comment!]!
    }
    """
    
    # Compare results from different providers
    ollama_result = await ollama_service.translate_to_graphql(natural_query, schema, model="llama2")
    openai_result = await openai_service.translate_to_graphql(natural_query, schema, model="gpt-3.5-turbo")
    
    print("Ollama Result:")
    print(f"Query: {ollama_result.graphql_query}")
    print(f"Confidence: {ollama_result.confidence}")
    
    print("\nOpenAI Result:")
    print(f"Query: {openai_result.graphql_query}")
    print(f"Confidence: {openai_result.confidence}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Custom Prompt System

```python
# backend/services/prompt_manager.py
from typing import Dict, Any
import jinja2

class PromptManager:
    """Advanced prompt management with templates."""
    
    def __init__(self):
        self.env = jinja2.Environment(
            loader=jinja2.DictLoader({
                "translation": """
You are an expert GraphQL query translator for {{ domain }}.

Context: {{ context }}
Schema: {{ schema }}

Rules:
{% for rule in rules %}
- {{ rule }}
{% endfor %}

Natural Language Query: "{{ query }}"

Generate a GraphQL query following the exact JSON format:
{
  "graphql": "query string here",
  "confidence": 0.95,
  "explanation": "brief explanation"
}
                """,
                "optimization": """
Optimize this GraphQL query for {{ target }} performance:

Original Query:
{{ query }}

Schema Context:
{{ schema }}

Focus on:
{% for focus in optimization_targets %}
- {{ focus }}
{% endfor %}

Return optimized query with explanations.
                """
            })
        )
    
    def render_prompt(self, template_name: str, **kwargs) -> str:
        """Render a prompt template with variables."""
        template = self.env.get_template(template_name)
        return template.render(**kwargs)

# Usage
prompt_manager = PromptManager()

translation_prompt = prompt_manager.render_prompt(
    "translation",
    domain="E-commerce",
    context="Product catalog with users, orders, and reviews",
    schema=schema_text,
    rules=[
        "Always include product availability",
        "Use fragments for address information",
        "Paginate lists with first/after pattern"
    ],
    query=natural_query
)
```

This guide provides a comprehensive foundation for extending your MPPW MCP system. You can mix and match these patterns to create exactly the functionality you need. Would you like me to elaborate on any specific section or create additional examples for particular use cases? 