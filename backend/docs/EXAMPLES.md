# MPPW MCP Examples

This document provides practical examples for using and extending the MPPW MCP system.

## Basic Usage Examples

### 1. Making Translation Requests

#### REST API
```bash
curl -X POST "http://localhost:8000/api/v1/translation/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "natural_query": "Get all users with their recent posts",
    "schema_context": "type User { id: ID!, posts: [Post!]! }",
    "model": "llama2"
  }'
```

#### Python Client
```python
import httpx

async def translate_query():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/translation/translate",
            json={
                "natural_query": "Find active users with email verification",
                "schema_context": "type User { id: ID!, email: String!, isActive: Boolean! }",
                "model": "gpt-3.5-turbo"
            }
        )
        return response.json()
```

### 2. Using Different Model Providers

```python
from backend.services.translation_service import TranslationService

# Ollama provider
ollama_service = TranslationService(provider_name="ollama")
result1 = await ollama_service.translate_to_graphql(
    "Get user profiles", schema_context, model="llama2"
)

# OpenAI provider  
openai_service = TranslationService(provider_name="openai")
result2 = await openai_service.translate_to_graphql(
    "Get user profiles", schema_context, model="gpt-4"
)
```

## Customization Examples

### 1. Custom Prompt Templates

```python
# backend/services/custom_prompts.py
class EcommercePromptManager:
    def __init__(self):
        self.templates = {
            "product_query": """
You are an e-commerce GraphQL expert.

Context: {{ context }}
Schema: {{ schema }}

Rules for e-commerce queries:
- Always include product availability
- Use pagination for product lists
- Include price and inventory data
- Follow camelCase conventions

Query: "{{ query }}"

Return JSON: {"graphql": "...", "confidence": 0.95}
            """,
            "user_query": """
You are a user management expert.

Generate GraphQL for: "{{ query }}"
Schema: {{ schema }}

Always include: id, email, profile.displayName
Use fragments for user data consistency.
            """
        }
    
    def get_prompt(self, query_type: str, **kwargs) -> str:
        from jinja2 import Template
        template = Template(self.templates.get(query_type, self.templates["product_query"]))
        return template.render(**kwargs)

# Usage in translation service
prompt_manager = EcommercePromptManager()
system_prompt = prompt_manager.get_prompt(
    "product_query",
    query=natural_query,
    context="Product catalog with inventory",
    schema=schema_text
)
```

### 2. Adding a Custom Model Provider

```python
# backend/services/huggingface_provider.py
import aiohttp
from .base_model_provider import BaseModelProvider, GenerationRequest, GenerationResponse

class HuggingFaceProvider(BaseModelProvider):
    def __init__(self, api_token: str):
        self.api_token = api_token
        self.endpoint = "https://api-inference.huggingface.co"
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        headers = {"Authorization": f"Bearer {self.api_token}"}
        payload = {
            "inputs": f"{request.system_prompt}\n\n{request.prompt}",
            "parameters": {
                "temperature": request.temperature,
                "max_new_tokens": request.max_tokens
            }
        }
        
        model_id = request.model or "microsoft/DialoGPT-large"
        url = f"{self.endpoint}/models/{model_id}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                result = await response.json()
                generated_text = result[0]["generated_text"] if result else ""
                
                return GenerationResponse(
                    content=generated_text,
                    model=model_id,
                    tokens_used=len(generated_text.split()),
                    cost=0.0,  # Free tier
                    metadata={"provider": "huggingface"}
                )
```

### 3. Custom MCP Tool

```python
# backend/mcp_server/tools/plugins/schema_analyzer.py
async def analyze_schema_complexity(arguments: dict):
    """Analyze GraphQL schema complexity and suggest optimizations."""
    schema_text = arguments.get("schema", "")
    
    # Parse schema
    type_count = schema_text.count("type ")
    field_count = schema_text.count(":")
    relation_count = schema_text.count("[") + schema_text.count("!")
    
    complexity_score = (type_count * 2) + (field_count * 0.5) + (relation_count * 1.5)
    
    recommendations = []
    if complexity_score > 100:
        recommendations.append("Consider breaking into smaller schema modules")
    if field_count > type_count * 10:
        recommendations.append("Some types may have too many fields")
    
    return {
        "complexity_score": complexity_score,
        "type_count": type_count,
        "field_count": field_count,
        "recommendations": recommendations,
        "schema_health": "good" if complexity_score < 50 else "needs_attention"
    }

def register_tool():
    return {
        "name": "analyze_schema_complexity",
        "handler": analyze_schema_complexity,
        "schema": {
            "type": "object",
            "properties": {
                "schema": {"type": "string", "description": "GraphQL schema to analyze"}
            },
            "required": ["schema"]
        }
    }
```

### 4. Domain-Specific Translation Strategy

```python
# backend/services/strategies/ecommerce_strategy.py
from ..translation_service import TranslationStrategy, TranslationResult

class EcommerceTranslationStrategy(TranslationStrategy):
    """Specialized translation for e-commerce queries."""
    
    async def translate(self, query: str, context: str) -> TranslationResult:
        # Classify e-commerce query type
        query_type = self._classify_ecommerce_query(query)
        
        # Use specialized prompts
        if query_type == "product_search":
            return await self._handle_product_search(query, context)
        elif query_type == "order_management":
            return await self._handle_order_query(query, context)
        elif query_type == "inventory":
            return await self._handle_inventory_query(query, context)
        else:
            return await self._handle_general_query(query, context)
    
    def _classify_ecommerce_query(self, query: str) -> str:
        query_lower = query.lower()
        if any(word in query_lower for word in ["product", "item", "search", "catalog"]):
            return "product_search"
        elif any(word in query_lower for word in ["order", "purchase", "transaction"]):
            return "order_management" 
        elif any(word in query_lower for word in ["inventory", "stock", "availability"]):
            return "inventory"
        return "general"
    
    async def _handle_product_search(self, query: str, context: str) -> TranslationResult:
        # E-commerce specific product search logic
        prompt = f"""
        Generate a GraphQL query for product search: "{query}"
        
        Always include:
        - Product ID, name, price
        - Availability status
        - Images and descriptions
        - Category information
        - Pagination (first, after)
        
        Schema context: {context}
        """
        
        # Use model to generate query
        response = await self.model_provider.generate(prompt)
        
        return TranslationResult(
            graphql_query=response.content,
            confidence=0.9,
            query_type="product_search",
            suggestions=["Consider adding filters for price range", "Include product ratings"]
        )

# Register the strategy
TranslationService.register_strategy("ecommerce", EcommerceTranslationStrategy)
```

## Advanced Usage Examples

### 1. Multi-Provider Comparison

```python
async def compare_providers():
    """Compare translation results from multiple providers."""
    providers = ["ollama", "openai", "anthropic"]
    query = "Get top-selling products with customer reviews"
    schema = "type Product { id: ID!, name: String!, reviews: [Review!]! }"
    
    results = {}
    for provider in providers:
        service = TranslationService(provider_name=provider)
        result = await service.translate_to_graphql(query, schema)
        results[provider] = {
            "query": result.graphql_query,
            "confidence": result.confidence,
            "processing_time": result.processing_time
        }
    
    # Analyze results
    best_confidence = max(results.values(), key=lambda x: x["confidence"])
    fastest = min(results.values(), key=lambda x: x["processing_time"])
    
    return {
        "results": results,
        "best_confidence": best_confidence,
        "fastest": fastest
    }
```

### 2. Batch Processing

```python
async def batch_translate_queries():
    """Process multiple queries in batch."""
    queries = [
        "Get all users",
        "Find products by category", 
        "Get order history for user",
        "List available payment methods"
    ]
    
    schema_context = "type User { orders: [Order!]! } type Product { category: Category! }"
    
    # Use batch endpoint
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/translation/translate/batch",
            json={
                "queries": queries,
                "schema_context": schema_context,
                "model": "gpt-3.5-turbo"
            }
        )
        return response.json()
```

### 3. Real-time Translation with WebSocket

```python
import asyncio
import websockets
import json

async def real_time_translation():
    """Real-time translation using WebSocket connection."""
    uri = "ws://localhost:8000/ws/translate"
    
    async with websockets.connect(uri) as websocket:
        # Send queries
        queries = [
            "Get user profile",
            "List recent orders", 
            "Find popular products"
        ]
        
        for query in queries:
            message = {
                "type": "translate",
                "data": {
                    "natural_query": query,
                    "schema_context": "type User { profile: Profile! }",
                    "model": "llama2"
                }
            }
            await websocket.send(json.dumps(message))
            
            # Receive response
            response = await websocket.recv()
            result = json.loads(response)
            print(f"Query: {query}")
            print(f"GraphQL: {result['data']['graphql_query']}")
            print("---")

# Run real-time translation
asyncio.run(real_time_translation())
```

### 4. Custom Configuration

```python
# config/custom_config.py
from pydantic import BaseSettings

class CustomSettings(BaseSettings):
    # Domain-specific settings
    domain: str = "ecommerce"
    default_pagination_size: int = 20
    enable_query_caching: bool = True
    
    # Model preferences by use case
    translation_models: dict = {
        "fast": "gpt-3.5-turbo",
        "accurate": "gpt-4", 
        "local": "llama2",
        "specialized": "codellama"
    }
    
    # Custom prompt templates
    custom_prompts: dict = {
        "ecommerce": "You are an e-commerce expert...",
        "analytics": "You are a data analytics expert...",
        "social": "You are a social media platform expert..."
    }
    
    class Config:
        env_prefix = "CUSTOM_"

# Usage
settings = CustomSettings()
model = settings.translation_models.get("accurate", "gpt-3.5-turbo")
prompt_template = settings.custom_prompts.get("ecommerce", "default")
```

These examples demonstrate the flexibility and extensibility of the MPPW MCP system. You can mix and match these patterns to create exactly the functionality you need for your specific use case. 