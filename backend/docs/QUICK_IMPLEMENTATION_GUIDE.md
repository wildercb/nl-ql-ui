# Quick Implementation Guide

This guide shows how to make common changes to the MPPW-MCP modular system.

## 🚀 Frontend Integration Complete

The frontend now uses the MCP server with these new agent options:

- **MCP Fast** - Translation only (fastest)
- **MCP Standard** - Rewrite → Translate → Review  
- **MCP Comprehensive** - All agents + optimization + data review
- **MCP Adaptive** - Auto-selected strategy based on query complexity

## 📝 Common Tasks

### 1. Add a New Prompt (2 minutes)

Create a YAML file in the appropriate directory:

```bash
# For agent-specific prompts
backend/prompts/agents/translator/my_prompt.yaml

# For domain-specific prompts  
backend/prompts/domains/ecommerce/my_prompt.yaml

# For strategy-specific prompts
backend/prompts/strategies/detailed/my_prompt.yaml
```

Example prompt file:

```yaml
name: ecommerce_advanced
agent_type: translator
domain: ecommerce
strategy: detailed
description: "Advanced e-commerce GraphQL generation"
version: "1.0"
tags: [ecommerce, advanced, products]
variables: [query, schema_context, user_preferences]
content: |
  You are an e-commerce GraphQL expert.
  
  Query: {{ query }}
  Schema: {{ schema_context }}
  
  Focus on:
  - Product catalog optimization
  - Inventory management
  - Price calculations
  - Customer segmentation
  
  Return JSON: {"graphql": "...", "confidence": 0.95}
```

The prompt is automatically loaded and available immediately!

### 2. Add a New LLM Provider (5 minutes)

Create a provider class:

```python
# backend/services/llm_providers/anthropic_provider.py
from .base import BaseLLMProvider
import httpx

class AnthropicProvider(BaseLLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com"
    
    async def chat_completion(self, messages, model, **kwargs):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/v1/messages",
                headers={"x-api-key": self.api_key},
                json={
                    "model": model,
                    "messages": messages,
                    **kwargs
                }
            )
            return response.json()
    
    async def health_check(self):
        return {"status": "healthy", "provider": "anthropic"}
    
    async def list_models(self):
        return ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
```

Register it:

```python
# backend/services/llm_providers/__init__.py
from .anthropic_provider import AnthropicProvider
from .factory import LLMProviderFactory

factory = LLMProviderFactory()
factory.register_provider("anthropic", AnthropicProvider)
```

Use it:

```python
# In your configuration or code
provider = get_llm_provider("anthropic")
result = await provider.chat_completion(messages, "claude-3-sonnet")
```

### 3. Add a Custom Agent (10 minutes)

```python
# backend/agents/specialized/analytics_agent.py
from agents.base import BaseAgent, AgentContext, AgentCapability
from agents.registry import agent
from prompts import get_prompt_manager
from prompts.types import PromptContext, AgentType, DomainType

@agent(
    name="analytics_processor",
    capabilities=[AgentCapability.TRANSLATE, AgentCapability.ANALYZE],
    description="Specialized agent for analytics queries"
)
class AnalyticsAgent(BaseAgent):
    
    async def run(self, ctx: AgentContext, config=None, **kwargs):
        # Get analytics-specific prompt
        pm = get_prompt_manager()
        
        prompt_context = PromptContext(
            query=ctx.original_query,
            domain=DomainType.ANALYTICS,
            schema_context=ctx.schema_context,
            metadata={"focus": "metrics_and_dimensions"}
        )
        
        # Generate appropriate prompt
        prompt_result = pm.generate_smart_prompt(
            context=prompt_context,
            agent_type=AgentType.TRANSLATOR
        )
        
        # Call LLM with optimized prompt
        model = config.get('model', 'gemma3:4b')
        llm_result = await self.call_llm(
            prompt_result.content,
            model=model,
            temperature=0.3  # Lower temp for analytical accuracy
        )
        
        # Process and store results
        processed_result = await self.process_analytics_result(llm_result)
        ctx.add_agent_output(self.name, processed_result)
    
    async def process_analytics_result(self, llm_result):
        # Add analytics-specific processing
        return {
            'graphql': llm_result.get('graphql', ''),
            'metrics_identified': self.extract_metrics(llm_result),
            'dimensions_identified': self.extract_dimensions(llm_result),
            'suggested_aggregations': self.suggest_aggregations(llm_result)
        }
```

### 4. Add a Custom MCP Tool (15 minutes)

```python
# backend/mcp_server/tools/specialized/analytics_optimizer.py
from mcp_server.tools.base import BaseMCPTool
from typing import Dict, Any

class AnalyticsOptimizerTool(BaseMCPTool):
    name = "optimize_analytics_query"
    description = "Optimizes GraphQL queries for analytics workloads"
    
    async def execute(self, **kwargs) -> Dict[str, Any]:
        query = kwargs.get('query', '')
        graphql = kwargs.get('graphql', '')
        
        # Analyze query for analytics patterns
        analysis = await self.analyze_analytics_query(query, graphql)
        
        # Generate optimizations
        optimizations = await self.generate_optimizations(analysis)
        
        return {
            'original_query': query,
            'original_graphql': graphql,
            'analysis': analysis,
            'optimized_graphql': optimizations.get('graphql'),
            'performance_improvements': optimizations.get('improvements'),
            'caching_recommendations': optimizations.get('caching')
        }
    
    async def analyze_analytics_query(self, query: str, graphql: str) -> Dict[str, Any]:
        # Implement analytics-specific analysis
        return {
            'query_type': 'aggregation',
            'time_series_detected': True,
            'grouping_fields': ['date', 'category'],
            'metrics': ['revenue', 'count', 'average']
        }
    
    async def generate_optimizations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        # Generate optimized query based on analysis
        return {
            'graphql': 'optimized GraphQL here',
            'improvements': ['Added pagination', 'Optimized field selection'],
            'caching': {'ttl': 300, 'key_pattern': 'analytics:${date}:${category}'}
        }

# Register the tool
from mcp_server.enhanced_agent import EnhancedMCPServer
server = EnhancedMCPServer()
server.add_tool(AnalyticsOptimizerTool())
```

### 5. Create a Custom Pipeline (20 minutes)

```python
# backend/agents/pipelines/analytics_pipeline.py
from agents.pipeline import Pipeline
from agents.context import AgentContext
from typing import Dict, Any

class AnalyticsPipeline(Pipeline):
    name = "analytics"
    description = "Specialized pipeline for analytics queries"
    
    async def execute(self, context: AgentContext) -> Dict[str, Any]:
        # Step 1: Analytics-specific preprocessing
        await self.run_agent("analytics_preprocessor", context, {
            'focus': 'metrics_extraction',
            'time_series_detection': True
        })
        
        # Step 2: Specialized translation
        await self.run_agent("analytics_processor", context, {
            'model': 'gemma3:4b',
            'temperature': 0.2
        })
        
        # Step 3: Performance optimization
        await self.run_agent("query_optimizer", context, {
            'optimization_level': 'aggressive',
            'caching_enabled': True
        })
        
        # Step 4: Analytics validation
        await self.run_agent("analytics_validator", context, {
            'check_aggregations': True,
            'validate_time_series': True
        })
        
        return self.collect_results(context)

# Register pipeline
from mcp_server.enhanced_agent import EnhancedMCPServer
server = EnhancedMCPServer()
server.register_pipeline("analytics", AnalyticsPipeline())
```

### 6. Update Frontend for New Features (5 minutes)

Add new pipeline option:

```typescript
// frontend/src/views/HomeView.vue
const pipelineOptions = [
  { label: 'MCP Fast', value: 'fast', description: 'Translation only' },
  { label: 'MCP Standard', value: 'standard', description: 'Standard pipeline' },
  { label: 'MCP Comprehensive', value: 'comprehensive', description: 'Full pipeline' },
  { label: 'MCP Adaptive', value: 'adaptive', description: 'Auto-selected strategy' },
  { label: 'MCP Analytics', value: 'analytics', description: 'Analytics-optimized processing' }, // NEW
] as const;
```

The new pipeline is immediately available in the dropdown!

## 🔧 Configuration Changes

### Environment Variables

```bash
# .env
ANTHROPIC_API_KEY=your_key_here
HUGGINGFACE_API_KEY=your_key_here
ANALYTICS_CACHE_TTL=300
```

### Provider Configuration

```yaml
# config/providers.yaml
llm_providers:
  default: ollama
  providers:
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}
      default_model: claude-3-sonnet
      max_tokens: 4096
    huggingface:
      api_key: ${HUGGINGFACE_API_KEY}
      base_url: https://api-inference.huggingface.co
      default_model: meta-llama/Llama-2-7b-chat-hf
```

## 🧪 Testing

### Test New Prompts

```python
# tests/test_prompts.py
def test_analytics_prompt():
    from prompts import get_prompt_manager
    from prompts.types import PromptContext, DomainType
    
    pm = get_prompt_manager()
    context = PromptContext(
        query="Show revenue by month for the last year",
        domain=DomainType.ANALYTICS
    )
    
    result = pm.generate_smart_prompt(context)
    assert "analytics" in result.content.lower()
    assert "revenue" in result.content.lower()
```

### Test New Agents

```python
# tests/test_agents.py
async def test_analytics_agent():
    from agents.specialized.analytics_agent import AnalyticsAgent
    from agents.context import AgentContext
    
    agent = AnalyticsAgent()
    context = AgentContext(
        original_query="Revenue by category last month",
        domain="analytics"
    )
    
    await agent.run(context)
    result = context.get_agent_output('analytics_processor')
    
    assert result is not None
    assert 'metrics_identified' in result
```

### Test MCP Integration

```bash
cd backend
python test_enhanced_mcp.py --strategy analytics --query "Show sales metrics by region"
```

## 📊 Monitoring

### Check Prompt Usage

```python
from prompts import get_prompt_manager

pm = get_prompt_manager()
stats = pm.get_statistics()

print(f"Total templates: {stats['total_templates']}")
print(f"By strategy: {stats['by_strategy']}")
print(f"By domain: {stats['by_domain']}")
```

### Monitor Performance

```python
import logging
logging.getLogger('mcp_server').setLevel(logging.INFO)
logging.getLogger('agents').setLevel(logging.INFO)
logging.getLogger('prompts').setLevel(logging.INFO)
```

## 🎯 Quick Fixes

### Prompt Not Working?

1. Check file syntax: `python -c "import yaml; yaml.safe_load(open('your_prompt.yaml'))"`
2. Reload prompts: `pm.reload_templates()`
3. Check logs: `tail -f logs/mcp_server.log`

### Agent Not Responding?

1. Check registration: `from agents.registry import list_agents; print(list_agents())`
2. Verify capabilities: Make sure agent has required capabilities
3. Check dependencies: Ensure all required agents are available

### Provider Connection Issues?

1. Test health: `await provider.health_check()`
2. Check credentials: Verify API keys and endpoints
3. Test models: `await provider.list_models()`

## 🔄 Hot Reload Development

During development, changes are automatically picked up:

- **Prompts**: Auto-reload every 30 seconds
- **Configuration**: Requires restart
- **Agents**: Requires restart  
- **Tools**: Requires restart

For faster development, use the test script:

```bash
# Test changes immediately
python test_enhanced_mcp.py --model gemma3:4b --strategy your_strategy
```

This guide covers the most common tasks. The modular system makes most changes quick and straightforward! 