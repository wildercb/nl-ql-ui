# Agent Interaction Guide

This guide explains how agents interact within the MPPW-MCP system, how to modify their behavior, and how to extend the system with new agents.

## Overview

The MPPW-MCP system uses a sophisticated multi-agent architecture where different agents specialize in specific tasks and collaborate to process natural language queries into GraphQL operations.

## Agent Architecture

### Core Components

1. **Agent Configuration System** (`backend/config/agent_config.py`)
   - Centralized configuration for all agents
   - Easy model switching and capability management
   - Pipeline definitions

2. **Enhanced Orchestration Service** (`backend/services/enhanced_orchestration_service.py`)
   - Coordinates agent execution
   - Manages streaming responses
   - Handles error recovery and fallbacks

3. **Agent Implementations** (`backend/agents/implementations.py`)
   - Concrete agent classes
   - Specialized capabilities for each agent type

## Agent Types and Roles

### 1. Rewriter Agent
- **Purpose**: Improves and clarifies natural language queries
- **Model**: `phi3:mini` (fast, low memory)
- **Input**: Raw user query
- **Output**: Refined, clearer query
- **When it runs**: First step in standard and comprehensive pipelines

### 2. Translator Agent
- **Purpose**: Converts natural language to GraphQL
- **Model**: `phi3:mini` (reliable, fast)
- **Input**: Natural language query (possibly rewritten)
- **Output**: GraphQL query with confidence score
- **When it runs**: Core step in all pipelines

### 3. Reviewer Agent
- **Purpose**: Reviews and validates GraphQL queries
- **Model**: `phi3:mini` (code analysis)
- **Input**: GraphQL query + original intent
- **Output**: Review result with optional suggested corrections
- **When it runs**: Final step in standard pipeline, middle step in comprehensive
- **Special feature**: Can suggest replacement queries that update the UI live

### 4. Data Reviewer Agent (NEW)
- **Purpose**: Analyzes actual data results for accuracy
- **Model**: `gemma3:4b` (multimodal support)
- **Input**: GraphQL query + returned data + original intent
- **Output**: Accuracy assessment with optional query refinements
- **When it runs**: Only in comprehensive pipeline
- **Special features**: 
  - Handles multimodal data (images, documents)
  - Iteratively refines queries until satisfied
  - Automatically executes new queries

### 5. Optimizer Agent
- **Purpose**: Optimizes queries for performance
- **Model**: `phi3:mini` (performance analysis)
- **Input**: GraphQL query + schema context
- **Output**: Optimized query with performance improvements
- **When it runs**: Comprehensive pipeline when needed

## Pipeline Strategies

### Fast Pipeline (`translate`)
```
User Query → Translator Agent → GraphQL Query
```
- **Use case**: Simple queries, high throughput
- **Agents**: translator
- **Timeout**: 15 seconds
- **Model**: phi3:mini

### Standard Pipeline (`multi-agent`)
```
User Query → Rewriter Agent → Translator Agent → Reviewer Agent → GraphQL Query
```
- **Use case**: General queries, production workloads
- **Agents**: rewriter, translator, reviewer
- **Timeout**: 45 seconds
- **Models**: All use phi3:mini for consistency

### Comprehensive Pipeline (`enhanced-agents`)
```
User Query → Rewriter Agent → Translator Agent → Reviewer Agent → Data Reviewer Agent → Final Query
```
- **Use case**: Complex queries, critical applications, multimodal data
- **Agents**: rewriter, translator, reviewer, data_reviewer
- **Timeout**: 90 seconds
- **Models**: phi3:mini for text processing, gemma3:4b for data review
- **Special behavior**: Data reviewer can iteratively refine queries

## Frontend Integration

### UI Flow
1. User selects pipeline strategy (Translate/Multi-Agent/Enhanced Agents)
2. `HomeView.vue` calls `runPipeline()` with appropriate strategy
3. Streaming events update the UI in real-time
4. Special handling for reviewer and data reviewer query suggestions

### Key Events
- `agent_start`: Agent begins processing
- `agent_token`: Streaming tokens from agent
- `agent_complete`: Agent finishes with results
- `pipeline_complete`: Entire pipeline finished

### Live Query Updates
When reviewer or data reviewer suggests a new query:
1. GraphQL query box updates automatically
2. `runDataQuery()` executes automatically
3. Results appear in data results section
4. Process continues until data reviewer is satisfied

## Configuration and Customization

### Easy Model Changes
```python
# In backend/config/agent_config.py
from config.agent_config import agent_config_manager

# Change model for specific agent
agent_config_manager.update_agent_model('translator', 'gemma3:4b')

# Change model for entire size category
agent_config_manager.set_model_for_size(ModelSize.SMALL, 'phi3:mini')
```

### Adding New Agents
1. **Create Agent Class** (in `backend/agents/implementations.py`):
```python
@agent(
    name="my_custom_agent",
    capabilities=[AgentCapability.CUSTOM],
    depends_on=["translator_agent"],
    description="My custom agent functionality"
)
class MyCustomAgent(BaseAgent):
    async def run(self, context: AgentContext, config: Dict[str, Any]) -> Dict[str, Any]:
        # Your agent logic here
        return {"result": "processed"}
```

2. **Add Agent Configuration**:
```python
# In agent_config.py
"my_custom_agent": AgentConfig(
    name="my_custom_agent",
    role=AgentRole.CUSTOM,
    model="phi3:mini",
    fallback_model="phi3:mini",
    timeout=30.0,
    required_capabilities=["custom_processing"]
)
```

3. **Add to Pipeline**:
```python
# Update pipeline configuration
"custom_pipeline": PipelineConfig(
    name="custom_pipeline",
    description="Pipeline with custom agent",
    agents=[
        self.agent_configs["translator"],
        self.agent_configs["my_custom_agent"]
    ]
)
```

### Creating Custom Pipelines
```python
# Create a custom pipeline
custom_pipeline = agent_config_manager.create_custom_pipeline(
    name="fast_review",
    agent_names=["translator", "reviewer"],
    description="Fast translation with review",
    timeout=30.0,
    optimization_level="speed"
)
```

## Error Handling and Fallbacks

### Model Fallbacks
- Each agent has a fallback model configured
- If primary model fails (memory issues, timeout), fallback is used
- Ollama memory issues automatically trigger lighter models

### Translation Fallbacks
- If translation fails completely, system provides emergency fallback query
- Fallback queries are domain-specific (e.g., thermal scans query)
- Confidence scores reflect fallback usage

### Pipeline Recovery
- Individual agent failures don't stop the pipeline
- Downstream agents receive fallback results
- Final output includes warnings about failures

## Performance Optimization

### Memory Management
- Small models (phi3:mini) for text processing
- Medium models (gemma3:4b) only for multimodal needs
- Large models (gemma3n:e4b) avoided due to memory constraints

### Streaming Optimization
- Real-time token streaming for immediate feedback
- Parallel processing where possible
- Early termination on satisfactory results

### Timeout Management
- Agent-specific timeouts prevent hanging
- Pipeline-level timeouts for overall guarantees
- Graceful degradation on timeout

## Monitoring and Debugging

### Logging
- Each agent logs start/completion with timing
- Model usage tracked for optimization
- Error conditions logged with context

### Metrics
- Processing time per agent
- Model success/failure rates
- Pipeline completion statistics

### Debug Mode
- Enable detailed logging in config
- Stream intermediate results
- Expose internal agent state

## Extension Points

### New Agent Types
1. **Validator Agent**: Additional validation logic
2. **Analyzer Agent**: Deep query analysis
3. **Formatter Agent**: Output formatting
4. **Security Agent**: Security scanning

### New Pipeline Strategies
1. **Parallel Pipeline**: Run multiple agents simultaneously
2. **Conditional Pipeline**: Dynamic agent selection
3. **Feedback Pipeline**: User feedback integration
4. **Learning Pipeline**: Adaptive improvement

### Integration Points
1. **External APIs**: Call external services from agents
2. **Database Integration**: Direct database queries
3. **File Processing**: Handle file uploads
4. **Real-time Data**: Stream live data updates

## Best Practices

### Agent Design
- Keep agents focused on single responsibilities
- Use appropriate model sizes for tasks
- Implement proper error handling
- Provide meaningful feedback

### Pipeline Design
- Order agents logically (dependencies)
- Set appropriate timeouts
- Plan for failure scenarios
- Monitor performance metrics

### Configuration Management
- Use centralized configuration
- Version control configuration changes
- Test configuration changes thoroughly
- Document custom configurations

## Troubleshooting

### Common Issues
1. **Ollama Memory Issues**: Use smaller models
2. **Timeout Errors**: Increase timeout or use faster models
3. **Translation Failures**: Check ICL examples and schema
4. **Agent Not Running**: Verify configuration and dependencies

### Debug Steps
1. Check logs for error messages
2. Verify agent configuration
3. Test individual agents
4. Check model availability
5. Validate pipeline configuration

This guide provides the foundation for understanding and extending the agent system. The modular design makes it easy to add new capabilities while maintaining system stability and performance. 