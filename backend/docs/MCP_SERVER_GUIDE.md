# Enhanced MCP Server Guide

## Overview

The Enhanced MCP Server provides the same functionality as the Enhanced Orchestration Service but through the Model Context Protocol (MCP) interface. It supports streaming responses, multiple pipeline strategies, and comprehensive error handling.

## Features

- **Multiple Pipeline Strategies**: Standard, Fast, Comprehensive, and Adaptive
- **Streaming Support**: Real-time processing updates
- **Batch Processing**: Handle multiple queries concurrently
- **Error Recovery**: Graceful handling of service failures
- **Performance Monitoring**: Built-in execution statistics
- **Python 3.9+ Compatible**: Works without external MCP dependencies

## Architecture

The Enhanced MCP Server is built around the `EnhancedMCPServer` class which wraps the `EnhancedOrchestrationService` and provides MCP-compatible interfaces.

### Key Components

1. **EnhancedMCPServer**: Main server class
2. **Tool Handlers**: Process different pipeline strategies
3. **Streaming Support**: Handle async generators from orchestration service
4. **Error Handling**: Comprehensive error recovery and logging

## Installation and Setup

### Prerequisites

- Python 3.9+
- MongoDB running locally or accessible
- Ollama service with required models

### Installation

1. Ensure all dependencies are installed:
   ```bash
   cd backend
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Configure environment variables (optional):
   ```bash
   export MONGODB_URL="mongodb://localhost:27017"
   export OLLAMA_BASE_URL="http://localhost:11434"
   export OLLAMA_DEFAULT_MODEL="phi3:mini"
   ```

## Running the Server

### Direct Execution

```bash
cd backend
source venv/bin/activate
python mcp_server/enhanced_agent.py
```

### Using the Startup Script

```bash
cd backend
source venv/bin/activate
python run_enhanced_mcp_server.py
```

### Testing the Server

```bash
cd backend
source venv/bin/activate
python test_enhanced_mcp.py
```

## Available Tools

The Enhanced MCP Server provides 5 main tools:

### 1. process_query_standard

**Description**: Process a query using the standard pipeline (rewrite → translate → review)

**Parameters**:
- `query` (required): The natural language query to process
- `pre_model` (optional): Model for query rewriting
- `translator_model` (optional): Model for translation
- `review_model` (optional): Model for review
- `domain_context` (optional): Domain-specific context
- `schema_context` (optional): GraphQL schema context
- `user_id` (optional): User identifier for tracking

**Example**:
```json
{
  "name": "process_query_standard",
  "arguments": {
    "query": "Show me the first 10 thermal scans with temperatures above 60 degrees",
    "translator_model": "phi3:mini",
    "user_id": "user123"
  }
}
```

### 2. process_query_fast

**Description**: Process a query using the fast pipeline (translation only)

**Parameters**:
- `query` (required): The natural language query to process
- `translator_model` (optional): Model for translation
- `schema_context` (optional): GraphQL schema context
- `user_id` (optional): User identifier for tracking

**Example**:
```json
{
  "name": "process_query_fast",
  "arguments": {
    "query": "Get maintenance logs from today",
    "translator_model": "phi3:mini"
  }
}
```

### 3. process_query_comprehensive

**Description**: Process a query using the comprehensive pipeline (all agents + optimization + data review)

**Parameters**:
- `query` (required): The natural language query to process
- `pre_model` (optional): Model for query rewriting
- `translator_model` (optional): Model for translation
- `review_model` (optional): Model for review
- `domain_context` (optional): Domain-specific context
- `schema_context` (optional): GraphQL schema context
- `user_id` (optional): User identifier for tracking

**Example**:
```json
{
  "name": "process_query_comprehensive",
  "arguments": {
    "query": "Find all equipment with critical temperature readings in the last 24 hours",
    "translator_model": "phi3:mini",
    "review_model": "phi3:mini",
    "domain_context": "industrial monitoring"
  }
}
```

### 4. process_query_adaptive

**Description**: Process a query using the adaptive pipeline (strategy selected based on query complexity)

**Parameters**:
- `query` (required): The natural language query to process
- `pre_model` (optional): Model for query rewriting
- `translator_model` (optional): Model for translation
- `review_model` (optional): Model for review
- `domain_context` (optional): Domain-specific context
- `schema_context` (optional): GraphQL schema context
- `user_id` (optional): User identifier for tracking

**Example**:
```json
{
  "name": "process_query_adaptive",
  "arguments": {
    "query": "Analyze thermal patterns and predict maintenance needs",
    "translator_model": "phi3:mini"
  }
}
```

### 5. batch_process_queries

**Description**: Process multiple queries in batch using the specified pipeline strategy

**Parameters**:
- `queries` (required): List of natural language queries to process
- `pipeline_strategy` (optional): Pipeline strategy to use ("standard", "fast", "comprehensive", "adaptive")
- `max_concurrent` (optional): Maximum number of concurrent query processing
- `translator_model` (optional): Model for translation
- `schema_context` (optional): GraphQL schema context
- `user_id` (optional): User identifier for tracking

**Example**:
```json
{
  "name": "batch_process_queries",
  "arguments": {
    "queries": [
      "Show me thermal scans from today",
      "Get maintenance logs for the last week",
      "Find all equipment with temperature above 80 degrees"
    ],
    "pipeline_strategy": "fast",
    "max_concurrent": 3,
    "translator_model": "phi3:mini"
  }
}
```

## Response Format

All tools return structured JSON responses with the following format:

### Successful Response

```json
{
  "original_query": "Show me the first 5 thermal scans",
  "rewritten_query": "Show me the first 5 thermal scans",
  "translation": {
    "graphql_query": "query { thermalScans(first: 5) { id temperature timestamp } }",
    "confidence": 0.95,
    "model_used": "phi3:mini"
  },
  "review": {
    "approved": true,
    "suggestions": [],
    "confidence": 0.92
  },
  "processing_time": 2.34,
  "session_id": "enhanced-mcp-abc123",
  "pipeline_strategy": "fast",
  "events_count": 4
}
```

### Error Response

```json
{
  "session_id": "enhanced-mcp-abc123",
  "error": "Orchestration service not initialized",
  "tool": "process_query_fast",
  "success": false
}
```

### Batch Response

```json
{
  "session_id": "enhanced-mcp-batch123",
  "pipeline_strategy": "fast",
  "total_queries": 3,
  "successful": 3,
  "failed": 0,
  "max_concurrent": 3,
  "results": [
    {
      "query_index": 0,
      "session_id": "enhanced-mcp-batch123-0",
      "original_query": "Show me thermal scans from today",
      "translation": { ... },
      "processing_time": 1.23
    },
    // ... more results
  ]
}
```

## Pipeline Strategies

### Fast Pipeline
- **Agents**: Translation only
- **Use Case**: Quick translations where speed is priority
- **Processing Time**: ~1-3 seconds
- **Recommended For**: Simple queries, real-time applications

### Standard Pipeline
- **Agents**: Rewrite → Translate → Review
- **Use Case**: Balanced approach for most queries
- **Processing Time**: ~3-8 seconds
- **Recommended For**: Production workloads, general use

### Comprehensive Pipeline
- **Agents**: Rewrite → Translate → Review + Optimization + Data Review
- **Use Case**: Complex queries requiring highest quality
- **Processing Time**: ~8-20 seconds
- **Recommended For**: Critical queries, complex analysis

### Adaptive Pipeline
- **Agents**: Automatically selected based on query complexity
- **Use Case**: Automatic strategy selection
- **Processing Time**: Variable based on selected strategy
- **Recommended For**: Mixed workloads, user-facing applications

## Configuration

### Environment Variables

```bash
# Database Configuration
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=mppw_mcp

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=phi3:mini
OLLAMA_TIMEOUT=120

# Neo4j Configuration (for GraphQL schema)
NEO4J_GRAPHQL_ENDPOINT=http://neo4j:4000/graphql

# Logging Configuration
LOG_LEVEL=INFO
```

### Model Configuration

The server supports various Ollama models:

- `phi3:mini` (recommended for development)
- `llama3.2:3b` (good balance of speed and quality)
- `llama3.1:8b` (higher quality, slower)
- `qwen2.5:7b` (excellent for technical queries)

## Error Handling

The Enhanced MCP Server includes comprehensive error handling:

### Common Errors

1. **Model Not Found**: Ollama model not available
   - **Solution**: Pull the model with `ollama pull <model_name>`

2. **Database Connection Error**: MongoDB not accessible
   - **Solution**: Ensure MongoDB is running and accessible

3. **GraphQL Schema Error**: Neo4j service not available
   - **Solution**: Start Neo4j service or update configuration

4. **Timeout Errors**: Query processing takes too long
   - **Solution**: Increase timeout settings or use faster pipeline

### Error Recovery

The server implements graceful error recovery:

- **Service Failures**: Continue processing with degraded functionality
- **Model Errors**: Fallback to default model
- **Network Issues**: Retry with exponential backoff
- **Resource Exhaustion**: Queue management and throttling

## Performance Monitoring

### Built-in Metrics

The server tracks various performance metrics:

- **Query Processing Time**: Per query and aggregate
- **Success/Failure Rates**: By pipeline strategy
- **Model Usage**: Which models are used most
- **Concurrent Processing**: Active query counts

### Accessing Metrics

```python
# Get server information including metrics
info = server.get_server_info()
print(info['capabilities'])
```

## Integration with UI

The Enhanced MCP Server can be integrated with the existing UI by:

1. **Running alongside the FastAPI server**
2. **Using as a replacement for enhanced orchestration endpoints**
3. **Connecting via MCP protocol for external tools**

### Example Integration

```python
# In your FastAPI route
from mcp_server.enhanced_agent import EnhancedMCPServer

server = EnhancedMCPServer()
await server.initialize()

result = await server.call_tool("process_query_fast", {
    "query": user_query,
    "translator_model": "phi3:mini"
})
```

## Development and Testing

### Running Tests

```bash
cd backend
source venv/bin/activate
python test_enhanced_mcp.py
```

### Development Mode

For development, you can run the server with additional logging:

```bash
cd backend
source venv/bin/activate
LOG_LEVEL=DEBUG python run_enhanced_mcp_server.py
```

### Custom Tools

You can extend the server with custom tools:

```python
class CustomEnhancedMCPServer(EnhancedMCPServer):
    async def list_tools(self):
        tools = await super().list_tools()
        tools.append({
            "name": "custom_tool",
            "description": "Custom processing tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                }
            }
        })
        return tools
    
    async def call_tool(self, name, arguments):
        if name == "custom_tool":
            return await self.custom_processing(arguments["input"])
        return await super().call_tool(name, arguments)
```

## Troubleshooting

### Common Issues

1. **Server Won't Start**
   - Check Python version (3.9+ required)
   - Verify all dependencies installed
   - Check MongoDB connectivity

2. **Queries Fail**
   - Ensure Ollama service is running
   - Verify required models are pulled
   - Check network connectivity

3. **Performance Issues**
   - Use faster pipeline strategies
   - Reduce concurrent processing
   - Optimize model selection

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

### Planned Features

1. **WebSocket Support**: Real-time streaming responses
2. **Model Auto-Discovery**: Automatic model selection
3. **Caching Layer**: Response caching for common queries
4. **Metrics Dashboard**: Web-based monitoring interface
5. **Plugin System**: Extensible architecture for custom agents

### Contributing

To contribute to the Enhanced MCP Server:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Update documentation
6. Submit a pull request

## License

This project is licensed under the MIT License. See the LICENSE file for details. 