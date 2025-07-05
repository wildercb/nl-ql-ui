# Model Management Guide

This guide explains how to use the new model management features in the MPPW-MCP application.

## Overview

The application now supports multiple AI models and provides a user-friendly interface for:
- Selecting different models
- Checking model availability
- Downloading models automatically
- Using models in the pipeline

## Supported Models

The following models are supported:

### Phi3 Models (Microsoft)
- **phi3:mini** - Fast, lightweight model (recommended for quick testing)
- **phi3:small** - Balanced performance and speed
- **phi3:medium** - Higher quality, slower inference

### Gemma3 Models (Google)
- **gemma3:2b** - Fast, efficient model
- **gemma3:7b** - High quality, balanced performance
- **gemma3:9b** - Maximum quality, slower inference

### Llama4 Models (Meta)
- **llama4:3b** - Fast, lightweight model
- **llama4:7b** - Balanced performance and quality
- **llama4:14b** - Maximum quality, requires more resources

## Using the Model Selector

### 1. Model Configuration Panel

The model configuration panel is located at the top of the main interface and includes:

- **Model Selector**: Dropdown to choose from available models
- **Model Status**: Visual indicator showing if the model is ready
- **Actions**: Buttons to check status and download models

### 2. Model Status Indicators

- 🟢 **Green**: Model is ready to use
- 🟡 **Yellow**: Model is currently downloading
- 🔴 **Red**: Model is not available (needs to be downloaded)

### 3. Model Actions

- **Check**: Verify the current status of the selected model
- **Download**: Download the selected model (only shown when model is not ready)

## Automatic Model Management

### Model Status Checking

The application automatically checks model status:
- When the page loads
- When you change the selected model
- After downloading a model

### Download Progress

When downloading a model, you'll see:
- Real-time progress bar
- Percentage completion
- Status updates

The download process uses streaming updates to provide accurate progress information.

## Using Models in Pipelines

### Model Selection in Pipelines

When you run a pipeline (Translate, Multi-Agent, or Enhanced Agents), the selected model is used for:
- **Pre-processing/Rewriting**: Improves the natural language query
- **Translation**: Converts natural language to GraphQL
- **Review**: Validates and optimizes the GraphQL query

### Model-Specific Behavior

Different models may produce slightly different results:
- **Smaller models** (phi3:mini, gemma3:2b): Faster responses, good for testing
- **Larger models** (llama4:14b, gemma3:9b): Higher quality results, slower responses

## API Endpoints

The model management functionality is exposed through these API endpoints:

### List Available Models
```http
GET /api/models/
```

### Check Model Status
```http
GET /api/models/{model_name}
```

### Download Model
```http
POST /api/models/{model_name}/pull
```

### Download Model with Progress
```http
POST /api/models/{model_name}/pull/stream
```

### Check Service Health
```http
GET /api/models/health/status
```

## Best Practices

### Model Selection

1. **Start with phi3:mini** for quick testing and development
2. **Use gemma3:7b** for production workloads requiring good quality
3. **Use llama4:14b** for maximum quality when speed isn't critical

### Resource Management

- Larger models require more RAM and processing power
- Consider your hardware capabilities when selecting models
- Monitor system resources during model downloads

### Performance Optimization

- Keep frequently used models downloaded
- Use smaller models for batch processing
- Use larger models for critical queries requiring high accuracy

## Troubleshooting

### Model Download Issues

If a model fails to download:

1. Check your internet connection
2. Verify you have sufficient disk space
3. Ensure Ollama is running and accessible
4. Try downloading a smaller model first

### Model Not Available

If a model shows as "not available":

1. Click the "Check" button to refresh status
2. Try downloading the model
3. Verify the model name is correct
4. Check if Ollama supports the model

### Performance Issues

If the application is slow:

1. Try using a smaller model
2. Check system resource usage
3. Ensure you have sufficient RAM
4. Consider upgrading your hardware

## Testing

You can test the model management functionality using the provided test script:

```bash
cd scripts
python test_model_management.py
```

This script will:
- Test all model management endpoints
- Verify model status checking
- Test model downloading
- Validate pipeline integration

## Configuration

Model settings can be configured in `backend/config/settings.py`:

```python
class OllamaSettings(BaseSettings):
    base_url: str = Field("http://localhost:11434")
    default_model: str = Field("phi3:mini")
    timeout: int = Field(120)
    max_tokens: int = Field(4096)
    temperature: float = Field(0.7)
```

## Future Enhancements

Planned improvements include:
- Model performance metrics
- Automatic model optimization
- Model comparison tools
- Batch model management
- Cloud model support 