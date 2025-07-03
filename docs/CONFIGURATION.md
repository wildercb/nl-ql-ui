# Configuration Guide

This guide covers all configuration options for the MPPW MCP system.

## Environment Variables

Create a `.env` file in the project root with these variables:

### Core Database & API
```bash
# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/mppw_mcp
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=true
SECRET_KEY=your-secret-key-here

# CORS Settings
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]
CORS_ALLOW_CREDENTIALS=true
RATE_LIMIT_REQUESTS_PER_MINUTE=60
```

### Model Providers

#### Ollama (Default)
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=llama2
OLLAMA_TIMEOUT=30
OLLAMA_MAX_RETRIES=3
```

#### OpenAI (Optional)
```bash
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_DEFAULT_MODEL=gpt-3.5-turbo
OPENAI_MAX_TOKENS=4096
```

#### Anthropic (Optional)
```bash
ANTHROPIC_API_KEY=your-anthropic-api-key-here
ANTHROPIC_DEFAULT_MODEL=claude-3-sonnet
ANTHROPIC_MAX_TOKENS=4096
```

#### HuggingFace (Optional)
```bash
HUGGINGFACE_API_TOKEN=your-huggingface-token-here
HUGGINGFACE_DEFAULT_MODEL=microsoft/DialoGPT-large
```

### MCP Server
```bash
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=3001
MCP_ENABLE_TOOLS=true
```

### Logging & Security
```bash
LOG_LEVEL=INFO
LOG_FORMAT=json
ENABLE_REQUEST_LOGGING=true
ENABLE_RATE_LIMITING=true
ENABLE_CORS=true
ENABLE_SECURITY_HEADERS=true
```

### Feature Flags
```bash
ENABLE_BATCH_TRANSLATION=true
ENABLE_WEBSOCKET_TRANSLATION=true
ENABLE_SCHEMA_ANALYZER=true
ENABLE_QUERY_OPTIMIZER=true
```

## Provider Configuration

### Adding Custom Providers
```python
# config/custom_providers.py
CUSTOM_PROVIDERS = {
    "azure_openai": {
        "endpoint": "https://your-resource.openai.azure.com/",
        "api_key": "your-azure-key",
        "api_version": "2023-05-15"
    },
    "cohere": {
        "api_key": "your-cohere-key",
        "model": "command-xlarge"
    }
}
```

### Model Preferences
```bash
# Model selection by use case
CUSTOM_TRANSLATION_MODEL_FAST=gpt-3.5-turbo
CUSTOM_TRANSLATION_MODEL_ACCURATE=gpt-4
CUSTOM_TRANSLATION_MODEL_LOCAL=llama2
CUSTOM_TRANSLATION_MODEL_SPECIALIZED=codellama
```

## Prompt Configuration

### Custom Domain Prompts
```python
# config/prompts.py
DOMAIN_PROMPTS = {
    "ecommerce": {
        "system": "You are an e-commerce GraphQL expert...",
        "rules": [
            "Always include product availability",
            "Use pagination for lists",
            "Include price information"
        ]
    },
    "social": {
        "system": "You are a social media platform expert...",
        "rules": [
            "Include privacy checks",
            "Use fragments for user data",
            "Consider content moderation"
        ]
    }
}
```

## Runtime Configuration

### Dynamic Model Switching
```python
# In your application
from backend.services.translation_service import TranslationService

# Switch providers at runtime
service = TranslationService()
await service.switch_provider("openai", api_key="new-key")

# Switch models
await service.switch_model("gpt-4")
```

### Feature Flag Management
```python
# config/feature_flags.py
class FeatureFlags:
    ENABLE_ADVANCED_VALIDATION = True
    ENABLE_COST_TRACKING = True
    ENABLE_A_B_TESTING = False
    ENABLE_REAL_TIME_ANALYTICS = True
```

## Performance Tuning

### Cache Configuration
```bash
ENABLE_REDIS_CACHE=true
CACHE_TTL_SECONDS=3600
ENABLE_QUERY_CACHE=true
MAX_CACHE_SIZE_MB=512
```

### Database Optimization
```bash
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600
```

### Model Provider Limits
```python
PROVIDER_LIMITS = {
    "openai": {
        "requests_per_minute": 60,
        "tokens_per_minute": 90000,
        "max_concurrent": 5
    },
    "anthropic": {
        "requests_per_minute": 50,
        "tokens_per_minute": 40000,
        "max_concurrent": 3
    }
}
```

## Security Configuration

### API Security
```bash
JWT_SECRET_KEY=your-jwt-secret
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
BCRYPT_ROUNDS=12
```

### Rate Limiting
```python
RATE_LIMITS = {
    "translation": "10/minute",
    "validation": "20/minute", 
    "models": "5/minute",
    "health": "100/minute"
}
```

## Development vs Production

### Development
```bash
ENV=development
API_DEBUG=true
LOG_LEVEL=DEBUG
ENABLE_HOT_RELOAD=true
DEV_ENABLE_DEBUG_TOOLBAR=true
```

### Production
```bash
ENV=production
API_DEBUG=false
LOG_LEVEL=INFO
ENABLE_HOT_RELOAD=false
SECURE_SSL_REDIRECT=true
SECURE_HSTS_SECONDS=31536000
```

## Docker Configuration

### Environment Variables in Docker
```yaml
# docker-compose.yml
services:
  backend:
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/mppw_mcp
      - REDIS_URL=redis://redis:6379/0
      - OLLAMA_BASE_URL=http://ollama:11434
```

### Health Check Configuration
```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## Monitoring Configuration

### Metrics Collection
```bash
ENABLE_METRICS=true
METRICS_PORT=9090
PROMETHEUS_ENDPOINT=/metrics
```

### Logging Configuration
```python
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "json": {
            "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
}
```

## Configuration Validation

### Startup Checks
```python
# backend/config/validation.py
def validate_configuration():
    """Validate all configuration settings on startup."""
    checks = [
        check_database_connection(),
        check_redis_connection(),
        check_ollama_connection(),
        validate_environment_variables(),
        check_model_availability()
    ]
    
    failed_checks = [check for check in checks if not check.passed]
    if failed_checks:
        raise ConfigurationError(f"Failed checks: {failed_checks}")
```

### Configuration Schema
```python
from pydantic import BaseSettings, validator

class Settings(BaseSettings):
    database_url: str
    redis_url: str
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    @validator('database_url')
    def validate_database_url(cls, v):
        if not v.startswith(('postgresql://', 'sqlite:///')):
            raise ValueError('Invalid database URL')
        return v
    
    @validator('api_port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError('Port must be between 1 and 65535')
        return v
```

This configuration system provides flexibility while maintaining security and performance. Adjust settings based on your specific deployment needs. 