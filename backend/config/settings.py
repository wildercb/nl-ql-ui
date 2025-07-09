"""
Settings Configuration - Compatibility layer for unified architecture.

This module provides backward compatibility for settings that are now
managed by the unified configuration system.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DatabaseSettings:
    """Database configuration settings."""
    
    url: str = "mongodb://localhost:27017"
    database: str = "mppw_main"
    min_connections: int = 1
    max_connections: int = 10
    
    def __post_init__(self):
        self.url = os.getenv("MONGODB_URL", self.url)
        self.database = os.getenv("MONGODB_DATABASE", self.database)
        self.min_connections = int(os.getenv("MONGODB_MIN_CONNECTIONS", self.min_connections))
        self.max_connections = int(os.getenv("MONGODB_MAX_CONNECTIONS", self.max_connections))


@dataclass
class OllamaSettings:
    """Ollama configuration settings."""
    
    base_url: str = "http://localhost:11434"
    default_model: str = "phi3:mini"  # Note: Models should be selected via UI
    timeout: int = 300
    
    def __post_init__(self):
        self.base_url = os.getenv("OLLAMA_BASE_URL", self.base_url)
        self.default_model = os.getenv("OLLAMA_DEFAULT_MODEL", self.default_model)
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", self.timeout))


@dataclass
class MCPSettings:
    """MCP server configuration settings."""
    
    name: str = "MPPW MCP Server"
    version: str = "2.0.0"
    port: int = 8001
    
    def __post_init__(self):
        self.port = int(os.getenv("MCP_PORT", self.port))


@dataclass
class LLMTrackingSettings:
    """LLM tracking configuration settings."""
    
    enabled: bool = True
    store_prompts: bool = True
    store_responses: bool = True
    batch_size: int = 100
    
    def __post_init__(self):
        self.enabled = os.getenv("LLM_TRACKING_ENABLED", "true").lower() == "true"
        self.store_prompts = os.getenv("LLM_TRACKING_STORE_PROMPTS", "true").lower() == "true"
        self.store_responses = os.getenv("LLM_TRACKING_STORE_RESPONSES", "true").lower() == "true"
        self.batch_size = int(os.getenv("LLM_TRACKING_BATCH_SIZE", self.batch_size))


# ---------------------------------------------------------------------------
# Rate Limiting Settings
# ---------------------------------------------------------------------------


@dataclass
class RateLimitSettings:
    """API rate limiting configuration."""

    requests_per_minute: int = 60  # default 60 RPM per IP

    def __post_init__(self):
        self.requests_per_minute = int(os.getenv("API_RATE_LIMIT_RPM", self.requests_per_minute))


@dataclass
class Settings:
    """Main application settings."""
    
    # Environment
    environment: str = "development"
    debug: bool = True
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Logging
    log_level: str = "INFO"
    
    # Component settings
    database: Optional[DatabaseSettings] = None
    ollama: Optional[OllamaSettings] = None
    mcp: Optional[MCPSettings] = None
    llm_tracking: Optional[LLMTrackingSettings] = None
    rate_limit: Optional[RateLimitSettings] = None
    
    def __post_init__(self):
        self.environment = os.getenv("ENVIRONMENT", self.environment)
        self.debug = os.getenv("DEBUG", "true").lower() == "true"
        self.api_host = os.getenv("API_HOST", self.api_host)
        self.api_port = int(os.getenv("API_PORT", self.api_port))
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        
        # Initialize component settings
        if self.database is None:
            self.database = DatabaseSettings()
        if self.ollama is None:
            self.ollama = OllamaSettings()
        if self.mcp is None:
            self.mcp = MCPSettings()
        if self.llm_tracking is None:
            self.llm_tracking = LLMTrackingSettings()
        if self.rate_limit is None:
            self.rate_limit = RateLimitSettings()
    
    def dict(self) -> Dict[str, Any]:
        """Get settings as dictionary for compatibility."""
        # These should never be None after __post_init__
        assert self.database is not None
        assert self.ollama is not None
        assert self.mcp is not None
        assert self.llm_tracking is not None
        assert self.rate_limit is not None
        
        return {
            "environment": self.environment,
            "debug": self.debug,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "log_level": self.log_level,
            "database": {
                "url": self.database.url,
                "database": self.database.database,
                "min_connections": self.database.min_connections,
                "max_connections": self.database.max_connections
            },
            "ollama": {
                "base_url": self.ollama.base_url,
                "default_model": self.ollama.default_model,
                "timeout": self.ollama.timeout
            },
            "mcp": {
                "name": self.mcp.name,
                "version": self.mcp.version,
                "port": self.mcp.port
            },
            "llm_tracking": {
                "enabled": self.llm_tracking.enabled,
                "store_prompts": self.llm_tracking.store_prompts,
                "store_responses": self.llm_tracking.store_responses,
                "batch_size": self.llm_tracking.batch_size
            },
            "rate_limit": {
                "requests_per_minute": self.rate_limit.requests_per_minute
            }
        }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Legacy compatibility aliases
def get_database_settings() -> DatabaseSettings:
    """Get database settings."""
    settings = get_settings()
    assert settings.database is not None
    return settings.database


def get_ollama_settings() -> OllamaSettings:
    """Get Ollama settings."""
    settings = get_settings()
    assert settings.ollama is not None
    return settings.ollama


def get_mcp_settings() -> MCPSettings:
    """Get MCP settings."""
    settings = get_settings()
    assert settings.mcp is not None
    return settings.mcp


# Environment helpers
def is_development() -> bool:
    """Check if running in development mode."""
    return get_settings().environment.lower() == "development"


def is_production() -> bool:
    """Check if running in production mode."""
    return get_settings().environment.lower() == "production" 