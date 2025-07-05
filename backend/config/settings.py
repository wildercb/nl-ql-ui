"""Application configuration settings."""

from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache


class DatabaseSettings(BaseSettings):
    """MongoDB database configuration."""
    
    url: str = Field("mongodb://localhost:27017")
    database: str = Field("mppw_mcp")
    min_connections: int = Field(10)
    max_connections: int = Field(100)
    
    model_config = SettingsConfigDict(env_prefix="MONGODB_")


class DataDatabaseSettings(BaseSettings):
    """Secondary MongoDB for content data."""
    
    url: str = Field("mongodb://mongo_data:27018")
    database: str = Field("mppw_content")
    min_connections: int = Field(10)
    max_connections: int = Field(100)
    
    model_config = SettingsConfigDict(env_prefix="DATA_MONGODB_")


class RedisSettings(BaseSettings):
    """Redis configuration."""
    
    url: str = Field("redis://localhost:6379")
    db: int = Field(0)
    max_connections: int = Field(100)
    
    model_config = SettingsConfigDict(env_prefix="REDIS_")


class OllamaSettings(BaseSettings):
    """Ollama AI model configuration."""
    
    base_url: str = Field("http://localhost:11434")
    default_model: str = Field("phi3:mini")
    timeout: int = Field(120)
    max_tokens: int = Field(4096)
    temperature: float = Field(0.7)
    
    model_config = SettingsConfigDict(env_prefix="OLLAMA_")


class APISettings(BaseSettings):
    """API server configuration."""
    
    host: str = Field("0.0.0.0")
    port: int = Field(8000)
    workers: int = Field(1)
    debug: bool = Field(False)
    title: str = Field("MPPW NLQ Translator API")
    description: str = Field("Natural Language to GraphQL Query Translation Service")
    version: str = Field("1.0.0")
    
    model_config = SettingsConfigDict(env_prefix="API_")


class MCPSettings(BaseSettings):
    """MCP Server configuration."""
    
    host: str = Field("0.0.0.0")
    port: int = Field(8001)
    name: str = Field("mppw-nlq-translator")
    description: str = Field("Natural Language to GraphQL Query Translation MCP Server")
    version: str = Field("1.0.0")
    
    model_config = SettingsConfigDict(env_prefix="MCP_SERVER_")


class SecuritySettings(BaseSettings):
    """Security and authentication configuration."""
    
    # NOTE: Provide a sensible default so that the application can boot in local/dev
    # environments.  The value can (and **should**) be overridden in production by
    # setting an environment variable named `SECRET_KEY` or `SECURITY_SECRET_KEY`.
    # By specifying both `alias="SECRET_KEY"` *and* an `env_prefix` (via
    # `SettingsConfigDict`) we make it possible to reference the value with either
    # naming scheme:
    #   * SECRET_KEY
    #   * SECURITY_SECRET_KEY
    # The latter follows the standard nested-settings convention, while the former
    # matches the variable already present in docker-compose.yml.
    secret_key: str = Field("change_me_please", alias="SECRET_KEY")
    access_token_expire_minutes: int = Field(30)
    refresh_token_expire_days: int = Field(30)
    algorithm: str = Field("HS256")

    # Allow both prefixed and un-prefixed env vars
    model_config = SettingsConfigDict(env_prefix="SECURITY_")


class CORSSettings(BaseSettings):
    """CORS configuration."""
    
    origins: List[str] = Field(["http://localhost:3000", "http://127.0.0.1:3000"])
    allow_credentials: bool = Field(True)
    allow_methods: List[str] = Field(["*"])
    allow_headers: List[str] = Field(["*"])
    
    model_config = SettingsConfigDict(env_prefix="CORS_")


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    level: str = Field("INFO")
    format: str = Field("json")
    file_path: Optional[str] = Field(None)
    
    model_config = SettingsConfigDict(env_prefix="LOG_")


class GraphQLSettings(BaseSettings):
    """GraphQL configuration."""
    
    introspection: bool = Field(True)
    playground: bool = Field(True)
    
    model_config = SettingsConfigDict(env_prefix="GRAPHQL_")


class CelerySettings(BaseSettings):
    """Celery background task configuration."""
    
    broker_url: str = Field("redis://localhost:6379/1")
    result_backend: str = Field("redis://localhost:6379/2")
    task_serializer: str = Field("json")
    result_serializer: str = Field("json")
    
    model_config = SettingsConfigDict(env_prefix="CELERY_")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration."""
    
    sentry_dsn: Optional[str] = Field(None)
    prometheus_metrics: bool = Field(True)
    
    model_config = SettingsConfigDict(env_prefix="MONITORING_")


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration."""
    
    requests_per_minute: int = Field(60)
    burst: int = Field(10)
    
    model_config = SettingsConfigDict(env_prefix="RATE_LIMIT_")


class QuerySettings(BaseSettings):
    """Query processing configuration."""
    
    max_query_length: int = Field(1000)
    max_results_per_page: int = Field(100)
    cache_ttl: int = Field(3600)
    
    model_config = SettingsConfigDict(env_prefix="QUERY_")


class LLMTrackingSettings(BaseSettings):
    """LLM interaction tracking configuration."""
    
    enabled: bool = Field(True)
    store_prompts: bool = Field(True)
    store_responses: bool = Field(True)
    retention_days: int = Field(90)
    max_prompt_length: int = Field(10000)
    max_response_length: int = Field(10000)
    
    model_config = SettingsConfigDict(env_prefix="LLM_TRACKING_")


class Neo4jSettings(BaseSettings):
    """Neo4j graph database for content via Cypher."""
    uri: str = Field("bolt://neo4j:7687")
    user: str = Field("neo4j")
    password: str = Field("password")

    model_config = SettingsConfigDict(env_prefix="NEO4J_")


class Settings(BaseSettings):
    """Main application settings."""
    
    env: str = Field("development")
    
    # Sub-configurations
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    data_database: DataDatabaseSettings = Field(default_factory=DataDatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    api: APISettings = Field(default_factory=APISettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    cors: CORSSettings = Field(default_factory=CORSSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    graphql: GraphQLSettings = Field(default_factory=GraphQLSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    query: QuerySettings = Field(default_factory=QuerySettings)
    llm_tracking: LLMTrackingSettings = Field(default_factory=LLMTrackingSettings)
    neo4j: Neo4jSettings = Field(default_factory=Neo4jSettings)
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.env.lower() in ("development", "dev", "local")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.env.lower() in ("production", "prod")
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.env.lower() in ("testing", "test")
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="allow")


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings() 