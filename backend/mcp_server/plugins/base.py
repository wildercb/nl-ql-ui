"""
Base Plugin Architecture

Provides the foundation for the MCP server plugin system with:
- Type-safe plugin interfaces
- Comprehensive metadata management
- Lifecycle management hooks
- Resource and tool abstraction
- Performance monitoring integration
"""

import abc
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class PluginCategory(Enum):
    """Categories for organizing plugins."""
    TRANSLATION = "translation"
    ANALYSIS = "analysis"
    DATA_QUERY = "data_query"
    ORCHESTRATION = "orchestration"
    MONITORING = "monitoring"
    UTILITY = "utility"
    INTEGRATION = "integration"
    EXPERIMENTAL = "experimental"


class PluginStatus(Enum):
    """Plugin lifecycle status."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class PluginMetadata:
    """Comprehensive metadata for plugins."""
    
    # Basic information
    name: str
    version: str
    description: str
    author: str
    
    # Categorization
    category: PluginCategory
    tags: Set[str] = field(default_factory=set)
    
    # Dependencies and compatibility
    required_mcp_version: str = "1.0.0"
    dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    
    # Capabilities
    provides_tools: List[str] = field(default_factory=list)
    provides_resources: List[str] = field(default_factory=list)
    provides_prompts: List[str] = field(default_factory=list)
    
    # Configuration
    config_schema: Optional[Dict[str, Any]] = None
    default_config: Dict[str, Any] = field(default_factory=dict)
    
    # Lifecycle
    auto_load: bool = True
    priority: int = 100  # Lower numbers load first
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    homepage: Optional[str] = None
    documentation: Optional[str] = None
    license: str = "MIT"


@dataclass 
class PluginStats:
    """Performance and usage statistics for plugins."""
    
    # Usage metrics
    tool_calls: int = 0
    resource_requests: int = 0
    prompt_generations: int = 0
    
    # Performance metrics
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    peak_memory_usage: int = 0
    
    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    
    # Lifecycle
    load_time: Optional[datetime] = None
    activation_count: int = 0
    last_used: Optional[datetime] = None


class BasePlugin(abc.ABC):
    """
    Abstract base class for all MCP server plugins.
    
    Provides a comprehensive framework for plugin development with:
    - Standardized lifecycle management
    - Resource and tool abstraction
    - Configuration management
    - Performance monitoring
    - Error handling and recovery
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the plugin with optional configuration.
        
        Args:
            config: Plugin-specific configuration dictionary
        """
        self._config = config or {}
        self._status = PluginStatus.UNLOADED
        self._stats = PluginStats()
        self._logger = logging.getLogger(f"plugin.{self.get_metadata().name}")
        
        # Validate configuration against schema
        self._validate_config()
    
    @classmethod
    @abc.abstractmethod
    def get_metadata(cls) -> PluginMetadata:
        """
        Get comprehensive metadata for this plugin.
        
        This method must be implemented by all plugins to provide
        essential information about capabilities and requirements.
        
        Returns:
            PluginMetadata object with complete plugin information
        """
        pass
    
    @abc.abstractmethod
    async def load(self) -> bool:
        """
        Load and initialize the plugin.
        
        This method is called when the plugin is first loaded into
        the system. Use this for one-time initialization tasks.
        
        Returns:
            True if loading was successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    async def activate(self) -> bool:
        """
        Activate the plugin for use.
        
        This method is called to make the plugin active and ready
        to handle requests. It may be called multiple times.
        
        Returns:
            True if activation was successful, False otherwise
        """
        pass
    
    async def deactivate(self) -> bool:
        """
        Deactivate the plugin temporarily.
        
        This method is called to temporarily disable the plugin
        while keeping it loaded in memory.
        
        Returns:
            True if deactivation was successful, False otherwise
        """
        try:
            self._status = PluginStatus.LOADED
            self._logger.info(f"Plugin {self.get_metadata().name} deactivated")
            return True
        except Exception as e:
            self._logger.error(f"Error deactivating plugin: {e}")
            return False
    
    async def unload(self) -> bool:
        """
        Unload the plugin completely.
        
        This method is called when the plugin needs to be removed
        from the system. Perform cleanup here.
        
        Returns:
            True if unloading was successful, False otherwise
        """
        try:
            await self.deactivate()
            self._status = PluginStatus.UNLOADED
            self._logger.info(f"Plugin {self.get_metadata().name} unloaded")
            return True
        except Exception as e:
            self._logger.error(f"Error unloading plugin: {e}")
            return False
    
    # Tool interface methods
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of tools provided by this plugin.
        
        Returns:
            List of tool definitions with metadata
        """
        return []
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool provided by this plugin.
        
        Args:
            name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            NotImplementedError: If the tool is not supported
        """
        self._stats.tool_calls += 1
        self._stats.last_used = datetime.utcnow()
        
        raise NotImplementedError(f"Tool '{name}' not implemented by plugin {self.get_metadata().name}")
    
    # Resource interface methods
    
    def get_available_resources(self) -> List[Dict[str, Any]]:
        """
        Get list of resources provided by this plugin.
        
        Returns:
            List of resource definitions with metadata
        """
        return []
    
    async def get_resource(self, uri: str, **kwargs) -> Dict[str, Any]:
        """
        Retrieve a resource provided by this plugin.
        
        Args:
            uri: Resource URI
            **kwargs: Additional resource parameters
            
        Returns:
            Resource content and metadata
            
        Raises:
            NotImplementedError: If the resource is not supported
        """
        self._stats.resource_requests += 1
        self._stats.last_used = datetime.utcnow()
        
        raise NotImplementedError(f"Resource '{uri}' not implemented by plugin {self.get_metadata().name}")
    
    # Prompt interface methods
    
    def get_available_prompts(self) -> List[Dict[str, Any]]:
        """
        Get list of prompts provided by this plugin.
        
        Returns:
            List of prompt definitions with metadata
        """
        return []
    
    async def get_prompt(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a prompt provided by this plugin.
        
        Args:
            name: Name of the prompt to generate
            arguments: Prompt arguments
            
        Returns:
            Generated prompt content and metadata
            
        Raises:
            NotImplementedError: If the prompt is not supported
        """
        self._stats.prompt_generations += 1
        self._stats.last_used = datetime.utcnow()
        
        raise NotImplementedError(f"Prompt '{name}' not implemented by plugin {self.get_metadata().name}")
    
    # Configuration management
    
    def get_config(self, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value(s).
        
        Args:
            key: Configuration key (None for all config)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if key is None:
            return self._config.copy()
        return self._config.get(key, default)
    
    def update_config(self, config: Dict[str, Any]) -> bool:
        """
        Update plugin configuration.
        
        Args:
            config: New configuration values
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Merge with existing config
            self._config.update(config)
            
            # Validate the updated configuration
            self._validate_config()
            
            self._logger.info(f"Configuration updated for plugin {self.get_metadata().name}")
            return True
            
        except Exception as e:
            self._logger.error(f"Configuration update failed: {e}")
            return False
    
    def _validate_config(self):
        """Validate configuration against schema if available."""
        metadata = self.get_metadata()
        if metadata.config_schema:
            # Basic validation - could be enhanced with jsonschema
            schema = metadata.config_schema
            for key, spec in schema.items():
                if spec.get('required', False) and key not in self._config:
                    raise ValueError(f"Required configuration key '{key}' missing")
    
    # Status and monitoring
    
    def get_status(self) -> PluginStatus:
        """Get current plugin status."""
        return self._status
    
    def get_stats(self) -> PluginStats:
        """Get comprehensive plugin statistics."""
        return self._stats
    
    def reset_stats(self):
        """Reset plugin statistics."""
        self._stats = PluginStats()
        self._logger.info(f"Statistics reset for plugin {self.get_metadata().name}")
    
    # Health and diagnostics
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check for this plugin.
        
        Returns:
            Health status and diagnostic information
        """
        try:
            # Basic health check
            is_healthy = (
                self._status in [PluginStatus.LOADED, PluginStatus.ACTIVE] and
                self._stats.error_count < 10  # Arbitrary threshold
            )
            
            return {
                'healthy': is_healthy,
                'status': self._status.value,
                'last_used': self._stats.last_used.isoformat() if self._stats.last_used else None,
                'error_count': self._stats.error_count,
                'last_error': self._stats.last_error
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'status': 'error'
            }
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostic information.
        
        Returns:
            Detailed diagnostic data for debugging
        """
        metadata = self.get_metadata()
        
        return {
            'metadata': {
                'name': metadata.name,
                'version': metadata.version,
                'category': metadata.category.value,
                'author': metadata.author
            },
            'status': self._status.value,
            'config': self._config,
            'stats': {
                'tool_calls': self._stats.tool_calls,
                'resource_requests': self._stats.resource_requests,
                'prompt_generations': self._stats.prompt_generations,
                'error_count': self._stats.error_count,
                'total_execution_time': self._stats.total_execution_time,
                'last_used': self._stats.last_used.isoformat() if self._stats.last_used else None
            },
            'capabilities': {
                'tools': len(self.get_available_tools()),
                'resources': len(self.get_available_resources()),
                'prompts': len(self.get_available_prompts())
            }
        }
    
    # Error handling
    
    def _record_error(self, error: Exception):
        """Record an error for statistics and monitoring."""
        self._stats.error_count += 1
        self._stats.last_error = str(error)
        self._stats.last_error_time = datetime.utcnow()
        self._logger.error(f"Plugin error: {error}")
    
    # Utility methods
    
    def __str__(self) -> str:
        """String representation of the plugin."""
        metadata = self.get_metadata()
        return f"Plugin({metadata.name} v{metadata.version})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        metadata = self.get_metadata()
        return (f"Plugin(name={metadata.name}, version={metadata.version}, "
                f"category={metadata.category.value}, status={self._status.value})")


def plugin(
    name: str,
    version: str = "1.0.0",
    description: str = "",
    author: str = "Unknown",
    category: PluginCategory = PluginCategory.UTILITY,
    **kwargs
):
    """
    Decorator for registering plugins with metadata.
    
    This decorator simplifies plugin registration by automatically
    creating metadata and marking classes as plugins.
    
    Args:
        name: Plugin name
        version: Plugin version
        description: Plugin description
        author: Plugin author
        category: Plugin category
        **kwargs: Additional metadata parameters
        
    Returns:
        Decorated plugin class with metadata
    """
    def decorator(cls):
        # Create metadata from decorator parameters
        metadata = PluginMetadata(
            name=name,
            version=version,
            description=description,
            author=author,
            category=category,
            **kwargs
        )
        
        # Store metadata on the class
        cls._plugin_metadata = metadata
        
        # Override get_metadata to return stored metadata
        cls.get_metadata = classmethod(lambda cls: metadata)
        
        return cls
    
    return decorator 