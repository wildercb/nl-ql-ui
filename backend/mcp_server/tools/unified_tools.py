"""
Unified MCP Tools System for MPPW-MCP

This module provides a modern, extensible MCP tools framework that:
1. Integrates with the unified configuration system
2. Uses consistent patterns across all tools
3. Provides automatic tool registration and discovery
4. Supports easy addition of new tools and categories
5. Includes comprehensive error handling and monitoring
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union, Callable, Type, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import time
import json
from pathlib import Path
import inspect

from config.unified_config import get_unified_config, ToolCategory, ToolConfig
from agents.unified_agents import AgentContext, create_agent, execute_pipeline
from services.llm_factory import resolve_llm
from services.validation_service import ValidationService
from services.translation_service import TranslationService
from services.ollama_service import OllamaService

logger = logging.getLogger(__name__)


# =============================================================================
# Core Tool Framework
# =============================================================================

@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_mcp_response(self) -> List[Dict[str, Any]]:
        """Convert to MCP server response format."""
        if self.success:
            return [{"type": "text", "text": json.dumps(self.data, indent=2)}]
        else:
            return [{"type": "text", "text": f"Error: {self.error}"}]


class BaseTool(ABC):
    """
    Base class for all MCP tools in the unified system.
    
    Provides consistent interface and common functionality:
    - Configuration integration
    - Parameter validation
    - Error handling
    - Performance tracking
    - Automatic registration
    """
    
    def __init__(self):
        self.config_manager = get_unified_config()
        self.tool_config = self._get_tool_config()
        
        if not self.tool_config:
            logger.warning(f"No configuration found for tool: {self.get_name()}")
            self.tool_config = self._create_default_config()
        
        logger.debug(f"Initialized tool: {self.get_name()} ({self.get_category().value})")
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the tool name."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get the tool description."""
        pass
    
    @abstractmethod
    def get_category(self) -> ToolCategory:
        """Get the tool category."""
        pass
    
    @abstractmethod
    def get_input_schema(self) -> Dict[str, Any]:
        """Get the input schema for the tool."""
        pass
    
    @abstractmethod
    async def _execute_impl(self, **kwargs) -> Any:
        """Execute the tool logic. Must be implemented by subclasses."""
        pass
    
    def _get_tool_config(self) -> Optional[ToolConfig]:
        """Get tool configuration from the unified config."""
        return self.config_manager.get_tool(self.get_name())
    
    def _create_default_config(self) -> ToolConfig:
        """Create a default configuration for this tool."""
        return ToolConfig(
            name=self.get_name(),
            category=self.get_category(),
            description=self.get_description(),
            function_name=self.get_name(),
            module_path=f"{self.__module__}.{self.__class__.__name__}",
            input_schema=self.get_input_schema()
        )
    
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with parameter validation and error handling."""
        start_time = time.time()
        
        try:
            # Validate inputs
            validation_error = self._validate_inputs(kwargs)
            if validation_error:
                return ToolResult(
                    success=False,
                    error=validation_error,
                    processing_time=time.time() - start_time
                )
            
            # Execute with timeout if configured
            if self.tool_config.timeout:
                result = await asyncio.wait_for(
                    self._execute_impl(**kwargs),
                    timeout=self.tool_config.timeout
                )
            else:
                result = await self._execute_impl(**kwargs)
            
            processing_time = time.time() - start_time
            
            return ToolResult(
                success=True,
                data=result,
                processing_time=processing_time,
                metadata={
                    "tool_name": self.get_name(),
                    "tool_category": self.get_category().value,
                    "execution_time": processing_time
                }
            )
            
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                error=f"Tool execution timed out after {self.tool_config.timeout}s",
                processing_time=time.time() - start_time
            )
        except Exception as e:
            logger.error(f"Tool {self.get_name()} failed: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
                metadata={"exception_type": type(e).__name__}
            )
    
    def _validate_inputs(self, kwargs: Dict[str, Any]) -> Optional[str]:
        """Validate input parameters against the schema."""
        schema = self.get_input_schema()
        required_fields = schema.get("required", [])
        
        # Check required fields
        for field in required_fields:
            if field not in kwargs:
                return f"Missing required parameter: {field}"
        
        # Check field types if specified
        properties = schema.get("properties", {})
        for field, value in kwargs.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type and not self._check_type(value, expected_type):
                    return f"Invalid type for {field}: expected {expected_type}"
        
        return None
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type:
            return isinstance(value, expected_python_type)
        
        return True  # Unknown type, allow it
    
    def to_mcp_definition(self) -> Dict[str, Any]:
        """Convert to MCP tool definition format."""
        return {
            "name": self.get_name(),
            "description": self.get_description(),
            "inputSchema": self.get_input_schema()
        }


# =============================================================================
# Translation Tools
# =============================================================================

class TranslateQueryTool(BaseTool):
    """Tool for translating natural language to GraphQL."""
    
    def __init__(self):
        super().__init__()
        self.translation_service = TranslationService()
    
    def get_name(self) -> str:
        return "translate_query"
    
    def get_description(self) -> str:
        return "Translate natural language queries to GraphQL using the unified agent system"
    
    def get_category(self) -> ToolCategory:
        return ToolCategory.TRANSLATION
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "natural_query": {"type": "string", "description": "The natural language query to translate"},
                "schema_context": {"type": "string", "description": "GraphQL schema context for better translation"},
                "domain": {"type": "string", "description": "Domain context (e.g., 'ecommerce', 'social')"},
                "model": {"type": "string", "description": "AI model to use for translation"},
                "strategy": {"type": "string", "description": "Translation strategy (fast, standard, comprehensive)"},
                "examples": {"type": "array", "description": "Example translations for context"}
            },
            "required": ["natural_query"]
        }
    
    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute translation using the unified agent system."""
        natural_query = kwargs["natural_query"]
        schema_context = kwargs.get("schema_context")
        domain = kwargs.get("domain")
        model = kwargs.get("model")
        strategy = kwargs.get("strategy", "standard")
        examples = kwargs.get("examples", [])
        
        # Create agent context
        ctx = AgentContext(
            original_query=natural_query,
            domain_context=domain,
            schema_context=schema_context,
            examples=examples
        )
        
        # Add model override if specified
        if model:
            ctx.model_overrides["translator"] = model
        
        # Execute translation pipeline
        if strategy == "fast":
            # Use only translator agent
            translator = create_agent("translator")
            if translator:
                result = await translator.execute(ctx)
                if result.success:
                    return result.output
                else:
                    raise Exception(result.error)
        else:
            # Use full pipeline
            pipeline_name = strategy if strategy in ["standard", "comprehensive"] else "standard"
            results = await execute_pipeline(pipeline_name, ctx)
            
            # Get translation result
            translator_result = results.get("translator")
            if translator_result and translator_result.success:
                return translator_result.output
            else:
                error = translator_result.error if translator_result else "Translation failed"
                raise Exception(error)
        
        raise Exception("No suitable translation agent available")


class ProcessQueryPipelineTool(BaseTool):
    """Tool for processing queries through configurable agent pipelines."""
    
    def get_name(self) -> str:
        return "process_query_pipeline"
    
    def get_description(self) -> str:
        return "Process a query through a configurable agent pipeline (rewrite → translate → review)"
    
    def get_category(self) -> ToolCategory:
        return ToolCategory.TRANSLATION
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The natural language query to process"},
                "pipeline": {"type": "string", "description": "Pipeline to use (fast, standard, comprehensive, adaptive)"},
                "schema_context": {"type": "string", "description": "GraphQL schema context"},
                "domain": {"type": "string", "description": "Domain context"},
                "model_overrides": {"type": "object", "description": "Model overrides for specific agents"},
                "user_id": {"type": "string", "description": "User identifier for tracking"}
            },
            "required": ["query"]
        }
    
    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute the full query processing pipeline."""
        query = kwargs["query"]
        pipeline = kwargs.get("pipeline", "standard")
        schema_context = kwargs.get("schema_context")
        domain = kwargs.get("domain")
        model_overrides = kwargs.get("model_overrides", {})
        user_id = kwargs.get("user_id")
        
        # Create agent context
        ctx = AgentContext(
            original_query=query,
            domain_context=domain,
            schema_context=schema_context,
            user_id=user_id,
            model_overrides=model_overrides
        )
        
        # Execute pipeline
        results = await execute_pipeline(pipeline, ctx)
        
        # Compile results
        response = {
            "original_query": query,
            "pipeline_used": pipeline,
            "execution_path": ctx.execution_path,
            "total_processing_time": ctx.total_processing_time,
            "results": {}
        }
        
        # Add individual agent results
        for agent_name, result in results.items():
            response["results"][agent_name] = {
                "success": result.success,
                "output": result.output,
                "processing_time": result.processing_time,
                "model_used": result.model_used,
                "error": result.error
            }
        
        # Add final outputs to top level for convenience
        if ctx.rewritten_query:
            response["rewritten_query"] = ctx.rewritten_query
        if ctx.graphql_query:
            response["graphql_query"] = ctx.graphql_query
        if ctx.review_result:
            response["review_result"] = ctx.review_result
        
        return response


# =============================================================================
# Validation Tools
# =============================================================================

class ValidateGraphQLTool(BaseTool):
    """Tool for validating GraphQL queries."""
    
    def __init__(self):
        super().__init__()
        self.validation_service = ValidationService()
    
    def get_name(self) -> str:
        return "validate_graphql"
    
    def get_description(self) -> str:
        return "Validate GraphQL query syntax, structure, and best practices"
    
    def get_category(self) -> ToolCategory:
        return ToolCategory.VALIDATION
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The GraphQL query to validate"},
                "schema": {"type": "string", "description": "GraphQL schema for validation"},
                "strict_mode": {"type": "boolean", "description": "Enable strict validation"},
                "check_performance": {"type": "boolean", "description": "Check for performance issues"},
                "check_security": {"type": "boolean", "description": "Check for security issues"}
            },
            "required": ["query"]
        }
    
    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Execute GraphQL validation."""
        query = kwargs["query"]
        schema = kwargs.get("schema")
        strict_mode = kwargs.get("strict_mode", False)
        check_performance = kwargs.get("check_performance", True)
        check_security = kwargs.get("check_security", True)
        
        # Basic validation
        validation_result = self.validation_service.validate_query(query)
        
        result = {
            "valid": validation_result.is_valid,
            "errors": validation_result.errors,
            "warnings": validation_result.warnings,
            "suggestions": validation_result.suggestions,
            "query_length": len(query)
        }
        
        # Schema validation if provided
        if schema:
            try:
                schema_validation = self.validation_service.validate_with_schema(query, schema)
                result["schema_validation"] = schema_validation
            except Exception as e:
                result["schema_validation"] = {"error": str(e)}
        
        # Performance analysis
        if check_performance:
            try:
                performance_analysis = self.validation_service.analyze_complexity(query)
                result["performance_analysis"] = performance_analysis
            except Exception as e:
                result["performance_analysis"] = {"error": str(e)}
        
        # Security analysis
        if check_security:
            security_issues = []
            query_lower = query.lower()
            
            if "password" in query_lower or "secret" in query_lower:
                security_issues.append("Query contains potentially sensitive field names")
            
            if query.count("{") > 10:
                security_issues.append("Query has high nesting depth - potential DoS risk")
            
            result["security_issues"] = security_issues
        
        return result


# =============================================================================
# Model Management Tools
# =============================================================================

class ListModelsTool(BaseTool):
    """Tool for listing available AI models."""
    
    def __init__(self):
        super().__init__()
        self.ollama_service = OllamaService()
    
    def get_name(self) -> str:
        return "list_models"
    
    def get_description(self) -> str:
        return "List all available AI models across all configured providers"
    
    def get_category(self) -> ToolCategory:
        return ToolCategory.MODEL
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "provider": {"type": "string", "description": "Filter by provider (ollama, groq, openrouter)"},
                "include_metrics": {"type": "boolean", "description": "Include performance metrics"}
            }
        }
    
    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """List available models from all providers."""
        provider_filter = kwargs.get("provider")
        include_metrics = kwargs.get("include_metrics", False)
        
        result = {
            "providers": {},
            "total_models": 0,
            "configured_models": {}
        }
        
        # Get configured models from unified config
        for model_name, model_config in self.config_manager.models.items():
            if provider_filter and model_config.provider.value != provider_filter:
                continue
            
            provider_name = model_config.provider.value
            if provider_name not in result["configured_models"]:
                result["configured_models"][provider_name] = []
            
            model_info = {
                "name": model_name,
                "size": model_config.size.value,
                "max_tokens": model_config.max_tokens,
                "capabilities": [c.value for c in model_config.capabilities]
            }
            
            if include_metrics:
                model_info["cost_per_token"] = model_config.cost_per_token
                model_info["metadata"] = model_config.metadata
            
            result["configured_models"][provider_name].append(model_info)
            result["total_models"] += 1
        
        # Get live models from Ollama if available
        if not provider_filter or provider_filter == "ollama":
            try:
                ollama_models = await self.ollama_service.list_models()
                result["providers"]["ollama"] = {
                    "available": True,
                    "models": ollama_models,
                    "count": len(ollama_models)
                }
            except Exception as e:
                result["providers"]["ollama"] = {
                    "available": False,
                    "error": str(e)
                }
        
        return result


class GetModelInfoTool(BaseTool):
    """Tool for getting detailed information about a specific model."""
    
    def get_name(self) -> str:
        return "get_model_info"
    
    def get_description(self) -> str:
        return "Get detailed information about a specific AI model"
    
    def get_category(self) -> ToolCategory:
        return ToolCategory.MODEL
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model_name": {"type": "string", "description": "Name of the model to get info for"},
                "include_performance": {"type": "boolean", "description": "Include performance metrics"}
            },
            "required": ["model_name"]
        }
    
    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Get model information."""
        model_name = kwargs["model_name"]
        include_performance = kwargs.get("include_performance", False)
        
        # Get from configuration
        model_config = self.config_manager.get_model(model_name)
        if not model_config:
            raise Exception(f"Model not found in configuration: {model_name}")
        
        result = {
            "name": model_config.name,
            "provider": model_config.provider.value,
            "size": model_config.size.value,
            "max_tokens": model_config.max_tokens,
            "temperature": model_config.temperature,
            "timeout": model_config.timeout,
            "capabilities": [c.value for c in model_config.capabilities],
            "metadata": model_config.metadata
        }
        
        # Get provider-specific information
        if model_config.provider.value == "ollama":
            try:
                ollama_service = OllamaService()
                provider_info = await ollama_service.show_model_info(model_name)
                result["provider_info"] = provider_info
            except Exception as e:
                result["provider_info"] = {"error": str(e)}
        
        # Get performance metrics if requested
        if include_performance:
            # This would typically come from stored metrics
            result["performance_metrics"] = {
                "avg_processing_time": 0.0,
                "success_rate": 1.0,
                "total_executions": 0
            }
        
        return result


# =============================================================================
# Utility Tools
# =============================================================================

class GetConfigTool(BaseTool):
    """Tool for retrieving system configuration information."""
    
    def get_name(self) -> str:
        return "get_config"
    
    def get_description(self) -> str:
        return "Get system configuration information and statistics"
    
    def get_category(self) -> ToolCategory:
        return ToolCategory.UTILITY
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "section": {"type": "string", "description": "Configuration section (agents, models, tools, pipelines)"},
                "include_stats": {"type": "boolean", "description": "Include statistics"}
            }
        }
    
    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """Get configuration information."""
        section = kwargs.get("section")
        include_stats = kwargs.get("include_stats", True)
        
        result = {}
        
        if not section or section == "agents":
            result["agents"] = {
                name: {
                    "type": config.agent_type.value,
                    "capabilities": [c.value for c in config.capabilities],
                    "primary_model": config.primary_model,
                    "enabled": config.enabled
                }
                for name, config in self.config_manager.agents.items()
            }
        
        if not section or section == "models":
            result["models"] = {
                name: {
                    "provider": config.provider.value,
                    "size": config.size.value,
                    "max_tokens": config.max_tokens
                }
                for name, config in self.config_manager.models.items()
            }
        
        if not section or section == "tools":
            result["tools"] = {
                name: {
                    "category": config.category.value,
                    "description": config.description,
                    "enabled": config.enabled
                }
                for name, config in self.config_manager.tools.items()
            }
        
        if not section or section == "pipelines":
            result["pipelines"] = {
                name: {
                    "strategy": config.strategy.value,
                    "agents": config.agents,
                    "timeout": config.timeout,
                    "enabled": config.enabled
                }
                for name, config in self.config_manager.pipelines.items()
            }
        
        if include_stats:
            result["stats"] = self.config_manager.get_stats()
        
        return result


# =============================================================================
# Tool Registry and Discovery
# =============================================================================

class ToolRegistry:
    """Registry for managing and discovering MCP tools."""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.config_manager = get_unified_config()
        
        # Auto-discover and register built-in tools
        self._discover_builtin_tools()
    
    def _discover_builtin_tools(self):
        """Discover and register all built-in tools."""
        builtin_tools = [
            TranslateQueryTool(),
            ProcessQueryPipelineTool(),
            ValidateGraphQLTool(),
            ListModelsTool(),
            GetModelInfoTool(),
            GetConfigTool()
        ]
        
        for tool in builtin_tools:
            self.register_tool(tool)
    
    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool in the registry."""
        self.tools[tool.get_name()] = tool
        
        # Register in unified config if not already there
        if not self.config_manager.get_tool(tool.get_name()):
            self.config_manager.register_tool(tool._create_default_config())
        
        logger.info(f"Registered tool: {tool.get_name()} ({tool.get_category().value})")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self, category: Optional[ToolCategory] = None) -> List[BaseTool]:
        """List all tools, optionally filtered by category."""
        tools = list(self.tools.values())
        if category:
            tools = [t for t in tools if t.get_category() == category]
        return tools
    
    def get_mcp_definitions(self) -> List[Dict[str, Any]]:
        """Get MCP tool definitions for all registered tools."""
        return [tool.to_mcp_definition() for tool in self.tools.values()]
    
    async def execute_tool(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool not found: {name}"
            )
        
        return await tool.execute(**kwargs)


# =============================================================================
# Global Registry Instance
# =============================================================================

# Global tool registry instance
tool_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return tool_registry


def register_custom_tool(tool: BaseTool) -> None:
    """Register a custom tool."""
    tool_registry.register_tool(tool)


def get_mcp_tools() -> List[Dict[str, Any]]:
    """Get all MCP tool definitions."""
    return tool_registry.get_mcp_definitions()


async def execute_mcp_tool(name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Execute an MCP tool and return formatted response."""
    result = await tool_registry.execute_tool(name, **arguments)
    return result.to_mcp_response()


# =============================================================================
# Tool Discovery and Extension
# =============================================================================

def discover_custom_tools(tools_dir: Path) -> None:
    """Discover and register custom tools from a directory."""
    if not tools_dir.exists():
        return
    
    # This would implement dynamic loading of custom tool modules
    # For now, it's a placeholder for future extensibility
    logger.info(f"Custom tool discovery from {tools_dir} - not yet implemented")


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_tools():
        registry = get_tool_registry()
        
        # Test translation tool
        result = await registry.execute_tool(
            "translate_query",
            natural_query="Get all users with names",
            schema_context="type User { id: ID! name: String! }"
        )
        print(f"Translation result: {result.success}")
        if result.success:
            print(f"Data: {result.data}")
        
        # Test pipeline tool
        result = await registry.execute_tool(
            "process_query_pipeline",
            query="Find products under $50",
            pipeline="standard"
        )
        print(f"Pipeline result: {result.success}")
        
        # Test validation tool
        result = await registry.execute_tool(
            "validate_graphql",
            query="query { users { id name } }"
        )
        print(f"Validation result: {result.success}")
    
    asyncio.run(test_tools()) 