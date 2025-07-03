"""Model management tools for AI models (Ollama)."""

import json
from typing import Dict, Any, List, Optional

from fastmcp import FastMCP, Context
from ...services.ollama_service import OllamaService


def register_model_tools(mcp: FastMCP, ollama_service: OllamaService):
    """Register all model management tools."""

    @mcp.tool()
    async def list_available_models(ctx: Context = None) -> Dict[str, Any]:
        """List all available AI models from Ollama."""
        await ctx.info("Retrieving available models from Ollama")
        
        try:
            models = await ollama_service.list_models()
            await ctx.info(f"Found {len(models)} available models")
            
            return {
                "models": models,
                "count": len(models),
                "default_model": ollama_service.default_model
            }
        except Exception as e:
            await ctx.error(f"Failed to list models: {str(e)}")
            return {"error": str(e), "models": []}

    @mcp.tool()
    async def get_model_info(model_name: str, ctx: Context = None) -> Dict[str, Any]:
        """Get detailed information about a specific AI model."""
        await ctx.info(f"Getting information for model: {model_name}")
        
        try:
            model_info = await ollama_service.get_model_info(model_name)
            await ctx.info("Model information retrieved successfully")
            return model_info
        except Exception as e:
            await ctx.error(f"Failed to get model info: {str(e)}")
            return {"error": str(e), "model": model_name}

    @mcp.tool()
    async def pull_model(model_name: str, ctx: Context = None) -> Dict[str, Any]:
        """Download and install a new AI model."""
        await ctx.info(f"Starting download of model: {model_name}")
        
        try:
            result = await ollama_service.pull_model(model_name)
            await ctx.info(f"Model {model_name} downloaded successfully")
            return {"success": True, "model": model_name, "details": result}
        except Exception as e:
            await ctx.error(f"Failed to pull model: {str(e)}")
            return {"error": str(e), "model": model_name}

    @mcp.tool()
    async def delete_model(model_name: str, ctx: Context = None) -> Dict[str, Any]:
        """Remove a model from the system."""
        await ctx.info(f"Deleting model: {model_name}")
        
        try:
            result = await ollama_service.delete_model(model_name)
            await ctx.info(f"Model {model_name} deleted successfully")
            return {"success": True, "model": model_name, "details": result}
        except Exception as e:
            await ctx.error(f"Failed to delete model: {str(e)}")
            return {"error": str(e), "model": model_name} 