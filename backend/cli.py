#!/usr/bin/env python3
"""CLI entry point for the FastMCP GraphQL Translation Server."""

import asyncio
import logging
import sys
from pathlib import Path

# Patch event loop for environments that already run asyncio
import nest_asyncio
nest_asyncio.apply()

# Add the backend directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from mcp_server.main import create_mcp_server
from config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def main():
    """Main entry point for the FastMCP server."""
    try:
        # Load settings
        settings = get_settings()
        
        logger.info("Initializing FastMCP GraphQL Translation Server...")
        logger.info(f"Server name: {settings.mcp.name}")
        logger.info(f"Server version: {settings.mcp.version}")
        logger.info(f"Ollama URL: {settings.ollama.base_url}")
        
        # Create and configure the FastMCP server
        server = create_mcp_server()
        
        logger.info("FastMCP server created successfully")
        
        # Start the server
        logger.info("Starting FastMCP server...")
        await server.run()
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server startup failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If already running, create a new loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("Server stopped")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1) 