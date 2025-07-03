#!/usr/bin/env python3
"""Script to run the MCP server."""

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from mcp_server.server import MCPServer
from config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class MCPServerRunner:
    """Runner for the MCP server with graceful shutdown."""
    
    def __init__(self):
        self.server = MCPServer()
        self.running = False
    
    async def start(self):
        """Start the MCP server."""
        self.running = True
        settings = get_settings()
        
        logger.info(f"Starting MCP server: {settings.mcp.name}")
        logger.info(f"Server description: {settings.mcp.description}")
        logger.info(f"Version: {settings.mcp.version}")
        
        try:
            await self.server.run()
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the MCP server gracefully."""
        if self.running:
            logger.info("Stopping MCP server...")
            self.running = False
            # Add any cleanup code here
            logger.info("MCP server stopped")
    
    def setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            if self.running:
                # This will trigger KeyboardInterrupt in the main loop
                raise KeyboardInterrupt()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


async def main():
    """Main entry point."""
    runner = MCPServerRunner()
    runner.setup_signal_handlers()
    
    try:
        await runner.start()
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown completed")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1) 