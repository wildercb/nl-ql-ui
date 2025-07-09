#!/usr/bin/env python3
"""
Enhanced MCP Server Startup Script

This script starts the Enhanced MCP Server with proper initialization,
error handling, and graceful shutdown.
"""

import asyncio
import logging
import sys
import signal
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from mcp_server.enhanced_agent import main

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/enhanced_mcp_server.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True

async def run_enhanced_mcp_server():
    """Run the enhanced MCP server with proper error handling."""
    try:
        logger.info("🚀 Starting Enhanced MCP Server...")
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Initialize and run the server
        await main()
        
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received")
    except Exception as e:
        logger.error(f"❌ Server error: {e}", exc_info=True)
        return 1
    finally:
        logger.info("🧹 Server shutdown complete")
    
    return 0

if __name__ == "__main__":
    try:
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)
        
        # Run the server
        exit_code = asyncio.run(run_enhanced_mcp_server())
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1) 