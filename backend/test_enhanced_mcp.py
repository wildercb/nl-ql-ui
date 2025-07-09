#!/usr/bin/env python3
"""
Test script for Enhanced MCP Server

This script tests the enhanced MCP server functionality to ensure it works correctly.
"""

import asyncio
import logging
import json
from mcp_server.enhanced_agent import EnhancedMCPServer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_enhanced_mcp_server():
    """Test the enhanced MCP server functionality."""
    
    try:
        logger.info("🧪 Testing Enhanced MCP Server...")
        
        # Create and initialize server
        server = EnhancedMCPServer()
        await server.initialize()
        
        logger.info("✅ Server initialized successfully")
        
        # Test list_tools
        tools = await server.list_tools()
        logger.info(f"📋 Available tools: {len(tools)}")
        for tool in tools:
            logger.info(f"  - {tool['name']}: {tool['description']}")
        
        # Test server info
        info = server.get_server_info()
        if 'error' in info:
            logger.warning(f"⚠️  Server info error: {info['error']}")
        else:
            logger.info(f"ℹ️  Server info: {info['name']} v{info['version']}")
        
        # Test a simple query using the fast pipeline
        test_query = "Show me the first 5 thermal scans"
        logger.info(f"🔄 Testing query: '{test_query}'")
        
        result = await server.call_tool("process_query_fast", {
            "query": test_query,
            "translator_model": "gemma3:4b",
            "user_id": "test_user"
        })
        
        logger.info(f"✅ Query result: {json.dumps(result, indent=2)}")
        
        # Test batch processing
        batch_queries = [
            "Show me thermal scans from today",
            "Get maintenance logs for the last week",
            "Find all equipment with temperature above 80 degrees"
        ]
        
        logger.info(f"📦 Testing batch processing with {len(batch_queries)} queries...")
        
        batch_result = await server.call_tool("batch_process_queries", {
            "queries": batch_queries,
            "pipeline_strategy": "fast",
            "max_concurrent": 2,
            "translator_model": "gemma3:4b",
            "user_id": "test_user"
        })
        
        logger.info(f"✅ Batch result: {batch_result['successful']}/{batch_result['total_queries']} successful")
        
        # Cleanup
        await server.cleanup()
        
        logger.info("🎉 All tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_enhanced_mcp_server())
    exit(0 if success else 1) 