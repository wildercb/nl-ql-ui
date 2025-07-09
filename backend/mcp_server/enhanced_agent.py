"""Enhanced MCP Agent Server

This MCP server provides the same functionality as the enhanced orchestration service
but through a basic Model Context Protocol interface. It supports streaming responses,
multiple pipeline strategies, and comprehensive error handling.

Compatible with Python 3.9+ without external MCP dependencies.
"""

import asyncio
import logging
import json
import uuid
import sys
from typing import Dict, List, Any, Optional, AsyncGenerator
from dataclasses import asdict

from config.settings import get_settings
from services.enhanced_orchestration_service import EnhancedOrchestrationService, PipelineStrategy
from services.database_service import initialize_database, close_database
from services.llm_tracking_service import get_tracking_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global settings and services
settings = get_settings()
orchestration_service = None
tracking_service = None

class EnhancedMCPServer:
    """Enhanced MCP Server that provides the same functionality as the enhanced orchestration service."""
    
    def __init__(self):
        self.name = "Enhanced Multi-Agent MCP Server"
        self.version = "1.0.0"
        self.initialized = False
    
    async def initialize(self):
        """Initialize the enhanced MCP server with all required services."""
        global orchestration_service, tracking_service
        
        try:
            logger.info("🚀 Initializing Enhanced MCP Server...")
            
            # Initialize database
            await initialize_database()
            logger.info("✅ Database initialized")
            
            # Initialize services
            orchestration_service = EnhancedOrchestrationService()
            tracking_service = get_tracking_service()
            
            self.initialized = True
            logger.info("✅ Enhanced MCP Server initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Enhanced MCP Server: {e}")
            raise

    async def cleanup(self):
        """Clean up Enhanced MCP server resources."""
        try:
            logger.info("🧹 Cleaning up Enhanced MCP Server...")
            
            # Close database connection
            await close_database()
            
            logger.info("✅ Enhanced MCP Server cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during Enhanced MCP Server cleanup: {e}")

    def create_session_id(self) -> str:
        """Create a unique session ID for tracking interactions."""
        return f"enhanced-mcp-{uuid.uuid4().hex[:8]}"

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return [
            {
                "name": "process_query_standard",
                "description": "Process a query using the standard pipeline (rewrite → translate → review)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The natural language query to process"},
                        "pre_model": {"type": "string", "description": "Model for query rewriting"},
                        "translator_model": {"type": "string", "description": "Model for translation"},
                        "review_model": {"type": "string", "description": "Model for review"},
                        "domain_context": {"type": "string", "description": "Domain-specific context"},
                        "schema_context": {"type": "string", "description": "GraphQL schema context"},
                        "user_id": {"type": "string", "description": "User identifier for tracking"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "process_query_fast",
                "description": "Process a query using the fast pipeline (translation only)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The natural language query to process"},
                        "translator_model": {"type": "string", "description": "Model for translation"},
                        "schema_context": {"type": "string", "description": "GraphQL schema context"},
                        "user_id": {"type": "string", "description": "User identifier for tracking"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "process_query_comprehensive",
                "description": "Process a query using the comprehensive pipeline (all agents + optimization + data review)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The natural language query to process"},
                        "pre_model": {"type": "string", "description": "Model for query rewriting"},
                        "translator_model": {"type": "string", "description": "Model for translation"},
                        "review_model": {"type": "string", "description": "Model for review"},
                        "domain_context": {"type": "string", "description": "Domain-specific context"},
                        "schema_context": {"type": "string", "description": "GraphQL schema context"},
                        "user_id": {"type": "string", "description": "User identifier for tracking"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "process_query_adaptive",
                "description": "Process a query using the adaptive pipeline (strategy selected based on query complexity)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The natural language query to process"},
                        "pre_model": {"type": "string", "description": "Model for query rewriting"},
                        "translator_model": {"type": "string", "description": "Model for translation"},
                        "review_model": {"type": "string", "description": "Model for review"},
                        "domain_context": {"type": "string", "description": "Domain-specific context"},
                        "schema_context": {"type": "string", "description": "GraphQL schema context"},
                        "user_id": {"type": "string", "description": "User identifier for tracking"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "batch_process_queries",
                "description": "Process multiple queries in batch using the specified pipeline strategy",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of natural language queries to process"
                        },
                        "pipeline_strategy": {
                            "type": "string",
                            "enum": ["standard", "fast", "comprehensive", "adaptive"],
                            "description": "Pipeline strategy to use for all queries"
                        },
                        "max_concurrent": {"type": "integer", "description": "Maximum number of concurrent query processing"},
                        "translator_model": {"type": "string", "description": "Model for translation"},
                        "schema_context": {"type": "string", "description": "GraphQL schema context"},
                        "user_id": {"type": "string", "description": "User identifier for tracking"}
                    },
                    "required": ["queries"]
                }
            }
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool calls."""
        
        session_id = self.create_session_id()
        
        try:
            if name == "process_query_standard":
                result = await self.process_query_standard(
                    query=arguments["query"],
                    pre_model=arguments.get("pre_model"),
                    translator_model=arguments.get("translator_model"),
                    review_model=arguments.get("review_model"),
                    domain_context=arguments.get("domain_context"),
                    schema_context=arguments.get("schema_context"),
                    user_id=arguments.get("user_id"),
                    session_id=session_id
                )
            elif name == "process_query_fast":
                result = await self.process_query_fast(
                    query=arguments["query"],
                    translator_model=arguments.get("translator_model"),
                    schema_context=arguments.get("schema_context"),
                    user_id=arguments.get("user_id"),
                    session_id=session_id
                )
            elif name == "process_query_comprehensive":
                result = await self.process_query_comprehensive(
                    query=arguments["query"],
                    pre_model=arguments.get("pre_model"),
                    translator_model=arguments.get("translator_model"),
                    review_model=arguments.get("review_model"),
                    domain_context=arguments.get("domain_context"),
                    schema_context=arguments.get("schema_context"),
                    user_id=arguments.get("user_id"),
                    session_id=session_id
                )
            elif name == "process_query_adaptive":
                result = await self.process_query_adaptive(
                    query=arguments["query"],
                    pre_model=arguments.get("pre_model"),
                    translator_model=arguments.get("translator_model"),
                    review_model=arguments.get("review_model"),
                    domain_context=arguments.get("domain_context"),
                    schema_context=arguments.get("schema_context"),
                    user_id=arguments.get("user_id"),
                    session_id=session_id
                )
            elif name == "batch_process_queries":
                result = await self.batch_process_queries(
                    queries=arguments["queries"],
                    pipeline_strategy=arguments.get("pipeline_strategy", "standard"),
                    max_concurrent=arguments.get("max_concurrent", 3),
                    translator_model=arguments.get("translator_model"),
                    schema_context=arguments.get("schema_context"),
                    user_id=arguments.get("user_id"),
                    session_id=session_id
                )
            else:
                raise ValueError(f"Unknown tool: {name}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Tool call failed: {e}")
            error_result = {
                "session_id": session_id,
                "error": str(e),
                "tool": name,
                "success": False
            }
            return error_result

    # =============================================================================
    # TOOL IMPLEMENTATION FUNCTIONS
    # =============================================================================

    async def process_query_standard(
        self,
        query: str,
        pre_model: Optional[str] = None,
        translator_model: Optional[str] = None,
        review_model: Optional[str] = None,
        domain_context: Optional[str] = None,
        schema_context: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a query using the standard pipeline."""
        
        if not session_id:
            session_id = self.create_session_id()
        
        try:
            logger.info(f"🔄 Processing query with standard pipeline: {session_id}")
            
            # Ensure orchestration service is initialized
            if not orchestration_service:
                raise Exception("Orchestration service not initialized")
            
            # Collect all streaming events
            events = []
            final_result = None
            
            async for event in orchestration_service.process_query_stream(
                query=query,
                pre_model=pre_model,
                translator_model=translator_model,
                review_model=review_model,
                pipeline_strategy=PipelineStrategy.STANDARD,
                domain_context=domain_context,
                schema_context=schema_context,
                user_id=user_id,
                session_id=session_id
            ):
                events.append(event)
                if event['event'] == 'complete':
                    final_result = event['data']['result']
            
            if not final_result:
                raise Exception("Pipeline failed to produce results")
            
            # Add session tracking info
            final_result['session_id'] = session_id
            final_result['pipeline_strategy'] = PipelineStrategy.STANDARD
            final_result['events_count'] = len(events)
            
            logger.info(f"✅ Standard pipeline completed: {session_id}")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Standard pipeline failed: {e}")
            return {
                "session_id": session_id,
                "error": str(e),
                "pipeline_strategy": PipelineStrategy.STANDARD,
                "success": False
            }

    async def process_query_fast(
        self,
        query: str,
        translator_model: Optional[str] = None,
        schema_context: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a query using the fast pipeline."""
        
        if not session_id:
            session_id = self.create_session_id()
        
        try:
            logger.info(f"⚡ Processing query with fast pipeline: {session_id}")
            
            # Ensure orchestration service is initialized
            if not orchestration_service:
                raise Exception("Orchestration service not initialized")
            
            # Collect all streaming events
            events = []
            final_result = None
            
            async for event in orchestration_service.process_query_stream(
                query=query,
                translator_model=translator_model,
                pipeline_strategy=PipelineStrategy.FAST,
                schema_context=schema_context,
                user_id=user_id,
                session_id=session_id
            ):
                events.append(event)
                if event['event'] == 'complete':
                    final_result = event['data']['result']
            
            if not final_result:
                raise Exception("Fast pipeline failed to produce results")
            
            # Add session tracking info
            final_result['session_id'] = session_id
            final_result['pipeline_strategy'] = PipelineStrategy.FAST
            final_result['events_count'] = len(events)
            
            logger.info(f"✅ Fast pipeline completed: {session_id}")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Fast pipeline failed: {e}")
            return {
                "session_id": session_id,
                "error": str(e),
                "pipeline_strategy": PipelineStrategy.FAST,
                "success": False
            }

    async def process_query_comprehensive(
        self,
        query: str,
        pre_model: Optional[str] = None,
        translator_model: Optional[str] = None,
        review_model: Optional[str] = None,
        domain_context: Optional[str] = None,
        schema_context: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a query using the comprehensive pipeline."""
        
        if not session_id:
            session_id = self.create_session_id()
        
        try:
            logger.info(f"🔧 Processing query with comprehensive pipeline: {session_id}")
            
            # Ensure orchestration service is initialized
            if not orchestration_service:
                raise Exception("Orchestration service not initialized")
            
            # Collect all streaming events
            events = []
            final_result = None
            
            async for event in orchestration_service.process_query_stream(
                query=query,
                pre_model=pre_model,
                translator_model=translator_model,
                review_model=review_model,
                pipeline_strategy=PipelineStrategy.COMPREHENSIVE,
                domain_context=domain_context,
                schema_context=schema_context,
                user_id=user_id,
                session_id=session_id
            ):
                events.append(event)
                if event['event'] == 'complete':
                    final_result = event['data']['result']
            
            if not final_result:
                raise Exception("Comprehensive pipeline failed to produce results")
            
            # Add session tracking info
            final_result['session_id'] = session_id
            final_result['pipeline_strategy'] = PipelineStrategy.COMPREHENSIVE
            final_result['events_count'] = len(events)
            
            logger.info(f"✅ Comprehensive pipeline completed: {session_id}")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Comprehensive pipeline failed: {e}")
            return {
                "session_id": session_id,
                "error": str(e),
                "pipeline_strategy": PipelineStrategy.COMPREHENSIVE,
                "success": False
            }

    async def process_query_adaptive(
        self,
        query: str,
        pre_model: Optional[str] = None,
        translator_model: Optional[str] = None,
        review_model: Optional[str] = None,
        domain_context: Optional[str] = None,
        schema_context: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a query using the adaptive pipeline."""
        
        if not session_id:
            session_id = self.create_session_id()
        
        try:
            logger.info(f"🧠 Processing query with adaptive pipeline: {session_id}")
            
            # Ensure orchestration service is initialized
            if not orchestration_service:
                raise Exception("Orchestration service not initialized")
            
            # Collect all streaming events
            events = []
            final_result = None
            
            async for event in orchestration_service.process_query_stream(
                query=query,
                pre_model=pre_model,
                translator_model=translator_model,
                review_model=review_model,
                pipeline_strategy=PipelineStrategy.ADAPTIVE,
                domain_context=domain_context,
                schema_context=schema_context,
                user_id=user_id,
                session_id=session_id
            ):
                events.append(event)
                if event['event'] == 'complete':
                    final_result = event['data']['result']
            
            if not final_result:
                raise Exception("Adaptive pipeline failed to produce results")
            
            # Add session tracking info
            final_result['session_id'] = session_id
            final_result['pipeline_strategy'] = PipelineStrategy.ADAPTIVE
            final_result['events_count'] = len(events)
            
            logger.info(f"✅ Adaptive pipeline completed: {session_id}")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Adaptive pipeline failed: {e}")
            return {
                "session_id": session_id,
                "error": str(e),
                "pipeline_strategy": PipelineStrategy.ADAPTIVE,
                "success": False
            }

    async def batch_process_queries(
        self,
        queries: List[str],
        pipeline_strategy: str = "standard",
        max_concurrent: int = 3,
        translator_model: Optional[str] = None,
        schema_context: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process multiple queries in batch."""
        
        if not session_id:
            session_id = self.create_session_id()
        
        try:
            logger.info(f"📦 Processing {len(queries)} queries in batch: {session_id}")
            
            # Process queries concurrently with semaphore
            semaphore = asyncio.Semaphore(max_concurrent)
            results = []
            
            async def process_single_query(query: str, index: int) -> Dict[str, Any]:
                async with semaphore:
                    query_session_id = f"{session_id}-{index}"
                    
                    try:
                        # Ensure orchestration service is initialized
                        if not orchestration_service:
                            raise Exception("Orchestration service not initialized")
                        
                        events = []
                        final_result = None
                        
                        async for event in orchestration_service.process_query_stream(
                            query=query,
                            translator_model=translator_model,
                            pipeline_strategy=pipeline_strategy,
                            schema_context=schema_context,
                            user_id=user_id,
                            session_id=query_session_id
                        ):
                            events.append(event)
                            if event['event'] == 'complete':
                                final_result = event['data']['result']
                        
                        if not final_result:
                            raise Exception("Query processing failed")
                        
                        final_result['query_index'] = index
                        final_result['session_id'] = query_session_id
                        return final_result
                        
                    except Exception as e:
                        logger.error(f"Query {index} failed: {e}")
                        return {
                            "query_index": index,
                            "session_id": query_session_id,
                            "error": str(e),
                            "success": False
                        }
            
            # Execute all queries
            tasks = [process_single_query(query, i) for i, query in enumerate(queries)]
            results = await asyncio.gather(*tasks)
            
            # Calculate batch statistics
            successful = sum(1 for r in results if not r.get("error"))
            failed = len(results) - successful
            
            batch_result = {
                "session_id": session_id,
                "pipeline_strategy": pipeline_strategy,
                "total_queries": len(queries),
                "successful": successful,
                "failed": failed,
                "max_concurrent": max_concurrent,
                "results": results
            }
            
            logger.info(f"✅ Batch processing completed: {successful}/{len(queries)} successful")
            return batch_result
            
        except Exception as e:
            logger.error(f"❌ Batch processing failed: {e}")
            return {
                "session_id": session_id,
                "error": str(e),
                "pipeline_strategy": pipeline_strategy,
                "success": False
            }

    def get_server_info(self) -> Dict[str, Any]:
        """Get enhanced MCP server information and capabilities."""
        
        try:
            info = {
                "name": "Enhanced Multi-Agent MCP Server",
                "version": "1.0.0",
                "description": "MCP server providing enhanced multi-agent orchestration capabilities",
                "capabilities": {
                    "pipeline_strategies": list(orchestration_service.pipeline_configs.keys()) if orchestration_service else [],
                    "streaming_support": True,
                    "batch_processing": True,
                    "error_recovery": True,
                    "performance_monitoring": True
                },
                "settings": {
                    "default_model": settings.ollama.default_model,
                    "ollama_base_url": settings.ollama.base_url,
                    "database_url": settings.database.url
                },
                "available_tools": [
                    "process_query_standard",
                    "process_query_fast", 
                    "process_query_comprehensive",
                    "process_query_adaptive",
                    "batch_process_queries"
                ]
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get server information: {e}")
            return {"error": str(e)}

    async def run_stdio(self):
        """Run the server using stdio transport."""
        
        if not self.initialized:
            await self.initialize()
        
        logger.info("🚀 Enhanced MCP Server starting...")
        logger.info("📋 Available Tools:")
        logger.info("   - process_query_standard - Standard pipeline processing")
        logger.info("   - process_query_fast - Fast pipeline processing")
        logger.info("   - process_query_comprehensive - Comprehensive pipeline processing")
        logger.info("   - process_query_adaptive - Adaptive pipeline processing")
        logger.info("   - batch_process_queries - Batch query processing")
        logger.info("✅ Enhanced MCP Server ready for connections!")
        
        # Simple stdio-based interaction loop
        try:
            while True:
                # Read input from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                
                try:
                    # Parse JSON input
                    request = json.loads(line.strip())
                    
                    # Handle different request types
                    if request.get("method") == "tools/list":
                        tools = await self.list_tools()
                        response = {"id": request.get("id"), "result": {"tools": tools}}
                    elif request.get("method") == "tools/call":
                        params = request.get("params", {})
                        result = await self.call_tool(params.get("name"), params.get("arguments", {}))
                        response = {"id": request.get("id"), "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}}
                    elif request.get("method") == "initialize":
                        response = {"id": request.get("id"), "result": {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}}}}
                    else:
                        response = {"id": request.get("id"), "error": {"code": -32601, "message": "Method not found"}}
                    
                    # Send response to stdout
                    print(json.dumps(response))
                    sys.stdout.flush()
                    
                except json.JSONDecodeError:
                    error_response = {"error": {"code": -32700, "message": "Parse error"}}
                    print(json.dumps(error_response))
                    sys.stdout.flush()
                except Exception as e:
                    error_response = {"error": {"code": -32603, "message": f"Internal error: {str(e)}"}}
                    print(json.dumps(error_response))
                    sys.stdout.flush()
                    
        except KeyboardInterrupt:
            logger.info("🛑 Server shutdown requested")
        except Exception as e:
            logger.error(f"❌ Server error: {e}")
            raise
        finally:
            # Cleanup
            await self.cleanup()

# =============================================================================
# SERVER INSTANCE AND MAIN FUNCTION
# =============================================================================

# Create server instance
server = EnhancedMCPServer()

async def main():
    """Main entry point for the enhanced MCP server."""
    await server.run_stdio()

if __name__ == "__main__":
    asyncio.run(main()) 