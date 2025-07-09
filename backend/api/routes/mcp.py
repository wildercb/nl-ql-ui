"""
MCP API Routes

Provides HTTP API endpoints that bridge to the Enhanced MCP Server
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from mcp_server.enhanced_agent import EnhancedMCPServer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mcp", tags=["mcp"])

# Global MCP server instance
_mcp_server: Optional[EnhancedMCPServer] = None

async def get_mcp_server() -> EnhancedMCPServer:
    """Get or create the MCP server instance"""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = EnhancedMCPServer()
        await _mcp_server.initialize()
    return _mcp_server

# Request/Response Models
class MCPQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query to process")
    translator_model: str = Field(default="gemma3:4b", description="LLM model to use")
    user_id: str = Field(default="api_user", description="User identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class MCPBatchRequest(BaseModel):
    queries: List[str] = Field(..., description="List of queries to process")
    pipeline_strategy: str = Field(default="standard", description="Pipeline strategy")
    max_concurrent: int = Field(default=3, description="Maximum concurrent queries")
    translator_model: str = Field(default="gemma3:4b", description="LLM model to use")
    user_id: str = Field(default="api_user", description="User identifier")

class MCPResponse(BaseModel):
    original_query: str
    rewritten_query: Optional[str] = None
    translation: Dict[str, Any]
    review: Dict[str, Any]
    processing_time: float
    session_id: str
    pipeline_strategy: str
    events_count: int

class MCPBatchResponse(BaseModel):
    successful: int
    total_queries: int
    results: List[MCPResponse]
    failed_queries: List[Dict[str, str]]
    processing_time: float
    session_id: str

class MCPToolInfo(BaseModel):
    name: str
    description: str

class MCPServerInfo(BaseModel):
    name: str
    version: str
    capabilities: List[str]

# API Endpoints

@router.get("/info", response_model=MCPServerInfo)
async def get_server_info():
    """Get MCP server information"""
    try:
        server = await get_mcp_server()
        info = server.get_server_info()
        
        if 'error' in info:
            raise HTTPException(status_code=500, detail=info['error'])
            
        return MCPServerInfo(
            name=info['name'],
            version=info['version'], 
            capabilities=info.get('capabilities', [])
        )
    except Exception as e:
        logger.error(f"Failed to get server info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tools", response_model=List[MCPToolInfo])
async def get_available_tools():
    """Get list of available MCP tools"""
    try:
        server = await get_mcp_server()
        tools = await server.list_tools()
        
        return [
            MCPToolInfo(name=tool['name'], description=tool['description'])
            for tool in tools
        ]
    except Exception as e:
        logger.error(f"Failed to get tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tools/process_query_standard", response_model=MCPResponse)
async def process_query_standard(request: MCPQueryRequest):
    """Process query using standard pipeline"""
    return await _process_query("process_query_standard", request)

@router.post("/tools/process_query_fast", response_model=MCPResponse)
async def process_query_fast(request: MCPQueryRequest):
    """Process query using fast pipeline"""
    return await _process_query("process_query_fast", request)

@router.post("/tools/process_query_comprehensive", response_model=MCPResponse)
async def process_query_comprehensive(request: MCPQueryRequest):
    """Process query using comprehensive pipeline"""
    return await _process_query("process_query_comprehensive", request)

@router.post("/tools/process_query_adaptive", response_model=MCPResponse)
async def process_query_adaptive(request: MCPQueryRequest):
    """Process query using adaptive pipeline"""
    return await _process_query("process_query_adaptive", request)

@router.post("/tools/batch_process_queries", response_model=MCPBatchResponse)
async def batch_process_queries(request: MCPBatchRequest):
    """Process multiple queries in batch"""
    try:
        server = await get_mcp_server()
        
        result = await server.call_tool("batch_process_queries", {
            "queries": request.queries,
            "pipeline_strategy": request.pipeline_strategy,
            "max_concurrent": request.max_concurrent,
            "translator_model": request.translator_model,
            "user_id": request.user_id
        })
        
        return MCPBatchResponse(**result)
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Streaming endpoints
@router.post("/tools/process_query_standard/stream")
async def process_query_standard_stream(request: MCPQueryRequest):
    """Stream process query using standard pipeline"""
    return StreamingResponse(
        _stream_query("process_query_standard", request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@router.post("/tools/process_query_fast/stream")
async def process_query_fast_stream(request: MCPQueryRequest):
    """Stream process query using fast pipeline"""
    return StreamingResponse(
        _stream_query("process_query_fast", request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@router.post("/tools/process_query_comprehensive/stream")
async def process_query_comprehensive_stream(request: MCPQueryRequest):
    """Stream process query using comprehensive pipeline"""
    return StreamingResponse(
        _stream_query("process_query_comprehensive", request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@router.post("/tools/process_query_adaptive/stream")
async def process_query_adaptive_stream(request: MCPQueryRequest):
    """Stream process query using adaptive pipeline"""
    return StreamingResponse(
        _stream_query("process_query_adaptive", request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# Helper functions

async def _process_query(tool_name: str, request: MCPQueryRequest) -> MCPResponse:
    """Process a single query using the specified tool"""
    try:
        server = await get_mcp_server()
        
        result = await server.call_tool(tool_name, {
            "query": request.query,
            "translator_model": request.translator_model,
            "user_id": request.user_id,
            "metadata": request.metadata
        })
        
        return MCPResponse(**result)
        
    except Exception as e:
        logger.error(f"Query processing failed with {tool_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _stream_query(tool_name: str, request: MCPQueryRequest):
    """Async generator yielding SSE lines from the orchestration service."""
    import json

    try:
        # Get the orchestration service directly
        from services.enhanced_orchestration_service import get_orchestration_service, PipelineStrategy

        orchestration = get_orchestration_service()
        if orchestration is None:
            raise RuntimeError("Orchestration service not initialised")

        strategy_map = {
            "process_query_fast": PipelineStrategy.FAST,
            "process_query_standard": PipelineStrategy.STANDARD,
            "process_query_comprehensive": PipelineStrategy.COMPREHENSIVE,
            "process_query_adaptive": PipelineStrategy.PARALLEL,  # Map adaptive to parallel
        }

        strategy = strategy_map.get(tool_name)
        if strategy is None:
            raise ValueError(f"Unknown streaming tool: {tool_name}")

        async for evt in orchestration.process_query_stream(
            query=request.query,
            translator_model=request.translator_model,
            pipeline_strategy=strategy,
            schema_context=request.metadata.get("schema_context") if request.metadata else None,
            user_id=request.user_id,
        ):
            yield f"data: {json.dumps(evt)}\n\n"

    except Exception as e:
        logger.error(f"Stream processing failed with {tool_name}: {e}")
        yield f"data: {json.dumps({'event': 'error', 'data': {'error': str(e)}})}\n\n"

# Cleanup function
async def cleanup_mcp_server():
    """Cleanup MCP server on shutdown"""
    global _mcp_server
    if _mcp_server:
        await _mcp_server.cleanup()
        _mcp_server = None 