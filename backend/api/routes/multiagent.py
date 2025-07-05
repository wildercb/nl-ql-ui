"""
Multi-Agent Orchestration API Routes

Enhanced routes that leverage the new sophisticated orchestration system
with multiple pipeline strategies, performance monitoring, and advanced features.
"""

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import json
import logging

from services.enhanced_orchestration_service import (
    EnhancedOrchestrationService, 
    PipelineStrategy,
    get_orchestration_service
)

router = APIRouter(prefix='/api/multiagent', tags=['multiagent'])

class StreamRequest(BaseModel):
    query: str = Field(..., description="Natural language query to process")
    pipeline_strategy: str = Field(PipelineStrategy.STANDARD, description="Pipeline strategy to use")
    translator_model: Optional[str] = Field(None, description="Model for translation")
    pre_model: Optional[str] = Field(None, description="Model for preprocessing/rewriting")
    review_model: Optional[str] = Field(None, description="Model for review")
    user_id: Optional[str] = Field(None, description="User identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")

async def stream_processor(service: EnhancedOrchestrationService, request: StreamRequest):
    """Processes a request and yields SSE-formatted events."""
    logger = logging.getLogger(__name__)
    logger.info(f"🎬 Starting stream processor for query: {request.query[:50]}...")
    
    stream = service.process_query_stream(
        query=request.query,
        pipeline_strategy=request.pipeline_strategy,
        translator_model=request.translator_model,
        pre_model=request.pre_model,
        review_model=request.review_model,
        user_id=request.user_id,
        session_id=request.session_id,
    )
    event_count = 0
    async for event in stream:
        event_count += 1
        event_type = event.get('event', 'unknown')
        event_data = event.get('data', {}) if isinstance(event, dict) else event
        
        # Log detailed event information
        if event_type == 'agent_token':
            logger.debug(f"🔄 SSE #{event_count}: {event_type} -> token: '{event_data.get('token', '')[:20]}...' for agent: {event_data.get('agent', 'unknown')}")
        else:
            logger.info(f"🔄 SSE #{event_count}: {event_type} -> {str(event_data)[:100]}...")
        
        yield {
            "event": event.get('event', 'message'),
            "data": json.dumps(event)
        }
    
    logger.info(f"✅ Stream processor completed, sent {event_count} events")

@router.post('/process/stream')
async def stream_multiagent_processing(
    request: StreamRequest,
    service: EnhancedOrchestrationService = Depends(get_orchestration_service)
):
    """
    Stream enhanced multi-agent processing with real-time updates.
    """
    # The 'enhanced' agent buttons will call this endpoint directly
    # with the chosen strategy in the request body.
    return EventSourceResponse(stream_processor(service, request))

@router.post('/translate/stream')
async def stream_translation(
    request: StreamRequest,
    service: EnhancedOrchestrationService = Depends(get_orchestration_service)
):
    """
    Stream translation-only pipeline. This is for the 'Translate' button.
    """
    request.pipeline_strategy = PipelineStrategy.FAST
    return EventSourceResponse(stream_processor(service, request))

@router.post('/legacy/stream')
async def stream_legacy_multiagent(
    request: StreamRequest,
    service: EnhancedOrchestrationService = Depends(get_orchestration_service)
):
    """
    Stream standard multi-agent pipeline. This is for the 'Multi-Agent' button.
    """
    request.pipeline_strategy = PipelineStrategy.STANDARD
    return EventSourceResponse(stream_processor(service, request)) 