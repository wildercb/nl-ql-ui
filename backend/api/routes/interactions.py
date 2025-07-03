from fastapi import APIRouter, Response
import asyncio
import json
from sse_starlette.sse import EventSourceResponse

from services.llm_tracking_service import get_live_interactions

router = APIRouter(prefix='/api', tags=['interactions'])

@router.get('/interactions/stream')
async def stream_interactions():
    """
    Stream live model interactions from the MCP server using Server-Sent Events (SSE).
    """
    async def event_generator():
        while True:
            interactions = await get_live_interactions()
            for interaction in interactions:
                yield {
                    'event': 'interaction',
                    'data': json.dumps(interaction.dict())
                }
            await asyncio.sleep(1)  # Check for new interactions every second

    return EventSourceResponse(event_generator()) 