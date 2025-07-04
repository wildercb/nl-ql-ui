from fastapi import APIRouter, Depends, Query
from sse_starlette.sse import EventSourceResponse
import json, asyncio, logging, urllib.parse
from typing import AsyncGenerator

from services.agent_orchestration_service import AgentOrchestrationService
from .auth import get_current_user

router = APIRouter(prefix="/api", tags=["Multi-Agent Demo"])

logger = logging.getLogger(__name__)

async def _agent_event_generator(
    query: str,
    pre_model: str | None,
    translator_model: str | None,
    review_model: str | None,
) -> AsyncGenerator[dict, None]:
    """Run the multi-agent pipeline and stream incremental events."""
    service = AgentOrchestrationService()

    # 1️⃣ Rewrite
    rewritten = await service._rewrite_query(query, pre_model or service.settings.ollama.default_model)
    yield {
        "event": "rewrite",
        "data": json.dumps({"rewritten_query": rewritten})
    }

    # 2️⃣ Translate (non-streaming for now)
    translation_result = await service.translation_service.translate_to_graphql(
        rewritten, model=translator_model
    )
    yield {
        "event": "graphql",
        "data": json.dumps({
            "graphql_query": translation_result.graphql_query,
            "confidence": translation_result.confidence,
            "explanation": translation_result.explanation,
            "warnings": translation_result.warnings,
            "suggested_improvements": translation_result.suggested_improvements,
        })
    }

    # 3️⃣ Review
    review = await service._review_translation(
        original_query=query,
        graphql_query=translation_result.graphql_query or "",
        model=review_model or translator_model or service.settings.ollama.default_model,
    )
    yield {
        "event": "review",
        "data": json.dumps({
            "passed": review.passed,
            "comments": review.comments,
            "suggestions": review.suggested_improvements,
        })
    }

    # 4️⃣ Complete
    yield {"event": "complete", "data": json.dumps({"status": "done"})}

@router.get("/agent-demo/stream")
async def agent_demo_stream(
    natural_query: str = Query(..., alias="natural_query"),
    pre_model: str | None = None,
    translator_model: str | None = None,
    review_model: str | None = None,
    current_user=Depends(get_current_user),
):
    """Stream multi-agent processing events via Server-Sent Events."""

    async def event_gen():
        try:
            async for evt in _agent_event_generator(natural_query, pre_model, translator_model, review_model):
                yield evt
        except Exception as e:
            logger.exception("Multi-agent stream error")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(event_gen()) 