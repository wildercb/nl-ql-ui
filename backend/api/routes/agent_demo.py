from typing import Optional, List

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field
import logging

from services.agent_orchestration_service import AgentOrchestrationService, MultiAgentResult
from .auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Multi-Agent Demo"])


class AgentDemoRequest(BaseModel):
    """Request body for multi-agent pipeline."""

    natural_query: str = Field(..., min_length=1, max_length=2000)
    pre_model: Optional[str] = Field(None, description="Model used for pre-processor agent")
    translator_model: Optional[str] = Field(None, description="Model used for translator agent")
    review_model: Optional[str] = Field(None, description="Model used for reviewer agent")


class AgentDemoResponse(BaseModel):
    """Flattened response for easy consumption in the UI."""

    original_query: str
    rewritten_query: str
    graphql_query: str
    confidence: float
    review_passed: bool
    review_comments: List[str]
    review_suggestions: List[str]
    processing_time: float

    @classmethod
    def from_result(cls, result: MultiAgentResult):
        return cls(
            original_query=result.original_query,
            rewritten_query=result.rewritten_query,
            graphql_query=result.translation.graphql_query or "",
            confidence=result.translation.confidence,
            review_passed=result.review.passed,
            review_comments=result.review.comments,
            review_suggestions=result.review.suggested_improvements,
            processing_time=result.processing_time,
        )


@router.post("/agent-demo", response_model=AgentDemoResponse)
async def run_agent_demo(
    request: AgentDemoRequest,
    current_user=Depends(get_current_user),
):
    """Execute the multi-agent NL→GraphQL pipeline in one call."""

    try:
        service = AgentOrchestrationService()
        result = await service.process_query(
            request.natural_query,
            pre_model=request.pre_model,
            translator_model=request.translator_model,
            review_model=request.review_model,
        )

        # (Optional) you could persist logs here if desired.
        return AgentDemoResponse.from_result(result)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception("Multi-agent processing failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Processing error") 