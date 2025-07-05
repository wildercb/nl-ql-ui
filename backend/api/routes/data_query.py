"""Routes for executing GraphQL data queries against the content database."""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Any, List, Dict

from services.data_query_service import get_data_query_service, DataQueryService

router = APIRouter(prefix="/api", tags=["Data Query"])


class DataQueryRequest(BaseModel):
    graphql_query: str = Field(..., description="GraphQL query string produced by translator")


class DataQueryResponse(BaseModel):
    results: List[Dict[str, Any]]


@router.post("/data/query", response_model=DataQueryResponse)
async def run_data_query(
    payload: DataQueryRequest,
    service: DataQueryService = Depends(get_data_query_service),
):
    """Run the provided GraphQL query against the content database and return JSON results."""
    try:
        docs = await service.run_query(payload.graphql_query)
        return DataQueryResponse(results=docs)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) 