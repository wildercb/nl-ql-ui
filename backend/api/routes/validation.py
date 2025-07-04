"""Validation API endpoints."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
import structlog

from services.validation_service import ValidationService, ValidationResult
from config import get_settings

logger = structlog.get_logger()
router = APIRouter()


class ValidationRequest(BaseModel):
    """Request model for GraphQL validation."""
    query: str = Field(..., min_length=1, description="GraphQL query to validate")
    graphql_schema: Optional[str] = Field(None, description="GraphQL schema for validation")


class BatchValidationRequest(BaseModel):
    """Request model for batch validation."""
    queries: List[str] = Field(..., min_items=1, max_items=20, description="List of GraphQL queries to validate")
    graphql_schema: Optional[str] = Field(None, description="GraphQL schema for validation")


class ValidationResponse(BaseModel):
    """Response model for validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    query_info: Optional[dict] = None

    @classmethod
    def from_result(cls, result: ValidationResult, query_info: Optional[dict] = None) -> "ValidationResponse":
        """Create response from service result."""
        return cls(
            is_valid=result.is_valid,
            errors=result.errors,
            warnings=result.warnings,
            suggestions=result.suggestions,
            query_info=query_info
        )


class BatchValidationResponse(BaseModel):
    """Response model for batch validation."""
    results: List[ValidationResponse]
    total_queries: int
    valid_queries: int
    error_count: int
    warning_count: int


async def get_validation_service() -> ValidationService:
    """Dependency to get validation service."""
    return ValidationService()


@router.post("/validate", response_model=ValidationResponse)
async def validate_query(
    request: ValidationRequest,
    validation_service: ValidationService = Depends(get_validation_service)
):
    """Validate a GraphQL query."""
    try:
        logger.info("Validation request", query_length=len(request.query), has_schema=bool(request.graphql_schema))
        
        result = await validation_service.validate_query(
            query=request.query,
            schema=request.graphql_schema
        )
        
        # Get additional query info
        query_info = validation_service.extract_query_info(request.query)
        
        response = ValidationResponse.from_result(result, query_info)
        
        logger.info(
            "Validation completed",
            is_valid=result.is_valid,
            error_count=len(result.errors),
            warning_count=len(result.warnings)
        )
        
        return response
        
    except Exception as e:
        logger.error("Validation failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Validation service error")


@router.post("/validate/batch", response_model=BatchValidationResponse)
async def validate_queries_batch(
    request: BatchValidationRequest,
    validation_service: ValidationService = Depends(get_validation_service)
):
    """Validate multiple GraphQL queries."""
    try:
        logger.info("Batch validation request", query_count=len(request.queries))
        
        results = await validation_service.validate_queries_batch(
            queries=request.queries,
            schema=request.graphql_schema
        )
        
        responses = []
        for i, result in enumerate(results):
            query_info = validation_service.extract_query_info(request.queries[i])
            responses.append(ValidationResponse.from_result(result, query_info))
        
        valid_count = sum(1 for r in responses if r.is_valid)
        error_count = sum(len(r.errors) for r in responses)
        warning_count = sum(len(r.warnings) for r in responses)
        
        logger.info(
            "Batch validation completed",
            total_queries=len(request.queries),
            valid_queries=valid_count,
            error_count=error_count,
            warning_count=warning_count
        )
        
        return BatchValidationResponse(
            results=responses,
            total_queries=len(request.queries),
            valid_queries=valid_count,
            error_count=error_count,
            warning_count=warning_count
        )
        
    except Exception as e:
        logger.error("Batch validation failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Validation service error")


@router.get("/rules")
async def get_validation_rules(
    validation_service: ValidationService = Depends(get_validation_service)
):
    """Get list of GraphQL validation rules."""
    try:
        rules = validation_service.get_validation_rules()
        return {"rules": rules}
    except Exception as e:
        logger.error("Failed to get validation rules", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve validation rules")


@router.post("/analyze")
async def analyze_query(
    request: ValidationRequest,
    validation_service: ValidationService = Depends(get_validation_service)
):
    """Analyze a GraphQL query and provide detailed information."""
    try:
        # Validate the query
        validation_result = await validation_service.validate_query(
            query=request.query,
            schema=request.graphql_schema
        )
        
        # Extract detailed query information
        query_info = validation_service.extract_query_info(request.query)
        
        return {
            "validation": ValidationResponse.from_result(validation_result),
            "query_analysis": query_info,
            "recommendations": {
                "performance": [],
                "security": [],
                "best_practices": validation_result.suggestions
            }
        }
        
    except Exception as e:
        logger.error("Query analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail="Analysis service error") 