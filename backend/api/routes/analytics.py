"""API endpoints for LLM interaction analytics and monitoring."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel, Field

from services.llm_tracking_service import get_tracking_service
from services.database_service import get_database_service
from models.query import LLMInteraction
from config import get_settings

router = APIRouter(prefix="/analytics", tags=["LLM Analytics"])


class InteractionSummary(BaseModel):
    """Summary of an LLM interaction."""
    
    id: str
    session_id: str
    model: str
    provider: str
    interaction_type: str
    prompt_preview: str
    response_preview: str
    processing_time: float
    tokens_used: Optional[int]
    confidence_score: Optional[float]
    is_successful: bool
    timestamp: datetime


class SessionAnalytics(BaseModel):
    """Analytics for a specific session."""
    
    session_id: str
    total_interactions: int
    successful_interactions: int
    success_rate: float
    total_processing_time: float
    avg_processing_time: float
    total_tokens: int
    avg_confidence: float
    models_used: List[str]
    interaction_types: List[str]
    timeline: List[InteractionSummary]


class ModelStats(BaseModel):
    """Statistics for a specific model."""
    
    model: str
    provider: str
    total_interactions: int
    successful_interactions: int
    success_rate: float
    avg_processing_time: float
    total_tokens: int
    avg_confidence: float


class AnalyticsOverview(BaseModel):
    """High-level analytics overview."""
    
    summary: Dict[str, Any]
    top_models: List[ModelStats]
    interaction_types: List[Dict[str, Any]]
    recent_activity: List[InteractionSummary]


@router.get("/overview", response_model=AnalyticsOverview)
async def get_analytics_overview(
    tracking_service = Depends(get_tracking_service)
) -> AnalyticsOverview:
    """Get high-level analytics overview of all LLM interactions."""
    
    try:
        # Get comprehensive analytics
        analytics = await tracking_service.get_interaction_analytics()
        
        # Get recent interactions for timeline
        recent_interactions = await LLMInteraction.find().sort(
            -LLMInteraction.timestamp
        ).limit(20).to_list()
        
        # Convert to summary format
        recent_activity = []
        for interaction in recent_interactions:
            recent_activity.append(InteractionSummary(
                id=str(interaction.id),
                session_id=interaction.session_id,
                model=interaction.model,
                provider=interaction.provider,
                interaction_type=interaction.interaction_type,
                prompt_preview=interaction.prompt[:100] + "..." if len(interaction.prompt) > 100 else interaction.prompt,
                response_preview=interaction.response[:100] + "..." if len(interaction.response) > 100 else interaction.response,
                processing_time=interaction.processing_time,
                tokens_used=interaction.tokens_used,
                confidence_score=interaction.confidence_score,
                is_successful=interaction.is_successful,
                timestamp=interaction.timestamp
            ))
        
        # Convert top models to response format
        top_models = []
        for model_data in analytics.get("top_models", []):
            top_models.append(ModelStats(
                model=model_data["model"],
                provider="ollama",  # Default provider
                total_interactions=model_data["interactions"],
                successful_interactions=int(model_data["interactions"] * 0.9),  # Estimate
                success_rate=0.9,  # Estimate
                avg_processing_time=model_data.get("avg_processing_time", 0),
                total_tokens=model_data["interactions"] * 100,  # Estimate
                avg_confidence=model_data.get("avg_confidence", 0)
            ))
        
        return AnalyticsOverview(
            summary=analytics.get("summary", {}),
            top_models=top_models,
            interaction_types=analytics.get("interaction_types", []),
            recent_activity=recent_activity
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics overview: {str(e)}")


@router.get("/sessions/{session_id}", response_model=SessionAnalytics)
async def get_session_analytics(
    session_id: str,
    tracking_service = Depends(get_tracking_service)
) -> SessionAnalytics:
    """Get detailed analytics for a specific session."""
    
    try:
        # Get all interactions for the session
        interactions = await tracking_service.get_session_interactions(session_id, limit=1000)
        
        if not interactions:
            raise HTTPException(status_code=404, detail=f"No interactions found for session {session_id}")
        
        # Calculate session statistics
        total_interactions = len(interactions)
        successful_interactions = sum(1 for i in interactions if i.is_successful)
        success_rate = successful_interactions / total_interactions if total_interactions > 0 else 0
        
        total_processing_time = sum(i.processing_time for i in interactions)
        avg_processing_time = total_processing_time / total_interactions if total_interactions > 0 else 0
        
        total_tokens = sum(i.tokens_used or 0 for i in interactions)
        
        confidence_scores = [i.confidence_score for i in interactions if i.confidence_score is not None]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        models_used = list(set(i.model for i in interactions))
        interaction_types = list(set(i.interaction_type for i in interactions))
        
        # Create timeline
        timeline = []
        for interaction in interactions:
            timeline.append(InteractionSummary(
                id=str(interaction.id),
                session_id=interaction.session_id,
                model=interaction.model,
                provider=interaction.provider,
                interaction_type=interaction.interaction_type,
                prompt_preview=interaction.prompt[:100] + "..." if len(interaction.prompt) > 100 else interaction.prompt,
                response_preview=interaction.response[:100] + "..." if len(interaction.response) > 100 else interaction.response,
                processing_time=interaction.processing_time,
                tokens_used=interaction.tokens_used,
                confidence_score=interaction.confidence_score,
                is_successful=interaction.is_successful,
                timestamp=interaction.timestamp
            ))
        
        return SessionAnalytics(
            session_id=session_id,
            total_interactions=total_interactions,
            successful_interactions=successful_interactions,
            success_rate=success_rate,
            total_processing_time=total_processing_time,
            avg_processing_time=avg_processing_time,
            total_tokens=total_tokens,
            avg_confidence=avg_confidence,
            models_used=models_used,
            interaction_types=interaction_types,
            timeline=timeline
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session analytics: {str(e)}")


@router.get("/interactions/{interaction_id}")
async def get_interaction_details(interaction_id: str):
    """Get full details of a specific LLM interaction."""
    
    try:
        from bson import ObjectId
        
        # Get the interaction by ID
        interaction = await LLMInteraction.get(ObjectId(interaction_id))
        
        if not interaction:
            raise HTTPException(status_code=404, detail=f"Interaction {interaction_id} not found")
        
        # Return full interaction details
        return {
            "id": str(interaction.id),
            "session_id": interaction.session_id,
            "user_id": str(interaction.user_id) if interaction.user_id else None,
            "model": interaction.model,
            "provider": interaction.provider,
            "interaction_type": interaction.interaction_type,
            "prompt": interaction.prompt,
            "response": interaction.response,
            "system_prompt": interaction.system_prompt,
            "processing_time": interaction.processing_time,
            "tokens_used": interaction.tokens_used,
            "prompt_tokens": interaction.prompt_tokens,
            "response_tokens": interaction.response_tokens,
            "temperature": interaction.temperature,
            "max_tokens": interaction.max_tokens,
            "context_data": interaction.context_data,
            "confidence_score": interaction.confidence_score,
            "error_message": interaction.error_message,
            "is_successful": interaction.is_successful,
            "timestamp": interaction.timestamp,
            "created_at": interaction.created_at,
            "updated_at": interaction.updated_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get interaction details: {str(e)}")


@router.get("/interactions")
async def get_interactions(
    limit: int = Query(50, ge=1, le=1000, description="Number of interactions to return"),
    offset: int = Query(0, ge=0, description="Number of interactions to skip"),
    model: Optional[str] = Query(None, description="Filter by model name"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
    interaction_type: Optional[str] = Query(None, description="Filter by interaction type"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    is_successful: Optional[bool] = Query(None, description="Filter by success status"),
    start_date: Optional[datetime] = Query(None, description="Filter by start date"),
    end_date: Optional[datetime] = Query(None, description="Filter by end date")
):
    """Get LLM interactions with filtering and pagination."""
    
    try:
        # Build query filters
        filters = {}
        
        if model:
            filters["model"] = model
        if provider:
            filters["provider"] = provider
        if interaction_type:
            filters["interaction_type"] = interaction_type
        if session_id:
            filters["session_id"] = session_id
        if user_id:
            from bson import ObjectId
            filters["user_id"] = ObjectId(user_id)
        if is_successful is not None:
            filters["is_successful"] = is_successful
        
        # Date range filtering
        if start_date or end_date:
            date_filter = {}
            if start_date:
                date_filter["$gte"] = start_date
            if end_date:
                date_filter["$lte"] = end_date
            filters["timestamp"] = date_filter
        
        # Execute query with pagination
        query = LLMInteraction.find(filters) if filters else LLMInteraction.find()
        
        total_count = await query.count()
        interactions = await query.sort(-LLMInteraction.timestamp).skip(offset).limit(limit).to_list()
        
        # Convert to response format
        results = []
        for interaction in interactions:
            results.append({
                "id": str(interaction.id),
                "session_id": interaction.session_id,
                "model": interaction.model,
                "provider": interaction.provider,
                "interaction_type": interaction.interaction_type,
                "prompt_preview": interaction.prompt[:200] + "..." if len(interaction.prompt) > 200 else interaction.prompt,
                "response_preview": interaction.response[:200] + "..." if len(interaction.response) > 200 else interaction.response,
                "processing_time": interaction.processing_time,
                "tokens_used": interaction.tokens_used,
                "confidence_score": interaction.confidence_score,
                "is_successful": interaction.is_successful,
                "timestamp": interaction.timestamp,
                "has_error": bool(interaction.error_message)
            })
        
        return {
            "total_count": total_count,
            "results": results,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_next": offset + limit < total_count,
                "has_prev": offset > 0
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get interactions: {str(e)}")


@router.get("/models/stats")
async def get_model_statistics(
    days: int = Query(7, ge=1, le=365, description="Number of days to analyze"),
    tracking_service = Depends(get_tracking_service)
):
    """Get detailed statistics about model usage over the specified period."""
    
    try:
        stats = await tracking_service.get_model_stats(days=days)
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model statistics: {str(e)}")


@router.get("/database/stats")
async def get_database_statistics(
    db_service = Depends(get_database_service)
):
    """Get database statistics and collection information."""
    
    try:
        stats = await db_service.get_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get database statistics: {str(e)}")


@router.post("/cleanup")
async def cleanup_old_data(
    days: int = Query(90, ge=1, le=365, description="Delete interactions older than this many days"),
    db_service = Depends(get_database_service)
):
    """Clean up old LLM interaction data."""
    
    try:
        from datetime import timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Delete old interactions
        delete_result = await LLMInteraction.find(
            LLMInteraction.timestamp < cutoff_date
        ).delete()
        
        return {
            "deleted_count": delete_result.deleted_count,
            "cutoff_date": cutoff_date,
            "days": days
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cleanup data: {str(e)}")


@router.get("/export/session/{session_id}")
async def export_session_data(session_id: str):
    """Export all data for a specific session in a detailed format."""
    
    try:
        # Get all interactions for the session
        interactions = await LLMInteraction.find(
            LLMInteraction.session_id == session_id
        ).sort(LLMInteraction.timestamp).to_list()
        
        if not interactions:
            raise HTTPException(status_code=404, detail=f"No interactions found for session {session_id}")
        
        # Export detailed session data
        export_data = {
            "session_id": session_id,
            "export_timestamp": datetime.utcnow().isoformat(),
            "total_interactions": len(interactions),
            "interactions": []
        }
        
        for interaction in interactions:
            export_data["interactions"].append({
                "id": str(interaction.id),
                "timestamp": interaction.timestamp.isoformat(),
                "model": interaction.model,
                "provider": interaction.provider,
                "interaction_type": interaction.interaction_type,
                "prompt": interaction.prompt,
                "response": interaction.response,
                "system_prompt": interaction.system_prompt,
                "processing_time": interaction.processing_time,
                "tokens_used": interaction.tokens_used,
                "prompt_tokens": interaction.prompt_tokens,
                "response_tokens": interaction.response_tokens,
                "temperature": interaction.temperature,
                "max_tokens": interaction.max_tokens,
                "context_data": interaction.context_data,
                "confidence_score": interaction.confidence_score,
                "error_message": interaction.error_message,
                "is_successful": interaction.is_successful
            })
        
        return export_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export session data: {str(e)}") 