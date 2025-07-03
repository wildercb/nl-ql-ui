from fastapi import APIRouter, HTTPException, status, Depends, Request, Query
from typing import List, Optional
import logging

from models.query import QueryLog
from models.user import User
from services.database_service import get_query_logs
from .auth import get_current_user, get_session_from_request

router = APIRouter(prefix='/api', tags=['history'])
logger = logging.getLogger(__name__)

@router.get('/history')
async def get_history(
    request: Request,
    limit: int = Query(100, ge=1, le=500),
    skip: int = Query(0, ge=0),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Retrieve the history of translation queries.
    
    For authenticated users: Returns all their historical queries.
    For guest users: Returns only queries from the current session.
    
    Args:
        limit (int): Maximum number of query logs to return. Defaults to 100.
        skip (int): Number of query logs to skip for pagination. Defaults to 0.
        current_user (Optional[User]): Current authenticated user (if any).
    
    Returns:
        Dict: Response containing history array and stats
    """
    try:
        # Get session information
        session_token = await get_session_from_request(request)
        
        if current_user:
            # Authenticated user - get all their history
            logger.info(f"Fetching history for authenticated user: {current_user.username}")
            query_logs = await get_query_logs(
                limit=limit, 
                skip=skip, 
                user_id=str(current_user.id)
            )
            
            # Calculate stats
            total_queries = current_user.total_queries or 0
            successful_queries = current_user.successful_queries or 0
            
        elif session_token:
            # Guest user - get only session-specific history
            logger.info(f"Fetching history for guest session: {session_token[:8]}...")
            query_logs = await get_query_logs(
                limit=limit, 
                skip=skip, 
                session_id=session_token
            )
            
            # Calculate stats for guest session
            total_queries = len(query_logs)
            successful_queries = len([q for q in query_logs if q.is_successful])
            
        else:
            # No authentication or session - return empty history
            logger.info("No authentication or session found, returning empty history")
            query_logs = []
            total_queries = 0
            successful_queries = 0
        
        # Calculate additional stats
        average_confidence = 0.0
        favorite_model = None
        
        if query_logs:
            confidences = [q.confidence for q in query_logs if q.confidence is not None]
            if confidences:
                average_confidence = sum(confidences) / len(confidences)
            
            # Find most used model
            model_counts = {}
            for q in query_logs:
                if q.model_used:
                    model_counts[q.model_used] = model_counts.get(q.model_used, 0) + 1
            if model_counts:
                favorite_model = max(model_counts.items(), key=lambda x: x[1])[0]
        
        logger.info(f"Retrieved {len(query_logs)} query logs")
        
        # Convert QueryLog objects to dictionaries for proper serialization
        history_list = []
        for log in query_logs:
            log_dict = {
                "id": str(log.id),
                "natural_query": log.natural_query,
                "graphql_query": log.graphql_query,
                "model_used": log.model_used,
                "confidence": log.confidence,
                "processing_time": log.processing_time,
                "is_successful": log.is_successful,
                "error_message": log.error_message,
                "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                "user_id": str(log.user_id) if log.user_id else None,
                "session_id": log.session_id
            }
            history_list.append(log_dict)
        
        return {
            "history": history_list,
            "stats": {
                "total_queries": total_queries,
                "successful_queries": successful_queries,
                "average_confidence": average_confidence,
                "favorite_model": favorite_model or ""
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch query history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch query history: {str(e)}"
        )


@router.get('/history/stats')
async def get_history_stats(
    request: Request,
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Get statistics about the user's query history.
    
    Returns:
        Dict: Statistics including total queries, success rate, most used models, etc.
    """
    try:
        session_token = await get_session_from_request(request)
        
        if current_user:
            # Get stats from user object
            total_queries = current_user.total_queries
            successful_queries = current_user.successful_queries
            success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
            
            # Get additional stats from database
            user_queries = await get_query_logs(limit=1000, user_id=current_user.id)
            
        elif session_token:
            # Get stats for guest session
            session_queries = await get_query_logs(limit=1000, session_id=session_token)
            total_queries = len(session_queries)
            successful_queries = len([q for q in session_queries if q.is_successful])
            success_rate = (successful_queries / total_queries * 100) if total_queries > 0 else 0
            user_queries = session_queries
            
        else:
            # No session
            return {
                "total_queries": 0,
                "successful_queries": 0,
                "success_rate": 0,
                "most_used_models": [],
                "recent_activity": []
            }
        
        # Calculate model usage stats
        model_usage = {}
        recent_activity = []
        
        for query in user_queries[:50]:  # Last 50 queries for activity
            if query.model_used:
                model_usage[query.model_used] = model_usage.get(query.model_used, 0) + 1
            
            recent_activity.append({
                "date": query.timestamp.isoformat(),
                "query": query.natural_query[:100],
                "success": query.is_successful,
                "model": query.model_used
            })
        
        most_used_models = [
            {"model": model, "count": count} 
            for model, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True)
        ][:5]
        
        return {
            "total_queries": total_queries,
            "successful_queries": successful_queries,
            "success_rate": round(success_rate, 2),
            "most_used_models": most_used_models,
            "recent_activity": recent_activity[:10],
            "user_type": "authenticated" if current_user else "guest"
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch history stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch history stats: {str(e)}"
        )


@router.delete('/history/clear')
async def clear_history(
    request: Request,
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Clear query history for the current user or session.
    
    For authenticated users: Clears all their history.
    For guest users: Clears only session-specific history.
    """
    try:
        session_token = await get_session_from_request(request)
        
        if current_user:
            # Clear all history for authenticated user
            deleted_count = await QueryLog.find({"user_id": current_user.id}).delete()
            logger.info(f"Cleared {deleted_count} history entries for user: {current_user.username}")
            
            # Reset user stats
            current_user.total_queries = 0
            current_user.successful_queries = 0
            await current_user.save()
            
        elif session_token:
            # Clear session-specific history for guest
            deleted_count = await QueryLog.find({"session_id": session_token}).delete()
            logger.info(f"Cleared {deleted_count} history entries for guest session: {session_token[:8]}...")
            
        else:
            return {"message": "No history to clear", "deleted_count": 0}
        
        return {
            "message": "History cleared successfully",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear history: {str(e)}"
        )


@router.get('/history/export')
async def export_history(
    request: Request,
    format: str = Query("json", regex="^(json|csv)$"),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Export query history in JSON or CSV format.
    
    Args:
        format (str): Export format - "json" or "csv"
        current_user (Optional[User]): Current authenticated user (if any)
    
    Returns:
        Exported history data
    """
    try:
        session_token = await get_session_from_request(request)
        
        if current_user:
            query_logs = await get_query_logs(limit=10000, user_id=current_user.id)
            filename_prefix = f"history_{current_user.username}"
        elif session_token:
            query_logs = await get_query_logs(limit=10000, session_id=session_token)
            filename_prefix = f"history_guest_{session_token[:8]}"
        else:
            query_logs = []
            filename_prefix = "history_empty"
        
        if format == "json":
            from fastapi.responses import JSONResponse
            import json
            from datetime import datetime
            
            # Convert to JSON-serializable format
            export_data = []
            for log in query_logs:
                export_data.append({
                    "id": str(log.id),
                    "natural_query": log.natural_query,
                    "graphql_query": log.graphql_query,
                    "model_used": log.model_used,
                    "confidence": log.confidence,
                    "processing_time": log.processing_time,
                    "is_successful": log.is_successful,
                    "error_message": log.error_message,
                    "timestamp": log.timestamp.isoformat()
                })
            
            return JSONResponse(
                content=export_data,
                headers={
                    "Content-Disposition": f"attachment; filename={filename_prefix}.json"
                }
            )
            
        elif format == "csv":
            from fastapi.responses import StreamingResponse
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "ID", "Natural Query", "GraphQL Query", "Model Used", 
                "Confidence", "Processing Time", "Success", "Error", "Timestamp"
            ])
            
            # Write data
            for log in query_logs:
                writer.writerow([
                    str(log.id),
                    log.natural_query,
                    log.graphql_query or "",
                    log.model_used or "",
                    log.confidence or "",
                    log.processing_time or "",
                    log.is_successful,
                    log.error_message or "",
                    log.timestamp.isoformat()
                ])
            
            output.seek(0)
            
            return StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename={filename_prefix}.csv"
                }
            )
        
    except Exception as e:
        logger.error(f"Failed to export history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export history: {str(e)}"
        ) 