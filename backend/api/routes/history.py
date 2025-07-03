from fastapi import APIRouter, HTTPException, status
from typing import List
import logging

from models.query import QueryLog
from services.database_service import get_query_logs

router = APIRouter(prefix='/api', tags=['history'])
logger = logging.getLogger(__name__)

@router.get('/history', response_model=List[QueryLog])
async def get_history(limit: int = 100, skip: int = 0):
    """
    Retrieve the history of translation queries from MongoDB.
    
    Args:
        limit (int): Maximum number of query logs to return. Defaults to 100.
        skip (int): Number of query logs to skip for pagination. Defaults to 0.
    
    Returns:
        List[QueryLog]: List of query log entries.
    """
    try:
        query_logs = await get_query_logs(limit=limit, skip=skip)
        return query_logs
    except Exception as e:
        logger.error(f"Failed to fetch query history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch query history: {str(e)}"
        ) 