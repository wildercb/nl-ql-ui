from fastapi import APIRouter, HTTPException, status, Body
import logging
from typing import List, Dict, Any

from services.translation_service import chat_with_model
from models.query import ChatMessage

router = APIRouter(prefix='/api', tags=['chat'])
logger = logging.getLogger(__name__)

@router.post('/chat')
async def chat(
    query: str = Body(..., embed=True),
    model: str = Body(..., embed=True),
    context: List[Dict[str, Any]] = Body(..., embed=True)
):
    """
    Handle chat interactions with the model based on user query and conversation context.
    
    Args:
        query (str): The user's chat message.
        model (str): The model to use for the chat response.
        context (List[Dict[str, Any]]): The conversation history/context.
    
    Returns:
        Dict: The model's response.
    """
    try:
        logger.info(f"Chat request received - Query: {query}, Model: {model}, Context length: {len(context)}")
        response = await chat_with_model(query, model, context)
        return {'response': response}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}"
        ) 