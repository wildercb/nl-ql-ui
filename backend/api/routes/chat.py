from fastapi import APIRouter, HTTPException, status, Body
import logging
from typing import List, Dict, Any, Optional

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


@router.post('/chat/enhanced')
async def enhanced_chat(
    message: str = Body(...),
    context: Dict[str, Any] = Body(...),
    model: str = Body(default="phi3:mini")
):
    """
    Enhanced chat endpoint that handles full agent context including:
    - Original query and pipeline strategy
    - Complete agent interaction history
    - Generated GraphQL and data results
    - All agent prompts and outputs for comprehensive context
    """
    try:
        logger.info(f"Enhanced chat request - Message: {message}, Strategy: {context.get('pipeline_strategy', 'unknown')}")
        
        # Build comprehensive prompt with all agent context
        enhanced_prompt = build_enhanced_prompt(message, context)
        
        # Use simple chat with enhanced context for now
        # Build enhanced prompt with all context
        enhanced_prompt = build_enhanced_prompt(message, context)
        
        # Use existing chat function with enhanced context
        response = await chat_with_model(
            query=enhanced_prompt,
            model=model,
            context=[]  # Context is already built into the prompt
        )
        
        return {'response': response}
        
    except Exception as e:
        logger.error(f"Enhanced chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Enhanced chat failed: {str(e)}"
        )


def build_enhanced_prompt(message: str, context: Dict[str, Any]) -> str:
    """
    Build a comprehensive prompt that includes all agent context for sophisticated chat continuation.
    """
    
    prompt_parts = [
        "You are an AI assistant with access to a complete multi-agent processing history.",
        "The user is continuing a conversation after running sophisticated agent pipelines.",
        "",
        f"=== ORIGINAL CONTEXT ===",
        f"User's Original Query: {context.get('original_query', 'Unknown')}",
        f"Pipeline Strategy Used: {context.get('pipeline_strategy', 'Unknown')}",
        ""
    ]
    
    # Add agent history
    if context.get('agent_history'):
        prompt_parts.append("=== AGENT PROCESSING HISTORY ===")
        for msg in context['agent_history']:
            agent = msg.get('agent', 'Unknown')
            content = msg.get('content', '')
            prompt_parts.append(f"{agent}: {content}")
        prompt_parts.append("")
    
    # Add generated GraphQL
    if context.get('generated_graphql'):
        prompt_parts.extend([
            "=== GENERATED GRAPHQL QUERY ===",
            f"```graphql\n{context['generated_graphql']}\n```",
            ""
        ])
    
    # Add data results if available
    if context.get('data_results'):
        prompt_parts.extend([
            "=== DATA RESULTS ===",
            f"Retrieved {len(context['data_results'])} records from the database.",
            f"Sample data: {str(context['data_results'][:2]) if context['data_results'] else 'No data'}",
            ""
        ])
    
    prompt_parts.extend([
        "=== CURRENT USER MESSAGE ===",
        f"User: {message}",
        "",
        "Please respond helpfully using all the above context. You can:",
        "- Answer questions about the query or results",
        "- Suggest modifications to the GraphQL query",
        "- Explain the agent processing steps",
        "- Help analyze the data results",
        "- Provide insights based on the pipeline strategy used",
        "",
        "Response:"
    ])
    
    return "\n".join(prompt_parts) 