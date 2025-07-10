"""
Chat API Routes

Handles live chat conversations with context from agent responses.
"""

from fastapi import APIRouter, HTTPException, status, Body, Depends
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
import logging
from typing import List, Dict, Any, Optional
import json
import asyncio

from services.translation_service import chat_with_model
from services.ollama_service import OllamaService
from config.settings import get_settings

from models.query import ChatMessage

router = APIRouter(prefix='/api/chat', tags=['chat'])
logger = logging.getLogger(__name__)

class ChatMessageRequest(BaseModel):
    role: str = Field(..., description="Role of the message sender ('user', 'assistant', 'system')")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    messages: List[ChatMessageRequest] = Field(..., description="List of chat messages")
    model: Optional[str] = Field(None, description="Model to use for chat")
    stream: bool = Field(True, description="Whether to stream the response")
    temperature: float = Field(0.7, description="Temperature for response generation")
    max_tokens: int = Field(2048, description="Maximum tokens to generate")

async def get_ollama_service() -> OllamaService:
    """Get Ollama service instance."""
    return OllamaService()

@router.post('/chat')
async def chat(
    messages: List[ChatMessageRequest] = Body(..., description="List of chat messages"),
    model: Optional[str] = Body(None, description="Model to use for chat")
):
    """
    Chat with the model using the provided messages.
    """
    try:
        logger.info(f"🦙 Chat request with model: {model or 'default'}")
        
        # Messages are already in the correct format
        formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        
        # Get the last user message as the query
        user_messages = [msg for msg in formatted_messages if msg["role"] == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")
        
        query = user_messages[-1]["content"]
        context = [msg for msg in formatted_messages if msg["role"] != "user"]
        
        # Get response from the model
        response = await chat_with_model(query, model or "phi3:mini", context)
        
        logger.info("✅ Chat request completed successfully")
        return {"response": response}
        
    except Exception as e:
        logger.error(f"❌ Chat request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def build_chat_prompt(messages: List[ChatMessageRequest]) -> str:
    """
    Build a chat prompt from the provided messages.
    """
    prompt_parts = []
    
    for message in messages:
        if message.role == "system":
            prompt_parts.append(f"System: {message.content}")
        elif message.role == "user":
            prompt_parts.append(f"User: {message.content}")
        elif message.role == "assistant":
            prompt_parts.append(f"Assistant: {message.content}")
    
    return "\n".join(prompt_parts)

@router.post("/stream")
async def stream_chat(
    request: ChatRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """
    Stream chat completion with context from agent responses.
    """
    logger.info(f"🦙 Starting chat stream with model: {request.model or 'default'}")
    
    async def generate_chat_stream():
        try:
            # Messages are already in the correct format
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            # Yield the prompt event
            prompt_content = build_chat_prompt(request.messages)
            yield {
                "event": "prompt",
                "data": json.dumps({"prompt": prompt_content})
            }

            logger.info(f"📝 Processing {len(messages)} messages for chat")
            
            # Stream the response from Ollama
            async for chunk in ollama_service.stream_chat_completion(
                messages=messages,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ):
                # Check for done signal
                if chunk.get('done', False):
                    logger.info("✅ Chat stream completed")
                    yield {
                        "event": "done",
                        "data": json.dumps({"status": "completed"})
                    }
                    break
                
                # Extract content from message
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    logger.debug(f"📤 Streaming chunk: {content}")
                    yield {
                        "event": "token",
                        "data": json.dumps({
                            "token": content
                        })
                    }
                
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"❌ Chat stream error: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
    
    return EventSourceResponse(generate_chat_stream())

@router.post("/")
async def chat_completion(
    request: ChatRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    """
    Non-streaming chat completion.
    """
    logger.info(f"🦙 Starting chat completion with model: {request.model or 'default'}")
    
    try:
        # Messages are already in the correct format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        logger.info(f"📝 Processing {len(messages)} messages for chat")
        
        # Get the complete response
        full_response = ""
        async for chunk in ollama_service.stream_chat_completion(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        ):
            if chunk.get('done', False):
                break
            
            if 'message' in chunk and 'content' in chunk['message']:
                full_response += chunk['message']['content']
        
        logger.info("✅ Chat completion finished")
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": full_response
                }
            }]
        }
        
    except Exception as e:
        logger.error(f"❌ Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 