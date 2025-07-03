"""Translation API endpoints."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import structlog
import json
import asyncio
import time
import logging
import httpx

from services.translation_service import TranslationService, TranslationResult
from config import get_settings
from services.database_service import get_database_service
from models.translation import TranslationResult
from .auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Translation"])


class TranslationRequest(BaseModel):
    """Request model for translation."""
    natural_query: str = Field(..., min_length=1, max_length=2000, description="Natural language query to translate")
    schema_context: Optional[str] = Field(None, max_length=10000, description="GraphQL schema context")
    model: Optional[str] = Field(None, description="Model to use for translation")
    session_id: Optional[str] = Field(None, description="Session ID for guest users")


class TranslationResponse(BaseModel):
    """Response model for translation."""
    graphql_query: str
    confidence: float
    explanation: str
    model: str
    processing_time: float
    warnings: List[str] = []
    suggested_improvements: List[str] = []
    
    @classmethod
    def from_result(cls, result: TranslationResult):
        return cls(
            graphql_query=result.graphql_query,
            confidence=result.confidence,
            explanation=result.explanation,
            model=result.model_used,
            processing_time=result.processing_time,
            warnings=result.warnings,
            suggested_improvements=result.suggested_improvements
        )


class BatchTranslationRequest(BaseModel):
    """Request model for batch translation."""
    queries: List[str] = Field(..., min_items=1, max_items=10, description="List of natural language queries")
    schema_context: Optional[str] = Field(None, description="GraphQL schema context")
    model: Optional[str] = Field(None, description="AI model to use")


class BatchTranslationResponse(BaseModel):
    """Response model for batch translation."""
    results: List[TranslationResponse]
    total_queries: int
    successful_translations: int
    average_confidence: float
    total_processing_time: float


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    model: Optional[str] = Field("phi3:mini", description="Model to use for chat")
    history: Optional[List[Dict[str, Any]]] = Field(None, description="Previous messages for context")


def get_translation_service() -> TranslationService:
    """Dependency to get translation service."""
    return TranslationService()


@router.post("/translate")
async def translate_query(
    request: TranslationRequest,
    current_user=Depends(get_current_user),
    db_service=Depends(get_database_service)
):
    """Translate natural language query to GraphQL."""
    try:
        translation_service = TranslationService()
        
        result = await translation_service.translate_to_graphql(
            natural_query=request.natural_query,
            schema_context=request.schema_context or "",
            model=request.model
        )
        
        # Store in database with user/session info
        from models.query import QueryLog
        
        log_entry = QueryLog(
            natural_query=request.natural_query,
            graphql_query=result.graphql_query,
            confidence=result.confidence,
            processing_time=result.processing_time,
            model_used=result.model_used,
            user_id=current_user.id if current_user else None,
            session_id=request.session_id,
            metadata={
                "explanation": result.explanation,
                "warnings": result.warnings,
                "suggested_improvements": result.suggested_improvements,
                "schema_context": request.schema_context
            }
        )
        
        await log_entry.save()
        
        return {
            "graphql_query": result.graphql_query,
            "confidence": result.confidence,
            "explanation": result.explanation,
            "processing_time": result.processing_time,
            "model_used": result.model_used,
            "warnings": result.warnings,
            "suggested_improvements": result.suggested_improvements,
            "query_id": str(log_entry.id)
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Translation service temporarily unavailable"
        )


@router.post("/translate/stream")
async def translate_query_stream(
    request: TranslationRequest,
    translation_service: TranslationService = Depends(get_translation_service)
):
    """Stream translation process with detailed step-by-step information."""
    
    async def generate_stream():
        """Generate streaming response with detailed translation steps."""
        start_time = time.time()
        
        try:
            # Step 1: Initialize
            yield f"data: {json.dumps({'step': 'init', 'message': 'Initializing translation process...', 'timestamp': time.time()})}\n\n"
            await asyncio.sleep(0.1)
            
            # Step 2: Validate input
            query_preview = request.natural_query[:50] + "..." if len(request.natural_query) > 50 else request.natural_query
            yield f"data: {json.dumps({'step': 'validate', 'message': f'Validating query: {query_preview}', 'timestamp': time.time()})}\n\n"
            await asyncio.sleep(0.1)
            
            if not request.natural_query.strip():
                yield f"data: {json.dumps({'step': 'error', 'message': 'Query cannot be empty', 'timestamp': time.time()})}\n\n"
                return
            
            # Step 3: Prepare model
            model = request.model or get_settings().ollama.default_model
            yield f"data: {json.dumps({'step': 'model', 'message': f'Using model: {model}', 'timestamp': time.time()})}\n\n"
            await asyncio.sleep(0.1)
            
            # Step 4: Build prompts
            yield f"data: {json.dumps({'step': 'prompt', 'message': 'Building system and user prompts...', 'timestamp': time.time()})}\n\n"
            await asyncio.sleep(0.1)
            
            # Step 5: Generate response
            yield f"data: {json.dumps({'step': 'generate', 'message': 'Generating GraphQL translation...', 'timestamp': time.time()})}\n\n"
            
            # Perform actual translation
            result = await translation_service.translate_to_graphql(
                natural_query=request.natural_query,
                schema_context=request.schema_context or "",
                model=model
            )
            
            # Step 6: Process response
            yield f"data: {json.dumps({'step': 'process', 'message': 'Processing and validating response...', 'timestamp': time.time()})}\n\n"
            await asyncio.sleep(0.1)
            
            # Step 7: Extract components
            yield f"data: {json.dumps({'step': 'extract', 'message': 'Extracting GraphQL query and metadata...', 'timestamp': time.time()})}\n\n"
            await asyncio.sleep(0.1)
            
            # Step 8: Finalize
            processing_time = time.time() - start_time
            yield f"data: {json.dumps({'step': 'complete', 'message': 'Translation completed successfully', 'timestamp': time.time()})}\n\n"
            
            # Final result
            final_response = {
                'step': 'result',
                'data': {
                    'graphql_query': result.graphql_query,
                    'confidence': result.confidence,
                    'explanation': result.explanation,
                    'model': result.model_used,
                    'processing_time': result.processing_time,
                    'warnings': result.warnings,
                    'suggested_improvements': result.suggested_improvements,
                    'prompt_analysis': {
                        'system_prompt': translation_service._build_system_prompt(request.schema_context or ""),
                        'user_prompt': request.natural_query,
                        'parameters': {
                            'model': model,
                            'temperature': 0.7,
                            'max_tokens': 2048
                        }
                    },
                    'response_analysis': {
                        'raw_response': result.explanation,
                        'extracted_graphql': result.graphql_query,
                        'processing_steps': [
                            'Received natural language query',
                            'Analyzed query intent and structure',
                            'Generated GraphQL syntax',
                            'Validated query structure',
                            'Extracted confidence score',
                            'Applied post-processing'
                        ]
                    }
                },
                'timestamp': time.time()
            }
            
            yield f"data: {json.dumps(final_response)}\n\n"
            
        except Exception as e:
            error_time = time.time() - start_time
            yield f"data: {json.dumps({'step': 'error', 'message': f'Translation failed: {str(e)}', 'processing_time': error_time, 'timestamp': time.time()})}\n\n"
        
        # End stream
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )


@router.post("/translate/batch", response_model=BatchTranslationResponse)
async def translate_queries_batch(
    request: BatchTranslationRequest,
    translation_service: TranslationService = Depends(get_translation_service)
):
    """Translate multiple natural language queries to GraphQL."""
    try:
        logger.info("Batch translation request", query_count=len(request.queries), model=request.model)
        
        results = await translation_service.batch_translate(
            queries=request.queries,
            schema_context=request.schema_context or "",
            model=request.model
        )
        
        responses = [TranslationResponse.from_result(result) for result in results]
        successful_count = sum(1 for r in responses if r.confidence > 0.5)
        average_confidence = sum(r.confidence for r in responses) / len(responses)
        total_processing_time = sum(r.processing_time for r in responses)
        
        logger.info(
            "Batch translation completed",
            total_queries=len(request.queries),
            successful_translations=successful_count,
            average_confidence=average_confidence
        )
        
        return BatchTranslationResponse(
            results=responses,
            total_queries=len(request.queries),
            successful_translations=successful_count,
            average_confidence=average_confidence,
            total_processing_time=total_processing_time
        )
        
    except ValueError as e:
        logger.warning("Batch translation validation error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Batch translation failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Translation service error")


@router.get("/examples")
async def get_translation_examples(
    translation_service: TranslationService = Depends(get_translation_service)
):
    """Get example translations."""
    try:
        examples = translation_service.get_translation_examples()
        return {"examples": examples}
    except Exception as e:
        logger.error("Failed to get examples", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve examples")


@router.post("/improve")
async def suggest_improvements(
    request: TranslationRequest,
    translation_service: TranslationService = Depends(get_translation_service)
):
    """Get suggestions for improving a natural language query."""
    try:
        # For now, just run translation and return suggestions
        result = await translation_service.translate_to_graphql(
            natural_query=request.natural_query,
            schema_context=request.schema_context or "",
            model=request.model
        )
        
        return {
            "original_query": request.natural_query,
            "suggestions": result.suggested_improvements,
            "warnings": result.warnings,
            "confidence": result.confidence
        }
        
    except Exception as e:
        logger.error("Failed to generate improvements", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate suggestions")


@router.post("/chat")
async def chat_with_model(
    request: ChatRequest,
    current_user=Depends(get_current_user)
):
    """Continue conversation with an AI model."""
    try:
        from services.ollama_service import OllamaService
        
        ollama_service = OllamaService()
        
        # Build conversation history
        messages = []
        
        # Add system message for GraphQL context
        messages.append({
            "role": "system",
            "content": "You are a helpful AI assistant specializing in GraphQL and data querying. "
                      "You can help with GraphQL queries, data analysis, and general programming questions. "
                      "Be concise but thorough in your responses."
        })
        
        # Add previous history if provided
        if request.history:
            for msg in request.history[-10:]:  # Last 10 messages
                if msg.get('sender') == 'user':
                    messages.append({
                        "role": "user", 
                        "content": msg.get('content', '')
                    })
                elif msg.get('sender') == 'assistant':
                    messages.append({
                        "role": "assistant", 
                        "content": msg.get('content', '')
                    })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": request.message
        })
        
        # Get AI response
        response = await ollama_service.chat_completion(
            messages=messages,
            model=request.model,
            temperature=0.7,
            max_tokens=1024
        )
        
        # Broadcast the chat interaction
        try:
            interaction_data = {
                'id': f"chat_{int(time.time() * 1000)}",
                'timestamp': time.time(),
                'model': request.model,
                'type': 'chat',
                'processing_time': 0,  # Could measure this
                'user_id': str(current_user.id) if current_user else None,
                'session_id': None,
                
                # Prompt details
                'system_prompt': messages[0]['content'],
                'user_prompt': request.message,
                'full_prompt': f"Context: {len(messages)-1} previous messages\nUser: {request.message}",
                'parameters': {
                    'temperature': 0.7,
                    'max_tokens': 1024,
                    'model': request.model
                },
                
                # Response details
                'raw_response': response.text,
                'processed_response': response.text,
                'confidence': 1.0,  # Chat doesn't have confidence scoring
                'warnings': [],
                'response_metadata': {},
                
                'status': 'completed',
                'error': None,
                'tokens_used': len(request.message.split()) + len(response.text.split()),
                'response_tokens': len(response.text.split()),
                'total_tokens': len(request.message.split()) + len(response.text.split())
            }
            
            # Broadcast to live interactions
            async with httpx.AsyncClient() as client:
                await client.post(
                    "http://localhost:8000/api/interactions/broadcast",
                    json=interaction_data,
                    timeout=5.0
                )
                
        except Exception as e:
            logger.warning(f"Failed to broadcast chat interaction: {e}")
        
        return {
            "response": response.text,
            "model": request.model,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat service temporarily unavailable"
        ) 