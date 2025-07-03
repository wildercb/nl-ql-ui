from fastapi import APIRouter, Response, Depends, BackgroundTasks
import asyncio
import json
import time
from sse_starlette.sse import EventSourceResponse
from typing import List, Dict, Any, Optional

from services.llm_tracking_service import get_tracking_service, LLMTrackingService
from models.query import LLMInteraction

router = APIRouter(prefix='/api', tags=['interactions'])

# Global store for live interactions
live_interactions_store: List[Dict[str, Any]] = []
active_connections: List[asyncio.Queue] = []

@router.get('/interactions/stream')
async def stream_interactions():
    """
    Stream live model interactions with full prompt and response details using Server-Sent Events (SSE).
    """
    
    # Create a queue for this connection
    connection_queue = asyncio.Queue()
    active_connections.append(connection_queue)
    
    async def event_generator():
        try:
            while True:
                try:
                    # Wait for new interaction data with timeout
                    interaction = await asyncio.wait_for(
                        connection_queue.get(), 
                        timeout=30.0  # 30 second timeout for keep-alive
                    )
                    
                    if interaction:
                        yield {
                            'event': 'interaction',
                            'data': json.dumps(interaction)
                        }
                    
                except asyncio.TimeoutError:
                    # Send keep-alive ping
                    yield {
                        'event': 'ping',
                        'data': json.dumps({'timestamp': time.time(), 'status': 'alive'})
                    }
                
                except Exception as e:
                    print(f"Error in event generator: {e}")
                    break
                    
        finally:
            # Clean up connection
            if connection_queue in active_connections:
                active_connections.remove(connection_queue)
    
    return EventSourceResponse(event_generator())


@router.post('/interactions/broadcast')
async def broadcast_interaction(
    interaction_data: Dict[str, Any],
    tracking_service: LLMTrackingService = Depends(get_tracking_service)
):
    """
    Broadcast a new interaction to all connected clients.
    This endpoint is called internally when new model interactions occur.
    """
    
    # Enhance interaction data with additional metadata
    enhanced_interaction = {
        'id': interaction_data.get('id', str(time.time())),
        'timestamp': interaction_data.get('timestamp', time.time()),
        'model': interaction_data.get('model', 'unknown'),
        'interaction_type': interaction_data.get('type', 'translation'),
        'processing_time': interaction_data.get('processing_time', 0),
        'user_id': interaction_data.get('user_id'),
        'session_id': interaction_data.get('session_id'),
        
        # Full prompt details
        'prompt_data': {
            'system_prompt': interaction_data.get('system_prompt', ''),
            'user_prompt': interaction_data.get('user_prompt', ''),
            'full_prompt': interaction_data.get('full_prompt', ''),
            'parameters': interaction_data.get('parameters', {})
        },
        
        # Full response details
        'response_data': {
            'raw_response': interaction_data.get('raw_response', ''),
            'processed_response': interaction_data.get('processed_response', ''),
            'confidence': interaction_data.get('confidence', 0),
            'warnings': interaction_data.get('warnings', []),
            'metadata': interaction_data.get('response_metadata', {})
        },
        
        # Status and metrics
        'status': interaction_data.get('status', 'completed'),
        'error': interaction_data.get('error'),
        'metrics': {
            'tokens_used': interaction_data.get('tokens_used', 0),
            'response_tokens': interaction_data.get('response_tokens', 0),
            'total_tokens': interaction_data.get('total_tokens', 0)
        }
    }
    
    # Add to global store (keep last 50)
    live_interactions_store.append(enhanced_interaction)
    if len(live_interactions_store) > 50:
        live_interactions_store.pop(0)
    
    # Save to tracking service
    try:
        await tracking_service.track_interaction(
            model=enhanced_interaction['model'],
            prompt_text=enhanced_interaction['prompt_data']['full_prompt'],
            response_text=enhanced_interaction['response_data']['raw_response'],
            processing_time_ms=enhanced_interaction['processing_time'] * 1000,
            metadata=enhanced_interaction
        )
    except Exception as e:
        print(f"Error saving interaction to tracking service: {e}")
    
    # Broadcast to all connected clients
    for connection_queue in active_connections:
        try:
            await connection_queue.put(enhanced_interaction)
        except Exception as e:
            print(f"Error broadcasting to connection: {e}")
    
    return {"status": "broadcasted", "connections": len(active_connections)}


@router.get('/interactions/recent')
async def get_recent_interactions(
    limit: int = 10,
    tracking_service: LLMTrackingService = Depends(get_tracking_service)
):
    """
    Get recent model interactions for initial page load.
    """
    
    # Return recent interactions from store
    recent = live_interactions_store[-limit:] if live_interactions_store else []
    
    # If store is empty, try to get from database
    if not recent:
        try:
            db_interactions = await LLMInteraction.find().sort("-timestamp").limit(limit).to_list()
            recent = [
                {
                    'id': str(interaction.id),
                    'timestamp': interaction.timestamp.timestamp(),
                    'model': interaction.model_name,
                    'interaction_type': 'database_record',
                    'processing_time': interaction.processing_time_ms / 1000,
                    'user_id': str(interaction.user_id) if interaction.user_id else None,
                    'session_id': interaction.session_id,
                    'prompt_data': {
                        'system_prompt': interaction.metadata.get('system_prompt', ''),
                        'user_prompt': interaction.prompt_text[:200] + '...' if len(interaction.prompt_text) > 200 else interaction.prompt_text,
                        'full_prompt': interaction.prompt_text,
                        'parameters': interaction.metadata.get('parameters', {})
                    },
                    'response_data': {
                        'raw_response': interaction.response_text,
                        'processed_response': interaction.metadata.get('processed_response', ''),
                        'confidence': interaction.metadata.get('confidence', 0),
                        'warnings': interaction.metadata.get('warnings', []),
                        'metadata': interaction.metadata or {}
                    },
                    'status': 'completed',
                    'error': None,
                    'metrics': {
                        'tokens_used': interaction.tokens_used or 0,
                        'response_tokens': interaction.metadata.get('response_tokens', 0),
                        'total_tokens': interaction.tokens_used or 0
                    }
                }
                for interaction in db_interactions
            ]
        except Exception as e:
            print(f"Error fetching interactions from database: {e}")
            recent = []
    
    return {"interactions": recent, "total": len(recent)}


@router.get('/interactions/stats')
async def get_interaction_stats(
    tracking_service: LLMTrackingService = Depends(get_tracking_service)
):
    """
    Get statistics about model interactions.
    """
    
    try:
        # Get analytics from tracking service
        analytics = await tracking_service.get_interaction_analytics()
        
        # Add live stats
        live_stats = {
            'active_connections': len(active_connections),
            'live_interactions_count': len(live_interactions_store),
            'recent_activity': len([
                i for i in live_interactions_store 
                if time.time() - i['timestamp'] < 300  # Last 5 minutes
            ])
        }
        
        return {
            'analytics': analytics,
            'live_stats': live_stats,
            'status': 'healthy'
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'live_stats': {
                'active_connections': len(active_connections),
                'live_interactions_count': len(live_interactions_store),
                'recent_activity': 0
            },
            'status': 'error'
        }


# Function to simulate interactions for testing
async def simulate_interaction(
    model: str = "phi3:mini",
    user_prompt: str = "Test query",
    processing_time: float = 1.5
):
    """
    Simulate a model interaction for testing purposes.
    This can be called from translation service to broadcast real interactions.
    """
    
    interaction_data = {
        'id': f"sim_{int(time.time())}",
        'timestamp': time.time(),
        'model': model,
        'type': 'translation',
        'processing_time': processing_time,
        'user_id': None,
        'session_id': 'simulation',
        
        # Prompt details
        'system_prompt': "You are an expert GraphQL query generator. Convert natural language to GraphQL.",
        'user_prompt': user_prompt,
        'full_prompt': f"System: You are an expert GraphQL query generator.\nUser: {user_prompt}",
        'parameters': {
            'temperature': 0.7,
            'max_tokens': 1024,
            'model': model
        },
        
        # Response details  
        'raw_response': f"Generated GraphQL query for: {user_prompt}",
        'processed_response': "query { users { id name } }",
        'confidence': 0.85,
        'warnings': [],
        'response_metadata': {},
        
        'status': 'completed',
        'error': None,
        'tokens_used': 150,
        'response_tokens': 50,
        'total_tokens': 200
    }
    
    # Broadcast the simulated interaction
    for connection_queue in active_connections:
        try:
            await connection_queue.put(interaction_data)
        except Exception as e:
            print(f"Error broadcasting simulated interaction: {e}")


@router.post('/interactions/simulate')
async def trigger_simulation(
    background_tasks: BackgroundTasks,
    model: str = "phi3:mini",
    prompt: str = "Show me all users"
):
    """
    Trigger a simulated interaction for testing.
    """
    
    background_tasks.add_task(simulate_interaction, model, prompt, 2.0)
    
    return {
        "message": "Simulation triggered",
        "model": model,
        "prompt": prompt,
        "active_connections": len(active_connections)
    } 