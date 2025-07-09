"""
Enhanced LLM interaction tracking service for comprehensive monitoring.

Simplified version for compatibility without beanie dependency.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMInteractionTracker:
    """Simple tracker for LLM interactions."""
    session_id: str
    user_id: Optional[str] = None
    model: str = "unknown"
    provider: str = "unknown"
    interaction_type: str = "generation"
    started_at: Optional[datetime] = None
    prompt: str = ""
    system_prompt: Optional[str] = None
    response: str = ""
    processing_time: Optional[float] = None
    tokens_used: Optional[int] = None
    prompt_tokens: Optional[int] = None
    response_tokens: Optional[int] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    context_data: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None
    error_message: Optional[str] = None
    is_successful: bool = True
    
    def __post_init__(self):
        if self.started_at is None:
            self.started_at = datetime.utcnow()
        if self.context_data is None:
            self.context_data = {}
    
    def set_prompt(self, prompt: str):
        """Set the prompt."""
        self.prompt = prompt
    
    def set_system_prompt(self, system_prompt: str):
        """Set the system prompt."""
        self.system_prompt = system_prompt
    
    def set_response(self, response: str):
        """Set the response."""
        self.response = response
    
    def set_parameters(self, **kwargs):
        """Set model parameters."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def set_performance_metrics(self, **kwargs):
        """Set performance metrics."""
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def set_error(self, error: str):
        """Set error information."""
        self.error_message = error
        self.is_successful = False


class LLMTrackingService:
    """
    Simple LLM tracking service for backward compatibility.
    
    This provides a lightweight tracking system without external dependencies.
    """
    
    def __init__(self):
        self.interactions: List[LLMInteractionTracker] = []
        logger.info("LLMTrackingService initialized (simplified mode)")
    
    @asynccontextmanager
    async def track_interaction(
        self,
        session_id: str,
        model: str = "unknown",
        provider: str = "unknown",
        interaction_type: str = "generation",
        user_id: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None
    ):
        """Context manager for tracking LLM interactions."""
        tracker = LLMInteractionTracker(
            session_id=session_id,
            model=model,
            provider=provider,
            interaction_type=interaction_type,
            user_id=user_id,
            context_data=context_data or {}
        )
        
        try:
            yield tracker
            
            # Log successful interaction
            logger.debug(
                f"Tracked interaction: {provider}/{model} - {interaction_type} - "
                f"Success: {tracker.is_successful}"
            )
            
            # Store in memory (could be persisted to database later)
            self.interactions.append(tracker)
            
        except Exception as e:
            tracker.set_error(str(e))
            logger.error(f"Error in tracked interaction: {e}")
            self.interactions.append(tracker)
            raise
    
    def get_recent_interactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent interactions."""
        recent = self.interactions[-limit:] if self.interactions else []
        return [
            {
                "session_id": interaction.session_id,
                "model": interaction.model,
                "provider": interaction.provider,
                "interaction_type": interaction.interaction_type,
                "started_at": interaction.started_at.isoformat(),
                "is_successful": interaction.is_successful,
                "processing_time": interaction.processing_time,
                "tokens_used": interaction.tokens_used
            }
            for interaction in recent
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        total = len(self.interactions)
        successful = sum(1 for i in self.interactions if i.is_successful)
        
        return {
            "total_interactions": total,
            "successful_interactions": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "providers": list(set(i.provider for i in self.interactions)),
            "models": list(set(i.model for i in self.interactions))
        }


# Global service instance
_tracking_service: Optional[LLMTrackingService] = None


def get_tracking_service() -> LLMTrackingService:
    """Get the global tracking service instance."""
    global _tracking_service
    if _tracking_service is None:
        _tracking_service = LLMTrackingService()
    return _tracking_service 


# Backward compatibility functions
async def track_interaction(agent: str, message: str, interaction_id: str) -> Dict[str, Any]:
    """
    Track a model interaction for backward compatibility.
    
    Args:
        agent: The name or identifier of the agent/model
        message: The message or interaction content
        interaction_id: Unique identifier for the interaction
    
    Returns:
        Dict with interaction data
    """
    try:
        tracking_service = get_tracking_service()
        
        # Create a mock interaction record
        interaction_data = {
            "agent": agent,
            "message": message,
            "timestamp": datetime.utcnow(),
            "interaction_id": interaction_id
        }
        
        logger.debug(f"Tracked interaction: {agent} - {interaction_id}")
        return interaction_data
        
    except Exception as e:
        logger.error(f"Failed to track interaction: {e}")
        return {
            "agent": agent,
            "message": message,
            "timestamp": datetime.utcnow(),
            "interaction_id": interaction_id,
            "error": str(e)
        }


async def get_live_interactions(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent interactions for backward compatibility.
    
    Args:
        limit: Maximum number of interactions to return
        
    Returns:
        List of recent interactions
    """
    try:
        tracking_service = get_tracking_service()
        return tracking_service.get_recent_interactions(limit)
        
    except Exception as e:
        logger.error(f"Failed to get live interactions: {e}")
        return [] 