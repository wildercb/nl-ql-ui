"""Enhanced LLM interaction tracking service for comprehensive monitoring."""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager
from beanie import Document, Indexed

from models.query import LLMInteraction
from config.settings import get_settings

logger = logging.getLogger(__name__)


class ModelInteraction(Document):
    agent: str
    message: str
    timestamp: datetime
    interaction_id: str

    class Settings:
        name = "model_interactions"
        indexes = [
            ("timestamp", {"kind": "descending"}),
            ("interaction_id", {"kind": "ascending", "unique": True})
        ]


class LLMTrackingService:
    """Service for tracking and storing all LLM model interactions."""
    
    def __init__(self):
        self.settings = get_settings()
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
    @asynccontextmanager
    async def track_interaction(
        self,
        session_id: str,
        model: str,
        provider: str,
        interaction_type: str,
        user_id: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracking LLM interactions.
        
        Usage:
            async with tracking_service.track_interaction(
                session_id="sess_123",
                model="llama2", 
                provider="ollama",
                interaction_type="translation"
            ) as tracker:
                # Send prompt to model
                tracker.set_prompt(prompt)
                tracker.set_system_prompt(system_prompt)
                tracker.set_parameters(temperature=0.7, max_tokens=2048)
                
                # Get response from model
                response = await model_call()
                
                # Track response
                tracker.set_response(response)
                tracker.set_performance_metrics(
                    processing_time=1.5,
                    tokens_used=256,
                    confidence_score=0.85
                )
        """
        
        interaction_tracker = InteractionTracker(
            session_id=session_id,
            model=model,
            provider=provider,
            interaction_type=interaction_type,
            user_id=user_id,
            context_data=context_data or {},
            tracking_service=self
        )
        
        # Store the tracker in active sessions
        tracker_id = str(uuid.uuid4())
        self.active_sessions[tracker_id] = {
            "tracker": interaction_tracker,
            "started_at": datetime.utcnow()
        }
        
        try:
            yield interaction_tracker
        finally:
            # Save the interaction when context exits
            await self._save_interaction(interaction_tracker)
            
            # Clean up
            if tracker_id in self.active_sessions:
                del self.active_sessions[tracker_id]
    
    async def _save_interaction(self, tracker: "InteractionTracker"):
        """Save the tracked interaction to MongoDB."""
        if not self.settings.llm_tracking.enabled:
            return
        
        try:
            # Truncate prompts and responses if they're too long
            prompt = tracker.prompt or ""
            response = tracker.response or ""
            
            if len(prompt) > self.settings.llm_tracking.max_prompt_length:
                prompt = prompt[:self.settings.llm_tracking.max_prompt_length] + "...[truncated]"
            
            if len(response) > self.settings.llm_tracking.max_response_length:
                response = response[:self.settings.llm_tracking.max_response_length] + "...[truncated]"
            
            # Create LLM interaction document
            interaction = LLMInteraction(
                session_id=tracker.session_id,
                user_id=tracker.user_id,
                model=tracker.model,
                provider=tracker.provider,
                prompt=prompt if self.settings.llm_tracking.store_prompts else "[not stored]",
                response=response if self.settings.llm_tracking.store_responses else "[not stored]",
                system_prompt=tracker.system_prompt,
                processing_time=tracker.processing_time or 0.0,
                tokens_used=tracker.tokens_used,
                prompt_tokens=tracker.prompt_tokens,
                response_tokens=tracker.response_tokens,
                temperature=tracker.temperature,
                max_tokens=tracker.max_tokens,
                interaction_type=tracker.interaction_type,
                context_data=tracker.context_data,
                confidence_score=tracker.confidence_score,
                error_message=tracker.error_message,
                is_successful=tracker.is_successful,
                timestamp=tracker.started_at
            )
            
            # Save to database
            await interaction.save()
            
            logger.debug(
                f"💾 Saved LLM interaction: {tracker.provider}/{tracker.model} - {tracker.interaction_type}"
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to save LLM interaction: {e}")
    
    async def get_session_interactions(
        self, 
        session_id: str, 
        limit: int = 100
    ) -> List[LLMInteraction]:
        """Get all interactions for a specific session."""
        try:
            interactions = await LLMInteraction.find(
                LLMInteraction.session_id == session_id
            ).sort(-LLMInteraction.timestamp).limit(limit).to_list()
            
            return interactions
            
        except Exception as e:
            logger.error(f"❌ Failed to get session interactions: {e}")
            return []
    
    async def get_user_interactions(
        self, 
        user_id: str, 
        limit: int = 100
    ) -> List[LLMInteraction]:
        """Get all interactions for a specific user."""
        try:
            interactions = await LLMInteraction.find(
                LLMInteraction.user_id == user_id
            ).sort(-LLMInteraction.timestamp).limit(limit).to_list()
            
            return interactions
            
        except Exception as e:
            logger.error(f"❌ Failed to get user interactions: {e}")
            return []
    
    async def get_model_stats(self, days: int = 7) -> Dict[str, Any]:
        """Get statistics about model usage over the specified number of days."""
        try:
            from datetime import timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Aggregate statistics by model and provider
            pipeline = [
                {"$match": {"timestamp": {"$gte": cutoff_date}}},
                {
                    "$group": {
                        "_id": {
                            "model": "$model",
                            "provider": "$provider"
                        },
                        "total_interactions": {"$sum": 1},
                        "successful_interactions": {
                            "$sum": {"$cond": ["$is_successful", 1, 0]}
                        },
                        "total_processing_time": {"$sum": "$processing_time"},
                        "total_tokens": {"$sum": "$tokens_used"},
                        "avg_confidence": {"$avg": "$confidence_score"}
                    }
                },
                {
                    "$project": {
                        "model": "$_id.model",
                        "provider": "$_id.provider",
                        "total_interactions": 1,
                        "successful_interactions": 1,
                        "success_rate": {
                            "$divide": ["$successful_interactions", "$total_interactions"]
                        },
                        "avg_processing_time": {
                            "$divide": ["$total_processing_time", "$total_interactions"]
                        },
                        "total_tokens": 1,
                        "avg_confidence": 1,
                        "_id": 0
                    }
                }
            ]
            
            # Execute aggregation
            stats = await LLMInteraction.aggregate(pipeline).to_list()
            
            return {
                "period_days": days,
                "model_stats": stats,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get model stats: {e}")
            return {"error": str(e)}
    
    async def get_interaction_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about LLM interactions."""
        try:
            # Get total counts
            total_interactions = await LLMInteraction.count()
            successful_interactions = await LLMInteraction.find(
                LLMInteraction.is_successful == True
            ).count()
            
            # Get recent activity (last 24 hours)
            from datetime import timedelta
            recent_cutoff = datetime.utcnow() - timedelta(hours=24)
            recent_interactions = await LLMInteraction.find(
                LLMInteraction.timestamp >= recent_cutoff
            ).count()
            
            # Get top models
            top_models_pipeline = [
                {
                    "$group": {
                        "_id": "$model",
                        "count": {"$sum": 1},
                        "avg_processing_time": {"$avg": "$processing_time"},
                        "avg_confidence": {"$avg": "$confidence_score"}
                    }
                },
                {"$sort": {"count": -1}},
                {"$limit": 10}
            ]
            
            top_models = await LLMInteraction.aggregate(top_models_pipeline).to_list()
            
            # Get interaction types breakdown
            types_pipeline = [
                {
                    "$group": {
                        "_id": "$interaction_type",
                        "count": {"$sum": 1}
                    }
                },
                {"$sort": {"count": -1}}
            ]
            
            interaction_types = await LLMInteraction.aggregate(types_pipeline).to_list()
            
            return {
                "summary": {
                    "total_interactions": total_interactions,
                    "successful_interactions": successful_interactions,
                    "success_rate": successful_interactions / total_interactions if total_interactions > 0 else 0,
                    "recent_24h": recent_interactions
                },
                "top_models": [
                    {
                        "model": model["_id"],
                        "interactions": model["count"],
                        "avg_processing_time": model.get("avg_processing_time", 0),
                        "avg_confidence": model.get("avg_confidence", 0)
                    }
                    for model in top_models
                ],
                "interaction_types": [
                    {
                        "type": item["_id"],
                        "count": item["count"]
                    }
                    for item in interaction_types
                ],
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get interaction analytics: {e}")
            return {"error": str(e)}


class InteractionTracker:
    """Tracks a single LLM interaction."""
    
    def __init__(
        self,
        session_id: str,
        model: str,
        provider: str,
        interaction_type: str,
        user_id: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None,
        tracking_service: Optional[LLMTrackingService] = None
    ):
        self.session_id = session_id
        self.model = model
        self.provider = provider
        self.interaction_type = interaction_type
        self.user_id = user_id
        self.context_data = context_data or {}
        self.tracking_service = tracking_service
        
        # Interaction data
        self.prompt: Optional[str] = None
        self.response: Optional[str] = None
        self.system_prompt: Optional[str] = None
        
        # Performance metrics
        self.processing_time: Optional[float] = None
        self.tokens_used: Optional[int] = None
        self.prompt_tokens: Optional[int] = None
        self.response_tokens: Optional[int] = None
        
        # Model parameters
        self.temperature: Optional[float] = None
        self.max_tokens: Optional[int] = None
        
        # Quality metrics
        self.confidence_score: Optional[float] = None
        self.error_message: Optional[str] = None
        self.is_successful: bool = True
        
        # Timestamps
        self.started_at = datetime.utcnow()
    
    def set_prompt(self, prompt: str):
        """Set the prompt sent to the model."""
        self.prompt = prompt
    
    def set_response(self, response: str):
        """Set the response received from the model."""
        self.response = response
    
    def set_system_prompt(self, system_prompt: str):
        """Set the system prompt used."""
        self.system_prompt = system_prompt
    
    def set_parameters(self, temperature: Optional[float] = None, max_tokens: Optional[int] = None):
        """Set model parameters."""
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
    
    def set_performance_metrics(
        self,
        processing_time: Optional[float] = None,
        tokens_used: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        response_tokens: Optional[int] = None,
        confidence_score: Optional[float] = None
    ):
        """Set performance and quality metrics."""
        if processing_time is not None:
            self.processing_time = processing_time
        if tokens_used is not None:
            self.tokens_used = tokens_used
        if prompt_tokens is not None:
            self.prompt_tokens = prompt_tokens
        if response_tokens is not None:
            self.response_tokens = response_tokens
        if confidence_score is not None:
            self.confidence_score = confidence_score
    
    def set_error(self, error_message: str):
        """Mark the interaction as failed with an error message."""
        self.error_message = error_message
        self.is_successful = False
    
    def add_context(self, key: str, value: Any):
        """Add additional context data."""
        self.context_data[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tracker to dictionary for logging/debugging."""
        return {
            "session_id": self.session_id,
            "model": self.model,
            "provider": self.provider,
            "interaction_type": self.interaction_type,
            "user_id": self.user_id,
            "prompt_length": len(self.prompt) if self.prompt else 0,
            "response_length": len(self.response) if self.response else 0,
            "processing_time": self.processing_time,
            "tokens_used": self.tokens_used,
            "confidence_score": self.confidence_score,
            "is_successful": self.is_successful,
            "started_at": self.started_at.isoformat()
        }


# Global tracking service instance
tracking_service = LLMTrackingService()


def get_tracking_service() -> LLMTrackingService:
    """Get the global LLM tracking service instance."""
    return tracking_service 


async def track_interaction(agent: str, message: str, interaction_id: str) -> ModelInteraction:
    """
    Track a model interaction and store it in MongoDB.
    
    Args:
        agent (str): The name or identifier of the agent/model.
        message (str): The message or interaction content.
        interaction_id (str): Unique identifier for the interaction.
    
    Returns:
        ModelInteraction: The saved interaction document.
    """
    interaction = ModelInteraction(
        agent=agent,
        message=message,
        timestamp=datetime.utcnow(),
        interaction_id=interaction_id
    )
    await interaction.save()
    return interaction


async def get_live_interactions(limit: int = 10) -> List[ModelInteraction]:
    """
    Fetch the most recent model interactions from MongoDB.
    
    Args:
        limit (int): Maximum number of interactions to return.
    
    Returns:
        List[ModelInteraction]: List of recent model interactions.
    """
    return await ModelInteraction.find().sort(-ModelInteraction.timestamp).limit(limit).to_list() 