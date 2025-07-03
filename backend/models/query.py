"""Query-related MongoDB document models."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from beanie import Document, Link
from pydantic import Field
from bson import ObjectId
from pydantic_settings import SettingsConfigDict

from .base import BaseDocument


class Query(BaseDocument):
    """Document for storing natural language queries and their translations."""
    
    # Query content
    natural_query: str = Field(..., description="Original natural language query")
    graphql_query: str = Field(..., description="Generated GraphQL query")
    schema_context: Optional[str] = Field(None, description="Schema context used for translation")
    
    # Translation metadata
    model_used: str = Field(..., description="AI model used for translation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of translation")
    explanation: Optional[str] = Field(None, description="Explanation of the translation")
    processing_time: float = Field(..., ge=0.0, description="Time taken to process in seconds")
    
    # Quality metrics
    suggested_improvements: Optional[Dict[str, Any]] = Field(None, description="Suggested improvements")
    warnings: Optional[List[str]] = Field(None, description="Translation warnings")
    
    # Validation results
    is_valid: bool = Field(True, description="Whether the GraphQL query is valid")
    validation_errors: Optional[List[str]] = Field(None, description="Validation errors")
    validation_warnings: Optional[List[str]] = Field(None, description="Validation warnings")
    
    # Usage tracking
    usage_count: int = Field(0, ge=0, description="Number of times this query was used")
    last_used_at: Optional[datetime] = Field(None, description="Last time this query was used")
    
    # User association (optional, for future user features)
    user_id: Optional[ObjectId] = Field(None, description="User who created this query")
    
    class Settings:
        name = "queries"
        indexes = [
            "uuid",
            "user_id",
            "model_used",
            [("confidence", -1)],
            [("created_at", -1)],
            [("natural_query", "text"), ("graphql_query", "text")],
            [("user_id", 1), ("created_at", -1)],
            [("model_used", 1), ("confidence", -1)]
        ]

    def __repr__(self):
        return f"<Query(uuid='{self.uuid}', natural_query='{self.natural_query[:50]}...')>"

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)


class QueryResult(BaseDocument):
    """Document for storing query execution results."""
    
    # Reference to the query
    query_id: ObjectId = Field(..., description="Reference to the Query document")
    
    # Execution details
    executed_query: str = Field(..., description="The actual query executed")
    endpoint_url: Optional[str] = Field(None, description="GraphQL endpoint URL")
    execution_time: Optional[float] = Field(None, ge=0.0, description="Query execution time in seconds")
    
    # Result data
    result_data: Optional[Dict[str, Any]] = Field(None, description="The actual GraphQL response")
    error_message: Optional[str] = Field(None, description="Error message if execution failed")
    is_successful: bool = Field(True, description="Whether the execution was successful")
    
    # Metadata
    headers_used: Optional[Dict[str, str]] = Field(None, description="HTTP headers used in request")
    variables_used: Optional[Dict[str, Any]] = Field(None, description="GraphQL variables used")
    
    class Settings:
        name = "query_results"
        indexes = [
            "query_id",
            "is_successful",
            [("created_at", -1)]
        ]

    def __repr__(self):
        return f"<QueryResult(uuid='{self.uuid}', successful={self.is_successful})>"

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)


class QuerySession(BaseDocument):
    """Document for tracking query sessions and conversations."""
    
    # Session identification
    session_name: Optional[str] = Field(None, max_length=200, description="Human-readable session name")
    
    # Session metadata
    total_queries: int = Field(0, ge=0, description="Total queries in this session")
    successful_queries: int = Field(0, ge=0, description="Number of successful queries")
    total_processing_time: float = Field(0.0, ge=0.0, description="Total processing time")
    average_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Average confidence score")
    
    # Session context
    schema_context: Optional[str] = Field(None, description="Schema used throughout the session")
    preferred_model: Optional[str] = Field(None, description="Preferred AI model for this session")
    
    # User association
    user_id: Optional[ObjectId] = Field(None, description="User who owns this session")
    
    # Session lifecycle
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Session start time")
    ended_at: Optional[datetime] = Field(None, description="Session end time")
    is_active: bool = Field(True, description="Whether session is currently active")
    
    class Settings:
        name = "query_sessions"
        indexes = [
            "user_id",
            "is_active",
            [("created_at", -1)]
        ]

    def __repr__(self):
        return f"<QuerySession(uuid='{self.uuid}', name='{self.session_name}')>"

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)


class QueryFeedback(BaseDocument):
    """Document for storing user feedback on query translations."""
    
    # References
    query_id: ObjectId = Field(..., description="Reference to the Query document")
    user_id: Optional[ObjectId] = Field(None, description="User who provided feedback")
    
    # Feedback data
    rating: Optional[int] = Field(None, ge=1, le=5, description="Overall rating (1-5 scale)")
    is_helpful: Optional[bool] = Field(None, description="Whether the translation was helpful")
    feedback_text: Optional[str] = Field(None, description="Detailed feedback text")
    
    # Specific feedback categories
    translation_accuracy: Optional[int] = Field(None, ge=1, le=5, description="Translation accuracy (1-5 scale)")
    query_efficiency: Optional[int] = Field(None, ge=1, le=5, description="Query efficiency (1-5 scale)")
    explanation_clarity: Optional[int] = Field(None, ge=1, le=5, description="Explanation clarity (1-5 scale)")
    
    # Suggestions for improvement
    suggested_query: Optional[str] = Field(None, description="User's suggested improved query")
    improvement_notes: Optional[str] = Field(None, description="Notes on how to improve")
    
    class Settings:
        name = "query_feedback"
        indexes = [
            "query_id",
            "user_id",
            [("created_at", -1)]
        ]

    def __repr__(self):
        return f"<QueryFeedback(query_id={self.query_id}, rating={self.rating})>"

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)


class LLMInteraction(BaseDocument):
    """Document for tracking all LLM model interactions for analysis and debugging."""
    
    # Session tracking
    session_id: str = Field(..., description="Session ID to group related interactions")
    user_id: Optional[ObjectId] = Field(None, description="User who initiated the interaction")
    
    # Model information
    model: str = Field(..., description="AI model used (e.g., 'llama2', 'gpt-4')")
    provider: str = Field(..., description="Model provider (e.g., 'ollama', 'openai')")
    
    # Interaction data
    prompt: str = Field(..., description="Full prompt sent to the model")
    response: str = Field(..., description="Full response from the model")
    system_prompt: Optional[str] = Field(None, description="System prompt if used")
    
    # Performance metrics
    processing_time: float = Field(..., ge=0.0, description="Time taken to process in seconds")
    tokens_used: Optional[int] = Field(None, ge=0, description="Total tokens used")
    prompt_tokens: Optional[int] = Field(None, ge=0, description="Tokens in prompt")
    response_tokens: Optional[int] = Field(None, ge=0, description="Tokens in response")
    
    # Model parameters
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature parameter")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens parameter")
    
    # Context and metadata
    interaction_type: str = Field(..., description="Type of interaction (translation, validation, etc.)")
    context_data: Optional[Dict[str, Any]] = Field(None, description="Additional context data")
    
    # Quality metrics
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence in the response")
    error_message: Optional[str] = Field(None, description="Error message if interaction failed")
    is_successful: bool = Field(True, description="Whether the interaction was successful")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the interaction occurred")
    
    class Settings:
        name = "llm_interactions"
        indexes = [
            "session_id",
            "user_id", 
            "model",
            "provider",
            "interaction_type",
            [("timestamp", -1)],
            [("session_id", 1), ("timestamp", -1)],
            [("model", 1), ("timestamp", -1)],
            [("user_id", 1), ("timestamp", -1)]
        ]

    def __repr__(self):
        return f"<LLMInteraction(model='{self.model}', type='{self.interaction_type}', tokens={self.tokens_used})>"

    model_config = SettingsConfigDict(arbitrary_types_allowed=True) 


class QueryLog(BaseDocument):
    """Document for logging queries, compatible with existing references."""
    
    # Query content
    natural_query: str = Field(..., description="Original natural language query")
    graphql_query: Optional[str] = Field(None, description="Generated GraphQL query")
    
    # Metadata
    model_used: Optional[str] = Field(None, description="AI model used for translation")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score of translation")
    processing_time: Optional[float] = Field(None, ge=0.0, description="Time taken to process in seconds")
    
    # Status
    is_successful: bool = Field(False, description="Whether the translation was successful")
    error_message: Optional[str] = Field(None, description="Error message if translation failed")
    
    # User and session association
    user_id: Optional[ObjectId] = Field(None, description="User who created this query log")
    session_id: Optional[str] = Field(None, description="Session ID for guest users")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the query was logged")
    
    class Settings:
        name = "query_logs"
        indexes = [
            "uuid",
            "user_id",
            "session_id",
            [("timestamp", -1)],
            [("user_id", 1), ("timestamp", -1)],
            [("session_id", 1), ("timestamp", -1)]
        ]

    def __repr__(self):
        return f"<QueryLog(uuid='{self.uuid}', natural_query='{self.natural_query[:50]}...')>"

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)


class ChatMessage(BaseDocument):
    """Document for storing chat messages in conversations."""
    
    # Message identification
    conversation_id: str = Field(..., description="ID of the conversation this message belongs to")
    user_id: Optional[ObjectId] = Field(None, description="User who sent the message")
    
    # Message content
    sender: str = Field(..., description="Who sent the message ('user', 'assistant', 'system')")
    content: str = Field(..., description="The message content")
    
    # Message metadata
    model_used: Optional[str] = Field(None, description="AI model used for assistant responses")
    processing_time: Optional[float] = Field(None, ge=0.0, description="Time taken to generate response")
    
    # Context and relationships
    parent_message_id: Optional[ObjectId] = Field(None, description="ID of the message this is replying to")
    related_query_id: Optional[ObjectId] = Field(None, description="Related query if this message is about a specific translation")
    
    # Message state
    is_edited: bool = Field(False, description="Whether this message has been edited")
    edit_count: int = Field(0, ge=0, description="Number of times this message has been edited")
    
    # Timestamp
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the message was sent")
    
    class Settings:
        name = "chat_messages"
        indexes = [
            "conversation_id",
            "user_id",
            "sender",
            [("conversation_id", 1), ("timestamp", 1)],
            [("user_id", 1), ("timestamp", -1)],
            [("timestamp", -1)]
        ]

    def __repr__(self):
        return f"<ChatMessage(conversation='{self.conversation_id}', sender='{self.sender}', content='{self.content[:50]}...')>"

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)