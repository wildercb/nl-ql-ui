"""User-related MongoDB document models."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from beanie import Document
from pydantic import Field, EmailStr
from bson import ObjectId

from .base import BaseDocument


class User(BaseDocument):
    """Document for user accounts and profiles."""
    
    # User identification
    username: str = Field(..., min_length=3, max_length=50, description="Unique username")
    email: EmailStr = Field(..., description="User email address")
    
    # Authentication
    hashed_password: str = Field(..., description="Hashed password")
    is_active: bool = Field(True, description="Whether user account is active")
    is_verified: bool = Field(False, description="Whether user email is verified")
    
    # Profile information
    full_name: Optional[str] = Field(None, max_length=200, description="User's full name")
    profile_picture_url: Optional[str] = Field(None, max_length=500, description="Profile picture URL")
    bio: Optional[str] = Field(None, description="User biography")
    
    # Preferences
    preferred_model: Optional[str] = Field(None, max_length=100, description="Preferred AI model")
    default_schema_context: Optional[str] = Field(None, description="Default GraphQL schema context")
    ui_preferences: Optional[Dict[str, Any]] = Field(None, description="UI preferences (theme, layout, etc.)")
    
    # Usage statistics
    total_queries: int = Field(0, ge=0, description="Total number of queries made")
    successful_queries: int = Field(0, ge=0, description="Number of successful queries")
    last_login_at: Optional[datetime] = Field(None, description="Last login timestamp")
    last_activity_at: Optional[datetime] = Field(None, description="Last activity timestamp")
    
    # Account metadata
    registration_source: Optional[str] = Field(None, max_length=50, description="Registration source (web, api, etc.)")
    subscription_tier: str = Field("free", description="Subscription tier (free, pro, enterprise)")
    
    class Settings:
        name = "users"
        indexes = [
            [("username", 1)],  # unique in MongoDB init script
            [("email", 1)],     # unique in MongoDB init script
            "uuid",
            "created_at"
        ]

    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"


class UserSession(BaseDocument):
    """Document for tracking user authentication sessions."""
    
    # User reference
    user_id: ObjectId = Field(..., description="Reference to the User document")
    
    # Session data
    session_token: str = Field(..., max_length=255, description="Unique session token")
    refresh_token: Optional[str] = Field(None, max_length=255, description="Refresh token for session renewal")
    
    # Session metadata
    ip_address: Optional[str] = Field(None, max_length=45, description="Client IP address (IPv6 compatible)")
    user_agent: Optional[str] = Field(None, description="Client user agent string")
    device_info: Optional[Dict[str, Any]] = Field(None, description="Device information")
    
    # Session lifecycle
    expires_at: datetime = Field(..., description="Session expiration time")
    last_activity_at: datetime = Field(default_factory=datetime.utcnow, description="Last activity timestamp")
    is_active: bool = Field(True, description="Whether session is active")
    
    # Security
    login_method: Optional[str] = Field(None, max_length=50, description="Login method (password, oauth, api_key, etc.)")
    is_persistent: bool = Field(False, description="Whether session should persist across browser restarts")
    
    class Settings:
        name = "user_sessions"
        indexes = [
            "user_id",
            [("session_token", 1)],  # unique in MongoDB init script
            "expires_at"
        ]

    def __repr__(self):
        return f"<UserSession(user_id={self.user_id}, active={self.is_active})>"


class UserAPIKey(BaseDocument):
    """Document for user API keys."""
    
    # User reference
    user_id: ObjectId = Field(..., description="Reference to the User document")
    
    # Key data
    key_name: str = Field(..., max_length=100, description="Human-readable key name")
    key_hash: str = Field(..., max_length=255, description="Hashed API key")
    key_prefix: str = Field(..., max_length=20, description="First few characters for identification")
    
    # Permissions and limits
    permissions: Optional[List[str]] = Field(None, description="List of allowed operations")
    rate_limit_per_minute: int = Field(60, ge=1, description="Rate limit per minute")
    rate_limit_per_day: int = Field(1000, ge=1, description="Rate limit per day")
    
    # Usage tracking
    usage_count: int = Field(0, ge=0, description="Number of times this key was used")
    last_used_at: Optional[datetime] = Field(None, description="Last time this key was used")
    
    # Key lifecycle
    expires_at: Optional[datetime] = Field(None, description="Key expiration time")
    is_active: bool = Field(True, description="Whether key is active")
    revoked_at: Optional[datetime] = Field(None, description="When key was revoked")
    revocation_reason: Optional[str] = Field(None, max_length=200, description="Reason for revocation")
    
    class Settings:
        name = "user_api_keys"
        indexes = [
            "user_id",
            [("key_hash", 1)],  # unique in MongoDB init script
            "is_active"
        ]

    def __repr__(self):
        return f"<UserAPIKey(name='{self.key_name}', user_id={self.user_id})>"


class UserPreferences(BaseDocument):
    """Document for detailed user preferences."""
    
    # User reference
    user_id: ObjectId = Field(..., description="Reference to the User document")
    
    # Model preferences
    preferred_models: Optional[List[str]] = Field(None, description="Ordered list of preferred models")
    model_settings: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Custom settings per model")
    
    # Query preferences
    default_temperature: float = Field(0.7, ge=0.0, le=2.0, description="Default temperature for model requests")
    default_max_tokens: int = Field(2048, ge=1, description="Default maximum tokens for responses")
    auto_validate_queries: bool = Field(True, description="Automatically validate generated queries")
    save_query_history: bool = Field(True, description="Save query history for this user")
    
    # UI preferences
    theme: str = Field("light", description="UI theme (light, dark, auto)")
    language: str = Field("en", max_length=10, description="Preferred language code")
    timezone: str = Field("UTC", max_length=50, description="User's timezone")
    date_format: str = Field("YYYY-MM-DD", max_length=20, description="Preferred date format")
    
    # Notification preferences
    email_notifications: bool = Field(True, description="Enable email notifications")
    query_completion_notifications: bool = Field(False, description="Notify on query completion")
    weekly_summary_emails: bool = Field(True, description="Send weekly summary emails")
    
    # Privacy preferences
    share_anonymous_usage_data: bool = Field(True, description="Allow anonymous usage data sharing")
    public_query_sharing: bool = Field(False, description="Allow public sharing of queries")
    
    class Settings:
        name = "user_preferences"
        indexes = [
            [("user_id", 1)]  # unique in MongoDB init script
        ]

    def __repr__(self):
        return f"<UserPreferences(user_id={self.user_id})>" 