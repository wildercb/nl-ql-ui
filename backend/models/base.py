"""Base model for all MongoDB document models."""

from datetime import datetime
from typing import Optional
from beanie import Document
from pydantic import Field
import uuid as uuid_lib
from pydantic_settings import SettingsConfigDict


class BaseDocument(Document):
    """Base document with common fields for all MongoDB collections."""
    
    # MongoDB ObjectId is automatic as _id
    uuid: str = Field(default_factory=lambda: str(uuid_lib.uuid4()), json_schema_extra={"index": True, "unique": True})
    created_at: datetime = Field(default_factory=datetime.utcnow, json_schema_extra={"index": True})
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        """Beanie document settings."""
        use_state_management = True
        validate_on_save = True
        
    def save(self, *args, **kwargs):
        """Override save to update timestamp."""
        self.updated_at = datetime.utcnow()
        return super().save(*args, **kwargs)
    
    async def update(self, *args, **kwargs):
        """Override update to update timestamp."""
        self.updated_at = datetime.utcnow()
        return await super().update(*args, **kwargs)
    
    def to_dict(self) -> dict:
        """Convert document to dictionary."""
        return self.dict(by_alias=True, exclude_unset=True)
    
    model_config = SettingsConfigDict(arbitrary_types_allowed=True) 