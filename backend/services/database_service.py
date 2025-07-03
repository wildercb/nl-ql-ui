"""MongoDB database connection and management service."""

import asyncio
import logging
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie, PydanticObjectId

from config.settings import get_settings
from models import (
    Query, QueryResult, QuerySession, QueryFeedback, LLMInteraction,
    User, UserSession, UserAPIKey, UserPreferences
)
from models.query import QueryLog

logger = logging.getLogger(__name__)


class DatabaseService:
    """Service for managing MongoDB database connections and operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client: AsyncIOMotorClient = None
        self.database = None
        
    async def initialize(self):
        """Initialize the database connection and Beanie ODM."""
        try:
            logger.info(f"🔌 Connecting to MongoDB at {self.settings.database.url}")
            
            # Create MongoDB client
            self.client = AsyncIOMotorClient(
                self.settings.database.url,
                minPoolSize=self.settings.database.min_connections,
                maxPoolSize=self.settings.database.max_connections,
            )
            
            # Get database
            self.database = self.client[self.settings.database.database]
            
            # Initialize Beanie with all document models
            await init_beanie(
                database=self.database,
                document_models=[
                    # Query-related models
                    Query,
                    QueryResult, 
                    QuerySession,
                    QueryFeedback,
                    LLMInteraction,
                    QueryLog,
                    # User-related models
                    User,
                    UserSession,
                    UserAPIKey,
                    UserPreferences
                ]
            )
            
            logger.info("✅ MongoDB connection and Beanie ODM initialized successfully")
            
            # Test the connection
            await self.health_check()
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MongoDB connection: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check database connection health."""
        try:
            # Ping the database
            result = await self.client.admin.command('ping')
            if result.get('ok') == 1:
                logger.debug("📊 MongoDB health check passed")
                return True
            else:
                logger.error("❌ MongoDB health check failed")
                return False
        except Exception as e:
            logger.error(f"❌ MongoDB health check error: {e}")
            return False
    
    async def close(self):
        """Close the database connection."""
        if self.client:
            logger.info("🔒 Closing MongoDB connection")
            self.client.close()
            self.client = None
            self.database = None
    
    async def get_stats(self) -> dict:
        """Get database statistics."""
        try:
            db_stats = await self.database.command("dbStats")
            
            # Get collection stats
            collection_stats = {}
            collections = await self.database.list_collection_names()
            
            for collection_name in collections:
                try:
                    stats = await self.database.command("collStats", collection_name)
                    collection_stats[collection_name] = {
                        "count": stats.get("count", 0),
                        "size": stats.get("size", 0),
                        "avgObjSize": stats.get("avgObjSize", 0),
                        "indexes": stats.get("nindexes", 0)
                    }
                except Exception:
                    # Skip collections that don't support collStats
                    pass
            
            return {
                "database": {
                    "name": db_stats.get("db"),
                    "collections": db_stats.get("collections", 0),
                    "dataSize": db_stats.get("dataSize", 0),
                    "storageSize": db_stats.get("storageSize", 0),
                    "indexes": db_stats.get("indexes", 0),
                    "indexSize": db_stats.get("indexSize", 0)
                },
                "collections": collection_stats,
                "connection": {
                    "url": self.settings.database.url,
                    "database": self.settings.database.database,
                    "minConnections": self.settings.database.min_connections,
                    "maxConnections": self.settings.database.max_connections
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get database stats: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_data(self):
        """Clean up old data based on retention policies."""
        try:
            from datetime import datetime, timedelta
            
            # Clean up old LLM interactions (based on retention days)
            retention_cutoff = datetime.utcnow() - timedelta(
                days=self.settings.llm_tracking.retention_days
            )
            
            # Delete old LLM interactions
            delete_result = await LLMInteraction.find(
                LLMInteraction.timestamp < retention_cutoff
            ).delete()
            
            if delete_result.deleted_count > 0:
                logger.info(f"🧹 Cleaned up {delete_result.deleted_count} old LLM interactions")
            
            # Clean up expired user sessions
            expired_sessions = await UserSession.find(
                UserSession.expires_at < datetime.utcnow(),
                UserSession.is_active == True
            ).update({"$set": {"is_active": False}})
            
            logger.info("✅ Database cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Database cleanup failed: {e}")


# Global database service instance
db_service = DatabaseService()


async def get_database_service() -> DatabaseService:
    """Get the global database service instance."""
    return db_service


async def initialize_database():
    """Initialize the database connection."""
    await db_service.initialize()


async def close_database():
    """Close the database connection."""
    await db_service.close()


async def get_query_logs(
    limit: int = 100, 
    skip: int = 0, 
    user_id: Optional[str] = None, 
    session_id: Optional[str] = None
) -> List[QueryLog]:
    """
    Retrieve query logs from MongoDB with optional filtering.
    
    Args:
        limit (int): Maximum number of logs to return
        skip (int): Number of logs to skip (for pagination)
        user_id (Optional[str]): Filter by user ID for authenticated users
        session_id (Optional[str]): Filter by session ID for guest users
    
    Returns:
        List[QueryLog]: List of query log documents
    """
    try:
        # Build query filter
        query_filter = {}
        
        if user_id:
            try:
                from bson import ObjectId
                # Try to convert user_id to ObjectId, but handle if it's already ObjectId
                if isinstance(user_id, str):
                    query_filter["user_id"] = ObjectId(user_id)
                else:
                    query_filter["user_id"] = user_id
            except Exception as e:
                logger.error(f"Invalid user_id format: {user_id}, error: {e}")
                # Return empty list if user_id is invalid
                return []
        elif session_id:
            # For guest sessions, session_id is stored as a string
            query_filter["session_id"] = session_id
        
        logger.info(f"Querying QueryLog with filter: {query_filter}")
        
        # Execute query with pagination and sorting
        query_logs = await QueryLog.find(query_filter) \
            .sort([("timestamp", -1)]) \
            .skip(skip) \
            .limit(limit) \
            .to_list()
        
        logger.info(f"Retrieved {len(query_logs)} query logs")
        return query_logs
        
    except Exception as e:
        logger.error(f"Error retrieving query logs: {e}")
        # Return empty list instead of raising exception to prevent 500 errors
        return [] 