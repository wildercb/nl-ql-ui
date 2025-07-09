"""Main FastAPI application with unified architecture - NO DEFAULT MODELS."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

# Import unified architecture components
from services.database_service import initialize_database, close_database

from .middleware import (
    RateLimitMiddleware,
    LoggingMiddleware,
    SecurityMiddleware,
    CacheMiddleware
)
from .routes import health, translation, validation, models, analytics, interactions, history, chat, auth, data_query, multiagent, content_seed, mcp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    
    # Startup
    logger.info("🚀 Starting MPPW MCP API with Unified Architecture...")
    logger.info("📝 Models will be selected via UI - NO default models configured")
    
    try:
        # Initialize MongoDB connection
        await initialize_database()
        logger.info("✅ MongoDB connection initialized")
        
        # Initialize unified architecture services
        try:
            from services.unified_providers import get_provider_service
            provider_service = get_provider_service()
            logger.info("🔌 Provider service initialized")
        except Exception as e:
            logger.warning(f"⚠️ Provider service not yet configured: {e}")
        
        try:
            from mcp_server.tools.unified_tools import get_tool_registry
            tool_registry = get_tool_registry()
            tools = tool_registry.list_tools()
            logger.info(f"🛠️ Tool registry initialized with {len(tools)} tools")
        except Exception as e:
            logger.warning(f"⚠️ Tool registry not yet configured: {e}")
        
        logger.info("🎉 API startup complete - ready for model selection via UI!")
        
    except Exception as e:
        logger.error(f"❌ Failed to start API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down MPPW MCP API...")
    
    try:
        # Close MongoDB connection
        await close_database()
        logger.info("🔒 MongoDB connection closed")
        
        logger.info("✅ API shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="MPPW-MCP Unified API",
    description="Multi-Agent Pipeline Processing with Model Context Protocol - Unified Architecture (Models selected via UI)",
    version="2.0.0",
    debug=True,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add custom middleware
app.add_middleware(RateLimitMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(SecurityMiddleware)
app.add_middleware(CacheMiddleware)

# Include API routes
app.include_router(health.router)
app.include_router(translation.router)
app.include_router(validation.router)
app.include_router(models.router)
app.include_router(analytics.router)
app.include_router(interactions.router)
app.include_router(history.router)
app.include_router(chat.router)
app.include_router(auth.router)
app.include_router(data_query.router)
app.include_router(multiagent.router)
app.include_router(content_seed.router)
app.include_router(mcp.router)

logger.info(f"📍 API routes registered: {len(app.routes)} endpoints")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "type": type(exc).__name__,
            "architecture": "unified"
        }
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    
    return {
        "name": "MPPW-MCP Unified API",
        "version": "2.0.0",
        "description": "Multi-Agent Pipeline Processing with Model Context Protocol - Unified Architecture",
        "status": "healthy",
        "architecture": "unified",
        "database": "MongoDB",
        "model_selection": "UI-based (no defaults)",
        "features": [
            "Unified Architecture",
            "Type-Safe Configuration",
            "Multi-Agent Pipeline Processing",
            "Streamlined Provider System",
            "MCP Tool Integration",
            "Real-time Analytics and Monitoring",
            "Natural Language to GraphQL Translation",
            "Query Validation and Optimization",
            "UI-Based Model Selection"
        ],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "translation": "/translation",
            "validation": "/validation",
            "models": "/models",
            "analytics": "/analytics",
            "multiagent": "/multiagent",
            "mcp": "/mcp"
        }
    }


@app.get("/info")
async def get_api_info():
    """Get detailed API and system information."""
    
    try:
        # Get database stats
        from services.database_service import get_database_service
        db_service = await get_database_service()
        db_stats = await db_service.get_stats()
        
        # Try to get unified architecture info if available
        unified_info = {
            "providers": "configured via UI",
            "models": "selected via UI", 
            "agents": "available",
            "tools": "available"
        }
        
        try:
            from services.unified_providers import get_provider_service
            provider_service = get_provider_service()
            unified_info["providers"] = "service initialized"
        except Exception:
            pass
            
        try:
            from mcp_server.tools.unified_tools import get_tool_registry
            tool_registry = get_tool_registry()
            tools = tool_registry.list_tools()
            unified_info["tools"] = f"{len(tools)} registered"
        except Exception:
            pass
        
        return {
            "api": {
                "title": "MPPW-MCP Unified API",
                "version": "2.0.0",
                "architecture": "unified",
                "debug": True
            },
            "unified_architecture": unified_info,
            "database": {
                "type": "MongoDB",
                "status": "connected" if db_stats.get("database") else "disconnected",
                "collections": db_stats.get("database", {}).get("collections", 0),
                "total_documents": sum(
                    collection.get("count", 0) 
                    for collection in db_stats.get("collections", {}).values()
                )
            },
            "system": {
                "unified_architecture": True,
                "type_safety": True,
                "modular_design": True,
                "extensible": True,
                "model_selection": "UI-based"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting API info: {e}")
        return {
            "error": "Failed to get system information",
            "message": str(e)
        } 