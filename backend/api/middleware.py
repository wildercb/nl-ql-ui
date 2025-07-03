"""Custom middleware for the API."""

import time
import asyncio
from typing import Dict, Optional
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import structlog

from config import get_settings

logger = structlog.get_logger()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app):
        super().__init__(app)
        self.settings = get_settings()
        self.requests: Dict[str, list] = {}
        self.lock = asyncio.Lock()
    
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting to requests."""
        # Get client identifier (IP address)
        client_ip = request.client.host if request.client else "unknown"
        
        # Skip rate limiting for health checks
        if request.url.path.startswith("/health"):
            return await call_next(request)
        
        current_time = time.time()
        window_start = current_time - 60  # 1-minute window
        
        async with self.lock:
            # Clean old requests
            if client_ip in self.requests:
                self.requests[client_ip] = [
                    req_time for req_time in self.requests[client_ip]
                    if req_time > window_start
                ]
            else:
                self.requests[client_ip] = []
            
            # Check rate limit
            if len(self.requests[client_ip]) >= self.settings.rate_limit.requests_per_minute:
                logger.warning(
                    "Rate limit exceeded",
                    client_ip=client_ip,
                    requests_count=len(self.requests[client_ip])
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "retry_after": 60,
                        "requests_per_minute": self.settings.rate_limit.requests_per_minute
                    }
                )
            
            # Add current request
            self.requests[client_ip].append(current_time)
        
        return await call_next(request)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware."""
    
    async def dispatch(self, request: Request, call_next):
        """Log requests and responses."""
        start_time = time.time()
        
        # Extract request information
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            client_ip=client_ip,
            user_agent=user_agent
        )
        
        try:
            response = await call_next(request)
            processing_time = time.time() - start_time
            
            logger.info(
                "Request completed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                processing_time=processing_time,
                client_ip=client_ip
            )
            
            # Add custom headers
            response.headers["X-Processing-Time"] = str(processing_time)
            response.headers["X-Request-ID"] = f"{int(start_time * 1000)}"
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            logger.error(
                "Request failed",
                method=request.method,
                url=str(request.url),
                error=str(e),
                processing_time=processing_time,
                client_ip=client_ip,
                exc_info=True
            )
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": f"{int(start_time * 1000)}"
                }
            )


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security headers middleware."""
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers to responses."""
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


class CacheMiddleware(BaseHTTPMiddleware):
    """Simple in-memory cache middleware for GET requests."""
    
    def __init__(self, app):
        super().__init__(app)
        self.cache: Dict[str, tuple] = {}  # url -> (response, timestamp)
        self.cache_ttl = 300  # 5 minutes
        self.lock = asyncio.Lock()
    
    async def dispatch(self, request: Request, call_next):
        """Cache GET requests."""
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Skip caching for certain endpoints
        if any(path in str(request.url) for path in ["/health", "/docs", "/redoc"]):
            return await call_next(request)
        
        cache_key = str(request.url)
        current_time = time.time()
        
        async with self.lock:
            # Check cache
            if cache_key in self.cache:
                cached_response, timestamp = self.cache[cache_key]
                if current_time - timestamp < self.cache_ttl:
                    logger.debug("Cache hit", url=cache_key)
                    return Response(
                        content=cached_response,
                        headers={"X-Cache": "HIT"}
                    )
                else:
                    # Remove expired entry
                    del self.cache[cache_key]
        
        # Process request
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            # Only cache responses that have a body (not streaming responses)
            if hasattr(response, 'body'):
                async with self.lock:
                    # Store in cache (simplified - would need proper serialization in production)
                    self.cache[cache_key] = (response.body, current_time)
                    
                    # Clean old entries (simple cleanup)
                    if len(self.cache) > 1000:  # Limit cache size
                        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                        del self.cache[oldest_key]
                
                response.headers["X-Cache"] = "MISS"
        
        return response 