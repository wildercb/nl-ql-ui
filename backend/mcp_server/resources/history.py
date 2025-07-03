"""History resources for user translation history and sessions."""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from fastmcp import FastMCP, Context


def register_history_resources(mcp: FastMCP):
    """Register all history-related resources."""

    @mcp.resource("history://user/{user_id}")
    async def get_user_history(user_id: str, ctx: Context = None) -> str:
        """Get translation history for a specific user."""
        await ctx.info(f"Retrieving history for user: {user_id}")
        
        try:
            # Mock user history
            user_history = {
                "user_id": user_id,
                "total_translations": 24,
                "translations": [
                    {
                        "id": "t1",
                        "natural_query": "Find active users",
                        "graphql_query": "{ users(where: { status: ACTIVE }) { id name email } }",
                        "confidence": 0.89,
                        "created_at": (datetime.now() - timedelta(hours=2)).isoformat(),
                        "model": "llama2"
                    },
                    {
                        "id": "t2", 
                        "natural_query": "Get product sales this month",
                        "graphql_query": "{ orders(where: { createdAt: { gte: \"2024-01-01\" } }) { items { product { name } quantity } } }",
                        "confidence": 0.76,
                        "created_at": (datetime.now() - timedelta(days=1)).isoformat(),
                        "model": "codellama"
                    }
                ],
                "statistics": {
                    "average_confidence": 0.82,
                    "most_used_model": "llama2",
                    "favorite_categories": ["user_management", "ecommerce"]
                }
            }
            
            return json.dumps(user_history, indent=2)
            
        except Exception as e:
            await ctx.error(f"Failed to retrieve user history: {str(e)}")
            return json.dumps({"error": str(e), "user_id": user_id})

    @mcp.resource("history://session/{session_id}")
    async def get_session_history(session_id: str, ctx: Context = None) -> str:
        """Get translation history for a specific session."""
        await ctx.info(f"Retrieving session history: {session_id}")
        
        try:
            session_data = {
                "session_id": session_id,
                "started_at": (datetime.now() - timedelta(hours=3)).isoformat(),
                "translations_count": 8,
                "translations": [
                    {
                        "sequence": 1,
                        "natural_query": "Show me all users",
                        "graphql_query": "{ users { id name email } }",
                        "confidence": 0.95
                    },
                    {
                        "sequence": 2,
                        "natural_query": "Filter users by gmail",
                        "graphql_query": "{ users(where: { email: { contains: \"gmail\" } }) { id name email } }",
                        "confidence": 0.88
                    }
                ]
            }
            
            return json.dumps(session_data, indent=2)
            
        except Exception as e:
            await ctx.error(f"Failed to retrieve session history: {str(e)}")
            return json.dumps({"error": str(e), "session_id": session_id}) 