"""Query resources for accessing saved translations and queries."""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from fastmcp import FastMCP, Context


def register_query_resources(mcp: FastMCP):
    """Register all query-related resources."""

    @mcp.resource("query://saved/{query_id}")
    async def get_saved_query(query_id: str, ctx: Context = None) -> str:
        """
        Retrieve a saved query by ID.
        
        Provides access to previously saved GraphQL translations
        including metadata and translation details.
        """
        await ctx.info(f"Retrieving saved query: {query_id}")
        
        try:
            # In a real implementation, this would query the database
            # For now, return a mock saved query
            mock_query = {
                "id": query_id,
                "natural_query": "Find all users with gmail addresses",
                "graphql_query": "{ users(where: { email: { contains: \"gmail\" } }) { id email name } }",
                "confidence": 0.85,
                "created_at": datetime.now().isoformat(),
                "model_used": "llama2",
                "metadata": {
                    "saved_by": "user_123",
                    "tags": ["users", "email", "filter"],
                    "usage_count": 5
                }
            }
            
            await ctx.info("Saved query retrieved successfully")
            return json.dumps(mock_query, indent=2)
            
        except Exception as e:
            await ctx.error(f"Failed to retrieve saved query: {str(e)}")
            return json.dumps({"error": str(e), "query_id": query_id})

    @mcp.resource("query://recent")
    async def get_recent_queries(ctx: Context = None) -> str:
        """
        Get recently translated queries.
        
        Returns a list of the most recent query translations
        for quick access and reference.
        """
        await ctx.info("Retrieving recent queries")
        
        try:
            # Mock recent queries - in practice this would come from database
            recent_queries = [
                {
                    "id": "q1",
                    "natural_query": "Get all products under $50",
                    "graphql_query": "{ products(where: { price: { lt: 50 } }) { id name price } }",
                    "confidence": 0.92,
                    "created_at": (datetime.now() - timedelta(hours=1)).isoformat()
                },
                {
                    "id": "q2", 
                    "natural_query": "Find users created in the last week",
                    "graphql_query": "{ users(where: { createdAt: { gte: \"2024-01-01\" } }) { id name email } }",
                    "confidence": 0.88,
                    "created_at": (datetime.now() - timedelta(hours=2)).isoformat()
                },
                {
                    "id": "q3",
                    "natural_query": "Count active subscriptions",
                    "graphql_query": "{ subscriptions(where: { status: ACTIVE }) { aggregate { count } } }",
                    "confidence": 0.75,
                    "created_at": (datetime.now() - timedelta(hours=4)).isoformat()
                }
            ]
            
            result = {
                "recent_queries": recent_queries,
                "count": len(recent_queries),
                "last_updated": datetime.now().isoformat()
            }
            
            await ctx.info(f"Retrieved {len(recent_queries)} recent queries")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            await ctx.error(f"Failed to retrieve recent queries: {str(e)}")
            return json.dumps({"error": str(e)})

    @mcp.resource("query://popular")
    async def get_popular_queries(ctx: Context = None) -> str:
        """
        Get popular/frequently used query patterns.
        
        Returns commonly requested translation patterns
        that can serve as examples and templates.
        """
        await ctx.info("Retrieving popular query patterns")
        
        try:
            popular_patterns = [
                {
                    "pattern": "User queries with email filters",
                    "example_natural": "Find users with gmail addresses",
                    "example_graphql": "{ users(where: { email: { contains: \"gmail\" } }) { id email name } }",
                    "usage_count": 45,
                    "category": "user_management"
                },
                {
                    "pattern": "Product filtering by price",
                    "example_natural": "Get products under $100",
                    "example_graphql": "{ products(where: { price: { lt: 100 } }) { id name price category } }",
                    "usage_count": 38,
                    "category": "ecommerce"
                },
                {
                    "pattern": "Date range queries",
                    "example_natural": "Show orders from last month",
                    "example_graphql": "{ orders(where: { createdAt: { gte: \"2024-01-01\", lte: \"2024-01-31\" } }) { id total } }",
                    "usage_count": 29,
                    "category": "time_based"
                },
                {
                    "pattern": "Nested relationships",
                    "example_natural": "Users with their orders and order items",
                    "example_graphql": "{ users { id name orders { id total items { id product { name } quantity } } } }",
                    "usage_count": 22,
                    "category": "relationships"
                }
            ]
            
            result = {
                "popular_patterns": popular_patterns,
                "count": len(popular_patterns),
                "categories": list(set(p["category"] for p in popular_patterns))
            }
            
            await ctx.info(f"Retrieved {len(popular_patterns)} popular patterns")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            await ctx.error(f"Failed to retrieve popular queries: {str(e)}")
            return json.dumps({"error": str(e)})

    @mcp.resource("query://templates/{category}")
    async def get_query_templates(category: str, ctx: Context = None) -> str:
        """
        Get query templates for a specific category.
        
        Provides pre-built query templates that can be customized
        for common use cases in different domains.
        """
        await ctx.info(f"Retrieving query templates for category: {category}")
        
        try:
            templates_by_category = {
                "ecommerce": [
                    {
                        "name": "Product Search",
                        "description": "Search products by name, category, or price range",
                        "template": "{ products(where: { ${filters} }) { id name price category description } }",
                        "variables": ["name", "category", "price_min", "price_max"],
                        "examples": [
                            "Products in electronics category",
                            "Products under $50",
                            "Products with 'phone' in name"
                        ]
                    },
                    {
                        "name": "Order Management",
                        "description": "Query orders with customer and item details",
                        "template": "{ orders(where: { ${filters} }) { id total status customer { name email } items { product { name } quantity } } }",
                        "variables": ["status", "date_range", "customer_id"],
                        "examples": [
                            "Pending orders",
                            "Orders from last week",
                            "Orders by specific customer"
                        ]
                    }
                ],
                "user_management": [
                    {
                        "name": "User Search",
                        "description": "Find users by various criteria",
                        "template": "{ users(where: { ${filters} }) { id name email createdAt profile { ${profile_fields} } } }",
                        "variables": ["email", "role", "created_date", "profile_fields"],
                        "examples": [
                            "Users with admin role",
                            "Recently registered users",
                            "Users with incomplete profiles"
                        ]
                    },
                    {
                        "name": "User Activity",
                        "description": "Track user activity and engagement",
                        "template": "{ users { id name lastLogin sessions(last: 10) { createdAt duration } activity { ${activity_fields} } } }",
                        "variables": ["activity_type", "time_range", "activity_fields"],
                        "examples": [
                            "User login patterns",
                            "Active users this week",
                            "User engagement metrics"
                        ]
                    }
                ]
            }
            
            templates = templates_by_category.get(category, [])
            
            if not templates:
                available_categories = list(templates_by_category.keys())
                result = {
                    "error": f"Category '{category}' not found",
                    "available_categories": available_categories,
                    "suggestion": f"Try one of: {', '.join(available_categories)}"
                }
            else:
                result = {
                    "category": category,
                    "templates": templates,
                    "count": len(templates)
                }
            
            await ctx.info(f"Retrieved {len(templates)} templates for {category}")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            await ctx.error(f"Failed to retrieve templates: {str(e)}")
            return json.dumps({"error": str(e), "category": category})

    @mcp.resource("query://statistics")
    async def get_query_statistics(ctx: Context = None) -> str:
        """
        Get usage statistics for the translation service.
        
        Provides insights into service usage patterns,
        popular models, success rates, and performance metrics.
        """
        await ctx.info("Retrieving query statistics")
        
        try:
            # Mock statistics - in practice this would come from analytics
            stats = {
                "total_translations": 1247,
                "successful_translations": 1089,
                "success_rate": 87.3,
                "average_confidence": 0.82,
                "popular_models": [
                    {"model": "llama2", "usage_count": 445, "avg_confidence": 0.84},
                    {"model": "codellama", "usage_count": 312, "avg_confidence": 0.89},
                    {"model": "mistral", "usage_count": 267, "avg_confidence": 0.79}
                ],
                "query_categories": [
                    {"category": "user_management", "count": 387, "percentage": 31.0},
                    {"category": "ecommerce", "count": 312, "percentage": 25.0},
                    {"category": "analytics", "count": 248, "percentage": 19.9},
                    {"category": "content", "count": 186, "percentage": 14.9},
                    {"category": "other", "count": 114, "percentage": 9.1}
                ],
                "performance_metrics": {
                    "average_response_time": "1.2s",
                    "p95_response_time": "3.1s",
                    "peak_usage_hour": "14:00-15:00 UTC",
                    "most_active_day": "Tuesday"
                },
                "recent_trends": [
                    "Increased use of complex nested queries (+15%)",
                    "Growing adoption of batch translation (+22%)",
                    "Improved success rate for schema-aware translations (+8%)"
                ],
                "last_updated": datetime.now().isoformat()
            }
            
            await ctx.info("Statistics retrieved successfully")
            return json.dumps(stats, indent=2)
            
        except Exception as e:
            await ctx.error(f"Failed to retrieve statistics: {str(e)}")
            return json.dumps({"error": str(e)}) 