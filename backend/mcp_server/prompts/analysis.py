"""Analysis prompts for schema and query analysis."""

from typing import List
from fastmcp import FastMCP, Context
from fastmcp.prompts.base import UserMessage, AssistantMessage


def register_analysis_prompts(mcp: FastMCP):
    """Register all analysis-related prompts."""

    @mcp.prompt()
    async def schema_exploration(ctx: Context = None) -> List[UserMessage | AssistantMessage]:
        """Guide for exploring and understanding GraphQL schemas."""
        await ctx.info("Generating schema exploration guidance")
        
        return [
            UserMessage("""
I need help understanding a GraphQL schema systematically. Can you guide me through:
1. How to analyze schema structure and relationships
2. Finding key query patterns and entry points  
3. Understanding the data model
4. Discovering useful queries for my needs

Please provide a step-by-step exploration approach.
            """),
            
            AssistantMessage("""
# GraphQL Schema Exploration Guide

## Step 1: Schema Overview
Start with a high-level view:

```graphql
query SchemaOverview {
  __schema {
    types { name kind description }
    queryType { name }
    mutationType { name }
  }
}
```

## Step 2: Explore Entry Points
Examine main query operations:

```graphql
query QueryOperations {
  __type(name: "Query") {
    fields {
      name
      description
      type { name kind }
      args { name type { name } }
    }
  }
}
```

## Step 3: Detailed Type Exploration
For any interesting type:

```graphql
query ExploreType($typeName: String!) {
  __type(name: $typeName) {
    name
    description
    fields {
      name
      description
      type { name kind }
      args { name type { name } }
    }
  }
}
```

## Exploration Strategy:
1. **Start Small**: Begin with simple field selections
2. **Build Gradually**: Add nesting one level at a time
3. **Test Filters**: Try different where conditions
4. **Document Patterns**: Save useful queries

## Common Patterns to Look For:
- **Pagination**: `first`, `after`, `edges`, `pageInfo`
- **Filtering**: `where`, `filter` arguments
- **Sorting**: `orderBy`, `sort` arguments

Ready to explore your schema?
            """)
        ]

    @mcp.prompt()
    async def performance_analysis(ctx: Context = None) -> List[UserMessage | AssistantMessage]:
        """Performance analysis and optimization guidance."""
        await ctx.info("Generating performance analysis guidance")
        
        return [
            UserMessage("""
I need help with GraphQL query performance analysis and optimization. Please help me with:
1. Identifying common performance bottlenecks
2. Query optimization techniques
3. Monitoring and measuring performance
4. Best practices for optimization

Provide practical guidance I can implement.
            """),
            
            AssistantMessage("""
# GraphQL Performance Optimization Guide

## Common Performance Issues

### 1. The N+1 Problem
```graphql
# This can cause N+1 queries
{
  users {
    name
    posts {  # Separate query for each user
      title
    }
  }
}
```
**Solution:** Use DataLoader pattern

### 2. Over-fetching
```graphql
# Bad: Requesting everything
{
  users {
    id name email bio avatar preferences settings
  }
}

# Good: Only what you need
{
  users {
    id name email
  }
}
```

### 3. Unbounded Queries
```graphql
# Dangerous: No limits
{
  users {
    posts {
      comments {
        # Could be infinite...
      }
    }
  }
}
```

## Optimization Strategies

### 1. Query Complexity Limits
Implement depth and complexity limits

### 2. Efficient Pagination
```graphql
query GetProducts($first: Int!, $after: String) {
  products(first: $first, after: $after) {
    edges {
      node { id name price }
      cursor
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
```

### 3. Smart Field Selection
Use fragments for repeated field sets

## Performance Checklist:
- [ ] Limit query depth (max 10-15 levels)
- [ ] Implement pagination for lists
- [ ] Use DataLoader for N+1 prevention
- [ ] Add query complexity limits
- [ ] Monitor execution times

What specific performance challenge are you facing?
            """)
        ] 