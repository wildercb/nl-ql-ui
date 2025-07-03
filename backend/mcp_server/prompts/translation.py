"""Translation prompts for interactive assistance."""

from typing import List
from fastmcp import FastMCP, Context
from fastmcp.prompts.base import UserMessage, AssistantMessage


def register_translation_prompts(mcp: FastMCP):
    """Register all translation-related prompts."""

    @mcp.prompt()
    async def translation_assistant(
        query_type: str = "general",
        difficulty: str = "beginner",
        ctx: Context = None
    ) -> List[UserMessage | AssistantMessage]:
        """
        Interactive translation assistant for different query types and skill levels.
        
        Provides contextual guidance for translating natural language to GraphQL
        based on the user's experience level and query complexity.
        """
        await ctx.info(f"Generating translation assistance for {query_type} queries at {difficulty} level")
        
        messages = []
        
        # Initial system context based on difficulty
        if difficulty == "beginner":
            messages.append(UserMessage(f"""
I'm new to GraphQL and need help translating natural language queries. I want to work with {query_type} queries. 

Can you help me understand:
1. Basic GraphQL syntax and structure
2. How to translate common natural language patterns
3. What information I need to provide for better translations
4. Common mistakes to avoid

Please start with simple examples and explain each part.
            """))
            
            messages.append(AssistantMessage(f"""
Welcome to GraphQL translation! I'll help you learn step by step.

## GraphQL Basics for {query_type.title()} Queries

### Basic Structure
Every GraphQL query follows this pattern:
```graphql
{{
  fieldName(arguments) {{
    subField1
    subField2
  }}
}}
```

### Common {query_type.title()} Patterns

**Simple Selection:**
- Natural: "Get all users"
- GraphQL: `{{ users {{ id name email }} }}`

**With Filters:**
- Natural: "Find users with gmail addresses"  
- GraphQL: `{{ users(where: {{ email: {{ contains: "gmail" }} }}) {{ id name email }} }}`

**With Pagination:**
- Natural: "Show first 10 users"
- GraphQL: `{{ users(first: 10) {{ id name email }} }}`

### Translation Tips:
1. **Be specific** about what data you want
2. **Mention filters** clearly (equals, contains, greater than, etc.)
3. **Specify relationships** if you need nested data
4. **Include pagination** for large datasets

### What You Need:
- Clear description of what data you want
- Any filtering criteria
- Which fields you need in the response
- Schema information (if available) for better accuracy

Try describing a query you'd like to translate, and I'll walk you through it step by step!
            """))
            
        elif difficulty == "intermediate":
            messages.append(UserMessage(f"""
I have some GraphQL experience and want to improve my translation skills for {query_type} queries.

I'd like help with:
1. More complex query patterns and relationships
2. Optimizing query performance
3. Using advanced features like fragments and variables
4. Handling edge cases in translation

Please provide examples that challenge my current understanding.
            """))
            
            messages.append(AssistantMessage(f"""
Great! Let's dive into more advanced {query_type} translation patterns.

## Advanced Translation Techniques

### Complex Relationships
```graphql
{{
  users {{
    id
    name
    orders(last: 5) {{
      id
      total
      items {{
        product {{
          name
          category
        }}
        quantity
      }}
    }}
  }}
}}
```

### Using Fragments for Reusability
```graphql
fragment UserInfo on User {{
  id
  name
  email
  profile {{
    avatar
    bio
  }}
}}

query {{
  activeUsers: users(where: {{ status: ACTIVE }}) {{
    ...UserInfo
  }}
  recentUsers: users(orderBy: {{ createdAt: DESC }}, first: 10) {{
    ...UserInfo
  }}
}}
```

### Variables and Dynamic Queries
```graphql
query GetUsers($filter: UserFilter!, $limit: Int = 10) {{
  users(where: $filter, first: $limit) {{
    id
    name
    email
  }}
}}
```

### Performance Optimization Tips:
1. **Request only needed fields** - avoid over-fetching
2. **Use pagination** for large datasets
3. **Leverage fragments** to avoid duplication
4. **Consider query depth** - deeply nested queries can be slow

### Translation Challenges:
Try translating these complex requests:
- "Show me users who have placed orders over $100 in the last month, including their order details and shipping addresses"
- "Find products that are out of stock but have pending orders, grouped by category"
- "Get analytics for user engagement including session count, average session duration, and most viewed pages"

What complex {query_type} scenario would you like to work on?
            """))
            
        else:  # advanced
            messages.append(UserMessage(f"""
I'm experienced with GraphQL and want to master advanced translation techniques for {query_type} queries.

Focus on:
1. Complex nested relationships and data aggregation
2. Performance optimization and query analysis
3. Schema-aware translation strategies
4. Error handling and validation patterns
5. Custom directives and advanced features

Show me sophisticated examples and edge cases.
            """))
            
            messages.append(AssistantMessage(f"""
Excellent! Let's explore expert-level {query_type} translation techniques.

## Expert Translation Strategies

### Schema-Aware Optimization
```graphql
# Leveraging schema introspection for better translations
query IntrospectFirst {{
  __schema {{
    types {{
      name
      fields {{
        name
        type {{
          name
          kind
        }}
      }}
    }}
  }}
}}
```

### Advanced Aggregation Patterns
```graphql
{{
  analytics {{
    userStats {{
      total: count
      active: count(where: {{ status: ACTIVE }})
      byCountry: groupBy(field: country) {{
        key
        count
        averageOrderValue: avg(orders: {{ field: total }})
      }}
    }}
    orderTrends(period: MONTHLY) {{
      date
      revenue: sum(field: total)
      orderCount: count
      averageOrderValue: avg(field: total)
      topProducts(limit: 5) {{
        product {{ name }}
        quantity: sum(field: quantity)
        revenue: sum(field: subtotal)
      }}
    }}
  }}
}}
```

### Error Handling & Validation
```graphql
# Query with comprehensive error handling
query GetUserData($userId: ID!) {{
  user(id: $userId) {{
    ... on User {{
      id
      name
      orders {{
        id
        status
        ... on FailedOrder {{
          errorReason
          retryCount
        }}
      }}
    }}
    ... on UserNotFound {{
      message
      suggestedActions
    }}
  }}
}}
```

### Performance Analysis Queries
```graphql
{{
  __schema {{
    queryType {{
      fields {{
        name
        type {{
          name
        }}
        args {{
          name
          type {{
            name
          }}
        }}
      }}
    }}
  }}
}}
```

### Advanced Translation Considerations:
1. **Schema Evolution** - Handle versioning and deprecated fields
2. **Federation** - Cross-service relationships and boundaries  
3. **Security** - Field-level permissions and data sensitivity
4. **Caching** - Query structure for optimal cache utilization
5. **Real-time** - Subscription patterns for live data

### Expert Challenges:
- Translate multi-dimensional analytics queries with complex grouping
- Handle polymorphic relationships and union types
- Optimize queries for federated GraphQL architectures
- Design queries that work across schema versions

What advanced {query_type} translation challenge interests you most?
            """))
        
        return messages

    @mcp.prompt()
    async def query_optimization(
        query: str = "",
        schema_context: str = "",
        ctx: Context = None
    ) -> List[UserMessage | AssistantMessage]:
        """
        Prompt for query optimization guidance and best practices.
        
        Analyzes existing queries and provides specific recommendations
        for improving performance, maintainability, and effectiveness.
        """
        await ctx.info("Generating query optimization guidance")
        
        if query:
            user_msg = f"""
I have this GraphQL query that I'd like to optimize:

```graphql
{query}
```

Schema context (if available):
{schema_context or "No schema provided"}

Please analyze this query and help me optimize it for:
1. Performance (speed and resource usage)
2. Maintainability (readability and reusability)
3. Best practices compliance
4. Potential issues or improvements

Provide specific recommendations with examples.
            """
            
            assistant_msg = f"""
I'll analyze your query and provide optimization recommendations.

## Query Analysis

### Current Query Structure:
```graphql
{query}
```

## Optimization Recommendations:

### 1. Performance Optimizations
- **Field Selection**: Only request fields you actually need
- **Pagination**: Use `first`/`last` with `after`/`before` for large datasets
- **Depth Limiting**: Avoid deeply nested queries that can cause N+1 problems
- **Batch Loading**: Group related requests when possible

### 2. Maintainability Improvements
- **Fragments**: Extract repeated field sets into reusable fragments
- **Variables**: Use variables instead of hardcoded values
- **Descriptive Names**: Use clear, descriptive operation names

### 3. Best Practices
- **Error Handling**: Include error-prone fields in fragments with error handling
- **Aliasing**: Use aliases to avoid field name conflicts
- **Directives**: Use `@include` and `@skip` for conditional fields

### 4. Suggested Optimizations:

**Optimized Version:**
```graphql
# I'll provide a specific optimized version based on your query
# This would include actual improvements specific to your query structure
```

**Performance Metrics to Monitor:**
- Query execution time
- Number of database queries generated
- Memory usage
- Cache hit rates

### Additional Recommendations:
{"Schema-specific optimizations would be provided here if schema context was available" if not schema_context else "Schema-aware optimizations based on your provided context"}

Would you like me to focus on any specific aspect of optimization?
            """
        else:
            user_msg = """
I want to learn about GraphQL query optimization. Can you teach me:

1. Common performance bottlenecks in GraphQL queries
2. Best practices for writing efficient queries
3. How to analyze and improve existing queries
4. Tools and techniques for performance monitoring

Please provide practical examples and actionable advice.
            """
            
            assistant_msg = """
# GraphQL Query Optimization Guide

## Common Performance Issues

### 1. The N+1 Problem
**Problem:** Fetching related data causes multiple database queries
```graphql
# This could cause N+1 queries
{
  users {
    name
    orders {  # Separate query for each user
      total
    }
  }
}
```

**Solution:** Use DataLoader or batch resolvers

### 2. Over-fetching
**Problem:** Requesting unnecessary data
```graphql
# Bad: Requesting all fields
{
  users {
    id name email bio avatar preferences settings metadata
  }
}

# Good: Only what you need
{
  users {
    id name email
  }
}
```

### 3. Deep Nesting
**Problem:** Unbounded depth can cause exponential complexity
```graphql
# Dangerous: Could fetch massive amounts of data
{
  users {
    friends {
      friends {
        friends {
          # ... potentially infinite
        }
      }
    }
  }
}
```

## Optimization Strategies

### 1. Smart Field Selection
```graphql
# Use fragments for common field sets
fragment UserSummary on User {
  id
  name
  email
  createdAt
}

query {
  activeUsers: users(where: { status: ACTIVE }) {
    ...UserSummary
  }
  recentUsers: users(orderBy: { createdAt: DESC }, first: 10) {
    ...UserSummary
  }
}
```

### 2. Pagination Patterns
```graphql
# Cursor-based pagination (recommended)
query GetUsers($after: String, $first: Int = 20) {
  users(after: $after, first: $first) {
    edges {
      node {
        id
        name
        email
      }
      cursor
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
```

### 3. Query Complexity Analysis
```graphql
# Monitor these metrics:
# - Query depth (max 10-15 levels)
# - Field count (max 100-200 fields)  
# - Complexity score (custom calculation)
```

## Performance Monitoring

### Query Analysis Tools:
1. **Apollo Studio** - Query performance insights
2. **GraphQL Voyager** - Schema visualization
3. **Custom Metrics** - Response time, error rates
4. **Database Monitoring** - Query counts, execution time

### Key Metrics to Track:
- Average query execution time
- 95th percentile response time
- Query complexity scores
- Cache hit/miss rates
- Error rates by query type

## Optimization Checklist:

✅ **Query Structure:**
- [ ] Use fragments for repeated fields
- [ ] Implement proper pagination
- [ ] Limit query depth
- [ ] Use aliases for clarity

✅ **Performance:**  
- [ ] Only request needed fields
- [ ] Implement query complexity limits
- [ ] Use caching strategies
- [ ] Monitor N+1 queries

✅ **Maintainability:**
- [ ] Use descriptive operation names
- [ ] Leverage variables for dynamic values
- [ ] Include error handling
- [ ] Document complex queries

Would you like me to analyze a specific query or dive deeper into any of these optimization techniques?
            """
        
        return [
            UserMessage(user_msg),
            AssistantMessage(assistant_msg)
        ]

    @mcp.prompt()
    async def domain_translation(
        domain: str = "general",
        schema_sample: str = "",
        ctx: Context = None
    ) -> List[UserMessage | AssistantMessage]:
        """
        Domain-specific translation guidance for specialized GraphQL schemas.
        
        Provides tailored assistance for different domains like e-commerce,
        social media, analytics, etc. with domain-specific patterns and examples.
        """
        await ctx.info(f"Generating domain-specific translation guidance for {domain}")
        
        domain_guides = {
            "ecommerce": {
                "intro": "E-commerce GraphQL queries focus on products, orders, customers, and inventory management.",
                "patterns": [
                    {
                        "name": "Product Catalog",
                        "natural": "Show products in electronics category under $500",
                        "graphql": "{ products(where: { category: \"electronics\", price: { lt: 500 } }) { id name price description images } }"
                    },
                    {
                        "name": "Order Management", 
                        "natural": "Get pending orders with customer details",
                        "graphql": "{ orders(where: { status: PENDING }) { id total customer { name email } items { product { name } quantity } } }"
                    },
                    {
                        "name": "Inventory Tracking",
                        "natural": "Find low stock products",
                        "graphql": "{ products(where: { inventory: { lt: 10 } }) { id name inventory supplier { name } } }"
                    }
                ],
                "considerations": [
                    "Product variants and SKUs",
                    "Pricing and currency handling", 
                    "Inventory levels and availability",
                    "Customer segmentation",
                    "Order states and fulfillment"
                ]
            },
            "social": {
                "intro": "Social media GraphQL queries handle users, posts, relationships, and engagement metrics.",
                "patterns": [
                    {
                        "name": "User Feed",
                        "natural": "Get recent posts from followed users",
                        "graphql": "{ posts(where: { author: { followedBy: { userId: $currentUser } } }, orderBy: { createdAt: DESC }) { id content author { name avatar } likes { count } } }"
                    },
                    {
                        "name": "Social Graph",
                        "natural": "Find mutual friends between two users",
                        "graphql": "{ user(id: $userId) { friends(where: { friends: { some: { id: $otherUserId } } }) { id name } } }"
                    }
                ],
                "considerations": [
                    "Privacy and visibility settings",
                    "Relationship types (friends, followers, etc.)",
                    "Content moderation and filtering",
                    "Real-time updates and notifications"
                ]
            }
        }
        
        guide = domain_guides.get(domain, {
            "intro": f"Domain-specific guidance for {domain} GraphQL queries.",
            "patterns": [],
            "considerations": ["Domain-specific data modeling", "Common query patterns", "Performance considerations"]
        })
        
        messages = [
            UserMessage(f"""
I'm working with a {domain} GraphQL schema and need help with domain-specific translation patterns.

Schema sample (if available):
{schema_sample or "No schema sample provided"}

Please help me understand:
1. Common query patterns for {domain}
2. Domain-specific terminology and conventions  
3. Best practices for this type of application
4. Common challenges and how to address them

Provide practical examples I can use as templates.
            """),
            
            AssistantMessage(f"""
# {domain.title()} GraphQL Translation Guide

{guide['intro']}

## Common Query Patterns

{chr(10).join([f"### {p['name']}{chr(10)}**Natural Language:** {p['natural']}{chr(10)}**GraphQL:**{chr(10)}```graphql{chr(10)}{p['graphql']}{chr(10)}```{chr(10)}" for p in guide['patterns']])}

## Domain Considerations

{chr(10).join([f"- {consideration}" for consideration in guide['considerations']])}

## Translation Tips for {domain.title()}:

1. **Understand the Data Model**: Learn the core entities and their relationships
2. **Know the Business Logic**: Understand domain rules and constraints
3. **Consider Performance**: Some domains have specific performance requirements
4. **Handle Edge Cases**: Plan for data consistency and validation needs

## Next Steps:
- Describe a specific {domain} query you'd like to translate
- Ask about relationships between entities in your domain
- Request help with complex {domain} scenarios

What {domain}-specific translation challenge would you like to work on?
            """)
        ]
        
        return messages 