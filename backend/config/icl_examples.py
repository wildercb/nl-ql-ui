"""Initial In-Context Learning (ICL) Examples for Natural Language to GraphQL Translation."""

from typing import List, Dict

# List of initial ICL examples for seeding
# Each example is a dictionary with 'natural' (natural language query) and 'graphql' (corresponding GraphQL query)
INITIAL_ICL_EXAMPLES: List[Dict[str, str]] = [
    {
        "natural": "Get all users with their names and emails",
        "graphql": "query { users { name email } }"
    },
    {
        "natural": "Find products in the electronics category under $500",
        "graphql": "query { products(where: { category: \"electronics\", price: { lt: 500 } }) { id name price } }"
    },
    {
        "natural": "Show me the latest 10 orders with customer details",
        "graphql": "query { orders(orderBy: { createdAt: DESC }, first: 10) { id total customer { name email } } }"
    },
    {
        "natural": "Get posts by user John published last month",
        "graphql": "query { posts(where: { author: { name: \"John\" }, createdAt: { gte: \"2023-10-01\", lt: \"2023-11-01\" } }) { id title content } }"
    },
    {
        "natural": "List all categories with more than 5 products",
        "graphql": "query { categories(where: { products: { count: { gt: 5 } } }) { name products { count } } }"
    }
]

def get_initial_icl_examples() -> List[str]:
    """Format initial ICL examples as strings for inclusion in prompts."""
    formatted_examples = []
    for example in INITIAL_ICL_EXAMPLES:
        formatted_examples.append(f"Natural: {example['natural']}\nGraphQL: {example['graphql']}")
    return formatted_examples 