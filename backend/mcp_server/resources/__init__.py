"""FastMCP Resources for GraphQL Translation."""

from .queries import register_query_resources
from .history import register_history_resources

__all__ = [
    "register_query_resources",
    "register_history_resources"
] 