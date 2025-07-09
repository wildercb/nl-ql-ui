"""
Validation Service - Compatibility layer for unified architecture.

This service provides GraphQL validation functionality using
the unified architecture components.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result from GraphQL validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    confidence: float = 1.0


class ValidationService:
    """
    Validation service for GraphQL queries.
    
    Provides basic GraphQL syntax validation and compatibility
    with existing API routes.
    """
    
    def __init__(self):
        logger.info("ValidationService initialized")
    
    async def validate_graphql(
        self,
        graphql_query: str,
        schema_context: Optional[str] = None,
        **kwargs
    ) -> ValidationResult:
        """
        Validate a GraphQL query.
        
        Args:
            graphql_query: The GraphQL query to validate
            schema_context: Schema context for validation
            
        Returns:
            ValidationResult with validation details
        """
        try:
            errors = []
            warnings = []
            suggestions = []
            
            # Basic syntax validation
            if not graphql_query.strip():
                errors.append("Query cannot be empty")
                return ValidationResult(False, errors, warnings, suggestions, 0.0)
            
            # Check for basic GraphQL structure
            query_lower = graphql_query.lower().strip()
            
            if not any(keyword in query_lower for keyword in ['query', 'mutation', 'subscription']):
                # Check if it starts with { (shorthand query)
                if not query_lower.startswith('{'):
                    warnings.append("Query should start with 'query', 'mutation', 'subscription', or '{'")
            
            # Check for balanced braces
            open_braces = graphql_query.count('{')
            close_braces = graphql_query.count('}')
            
            if open_braces != close_braces:
                errors.append(f"Unbalanced braces: {open_braces} opening, {close_braces} closing")
            
            # Check for balanced parentheses
            open_parens = graphql_query.count('(')
            close_parens = graphql_query.count(')')
            
            if open_parens != close_parens:
                errors.append(f"Unbalanced parentheses: {open_parens} opening, {close_parens} closing")
            
            # Basic field validation
            if '{' in graphql_query and '}' in graphql_query:
                # Extract field content
                try:
                    start = graphql_query.find('{')
                    end = graphql_query.rfind('}')
                    field_content = graphql_query[start+1:end].strip()
                    
                    if not field_content:
                        warnings.append("Query has no fields selected")
                    
                except Exception:
                    warnings.append("Could not parse field content")
            
            # Suggestions for improvement
            if 'id' not in query_lower:
                suggestions.append("Consider including 'id' field for better caching")
            
            if len(graphql_query) > 1000:
                suggestions.append("Query is quite long - consider breaking into smaller queries")
            
            is_valid = len(errors) == 0
            confidence = 1.0 if is_valid else 0.5
            
            logger.info(f"Validation complete: valid={is_valid}, errors={len(errors)}, warnings={len(warnings)}")
            
            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation failed: {str(e)}"],
                warnings=[],
                suggestions=[],
                confidence=0.0
            )
    
    async def validate_syntax(self, graphql_query: str) -> bool:
        """Simple syntax validation that returns True/False."""
        result = await self.validate_graphql(graphql_query)
        return result.is_valid 