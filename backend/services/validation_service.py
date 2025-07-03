"""Validation service for GraphQL queries."""

import re
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from graphql import parse, validate, build_schema, DocumentNode
from graphql.error import GraphQLError
from config.settings import get_settings
from services.translation_service import TranslationService

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of GraphQL query validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    parsed_query: Optional[DocumentNode] = None


class ValidationService:
    """Service for validating GraphQL queries."""

    def __init__(self):
        self.default_rules = [
            "NoUnusedFragments",
            "NoUndefinedVariables",
            "KnownArgumentNames",
            "KnownDirectives",
            "KnownFragmentNames",
            "KnownTypeNames",
            "LoneAnonymousOperation",
            "NoFragmentCycles",
            "OverlappingFieldsCanBeMerged",
            "PossibleFragmentSpreads",
            "ProvidedRequiredArguments",
            "ScalarLeafs",
            "SingleFieldSubscriptions",
            "UniqueArgumentNames",
            "UniqueDirectivesPerLocation",
            "UniqueFragmentNames",
            "UniqueInputFieldNames",
            "UniqueOperationNames",
            "UniqueVariableNames",
            "ValuesOfCorrectType",
            "VariablesAreInputTypes",
            "VariablesInAllowedPosition"
        ]

    def _basic_syntax_check(self, query: str) -> List[str]:
        """Perform basic syntax checks without full GraphQL parsing."""
        errors = []
        
        if not query.strip():
            errors.append("Query is empty")
            return errors
        
        # Check for balanced braces
        open_braces = query.count('{')
        close_braces = query.count('}')
        if open_braces != close_braces:
            errors.append(f"Unbalanced braces: {open_braces} open, {close_braces} close")
        
        # Check for balanced parentheses
        open_parens = query.count('(')
        close_parens = query.count(')')
        if open_parens != close_parens:
            errors.append(f"Unbalanced parentheses: {open_parens} open, {close_parens} close")
        
        # Check for balanced brackets
        open_brackets = query.count('[')
        close_brackets = query.count(']')
        if open_brackets != close_brackets:
            errors.append(f"Unbalanced brackets: {open_brackets} open, {close_brackets} close")
        
        # Check for proper string quotes
        single_quotes = query.count("'")
        double_quotes = query.count('"')
        if single_quotes % 2 != 0:
            errors.append("Unmatched single quotes")
        if double_quotes % 2 != 0:
            errors.append("Unmatched double quotes")
        
        return errors

    def _get_style_warnings(self, query: str) -> List[str]:
        """Get style and best practice warnings."""
        warnings = []
        
        # Check for operation name
        if re.search(r'^[\s]*(?:query|mutation|subscription)[\s]*\{', query.strip()):
            warnings.append("Consider adding an operation name for better debugging")
        
        # Check for very deep nesting (potential performance issue)
        max_depth = 0
        current_depth = 0
        for char in query:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth -= 1
        
        if max_depth > 8:
            warnings.append(f"Query has deep nesting (depth: {max_depth}), consider using fragments")
        
        # Check for potential over-fetching (many fields at same level)
        lines = query.split('\n')
        for i, line in enumerate(lines):
            # Count fields at the same indentation level
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and '{' not in stripped and '}' not in stripped:
                # This is a simplified check - could be more sophisticated
                pass
        
        # Check for missing variables when parameters are used
        if '(' in query and '$' not in query:
            warnings.append("Query has arguments but no variables defined")
        
        return warnings

    def _get_suggestions(self, query: str, errors: List[str]) -> List[str]:
        """Generate suggestions based on query and errors."""
        suggestions = []
        
        if not query.strip():
            suggestions.append("Start with a basic query structure: query { field }")
            return suggestions
        
        # Suggestions based on errors
        for error in errors:
            if "unbalanced" in error.lower():
                suggestions.append("Check that all opening braces, parentheses, and brackets are properly closed")
            elif "unmatched" in error.lower():
                suggestions.append("Ensure all string literals are properly quoted")
        
        # General suggestions
        if 'query' not in query.lower() and 'mutation' not in query.lower() and 'subscription' not in query.lower():
            suggestions.append("Consider specifying the operation type (query, mutation, or subscription)")
        
        if len(query.split('\n')) == 1 and len(query) > 100:
            suggestions.append("Consider formatting the query across multiple lines for better readability")
        
        return suggestions

    async def validate_query(
        self, 
        query: str, 
        schema: Optional[str] = None
    ) -> ValidationResult:
        """Validate a GraphQL query."""
        errors = []
        warnings = []
        parsed_query = None
        
        # Basic syntax checks first
        basic_errors = self._basic_syntax_check(query)
        errors.extend(basic_errors)
        
        # If basic syntax is invalid, don't proceed with full parsing
        if basic_errors:
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                suggestions=self._get_suggestions(query, errors)
            )
        
        # Try to parse the query
        try:
            parsed_query = parse(query)
        except GraphQLError as e:
            errors.append(f"Parse error: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                suggestions=self._get_suggestions(query, errors)
            )
        except Exception as e:
            errors.append(f"Unexpected parse error: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                suggestions=self._get_suggestions(query, errors)
            )
        
        # If we have a schema, validate against it
        if schema and parsed_query:
            try:
                schema_obj = build_schema(schema)
                validation_errors = validate(schema_obj, parsed_query)
                
                for error in validation_errors:
                    errors.append(str(error))
                    
            except GraphQLError as e:
                warnings.append(f"Schema validation warning: {str(e)}")
            except Exception as e:
                warnings.append(f"Could not validate against schema: {str(e)}")
        
        # Get style warnings
        style_warnings = self._get_style_warnings(query)
        warnings.extend(style_warnings)
        
        # Generate suggestions
        suggestions = self._get_suggestions(query, errors)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            parsed_query=parsed_query
        )

    async def validate_queries_batch(
        self, 
        queries: List[str], 
        schema: Optional[str] = None
    ) -> List[ValidationResult]:
        """Validate multiple GraphQL queries."""
        results = []
        
        for query in queries:
            try:
                result = await self.validate_query(query, schema)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to validate query: {e}")
                results.append(ValidationResult(
                    is_valid=False,
                    errors=[f"Validation failed: {str(e)}"],
                    warnings=[],
                    suggestions=["Check query syntax and try again"]
                ))
        
        return results

    def get_validation_rules(self) -> List[str]:
        """Get list of validation rules used."""
        return self.default_rules.copy()

    def extract_query_info(self, query: str) -> Dict[str, Any]:
        """Extract information about a GraphQL query."""
        try:
            parsed = parse(query)
            info = {
                "operation_type": None,
                "operation_name": None,
                "field_count": 0,
                "has_variables": False,
                "has_fragments": False,
                "max_depth": 0
            }
            
            # This would need more sophisticated AST traversal
            # For now, return basic info
            if 'query' in query:
                info["operation_type"] = "query"
            elif 'mutation' in query:
                info["operation_type"] = "mutation"
            elif 'subscription' in query:
                info["operation_type"] = "subscription"
            
            if '$' in query:
                info["has_variables"] = True
            
            if 'fragment' in query:
                info["has_fragments"] = True
            
            # Count approximate field selections
            info["field_count"] = len(re.findall(r'\w+(?=\s*[\{\(]|\s*$)', query))
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to extract query info: {e}")
            return {"error": str(e)} 