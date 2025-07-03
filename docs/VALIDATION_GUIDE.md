# GraphQL Validation Service Guide

This guide covers the default validation rules in the MPPW MCP system and how to customize them for your specific use case.

## Default Validation Rules

The validation service implements multiple layers of validation:

### 1. Standard GraphQL Validation Rules
```python
default_rules = [
    "NoUnusedFragments",           # Remove unused fragment definitions
    "NoUndefinedVariables",        # Ensure all variables are defined
    "KnownArgumentNames",          # Validate argument names exist
    "KnownDirectives",             # Ensure directives are valid
    "KnownFragmentNames",          # Validate fragment references
    "KnownTypeNames",              # Ensure type names exist in schema
    "LoneAnonymousOperation",      # Only one anonymous operation per document
    "NoFragmentCycles",            # Prevent circular fragment references
    "OverlappingFieldsCanBeMerged", # Handle field merging conflicts
    "PossibleFragmentSpreads",     # Validate fragment usage context
    "ProvidedRequiredArguments",   # Ensure required args are provided
    "ScalarLeafs",                 # Scalar fields cannot have selections
    "SingleFieldSubscriptions",    # Subscriptions must have single root field
    "UniqueArgumentNames",         # Argument names must be unique
    "UniqueDirectivesPerLocation", # No duplicate directives
    "UniqueFragmentNames",         # Fragment names must be unique
    "UniqueInputFieldNames",       # Input field names must be unique
    "UniqueOperationNames",        # Operation names must be unique
    "UniqueVariableNames",         # Variable names must be unique
    "ValuesOfCorrectType",         # Values must match expected types
    "VariablesAreInputTypes",      # Variables must be input types
    "VariablesInAllowedPosition"   # Variables must be in valid positions
]
```

### 2. Basic Syntax Checks
- **Balanced braces, parentheses, brackets**
- **Proper string quote matching**
- **Non-empty query validation**

### 3. Style & Best Practice Warnings
- **Missing operation names** - "Consider adding an operation name for better debugging"
- **Deep nesting detection** - Warns when query depth > 8 levels
- **Missing variables** - When arguments exist but no variables defined
- **Over-fetching potential** - Too many fields at same level

### 4. Smart Suggestions
- **Syntax error recovery** - Suggests fixes for common syntax issues
- **Operation type specification** - Recommends explicit query/mutation/subscription
- **Formatting suggestions** - Multi-line formatting for readability

## Customizing Validation Rules

### 1. Domain-Specific Validation

Create custom validators for your specific domain:

```python
# backend/services/custom_validation.py
from .validation_service import ValidationService, ValidationResult
from typing import List, Optional
import re

class EcommerceValidationService(ValidationService):
    """E-commerce specific GraphQL validation."""
    
    def __init__(self):
        super().__init__()
        # Add e-commerce specific rules
        self.ecommerce_rules = [
            "RequireProductAvailability",
            "LimitProductFields",
            "RequirePagination",
            "ValidatePriceQueries"
        ]
    
    def _get_ecommerce_warnings(self, query: str) -> List[str]:
        """Get e-commerce specific warnings."""
        warnings = []
        
        # Check for product queries without availability
        if 'product' in query.lower() and 'available' not in query.lower():
            warnings.append("Product queries should include availability status")
        
        # Check for missing pagination on list queries
        if re.search(r'products?\s*\{', query) and 'first' not in query and 'limit' not in query:
            warnings.append("Product list queries should include pagination (first/limit)")
        
        # Check for price queries without currency
        if 'price' in query.lower() and 'currency' not in query.lower():
            warnings.append("Price queries should include currency information")
        
        # Check for inventory queries without location
        if 'inventory' in query.lower() and 'location' not in query.lower():
            warnings.append("Inventory queries should specify location/warehouse")
        
        return warnings
    
    def _get_ecommerce_suggestions(self, query: str) -> List[str]:
        """Get e-commerce specific suggestions."""
        suggestions = []
        
        if 'product' in query.lower():
            suggestions.extend([
                "Consider adding product images with alt text",
                "Include SEO-friendly URLs in product data",
                "Add product category and brand information"
            ])
        
        if 'user' in query.lower() or 'customer' in query.lower():
            suggestions.extend([
                "Include customer preferences for personalization",
                "Add loyalty program status if applicable"
            ])
        
        return suggestions
    
    async def validate_query(self, query: str, schema: Optional[str] = None) -> ValidationResult:
        """Enhanced validation with e-commerce rules."""
        # Get standard validation result
        result = await super().validate_query(query, schema)
        
        # Add e-commerce specific warnings and suggestions
        ecommerce_warnings = self._get_ecommerce_warnings(query)
        ecommerce_suggestions = self._get_ecommerce_suggestions(query)
        
        return ValidationResult(
            is_valid=result.is_valid,
            errors=result.errors,
            warnings=result.warnings + ecommerce_warnings,
            suggestions=result.suggestions + ecommerce_suggestions,
            parsed_query=result.parsed_query
        )
```

### 2. Performance-Focused Validation

```python
class PerformanceValidationService(ValidationService):
    """Performance-focused GraphQL validation."""
    
    def __init__(self, max_depth: int = 6, max_fields: int = 20):
        super().__init__()
        self.max_depth = max_depth
        self.max_fields = max_fields
    
    def _check_performance_issues(self, query: str) -> tuple[List[str], List[str]]:
        """Check for performance-related issues."""
        errors = []
        warnings = []
        
        # Check query depth
        depth = self._calculate_query_depth(query)
        if depth > self.max_depth:
            errors.append(f"Query depth ({depth}) exceeds maximum allowed ({self.max_depth})")
        elif depth > self.max_depth - 2:
            warnings.append(f"Query depth ({depth}) is close to maximum limit")
        
        # Check field count
        field_count = len(re.findall(r'\w+(?=\s*[\{\(]|\s*$)', query))
        if field_count > self.max_fields:
            warnings.append(f"Query selects many fields ({field_count}), consider using fragments")
        
        # Check for N+1 potential
        if self._has_n_plus_one_potential(query):
            warnings.append("Query structure may cause N+1 queries, consider using fragments or batching")
        
        # Check for missing aliases on multiple same fields
        if self._has_duplicate_fields_without_aliases(query):
            errors.append("Multiple selections of same field require aliases")
        
        return errors, warnings
    
    def _calculate_query_depth(self, query: str) -> int:
        """Calculate the maximum nesting depth of the query."""
        max_depth = 0
        current_depth = 0
        
        for char in query:
            if char == '{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == '}':
                current_depth -= 1
        
        return max_depth
    
    def _has_n_plus_one_potential(self, query: str) -> bool:
        """Detect potential N+1 query patterns."""
        # Look for list fields with nested object selections
        pattern = r'\w+\s*\{\s*\w+\s*\{\s*\w+'
        return bool(re.search(pattern, query))
    
    def _has_duplicate_fields_without_aliases(self, query: str) -> bool:
        """Check for duplicate fields without aliases."""
        # This is a simplified check - would need AST parsing for accuracy
        lines = query.split('\n')
        field_names = []
        
        for line in lines:
            stripped = line.strip()
            if ':' not in stripped and '{' not in stripped and '}' not in stripped:
                field_match = re.match(r'^(\w+)', stripped)
                if field_match:
                    field_names.append(field_match.group(1))
        
        return len(field_names) != len(set(field_names))
```

### 3. Security-Focused Validation

```python
class SecurityValidationService(ValidationService):
    """Security-focused GraphQL validation."""
    
    def __init__(self):
        super().__init__()
        self.sensitive_fields = ['password', 'token', 'secret', 'key', 'hash']
        self.admin_only_fields = ['adminNotes', 'internalId', 'systemFlags']
    
    def _check_security_issues(self, query: str) -> tuple[List[str], List[str]]:
        """Check for security-related issues."""
        errors = []
        warnings = []
        
        # Check for sensitive field selections
        for field in self.sensitive_fields:
            if field in query.lower():
                errors.append(f"Sensitive field '{field}' should not be queried directly")
        
        # Check for admin-only fields (would need user context in real implementation)
        for field in self.admin_only_fields:
            if field in query:
                warnings.append(f"Admin field '{field}' requires elevated permissions")
        
        # Check for potential injection patterns
        if self._has_injection_patterns(query):
            errors.append("Query contains potentially malicious patterns")
        
        # Check for overly broad queries
        if self._is_overly_broad_query(query):
            warnings.append("Query selects many fields - ensure this is intentional")
        
        return errors, warnings
    
    def _has_injection_patterns(self, query: str) -> bool:
        """Check for potential GraphQL injection patterns."""
        suspicious_patterns = [
            r'__schema',     # Introspection attempts
            r'__type',       # Type introspection
            r'\.\.\.',       # Path traversal attempts
            r'union.*select', # SQL injection patterns
            r'script.*>',    # XSS attempts
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        return False
    
    def _is_overly_broad_query(self, query: str) -> bool:
        """Detect queries that select too many fields."""
        # Count top-level selections
        selections = len(re.findall(r'^\s*\w+', query, re.MULTILINE))
        return selections > 15
```

## Configuration-Based Validation

### 1. Configurable Rules

```python
# backend/config/validation_config.py
from pydantic import BaseSettings
from typing import List, Dict, Any

class ValidationSettings(BaseSettings):
    """Validation configuration settings."""
    
    # Performance limits
    max_query_depth: int = 8
    max_field_count: int = 50
    max_aliases: int = 20
    
    # Security settings
    allow_introspection: bool = False
    blocked_fields: List[str] = ['password', 'secret', 'token']
    admin_only_fields: List[str] = ['adminNotes', 'systemFlags']
    
    # Domain-specific rules
    domain_rules: Dict[str, Any] = {
        "ecommerce": {
            "require_product_availability": True,
            "require_pagination": True,
            "require_currency_with_price": True
        },
        "social": {
            "require_privacy_checks": True,
            "limit_friend_depth": 3,
            "require_content_warnings": True
        }
    }
    
    # Custom validation messages
    custom_messages: Dict[str, str] = {
        "deep_query": "Query depth exceeds recommended limit for performance",
        "missing_pagination": "List queries should include pagination for better UX",
        "sensitive_field": "Sensitive field access requires special handling"
    }
    
    class Config:
        env_prefix = "VALIDATION_"
```

### 2. Dynamic Rule Loading

```python
class ConfigurableValidationService(ValidationService):
    """Validation service with configurable rules."""
    
    def __init__(self, config: ValidationSettings):
        super().__init__()
        self.config = config
        self.domain = config.domain_rules
    
    async def validate_query(self, query: str, schema: Optional[str] = None, 
                           domain: str = "default") -> ValidationResult:
        """Validate with domain-specific rules."""
        result = await super().validate_query(query, schema)
        
        # Apply domain-specific validation
        if domain in self.domain:
            domain_errors, domain_warnings = self._apply_domain_rules(query, domain)
            result.errors.extend(domain_errors)
            result.warnings.extend(domain_warnings)
        
        return result
    
    def _apply_domain_rules(self, query: str, domain: str) -> tuple[List[str], List[str]]:
        """Apply domain-specific validation rules."""
        errors = []
        warnings = []
        rules = self.domain.get(domain, {})
        
        if rules.get("require_product_availability") and 'product' in query.lower():
            if 'available' not in query.lower() and 'inStock' not in query.lower():
                warnings.append(self.config.custom_messages.get(
                    "missing_availability", 
                    "Product queries should include availability"
                ))
        
        if rules.get("require_pagination"):
            if self._is_list_query(query) and not self._has_pagination(query):
                warnings.append(self.config.custom_messages.get(
                    "missing_pagination",
                    "List queries should include pagination"
                ))
        
        return errors, warnings
```

## Usage Examples

### 1. Using Custom Validation Service

```python
# In your application
from backend.services.custom_validation import EcommerceValidationService

# Initialize with custom validator
validation_service = EcommerceValidationService()

# Validate e-commerce query
result = await validation_service.validate_query("""
    query GetProducts {
        products {
            id
            name
            price
        }
    }
""")

# Result will include e-commerce-specific warnings:
# - "Product queries should include availability status"
# - "Product list queries should include pagination (first/limit)"
# - "Price queries should include currency information"
```

### 2. Performance Validation

```python
# Initialize with performance limits
perf_validator = PerformanceValidationService(max_depth=5, max_fields=15)

# This query would trigger warnings/errors
complex_query = """
    query DeepQuery {
        user {
            profile {
                preferences {
                    settings {
                        advanced {
                            options {
                                details {
                                    data # Depth = 7, exceeds limit of 5
                                }
                            }
                        }
                    }
                }
            }
        }
    }
"""

result = await perf_validator.validate_query(complex_query)
# Would return: "Query depth (7) exceeds maximum allowed (5)"
```

### 3. Configuration-Based Validation

```python
# Load configuration
config = ValidationSettings(
    max_query_depth=6,
    domain_rules={
        "ecommerce": {
            "require_product_availability": True,
            "require_pagination": True
        }
    }
)

# Initialize configurable validator
validator = ConfigurableValidationService(config)

# Validate with domain context
result = await validator.validate_query(query, domain="ecommerce")
```

## Integration with Translation Service

You can integrate custom validation with the translation service:

```python
# backend/services/translation_service.py (modification)
class TranslationService:
    def __init__(self, validation_service: ValidationService = None):
        self.validation_service = validation_service or ValidationService()
    
    async def translate_to_graphql(self, natural_query: str, **kwargs) -> TranslationResult:
        # ... translation logic ...
        
        # Validate the generated query
        validation_result = await self.validation_service.validate_query(
            translated_query, 
            schema_context
        )
        
        # Include validation feedback in translation result
        return TranslationResult(
            graphql_query=translated_query,
            confidence=confidence_score,
            validation_result=validation_result,
            warnings=validation_result.warnings,
            suggestions=validation_result.suggestions
        )
```

This validation system provides a flexible foundation that you can customize for your specific domain, performance requirements, and security needs. 