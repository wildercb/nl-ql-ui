"""
Practical examples of custom validation services for the MPPW MCP system.
Run these examples to see how custom validation works in practice.
"""

import asyncio
from typing import List, Optional
from backend.services.validation_service import ValidationService, ValidationResult


class EcommerceValidationService(ValidationService):
    """E-commerce specific GraphQL validation."""
    
    def __init__(self):
        super().__init__()
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
        import re
        if re.search(r'products?\s*\{', query) and 'first' not in query and 'limit' not in query:
            warnings.append("Product list queries should include pagination (first/limit)")
        
        # Check for price queries without currency
        if 'price' in query.lower() and 'currency' not in query.lower():
            warnings.append("Price queries should include currency information")
        
        return warnings
    
    async def validate_query(self, query: str, schema: Optional[str] = None) -> ValidationResult:
        """Enhanced validation with e-commerce rules."""
        # Get standard validation result
        result = await super().validate_query(query, schema)
        
        # Add e-commerce specific warnings
        ecommerce_warnings = self._get_ecommerce_warnings(query)
        
        return ValidationResult(
            is_valid=result.is_valid,
            errors=result.errors,
            warnings=result.warnings + ecommerce_warnings,
            suggestions=result.suggestions + [
                "Consider adding product images with alt text",
                "Include SEO-friendly URLs in product data"
            ],
            parsed_query=result.parsed_query
        )


class PerformanceValidationService(ValidationService):
    """Performance-focused GraphQL validation."""
    
    def __init__(self, max_depth: int = 6, max_fields: int = 20):
        super().__init__()
        self.max_depth = max_depth
        self.max_fields = max_fields
    
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
    
    async def validate_query(self, query: str, schema: Optional[str] = None) -> ValidationResult:
        """Enhanced validation with performance checks."""
        result = await super().validate_query(query, schema)
        
        # Check query depth
        depth = self._calculate_query_depth(query)
        if depth > self.max_depth:
            result.errors.append(f"Query depth ({depth}) exceeds maximum allowed ({self.max_depth})")
        elif depth > self.max_depth - 2:
            result.warnings.append(f"Query depth ({depth}) is close to maximum limit")
        
        # Check field count (simplified)
        import re
        field_count = len(re.findall(r'\w+(?=\s*[\{\(]|\s*$)', query))
        if field_count > self.max_fields:
            result.warnings.append(f"Query selects many fields ({field_count}), consider using fragments")
        
        return result


async def example_standard_validation():
    """Example of standard validation."""
    print("=== Standard Validation Example ===")
    
    validator = ValidationService()
    
    # Test various queries
    test_queries = [
        # Valid query
        """
        query GetUser {
            user(id: "123") {
                id
                name
                email
            }
        }
        """,
        
        # Invalid query - unbalanced braces
        """
        query GetUser {
            user(id: "123") {
                id
                name
                email
            
        """,
        
        # Query with style issues
        """
        query {
            user {
                profile {
                    settings {
                        preferences {
                            advanced {
                                options {
                                    details {
                                        data {
                                            nested
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        """
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i} ---")
        result = await validator.validate_query(query.strip())
        
        print(f"Valid: {result.is_valid}")
        if result.errors:
            print(f"Errors: {result.errors}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        if result.suggestions:
            print(f"Suggestions: {result.suggestions}")


async def example_ecommerce_validation():
    """Example of e-commerce specific validation."""
    print("\n=== E-commerce Validation Example ===")
    
    validator = EcommerceValidationService()
    
    # E-commerce test queries
    ecommerce_queries = [
        # Good e-commerce query
        """
        query GetProducts($first: Int, $after: String) {
            products(first: $first, after: $after) {
                edges {
                    node {
                        id
                        name
                        price {
                            amount
                            currency
                        }
                        availability {
                            inStock
                            quantity
                        }
                    }
                }
                pageInfo {
                    hasNextPage
                    endCursor
                }
            }
        }
        """,
        
        # Problematic e-commerce query
        """
        query GetProducts {
            products {
                id
                name
                price
            }
        }
        """
    ]
    
    for i, query in enumerate(ecommerce_queries, 1):
        print(f"\n--- E-commerce Query {i} ---")
        result = await validator.validate_query(query.strip())
        
        print(f"Valid: {result.is_valid}")
        if result.errors:
            print(f"Errors: {result.errors}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        if result.suggestions:
            print(f"Suggestions: {result.suggestions[:2]}")  # Show first 2 suggestions


async def example_performance_validation():
    """Example of performance-focused validation."""
    print("\n=== Performance Validation Example ===")
    
    validator = PerformanceValidationService(max_depth=5, max_fields=15)
    
    # Deep nested query
    deep_query = """
    query DeepQuery {
        user {
            profile {
                preferences {
                    settings {
                        advanced {
                            options {
                                details {
                                    data
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    """
    
    print("--- Deep Nested Query ---")
    result = await validator.validate_query(deep_query.strip())
    
    print(f"Valid: {result.is_valid}")
    if result.errors:
        print(f"Errors: {result.errors}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")


async def example_comparative_validation():
    """Compare different validation approaches."""
    print("\n=== Comparative Validation Example ===")
    
    test_query = """
    query GetProducts {
        products {
            id
            name
            price
        }
    }
    """
    
    validators = {
        "Standard": ValidationService(),
        "E-commerce": EcommerceValidationService(),
        "Performance": PerformanceValidationService(max_depth=3)
    }
    
    print("Query:", test_query.strip())
    print("\nValidation Results:")
    
    for name, validator in validators.items():
        result = await validator.validate_query(test_query.strip())
        print(f"\n{name} Validator:")
        print(f"  Valid: {result.is_valid}")
        print(f"  Warnings: {len(result.warnings)}")
        print(f"  Suggestions: {len(result.suggestions)}")
        
        if result.warnings:
            print(f"  First warning: {result.warnings[0]}")


async def main():
    """Run all validation examples."""
    await example_standard_validation()
    await example_ecommerce_validation()
    await example_performance_validation()
    await example_comparative_validation()


if __name__ == "__main__":
    asyncio.run(main()) 