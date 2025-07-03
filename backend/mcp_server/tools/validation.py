"""Validation tools for GraphQL queries."""

import json
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from fastmcp import FastMCP, Context
from ...services.validation_service import ValidationService


class ValidationRequest(BaseModel):
    """Request model for GraphQL validation."""
    query: str = Field(..., description="GraphQL query to validate")
    schema: Optional[str] = Field(None, description="GraphQL schema for validation")
    strict_mode: bool = Field(False, description="Enable strict validation mode")


def register_validation_tools(mcp: FastMCP, validation_service: ValidationService):
    """Register all validation-related tools."""

    @mcp.tool()
    async def validate_graphql(
        query: str,
        schema: Optional[str] = None,
        strict_mode: bool = False,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Validate GraphQL query syntax and structure.
        
        Performs comprehensive validation including syntax checking,
        field validation, type checking, and best practice analysis.
        """
        await ctx.info(f"Starting validation for GraphQL query")
        
        if not query.strip():
            return {
                "valid": False,
                "error": "Empty query provided",
                "suggestions": ["Provide a non-empty GraphQL query"]
            }
        
        try:
            start_time = time.time()
            
            # Perform basic validation
            validation_result = validation_service.validate_query(query)
            
            # Enhanced validation with schema if provided
            schema_validation = None
            if schema:
                try:
                    await ctx.info("Performing schema-based validation")
                    schema_validation = validation_service.validate_with_schema(query, schema)
                except Exception as e:
                    await ctx.warning(f"Schema validation failed: {str(e)}")
                    schema_validation = {"error": str(e)}
            
            # Style and best practice analysis
            style_analysis = validation_service.analyze_query_style(query)
            performance_hints = validation_service.get_performance_hints(query)
            
            processing_time = time.time() - start_time
            await ctx.info(f"Validation completed in {processing_time:.3f}s")
            
            return {
                "valid": validation_result.is_valid,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "suggestions": validation_result.suggestions,
                "schema_validation": schema_validation,
                "style_analysis": style_analysis,
                "performance_hints": performance_hints,
                "metadata": {
                    "query_length": len(query),
                    "has_schema": bool(schema),
                    "strict_mode": strict_mode,
                    "processing_time": processing_time
                }
            }
            
        except Exception as e:
            await ctx.error(f"Validation failed: {str(e)}")
            return {
                "valid": False,
                "error": str(e),
                "query": query,
                "suggestions": [
                    "Check GraphQL syntax",
                    "Verify field names and types",
                    "Ensure proper query structure"
                ]
            }

    @mcp.tool()
    async def validate_with_schema(
        query: str,
        schema: str,
        check_permissions: bool = False,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Validate GraphQL query against a specific schema.
        
        Performs comprehensive schema-aware validation including
        field existence, type compatibility, and argument validation.
        """
        await ctx.info("Starting schema-based validation")
        
        if not query.strip():
            return {"valid": False, "error": "Empty query provided"}
        
        if not schema.strip():
            return {"valid": False, "error": "Empty schema provided"}
        
        try:
            # Parse and validate schema first
            schema_validation = validation_service.validate_schema(schema)
            if not schema_validation.is_valid:
                return {
                    "valid": False,
                    "error": "Invalid schema provided",
                    "schema_errors": schema_validation.errors
                }
            
            await ctx.info("Schema is valid, validating query")
            
            # Validate query against schema
            result = validation_service.validate_with_schema(query, schema)
            
            # Additional checks
            field_analysis = validation_service.analyze_fields(query, schema)
            type_analysis = validation_service.analyze_types(query, schema)
            
            await ctx.info("Schema validation completed")
            
            return {
                "valid": result.is_valid,
                "errors": result.errors,
                "warnings": result.warnings,
                "suggestions": result.suggestions,
                "field_analysis": field_analysis,
                "type_analysis": type_analysis,
                "schema_info": {
                    "types_count": len(schema_validation.types) if hasattr(schema_validation, 'types') else 0,
                    "fields_checked": len(field_analysis.get("fields", [])),
                    "permissions_checked": check_permissions
                }
            }
            
        except Exception as e:
            await ctx.error(f"Schema validation failed: {str(e)}")
            return {
                "valid": False,
                "error": str(e),
                "suggestions": [
                    "Verify schema syntax is correct",
                    "Check field names exist in schema",
                    "Ensure argument types match schema",
                    "Validate query structure"
                ]
            }

    @mcp.tool()
    async def get_validation_suggestions(
        query: str,
        errors: Optional[List[str]] = None,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Get detailed suggestions for improving a GraphQL query.
        
        Analyzes common issues and provides specific recommendations
        for fixing validation errors and improving query quality.
        """
        await ctx.info("Generating validation suggestions")
        
        try:
            # Get validation results
            validation_result = validation_service.validate_query(query)
            
            # Analyze common issues
            suggestions = []
            
            if not validation_result.is_valid:
                # Specific error-based suggestions
                for error in validation_result.errors:
                    if "syntax" in error.lower():
                        suggestions.append({
                            "type": "syntax_error",
                            "issue": error,
                            "suggestion": "Check for missing braces, parentheses, or commas",
                            "example": "{ user(id: \"123\") { name email } }"
                        })
                    elif "field" in error.lower():
                        suggestions.append({
                            "type": "field_error", 
                            "issue": error,
                            "suggestion": "Verify field names exist in your schema",
                            "example": "Check schema documentation for available fields"
                        })
                    elif "argument" in error.lower():
                        suggestions.append({
                            "type": "argument_error",
                            "issue": error,
                            "suggestion": "Check argument types and required parameters",
                            "example": "user(id: \"string_value\") not user(id: 123)"
                        })
            
            # Best practice suggestions
            best_practices = []
            
            if len(query) > 1000:
                best_practices.append({
                    "type": "performance",
                    "suggestion": "Consider breaking large queries into smaller ones",
                    "impact": "Improved performance and readability"
                })
            
            if "password" in query.lower() or "secret" in query.lower():
                best_practices.append({
                    "type": "security",
                    "suggestion": "Avoid querying sensitive fields unnecessarily",
                    "impact": "Enhanced security"
                })
            
            if query.count("{") > 5:
                best_practices.append({
                    "type": "complexity",
                    "suggestion": "Consider using fragments for repeated field sets",
                    "impact": "Reduced query complexity and better maintainability"
                })
            
            await ctx.info(f"Generated {len(suggestions + best_practices)} suggestions")
            
            return {
                "query": query,
                "validation_errors": validation_result.errors,
                "error_suggestions": suggestions,
                "best_practices": best_practices,
                "overall_score": validation_result.confidence if hasattr(validation_result, 'confidence') else 0.0,
                "improvement_areas": [
                    "Syntax correctness",
                    "Field validation", 
                    "Performance optimization",
                    "Security considerations",
                    "Code maintainability"
                ]
            }
            
        except Exception as e:
            await ctx.error(f"Failed to generate suggestions: {str(e)}")
            return {
                "error": str(e),
                "suggestions": [
                    "Ensure query is valid GraphQL syntax",
                    "Check field names and types",
                    "Verify argument structure"
                ]
            }

    @mcp.tool()
    async def validate_batch_queries(
        queries: List[str],
        schema: Optional[str] = None,
        stop_on_first_error: bool = False,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Validate multiple GraphQL queries in batch.
        
        Efficiently processes multiple queries with detailed reporting
        and optional early termination on errors.
        """
        await ctx.info(f"Starting batch validation for {len(queries)} queries")
        
        if not queries:
            return {"error": "No queries provided for validation"}
        
        try:
            results = []
            total_queries = len(queries)
            errors_found = 0
            
            for i, query in enumerate(queries):
                await ctx.report_progress(i + 1, total_queries)
                
                try:
                    if schema:
                        validation_result = validation_service.validate_with_schema(query, schema)
                    else:
                        validation_result = validation_service.validate_query(query)
                    
                    result = {
                        "index": i,
                        "query": query[:100] + "..." if len(query) > 100 else query,
                        "valid": validation_result.is_valid,
                        "errors": validation_result.errors,
                        "warnings": validation_result.warnings
                    }
                    
                    if not validation_result.is_valid:
                        errors_found += 1
                        if stop_on_first_error:
                            await ctx.warning(f"Stopping batch validation at query {i + 1} due to error")
                            results.append(result)
                            break
                    
                    results.append(result)
                    
                except Exception as e:
                    errors_found += 1
                    await ctx.warning(f"Validation failed for query {i + 1}: {str(e)}")
                    results.append({
                        "index": i,
                        "query": query[:100] + "..." if len(query) > 100 else query,
                        "valid": False,
                        "error": str(e)
                    })
                    
                    if stop_on_first_error:
                        break
            
            success_rate = ((len(results) - errors_found) / len(results)) * 100 if results else 0
            
            await ctx.info(f"Batch validation completed: {len(results) - errors_found}/{len(results)} valid")
            
            return {
                "total_queries": total_queries,
                "processed": len(results),
                "valid_queries": len(results) - errors_found,
                "invalid_queries": errors_found,
                "success_rate": success_rate,
                "results": results,
                "summary": {
                    "stopped_early": len(results) < total_queries,
                    "common_errors": self._analyze_common_errors([r for r in results if not r.get("valid", True)])
                }
            }
            
        except Exception as e:
            await ctx.error(f"Batch validation failed: {str(e)}")
            return {
                "error": str(e),
                "total_queries": len(queries),
                "processed": 0
            }

    def _analyze_common_errors(self, failed_results: List[Dict]) -> List[str]:
        """Analyze common errors across failed validations."""
        error_patterns = {}
        
        for result in failed_results:
            errors = result.get("errors", [])
            for error in errors:
                # Categorize errors
                if "syntax" in error.lower():
                    error_patterns["syntax_errors"] = error_patterns.get("syntax_errors", 0) + 1
                elif "field" in error.lower():
                    error_patterns["field_errors"] = error_patterns.get("field_errors", 0) + 1
                elif "type" in error.lower():
                    error_patterns["type_errors"] = error_patterns.get("type_errors", 0) + 1
                else:
                    error_patterns["other_errors"] = error_patterns.get("other_errors", 0) + 1
        
        return [f"{category}: {count}" for category, count in error_patterns.items()]

    @mcp.tool()
    async def analyze_query_complexity(
        query: str,
        max_depth: int = 10,
        max_nodes: int = 100,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Analyze GraphQL query complexity and performance characteristics.
        
        Provides detailed analysis of query depth, node count, and
        potential performance issues with recommendations.
        """
        await ctx.info("Analyzing query complexity")
        
        try:
            # Calculate complexity metrics
            complexity_analysis = validation_service.analyze_complexity(query)
            
            # Performance assessment
            performance_score = 100
            issues = []
            recommendations = []
            
            depth = complexity_analysis.get("depth", 0)
            if depth > max_depth:
                performance_score -= 20
                issues.append(f"Query depth ({depth}) exceeds recommended maximum ({max_depth})")
                recommendations.append("Consider breaking deep nested queries into multiple requests")
            
            node_count = complexity_analysis.get("node_count", 0)
            if node_count > max_nodes:
                performance_score -= 15
                issues.append(f"Node count ({node_count}) exceeds recommended maximum ({max_nodes})")
                recommendations.append("Reduce the number of fields or use pagination")
            
            if "password" in query.lower() or "secret" in query.lower():
                performance_score -= 10
                issues.append("Potentially sensitive fields detected")
                recommendations.append("Avoid querying sensitive data unless necessary")
            
            await ctx.info(f"Complexity analysis completed. Performance score: {performance_score}")
            
            return {
                "query": query,
                "complexity_metrics": complexity_analysis,
                "performance_score": performance_score,
                "issues": issues,
                "recommendations": recommendations,
                "limits": {
                    "max_depth": max_depth,
                    "max_nodes": max_nodes,
                    "current_depth": depth,
                    "current_nodes": node_count
                },
                "assessment": {
                    "performance_rating": "excellent" if performance_score >= 90 else 
                                        "good" if performance_score >= 70 else
                                        "needs_improvement" if performance_score >= 50 else "poor",
                    "optimization_needed": len(recommendations) > 0
                }
            }
            
        except Exception as e:
            await ctx.error(f"Complexity analysis failed: {str(e)}")
            return {
                "error": str(e),
                "query": query,
                "suggestions": [
                    "Ensure query is valid GraphQL",
                    "Check for syntax errors",
                    "Simplify complex nested structures"
                ]
            } 