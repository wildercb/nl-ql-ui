"""Schema introspection and analysis tools."""

import json
from typing import Dict, Any, List, Optional
import httpx

from fastmcp import FastMCP, Context


def register_schema_tools(mcp: FastMCP):
    """Register all schema-related tools."""

    @mcp.tool()
    async def introspect_schema(
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Introspect a GraphQL schema from an endpoint.
        
        Retrieves the full schema definition including types, fields,
        and relationships for better translation context.
        """
        await ctx.info(f"Starting schema introspection for: {endpoint}")
        
        introspection_query = """
        query IntrospectionQuery {
          __schema {
            queryType { name }
            mutationType { name }
            subscriptionType { name }
            types {
              ...FullType
            }
          }
        }
        
        fragment FullType on __Type {
          kind
          name
          description
          fields(includeDeprecated: true) {
            name
            description
            args {
              ...InputValue
            }
            type {
              ...TypeRef
            }
            isDeprecated
            deprecationReason
          }
          inputFields {
            ...InputValue
          }
          interfaces {
            ...TypeRef
          }
          enumValues(includeDeprecated: true) {
            name
            description
            isDeprecated
            deprecationReason
          }
          possibleTypes {
            ...TypeRef
          }
        }
        
        fragment InputValue on __InputValue {
          name
          description
          type { ...TypeRef }
          defaultValue
        }
        
        fragment TypeRef on __Type {
          kind
          name
          ofType {
            kind
            name
            ofType {
              kind
              name
              ofType {
                kind
                name
                ofType {
                  kind
                  name
                  ofType {
                    kind
                    name
                    ofType {
                      kind
                      name
                      ofType {
                        kind
                        name
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    endpoint,
                    json={"query": introspection_query},
                    headers=headers or {}
                )
                
                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                
                result = response.json()
                
                if "errors" in result:
                    raise Exception(f"GraphQL errors: {result['errors']}")
                
                schema_data = result["data"]["__schema"]
                
                # Analyze the schema
                analysis = await _analyze_schema_structure(schema_data, ctx)
                
                await ctx.info("Schema introspection completed successfully")
                
                return {
                    "schema": schema_data,
                    "analysis": analysis,
                    "endpoint": endpoint,
                    "success": True
                }
                
        except Exception as e:
            await ctx.error(f"Schema introspection failed: {str(e)}")
            return {
                "error": str(e),
                "endpoint": endpoint,
                "success": False,
                "suggestions": [
                    "Verify the endpoint URL is correct",
                    "Check if authentication headers are needed",
                    "Ensure the endpoint supports introspection",
                    "Try with a longer timeout"
                ]
            }

    @mcp.tool()
    async def analyze_schema(
        schema_text: str,
        include_examples: bool = True,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Analyze a GraphQL schema and provide insights.
        
        Parses schema text and provides detailed analysis including
        complexity metrics, relationships, and usage examples.
        """
        await ctx.info("Starting schema analysis")
        
        try:
            from graphql import build_schema, introspection_from_schema
            
            # Parse the schema
            schema = build_schema(schema_text)
            introspection = introspection_from_schema(schema)
            schema_data = introspection["data"]["__schema"]
            
            # Perform analysis
            analysis = await _analyze_schema_structure(schema_data, ctx, include_examples)
            
            await ctx.info("Schema analysis completed")
            
            return {
                "analysis": analysis,
                "schema_valid": True,
                "success": True
            }
            
        except Exception as e:
            await ctx.error(f"Schema analysis failed: {str(e)}")
            return {
                "error": str(e),
                "schema_valid": False,
                "success": False,
                "suggestions": [
                    "Verify the schema syntax is correct",
                    "Check for missing type definitions",
                    "Ensure proper GraphQL formatting"
                ]
            }

    @mcp.tool()
    async def generate_query_examples(
        schema_text: str,
        example_count: int = 5,
        complexity_level: str = "medium",
        ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Generate example queries based on a GraphQL schema.
        
        Creates realistic query examples that demonstrate how to use
        the schema effectively for different use cases.
        """
        await ctx.info(f"Generating {example_count} example queries")
        
        try:
            from graphql import build_schema, introspection_from_schema
            
            schema = build_schema(schema_text)
            introspection = introspection_from_schema(schema)
            schema_data = introspection["data"]["__schema"]
            
            # Generate examples based on complexity level
            examples = await _generate_schema_examples(
                schema_data, example_count, complexity_level, ctx
            )
            
            await ctx.info(f"Generated {len(examples)} example queries")
            
            return {
                "examples": examples,
                "complexity_level": complexity_level,
                "count": len(examples),
                "success": True
            }
            
        except Exception as e:
            await ctx.error(f"Example generation failed: {str(e)}")
            return {
                "error": str(e),
                "success": False,
                "suggestions": [
                    "Verify schema is valid",
                    "Try a lower complexity level",
                    "Reduce the example count"
                ]
            }


async def _analyze_schema_structure(schema_data: Dict, ctx: Context, include_examples: bool = False) -> Dict[str, Any]:
    """Analyze schema structure and provide insights."""
    await ctx.info("Analyzing schema structure")
    
    types = schema_data.get("types", [])
    query_type = schema_data.get("queryType", {}).get("name", "Query")
    mutation_type = schema_data.get("mutationType", {}).get("name") if schema_data.get("mutationType") else None
    
    # Categorize types
    object_types = [t for t in types if t["kind"] == "OBJECT" and not t["name"].startswith("__")]
    input_types = [t for t in types if t["kind"] == "INPUT_OBJECT"]
    enum_types = [t for t in types if t["kind"] == "ENUM"]
    scalar_types = [t for t in types if t["kind"] == "SCALAR" and not t["name"].startswith("__")]
    
    # Find the Query type
    query_fields = []
    for type_def in object_types:
        if type_def["name"] == query_type:
            query_fields = type_def.get("fields", [])
            break
    
    # Analyze complexity
    total_fields = sum(len(t.get("fields", [])) for t in object_types)
    avg_fields_per_type = total_fields / len(object_types) if object_types else 0
    
    analysis = {
        "summary": {
            "total_types": len(types),
            "object_types": len(object_types),
            "input_types": len(input_types),
            "enum_types": len(enum_types),
            "scalar_types": len(scalar_types),
            "total_fields": total_fields,
            "avg_fields_per_type": round(avg_fields_per_type, 2),
            "has_mutations": mutation_type is not None
        },
        "query_operations": [
            {
                "name": field["name"],
                "description": field.get("description", ""),
                "args": len(field.get("args", [])),
                "return_type": _extract_type_name(field["type"])
            }
            for field in query_fields[:10]  # Limit to first 10
        ],
        "common_patterns": _identify_common_patterns(object_types),
        "recommendations": _generate_recommendations(schema_data, object_types)
    }
    
    if include_examples:
        analysis["example_queries"] = await _generate_simple_examples(query_fields[:5], ctx)
    
    return analysis


def _extract_type_name(type_ref: Dict) -> str:
    """Extract the actual type name from a type reference."""
    if type_ref.get("name"):
        return type_ref["name"]
    elif type_ref.get("ofType"):
        return _extract_type_name(type_ref["ofType"])
    else:
        return "Unknown"


def _identify_common_patterns(object_types: List[Dict]) -> List[str]:
    """Identify common GraphQL patterns in the schema."""
    patterns = []
    
    # Check for pagination pattern
    for type_def in object_types:
        if "Connection" in type_def["name"] or "Edge" in type_def["name"]:
            patterns.append("Relay-style pagination")
            break
    
    # Check for common field names
    all_field_names = []
    for type_def in object_types:
        all_field_names.extend([f["name"] for f in type_def.get("fields", [])])
    
    if "id" in all_field_names:
        patterns.append("ID fields present")
    if "createdAt" in all_field_names or "created_at" in all_field_names:
        patterns.append("Timestamp tracking")
    if any("user" in name.lower() for name in all_field_names):
        patterns.append("User relationships")
    
    return patterns


def _generate_recommendations(schema_data: Dict, object_types: List[Dict]) -> List[str]:
    """Generate recommendations for using the schema effectively."""
    recommendations = []
    
    query_type = schema_data.get("queryType", {}).get("name", "Query")
    
    # Find Query type operations
    query_operations = []
    for type_def in object_types:
        if type_def["name"] == query_type:
            query_operations = type_def.get("fields", [])
            break
    
    if query_operations:
        recommendations.append(f"Start with basic queries using fields like: {', '.join([op['name'] for op in query_operations[:3]])}")
    
    if schema_data.get("mutationType"):
        recommendations.append("Schema supports mutations for data modification")
    
    if len(object_types) > 10:
        recommendations.append("Complex schema - consider using fragments for repeated field sets")
    
    return recommendations


async def _generate_simple_examples(query_fields: List[Dict], ctx: Context) -> List[Dict]:
    """Generate simple query examples."""
    examples = []
    
    for field in query_fields:
        field_name = field["name"]
        args = field.get("args", [])
        return_type = _extract_type_name(field["type"])
        
        # Generate a simple query
        if not args:
            query = f"{{ {field_name} }}"
        else:
            # Create example with first argument
            first_arg = args[0]
            arg_name = first_arg["name"]
            arg_type = _extract_type_name(first_arg["type"])
            
            if arg_type in ["String", "ID"]:
                example_value = '"example"'
            elif arg_type in ["Int", "Float"]:
                example_value = "1"
            elif arg_type == "Boolean":
                example_value = "true"
            else:
                example_value = '"example"'
            
            query = f"{{ {field_name}({arg_name}: {example_value}) }}"
        
        examples.append({
            "description": f"Query {field_name}",
            "query": query,
            "field": field_name,
            "return_type": return_type
        })
    
    return examples


async def _generate_schema_examples(schema_data: Dict, count: int, complexity: str, ctx: Context) -> List[Dict]:
    """Generate more complex schema examples."""
    # This is a simplified version - in practice, you'd want more sophisticated example generation
    examples = []
    
    types = schema_data.get("types", [])
    object_types = [t for t in types if t["kind"] == "OBJECT" and not t["name"].startswith("__")]
    
    # Generate examples based on complexity
    if complexity == "simple":
        examples = await _generate_simple_examples(
            object_types[0].get("fields", [])[:count] if object_types else [], ctx
        )
    elif complexity == "medium":
        # Add nested field examples
        for i, obj_type in enumerate(object_types[:count]):
            fields = obj_type.get("fields", [])[:3]
            field_list = " ".join([f["name"] for f in fields])
            
            examples.append({
                "description": f"Query {obj_type['name']} with multiple fields",
                "query": f"{{ {obj_type['name'].lower()} {{ {field_list} }} }}",
                "complexity": "medium"
            })
    else:  # complex
        # Add deeply nested examples
        for i, obj_type in enumerate(object_types[:count]):
            examples.append({
                "description": f"Complex nested query for {obj_type['name']}",
                "query": f"{{ {obj_type['name'].lower()} {{ id name ... on {obj_type['name']} {{ description }} }} }}",
                "complexity": "complex"
            })
    
    return examples 