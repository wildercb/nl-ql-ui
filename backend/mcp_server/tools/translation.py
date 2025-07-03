"""Translation tools for converting natural language to GraphQL."""

import json
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from fastmcp import FastMCP, Context
from ...services.translation_service import TranslationService


class TranslationRequest(BaseModel):
    """Request model for GraphQL translation."""
    natural_query: str = Field(..., description="Natural language query to translate")
    schema_context: Optional[str] = Field(None, description="GraphQL schema context for better translation")
    model: Optional[str] = Field(None, description="AI model to use for translation")
    temperature: Optional[float] = Field(0.3, description="Temperature for AI model (0.0-1.0)")
    max_tokens: Optional[int] = Field(1000, description="Maximum tokens for response")


class BatchTranslationRequest(BaseModel):
    """Request model for batch translation."""
    queries: List[str] = Field(..., description="List of natural language queries to translate")
    schema_context: Optional[str] = Field(None, description="GraphQL schema context")
    model: Optional[str] = Field(None, description="AI model to use")
    concurrent_limit: Optional[int] = Field(3, description="Maximum concurrent translations")


def register_translation_tools(mcp: FastMCP, translation_service: TranslationService):
    """Register all translation-related tools."""

    @mcp.tool()
    async def translate_query(
        natural_query: str,
        schema_context: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = 0.3,
        max_tokens: Optional[int] = 1000,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Translate a natural language query to GraphQL.
        
        This is the primary translation tool that converts human-readable
        queries into valid GraphQL syntax with explanations and confidence scores.
        """
        await ctx.info(f"Starting translation for query: {natural_query[:50]}...")
        
        try:
            start_time = time.time()
            
            # Perform the translation
            result = await translation_service.translate_to_graphql(
                natural_query=natural_query,
                schema_context=schema_context or "",
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            processing_time = time.time() - start_time
            await ctx.info(f"Translation completed in {processing_time:.2f}s")
            
            return {
                "graphql_query": result.graphql_query,
                "confidence": result.confidence,
                "explanation": result.explanation,
                "model_used": result.model_used,
                "processing_time": processing_time,
                "suggestions": result.suggestions if hasattr(result, 'suggestions') else [],
                "metadata": {
                    "natural_query": natural_query,
                    "has_schema_context": bool(schema_context),
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            }
            
        except Exception as e:
            await ctx.error(f"Translation failed: {str(e)}")
            return {
                "error": str(e),
                "natural_query": natural_query,
                "suggestions": [
                    "Check if the Ollama service is running",
                    "Verify the model is available",
                    "Try a simpler query",
                    "Provide schema context for better results"
                ]
            }

    @mcp.tool() 
    async def batch_translate(
        queries: List[str],
        schema_context: Optional[str] = None,
        model: Optional[str] = None,
        concurrent_limit: int = 3,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Translate multiple natural language queries to GraphQL in batch.
        
        Efficiently processes multiple queries with concurrency control
        and detailed progress reporting.
        """
        await ctx.info(f"Starting batch translation for {len(queries)} queries")
        
        if not queries:
            return {"error": "No queries provided for translation"}
        
        if len(queries) > 20:
            await ctx.warning("Large batch detected. Consider splitting into smaller batches.")
        
        try:
            import asyncio
            from itertools import islice
            
            results = []
            total_queries = len(queries)
            
            # Process in chunks to respect concurrent_limit
            async def translate_chunk(chunk_queries: List[str], chunk_start: int):
                chunk_results = []
                tasks = []
                
                for i, query in enumerate(chunk_queries):
                    task = translation_service.translate_to_graphql(
                        natural_query=query,
                        schema_context=schema_context or "",
                        model=model
                    )
                    tasks.append((i + chunk_start, query, task))
                
                # Execute with concurrency limit
                semaphore = asyncio.Semaphore(concurrent_limit)
                
                async def translate_with_semaphore(index, query, task):
                    async with semaphore:
                        try:
                            result = await task
                            await ctx.report_progress(index + 1, total_queries)
                            return {
                                "index": index,
                                "natural_query": query,
                                "graphql_query": result.graphql_query,
                                "confidence": result.confidence,
                                "explanation": result.explanation,
                                "success": True
                            }
                        except Exception as e:
                            await ctx.warning(f"Failed to translate query {index + 1}: {str(e)}")
                            return {
                                "index": index,
                                "natural_query": query,
                                "error": str(e),
                                "success": False
                            }
                
                chunk_results = await asyncio.gather(*[
                    translate_with_semaphore(idx, query, task)
                    for idx, query, task in tasks
                ])
                
                return chunk_results
            
            # Process all queries in chunks
            for i in range(0, len(queries), concurrent_limit * 2):
                chunk = queries[i:i + concurrent_limit * 2]
                chunk_results = await translate_chunk(chunk, i)
                results.extend(chunk_results)
            
            # Calculate summary statistics
            successful = [r for r in results if r.get("success", False)]
            failed = [r for r in results if not r.get("success", False)]
            avg_confidence = sum(r.get("confidence", 0) for r in successful) / len(successful) if successful else 0
            
            await ctx.info(f"Batch translation completed: {len(successful)}/{len(queries)} successful")
            
            return {
                "total_queries": len(queries),
                "successful": len(successful),
                "failed": len(failed),
                "average_confidence": avg_confidence,
                "results": results,
                "summary": {
                    "success_rate": len(successful) / len(queries) * 100,
                    "failed_queries": [r["natural_query"] for r in failed]
                }
            }
            
        except Exception as e:
            await ctx.error(f"Batch translation failed: {str(e)}")
            return {
                "error": str(e),
                "total_queries": len(queries),
                "processed": 0
            }

    @mcp.tool()
    async def translate_with_context(
        natural_query: str,
        schema_context: str,
        examples: Optional[List[str]] = None,
        domain_context: Optional[str] = None,
        model: Optional[str] = None,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Translate with rich context including schema, examples, and domain knowledge.
        
        This enhanced translation tool provides the best results by using
        comprehensive context to understand the query intent.
        """
        await ctx.info("Starting context-aware translation")
        
        try:
            # Build enhanced context
            enhanced_context = schema_context
            
            if examples:
                enhanced_context += "\n\n# Example Queries:\n"
                for i, example in enumerate(examples, 1):
                    enhanced_context += f"{i}. {example}\n"
            
            if domain_context:
                enhanced_context += f"\n\n# Domain Context:\n{domain_context}\n"
            
            await ctx.info("Enhanced context prepared, starting translation")
            
            result = await translation_service.translate_to_graphql(
                natural_query=natural_query,
                schema_context=enhanced_context,
                model=model
            )
            
            # Additional context-aware validation
            context_score = 0.0
            if schema_context and len(schema_context) > 100:
                context_score += 0.3
            if examples and len(examples) > 0:
                context_score += 0.2
            if domain_context:
                context_score += 0.1
            
            enhanced_confidence = min(1.0, result.confidence + context_score)
            
            await ctx.info(f"Context-aware translation completed with enhanced confidence: {enhanced_confidence:.2f}")
            
            return {
                "graphql_query": result.graphql_query,
                "confidence": enhanced_confidence,
                "base_confidence": result.confidence,
                "context_boost": context_score,
                "explanation": result.explanation,
                "model_used": result.model_used,
                "context_analysis": {
                    "schema_provided": bool(schema_context),
                    "examples_count": len(examples) if examples else 0,
                    "domain_context_provided": bool(domain_context),
                    "total_context_length": len(enhanced_context)
                }
            }
            
        except Exception as e:
            await ctx.error(f"Context-aware translation failed: {str(e)}")
            return {"error": str(e)}

    @mcp.tool()
    async def improve_translation(
        original_query: str,
        graphql_result: str,
        feedback: str,
        schema_context: Optional[str] = None,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Improve an existing GraphQL translation based on feedback.
        
        Takes an original translation and user feedback to produce
        an improved version with explanations of changes made.
        """
        await ctx.info("Starting translation improvement")
        
        try:
            # Create improvement prompt
            improvement_prompt = f"""
Original natural language query: "{original_query}"
Current GraphQL translation: {graphql_result}
User feedback: "{feedback}"

Please provide an improved GraphQL translation that addresses the feedback.
"""
            
            if schema_context:
                improvement_prompt += f"\nSchema context: {schema_context}"
            
            result = await translation_service.translate_to_graphql(
                natural_query=improvement_prompt,
                schema_context=schema_context or "",
            )
            
            await ctx.info("Translation improvement completed")
            
            return {
                "improved_query": result.graphql_query,
                "original_query": graphql_result,
                "confidence": result.confidence,
                "explanation": result.explanation,
                "changes_made": result.explanation,
                "feedback_addressed": feedback
            }
            
        except Exception as e:
            await ctx.error(f"Translation improvement failed: {str(e)}")
            return {"error": str(e)}

    @mcp.tool()
    async def select_icl_examples(
        natural_query: str,
        domain: Optional[str] = None,
        max_examples: int = 3,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Select the most relevant in-context learning (ICL) examples for a given natural language query.
        
        This tool analyzes the query and domain to pick the best examples from a database or predefined set
        to enhance the translation prompt.
        
        Args:
            natural_query: The natural language query to analyze for relevant examples
            domain: Optional domain context (e.g., 'e-commerce', 'social-media') to filter examples
            max_examples: Maximum number of examples to return (default: 3)
        
        Returns:
            Dictionary with selected ICL examples and metadata
        """
        await ctx.info(f"Selecting ICL examples for query: {natural_query[:50]}...")
        
        try:
            from config.icl_examples import INITIAL_ICL_EXAMPLES
            
            # Simple relevance scoring based on keyword overlap (to be enhanced with NLP later)
            selected_examples = []
            query_lower = natural_query.lower()
            domain_lower = domain.lower() if domain else ""
            
            # Score examples based on matching keywords
            scored_examples = []
            for example in INITIAL_ICL_EXAMPLES:
                natural_text = example['natural'].lower()
                score = 0
                if any(word in natural_text for word in query_lower.split() if len(word) > 3):
                    score += 1
                if domain_lower and domain_lower in natural_text:
                    score += 2
                scored_examples.append((example, score))
            
            # Sort by score and take the top max_examples
            scored_examples.sort(key=lambda x: x[1], reverse=True)
            selected_examples = [f"Natural: {ex['natural']}\nGraphQL: {ex['graphql']}" for ex, _ in scored_examples[:max_examples] if ex[1] > 0]
            
            # If no relevant examples found, use the top initial examples
            if not selected_examples:
                selected_examples = [f"Natural: {ex['natural']}\nGraphQL: {ex['graphql']}" for ex in INITIAL_ICL_EXAMPLES[:max_examples]]
            
            await ctx.info(f"Selected {len(selected_examples)} relevant ICL examples")
            
            return {
                "selected_examples": selected_examples,
                "total_available": len(INITIAL_ICL_EXAMPLES),
                "relevance_criteria": "keyword overlap" if selected_examples else "default selection",
                "domain_filter": domain if domain else "none"
            }
        except Exception as e:
            await ctx.error(f"Failed to select ICL examples: {str(e)}")
            return {"error": str(e)}

    @mcp.tool()
    async def translate_with_dynamic_icl(
        natural_query: str,
        schema_context: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = 0.3,
        domain: Optional[str] = None,
        max_icl_examples: int = 3,
        ctx: Context = None
    ) -> Dict[str, Any]:
        """
        Translate a natural language query to GraphQL using dynamically selected in-context learning (ICL) examples.
        
        This tool first selects the most relevant ICL examples based on the query and domain,
        then uses them to enhance the translation process.
        
        Args:
            natural_query: The natural language query to translate
            schema_context: Optional GraphQL schema context for better translation
            model: AI model to use for translation
            temperature: Temperature for AI model (0.0-1.0)
            domain: Optional domain context to filter relevant examples
            max_icl_examples: Maximum number of ICL examples to include in the prompt
        
        Returns:
            Translation result with GraphQL query and metadata about ICL usage
        """
        await ctx.info(f"Starting translation with dynamic ICL for query: {natural_query[:50]}...")
        
        try:
            start_time = time.time()
            
            # Step 1: Select relevant ICL examples
            icl_result = await select_icl_examples(natural_query, domain, max_icl_examples, ctx)
            selected_examples = icl_result.get("selected_examples", [])
            
            # Step 2: Perform translation with selected examples
            result = await translation_service.translate_to_graphql(
                natural_query=natural_query,
                schema_context=schema_context or "",
                model=model,
                icl_examples=selected_examples
            )
            
            processing_time = time.time() - start_time
            await ctx.info(f"Translation with dynamic ICL completed in {processing_time:.2f}s")
            
            return {
                "graphql_query": result.graphql_query,
                "confidence": result.confidence,
                "explanation": result.explanation,
                "model_used": result.model_used,
                "processing_time": processing_time,
                "icl_metadata": {
                    "examples_used": len(selected_examples),
                    "relevance_criteria": icl_result.get("relevance_criteria", "unknown"),
                    "domain_filter": domain if domain else "none"
                },
                "suggestions": result.suggestions if hasattr(result, 'suggestions') else [],
                "warnings": result.warnings if hasattr(result, 'warnings') else []
            }
        except Exception as e:
            await ctx.error(f"Translation with dynamic ICL failed: {str(e)}")
            return {"error": str(e)} 