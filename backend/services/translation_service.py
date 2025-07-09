"""
Translation Service - Compatibility layer for unified architecture.

This service provides backward compatibility with the existing API routes
while using the new unified architecture under the hood.
"""

import logging
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from .unified_providers import get_provider_service, GenerationRequest
from agents.unified_agents import AgentFactory, AgentContext, AgentType
from prompts.unified_prompts import get_prompt_manager
from config.unified_config import get_unified_config
from config.icl_examples import get_smart_examples, get_icl_examples

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Result from translation operation."""
    graphql_query: str
    confidence: float
    explanation: str
    model_used: str
    processing_time: float
    warnings: List[str] = field(default_factory=list)
    suggested_improvements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class TranslationService:
    """
    Translation service that wraps the unified architecture.
    
    Provides backward compatibility for existing API routes while
    using the new unified agents and providers system.
    """
    
    def __init__(self, provider_name: Optional[str] = None):
        self.provider_service = get_provider_service()
        self.prompt_manager = get_prompt_manager()
        self.config = get_unified_config()
        self.default_provider = provider_name
        
        logger.info(f"TranslationService initialized with provider: {provider_name or 'auto'}")
    
    async def translate_to_graphql(
        self,
        natural_query: str,
        schema_context: Optional[str] = None,
        domain: Optional[str] = None,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        examples: Optional[List[Any]] = None
    ) -> TranslationResult:
        """
        Translate natural language to GraphQL (alias for translate_query).
        """
        return await self.translate_query(
            natural_query=natural_query,
            schema_context=schema_context,
            domain=domain,
            model=model,
            session_id=session_id,
            user_id=user_id,
            examples=examples
        )
    
    async def translate_natural_to_graphql(
        self,
        natural_query: str,
        schema_context: Optional[str] = None,
        domain: Optional[str] = None,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        examples: Optional[List[Any]] = None
    ) -> TranslationResult:
        """
        Translate natural language to GraphQL (alias for translate_query).
        """
        return await self.translate_query(
            natural_query=natural_query,
            schema_context=schema_context,
            domain=domain,
            model=model,
            session_id=session_id,
            user_id=user_id,
            examples=examples
        )
    
    async def batch_translate(
        self,
        queries: List[str],
        schema_context: Optional[str] = None,
        domain: Optional[str] = None,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[TranslationResult]:
        """
        Translate multiple natural language queries to GraphQL.
        """
        results = []
        for query in queries:
            try:
                result = await self.translate_query(
                    natural_query=query,
                    schema_context=schema_context,
                    domain=domain,
                    model=model,
                    session_id=session_id,
                    user_id=user_id
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Batch translation failed for query '{query}': {e}")
                results.append(TranslationResult(
                    graphql_query="",
                    confidence=0.0,
                    explanation=f"Translation failed: {str(e)}",
                    model_used=model or "unknown",
                    processing_time=0.0,
                    warnings=[f"Error: {str(e)}"]
                ))
        return results
    
    def get_translation_examples(self, domain: str = "manufacturing", limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get translation examples for the UI.
        """
        from config.icl_examples import get_icl_manager
        
        manager = get_icl_manager()
        examples = manager.get_random_examples(limit)
        
        return [
            {
                "natural": ex.natural,
                "graphql": ex.graphql,
                "tags": ex.tags,
                "complexity": ex.complexity_score
            }
            for ex in examples
        ]
    
    def _build_system_prompt(self, schema_context: str = "", domain: str = "manufacturing") -> str:
        """
        Build system prompt for translation.
        """
        from config.icl_examples import get_icl_manager
        
        manager = get_icl_manager()
        examples = manager.get_random_examples(3)
        
        examples_text = "\n\n".join([
            f"Natural: {ex.natural}\nGraphQL: {ex.graphql}"
            for ex in examples
        ])
        
        return f"""You are an expert GraphQL query translator for manufacturing and 3D printing systems.

Convert natural language queries to GraphQL queries for manufacturing data.

Schema Context:
{schema_context}

Examples:
{examples_text}

Guidelines:
- Use appropriate GraphQL syntax
- Handle filters, aggregations, and nested queries
- Focus on manufacturing, thermal, quality, and process data
- Return only the GraphQL query without explanations"""
    
    async def switch_provider(self, provider_name: str, api_key: Optional[str] = None):
        """Switch to a different provider."""
        self.default_provider = provider_name
        logger.info(f"Switched to provider: {provider_name}")
    
    async def switch_model(self, model: str):
        """Switch to a different model."""
        logger.info(f"Model preference set to: {model}")

    async def translate_query(
        self,
        natural_query: str,
        schema_context: Optional[str] = None,
        domain: Optional[str] = None,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        examples: Optional[List[Any]] = None
    ) -> TranslationResult:
        """
        Main translation method using unified architecture.
        
        Args:
            natural_query: The natural language query to translate
            schema_context: GraphQL schema context
            domain: Domain context for better translation
            model: Model to use (optional, will use UI-selected model)
            session_id: Session ID for tracking
            user_id: User ID for tracking
            examples: ICL examples for few-shot learning
            
        Returns:
            TranslationResult with GraphQL query and metadata
        """
        start_time = time.time()
 
        try:
            # Ensure providers are registered
            provider_service = get_provider_service()
            if not provider_service.registry.list_providers():
                raise RuntimeError("No LLM providers are configured. Please configure Ollama or another provider before using the translation service.")

            # Create translator agent
            agent = AgentFactory.create_agent("translator")
            
            if not agent:
                raise Exception("Translator agent not available")
            
            # Get ICL examples if not provided
            if not examples and domain:
                examples = get_smart_examples(natural_query, domain, limit=3)
            elif not examples:
                examples = get_icl_examples("manufacturing", limit=3)
            
            # Ensure schema context is provided (fallback placeholder if missing)
            if not schema_context:
                schema_context = "type Query { thermalScans: [ThermalScan] } type ThermalScan { scanID: ID jobID: String maxTemperature: Float hotspotCoordinates: [Int] thermalImage: String }"

            # Prepare context with ICL examples
            context = AgentContext(
                original_query=natural_query,
                session_id=session_id or f"translation-{int(time.time())}",
                user_id=user_id,
                domain_context=domain or "manufacturing",
                schema_context=schema_context or "",
                examples=examples or [],
                metadata={
                    "model_override": model,
                    "provider_override": self.default_provider
                }
            )
            
            # Execute translation
            result = await agent.execute(context)
            
            if result.success:
                processing_time = time.time() - start_time
                
                # Support both string and dict outputs from translator agent
                graphql_query_val: str
                if isinstance(result.output, str):
                    graphql_query_val = result.output
                elif isinstance(result.output, dict) and "graphql" in result.output:
                    graphql_query_val = result.output["graphql"]
                else:
                    graphql_query_val = ""

                return TranslationResult(
                    graphql_query=graphql_query_val,
                    confidence=result.confidence or 0.0,
                    explanation=result.metadata.get("explanation", "Translation completed"),
                    model_used=result.metadata.get("model_used", model or "unknown"),
                    processing_time=processing_time,
                    warnings=result.metadata.get("warnings", []),
                    suggested_improvements=result.metadata.get("suggestions", []),
                    metadata=result.metadata
                )
            else:
                raise Exception(f"Translation failed: {result.error}")
                
        except Exception as e:
            logger.error(f"Translation error: {e}")
            processing_time = time.time() - start_time
            
            # Return error result
            return TranslationResult(
                graphql_query="",
                confidence=0.0,
                explanation=f"Translation failed: {str(e)}",
                model_used=model or "unknown",
                processing_time=processing_time,
                warnings=[str(e)],
                metadata={"error": str(e)}
            )


# Compatibility function for existing imports
async def chat_with_model(
    message: str,
    model: Optional[str] = None,
    history: Optional[List[Dict[str, Any]]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Chat with model function for compatibility.
    
    Uses the unified provider system for chat completions.
    """
    try:
        provider_service = get_provider_service()
        
        # Prepare messages
        messages = []
        if history:
            for msg in history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        messages.append({"role": "user", "content": message})
        
        # Generate response
        response = await provider_service.chat(
            messages=messages,
            model=model or "phi3:mini",
            **kwargs
        )
        
        return {
            "response": response.text,
            "model": response.model,
            "processing_time": response.processing_time,
            "provider": response.provider
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {
            "response": f"Error: {str(e)}",
            "model": model or "unknown",
            "processing_time": 0.0,
            "error": str(e)
        } 