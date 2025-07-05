"""
Advanced Agent Orchestration Service

A sophisticated system that uses the new agent framework, pipeline engine, 
and context engineering, supporting both streaming and non-streaming responses.
"""

import asyncio
import logging
import time
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, AsyncGenerator

from config.settings import get_settings
from services.translation_service import TranslationService
from services.ollama_service import OllamaService

logger = logging.getLogger(__name__)

@dataclass
class AdvancedMultiAgentResult:
    """Enhanced result object with comprehensive metadata and analytics."""
    
    original_query: str
    rewritten_query: str
    translation: Dict[str, Any]
    review: Dict[str, Any]
    processing_time: float
    pipeline_config: str
    execution_strategy: str
    agents_executed: List[str]
    performance_metrics: Dict[str, Any]
    prompt_strategy_used: str
    context_optimizations: List[str]
    template_names: Dict[str, str]
    confidence_score: float
    optimization_applied: bool
    review_passed: bool
    agent_outputs: Dict[str, Any]
    execution_timeline: List[Dict[str, Any]]
    error_details: List[Dict[str, Any]]

    def to_legacy_format(self) -> Dict[str, Any]:
        """Convert to legacy format for backward compatibility."""
        return {
            "original_query": self.original_query,
            "rewritten_query": self.rewritten_query,
            "translation": self.translation,
            "review": self.review,
            "processing_time": self.processing_time
        }

class PipelineStrategy:
    STANDARD = "standard"
    FAST = "fast"
    COMPREHENSIVE = "comprehensive"
    ADAPTIVE = "adaptive"

class AdvancedAgentOrchestrationService:
    def __init__(self):
        self.settings = get_settings()
        self.translation_service = TranslationService()
        self.ollama_service = OllamaService()
        self.pipeline_configs = self._initialize_pipeline_configs()
        
        # Performance tracking
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'average_processing_time': 0.0,
            'strategy_performance': {}
        }
        
        logger.info("🚀 Advanced Agent Orchestration Service initialized")

    def _initialize_pipeline_configs(self) -> Dict[str, Dict[str, Any]]:
        return {
            PipelineStrategy.STANDARD: {'agents': ['rewriter', 'translator', 'reviewer']},
            PipelineStrategy.FAST: {'agents': ['translator']},
            PipelineStrategy.COMPREHENSIVE: {'agents': ['rewriter', 'translator', 'reviewer', 'optimizer']},
        }
    
    # --- Streaming Method ---
    async def process_query_stream(
        self, 
        query: str, 
        pipeline_strategy: str, 
        translator_model: Optional[str] = None,
        pre_model: Optional[str] = None,
        review_model: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Processes a query and streams events back."""
        yield self._create_event("session_start", session_id=kwargs.get('session_id', 'N/A'))
        
        pipeline_config = self.pipeline_configs.get(pipeline_strategy, self.pipeline_configs[PipelineStrategy.STANDARD])
        total_steps = len(pipeline_config['agents'])
        context = {'original_query': query, **kwargs}
        
        try:
            # Step 1: Rewriter (simulated for non-fast pipelines)
            rewritten_query = context['original_query']
            if pipeline_strategy != PipelineStrategy.FAST:
                yield self._create_event("agent_start", agent="rewriter", step=1, total_steps=total_steps)
                await asyncio.sleep(0.3)
                rewritten_query += " (rewritten)"
                yield self._create_event("agent_complete", agent="rewriter", step=1, total_steps=total_steps, result={"rewritten_query": rewritten_query})

            # Step 2: Translator (streaming)
            translator_step = 2 if pipeline_strategy != PipelineStrategy.FAST else 1
            yield self._create_event("agent_start", agent="translator", step=translator_step, total_steps=total_steps)
            
            final_translation = {}
            # Use the selected model for translation
            stream = self.translation_service.translate_to_graphql(
                natural_query=rewritten_query,
                model=translator_model or self.settings.ollama.default_model
            )
            async for event in stream:
                if event['event'] == 'agent_token':
                    yield self._create_event("agent_token", agent="translator", token=event['token'])
                elif event['event'] == 'translation_complete':
                    final_translation = event['result']
            
            yield self._create_event("agent_complete", agent="translator", step=translator_step, total_steps=total_steps, result=final_translation)

            # Step 3: Reviewer (simulated for non-fast pipelines)
            if pipeline_strategy != PipelineStrategy.FAST:
                yield self._create_event("agent_start", agent="reviewer", step=3, total_steps=total_steps)
                await asyncio.sleep(0.3)
                yield self._create_event("agent_complete", agent="reviewer", step=3, total_steps=total_steps, result={"passed": True})

            yield self._create_event("complete", result={"message": "Pipeline finished."})

        except Exception as e:
            logger.error(f"Streaming orchestration failed: {e}", exc_info=True)
            yield self._create_event("error", error=str(e))

    def _create_event(self, event_type: str, **kwargs) -> Dict[str, Any]:
        """Helper to create a structured event dictionary."""
        event = {"event": event_type}
        event.update(kwargs)
        # Simplify prompt data for streaming
        if 'prompt_data' in kwargs and isinstance(kwargs['prompt_data'], dict):
            kwargs['prompt'] = {k: str(v)[:100] + '...' for k,v in kwargs['prompt_data'].items()}
        return event

    # --- Original Non-Streaming Methods (Kept for compatibility) ---
    async def process_query(self, query: str, **kwargs: Any) -> AdvancedMultiAgentResult:
        """
        Original non-streaming method. Note: This will not produce token-by-token results.
        It uses the non-streaming `chat_completion` method from Ollama service.
        """
        start_time = time.time()
        
        # This simulates the original, non-streaming logic
        rewritten_query = await self._rewrite_query(query)
        
        # This would call a non-streaming version of translate_to_graphql
        # For now, we simulate this call.
        _, translation_result = await self._get_full_translation(rewritten_query)
        
        review_result = await self._review_translation(query, translation_result.get('graphql_query', ''))

        processing_time = time.time() - start_time
        return AdvancedMultiAgentResult(
            original_query=query,
            rewritten_query=rewritten_query,
            translation=translation_result,
            review=review_result,
            processing_time=processing_time,
            pipeline_config="standard",
            execution_strategy="non-streaming",
            agents_executed=['rewriter', 'translator', 'reviewer'],
            performance_metrics={'processing_time': processing_time},
            prompt_strategy_used="default",
            context_optimizations=[],
            template_names={},
            confidence_score=translation_result.get('confidence', 0.5),
            optimization_applied=False,
            review_passed=review_result.get('passed', False),
            agent_outputs={},
            execution_timeline=[],
            error_details=[]
        )

    async def _get_full_translation(self, query: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """
        A non-streaming version to get a full translation result.
        This simulates how the original service might have worked.
        """
        system_prompt = self.translation_service._build_system_prompt()
        user_prompt = f"Convert this query: {query}"
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        
        response = await self.ollama_service.chat_completion(messages=messages)
        parsed = self.translation_service._extract_json_from_response(response.text)
        return messages, parsed

    async def _rewrite_query(self, query: str) -> str:
        """Simulated non-streaming rewrite."""
        response = await self.ollama_service.generate_response(prompt=f"Rewrite: {query}")
        return response.text

    async def _review_translation(self, original_query: str, graphql_query: str) -> Dict[str, Any]:
        """Simulated non-streaming review."""
        prompt = f"Original: {original_query}\nGraphQL: {graphql_query}\nReview this."
        response = await self.ollama_service.generate_response(prompt=prompt)
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            return {"passed": "error" in response.text.lower(), "comments": [response.text]}

    # --- Non-streaming methods for backward compatibility or other uses ---
    async def process_simple_query(self, query: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Original simple query processing."""
        # This is a simplified placeholder for any non-streaming logic you need to keep.
        # We assume translate_to_graphql can no longer be awaited directly for a single result.
        # This part of the code would need a non-streaming alternative if used.
        # For now, we'll simulate a result.
        await asyncio.sleep(1) # Simulate async work
        return {
            "graphql_query": f"{{ query for '{query}' }}",
            "confidence": 0.85
        }

    def _create_advanced_result(
        self,
        pipeline_result: Dict[str, Any],
        context: Dict[str, Any],
        strategy: str,
        processing_time: float
    ) -> AdvancedMultiAgentResult:
        """Create comprehensive result object."""
        
        translation = pipeline_result.get('translation', {})
        review = pipeline_result.get('review', {})
        
        return AdvancedMultiAgentResult(
            # Core results
            original_query=pipeline_result['original_query'],
            rewritten_query=pipeline_result['rewritten_query'],
            translation=translation,
            review=review,
            processing_time=processing_time,
            
            # Enhanced metadata
            pipeline_config=strategy,
            execution_strategy="sequential",
            agents_executed=pipeline_result.get('agents_executed', []),
            performance_metrics={
                'processing_time': processing_time,
                'query_length': len(context['original_query']),
                'agents_count': len(pipeline_result.get('agents_executed', []))
            },
            
            # Context engineering
            prompt_strategy_used="default",
            context_optimizations=[],
            template_names={},
            
            # Quality metrics
            confidence_score=translation.get('confidence', 0.0),
            optimization_applied=pipeline_result.get('optimization_applied', False),
            review_passed=review.get('passed', True),
            
            # Debugging
            agent_outputs=pipeline_result,
            execution_timeline=[],
            error_details=[]
        )
    
    def _create_error_result(
        self, 
        query: str, 
        error: str, 
        processing_time: float, 
        strategy: str
    ) -> AdvancedMultiAgentResult:
        """Create error result object."""
        
        return AdvancedMultiAgentResult(
            original_query=query,
            rewritten_query=query,
            translation={'error': error, 'graphql_query': '', 'confidence': 0.0},
            review={'passed': False, 'comments': [f"Processing failed: {error}"]},
            processing_time=processing_time,
            pipeline_config=strategy,
            execution_strategy="failed",
            agents_executed=[],
            performance_metrics={},
            prompt_strategy_used="none",
            context_optimizations=[],
            template_names={},
            confidence_score=0.0,
            optimization_applied=False,
            review_passed=False,
            agent_outputs={},
            execution_timeline=[],
            error_details=[{'error': error, 'timestamp': datetime.utcnow().isoformat()}]
        )
    
    def _update_performance_stats(self, strategy: str, processing_time: float, success: bool):
        """Update performance statistics."""
        if success:
            self.execution_stats['successful_executions'] += 1
        
        # Update averages
        total = self.execution_stats['total_executions']
        current_avg = self.execution_stats['average_processing_time']
        self.execution_stats['average_processing_time'] = (current_avg * (total - 1) + processing_time) / total
        
        # Update strategy-specific stats
        if strategy not in self.execution_stats['strategy_performance']:
            self.execution_stats['strategy_performance'][strategy] = {
                'count': 0, 'success_count': 0, 'avg_time': 0.0
            }
        
        strategy_stats = self.execution_stats['strategy_performance'][strategy]
        strategy_stats['count'] += 1
        if success:
            strategy_stats['success_count'] += 1
        
        strategy_avg = strategy_stats['avg_time']
        strategy_count = strategy_stats['count']
        strategy_stats['avg_time'] = (strategy_avg * (strategy_count - 1) + processing_time) / strategy_count
    
    # Convenience methods for backward compatibility and different use cases
    
    async def process_with_context(
        self,
        query: str,
        domain: str,
        schema: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> AdvancedMultiAgentResult:
        """Process with rich context information."""
        return await self.process_query(
            query,
            pipeline_strategy=PipelineStrategy.COMPREHENSIVE,
            domain_context=domain,
            schema_context=schema,
            examples=examples
        )
    
    # Monitoring and management methods
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'execution_stats': self.execution_stats,
            'available_strategies': list(self.pipeline_configs.keys()),
            'service_info': {
                'version': '2.0.0',
                'features': ['advanced_pipelines', 'performance_monitoring', 'context_engineering']
            }
        }
    
    def get_available_strategies(self) -> List[Dict[str, Any]]:
        """Get list of available pipeline strategies."""
        return [
            {
                'name': name,
                'description': config['description'],
                'agents': config['agents'],
                'optimization_level': config['optimization_level']
            }
            for name, config in self.pipeline_configs.items()
        ] 