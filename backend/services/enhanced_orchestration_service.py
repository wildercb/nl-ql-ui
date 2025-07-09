"""
Enhanced Orchestration Service - Compatibility layer for unified architecture.
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PipelineStrategy(Enum):
    """Pipeline execution strategies."""
    FAST = "fast"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    PARALLEL = "parallel"


@dataclass
class OrchestrationResult:
    """Result from orchestration execution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    pipeline_used: str = "default"
    agents_executed: List[str] = None
    
    def __post_init__(self):
        if self.agents_executed is None:
            self.agents_executed = []


class EnhancedOrchestrationService:
    """
    Enhanced orchestration service that wraps the unified architecture.
    
    Provides backward compatibility for existing orchestration calls
    while using the new unified agents and pipeline system.
    """
    
    def __init__(self):
        logger.info("EnhancedOrchestrationService initialized")
    
    async def execute_pipeline(
        self,
        query: str,
        strategy: PipelineStrategy = PipelineStrategy.STANDARD,
        model: Optional[str] = None,
        schema_context: Optional[str] = None,
        domain: Optional[str] = None,
        **kwargs
    ) -> OrchestrationResult:
        """
        Execute a multi-agent pipeline.
        
        Args:
            query: Natural language query
            strategy: Pipeline strategy to use
            model: Model to use for agents
            schema_context: GraphQL schema context
            domain: Domain context
            
        Returns:
            OrchestrationResult with pipeline execution results
        """
        try:
            # Import unified agents
            from agents.unified_agents import PipelineExecutor, AgentContext
            
            executor = PipelineExecutor()
            
            # Create context
            context = AgentContext(
                original_query=query,
                schema_context=schema_context or "",
                domain_context=domain,
                metadata={
                    "model_override": model,
                    "strategy": strategy.value
                }
            )
            
            # Execute pipeline based on strategy
            pipeline_name = self._get_pipeline_name(strategy)
            results = await executor.execute_pipeline(pipeline_name, context)
            
            # Extract results
            success = any(result.success for result in results.values())
            output = self._extract_output(results)
            agents_executed = list(results.keys())
            
            return OrchestrationResult(
                success=success,
                output=output,
                processing_time=context.total_processing_time,
                pipeline_used=pipeline_name,
                agents_executed=agents_executed
            )
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return OrchestrationResult(
                success=False,
                error=str(e),
                pipeline_used=strategy.value
            )
    
    def _get_pipeline_name(self, strategy: PipelineStrategy) -> str:
        """Map strategy to pipeline name."""
        mapping = {
            PipelineStrategy.FAST: "fast",
            PipelineStrategy.STANDARD: "standard", 
            PipelineStrategy.COMPREHENSIVE: "comprehensive",
            PipelineStrategy.PARALLEL: "parallel"
        }
        return mapping.get(strategy, "standard")
    
    def _extract_output(self, results: Dict[str, Any]) -> Any:
        """Extract the main output from pipeline results."""
        # Look for translator output first
        for agent_name, result in results.items():
            if "translator" in agent_name.lower() and result.success:
                return result.output
        
        # Fallback to any successful result
        for result in results.values():
            if result.success and result.output:
                return result.output
        
        return None
    
    async def translate_query(
        self,
        query: str,
        model: Optional[str] = None,
        schema_context: Optional[str] = None,
        domain: Optional[str] = None,
        **kwargs
    ) -> OrchestrationResult:
        """Simple translation using just the translator agent."""
        return await self.execute_pipeline(
            query=query,
            strategy=PipelineStrategy.FAST,
            model=model,
            schema_context=schema_context,
            domain=domain,
            **kwargs
        )
    
    async def review_query(
        self,
        query: str,
        graphql_query: str,
        model: Optional[str] = None,
        **kwargs
    ) -> OrchestrationResult:
        """Review a GraphQL query using the LLM."""
        try:
            from agents.unified_agents import AgentFactory, AgentContext
            
            # Create reviewer agent
            agent = AgentFactory.create_agent("reviewer")
            if not agent:
                raise Exception("Reviewer agent not available")
            
            # Create context
            context = AgentContext(
                original_query=query,
                graphql_query=graphql_query,
                metadata={"model_override": model}
            )
            
            # Execute review
            result = await agent.execute(context)
            
            return OrchestrationResult(
                success=result.success,
                output=result.output,
                error=result.error,
                processing_time=result.processing_time,
                pipeline_used="review",
                agents_executed=["reviewer"]
            )
            
        except Exception as e:
            logger.error(f"Query review failed: {e}")
            return OrchestrationResult(
                success=False,
                error=str(e),
                pipeline_used="review"
            ) 

    # ------------------------------------------------------------------
    # Streaming Support (used by /api/mcp and /api/multiagent routes)
    # ------------------------------------------------------------------

    async def process_query_stream(
        self,
        query: str,
        pipeline_strategy: PipelineStrategy = PipelineStrategy.STANDARD,
        translator_model: Optional[str] = None,
        pre_model: Optional[str] = None,
        review_model: Optional[str] = None,
        schema_context: Optional[str] = None,
        domain_context: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Yield Server-Sent Events while a pipeline is executing.

        The current implementation is pragmatic – it executes the whole
        pipeline with `execute_pipeline` and then chunks the final result so
        that the UI gets progressive updates.  When we integrate true streaming
        providers we can push incremental LLM tokens here without changing the
        API contract.
        """

        import json, asyncio, time

        start_ts = time.time()

        # Emit pipeline start event (legacy label 'start' expected by UI)
        yield {
            "event": "start",
            "data": {
                "query": query,
                "strategy": pipeline_strategy.value,
                "session_id": session_id,
            },
        }

        # Emit translator agent start (for UI progress bars)
        yield {
            "event": "agent_start",
            "data": {"agent": "translator"},
        }

        # Ensure schema context provided to satisfy translator validation
        if not schema_context:
            schema_context = (
                "type Query { thermalScans: [ThermalScan] }\n"
                "type ThermalScan { scanID: ID maxTemperature: Float hotspotCoordinates: [Int] }"
            )

        # Run pipeline
        result = await self.execute_pipeline(
            query=query,
            strategy=pipeline_strategy,
            model=translator_model,  # translator_model overrides agent models
            schema_context=schema_context,
            domain=domain_context,
        )

        # Stream translator output token-by-token if available
        graphql = ""
        if result.output and isinstance(result.output, dict):
            graphql = result.output.get("graphql") or ""
        elif isinstance(result.output, str):
            graphql = result.output

        if graphql:
            tokens = graphql.split()
            current = ""
            for tok in tokens:
                current += tok + " "
                yield {
                    "event": "agent_token",
                    "data": {
                        "agent": "translator",
                        "token": tok,
                    },
                }
                # small pause so frontend can render progressively
                await asyncio.sleep(0.01)

            # Final translator result
            yield {
                "event": "agent_complete",
                "data": {
                    "agent": "translator",
                    "result": {
                        "graphql_query": graphql.strip(),
                        "confidence": (result.output.get("confidence") if isinstance(result.output, dict) else 1.0),
                        "explanation": (result.output.get("explanation") if isinstance(result.output, dict) else ""),
                    },
                    "prompt": None,
                },
            }

        # Emit pipeline_complete
        summary_payload = {
            "success": result.success,
            "translation": {
                "graphql_query": graphql.strip() if graphql else "",
                "confidence": (result.output.get("confidence") if isinstance(result.output, dict) else 1.0) if result.success else 0.0,
                "explanation": (result.output.get("explanation") if isinstance(result.output, dict) else "") if isinstance(result.output, dict) else "",
                "model_used": translator_model or "auto",
                "warnings": [],
            },
            "review": {},
            "error": result.error,
            "processing_time": time.time() - start_ts,
            "pipeline_strategy": pipeline_strategy.value,
            "pipeline_used": result.pipeline_used,
            "agents_executed": result.agents_executed,
        }

        yield {"event": "pipeline_complete", "data": summary_payload}

        # Legacy compatibility – final "complete" event expected by MCP server
        yield {"event": "complete", "data": {"result": summary_payload}}

        # Done
        return

# Global service instance
_orchestration_service: Optional[EnhancedOrchestrationService] = None


def get_orchestration_service() -> EnhancedOrchestrationService:
    """Get the global orchestration service instance."""
    global _orchestration_service
    if _orchestration_service is None:
        _orchestration_service = EnhancedOrchestrationService()
    return _orchestration_service 