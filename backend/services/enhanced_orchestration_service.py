"""
Enhanced Orchestration Service for MPPW-MCP

This service provides advanced pipeline orchestration with real-time streaming
capabilities for multi-agent processing.
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
import time
import uuid
import asyncio

from config.unified_config import get_unified_config

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


@dataclass
class AgentResult:
    """Result from an agent execution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    model_used: Optional[str] = None
    prompt_used: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EnhancedOrchestrationService:
    """
    Enhanced orchestration service that provides real-time streaming
    of multi-agent pipeline execution.
    """
    
    def __init__(self):
        self.config_manager = get_unified_config()
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

        This method executes the pipeline and emits events for each agent
        as they start, process, and complete in real-time.
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

        # Get Neo4j schema context for translator and reviewer agents
        if not schema_context:
            try:
                from services.neo4j_schema_service import get_neo4j_graphql_schema
                neo4j_schema = await get_neo4j_graphql_schema()
                if neo4j_schema:
                    schema_context = neo4j_schema
                    logger.info("✅ Using Neo4j-generated GraphQL schema")
                else:
                    # Fallback to default schema
                    schema_context = (
                        "type Query {\n"
                        "  thermalScans: [ThermalScan]\n"
                        "  findAllThermalScans: [ThermalScan]\n"
                        "}\n"
                        "type ThermalScan {\n"
                        "  scanID: ID\n"
                        "  scanId: String\n"
                        "  maxTemperature: Float\n"
                        "  temperatureReadings: [Float]\n"
                        "  hotspotCoordinates: [Int]\n"
                        "  dateTaken: String\n"
                        "  locationDetails: String\n"
                        "}"
                    )
                    logger.info("⚠️ Using fallback GraphQL schema")
            except Exception as e:
                logger.error(f"❌ Error fetching Neo4j schema: {e}")
                # Fallback to default schema
                schema_context = (
                    "type Query {\n"
                    "  thermalScans: [ThermalScan]\n"
                    "  findAllThermalScans: [ThermalScan]\n"
                    "}\n"
                    "type ThermalScan {\n"
                    "  scanID: ID\n"
                    "  scanId: String\n"
                    "  maxTemperature: Float\n"
                    "  temperatureReadings: [Float]\n"
                    "  hotspotCoordinates: [Int]\n"
                    "  dateTaken: String\n"
                    "  locationDetails: String\n"
                    "}"
                )

        try:
            # Import unified agents and prompt manager
            from agents.unified_agents import PipelineExecutor, AgentContext, AgentFactory
            from prompts.unified_prompts import get_prompt_manager
            from config.icl_examples import get_smart_examples, get_icl_examples
            
            # Get ICL examples for the translator agent
            examples = []
            if domain_context:
                examples = get_smart_examples(query, domain_context, limit=3)
            else:
                examples = get_icl_examples("manufacturing", limit=3)
            
            # Convert ICL examples to the format expected by the prompt templates
            formatted_examples = []
            for example in examples:
                formatted_examples.append({
                    "natural": example.natural,
                    "graphql": example.graphql
                })
            
            # Create context with ICL examples
            context = AgentContext(
                original_query=query,
                schema_context=schema_context,
                domain_context=domain_context,
                examples=formatted_examples,
                user_id=user_id,
                session_id=session_id or str(uuid.uuid4()),
                metadata={
                    "model_override": translator_model,
                    "strategy": pipeline_strategy.value
                }
            )
            
            # Initialize prompt manager
            prompt_manager = get_prompt_manager()

            # Get pipeline configuration
            config_manager = get_unified_config()
            pipeline_name = self._get_pipeline_name(pipeline_strategy)
            pipeline_config = config_manager.get_pipeline(pipeline_name)
            
            if not pipeline_config:
                raise ValueError(f"Pipeline not found: {pipeline_name}")

            # Execute agents one by one with real-time streaming
            agent_factory = AgentFactory()
            results = {}
            
            # Execute agents based on strategy
            for agent_name in pipeline_config.agents:
                agent = agent_factory.create_agent(agent_name)
                if not agent:
                    logger.error(f"Agent not found: {agent_name}")
                    continue

                # Skip analyzer if no data is available
                if agent_name == "analyzer" and not (context.agent_outputs.get('query_data') or context.agent_outputs.get('graphql_executed')):
                    logger.info(f"Skipping analyzer agent - no query data available for analysis")
                    continue
                
                # Get prompt for this agent
                # Convert AgentContext to dictionary with appropriate variables for each agent
                if agent_name == "analyzer":
                    # For analyzer, we need the query data and other context
                    prompt_context = {
                        "query_data": context.agent_outputs.get('query_data', {}),
                        "graphql_executed": context.agent_outputs.get('graphql_executed', ''),
                        "original_query": context.original_query,
                        "analysis_type": "comprehensive"
                    }
                else:
                    # For other agents, use the standard context conversion
                    prompt_context = context.get_prompt_context()
                
                prompt = prompt_manager.get_prompt_for_agent(agent_name, prompt_context)
                if not prompt:
                    logger.error(f"Failed to generate prompt for agent {agent_name}")
                    continue
                
                # Emit agent start event
                yield {
                    "event": "agent_start",
                    "data": {
                        "agent": agent_name,
                        "agent_type": "processing",
                        "timestamp": time.time()
                    },
                }

                # Emit agent prompt event
                yield {
                    "event": "agent_prompt",
                    "data": {
                        "agent": agent_name,
                        "prompt": prompt,
                        "timestamp": time.time()
                    },
                }

                # Execute agent with streaming
                agent_start_time = time.time()
                
                # Execute the agent properly
                logger.info(f"Starting execution of agent {agent_name}")
                try:
                    agent_result = await agent.execute(context)
                    agent_end_time = time.time()
                    logger.info(f"Agent {agent_name} execution completed successfully: success={agent_result.success}, output={agent_result.output}")
                except Exception as e:
                    agent_end_time = time.time()
                    logger.error(f"Agent {agent_name} execution failed: {e}")
                    agent_result = AgentResult(
                        success=False,
                        error=str(e),
                        processing_time=agent_end_time - agent_start_time
                    )
                
                # Debug logging
                logger.info(f"Agent {agent_name} execution result: success={agent_result.success}, output={agent_result.output}")
                
                # Store result
                results[agent_name] = agent_result
                
                # Update context with agent output
                context.add_agent_output(agent_name, agent_result.output)
                
                # If this is the translator agent and it was successful, execute the GraphQL query
                if agent_name == "translator" and agent_result.success and agent_result.output:
                    # Extract GraphQL query from translator output
                    if isinstance(agent_result.output, dict):
                        graphql_query = agent_result.output.get("graphql_query") or agent_result.output.get("graphql", "")
                    else:
                        graphql_query = str(agent_result.output)
                    
                    if graphql_query:
                        logger.info(f"Executing GraphQL query: {graphql_query}")
                        
                        # Execute the GraphQL query to get data for analysis
                        try:
                            from services.data_query_service import get_data_query_service
                            data_service = await get_data_query_service()
                            
                            # Execute the query
                            query_result = await data_service.execute_query(graphql_query)
                            
                            # Store the data in context for the analyzer
                            context.agent_outputs['query_data'] = query_result
                            context.agent_outputs['graphql_executed'] = graphql_query
                            
                            logger.info(f"GraphQL query executed successfully, data available for analysis")
                            
                        except Exception as e:
                            logger.error(f"Failed to execute GraphQL query: {e}")
                            context.agent_outputs['query_error'] = str(e)
                
                # Format the result for frontend
                formatted_result = self._format_agent_result(agent_name, agent_result, context)
                logger.info(f"Formatted result for {agent_name}: {formatted_result}")
                
                # Emit agent completion event with full data
                yield {
                    "event": "agent_complete",
                    "data": {
                        "agent": agent_name,
                        "result": formatted_result,
                        "prompt": prompt,
                        "processing_time": agent_end_time - agent_start_time,
                        "timestamp": time.time(),
                        "raw_output": agent_result.output,  # Include raw output for debugging
                        "success": agent_result.success,
                        "error": agent_result.error
                    },
                }

                # Stream tokens for the agent response
                if agent_result.success and agent_result.output:
                    if agent_name == "translator":
                        # Stream GraphQL query tokens
                        if isinstance(agent_result.output, dict):
                            graphql = agent_result.output.get("graphql", "") or agent_result.output.get("graphql_query", "")
                        else:
                            graphql = str(agent_result.output)
                        
                        if graphql:
                            tokens = graphql.split()
                            for tok in tokens:
                                yield {
                                    "event": "agent_token",
                                    "data": {
                                        "agent": agent_name,
                                        "token": tok + " ",
                                        "timestamp": time.time()
                                    },
                                }
                                await asyncio.sleep(0.01)
                    else:
                        # For other agents, simulate processing
                        processing_messages = [
                            "Analyzing query...",
                            "Processing data...",
                            "Generating results...",
                            "Finalizing output..."
                        ]
                        for msg in processing_messages:
                            yield {
                                "event": "agent_token",
                                "data": {
                                    "agent": agent_name,
                                    "token": msg + " ",
                                    "timestamp": time.time()
                                },
                            }
                            await asyncio.sleep(0.1)

            # Build final summary
            success = any(result.success for result in results.values())
            output = self._extract_output(results)
            agents_executed = list(results.keys())

            # Emit pipeline_complete
            graphql = ""
            if output and isinstance(output, dict):
                graphql = output.get("graphql") or output.get("graphql_query") or ""
            elif isinstance(output, str):
                graphql = output

            summary_payload = {
                "success": success,
                "translation": {
                    "graphql_query": graphql.strip() if graphql else "",
                    "confidence": (output.get("confidence") if isinstance(output, dict) else 1.0) if success else 0.0,
                    "explanation": (output.get("explanation") if isinstance(output, dict) else "") if success else "",
                    "model_used": translator_model or "auto",
                    "warnings": [],
                },
                "review": {},
                "error": None,
                "processing_time": time.time() - start_ts,
                "pipeline_strategy": pipeline_strategy.value,
                "pipeline_used": pipeline_name,
                "agents_executed": agents_executed,
            }

            yield {"event": "pipeline_complete", "data": summary_payload}

            # Legacy compatibility – final "complete" event expected by MCP server
            yield {"event": "complete", "data": {"result": summary_payload}}

        except Exception as e:
            logger.error(f"Pipeline streaming failed: {e}")
            yield {
                "event": "error",
                "data": {
                    "error": str(e),
                    "processing_time": time.time() - start_ts
                }
            }

    async def _execute_agent_simple(self, agent, context, prompt):
        """Execute an agent with real-time streaming of LLM responses."""
        try:
            # Get model for this agent
            model = agent._select_model(context)
            if not model:
                return AgentResult(
                    success=False,
                    error="No suitable model available"
                )

            # Stream the LLM response
            response_text = ""
            async for token in self._stream_llm_response(prompt, model, context):
                response_text += token

            # Process the complete response
            result = await agent._execute_core_logic(context, prompt, model)
            
            # Update context with agent output
            context.add_agent_output(agent.name, result)
            
            return AgentResult(
                success=True,
                output=result,
                model_used=model,
                prompt_used=prompt
            )

        except Exception as e:
            logger.error(f"Agent {agent.name} execution failed: {e}")
            return AgentResult(
                success=False,
                error=str(e)
            )

    async def _stream_llm_response(self, prompt, model, context):
        """Stream LLM response token by token."""
        try:
            from services.unified_providers import get_provider_service
            import json
            
            provider_service = get_provider_service()
            
            # Use chat format for better streaming
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]
            
            # Stream the response
            async for chunk in provider_service.stream_chat(
                messages=messages,
                model=f"ollama::{model}",
                temperature=0.7,
                max_tokens=2048
            ):
                # Parse Ollama streaming format
                if isinstance(chunk, str) and chunk.strip():
                    try:
                        # Ollama returns JSON lines
                        data = json.loads(chunk)
                        if 'message' in data and 'content' in data['message']:
                            content = data['message']['content']
                            if content:
                                yield content
                        elif 'response' in data:
                            yield data['response']
                    except json.JSONDecodeError:
                        # If not JSON, treat as plain text
                        if chunk.strip():
                            yield chunk
                elif isinstance(chunk, dict):
                    if 'content' in chunk:
                        yield chunk['content']
                    elif 'text' in chunk:
                        yield chunk['text']
                    
        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            # Fallback to non-streaming
            try:
                from services.unified_providers import get_provider_service
                provider_service = get_provider_service()
                
                response = await provider_service.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=f"ollama::{model}",
                    temperature=0.7,
                    max_tokens=2048
                )
                
                # Simulate streaming by yielding tokens
                tokens = response.text.split()
                for token in tokens:
                    yield token + " "
                    await asyncio.sleep(0.01)
                    
            except Exception as fallback_error:
                logger.error(f"Fallback LLM also failed: {fallback_error}")
                yield "Error: LLM service unavailable"

    def _format_agent_result(self, agent_name, result, context):
        """Format agent result for frontend display."""
        if not result.success:
            return {"error": result.error or "Agent execution failed"}
        
        # Get the raw output from the agent
        output = result.output
        
        if agent_name == "translator":
            # For translator, show the raw response and try to extract GraphQL
            if isinstance(output, dict):
                raw_response = output.get("raw_response", str(output))
                graphql_query = output.get("graphql_query", raw_response)
                confidence = output.get("confidence", 0.8)
                explanation = output.get("explanation", "Raw model response")
            else:
                raw_response = str(output)
                graphql_query = str(output)
                confidence = 0.8
                explanation = "Raw model response"
            
            return {
                "raw_response": raw_response,
                "graphql_query": graphql_query,
                "confidence": confidence,
                "explanation": explanation,
            }
        elif agent_name == "reviewer":
            if isinstance(output, dict):
                raw_response = output.get("raw_response", str(output))
                passed = output.get("passed", True)
                improvements = output.get("improvements", [])
            else:
                raw_response = str(output)
                passed = True
                improvements = []
            
            return {
                "raw_response": raw_response,
                "passed": passed,
                "comments": [raw_response],
                "suggested_improvements": improvements,
            }
        elif agent_name == "rewriter":
            # Rewriter returns a string directly
            raw_response = str(output)
            return {
                "raw_response": raw_response,
                "rewritten_query": raw_response,
                "improvements": ["Query was processed"],
            }
        elif agent_name == "analyzer":
            if isinstance(output, dict):
                raw_response = output.get("raw_response", str(output))
                recommendations = output.get("recommendations", [])
            else:
                raw_response = str(output)
                recommendations = []
            
            return {
                "raw_response": raw_response,
                "analysis": raw_response,
                "recommendations": recommendations,
            }
        
        # For any other agent, return the raw output
        return {
            "raw_response": str(output),
            "output": output
        }

# Global service instance
_orchestration_service: Optional[EnhancedOrchestrationService] = None


def get_orchestration_service() -> EnhancedOrchestrationService:
    """Get the global orchestration service instance."""
    global _orchestration_service
    if _orchestration_service is None:
        _orchestration_service = EnhancedOrchestrationService()
    return _orchestration_service 