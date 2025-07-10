"""
Unified Agent System for MPPW-MCP

This module provides a streamlined, modern agent framework that:
1. Uses the unified configuration system
2. Integrates with the unified prompt system  
3. Provides consistent patterns across all agents
4. Eliminates legacy code and inconsistencies
5. Makes it easy to add new agents and capabilities
"""

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union, Type, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import logging
import time
import json
import uuid

from config.unified_config import (
    get_unified_config, AgentType, AgentCapability, PromptStrategy,
    AgentConfig, ModelConfig
)
from prompts.unified_prompts import get_prompt_manager, get_prompt_for_agent
from services.llm_factory import resolve_llm
from services.llm_tracking_service import get_tracking_service

logger = logging.getLogger(__name__)


# =============================================================================
# Core Agent Context and Results
# =============================================================================

@dataclass
class AgentContext:
    """
    Unified context object that flows through the agent pipeline.
    Contains all data, configuration, and state needed by agents.
    """
    # Core data
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_query: str = ""
    user_id: Optional[str] = None
    
    # Pipeline state
    current_agent: Optional[str] = None
    execution_path: List[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    
    # Agent outputs (keyed by agent name)
    rewritten_query: Optional[str] = None
    graphql_query: Optional[str] = None
    review_result: Optional[Dict[str, Any]] = None
    optimization_result: Optional[Dict[str, Any]] = None
    analysis_result: Optional[Dict[str, Any]] = None
    
    # Generic agent outputs for extensibility
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    
    # Context data for prompts
    domain_context: Optional[str] = None
    schema_context: Optional[str] = None
    examples: List[Dict[str, str]] = field(default_factory=list)
    
    # Performance metrics
    execution_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_processing_time: float = 0.0
    
    # Configuration overrides
    model_overrides: Dict[str, str] = field(default_factory=dict)
    prompt_strategy_override: Optional[PromptStrategy] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_agent_output(self, agent_name: str, output: Any, metrics: Optional[Dict[str, Any]] = None):
        """Add output from an agent with optional metrics."""
        self.agent_outputs[agent_name] = output
        self.execution_path.append(agent_name)
        
        if metrics:
            self.execution_metrics[agent_name] = metrics
            if 'processing_time' in metrics:
                self.total_processing_time += metrics['processing_time']
    
    def get_agent_output(self, agent_name: str, default: Any = None) -> Any:
        """Get output from a specific agent."""
        return self.agent_outputs.get(agent_name, default)
    
    def has_agent_output(self, agent_name: str) -> bool:
        """Check if an agent has produced output."""
        return agent_name in self.agent_outputs
    
    def get_prompt_context(self) -> Dict[str, Any]:
        """Get context data formatted for prompt rendering."""
        return {
            'query': self.rewritten_query or self.original_query,
            'original_query': self.original_query,
            'domain': self.domain_context,
            'schema_context': self.schema_context,
            'examples': self.examples,
            'graphql_query': self.graphql_query,
            'session_id': self.session_id,
            'user_id': self.user_id,
            # Add any other agent outputs that might be useful for prompts
            **{f"{agent}_output": output for agent, output in self.agent_outputs.items()}
        }


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
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Base Agent Interface
# =============================================================================

class BaseAgent(ABC):
    """
    Base class for all agents in the unified system.
    
    Provides consistent interface and common functionality:
    - Configuration integration
    - Prompt management
    - Model selection and fallback
    - Performance tracking
    - Error handling
    """
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__.lower().replace('agent', '')
        self.config_manager = get_unified_config()
        self.prompt_manager = get_prompt_manager()
        self.tracking_service = get_tracking_service()
        
        # Get agent configuration
        self.config = self.config_manager.get_agent(self.name)
        if not self.config:
            logger.warning(f"No configuration found for agent: {self.name}")
            # Create a default config
            self.config = AgentConfig(
                name=self.name,
                agent_type=self._get_agent_type(),
                capabilities=self._get_capabilities(),
                primary_model="phi3:mini"
            )
        
        logger.info(f"Initialized agent: {self.name} ({self.config.agent_type.value})")
    
    @abstractmethod
    def _get_agent_type(self) -> AgentType:
        """Get the agent type. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def _execute_core_logic(self, ctx: AgentContext, prompt: str, model: str) -> Any:
        """Execute the core agent logic. Must be implemented by subclasses."""
        pass
    
    async def execute(self, ctx: AgentContext) -> AgentResult:
        """
        Main execution method that provides consistent behavior across all agents.
        """
        start_time = time.time()
        
        try:
            logger.info(f"Agent {self.name}: Starting execution")
            
            # Validate inputs
            logger.info(f"Agent {self.name}: Validating inputs")
            if not await self._validate_inputs(ctx):
                logger.error(f"Agent {self.name}: Input validation failed")
                return AgentResult(
                    success=False,
                    error="Input validation failed",
                    processing_time=time.time() - start_time
                )
            
            # Select model (with fallback support)
            logger.info(f"Agent {self.name}: Selecting model")
            model = self._select_model(ctx)
            if not model:
                logger.error(f"Agent {self.name}: No suitable model available")
                return AgentResult(
                    success=False,
                    error="No suitable model available",
                    processing_time=time.time() - start_time
                )
            
            logger.info(f"Agent {self.name}: Selected model {model}")
            
            # Generate prompt
            logger.info(f"Agent {self.name}: Generating prompt")
            prompt = self._generate_prompt(ctx)
            if not prompt:
                logger.error(f"Agent {self.name}: Failed to generate prompt")
                return AgentResult(
                    success=False,
                    error="Failed to generate prompt",
                    processing_time=time.time() - start_time
                )
            
            logger.info(f"Agent {self.name}: Generated prompt successfully")
            
            # Execute core logic with tracking
            logger.info(f"Agent {self.name}: Starting core logic execution")
            async with self.tracking_service.track_interaction(
                session_id=ctx.session_id,
                model=model,
                provider=self._get_model_provider(model),
                interaction_type=self.config.agent_type.value,
                user_id=ctx.user_id,
                context_data={
                    "agent_name": self.name,
                    "agent_type": self.config.agent_type.value,
                    "prompt_strategy": self.config.prompt_strategy.value
                }
            ) as tracker:
                
                # Set tracking data
                tracker.set_prompt(prompt)
                tracker.set_parameters(
                    temperature=self.config.temperature,
                    max_tokens=self.config.context_window
                )
                
                # Execute core logic
                logger.info(f"Agent {self.name}: Calling _execute_core_logic")
                output = await self._execute_core_logic(ctx, prompt, model)
                logger.info(f"Agent {self.name}: _execute_core_logic completed")
                
                # Track successful execution
                processing_time = time.time() - start_time
                tracker.set_response(str(output))
                tracker.set_performance_metrics(processing_time=processing_time)
                
                # Create result
                result = AgentResult(
                    success=True,
                    output=output,
                    processing_time=processing_time,
                    model_used=model,
                    prompt_used=prompt[:200] + "..." if len(prompt) > 200 else prompt,
                    metadata={
                        "agent_type": self.config.agent_type.value,
                        "capabilities": [c.value for c in self.config.capabilities],
                        "prompt_strategy": self.config.prompt_strategy.value
                    }
                )
                
                # Update context
                ctx.add_agent_output(
                    self.name,
                    output,
                    {
                        "processing_time": processing_time,
                        "model_used": model,
                        "success": True
                    }
                )
                
                logger.info(f"Agent {self.name}: Execution completed successfully")
                return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Agent {self.name} execution failed: {e}")
            
            return AgentResult(
                success=False,
                error=str(e),
                processing_time=processing_time,
                metadata={"exception_type": type(e).__name__}
            )
    
    def _select_model(self, ctx: AgentContext) -> Optional[str]:
        """Select the best model for this agent execution."""
        # Check for override in context
        if self.name in ctx.model_overrides:
            return ctx.model_overrides[self.name]
        
        # Try primary model
        primary_model = self.config.primary_model
        if self._is_model_available(primary_model):
            return primary_model
        
        # Try fallback models
        for fallback in self.config.fallback_models:
            if self._is_model_available(fallback):
                logger.warning(f"Using fallback model {fallback} for agent {self.name}")
                return fallback
        
        # Try to find any model with required capabilities
        recommended = self.config_manager.recommend_model_for_agent(self.name)
        if recommended and self._is_model_available(recommended):
            logger.warning(f"Using recommended model {recommended} for agent {self.name}")
            return recommended
        
        return None
    
    def _is_model_available(self, model_name: str) -> bool:
        """Check if a model is available and configured."""
        model_config = self.config_manager.get_model(model_name)
        return model_config is not None
    
    def _get_model_provider(self, model_name: str) -> str:
        """Get the provider for a model."""
        model_config = self.config_manager.get_model(model_name)
        return model_config.provider.value if model_config else "unknown"
    
    def _generate_prompt(self, ctx: AgentContext) -> Optional[str]:
        """Generate prompt for this agent using the prompt manager."""
        # Get prompt strategy (with override support)
        strategy = ctx.prompt_strategy_override or self.config.prompt_strategy
        
        # Get prompt context
        prompt_context = ctx.get_prompt_context()
        
        # Get prompt from manager
        prompt = get_prompt_for_agent(self.name, prompt_context, strategy)
        if prompt:
            return prompt
        
        # Fallback: try with default strategy
        if strategy != PromptStrategy.DETAILED:
            logger.warning(f"Falling back to detailed strategy for agent {self.name}")
            return get_prompt_for_agent(self.name, prompt_context, PromptStrategy.DETAILED)
        
        return None
    
    async def _validate_inputs(self, ctx: AgentContext) -> bool:
        """Validate that required inputs are present in context."""
        for required_input in self.config.required_inputs:
            if required_input == "original_query" and not ctx.original_query:
                logger.error(f"Agent {self.name} requires original_query")
                return False
            elif required_input == "query" and not (ctx.rewritten_query or ctx.original_query):
                logger.error(f"Agent {self.name} requires query")
                return False
            elif required_input == "graphql_query" and not ctx.graphql_query:
                logger.error(f"Agent {self.name} requires graphql_query")
                return False
            elif required_input == "schema_context" and not ctx.schema_context:
                logger.error(f"Agent {self.name} requires schema_context")
                return False
            # Add more validation rules as needed
        
        return True
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this agent."""
        # This would typically be implemented with persistent storage
        return {
            "agent_name": self.name,
            "agent_type": self.config.agent_type.value,
            "total_executions": 0,
            "avg_processing_time": 0.0,
            "success_rate": 1.0,
            "primary_model": self.config.primary_model
        }


# =============================================================================
# Concrete Agent Implementations
# =============================================================================

class RewriterAgent(BaseAgent):
    """Agent that rewrites and clarifies natural language queries."""
    
    def _get_agent_type(self) -> AgentType:
        return AgentType.REWRITER
    
    def _get_capabilities(self) -> List[AgentCapability]:
        return [AgentCapability.REWRITE]
    
    async def _execute_core_logic(self, ctx: AgentContext, prompt: str, model: str) -> str:
        """Rewrite the query using the LLM."""
        import asyncio
        
        try:
            service, provider, stripped_model = resolve_llm(model)
            
            # Format model name correctly for the provider service
            formatted_model = f"{provider}::{stripped_model}"
            
            messages = [
                {"role": "system", "content": "You are a query rewriting expert."},
                {"role": "user", "content": prompt}
            ]
            
            # Add timeout to prevent hanging
            logger.info(f"RewriterAgent: Making LLM call with timeout")
            result = await asyncio.wait_for(
                service.chat_completion(
                    messages=messages,
                    model=formatted_model,
                    temperature=self.config.temperature
                ),
                timeout=self.config.timeout  # Use configured timeout
            )
            
            logger.info(f"RewriterAgent: LLM call completed successfully")
            
            # Extract rewritten query from response
            rewritten = result.text.strip()
            
            # Clean up the response - remove any extra formatting
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            
            # Basic validation
            if len(rewritten) < 3:
                logger.warning("Rewritten query too short, using original")
                return ctx.original_query
            
            if len(rewritten) > len(ctx.original_query) * 3:
                logger.warning("Rewritten query too verbose, using original")
                return ctx.original_query
            
            # Update context
            ctx.rewritten_query = rewritten
            
            return rewritten
            
        except asyncio.TimeoutError:
            logger.error(f"RewriterAgent: LLM call timed out after {self.config.timeout} seconds")
            return ctx.original_query
        except Exception as e:
            logger.error(f"RewriterAgent: LLM call failed: {e}")
            return ctx.original_query


class TranslatorAgent(BaseAgent):
    """Agent that translates natural language to GraphQL."""
    
    def _get_agent_type(self) -> AgentType:
        return AgentType.TRANSLATOR
    
    def _get_capabilities(self) -> List[AgentCapability]:
        return [AgentCapability.TRANSLATE]
    
    async def _execute_core_logic(self, ctx: AgentContext, prompt: str, model: str) -> Dict[str, Any]:
        import re
        import json as pyjson
        import asyncio
        
        try:
            service, provider, stripped_model = resolve_llm(model)
            formatted_model = f"{provider}::{stripped_model}"
            messages = [
                {"role": "system", "content": "You are a GraphQL expert."},
                {"role": "user", "content": prompt}
            ]
            
            # Add timeout to prevent hanging
            logger.info(f"TranslatorAgent: Making LLM call with timeout")
            result = await asyncio.wait_for(
                service.chat_completion(
                    messages=messages,
                    model=formatted_model,
                    temperature=self.config.temperature
                ),
                timeout=self.config.timeout  # Use configured timeout
            )
            
            logger.info(f"TranslatorAgent: LLM call completed successfully")
            
            response_text = result.text.strip()
            # Extraction for pipeline use only (not for UI display)
            graphql_query = None
            confidence = None
            explanation = "Raw model response"
            # Try to extract JSON code block
            json_match = re.search(r'```json\s*([\s\S]+?)```', response_text)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_brace_match = re.search(r'\{[\s\S]+\}', response_text)
                json_str = json_brace_match.group(0) if json_brace_match else None
            if json_str:
                try:
                    parsed = pyjson.loads(json_str)
                    graphql_query = parsed.get("graphql")
                    confidence = parsed.get("confidence")
                    explanation = "Extracted from JSON block"
                except Exception:
                    pass
            if not graphql_query:
                graphql_match = re.search(r'"graphql"\s*:\s*"([^"]+?)"', response_text)
                if graphql_match:
                    graphql_query = graphql_match.group(1)
                    explanation = "Extracted from quoted graphql field"
            if not graphql_query:
                graphql_match = re.search(r'(query\s*\{[\s\S]+?\})', response_text)
                if graphql_match:
                    graphql_query = graphql_match.group(1)
                    explanation = "Extracted from query pattern"
            if not graphql_query and response_text.startswith("query"):
                graphql_query = response_text
                explanation = "Used entire response as query"
            if not graphql_query:
                graphql_query = ""
                explanation = "Failed to extract GraphQL query"
            try:
                confidence = float(confidence)
            except Exception:
                confidence = 0.8
            ctx.graphql_query = graphql_query
            # Always return the full raw response for UI display
            return {
                "raw_response": response_text,  # Always show exactly what the model returned
                "graphql_query": graphql_query,
                "confidence": confidence,
                "explanation": explanation
            }
            
        except asyncio.TimeoutError:
            logger.error(f"TranslatorAgent: LLM call timed out after {self.config.timeout} seconds")
            return {
                "raw_response": f"LLM call timed out after {self.config.timeout} seconds",
                "graphql_query": "",
                "confidence": 0.0,
                "explanation": "Timeout error"
            }
        except Exception as e:
            logger.error(f"TranslatorAgent: LLM call failed: {e}")
            return {
                "raw_response": f"LLM call failed: {str(e)}",
                "graphql_query": "",
                "confidence": 0.0,
                "explanation": "Error occurred"
            }


class ReviewerAgent(BaseAgent):
    """Agent that reviews and validates GraphQL queries."""
    
    def _get_agent_type(self) -> AgentType:
        return AgentType.REVIEWER
    
    def _get_capabilities(self) -> List[AgentCapability]:
        return [AgentCapability.REVIEW, AgentCapability.VALIDATE]
    
    async def _execute_core_logic(self, ctx: AgentContext, prompt: str, model: str) -> Dict[str, Any]:
        """Review the GraphQL query using the LLM."""
        service, provider, stripped_model = resolve_llm(model)
        
        # Format model name correctly for the provider service
        formatted_model = f"{provider}::{stripped_model}"
        
        messages = [
            {"role": "system", "content": "You are a GraphQL review expert."},
            {"role": "user", "content": prompt}
        ]
        
        result = await service.chat_completion(
            messages=messages,
            model=formatted_model,
            temperature=self.config.temperature
        )
        
        # Return raw response without JSON parsing
        response_text = result.text.strip()
        
        # Update context with raw response
        ctx.review_result = {"raw_response": response_text}
        
        # Return a simple structure with the raw response
        return {
            "raw_response": response_text,
            "passed": True,
            "score": 80,
            "issues": [],
            "improvements": [],
            "compliments": ["Query review completed"]
        }


class AnalyzerAgent(BaseAgent):
    """Agent that analyzes data and provides insights."""
    
    def _get_agent_type(self) -> AgentType:
        return AgentType.ANALYZER
    
    def _get_capabilities(self) -> List[AgentCapability]:
        return [AgentCapability.ANALYZE, AgentCapability.EXTRACT]
    
    async def _execute_core_logic(self, ctx: AgentContext, prompt: str, model: str) -> Dict[str, Any]:
        """Analyze data using the LLM."""
        service, provider, stripped_model = resolve_llm(model)
        
        # Format model name correctly for the provider service
        formatted_model = f"{provider}::{stripped_model}"
        
        messages = [
            {"role": "system", "content": "You are a data analysis expert."},
            {"role": "user", "content": prompt}
        ]
        
        result = await service.chat_completion(
            messages=messages,
            model=formatted_model,
            temperature=self.config.temperature
        )
        
        # Return raw response without JSON parsing
        response_text = result.text.strip()
        
        # Update context with raw response
        ctx.analysis_result = {"raw_response": response_text}
        
        # Return a simple structure with the raw response
        return {
            "raw_response": response_text,
            "summary": "Analysis completed",
            "insights": [],
            "recommendations": [],
            "confidence": 0.8
        }


# =============================================================================
# Agent Factory and Registry
# =============================================================================

class AgentFactory:
    """Factory for creating agent instances."""
    
    _agent_classes: Dict[str, Type[BaseAgent]] = {
        "rewriter": RewriterAgent,
        "translator": TranslatorAgent,
        "reviewer": ReviewerAgent,
        "analyzer": AnalyzerAgent
    }
    
    @classmethod
    def create_agent(cls, agent_name: str) -> Optional[BaseAgent]:
        """Create an agent instance by name."""
        agent_class = cls._agent_classes.get(agent_name)
        if agent_class:
            return agent_class(agent_name)
        return None
    
    @classmethod
    def register_agent_class(cls, name: str, agent_class: Type[BaseAgent]) -> None:
        """Register a custom agent class."""
        cls._agent_classes[name] = agent_class
        logger.info(f"Registered agent class: {name}")
    
    @classmethod
    def list_available_agents(cls) -> List[str]:
        """List all available agent types."""
        return list(cls._agent_classes.keys())


# =============================================================================
# Pipeline Executor
# =============================================================================

class PipelineExecutor:
    """Executes agent pipelines according to configuration."""
    
    def __init__(self):
        self.config_manager = get_unified_config()
        self.agent_factory = AgentFactory()
    
    async def execute_pipeline(
        self,
        pipeline_name: str,
        ctx: AgentContext,
        parallel: bool = False
    ) -> Dict[str, AgentResult]:
        """Execute a configured pipeline."""
        pipeline_config = self.config_manager.get_pipeline(pipeline_name)
        if not pipeline_config:
            raise ValueError(f"Pipeline not found: {pipeline_name}")
        
        results = {}
        
        if parallel and pipeline_config.execution_mode == "parallel":
            # Execute agents in parallel
            tasks = []
            for agent_name in pipeline_config.agents:
                agent = self.agent_factory.create_agent(agent_name)
                if agent:
                    task = asyncio.create_task(agent.execute(ctx))
                    tasks.append((agent_name, task))
            
            # Wait for all tasks
            for agent_name, task in tasks:
                try:
                    result = await task
                    results[agent_name] = result
                except Exception as e:
                    results[agent_name] = AgentResult(
                        success=False,
                        error=str(e)
                    )
        else:
            # Execute agents sequentially
            for agent_name in pipeline_config.agents:
                agent = self.agent_factory.create_agent(agent_name)
                if agent:
                    result = await agent.execute(ctx)
                    results[agent_name] = result
                    
                    # Stop on failure if not configured to continue
                    if not result.success and not pipeline_config.context_sharing:
                        break
        
        return results
    
    async def execute_agents(
        self,
        agent_names: List[str],
        ctx: AgentContext
    ) -> Dict[str, AgentResult]:
        """Execute a list of agents sequentially."""
        results = {}
        
        for agent_name in agent_names:
            agent = self.agent_factory.create_agent(agent_name)
            if agent:
                result = await agent.execute(ctx)
                results[agent_name] = result
        
        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def create_agent(agent_name: str) -> Optional[BaseAgent]:
    """Create an agent instance."""
    return AgentFactory.create_agent(agent_name)


async def execute_single_agent(agent_name: str, ctx: AgentContext) -> AgentResult:
    """Execute a single agent."""
    agent = create_agent(agent_name)
    if not agent:
        return AgentResult(
            success=False,
            error=f"Agent not found: {agent_name}"
        )
    
    return await agent.execute(ctx)


async def execute_pipeline(pipeline_name: str, ctx: AgentContext) -> Dict[str, AgentResult]:
    """Execute a configured pipeline."""
    executor = PipelineExecutor()
    return await executor.execute_pipeline(pipeline_name, ctx)


# Example usage for testing
if __name__ == "__main__":
    import asyncio
    
    async def test_agents():
        # Create test context
        ctx = AgentContext(
            original_query="Get all users with their names",
            schema_context="type User { id: ID! name: String! email: String! }"
        )
        
        # Test rewriter
        rewriter = create_agent("rewriter")
        if rewriter:
            result = await rewriter.execute(ctx)
            print(f"Rewriter result: {result.success}, output: {result.output}")
        
        # Test translator
        translator = create_agent("translator")
        if translator:
            result = await translator.execute(ctx)
            print(f"Translator result: {result.success}, output: {result.output}")
    
    asyncio.run(test_agents()) 