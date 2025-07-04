"""
Advanced Pipeline Execution Engine

Provides sophisticated orchestration of agent pipelines with:
- Configurable execution flows and conditional logic
- Dynamic agent selection and routing
- Performance monitoring and optimization
- Error handling and recovery strategies
- Pipeline versioning and A/B testing
"""

import asyncio
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field

from .base import BaseAgent, AgentContext, AgentCapability
from .registry import agent_registry

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Pipeline execution strategies."""
    SEQUENTIAL = "sequential"      # Execute agents one by one
    PARALLEL = "parallel"         # Execute compatible agents in parallel
    CONDITIONAL = "conditional"   # Execute based on conditions
    DYNAMIC = "dynamic"          # Runtime agent selection


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentConfig(BaseModel):
    """Configuration for an agent in a pipeline."""
    name: str
    enabled: bool = True
    
    # Execution conditions
    execute_if: Optional[str] = None  # Python expression evaluated at runtime
    skip_if: Optional[str] = None     # Python expression to skip execution
    
    # Configuration parameters
    config: Dict[str, Any] = Field(default_factory=dict)
    
    # Resource limits
    timeout: Optional[float] = None
    max_retries: int = 0
    
    # Dependencies
    depends_on: List[str] = Field(default_factory=list)
    required_capabilities: List[AgentCapability] = Field(default_factory=list)


class PipelineConfig(BaseModel):
    """Configuration for pipeline execution."""
    name: str
    version: str = "1.0.0"
    description: str = ""
    
    # Agent configuration
    agents: List[AgentConfig] = Field(default_factory=list)
    
    # Execution settings
    strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    timeout: Optional[float] = None
    max_parallel: int = 3
    
    # Context engineering
    prompt_strategy: Optional[str] = None
    domain_context: Optional[str] = None
    examples: List[Dict[str, str]] = Field(default_factory=list)
    
    # Error handling
    continue_on_error: bool = False
    error_recovery: Dict[str, str] = Field(default_factory=dict)
    
    # Monitoring
    enable_monitoring: bool = True
    log_level: str = "INFO"
    
    # Metadata
    tags: Set[str] = Field(default_factory=set)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ExecutionResult(BaseModel):
    """Result of pipeline execution."""
    pipeline_name: str
    pipeline_version: str
    status: PipelineStatus
    
    # Execution details
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_duration: float = 0.0
    
    # Agent results
    agent_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    execution_order: List[str] = Field(default_factory=list)
    
    # Context state
    final_context: Optional[AgentContext] = None
    
    # Performance metrics
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # Error information
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class Pipeline:
    """
    Advanced pipeline execution engine for agent orchestration.
    
    Features:
    - Multiple execution strategies (sequential, parallel, conditional)
    - Dynamic agent selection based on context
    - Comprehensive error handling and recovery
    - Performance monitoring and optimization
    - A/B testing support
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.status = PipelineStatus.PENDING
        self._execution_context = None
        
    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "Pipeline":
        """Create pipeline from configuration dictionary."""
        config = PipelineConfig(**config_dict)
        return cls(config)
    
    @classmethod
    def from_agents(
        cls, 
        agents: List[str], 
        name: str = "dynamic_pipeline",
        strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    ) -> "Pipeline":
        """Create pipeline from list of agent names."""
        agent_configs = [AgentConfig(name=agent_name) for agent_name in agents]
        config = PipelineConfig(
            name=name,
            agents=agent_configs,
            strategy=strategy
        )
        return cls(config)
    
    async def execute(
        self, 
        context: AgentContext,
        override_config: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """
        Execute the pipeline with the given context.
        
        Args:
            context: Execution context containing query and state
            override_config: Runtime configuration overrides
            
        Returns:
            Execution result with performance metrics and outputs
        """
        start_time = time.time()
        
        # Apply configuration overrides
        if override_config:
            self._apply_config_overrides(override_config)
        
        # Initialize execution result
        result = ExecutionResult(
            pipeline_name=self.config.name,
            pipeline_version=self.config.version,
            status=PipelineStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        self.status = PipelineStatus.RUNNING
        
        try:
            # Apply context engineering
            await self._prepare_context(context)
            
            # Resolve agent execution order
            execution_order = await self._resolve_execution_order()
            result.execution_order = execution_order
            
            # Execute agents based on strategy
            if self.config.strategy == ExecutionStrategy.SEQUENTIAL:
                await self._execute_sequential(context, result)
            elif self.config.strategy == ExecutionStrategy.PARALLEL:
                await self._execute_parallel(context, result)
            elif self.config.strategy == ExecutionStrategy.CONDITIONAL:
                await self._execute_conditional(context, result)
            elif self.config.strategy == ExecutionStrategy.DYNAMIC:
                await self._execute_dynamic(context, result)
            
            result.status = PipelineStatus.COMPLETED
            self.status = PipelineStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            result.status = PipelineStatus.FAILED
            result.errors.append({
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "stage": "execution"
            })
            self.status = PipelineStatus.FAILED
            
            if not self.config.continue_on_error:
                raise
        
        finally:
            # Finalize results
            result.completed_at = datetime.utcnow()
            result.total_duration = time.time() - start_time
            result.final_context = context
            result.performance_metrics = self._collect_performance_metrics(context)
        
        return result
    
    def _apply_config_overrides(self, overrides: Dict[str, Any]):
        """Apply runtime configuration overrides."""
        for key, value in overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    async def _prepare_context(self, context: AgentContext):
        """Prepare context with pipeline-specific settings."""
        if self.config.domain_context:
            context.domain_context = self.config.domain_context
        
        if self.config.examples:
            context.examples = self.config.examples
        
        if self.config.prompt_strategy:
            context.prompt_strategy = self.config.prompt_strategy
        
        # Add pipeline metadata to context
        context.metadata.update({
            "pipeline_name": self.config.name,
            "pipeline_version": self.config.version,
            "execution_strategy": self.config.strategy.value
        })
    
    async def _resolve_execution_order(self) -> List[str]:
        """Resolve the order in which agents should be executed."""
        enabled_agents = [ac.name for ac in self.config.agents if ac.enabled]
        
        if self.config.strategy in [ExecutionStrategy.SEQUENTIAL, ExecutionStrategy.CONDITIONAL]:
            # Use dependency resolution for ordered execution
            return agent_registry.get_execution_order(enabled_agents)
        else:
            # For parallel and dynamic, return as-is
            return enabled_agents
    
    async def _execute_sequential(self, context: AgentContext, result: ExecutionResult):
        """Execute agents sequentially."""
        for agent_name in result.execution_order:
            await self._execute_agent(agent_name, context, result)
    
    async def _execute_parallel(self, context: AgentContext, result: ExecutionResult):
        """Execute compatible agents in parallel."""
        # Group agents by dependency level
        dependency_levels = self._group_by_dependency_level(result.execution_order)
        
        for level_agents in dependency_levels:
            # Execute agents at the same level in parallel
            tasks = []
            for agent_name in level_agents:
                task = asyncio.create_task(
                    self._execute_agent(agent_name, context, result)
                )
                tasks.append(task)
            
            # Wait for all agents at this level to complete
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_conditional(self, context: AgentContext, result: ExecutionResult):
        """Execute agents based on conditions."""
        for agent_config in self.config.agents:
            if not agent_config.enabled:
                continue
            
            # Check execution conditions
            should_execute = await self._should_execute_agent(agent_config, context)
            
            if should_execute:
                await self._execute_agent(agent_config.name, context, result)
    
    async def _execute_dynamic(self, context: AgentContext, result: ExecutionResult):
        """Dynamically select and execute agents based on runtime conditions."""
        # This is a sophisticated strategy that could select agents based on:
        # - Context analysis
        # - Previous agent results
        # - Performance metrics
        # - User preferences
        
        # For now, fall back to conditional execution
        await self._execute_conditional(context, result)
    
    async def _execute_agent(
        self, 
        agent_name: str, 
        context: AgentContext, 
        result: ExecutionResult
    ):
        """Execute a single agent with error handling and metrics collection."""
        agent_config = self._get_agent_config(agent_name)
        agent_class = agent_registry.get_agent(agent_name)
        
        if not agent_class:
            error_msg = f"Agent {agent_name} not found in registry"
            logger.error(error_msg)
            result.errors.append({
                "agent": agent_name,
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            })
            return
        
        agent_start_time = time.time()
        context.current_agent = agent_name
        
        try:
            # Create agent instance
            agent = agent_class()
            
            # Execute with timeout if specified
            if agent_config and agent_config.timeout:
                await asyncio.wait_for(
                    agent.run(context, config=agent_config.config if agent_config else {}),
                    timeout=agent_config.timeout
                )
            else:
                await agent.run(context, config=agent_config.config if agent_config else {})
            
            # Collect agent metrics
            processing_time = time.time() - agent_start_time
            result.agent_results[agent_name] = {
                "status": "completed",
                "processing_time": processing_time,
                "output": context.get_agent_output(agent_name)
            }
            
            # Update agent performance metrics
            agent.update_performance_metrics(processing_time, True)
            
        except asyncio.TimeoutError:
            error_msg = f"Agent {agent_name} timed out"
            logger.error(error_msg)
            result.errors.append({
                "agent": agent_name,
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            error_msg = f"Agent {agent_name} failed: {str(e)}"
            logger.error(error_msg)
            result.errors.append({
                "agent": agent_name,
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Update agent performance metrics
            if agent_class:
                agent = agent_class()
                agent.update_performance_metrics(time.time() - agent_start_time, False)
        
        finally:
            context.current_agent = None
    
    def _get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent."""
        for config in self.config.agents:
            if config.name == agent_name:
                return config
        return None
    
    async def _should_execute_agent(
        self, 
        agent_config: AgentConfig, 
        context: AgentContext
    ) -> bool:
        """Determine if an agent should be executed based on conditions."""
        # Check skip conditions
        if agent_config.skip_if:
            try:
                if eval(agent_config.skip_if, {"context": context}):
                    return False
            except Exception as e:
                logger.warning(f"Error evaluating skip condition for {agent_config.name}: {e}")
        
        # Check execution conditions
        if agent_config.execute_if:
            try:
                return eval(agent_config.execute_if, {"context": context})
            except Exception as e:
                logger.warning(f"Error evaluating execute condition for {agent_config.name}: {e}")
                return False
        
        return True
    
    def _group_by_dependency_level(self, agent_names: List[str]) -> List[List[str]]:
        """Group agents by dependency level for parallel execution."""
        levels = []
        remaining = set(agent_names)
        
        while remaining:
            current_level = []
            
            for agent_name in list(remaining):
                # Check if all dependencies are already processed
                agent_config = self._get_agent_config(agent_name)
                dependencies = agent_config.depends_on if agent_config else []
                
                if all(dep not in remaining for dep in dependencies):
                    current_level.append(agent_name)
            
            if not current_level:
                # Circular dependency or other issue
                current_level = list(remaining)  # Execute remaining agents
            
            levels.append(current_level)
            remaining -= set(current_level)
        
        return levels
    
    def _collect_performance_metrics(self, context: AgentContext) -> Dict[str, Any]:
        """Collect comprehensive performance metrics."""
        return {
            "total_agents_executed": len(context.execution_path),
            "total_processing_time": context.total_processing_time,
            "agent_metrics": context.execution_metrics,
            "context_size": len(str(context)),
            "execution_path": context.execution_path
        } 