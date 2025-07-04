"""
Advanced base classes for LLM agents with sophisticated orchestration capabilities.

This module provides the foundation for a scalable agent framework with:
- Rich agent metadata and capabilities
- Advanced context management
- Performance monitoring integration
- Dynamic configuration support
"""
from __future__ import annotations

import abc
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from pydantic import BaseModel, Field


class AgentCapability(Enum):
    """Defines what an agent can do."""
    REWRITE = "rewrite"           # Can rewrite/clarify queries
    TRANSLATE = "translate"       # Can translate NL to GraphQL
    REVIEW = "review"            # Can review and validate output
    OPTIMIZE = "optimize"        # Can optimize queries/responses
    ANALYZE = "analyze"          # Can analyze schemas/data
    VALIDATE = "validate"        # Can validate syntax/semantics
    GENERATE = "generate"        # Can generate content/code
    EXTRACT = "extract"          # Can extract information
    TRANSFORM = "transform"      # Can transform data formats


class AgentMetadata(BaseModel):
    """Rich metadata for agent configuration and discovery."""
    name: str
    version: str = "1.0.0"
    description: str
    capabilities: Set[AgentCapability] = Field(default_factory=set)
    
    # Model requirements
    preferred_models: List[str] = Field(default_factory=list)
    min_context_length: int = 2048
    max_context_length: int = 32768
    
    # Performance characteristics
    avg_processing_time: float = 0.0
    success_rate: float = 1.0
    cost_per_request: float = 0.0
    
    # Dependencies and requirements
    requires_schema: bool = False
    requires_examples: bool = False
    depends_on: List[str] = Field(default_factory=list)
    
    # Configuration
    configurable_params: Dict[str, Any] = Field(default_factory=dict)
    tags: Set[str] = Field(default_factory=set)
    
    # Runtime tracking
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    usage_count: int = 0


class AgentContext(BaseModel):
    """
    Advanced shared context for agent pipeline execution.
    
    This context flows through the entire pipeline, allowing agents to:
    - Access previous agent outputs
    - Store intermediate results
    - Track performance metrics
    - Maintain execution history
    """
    # Core pipeline data
    original_query: str
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pipeline_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Agent outputs (keyed by agent name)
    rewritten_query: Optional[str] = None
    graphql_query: Optional[str] = None
    review_result: Optional[Dict[str, Any]] = None
    optimization_result: Optional[Dict[str, Any]] = None
    
    # Dynamic data storage for any agent
    agent_outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Context engineering data
    schema_context: Optional[str] = None
    domain_context: Optional[str] = None
    examples: List[Dict[str, str]] = Field(default_factory=list)
    prompt_strategy: Optional[str] = None
    
    # Performance tracking
    execution_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    total_processing_time: float = 0.0
    
    # Metadata
    user_id: Optional[str] = None
    priority: int = 0
    tags: Set[str] = Field(default_factory=set)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Execution state
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    current_agent: Optional[str] = None
    execution_path: List[str] = Field(default_factory=list)
    
    def add_agent_output(self, agent_name: str, output: Any, metrics: Optional[Dict[str, Any]] = None):
        """Add output from an agent with optional performance metrics."""
        self.agent_outputs[agent_name] = output
        self.execution_path.append(agent_name)
        
        if metrics:
            self.execution_metrics[agent_name] = metrics
            if 'processing_time' in metrics:
                self.total_processing_time += metrics['processing_time']
    
    def get_agent_output(self, agent_name: str, default: Any = None) -> Any:
        """Get output from a specific agent."""
        return self.agent_outputs.get(agent_name, default)
    
    def has_capability_output(self, capability: AgentCapability) -> bool:
        """Check if any agent with given capability has produced output."""
        # This would need agent registry to map capabilities to agents
        # For now, simple name-based heuristic
        capability_map = {
            AgentCapability.REWRITE: 'rewritten_query',
            AgentCapability.TRANSLATE: 'graphql_query', 
            AgentCapability.REVIEW: 'review_result',
            AgentCapability.OPTIMIZE: 'optimization_result'
        }
        field_name = capability_map.get(capability)
        if field_name:
            return getattr(self, field_name) is not None
        return False


class BaseAgent(abc.ABC):
    """
    Abstract base class for all agents in the system.
    
    Agents are the core processing units that transform the context
    by adding their specific capabilities (rewriting, translation, etc.)
    """
    
    def __init__(self, metadata: Optional[AgentMetadata] = None):
        self.metadata = metadata or AgentMetadata(
            name=self.__class__.__name__,
            description=self.__class__.__doc__ or "No description provided"
        )
        self._performance_metrics = {
            'total_executions': 0,
            'total_processing_time': 0.0,
            'success_count': 0,
            'error_count': 0
        }
    
    @property
    def name(self) -> str:
        """Agent name from metadata."""
        return self.metadata.name
    
    @property
    def capabilities(self) -> Set[AgentCapability]:
        """Agent capabilities from metadata."""
        return self.metadata.capabilities
    
    @abc.abstractmethod
    async def run(
        self, 
        ctx: AgentContext, 
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """
        Execute the agent's main functionality.
        
        Args:
            ctx: Shared context object to read from and modify
            config: Runtime configuration parameters
            **kwargs: Additional parameters passed from pipeline
            
        The agent should modify the context in-place to add its contribution.
        """
        pass
    
    async def can_execute(self, ctx: AgentContext) -> bool:
        """
        Check if this agent can execute given the current context.
        
        Override this to implement conditional execution logic.
        """
        return True
    
    async def validate_input(self, ctx: AgentContext) -> bool:
        """Validate that the context has required inputs for this agent."""
        return True
    
    async def post_process(self, ctx: AgentContext, result: Any) -> Any:
        """Post-process the agent's output before storing in context."""
        return result
    
    def update_performance_metrics(self, processing_time: float, success: bool):
        """Update internal performance tracking."""
        self._performance_metrics['total_executions'] += 1
        self._performance_metrics['total_processing_time'] += processing_time
        if success:
            self._performance_metrics['success_count'] += 1
        else:
            self._performance_metrics['error_count'] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of agent performance metrics."""
        metrics = self._performance_metrics.copy()
        if metrics['total_executions'] > 0:
            metrics['avg_processing_time'] = metrics['total_processing_time'] / metrics['total_executions']
            metrics['success_rate'] = metrics['success_count'] / metrics['total_executions']
        else:
            metrics['avg_processing_time'] = 0.0
            metrics['success_rate'] = 0.0
        return metrics 